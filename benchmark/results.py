"""
Result records and aggregation for CardiomniBench-VD.

One record per (method, task, case). Everything downstream - per-method summary,
the paper tables, the failure-mode analysis - is derived from that flat list, so
there is a single place where a number can come from.

Two rules this module exists to enforce:

1. A missing or failed prediction is recorded as a failure with a reason. It is
   never silently dropped and never imputed. A method that crashes on 5 of 20
   cases must not look better than one that scores badly on all 20.
2. Aggregates always carry n. A mean over 3 cases and a mean over 60 do not get
   printed the same way.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays into JSON-native types.

    Metric functions built on numpy return ``np.float32``/``np.int64`` rather than
    Python floats, and those are not JSON serialisable even though they print
    identically. Converting here keeps every writer safe instead of relying on
    each ``json.dumps`` call site to pass ``default=str``, which would silently
    stringify numbers into unusable output.
    """
    import numpy as np

    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class CaseResult:
    """Outcome of one method on one case."""

    method: str
    task: str
    case_id: str
    status: str  # "ok" | "failed" | "skipped"
    metrics: dict[str, float] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    runtime_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSONL persistence.

        Instance masks are dropped rather than encoded. A per-instance mask is a
        512x512 array, so keeping it would make the JSONL orders of magnitude
        larger than the numbers anyone reads from it, and ``asdict`` leaves it as
        a raw ndarray that no JSON encoder accepts. The mask is an intermediate
        for scoring, not a result; ``n_instances`` and the metrics already record
        what it contributed. Everything else is preserved verbatim.
        """
        payload = asdict(self)
        prediction = payload.get("prediction")
        if isinstance(prediction, dict):
            instances = prediction.get("instances")
            if isinstance(instances, list):
                prediction["instances"] = [
                    {k: v for k, v in inst.items() if k != "mask"}
                    if isinstance(inst, dict)
                    else inst
                    for inst in instances
                ]
        return _jsonable(payload)


@dataclass
class MethodSummary:
    """Aggregate of one method over one task."""

    method: str
    task: str
    family: str
    n_total: int
    n_ok: int
    n_failed: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    source: str = ""
    reported: str = ""
    cross_domain: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return self.n_ok / self.n_total if self.n_total else 0.0


def aggregate(
    results: list[CaseResult],
    method: str,
    task: str,
    family: str = "",
    source: str = "",
    reported: str = "",
    cross_domain: bool = False,
) -> MethodSummary:
    """Collapse per-case results into a summary with mean/sd/median per metric."""
    subset = [r for r in results if r.method == method and r.task == task]
    ok = [r for r in subset if r.ok]
    failed = [r for r in subset if r.status == "failed"]

    metric_names: set[str] = set()
    for record in ok:
        metric_names.update(record.metrics)

    metrics: dict[str, dict[str, float]] = {}
    for name in sorted(metric_names):
        values = [
            r.metrics[name]
            for r in ok
            if name in r.metrics and _is_finite(r.metrics[name])
        ]
        if not values:
            continue
        metrics[name] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    # Keep a few distinct error messages: enough to diagnose, not a dump.
    unique_errors: list[str] = []
    for record in failed:
        message = (record.error or "unknown").splitlines()[0][:200]
        if message not in unique_errors:
            unique_errors.append(message)

    return MethodSummary(
        method=method,
        task=task,
        family=family,
        n_total=len(subset),
        n_ok=len(ok),
        n_failed=len(failed),
        metrics=metrics,
        source=source,
        reported=reported,
        cross_domain=cross_domain,
        errors=unique_errors[:5],
    )


#: Where each task keeps its difficulty label. Read at report time from the case
#: directory, never passed to a method: a model told a case is hard is no longer
#: being measured on the same task as one that was not. Stratification is a
#: property of the analysis, not of the input.
DIFFICULTY_FIELD = "difficulty_level"


def case_difficulty(case_dir: Path) -> str | None:
    """Read a case's difficulty label from its task.yaml.

    Returns None when the field is absent so a task without difficulty labels
    produces an empty stratification rather than a fabricated bucket.
    """
    import yaml

    spec_path = case_dir / "task.yaml"
    if not spec_path.exists():
        return None
    try:
        with spec_path.open() as handle:
            spec = yaml.safe_load(handle) or {}
    except Exception:
        return None
    value = (spec.get("case_metadata") or {}).get(DIFFICULTY_FIELD)
    return str(value) if value else None


def stratify_by_difficulty(
    results: list[CaseResult],
    method: str,
    task: str,
    difficulty_by_case: dict[str, str],
) -> dict[str, MethodSummary]:
    """Aggregate one method/task separately within each difficulty bucket.

    Buckets are whatever the data uses: ARCADE labels easy/medium/hard and
    CardioSYNTAX labels low/intermediate/high, and they are deliberately not
    mapped onto a shared scale. Forcing one axis would imply the two vocabularies
    are commensurable, which is not established; reporting each task against its
    own labels keeps every cell interpretable.

    Cells are reported at whatever size they are, including n=2. ``MethodSummary``
    already carries per-metric ``n``, so a small cell is visible rather than
    hidden behind a mean. Rebalancing the label distribution to produce even
    cells would mean editing the data to suit the table.
    """
    subset = [r for r in results if r.method == method and r.task == task]
    buckets: dict[str, list[CaseResult]] = {}
    for record in subset:
        label = difficulty_by_case.get(record.case_id)
        if label is None:
            continue
        buckets.setdefault(label, []).append(record)

    return {
        label: aggregate(records, method=method, task=task)
        for label, records in sorted(buckets.items())
    }


def _is_finite(value: Any) -> bool:
    """Guard aggregates against inf/nan, which Hausdorff produces on empty masks."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def write_results(
    out_dir: Path,
    results: list[CaseResult],
    summaries: list[MethodSummary],
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist raw per-case records and the aggregated summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "cases.jsonl").open("w") as handle:
        for record in results:
            handle.write(json.dumps(_jsonable(record.to_dict()), ensure_ascii=False) + "\n")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "summaries": [_jsonable(asdict(s)) for s in summaries],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------

# Metrics where a lower value is better, so tables can mark direction honestly.
LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "syntax_score_mae",
    "hausdorff95_mm",
    "hausdorff_max_mm",
    "mean_surface_distance_mm",
}

# The metric each task is ranked by, and the columns worth showing.
TASK_HEADLINE: dict[str, str] = {
    "cardiosyntax_scoring": "mae",
    "cca_segmentation": "dice",
    "arcade_segmentation": "f1",
    "arcade_stenosis": "f1",
}

TASK_COLUMNS: dict[str, list[str]] = {
    "cardiosyntax_scoring": ["mae", "rmse", "pearson_r", "tier_accuracy"],
    "cca_segmentation": ["dice", "cldice", "hausdorff95_mm", "precision", "recall"],
    "arcade_segmentation": [
        "f1",
        "f1_label_agnostic",
        "pixel_dice",
        "mean_matched_iou_label_agnostic",
        "precision",
        "recall",
        "label_set_precision",
        "label_set_recall",
    ],
    "arcade_stenosis": ["f1", "f1_at_iou025", "pixel_dice", "precision", "recall"],
}


def format_table(summaries: list[MethodSummary], task: str) -> str:
    """Render a fixed-width results table for one task."""
    rows = [s for s in summaries if s.task == task]
    if not rows:
        return f"({task}: no results)"

    columns = [
        c
        for c in TASK_COLUMNS.get(task, [])
        if any(c in s.metrics for s in rows)
    ]
    headline = TASK_HEADLINE.get(task)

    # Rank by the headline metric, respecting its direction. Methods with no
    # headline value sort last rather than being dropped.
    def sort_key(summary: MethodSummary) -> tuple[int, float]:
        if not headline or headline not in summary.metrics:
            return (1, 0.0)
        value = summary.metrics[headline]["mean"]
        return (0, value if headline in LOWER_IS_BETTER else -value)

    rows.sort(key=sort_key)

    header = f"{'method':<22} {'family':<11} {'n':>7}  " + "  ".join(
        f"{c:>16}" for c in columns
    )
    lines = [f"== {task} ==", header, "-" * len(header)]

    for summary in rows:
        count = f"{summary.n_ok}/{summary.n_total}"
        cells = []
        for column in columns:
            if column in summary.metrics:
                stats = summary.metrics[column]
                cells.append(f"{stats['mean']:8.4f}±{stats['sd']:<7.4f}")
            else:
                cells.append(f"{'-':>16}")
        flag = " *" if summary.cross_domain else ""
        lines.append(
            f"{summary.method + flag:<22} {summary.family:<11} {count:>7}  "
            + "  ".join(cells)
        )
        if summary.n_failed:
            reason = summary.errors[0] if summary.errors else "unknown"
            lines.append(f"{'':<22} {summary.n_failed} failed: {reason[:70]}")

    if any(s.cross_domain for s in rows):
        lines.append("")
        lines.append("* cross-domain: weights trained on a different dataset")
    return "\n".join(lines)
