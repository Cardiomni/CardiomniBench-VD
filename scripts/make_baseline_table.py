"""Collect finished baseline runs into one reproducible experimental table.

Design constraints
------------------
Every number here is zero-shot. No threshold, no calibration coefficient, and no
hyperparameter was selected against these cases: the 60 CardioSYNTAX and 20 CCA
cases are the test set, and there is no training or validation split. Decisions
use each checkpoint's native argmax.

The one exception is marked explicitly in the table: CardioSYNTAX ships
per-fold linear coefficients in ``scaling_coeffs.json`` fitted on the authors'
own training data. That is part of the released checkpoint, not a knob tuned on
our cases, so it stays zero-shot with respect to this benchmark.

Methods that could not be run are listed with the verified reason rather than
omitted, so the table cannot be misread as "all published baselines were
evaluated".

Metric names are read from the run output, not assumed, because the segmentation
and scoring tasks emit different keys.

Usage
-----
    python -m scripts.make_baseline_table --runs-root runs --out results/baseline_table.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

# Published weights that exist but cannot be evaluated on this task, with the
# reason each was ruled out. Keeping them visible is part of the result.
NOT_RUN: list[dict[str, str]] = [
    {
        "method": "coronary_att_mamba2",
        "task": "cca_segmentation",
        "reason": (
            "Checkpoint expects 2 input channels (CT + Frangi vesselness); first "
            "conv is (32, 2, 7, 7, 7). Upstream model factory is not available "
            "locally and the Frangi parameters are not recorded in the release, "
            "so preprocessing cannot be reproduced faithfully."
        ),
    },
    {
        "method": "coronary_umamba",
        "task": "cca_segmentation",
        "reason": (
            "plans.json declares UNet_class_name=PlainConvUNet, but the checkpoint's "
            "encoder is Mamba SSM (mamba_layer.mamba.A_log / .D / .dt_proj): loading "
            "into PlainConvUNet gives 282 missing / 387 unexpected keys. The real "
            "class needs mamba_ssm + causal_conv1d CUDA extensions, not installed."
        ),
    },
    {
        "method": "coronary_cm_unet",
        "task": "cca_segmentation",
        "reason": (
            "2D X-ray angiography model (256x256 input), not applicable to 3D CTA "
            "volumes. Its natural task is arcade_segmentation (2D XCA), which is "
            "outside the current CCA + CardioSYNTAX scope."
        ),
    },
]

# Segmentation: per-case metrics aggregated to mean ± SD across cases.
SEG_COLUMNS: list[tuple[str, str, int]] = [
    ("dice", "Dice", 4),
    ("cldice", "clDice", 4),
    ("precision", "Precision", 4),
    ("recall", "Recall", 4),
    ("hausdorff95_mm", "HD95 (mm)", 2),
]

# Scoring: taken from summary.json, which already aggregates over all 60 cases.
# Per-case averaging would give a different (and wrong) number for RMSE and r.
SYNTAX_FIELDS: list[tuple[str, str, int]] = [
    ("mae", "MAE", 2),
    ("median_ae", "Median AE", 2),
    ("rmse", "RMSE", 2),
    ("pearson_r", "Pearson r", 3),
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_cases(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "cases.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def agg(cases: list[dict[str, Any]], key: str) -> tuple[float, float, int] | None:
    vals = [
        float(v)
        for c in cases
        if isinstance(v := (c.get("metrics") or {}).get(key), (int, float))
    ]
    if not vals:
        return None
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), sd, len(vals)


def fmt_agg(stat: tuple[float, float, int] | None, digits: int) -> str:
    if stat is None:
        return "n/a"
    mean, sd, _ = stat
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def fmt_num(val: Any, digits: int) -> str:
    if not isinstance(val, (int, float)):
        return "n/a"
    return f"{float(val):.{digits}f}"


def render_segmentation(runs_root: Path) -> list[str]:
    rows: list[tuple[str, list[dict[str, Any]]]] = []
    for run_dir in sorted(runs_root.glob("baseline_cca_*")):
        cases = [c for c in read_cases(run_dir) if c.get("status") == "ok"]
        failed = [c for c in read_cases(run_dir) if c.get("status") != "ok"]
        if cases or failed:
            name = (cases or failed)[0].get("method") or run_dir.name
            rows.append((name, cases))
    if not rows:
        return []

    out = ["## CCA coronary artery segmentation", ""]
    out.append("Cross-domain zero-shot. Decision rule: native argmax, no threshold.")
    out.append("")
    headers = ["Method", "n"] + [label for _, label, _ in SEG_COLUMNS]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "---|" * len(headers))
    for name, cases in sorted(rows):
        cells = [f"`{name}`", str(len(cases))]
        cells += [fmt_agg(agg(cases, key), d) for key, _, d in SEG_COLUMNS]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def render_scoring(runs_root: Path) -> list[str]:
    # Prefer summary.json: RMSE and Pearson r are not averages of per-case values.
    candidates: list[tuple[str, dict[str, Any]]] = []
    for run_dir in sorted(runs_root.glob("baseline_cardiosyntax*")):
        summary = read_json(run_dir / "summary.json")
        if summary:
            label = "cardiosyntax_r3d_calibrated" if summary.get("calibrated") else "cardiosyntax_r3d"
            candidates.append((label, summary))
    for legacy in ("cardiosyntax_r3d_baseline", "cardiosyntax_r3d_calibrated"):
        summary = read_json(runs_root / legacy / "summary.json")
        if summary and not any(lbl for lbl, _ in candidates if lbl == legacy):
            label = "cardiosyntax_r3d_calibrated" if summary.get("calibrated") else "cardiosyntax_r3d"
            if label not in {l for l, _ in candidates}:
                candidates.append((label, summary))
    if not candidates:
        return []

    out = ["## CardioSYNTAX score regression", ""]
    out.append(
        "Cross-domain zero-shot, 5-fold ensemble. SYNTAX score is a continuous "
        "value; the >22 row reports the PCI/CABG decision boundary."
    )
    out.append("")
    headers = ["Method", "n"] + [label for _, label, _ in SYNTAX_FIELDS] + [
        "Sens. >22",
        "Spec. >22",
        "Bal. acc. >22",
        "In expert band",
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "---|" * len(headers))
    for label, s in candidates:
        hc = s.get("high_complexity") or {}
        band = s.get("within_expert_band") or {}
        cells = [f"`{label}`", str(s.get("n_scored", s.get("n_cases", "?")))]
        cells += [fmt_num(s.get(key), d) for key, _, d in SYNTAX_FIELDS]
        cells += [
            fmt_num(hc.get("sensitivity"), 3),
            fmt_num(hc.get("specificity"), 3),
            fmt_num(hc.get("balanced_accuracy"), 3),
            fmt_num(band.get("rate"), 3),
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "`cardiosyntax_r3d_calibrated` applies the per-fold linear coefficients "
        "shipped in the authors' `scaling_coeffs.json` (a in 1.11-1.65), fitted "
        "on their training data and released with the weights. It is not a "
        "threshold tuned on these cases."
    )
    out.append("")
    return out


def render_expert_spread(runs_root: Path) -> list[str]:
    """Inter-observer spread, which bounds what any model error can mean."""
    cases = read_cases(runs_root / "baseline_cardiosyntax")
    spreads = [
        float(v)
        for c in cases
        if isinstance(v := (c.get("metrics") or {}).get("expert_spread"), (int, float))
    ]
    if not spreads:
        return []
    spreads.sort()
    over10 = sum(1 for s in spreads if s > 10)
    return [
        "## Inter-observer reference",
        "",
        f"Across {len(spreads)} cases the annotating cardiologists disagree by a "
        f"median of {statistics.median(spreads):.1f} SYNTAX points "
        f"(max {max(spreads):.1f}; {over10} cases above 10). Model error should be "
        "read against this spread, not against an assumption of exact ground truth.",
        "",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/baseline_table.md"))
    args = ap.parse_args()

    lines = [
        "# Baseline experimental table",
        "",
        "All rows are zero-shot on the full test set: 60 CardioSYNTAX cases and "
        "20 CCA segmentation cases. There is no training or validation split, so "
        "no threshold or hyperparameter was selected against these cases. "
        "Segmentation decisions use each checkpoint's native argmax.",
        "",
        "Every method here is a **cross-domain specialist**: trained on a "
        "different dataset and applied unchanged. They serve as tools and "
        "reference points, not as competing systems.",
        "",
    ]
    lines += render_scoring(args.runs_root)
    lines += render_segmentation(args.runs_root)
    lines += render_expert_spread(args.runs_root)

    if NOT_RUN:
        lines += ["## Published weights not evaluated", ""]
        lines.append("| Method | Task | Reason |")
        lines.append("|---|---|---|")
        for entry in NOT_RUN:
            lines.append(
                f"| `{entry['method']}` | {entry['task']} | {entry['reason']} |"
            )
        lines.append("")
        lines.append(
            "These are listed so the table is not read as covering every "
            "published baseline. Producing a number for either would require "
            "guessing preprocessing that the releases do not document."
        )
        lines.append("")

    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
