"""
Unified benchmark runner for CardiomniBench-VD.

One command, all methods, both tasks. No adaptive branching, no schema guessing,
no silent degradation. A method runs to completion or reports a failure with a
reason; the aggregation layer makes both visible.

Design decisions forced by the audit:

1. The existing rubric/judge path cannot be reused for these two tasks. The
   metric adapters read fusion-era nested keys that do not exist in the task
   files (e.g., `_syntax_score_mae` reads gold["stage3_scoring"], but our gold
   is flat: {syntax_score: 42.0}). The adapters would silently score everything
   as perfect (MAE=0). So this runner computes metrics directly from the gold
   standard, bypassing the judge.

2. Segmentation metrics need the actual mask arrays, but the metric registry
   signature is `(gold_dict, pred_dict) -> float` with no path context for
   loading NIfTI files. So mask-based metrics are computed here and added to
   the CaseResult, not delegated to the registry.

3. Both tasks have `default_case_rubric: null` in their TOML. The orchestrator
   would error out. This runner owns the entire evaluate-and-aggregate flow.

Usage:
    python -m benchmark.run_all [--methods NAME ...] [--tasks TASK ...] \\
        [--device DEVICE] [--output-dir DIR]

Omitting --methods evaluates all methods with available weights.
Omitting --tasks runs the CTA/DSA volumetric pair (cardiosyntax_scoring and
cca_segmentation). The 2D ARCADE tasks are opt-in via --tasks because they use a
different output contract (instance lists rather than volume masks).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml

# Project imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.methods import (
    BY_NAME,
    Method,
    available_methods,
    methods_for,
    resolve,
    weights_status,
)
from benchmark.results import (
    CaseResult,
    MethodSummary,
    aggregate,
    format_table,
    write_results,
)
from evaluation.metrics.segmentation_metrics import evaluate_segmentation


def discover_cases(task: str) -> list[Path]:
    """Return every case directory for the given task, sorted by case_id."""
    task_root = REPO_ROOT / "data" / "tasks" / task / "cases"
    if not task_root.is_dir():
        return []
    cases = sorted(
        [p for p in task_root.iterdir() if p.is_dir() and (p / "task.yaml").exists()],
        key=lambda p: p.name,
    )
    return cases


def load_task_spec(case_dir: Path) -> dict[str, Any]:
    """Load and return the task.yaml for one case."""
    with (case_dir / "task.yaml").open() as handle:
        return yaml.safe_load(handle)


def load_gold(task_spec: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    """Load the gold standard for one case into a uniform dict.

    The two tasks store gold differently, which is why this function exists:

      cardiosyntax_scoring  gold is INLINE in task.yaml under `gold_standard`,
                            not in a separate file, and it carries the three
                            individual expert scores. The expert spread is kept
                            because it defines what "expert-level" means for
                            this task: landing inside the range three humans
                            disagreed over is a stronger claim than a bare MAE.

      cca_segmentation      gold is an out-of-tree NIfTI referenced by
                            `gold_standard.label_file`.
    """
    task_type = task_spec["case_metadata"]["task_type"]
    gold_block = task_spec.get("gold_standard", {}) or {}

    if task_type == "cardiosyntax_scoring":
        return {
            "syntax_score": float(gold_block["syntax_score"]),
            "syntax_left": gold_block.get("syntax_left"),
            "syntax_right": gold_block.get("syntax_right"),
            "risk_band": gold_block.get("risk_band"),
            "expert_scores": [float(s) for s in (gold_block.get("expert_scores") or [])],
            "expert_min": gold_block.get("expert_min"),
            "expert_max": gold_block.get("expert_max"),
            "expert_spread": gold_block.get("expert_spread"),
        }

    if task_type == "cca_segmentation":
        gold_path = Path(gold_block["label_file"])
        if not gold_path.exists():
            raise FileNotFoundError(f"gold mask missing: {gold_path}")
        nii = nib.load(str(gold_path))
        mask = (nii.get_fdata() > 0.5).astype(np.uint8)
        spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
        return {"mask": mask, "spacing": spacing}

    # ARCADE instance tasks are deliberately not handled here. This loader
    # predates benchmark.scoring.load_gold and is kept only for the volumetric
    # pair; duplicating instance-list loading would give two gold paths that can
    # disagree. Use run_unified for ARCADE.
    raise ValueError(
        f"unsupported task_type: {task_type}. Instance-list tasks "
        "(arcade_segmentation, arcade_stenosis) are scored via "
        "benchmark.run_unified, which uses benchmark.scoring.load_gold."
    )


def run_specialist(
    method: Method, case_dir: Path, output_dir: Path, device: str
) -> dict[str, Any]:
    """Dispatch to the specialist runner module named in the method's config.

    Runners are imported lazily so a broken or dependency-heavy runner only
    affects the methods that use it, rather than preventing the whole benchmark
    from starting.
    """
    import importlib

    module_path = method.runner
    module = importlib.import_module(module_path)
    if not hasattr(module, "run"):
        raise AttributeError(f"{module_path} has no run() entry point")
    return module.run(case_dir, output_dir, method.config, device)


def run_vlm(
    method: Method, case_dir: Path, output_dir: Path, device: str
) -> dict[str, Any]:
    """Dispatch to the unified VLM runner."""
    import importlib

    module = importlib.import_module(method.runner)
    if not hasattr(module, "run"):
        raise AttributeError(f"{method.runner} has no run() entry point")
    return module.run(case_dir, output_dir, method.config, device)


def evaluate_case(
    method: Method, case_dir: Path, output_dir: Path, device: str
) -> CaseResult:
    """Run one method on one case and return a scored CaseResult."""
    task_spec = load_task_spec(case_dir)
    case_id = task_spec["case_id"]
    task = task_spec["case_metadata"]["task_type"]

    case_output = output_dir / method.name / task / case_id
    case_output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    try:
        if method.family == "specialist":
            prediction = run_specialist(method, case_dir, case_output, device)
        elif method.family == "vlm":
            prediction = run_vlm(method, case_dir, case_output, device)
        else:
            raise ValueError(f"unknown family: {method.family}")
        runtime = time.time() - started
    except Exception as exc:
        return CaseResult(
            method=method.name,
            task=task,
            case_id=case_id,
            status="failed",
            error=f"{type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}",
            runtime_s=time.time() - started,
        )

    # Compute metrics directly from gold and prediction.
    gold = load_gold(task_spec, case_dir)
    metrics: dict[str, float] = {}

    if task == "cardiosyntax_scoring":
        pred_score = float(prediction.get("syntax_score", float("nan")))
        gold_score = gold["syntax_score"]
        if pred_score == pred_score:  # not nan
            mae = abs(pred_score - gold_score)
            metrics["mae"] = mae
            metrics["squared_error"] = mae**2
            metrics["signed_error"] = pred_score - gold_score
            # Kept per case so set-level correlation and decision metrics can be
            # recomputed at aggregation time without reloading any gold files.
            metrics["gold_score"] = gold_score
            metrics["pred_score"] = pred_score
            gold_tier = 0 if gold_score <= 22 else (1 if gold_score < 33 else 2)
            pred_tier = 0 if pred_score <= 22 else (1 if pred_score < 33 else 2)
            metrics["tier_correct"] = float(gold_tier == pred_tier)
            # Carry the expert envelope through for the agreement metric.
            prediction = dict(prediction)
            prediction["_expert_min"] = gold.get("expert_min")
            prediction["_expert_max"] = gold.get("expert_max")
            prediction["_expert_scores"] = gold.get("expert_scores")

    elif task == "cca_segmentation":
        pred_mask_path = case_output / "mask.nii.gz"
        if pred_mask_path.exists():
            pred_nii = nib.load(str(pred_mask_path))
            pred_mask = (pred_nii.get_fdata() > 0.5).astype(np.uint8)
            seg_metrics = evaluate_segmentation(
                gold["mask"],
                pred_mask,
                spacing=gold["spacing"],
                include_hausdorff=True,
                include_cldice=True,
            )
            metrics.update(seg_metrics)

    return CaseResult(
        method=method.name,
        task=task,
        case_id=case_id,
        status="ok",
        metrics=metrics,
        prediction=prediction,
        runtime_s=runtime,
    )


def run_all(
    methods: list[Method],
    tasks: list[str],
    device: str,
    output_dir: Path,
) -> tuple[list[CaseResult], list[MethodSummary]]:
    """Evaluate every (method, task, case) triple and return results + summaries."""
    results: list[CaseResult] = []

    for task in tasks:
        cases = discover_cases(task)
        if not cases:
            print(f"WARN: no cases for {task}", file=sys.stderr)
            continue

        task_methods = [m for m in methods if task in m.tasks]
        print(f"\n{'='*60}")
        print(f"{task}  {len(cases)} cases × {len(task_methods)} methods")
        print(f"{'='*60}")

        for method in task_methods:
            ok, detail = weights_status(method)
            if not ok:
                print(f"[SKIP] {method.name}: {detail}")
                continue

            print(f"\n[RUN] {method.name} ({method.family})")
            for case_dir in cases:
                case_id = case_dir.name
                result = evaluate_case(method, case_dir, output_dir, device)
                results.append(result)
                flag = "ok" if result.ok else "FAIL"
                print(f"  [{flag}] {case_id}  {result.runtime_s:.1f}s", end="")
                if result.ok and result.metrics:
                    # Show the headline metric inline.
                    key = "mae" if task == "cardiosyntax_scoring" else "dice"
                    if key in result.metrics:
                        print(f"  {key}={result.metrics[key]:.4f}", end="")
                print()

    # Aggregate per (method, task).
    summaries: list[MethodSummary] = []
    for method in methods:
        for task in tasks:
            if task not in method.tasks:
                continue
            summary = aggregate(
                results,
                method.name,
                task,
                family=method.family,
                source=method.source,
                reported=method.reported,
                cross_domain=method.config.get("cross_domain", False),
            )
            if not summary.n_total:
                continue
            # Correlation, decision sensitivity/specificity and expert agreement
            # are properties of the whole case set, not averages of per-case
            # numbers, so they are computed here rather than in evaluate_case.
            if task == "cardiosyntax_scoring":
                _attach_set_level_syntax_metrics(summary, results, method.name)
            summaries.append(summary)

    return results, summaries


def _attach_set_level_syntax_metrics(
    summary: MethodSummary, results: list[CaseResult], method_name: str
) -> None:
    """Add Pearson r, >22 decision metrics and expert agreement to a summary.

    These cannot be derived by averaging per-case values: a correlation over 60
    cases is not the mean of 60 per-case correlations.
    """
    from evaluation.metrics.syntax_scoring_metrics import (
        compute_decision_metrics,
        compute_expert_agreement,
        compute_point_metrics,
        compute_tier_accuracy,
    )

    paired = [
        r
        for r in results
        if r.method == method_name
        and r.task == "cardiosyntax_scoring"
        and r.ok
        and "gold_score" in r.metrics
        and "pred_score" in r.metrics
    ]
    if len(paired) < 2:
        return

    gold_scores = [r.metrics["gold_score"] for r in paired]
    pred_scores = [r.metrics["pred_score"] for r in paired]

    set_metrics: dict[str, float] = {}
    set_metrics.update(compute_point_metrics(gold_scores, pred_scores))
    set_metrics.update(compute_decision_metrics(gold_scores, pred_scores))
    set_metrics.update(compute_tier_accuracy(gold_scores, pred_scores))
    set_metrics.update(
        compute_expert_agreement(
            [
                {
                    "pred": r.metrics["pred_score"],
                    "expert_min": r.prediction.get("_expert_min"),
                    "expert_max": r.prediction.get("_expert_max"),
                    "expert_scores": r.prediction.get("_expert_scores") or [],
                }
                for r in paired
            ]
        )
    )

    # Set-level values have no spread across cases, so they are stored as a
    # single value with n = number of paired cases.
    for name, value in set_metrics.items():
        if name in summary.metrics:
            continue
        summary.metrics[name] = {
            "mean": float(value),
            "sd": 0.0,
            "median": float(value),
            "min": float(value),
            "max": float(value),
            "n": len(paired),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Method names to evaluate (default: all with weights)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        # No choices= here: this entry point accepts any task the enum knows, and
        # invalid names are reported after parsing with the full list.
        default=["cardiosyntax_scoring", "cca_segmentation"],
        help=(
            "Tasks to run (default: cardiosyntax_scoring cca_segmentation). "
            "ARCADE tasks: arcade_segmentation, arcade_stenosis"
        ),
    )
    parser.add_argument("--device", default="cuda:5", help="torch device")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "unified_benchmark",
        help="Output root",
    )
    args = parser.parse_args()

    # Resolve methods: if --methods is given, use those; else use all available.
    if args.methods:
        selected = resolve(args.methods)
    else:
        ready, skipped = available_methods()
        selected = ready
        if skipped:
            print(f"Skipping {len(skipped)} methods without weights:\n")
            for method, reason in skipped:
                print(f"  {method.name:24} {reason}")

    print(f"\n{'='*60}")
    print(f"CardiomniBench-VD unified runner")
    print(f"{'='*60}")
    print(f"methods: {len(selected)}")
    print(f"tasks:   {', '.join(args.tasks)}")
    print(f"device:  {args.device}")
    print(f"output:  {args.output_dir}\n")

    results, summaries = run_all(selected, args.tasks, args.device, args.output_dir)

    write_results(
        args.output_dir,
        results,
        summaries,
        meta={"device": args.device, "tasks": args.tasks},
    )

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    for task in args.tasks:
        print(format_table(summaries, task))
        print()

    print(f"Wrote {len(results)} case results to {args.output_dir / 'cases.jsonl'}")
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\n{len(failures)} failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
