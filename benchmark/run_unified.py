"""
Unified benchmark runner for CardiomniBench-VD.

One command evaluates all methods with available weights on the selected tasks.
Results are written as (a) per-case JSONL for detailed analysis and (b) a summary
JSON with aggregate statistics for the paper tables.

Usage:
    python -m benchmark.run_unified [--methods NAME ...] [--tasks TASK ...] \\
        [--device DEVICE] [--limit N]

Omitting --methods runs all methods with verified weights. Omitting --tasks runs
the volumetric pair (cardiosyntax_scoring, cca_segmentation); the 2D ARCADE tasks
are opt-in since they score instance lists rather than volume masks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from benchmark.core import Prediction, Task
from benchmark.io_spec import load_case_input
from benchmark.results import CaseResult, MethodSummary, aggregate, format_table, write_results
from benchmark.scoring import load_gold, score_case, set_level_metrics
from benchmark.harnesses import ALL_HARNESSES
from benchmark.specialists import ALL_SPECIALISTS
from benchmark.vlms import ALL_VLMS

#: Every evaluable method. Specialists first (trained models and upper-bound
#: tools), then VLMs (base models unaided), then harnesses (orchestration over a
#: fixed base + tools).
ALL_METHODS = ALL_SPECIALISTS + ALL_VLMS + ALL_HARNESSES

REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_cases(task: Task) -> list[Path]:
    """Return every case directory for the given task, sorted by case_id."""
    task_root = REPO_ROOT / "data" / "tasks" / task.value / "cases"
    if not task_root.is_dir():
        return []
    cases = sorted(
        [p for p in task_root.iterdir() if p.is_dir() and (p / "task.yaml").exists()],
        key=lambda p: p.name,
    )
    return cases


def evaluate_case(
    method,
    case_dir: Path,
    output_dir: Path,
    device: str,
) -> CaseResult:
    """Run one method on one case and return a scored CaseResult."""
    started = time.time()
    case_input = None
    try:
        case_input = load_case_input(case_dir)
        prediction = method.predict(case_input, output_dir, device)
        prediction.validate()
        runtime = time.time() - started
    except Exception as exc:
        return CaseResult(
            method=method.name,
            # case_input may be None if loading itself failed, so fall back to
            # the directory name rather than masking the error with a KeyError.
            task=case_input.task.value if case_input else "unknown",
            case_id=case_input.case_id if case_input else case_dir.name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            runtime_s=time.time() - started,
        )

    # Score against gold. A scoring failure is recorded separately from an
    # inference failure, because the two have very different causes.
    try:
        gold = load_gold(case_dir)
        metrics = score_case(gold, prediction, output_dir)
    except Exception as exc:
        return CaseResult(
            method=method.name,
            task=case_input.task.value,
            case_id=case_input.case_id,
            status="failed",
            error=f"scoring failed: {type(exc).__name__}: {exc}",
            prediction=prediction.to_dict(),
            runtime_s=runtime,
        )

    return CaseResult(
        method=method.name,
        task=case_input.task.value,
        case_id=case_input.case_id,
        status="ok",
        metrics=metrics,
        prediction=prediction.to_dict(),
        runtime_s=runtime,
    )


def run_all(
    methods: list,
    tasks: list[Task],
    device: str,
    output_dir: Path,
    limit: int | None = None,
) -> tuple[list[CaseResult], list[MethodSummary]]:
    """Evaluate every (method, task, case) triple and return results + summaries.

    Each result is appended to cases.jsonl as soon as it is produced. A run that
    dies at case 700 of 1360 therefore keeps its first 700 results, and progress
    is inspectable while the run is still going.
    """
    results: list[CaseResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / "cases.jsonl"
    stream = stream_path.open("a", encoding="utf-8")

    # Total work, for progress reporting.
    total_units = sum(
        len(discover_cases(task)) if limit is None else min(limit, len(discover_cases(task)))
        for task in tasks
        for m in methods
        if m.supports(task)
    )
    done = 0

    try:
        for task in tasks:
            cases = discover_cases(task)
            if limit is not None:
                cases = cases[:limit]
            if not cases:
                print(f"WARN: no cases for {task.value}", file=sys.stderr, flush=True)
                continue

            task_methods = [m for m in methods if m.supports(task)]
            print(f"\n{'='*66}", flush=True)
            print(
                f"{task.value}  {len(cases)} cases x {len(task_methods)} methods",
                flush=True,
            )
            print(f"{'='*66}", flush=True)

            for method in task_methods:
                ok, detail = method.check_available()
                if not ok:
                    print(f"[SKIP] {method.name}: {detail}", flush=True)
                    continue

                print(f"\n[RUN] {method.name} ({method.family.value})", flush=True)
                for case_dir in cases:
                    case_output = output_dir / method.name / task.value / case_dir.name
                    result = evaluate_case(method, case_dir, case_output, device)
                    results.append(result)

                    stream.write(json.dumps(result.to_dict(), default=str) + "\n")
                    stream.flush()

                    done += 1
                    print(f"@@progress {done / max(total_units, 1):.4f}", flush=True)

                    headline = "mae" if task is Task.CARDIOSYNTAX_SCORING else "dice"
                    detail_str = ""
                    if result.ok and headline in result.metrics:
                        detail_str = f"  {headline}={result.metrics[headline]:.4f}"
                    elif not result.ok:
                        detail_str = f"  {(result.error or '').splitlines()[0][:70]}"
                    print(
                        f"  [{'ok' if result.ok else 'FAIL'}] {case_dir.name}"
                        f"  {result.runtime_s:.1f}s{detail_str}",
                        flush=True,
                    )
    finally:
        stream.close()

    # Aggregate per (method, task).
    summaries: list[MethodSummary] = []
    for method in methods:
        for task in tasks:
            if not method.supports(task):
                continue
            summary = aggregate(
                results,
                method.name,
                task.value,
                family=method.family.value,
                source=method.provenance.source,
                reported=method.provenance.reported_metric,
                cross_domain=(method.provenance.domain_relation.value == "cross_domain"),
            )
            if not summary.n_total:
                continue
            # Set-level metrics (correlation, decision sensitivity, expert
            # agreement) are properties of the whole case set and cannot be
            # obtained by averaging per-case values.
            set_metrics = set_level_metrics(
                task,
                [r.metrics for r in results
                 if r.method == method.name and r.task == task.value and r.ok],
            )
            for name, value in set_metrics.items():
                summary.metrics.setdefault(
                    name,
                    {"mean": value, "sd": 0.0, "median": value,
                     "min": value, "max": value, "n": summary.n_ok},
                )
            summaries.append(summary)

    return results, summaries


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Separate from :func:`main` so tests can assert that the exposed choices stay
    in step with the Task enum without executing a run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="*",
        help="Method names to evaluate (default: all with weights)",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=[t.value for t in Task],
        default=[Task.CARDIOSYNTAX_SCORING.value, Task.CCA_SEGMENTATION.value],
        help="Tasks to run (default: CardioSYNTAX and CCA)",
    )
    parser.add_argument("--device", default="cuda:7", help="torch device")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "unified_benchmark",
        help="Output root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit cases per task (for smoke tests)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tasks = [Task(t) for t in args.tasks]

    # Resolve methods: if --methods is given, filter; else use all available.
    if args.methods:
        by_name = {m.name: m for m in ALL_METHODS}
        unknown = [n for n in args.methods if n not in by_name]
        if unknown:
            print(f"ERROR: unknown methods: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(by_name))}", file=sys.stderr)
            return 1
        selected = [by_name[n] for n in args.methods]
    else:
        selected = []
        skipped = []
        for m in ALL_METHODS:
            ok, detail = m.check_available()
            if ok:
                selected.append(m)
            else:
                skipped.append((m, detail))
        if skipped:
            print(f"Skipping {len(skipped)} methods without weights:\\n")
            for method, reason in skipped:
                print(f"  {method.name:24} {reason}")

    print(f"\\n{'='*60}")
    print(f"CardiomniBench-VD unified runner")
    print(f"{'='*60}")
    print(f"methods: {len(selected)}")
    print(f"tasks:   {', '.join(t.value for t in tasks)}")
    print(f"device:  {args.device}")
    print(f"output:  {args.output_dir}\\n")

    results, summaries = run_all(selected, tasks, args.device, args.output_dir, args.limit)

    write_results(
        args.output_dir,
        results,
        summaries,
        meta={"device": args.device, "tasks": [t.value for t in tasks]},
    )

    print(f"\\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\\n")
    for task in tasks:
        print(format_table(summaries, task.value))
        print()

    print(f"Wrote {len(results)} case results to {args.output_dir / 'cases.jsonl'}")
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")

    failures = [r for r in results if not r.ok]
    if failures:
        print(f"\\n{len(failures)} failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
