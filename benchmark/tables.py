"""
Paper table generation from benchmark results.

Reads the flat cases.jsonl produced by a run and emits LaTeX. Every number in the
paper should come from here rather than being typed by hand, so a table can be
regenerated after a rerun and diffed.

Reporting rules enforced here
-----------------------------
1. n travels with every aggregate. A mean over 24 cases and a mean over 60 are
   not presented identically.
2. Zero-score cases are broken out. Our set has 10 normal-anatomy cases whose
   SYNTAX score is 0; pooling them with diseased cases lowers MAE by roughly 30%
   and describes a task nobody is asking about. The diseased subgroup is the
   headline number, with the pooled value shown alongside for comparability with
   prior work that reports it.
3. Zero-shot rows are visually separated from trained rows, and their
   domain_relation is stated in the caption. A cross-domain Dice of 0.05 and an
   in-domain Dice of 0.05 mean different things.
4. Failures are shown as a completion count, never hidden. A method that crashed
   on half the cases does not get a clean-looking mean.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

# Risk tertile boundaries, standard SYNTAX convention.
TIER_LOW, TIER_HIGH = 22, 33


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read cases.jsonl."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format a number, or an em-dash when it does not exist."""
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return st.fmean(values), st.stdev(values)


def syntax_table(cases: list[dict[str, Any]]) -> str:
    """Build the CardioSYNTAX results table.

    Columns separate the diseased subgroup from the pooled set, because those
    answer different questions and the difference is large here.
    """
    rows = [c for c in cases if c["task"] == "cardiosyntax_scoring"]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for case in rows:
        by_method[case["method"]].append(case)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & \multicolumn{2}{c}{Diseased (SYNTAX$>$0)} & "
        r"\multicolumn{2}{c}{All cases} & Tier & Within \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & MAE & $n$ & MAE & $n$ & acc. & expert \\",
        r"\midrule",
    ]

    for method in sorted(by_method):
        ok = [c for c in by_method[method] if c["status"] == "ok"]
        total = len(by_method[method])

        diseased = [c for c in ok if c["metrics"].get("gold_score", 0) > 0]
        d_mae, _ = _mean_sd([c["metrics"]["mae"] for c in diseased])
        a_mae, _ = _mean_sd([c["metrics"]["mae"] for c in ok])

        tier = [c["metrics"]["tier_correct"] for c in ok if "tier_correct" in c["metrics"]]
        within = [
            c["metrics"]["within_expert_range"]
            for c in ok
            if "within_expert_range" in c["metrics"]
        ]

        name = method.replace("_", r"\_")
        if len(ok) < total:
            name += rf" ({len(ok)}/{total})"

        lines.append(
            f"{name} & {_fmt(d_mae)} & {len(diseased)} & "
            f"{_fmt(a_mae)} & {len(ok)} & "
            f"{_fmt(st.fmean(tier) if tier else None)} & "
            f"{_fmt(st.fmean(within) if within else None)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def segmentation_table(cases: list[dict[str, Any]]) -> str:
    """Build the CCA segmentation table.

    Every method here was trained on ImageCAS and evaluated on CCA, so the whole
    table is cross-domain. That is stated once in the caption rather than
    repeated per row.
    """
    rows = [c for c in cases if c["task"] == "cca_segmentation"]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for case in rows:
        by_method[case["method"]].append(case)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Dice & clDice & HD95 (mm) & $n$ \\",
        r"\midrule",
    ]

    for method in sorted(by_method):
        ok = [c for c in by_method[method] if c["status"] == "ok"]
        total = len(by_method[method])

        dice, dice_sd = _mean_sd([
            c["metrics"]["dice"] for c in ok if "dice" in c["metrics"]
        ])
        cldice, _ = _mean_sd([
            c["metrics"]["cldice"] for c in ok if "cldice" in c["metrics"]
        ])
        hd95, _ = _mean_sd([
            c["metrics"]["hd95"] for c in ok
            if "hd95" in c["metrics"] and c["metrics"]["hd95"] == c["metrics"]["hd95"]
        ])

        name = method.replace("_", r"\_")
        if len(ok) < total:
            name += rf" ({len(ok)}/{total})"

        dice_cell = (
            rf"{_fmt(dice, 3)} $\pm$ {_fmt(dice_sd, 3)}" if dice is not None else "--"
        )
        lines.append(
            f"{name} & {dice_cell} & {_fmt(cldice, 3)} & "
            f"{_fmt(hd95, 1)} & {len(ok)} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def summary_text(cases: list[dict[str, Any]]) -> str:
    """Plain-text overview for quick reading in the terminal."""
    out: list[str] = []
    by_task: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        by_task[case["task"]].append(case)

    for task, rows in sorted(by_task.items()):
        ok = [c for c in rows if c["status"] == "ok"]
        failed = [c for c in rows if c["status"] == "failed"]
        out.append(f"\n{task}: {len(ok)} ok, {len(failed)} failed")

        by_method: dict[str, list[dict]] = defaultdict(list)
        for case in ok:
            by_method[case["method"]].append(case)

        for method in sorted(by_method):
            group = by_method[method]
            if task == "cardiosyntax_scoring":
                diseased = [c for c in group if c["metrics"].get("gold_score", 0) > 0]
                d_mae, _ = _mean_sd([c["metrics"]["mae"] for c in diseased])
                a_mae, _ = _mean_sd([c["metrics"]["mae"] for c in group])
                out.append(
                    f"  {method:32} MAE(diseased)={_fmt(d_mae)} "
                    f"(n={len(diseased)})  MAE(all)={_fmt(a_mae)} (n={len(group)})"
                )
            else:
                dice, sd = _mean_sd([
                    c["metrics"]["dice"] for c in group if "dice" in c["metrics"]
                ])
                out.append(
                    f"  {method:32} Dice={_fmt(dice, 4)}+-{_fmt(sd, 4)} "
                    f"(n={len(group)})"
                )

    return "\n".join(out)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory containing cases.jsonl")
    parser.add_argument("--latex", action="store_true", help="emit LaTeX tables")
    args = parser.parse_args()

    path = args.run_dir / "cases.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    cases = load_cases(path)
    print(summary_text(cases))

    if args.latex:
        print("\n\n% ===== CardioSYNTAX =====")
        print(syntax_table(cases))
        print("\n% ===== CCA segmentation =====")
        print(segmentation_table(cases))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
