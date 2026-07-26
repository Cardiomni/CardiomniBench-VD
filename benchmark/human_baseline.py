"""Human expert baseline for CardioSYNTAX scoring.

Why this exists
---------------
A SYNTAX score is a human judgement, not a physical measurement. The three
experts who scored these 60 studies disagree with each other by 5.71 points on
average (mean pairwise absolute difference), and their min-max spread averages
8.57 points. Reporting a model MAE without that context invites the reader to
compare against an implied ideal of 0, which does not exist for this task.

So every score-regression table gets a human reference row. It is the noise
floor: a model at the human level cannot be distinguished from a fourth expert.

How the reference is computed
-----------------------------
Leave-one-expert-out. For each study and each expert in turn, that expert's
score is treated as a "prediction" and the median of the remaining experts as the
"gold". This gives an unbiased estimate of how well one competent reader agrees
with a consensus of peers, measured exactly the way we measure models.

The naive alternative -- comparing each expert against the median of all three,
including themselves -- leaks the prediction into its own reference and reports a
human MAE that is too optimistic (3.68 instead of 5.01 on this data).

Only studies with at least two expert scores contribute, since a single score has
no peer consensus to compare against.
"""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

#: Threshold separating intermediate from high anatomical complexity. Clinically
#: this is the boundary where PCI-vs-CABG discussion changes, so tier agreement
#: matters independently of the raw score error.
HIGH_COMPLEXITY_THRESHOLD = 22.0


@dataclass(frozen=True)
class HumanBaseline:
    """Inter-reader agreement expressed in the same metrics used for models."""

    mae: float
    rmse: float
    median_ae: float
    tier_agreement: float
    n_comparisons: int
    n_studies: int

    #: Descriptive statistics of expert disagreement, useful in prose.
    mean_pairwise_ae: float
    mean_spread: float
    median_spread: float

    def as_row(self) -> dict[str, Any]:
        """Render as a table row that lines up with model result rows."""
        return {
            "method": "human expert (leave-one-out)",
            "family": "human",
            "n": self.n_studies,
            "mae": self.mae,
            "rmse": self.rmse,
            "median_ae": self.median_ae,
            "tier_accuracy": self.tier_agreement,
            "note": (
                f"{self.n_comparisons} expert-vs-peer-consensus comparisons; "
                f"mean pairwise disagreement {self.mean_pairwise_ae:.2f}"
            ),
        }


def _iter_expert_scores(
    cases_dir: Path,
) -> Iterable[tuple[float, list[float]]]:
    """Yield ``(consensus_gold, expert_scores)`` for studies with >=2 experts."""
    for task_file in sorted(cases_dir.glob("*/task.yaml")):
        try:
            doc = yaml.safe_load(task_file.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        gold = (doc or {}).get("gold_standard") or {}
        experts = gold.get("expert_scores") or []
        experts = [float(e) for e in experts if isinstance(e, (int, float))]
        consensus = gold.get("syntax_score")
        if len(experts) >= 2 and isinstance(consensus, (int, float)):
            yield float(consensus), experts


def compute_human_baseline(cases_dir: str | Path) -> HumanBaseline | None:
    """Measure inter-reader agreement on the CardioSYNTAX studies.

    Returns ``None`` when no study carries multiple expert scores, rather than
    fabricating a reference row from insufficient annotation.
    """
    cases_dir = Path(cases_dir)
    studies = list(_iter_expert_scores(cases_dir))
    if not studies:
        return None

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    tier_hits: list[bool] = []
    pairwise: list[float] = []
    spreads: list[float] = []

    for _consensus, experts in studies:
        spreads.append(max(experts) - min(experts))
        pairwise.extend(
            abs(a - b) for a, b in itertools.combinations(experts, 2)
        )
        # Leave-one-out: each expert scored against the median of the others.
        for idx, held_out in enumerate(experts):
            peers = [e for j, e in enumerate(experts) if j != idx]
            reference = statistics.median(peers)
            abs_errors.append(abs(held_out - reference))
            sq_errors.append((held_out - reference) ** 2)
            tier_hits.append(
                (held_out > HIGH_COMPLEXITY_THRESHOLD)
                == (reference > HIGH_COMPLEXITY_THRESHOLD)
            )

    return HumanBaseline(
        mae=statistics.fmean(abs_errors),
        rmse=math.sqrt(statistics.fmean(sq_errors)),
        median_ae=statistics.median(abs_errors),
        tier_agreement=statistics.fmean(1.0 if h else 0.0 for h in tier_hits),
        n_comparisons=len(abs_errors),
        n_studies=len(studies),
        mean_pairwise_ae=statistics.fmean(pairwise) if pairwise else 0.0,
        mean_spread=statistics.fmean(spreads),
        median_spread=statistics.median(spreads),
    )


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    import json

    result = compute_human_baseline("data/tasks/cardiosyntax_scoring/cases")
    if result is None:
        print("no multi-expert annotations found")
    else:
        print(json.dumps(result.as_row(), indent=2, ensure_ascii=False))
