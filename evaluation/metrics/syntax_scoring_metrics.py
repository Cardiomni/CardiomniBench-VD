"""
SYNTAX-score regression metrics for CardiomniBench-VD.

Used by the CardioSYNTAX scoring task. The SYNTAX score drives a real clinical
decision - roughly, >22 shifts the PCI-versus-CABG conversation - so a plain MAE
hides the failures that matter. This module reports four things:

  point accuracy       MAE / RMSE / Pearson r over all cases
  decision accuracy    sensitivity and specificity at the >22 threshold
  expert agreement     fraction of cases landing inside the range that three
                       human experts spanned on that same case
  ceiling behaviour    error broken out by gold severity, which is where a
                       regressor that has learned the mean gets caught

The expert-agreement number is the one worth arguing from. Three cardiologists
scoring the same angiogram disagreed by 9 points on the first case in this set,
so "MAE 6.9" is not obviously worse than a human. Reporting the inter-expert
range makes the comparison honest in both directions.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

# The clinically meaningful cut. SYNTAX <=22 is low tertile; above it, surgical
# revascularisation enters the discussion.
DECISION_THRESHOLD = 22.0

# Conventional SYNTAX risk tertiles.
RISK_TIERS: tuple[tuple[str, float, float], ...] = (
    ("low", 0.0, 22.0),
    ("intermediate", 22.0, 33.0),
    ("high", 33.0, float("inf")),
)


def risk_tier(score: float) -> str:
    """Map a SYNTAX score to its risk tertile."""
    for name, low, high in RISK_TIERS:
        if low <= score < high:
            return name
    return "high"


def compute_point_metrics(
    gold_scores: Sequence[float], pred_scores: Sequence[float]
) -> dict[str, float]:
    """MAE, RMSE, bias and Pearson correlation over paired scores."""
    if len(gold_scores) != len(pred_scores):
        raise ValueError(
            f"length mismatch: {len(gold_scores)} gold vs {len(pred_scores)} pred"
        )
    if not gold_scores:
        return {}

    errors = [p - g for g, p in zip(gold_scores, pred_scores)]
    absolute = [abs(e) for e in errors]

    metrics = {
        "mae": statistics.fmean(absolute),
        "rmse": (statistics.fmean([e * e for e in errors])) ** 0.5,
        # Signed mean error: separates "noisy" from "systematically low", which
        # is exactly the ceiling effect we expect from a saturating regressor.
        "bias": statistics.fmean(errors),
        "max_abs_error": max(absolute),
    }

    if len(gold_scores) > 1:
        try:
            metrics["pearson_r"] = _pearson(gold_scores, pred_scores)
        except (ZeroDivisionError, statistics.StatisticsError):
            # Degenerate when a method emits a constant prediction. That is a
            # finding, not an error, so record it explicitly.
            metrics["pearson_r"] = float("nan")
        metrics["pred_sd"] = statistics.stdev(pred_scores)
        metrics["gold_sd"] = statistics.stdev(gold_scores)

    # Prediction range versus gold range exposes a compressed output space: a
    # model whose max prediction sits far below the gold max cannot flag the
    # severe cases at all, no matter what its MAE says.
    metrics["pred_min"] = min(pred_scores)
    metrics["pred_max"] = max(pred_scores)
    metrics["gold_min"] = min(gold_scores)
    metrics["gold_max"] = max(gold_scores)
    metrics["range_coverage"] = (
        (max(pred_scores) - min(pred_scores)) / (max(gold_scores) - min(gold_scores))
        if max(gold_scores) > min(gold_scores)
        else 0.0
    )

    return metrics


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation without a scipy dependency."""
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denominator = (sum(a * a for a in dx) ** 0.5) * (sum(b * b for b in dy) ** 0.5)
    if denominator == 0:
        raise ZeroDivisionError("zero variance in one series")
    return numerator / denominator


def compute_decision_metrics(
    gold_scores: Sequence[float],
    pred_scores: Sequence[float],
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Sensitivity/specificity for the >threshold revascularisation decision."""
    tp = fp = tn = fn = 0
    for gold, pred in zip(gold_scores, pred_scores):
        gold_positive = gold > threshold
        pred_positive = pred > threshold
        if gold_positive and pred_positive:
            tp += 1
        elif gold_positive and not pred_positive:
            fn += 1
        elif not gold_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    return {
        f"sensitivity_gt{threshold:.0f}": tp / (tp + fn) if (tp + fn) else float("nan"),
        f"specificity_gt{threshold:.0f}": tn / (tn + fp) if (tn + fp) else float("nan"),
        f"accuracy_gt{threshold:.0f}": (tp + tn) / len(gold_scores)
        if gold_scores
        else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_tier_accuracy(
    gold_scores: Sequence[float], pred_scores: Sequence[float]
) -> dict[str, float]:
    """Exact and within-one-tier accuracy over the SYNTAX tertiles."""
    tier_order = [name for name, _, _ in RISK_TIERS]
    exact = 0
    within_one = 0
    for gold, pred in zip(gold_scores, pred_scores):
        gold_tier = risk_tier(gold)
        pred_tier = risk_tier(pred)
        if gold_tier == pred_tier:
            exact += 1
            within_one += 1
        elif abs(tier_order.index(gold_tier) - tier_order.index(pred_tier)) == 1:
            within_one += 1

    total = len(gold_scores)
    return {
        "tier_accuracy": exact / total if total else float("nan"),
        "tier_within_one": within_one / total if total else float("nan"),
    }


def compute_expert_agreement(
    records: Sequence[dict[str, Any]]
) -> dict[str, float]:
    """How often a prediction lands inside the inter-expert range.

    `records` entries need "pred", "expert_min" and "expert_max". Cases without
    expert annotations are skipped and counted, never imputed.
    """
    inside = 0
    evaluated = 0
    spreads: list[float] = []
    distances: list[float] = []

    for record in records:
        low = record.get("expert_min")
        high = record.get("expert_max")
        pred = record.get("pred")
        if low is None or high is None or pred is None:
            continue
        evaluated += 1
        spreads.append(float(high) - float(low))
        if float(low) <= float(pred) <= float(high):
            inside += 1
            distances.append(0.0)
        else:
            distances.append(
                min(abs(float(pred) - float(low)), abs(float(pred) - float(high)))
            )

    if not evaluated:
        return {"within_expert_range": float("nan"), "n_with_experts": 0}

    return {
        "within_expert_range": inside / evaluated,
        "n_with_experts": evaluated,
        # Mean inter-expert spread is the yardstick any MAE should be read
        # against: an MAE below this is within human disagreement.
        "mean_expert_spread": statistics.fmean(spreads),
        "mean_distance_to_expert_range": statistics.fmean(distances),
    }


def compute_severity_breakdown(
    gold_scores: Sequence[float], pred_scores: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Error statistics per gold risk tier.

    This is where a saturating regressor is exposed: strong overall numbers with
    a large negative bias confined to the high tier means the model systematically
    under-calls the cases that carry the most clinical weight.
    """
    breakdown: dict[str, dict[str, float]] = {}
    for tier_name, low, high in RISK_TIERS:
        pairs = [
            (g, p) for g, p in zip(gold_scores, pred_scores) if low <= g < high
        ]
        if not pairs:
            continue
        errors = [p - g for g, p in pairs]
        breakdown[tier_name] = {
            "n": len(pairs),
            "mae": statistics.fmean([abs(e) for e in errors]),
            "bias": statistics.fmean(errors),
            "gold_mean": statistics.fmean([g for g, _ in pairs]),
            "pred_mean": statistics.fmean([p for _, p in pairs]),
        }
    return breakdown


def evaluate_syntax_scoring(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Full metric suite for the CardioSYNTAX task.

    Each record needs "gold" and "pred"; "expert_min"/"expert_max" are optional.
    Returns a flat metrics dict plus nested severity/expert sections.
    """
    usable = [r for r in records if r.get("pred") is not None]
    if not usable:
        return {"n": 0, "error": "no usable predictions"}

    gold_scores = [float(r["gold"]) for r in usable]
    pred_scores = [float(r["pred"]) for r in usable]

    metrics: dict[str, Any] = {"n": len(usable)}
    metrics.update(compute_point_metrics(gold_scores, pred_scores))
    metrics.update(compute_decision_metrics(gold_scores, pred_scores))
    metrics.update(compute_tier_accuracy(gold_scores, pred_scores))
    metrics.update(compute_expert_agreement(usable))
    metrics["by_severity"] = compute_severity_breakdown(gold_scores, pred_scores)
    return metrics
