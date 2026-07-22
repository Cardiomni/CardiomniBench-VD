"""
Scoring metrics for CardiomniBench-VD evaluation.

Evaluates SYNTAX scoring, CAD-RADS grading accuracy, and risk stratification.
Anchors to the gold_standard.stage3_scoring structure.
"""

from typing import Dict, List, Any, Optional
import numpy as np


def syntax_score_error(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate SYNTAX score absolute error and segment-weighted error.

    Args:
        predicted: Agent's stage3_scoring output
        gold: Gold standard stage3_scoring

    Returns:
        Dict with 'syntax_score_mae', 'syntax_score_percentage_error', 'segment_weighted_error'
    """
    pred_score = predicted.get('syntax_score', {}).get('total_score', 0)
    gold_score = gold.get('syntax_score', {}).get('total_score', 0)

    mae = abs(pred_score - gold_score)
    pct_error = (mae / max(gold_score, 1)) * 100  # avoid div by zero

    # TODO: segment-weighted error requires per-segment comparison
    # pred_segments = predicted.get('syntax_score', {}).get('segments', [])
    # gold_segments = gold.get('syntax_score', {}).get('segments', [])

    return {
        'syntax_score_mae': mae,
        'syntax_score_percentage_error': pct_error,
        'segment_weighted_error': 0.0,  # placeholder
    }


def syntax_risk_tier_accuracy(predicted: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """
    Binary accuracy: does the predicted SYNTAX risk tier match gold?

    Tiers: 'low' (0-22), 'intermediate' (23-32), 'high' (≥33)

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_tier = predicted.get('syntax_score', {}).get('risk_tier', '').lower()
    gold_tier = gold.get('syntax_score', {}).get('risk_tier', '').lower()
    return 1.0 if pred_tier == gold_tier else 0.0


def cad_rads_per_vessel_accuracy(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, float]:
    """
    Per-vessel CAD-RADS grading accuracy (exact match + ±1 tier tolerance).

    Args:
        predicted: stage3_scoring with cad_rads_per_vessel list
        gold: gold standard cad_rads_per_vessel

    Returns:
        Dict with 'exact_match_rate', 'within_1_tier_rate'
    """
    pred_vessels = {v['vessel']: v['grade'] for v in predicted.get('cad_rads_per_vessel', [])}
    gold_vessels = {v['vessel']: v['grade'] for v in gold.get('cad_rads_per_vessel', [])}

    # CAD-RADS ordinal mapping: 0, 1, 2, 3, 4, 5 (ignore modifiers like 4a/4b for now)
    def parse_grade(g: str) -> int:
        g = str(g).lower().replace('cad-rads ', '').strip()
        if g.startswith('0'): return 0
        if g.startswith('1'): return 1
        if g.startswith('2'): return 2
        if g.startswith('3'): return 3
        if g.startswith('4'): return 4
        if g.startswith('5'): return 5
        return -1

    exact = 0
    within1 = 0
    total = len(gold_vessels)

    for vessel, gold_g in gold_vessels.items():
        pred_g = pred_vessels.get(vessel, '')
        p_val = parse_grade(pred_g)
        g_val = parse_grade(gold_g)
        if p_val == g_val:
            exact += 1
            within1 += 1
        elif abs(p_val - g_val) <= 1:
            within1 += 1

    return {
        'exact_match_rate': exact / max(total, 1),
        'within_1_tier_rate': within1 / max(total, 1),
    }


def agatston_category_match(predicted: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """
    Binary: does the Agatston category (minimal/mild/moderate/severe) match?

    Returns:
        1.0 if match, 0.0 otherwise
    """
    pred_cat = predicted.get('agatston_score', {}).get('category', '').lower()
    gold_cat = gold.get('agatston_score', {}).get('category', '').lower()
    return 1.0 if pred_cat == gold_cat else 0.0


def evaluate_scoring(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate scoring metrics for a single case.

    Args:
        predicted: Agent's stage3_scoring output
        gold: Gold standard stage3_scoring

    Returns:
        Dict of all scoring metrics
    """
    metrics = {}
    metrics.update(syntax_score_error(predicted, gold))
    metrics['syntax_risk_tier_accuracy'] = syntax_risk_tier_accuracy(predicted, gold)
    metrics.update(cad_rads_per_vessel_accuracy(predicted, gold))
    metrics['agatston_category_match'] = agatston_category_match(predicted, gold)
    return metrics
