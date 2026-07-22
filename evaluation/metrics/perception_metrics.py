"""
Perception Metrics for CardiomniBench-VD

Implements objective metrics for Stage 0 and Stage 1 (CTA/DSA perception):
- Vessel segment identification (F1, precision, recall)
- Dominance classification accuracy
- Stenosis quantification (MAE, RMSE, ±1-tier tolerance)
- CAD-RADS grading accuracy
- Plaque classification accuracy
- Calcium scoring accuracy
- High-risk plaque feature detection
- TIMI flow grading accuracy
- Rentrop collateral grading accuracy
- ACC/AHA lesion morphology accuracy
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter


def compute_segment_f1(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id"
) -> Dict[str, float]:
    """
    Compute F1 score for vessel segment identification.

    Args:
        gold_segments: Gold standard segment list
        pred_segments: Predicted segment list
        match_key: Key to match segments (typically "segment_id" or "segment_name")

    Returns:
        Dictionary with precision, recall, f1
    """
    gold_ids = set(seg[match_key] for seg in gold_segments if seg.get('present', True))
    pred_ids = set(seg[match_key] for seg in pred_segments if seg.get('present', True))

    if len(pred_ids) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gold_ids)}

    tp = len(gold_ids & pred_ids)
    fp = len(pred_ids - gold_ids)
    fn = len(gold_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_dominance_accuracy(gold_dominance: str, pred_dominance: str) -> float:
    """
    Compute accuracy for dominance classification.

    Args:
        gold_dominance: Gold standard ("right" or "left")
        pred_dominance: Prediction ("right" or "left")

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    return 1.0 if gold_dominance.lower() == pred_dominance.lower() else 0.0


def compute_stenosis_mae(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id"
) -> Dict[str, float]:
    """
    Compute mean absolute error for stenosis percentage quantification.

    Args:
        gold_segments: Gold standard with stenosis_percent
        pred_segments: Prediction with stenosis_percent
        match_key: Key to match segments

    Returns:
        Dictionary with MAE, RMSE, and per-segment errors
    """
    errors = []
    matched_segments = []

    # Create lookup dict for predictions
    pred_dict = {seg[match_key]: seg for seg in pred_segments}

    for gold_seg in gold_segments:
        seg_id = gold_seg[match_key]
        if seg_id not in pred_dict:
            # Missing segment — treat as maximum error
            errors.append(100.0)
            continue

        pred_seg = pred_dict[seg_id]

        gold_stenosis = gold_seg.get('stenosis_percent')
        pred_stenosis = pred_seg.get('stenosis_percent')

        if gold_stenosis is None or pred_stenosis is None:
            continue

        error = abs(gold_stenosis - pred_stenosis)
        errors.append(error)
        matched_segments.append({
            'segment_id': seg_id,
            'gold': gold_stenosis,
            'pred': pred_stenosis,
            'error': error
        })

    if len(errors) == 0:
        return {'mae': 0.0, 'rmse': 0.0, 'num_segments': 0}

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'num_segments': len(errors),
        'per_segment_errors': matched_segments
    }


def compute_cadrads_accuracy(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id",
    tolerance: int = 1
) -> Dict[str, float]:
    """
    Compute CAD-RADS grading accuracy.

    Args:
        gold_segments: Gold standard with cad_rads field
        pred_segments: Prediction with cad_rads field
        match_key: Key to match segments
        tolerance: Allow ±N tier difference (default 1)

    Returns:
        Dictionary with exact accuracy, ±1-tier accuracy, and confusion matrix
    """
    # CAD-RADS ordering for tier distance calculation
    cadrads_order = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4A': 4,
        '4B': 5,
        '5': 6,
        'N': -1  # Non-diagnostic
    }

    pred_dict = {seg[match_key]: seg for seg in pred_segments}
    exact_matches = 0
    within_tolerance_matches = 0
    total = 0
    confusion = []

    for gold_seg in gold_segments:
        seg_id = gold_seg[match_key]
        if seg_id not in pred_dict:
            continue

        pred_seg = pred_dict[seg_id]
        gold_grade = gold_seg.get('cad_rads', '').upper()
        pred_grade = pred_seg.get('cad_rads', '').upper()

        if not gold_grade or not pred_grade:
            continue

        total += 1

        # Exact match
        if gold_grade == pred_grade:
            exact_matches += 1
            within_tolerance_matches += 1
        else:
            # Check tier distance
            gold_tier = cadrads_order.get(gold_grade, -999)
            pred_tier = cadrads_order.get(pred_grade, -999)

            if gold_tier >= 0 and pred_tier >= 0:
                tier_distance = abs(gold_tier - pred_tier)
                if tier_distance <= tolerance:
                    within_tolerance_matches += 1

        confusion.append({
            'segment_id': seg_id,
            'gold': gold_grade,
            'pred': pred_grade
        })

    if total == 0:
        return {'exact_accuracy': 0.0, 'tolerance_accuracy': 0.0, 'num_segments': 0}

    return {
        'exact_accuracy': exact_matches / total,
        'tolerance_accuracy': within_tolerance_matches / total,
        'num_segments': total,
        'confusion': confusion
    }


def compute_plaque_classification_accuracy(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id"
) -> Dict[str, float]:
    """
    Compute accuracy for plaque type classification (calcified/soft/mixed).

    Returns overall accuracy and per-class precision/recall.
    """
    pred_dict = {seg[match_key]: seg for seg in pred_segments}

    gold_labels = []
    pred_labels = []

    for gold_seg in gold_segments:
        seg_id = gold_seg[match_key]
        if seg_id not in pred_dict:
            continue

        gold_type = gold_seg.get('plaque_type', '').lower()
        pred_type = pred_dict[seg_id].get('plaque_type', '').lower()

        if not gold_type or not pred_type:
            continue

        gold_labels.append(gold_type)
        pred_labels.append(pred_type)

    if len(gold_labels) == 0:
        return {'accuracy': 0.0, 'num_segments': 0}

    accuracy = sum(g == p for g, p in zip(gold_labels, pred_labels)) / len(gold_labels)

    # Per-class metrics
    plaque_types = ['calcified', 'soft', 'mixed']
    per_class = {}

    for ptype in plaque_types:
        tp = sum((g == ptype) and (p == ptype) for g, p in zip(gold_labels, pred_labels))
        fp = sum((g != ptype) and (p == ptype) for g, p in zip(gold_labels, pred_labels))
        fn = sum((g == ptype) and (p != ptype) for g, p in zip(gold_labels, pred_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[ptype] = {'precision': precision, 'recall': recall, 'f1': f1}

    return {
        'accuracy': accuracy,
        'num_segments': len(gold_labels),
        'per_class': per_class
    }


def compute_calcium_score_accuracy(
    gold_agatston_category: str,
    pred_agatston_category: str
) -> float:
    """
    Compute accuracy for Agatston calcium score category.

    Categories: "0", "1-99", "100-399", "400-999", ">=1000"

    Returns 1.0 if correct, 0.0 if incorrect.
    """
    return 1.0 if gold_agatston_category == pred_agatston_category else 0.0


def compute_high_risk_plaque_f1(
    gold_hrp_features: Dict[str, bool],
    pred_hrp_features: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Compute F1 for high-risk plaque feature detection.

    Features:
    - low_attenuation_plaque (<30 HU)
    - positive_remodeling (≥1.1)
    - napkin_ring_sign
    - spotty_calcification (<3mm)

    Returns per-feature and overall metrics.
    """
    features = ['low_attenuation_plaque', 'positive_remodeling', 'napkin_ring_sign', 'spotty_calcification']

    per_feature = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for feature in features:
        gold_present = gold_hrp_features.get(feature, False)
        pred_present = pred_hrp_features.get(feature, False)

        tp = int(gold_present and pred_present)
        fp = int((not gold_present) and pred_present)
        fn = int(gold_present and (not pred_present))

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_feature[feature] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    return {
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'per_feature': per_feature
    }


def compute_timi_flow_accuracy(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id"
) -> float:
    """
    Compute accuracy for TIMI flow grading (0-3).

    Returns exact accuracy.
    """
    pred_dict = {seg[match_key]: seg for seg in pred_segments}
    correct = 0
    total = 0

    for gold_seg in gold_segments:
        seg_id = gold_seg[match_key]
        if seg_id not in pred_dict:
            continue

        gold_timi = gold_seg.get('timi_flow')
        pred_timi = pred_dict[seg_id].get('timi_flow')

        if gold_timi is None or pred_timi is None:
            continue

        total += 1
        if gold_timi == pred_timi:
            correct += 1

    return correct / total if total > 0 else 0.0


def compute_rentrop_accuracy(gold_rentrop: int, pred_rentrop: int) -> float:
    """
    Compute accuracy for Rentrop collateral grading (0-3).

    Returns 1.0 if correct, 0.0 if incorrect.
    """
    if gold_rentrop is None or pred_rentrop is None:
        return 0.0
    return 1.0 if gold_rentrop == pred_rentrop else 0.0


def compute_lesion_morphology_accuracy(
    gold_segments: List[Dict[str, Any]],
    pred_segments: List[Dict[str, Any]],
    match_key: str = "segment_id"
) -> float:
    """
    Compute accuracy for ACC/AHA lesion morphology classification (A/B1/B2/C).

    Returns exact accuracy.
    """
    pred_dict = {seg[match_key]: seg for seg in pred_segments}
    correct = 0
    total = 0

    for gold_seg in gold_segments:
        seg_id = gold_seg[match_key]
        if seg_id not in pred_dict:
            continue

        gold_type = gold_seg.get('lesion_type', '').upper()
        pred_type = pred_dict[seg_id].get('lesion_type', '').upper()

        if not gold_type or not pred_type:
            continue

        total += 1
        if gold_type == pred_type:
            correct += 1

    return correct / total if total > 0 else 0.0
