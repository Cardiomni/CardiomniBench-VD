"""
Fusion Reasoning Metrics for CardiomniBench-VD

Implements metrics for Stage 2 (cross-modal CTA-DSA integration):
- Calcium blooming correction appropriateness
- CTO assessment completeness
- Culprit lesion identification accuracy
- CTA-DSA discrepancy explanation quality
"""

from typing import Dict, List, Any, Optional
import numpy as np


def compute_blooming_correction_score(
    gold_fusion: Dict[str, Any],
    pred_fusion: Dict[str, Any],
    gold_segments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate calcium blooming correction appropriateness.

    Checks whether agent correctly:
    1. Identified segments with heavy calcification
    2. Used DSA instead of CTA for stenosis quantification in those segments
    3. Documented the correction decision

    Args:
        gold_fusion: Gold standard fusion reasoning
        pred_fusion: Predicted fusion reasoning
        gold_segments: Gold standard segments with calcium information

    Returns:
        Score and detailed breakdown
    """
    # Identify segments requiring blooming correction (heavy calcium + significant stenosis)
    gold_corrections = gold_fusion.get('blooming_corrections', [])
    pred_corrections = pred_fusion.get('blooming_corrections', [])

    # Extract segment IDs requiring correction
    gold_segment_ids = set(corr.get('segment_id') for corr in gold_corrections)
    pred_segment_ids = set(corr.get('segment_id') for corr in pred_corrections)

    # Compute precision/recall for identifying segments needing correction
    tp = len(gold_segment_ids & pred_segment_ids)
    fp = len(pred_segment_ids - gold_segment_ids)
    fn = len(gold_segment_ids - pred_segment_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Check if DSA values were actually used
    correctly_applied = 0
    for seg_id in gold_segment_ids:
        pred_corr = next((c for c in pred_corrections if c.get('segment_id') == seg_id), None)
        if pred_corr and pred_corr.get('used_dsa_value', False):
            correctly_applied += 1

    application_rate = correctly_applied / len(gold_segment_ids) if len(gold_segment_ids) > 0 else 0.0

    return {
        'identification_f1': f1,
        'identification_precision': precision,
        'identification_recall': recall,
        'application_rate': application_rate,
        'score': (f1 + application_rate) / 2,  # Combined score
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'correctly_applied': correctly_applied,
        'required_corrections': len(gold_segment_ids)
    }


def compute_cto_assessment_completeness(
    gold_cto: Dict[str, Any],
    pred_cto: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate CTO assessment completeness.

    CTO assessment should include:
    - CTA stump morphology (blunt/tapered)
    - Occlusion length
    - Bridging collaterals from DSA (Rentrop grade)
    - Side branch involvement
    - Age estimation (if available)

    Args:
        gold_cto: Gold standard CTO assessment
        pred_cto: Predicted CTO assessment

    Returns:
        Completeness score and component breakdown
    """
    required_components = [
        'stump_morphology',
        'occlusion_length_mm',
        'bridging_collaterals',
        'rentrop_grade',
        'side_branch_involvement'
    ]

    components_present = 0
    components_correct = 0
    total_components = len(required_components)

    details = {}

    for component in required_components:
        gold_value = gold_cto.get(component)
        pred_value = pred_cto.get(component)

        present = pred_value is not None
        correct = gold_value == pred_value if gold_value is not None else False

        if present:
            components_present += 1
        if correct:
            components_correct += 1

        details[component] = {
            'present': present,
            'correct': correct,
            'gold': gold_value,
            'pred': pred_value
        }

    completeness = components_present / total_components
    accuracy = components_correct / total_components

    # Combined score: completeness weighted 60%, accuracy 40%
    score = 0.6 * completeness + 0.4 * accuracy

    return {
        'score': score,
        'completeness': completeness,
        'accuracy': accuracy,
        'components_present': components_present,
        'components_correct': components_correct,
        'total_components': total_components,
        'details': details
    }


def compute_culprit_lesion_accuracy(
    gold_culprit: Optional[Dict[str, Any]],
    pred_culprit: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate culprit lesion identification accuracy.

    Relevant for acute coronary syndrome cases.

    Args:
        gold_culprit: Gold standard culprit lesion (segment_id, rationale)
        pred_culprit: Predicted culprit lesion

    Returns:
        Binary accuracy and rationale quality
    """
    if gold_culprit is None:
        # Not applicable (stable CAD case)
        return {
            'applicable': False,
            'score': 1.0,
            'note': 'No culprit lesion expected (stable CAD)'
        }

    if pred_culprit is None:
        return {
            'applicable': True,
            'segment_correct': False,
            'score': 0.0,
            'note': 'Missed culprit lesion identification'
        }

    gold_segment = gold_culprit.get('segment_id')
    pred_segment = pred_culprit.get('segment_id')

    segment_correct = (gold_segment == pred_segment)

    # Check rationale mentions key features
    gold_rationale_features = set(gold_culprit.get('features', []))
    pred_rationale_features = set(pred_culprit.get('features', []))

    rationale_overlap = len(gold_rationale_features & pred_rationale_features)
    rationale_score = rationale_overlap / len(gold_rationale_features) if len(gold_rationale_features) > 0 else 0.0

    # Combined score: segment ID is primary (70%), rationale quality is secondary (30%)
    score = 0.7 * (1.0 if segment_correct else 0.0) + 0.3 * rationale_score

    return {
        'applicable': True,
        'segment_correct': segment_correct,
        'rationale_score': rationale_score,
        'score': score,
        'gold_segment': gold_segment,
        'pred_segment': pred_segment,
        'gold_features': list(gold_rationale_features),
        'pred_features': list(pred_rationale_features)
    }


def compute_discrepancy_explanation_quality(
    gold_fusion: Dict[str, Any],
    pred_fusion: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate quality of CTA-DSA discrepancy explanations.

    When CTA and DSA show different stenosis grades, agent should provide
    a reasonable explanation (e.g., calcium blooming, imaging angle, vasospasm).

    Args:
        gold_fusion: Gold standard with documented discrepancies
        pred_fusion: Predicted fusion analysis

    Returns:
        Score based on whether discrepancies are identified and explained
    """
    gold_discrepancies = gold_fusion.get('discrepancies', [])
    pred_discrepancies = pred_fusion.get('discrepancies', [])

    if len(gold_discrepancies) == 0:
        # No discrepancies expected
        false_positives = len(pred_discrepancies)
        return {
            'applicable': False,
            'score': 1.0 if false_positives == 0 else 0.5,
            'note': 'No discrepancies expected' if false_positives == 0 else f'{false_positives} false positive discrepancies'
        }

    # Match discrepancies by segment_id
    gold_segments = set(d.get('segment_id') for d in gold_discrepancies)
    pred_segments = set(d.get('segment_id') for d in pred_discrepancies)

    identified = len(gold_segments & pred_segments)
    missed = len(gold_segments - pred_segments)

    identification_rate = identified / len(gold_segments)

    # Check explanation quality for matched discrepancies
    explanation_scores = []
    for gold_disc in gold_discrepancies:
        seg_id = gold_disc.get('segment_id')
        pred_disc = next((d for d in pred_discrepancies if d.get('segment_id') == seg_id), None)

        if pred_disc:
            # Simple heuristic: check if explanation is non-empty and mentions plausible causes
            explanation = pred_disc.get('explanation', '').lower()
            gold_explanation = gold_disc.get('explanation', '').lower()

            # Check for key terms
            plausible_causes = ['calcium', 'blooming', 'angle', 'foreshortening', 'overlap', 'vasospasm', 'tortuosity']
            mentions_cause = any(cause in explanation for cause in plausible_causes)

            # Rough similarity check
            if mentions_cause:
                explanation_scores.append(1.0)
            else:
                explanation_scores.append(0.3)  # Has explanation but weak
        else:
            explanation_scores.append(0.0)

    avg_explanation_quality = np.mean(explanation_scores) if explanation_scores else 0.0

    # Combined score: 50% identification, 50% explanation quality
    score = 0.5 * identification_rate + 0.5 * avg_explanation_quality

    return {
        'applicable': True,
        'score': score,
        'identification_rate': identification_rate,
        'explanation_quality': avg_explanation_quality,
        'identified': identified,
        'missed': missed,
        'total_discrepancies': len(gold_segments)
    }


def evaluate_fusion_reasoning(
    gold_fusion: Dict[str, Any],
    pred_fusion: Dict[str, Any],
    gold_segments: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Aggregate all fusion reasoning metrics.

    Args:
        gold_fusion: Gold standard stage2_fusion
        pred_fusion: Predicted fusion_analysis
        gold_segments: Optional gold standard segments for context

    Returns:
        Aggregated fusion score and component breakdowns
    """
    results = {}

    # Blooming correction
    if gold_segments:
        results['blooming_correction'] = compute_blooming_correction_score(
            gold_fusion, pred_fusion, gold_segments
        )
    else:
        results['blooming_correction'] = {'score': 0.0, 'note': 'No segment data provided'}

    # CTO assessment
    gold_cto = gold_fusion.get('cto_assessment', {})
    pred_cto = pred_fusion.get('cto_assessment', {})
    if gold_cto:
        results['cto_assessment'] = compute_cto_assessment_completeness(gold_cto, pred_cto)
    else:
        results['cto_assessment'] = {'applicable': False, 'score': 1.0}

    # Culprit lesion
    gold_culprit = gold_fusion.get('culprit_lesion')
    pred_culprit = pred_fusion.get('culprit_lesion')
    results['culprit_lesion'] = compute_culprit_lesion_accuracy(gold_culprit, pred_culprit)

    # Discrepancy explanations
    results['discrepancy_explanation'] = compute_discrepancy_explanation_quality(gold_fusion, pred_fusion)

    # Aggregate score
    component_scores = [
        results['blooming_correction'].get('score', 0.0),
        results['cto_assessment'].get('score', 1.0),
        results['culprit_lesion'].get('score', 1.0),
        results['discrepancy_explanation'].get('score', 1.0)
    ]

    results['aggregate_score'] = np.mean(component_scores)

    return results


def compute_fusion_lift(
    cohort_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Cohort-level FUSION LIFT — the core experiment behind CardiomniBench-VD's
    "why paired CTA+DSA" claim (see PROPOSAL §2.3, C-class).

    Hypothesis: an agent given BOTH modalities should significantly outperform a
    single-modality agent specifically on the `fusion_required` subset, while the
    two arms are comparable on `cta_only` / `dsa_only` subsets. A large lift on
    the fusion_required subset (and a small lift elsewhere) is the measurable
    evidence that fusion reasoning — not just extra pixels — is what helps.

    Args:
        cohort_results: list of per-case dicts, each with:
            - 'fusion_category': 'cta_only' | 'dsa_only' | 'fusion_required'
            - 'score_paired':    overall case score for the paired (CTA+DSA) agent
            - 'score_single':    overall case score for the single-modality agent
              (the ablation arm that only receives the dominant modality)

    Returns:
        Per-subset mean scores for both arms, the lift (paired - single), and the
        counts — ready to drop into the paper's fusion-ablation table.
    """
    subsets = ("cta_only", "dsa_only", "fusion_required")
    out: Dict[str, Any] = {}

    for cat in subsets:
        cases = [c for c in cohort_results if c.get("fusion_category") == cat]
        if not cases:
            out[cat] = {"n": 0, "paired_mean": None, "single_mean": None, "lift": None}
            continue
        paired = [float(c.get("score_paired", 0.0)) for c in cases]
        single = [float(c.get("score_single", 0.0)) for c in cases]
        paired_mean = float(np.mean(paired))
        single_mean = float(np.mean(single))
        out[cat] = {
            "n": len(cases),
            "paired_mean": paired_mean,
            "single_mean": single_mean,
            "lift": paired_mean - single_mean,
        }

    # The headline number: lift on the fusion_required subset. The claim holds if
    # this is clearly positive AND larger than the lift on the single-modality subsets.
    fr = out.get("fusion_required", {})
    out["headline"] = {
        "fusion_required_lift": fr.get("lift"),
        "interpretation": (
            "Fusion value is demonstrated when fusion_required_lift is clearly "
            "positive and exceeds the lift on cta_only / dsa_only subsets. "
            "A near-zero lift everywhere means paired input adds pixels, not reasoning."
        ),
    }
    return out
