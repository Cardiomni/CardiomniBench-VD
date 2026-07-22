"""Metric registry — maps rubric ``metric`` names to callables.

The rubric YAMLs reference objective metrics by string name (e.g.
``segment_f1_score``, ``stenosis_mae``). This registry is the single place that
binds those names to the functions in ``evaluation/metrics/`` and adapts their
varied signatures to one uniform contract::

    adapter(gold_standard: dict, prediction: dict) -> float

scoring.py then maps that float to a grade via the criterion's threshold ranges.

Adapters are deliberately defensive: on missing/empty fields they return a
neutral value rather than raising, so the pipeline runs end-to-end on mock data
with no real annotations. Register a new metric by adding one entry to REGISTRY —
this is the '换 rubric metric' extension point.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from evaluation.metrics import perception_metrics as pm
from evaluation.metrics import scoring_metrics as sm
from . import report_facts as rf

logger = logging.getLogger(__name__)

Adapter = Callable[[Dict[str, Any], Dict[str, Any]], float]


# --- field extraction helpers -------------------------------------------------

def _gold_cta_segments(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (gold.get("stage1a_cta", {}) or {}).get("segments", []) or []


def _gold_dsa_segments(gold: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (gold.get("stage1b_dsa", {}) or {}).get("segments", []) or []


def _pred_cta_segments(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (pred.get("cta_findings", {}) or {}).get("segments", []) or []


def _pred_dsa_segments(pred: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (pred.get("dsa_findings", {}) or {}).get("segments", []) or []


# --- adapters (uniform signature: gold, pred -> float) ------------------------

def _segment_f1(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = (gold.get("stage0_anatomy", {}) or {}).get("segments", []) or []
    pred_segs = (pred.get("anatomical_localization", {}) or {}).get("segments_identified", []) or []
    if not gold_segs:
        return 0.0
    return pm.compute_segment_f1(gold_segs, pred_segs)["f1"]


def _dominance_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    g = (gold.get("stage0_anatomy", {}) or {}).get("dominance", "") or ""
    p = (pred.get("anatomical_localization", {}) or {}).get("dominance", "") or ""
    if not g:
        return 0.0
    return pm.compute_dominance_accuracy(g, p)


def _stenosis_mae(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = _gold_dsa_segments(gold) or _gold_cta_segments(gold)
    pred_segs = _pred_dsa_segments(pred) or _pred_cta_segments(pred)
    if not gold_segs:
        return 0.0
    return pm.compute_stenosis_mae(gold_segs, pred_segs)["mae"]


def _cadrads_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = _gold_cta_segments(gold)
    pred_segs = _pred_cta_segments(pred)
    if not gold_segs:
        return 0.0
    return pm.compute_cadrads_accuracy(gold_segs, pred_segs)["tolerance_accuracy"]


def _plaque_classification_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = _gold_cta_segments(gold)
    pred_segs = _pred_cta_segments(pred)
    if not gold_segs:
        return 0.0
    return pm.compute_plaque_classification_accuracy(gold_segs, pred_segs)["accuracy"]


def _high_risk_plaque_f1(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    g = (gold.get("stage1a_cta", {}) or {}).get("high_risk_plaque_features", {}) or {}
    p = (pred.get("cta_findings", {}) or {}).get("high_risk_plaque_features", {}) or {}
    return pm.compute_high_risk_plaque_f1(g, p)["overall_f1"]


def _timi_flow_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = _gold_dsa_segments(gold)
    pred_segs = _pred_dsa_segments(pred)
    if not gold_segs:
        return 0.0
    return pm.compute_timi_flow_accuracy(gold_segs, pred_segs)


def _lesion_type_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = _gold_dsa_segments(gold)
    pred_segs = _pred_dsa_segments(pred)
    if not gold_segs:
        return 0.0
    return pm.compute_lesion_morphology_accuracy(gold_segs, pred_segs)


def _syntax_score_mae(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    g = gold.get("stage3_scoring", {}) or {}
    p = pred.get("comprehensive_scoring", {}) or {}
    return sm.syntax_score_error(p, g)["syntax_score_mae"]


def _syntax_risk_tier_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    g = gold.get("stage3_scoring", {}) or {}
    p = pred.get("comprehensive_scoring", {}) or {}
    return sm.syntax_risk_tier_accuracy(p, g)


# --- prose DSA-report tolerance metrics ---------------------------------------
# The locked-in DSA task scores a PROSE report, not structured fields. Gold facts
# come from a dedicated ``dsa_report_facts`` block if present, else are derived
# from the existing ``stage1b_dsa.segments`` (segment_id + stenosis_percent).
# Pred facts are extracted from ``prediction["report"]`` (see report_facts.py).
# Comparison is tolerance-based (±% or same clinical tier), so these are the
# "带容差打分" adapters — deterministic and API-free (heuristic extraction).


def _gold_report_facts(gold: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the gold standard into report-facts shape for comparison.

    Prefers an explicit ``dsa_report_facts`` block (authored per the prose-report
    schema). Falls back to deriving facts from ``stage1b_dsa.segments`` so legacy
    structured gold standards still score.
    """
    explicit = gold.get("dsa_report_facts")
    if isinstance(explicit, dict) and explicit.get("segments") is not None:
        return explicit
    segs = _gold_dsa_segments(gold)
    dominance = (gold.get("stage0_anatomy", {}) or {}).get("dominance", "") or ""
    return {
        "dominance": dominance,
        "segments": [
            {
                "segment_id": s.get("segment_id"),
                "vessel": s.get("vessel"),
                "position": s.get("position"),
                "name": s.get("segment_name") or s.get("name"),
                "stenosis_percent": s.get("stenosis_percent"),
            }
            for s in segs
        ],
    }


def _fact_comparison(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pred facts + compare against gold, memoized on the pred dict.

    Cached under a private key so the five sub-score adapters share one parse
    within a single case evaluation. Keyed by id(gold) to stay correct if the
    same pred object were ever scored against different gold (it isn't today).
    """
    cache = pred.setdefault("_dsa_fact_cache", {})
    key = id(gold)
    if key not in cache:
        gold_facts = _gold_report_facts(gold)
        # No judge here: the automatic path uses deterministic heuristic
        # extraction. An orchestrator may pre-populate pred["extracted_facts"]
        # with LLM output, which extract_facts() will prefer.
        pred_facts = rf.extract_facts(pred, judge=None, mode="heuristic")
        cache[key] = rf.compare_facts(gold_facts, pred_facts)
    return cache[key]


def _report_coverage_recall(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return _fact_comparison(gold, pred)["coverage_recall"]


def _report_naming_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return _fact_comparison(gold, pred)["naming_accuracy"]


def _report_stenosis_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return _fact_comparison(gold, pred)["stenosis_accuracy"]


def _report_dominance_correct(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return _fact_comparison(gold, pred)["dominance_correct"]


def _report_anti_hallucination(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    return _fact_comparison(gold, pred)["anti_hallucination"]


def _binary_present(field_path: str) -> Adapter:
    """Build an adapter that scores 1.0 when a nested prediction field is non-empty.

    Used for data-handling checks (DICOM parsed, HU available) where, without a
    real execution trace, the best objective signal is 'did the agent populate
    this section at all'. Real runs can replace these with trace-based checks.
    """

    def adapter(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
        node: Any = pred
        for key in field_path.split("."):
            if isinstance(node, dict) and key in node and node[key]:
                node = node[key]
            else:
                return 0.0
        return 1.0

    return adapter


REGISTRY: Dict[str, Adapter] = {
    # data_handling — presence-based proxies until real traces exist
    "dicom_parse_success": _binary_present("cta_findings"),
    "modality_identification_accuracy": _binary_present("dsa_findings"),
    "hu_value_available": _binary_present("cta_findings.calcium_score"),
    # perception_accuracy
    "segment_f1_score": _segment_f1,
    "dominance_accuracy": _dominance_accuracy,
    "stenosis_mae": _stenosis_mae,
    "cadrads_accuracy": _cadrads_accuracy,
    "plaque_classification_accuracy": _plaque_classification_accuracy,
    "agatston_tier_accuracy": _binary_present("cta_findings.calcium_score"),
    "high_risk_plaque_f1": _high_risk_plaque_f1,
    "timi_flow_accuracy": _timi_flow_accuracy,
    "rentrop_accuracy": _binary_present("dsa_findings.collaterals"),
    "lesion_type_accuracy": _lesion_type_accuracy,
    # clinical_interpretation (scoring)
    "cadrads_per_patient_accuracy": _binary_present("comprehensive_scoring.cadrads_per_patient"),
    "syntax_score_mae": _syntax_score_mae,
    "syntax_risk_tier_accuracy": _syntax_risk_tier_accuracy,
    # prose DSA-report tolerance metrics (the locked-in narrow task)
    "report_segment_coverage_recall": _report_coverage_recall,
    "report_naming_accuracy": _report_naming_accuracy,
    "report_stenosis_accuracy": _report_stenosis_accuracy,
    "report_dominance_correct": _report_dominance_correct,
    "report_anti_hallucination": _report_anti_hallucination,
}


def get_metric(name: str) -> Optional[Adapter]:
    """Return the adapter for a metric name, or None if unregistered."""
    adapter = REGISTRY.get(name)
    if adapter is None:
        logger.warning("no registered metric adapter for %r", name)
    return adapter


def list_metrics() -> List[str]:
    return sorted(REGISTRY)
