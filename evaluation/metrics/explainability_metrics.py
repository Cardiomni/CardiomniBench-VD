"""
Explainability metrics for CardiomniBench-VD evaluation.

Evaluates the quality of the clinical reasoning trace and evidence anchoring.
"""

from typing import Dict, List, Any, Optional


def has_reasoning_trace(report: str) -> bool:
    """
    Binary check: does the report contain an explicit reasoning section?

    Looks for keywords: "reasoning", "rationale", "因为", "由于", "考虑到"
    """
    keywords = ['reasoning', 'rationale', 'justification', '推理', '因为', '由于', '考虑到', '根据']
    report_lower = report.lower()
    return any(kw in report_lower for kw in keywords)


def count_evidence_citations(report: str) -> int:
    """
    Count explicit evidence citations in the report.

    Looks for patterns: "based on CTA slice X", "DSA frame Y shows", "见 CTA", "DSA 可见"

    Returns:
        Number of evidence citations found
    """
    patterns = [
        'based on cta', 'based on dsa', 'cta slice', 'dsa frame', 'cta shows', 'dsa shows',
        '见 cta', '见 dsa', 'cta 可见', 'dsa 可见', '根据 cta', '根据 dsa'
    ]
    report_lower = report.lower()
    return sum(1 for p in patterns if p in report_lower)


def fusion_reasoning_present(report: str) -> bool:
    """
    Binary: does the report explain CTA-DSA cross-modal fusion reasoning?

    Looks for: "cta-dsa", "cross-modal", "fusion", "钙化校正", "blooming"
    """
    keywords = ['cta-dsa', 'cta and dsa', 'cross-modal', 'fusion', '钙化校正', 'blooming', '融合', '互补']
    report_lower = report.lower()
    return any(kw in report_lower for kw in keywords)


def capability_boundary_declared(report: str) -> bool:
    """
    Binary: does the report explicitly declare capability boundaries?

    Looks for: "cannot determine", "requires FFR", "needs IVUS", "无法判断", "需要"
    """
    keywords = [
        'cannot determine', 'unable to assess', 'requires ffr', 'needs ivus', 'requires lab',
        'beyond imaging', '无法判断', '无法确定', '需要 ffr', '需要 ivus', '需实验室', '超出影像'
    ]
    report_lower = report.lower()
    return any(kw in report_lower for kw in keywords)


def standard_anchoring_count(report: str) -> int:
    """
    Count references to clinical grading standards.

    Looks for: CAD-RADS, SYNTAX, TIMI, Rentrop, ACC/AHA, Agatston

    Returns:
        Number of standard mentions
    """
    standards = ['cad-rads', 'syntax', 'timi', 'rentrop', 'acc/aha', 'agatston']
    report_lower = report.lower()
    return sum(report_lower.count(std) for std in standards)


def evaluate_explainability(report: str, predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate explainability metrics for a single case.

    Args:
        report: Agent's full diagnostic report (text)
        predicted: Agent's structured output (for cross-check)
        gold: Gold standard (for capability_boundary cross-check)

    Returns:
        Dict of explainability metrics
    """
    return {
        'has_reasoning_trace': has_reasoning_trace(report),
        'evidence_citation_count': count_evidence_citations(report),
        'fusion_reasoning_present': fusion_reasoning_present(report),
        'capability_boundary_declared': capability_boundary_declared(report),
        'standard_anchoring_count': standard_anchoring_count(report),
        # Derived: anti-hallucination check (does the agent fabricate findings not in images?)
        # This requires deep semantic check — placeholder for LLM-judge to handle
        'fabrication_detected': False,  # TODO: LLM-judge anti-hallucination criterion
    }
