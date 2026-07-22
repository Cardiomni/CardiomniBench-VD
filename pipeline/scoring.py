"""Rubric scoring — turn one criterion into points, mirroring BiomniBench.

For each criterion the evaluation_method decides the path:
    automatic  — run the registered metric adapter, map its value to a grade via
                 the criterion's threshold ranges, then to points.
    llm_judge  — render a prompt, ask the judge backend for a categorical grade,
                 then map grade -> points from the rubric table.
    hybrid     — run the metric AND the judge; take the lower grade (conservative).

The judge only ever emits a grade label; points come from the rubric's fixed
grade->points table, computed here in code. Missing metric or missing grade
degrades to the lowest-point grade rather than crashing — the pipeline must run
end-to-end on mock data.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .judge_backends import Judge
from .metric_registry import get_metric

logger = logging.getLogger(__name__)


def _grades(criterion: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (criterion.get("grading_scale", {}) or {}).get("grades", []) or []


def _max_points(criterion: Dict[str, Any]) -> float:
    grades = _grades(criterion)
    return max((g.get("points", 0) for g in grades), default=0.0)


def _min_points(criterion: Dict[str, Any]) -> float:
    grades = _grades(criterion)
    return min((g.get("points", 0) for g in grades), default=0.0)


def _points_for_grade(criterion: Dict[str, Any], grade: Optional[str]) -> float:
    for g in _grades(criterion):
        if g.get("grade") == grade:
            return float(g.get("points", 0))
    # Unknown/None grade -> lowest points (fail closed).
    return float(_min_points(criterion))


def _grade_from_metric_value(criterion: Dict[str, Any], value: float) -> Dict[str, Any]:
    """Map a numeric metric value to a grade using threshold.metric_range.

    Grades are checked in declared order; the first whose [min, max) contains the
    value wins. If no grade declares a threshold, fall back to the top grade for
    non-error metrics (kept simple — real thresholds are expected in the rubric).
    """
    grades = _grades(criterion)
    if not grades:
        return {"grade": None, "points": 0.0}

    has_thresholds = any((g.get("threshold", {}) or {}).get("metric_range") for g in grades)
    if has_thresholds:
        for g in grades:
            rng = (g.get("threshold", {}) or {}).get("metric_range")
            if not rng:
                continue
            lo = rng.get("min", float("-inf"))
            hi = rng.get("max", float("inf"))
            # Inclusive of hi so the top bucket's max (e.g. 1.0) is reachable.
            if lo <= value <= hi:
                return g
        # A thresholded scale with no matching bucket -> lowest grade.
        return min(grades, key=lambda g: g.get("points", 0))

    # No thresholds declared (e.g. binary scales): treat the metric as a
    # correctness score in [0, 1]. >= 0.5 earns the top grade, else the bottom.
    top = max(grades, key=lambda g: g.get("points", 0))
    bottom = min(grades, key=lambda g: g.get("points", 0))
    return top if value >= 0.5 else bottom


def score_criterion(
    criterion: Dict[str, Any],
    gold_standard: Dict[str, Any],
    prediction: Dict[str, Any],
    judge: Judge,
    prompt_renderer=None,
) -> Dict[str, Any]:
    """Score one criterion; returns points, max_points, grade, method, reasoning."""
    method = criterion.get("evaluation_method", "automatic")
    result: Dict[str, Any] = {
        "criterion_id": criterion.get("criterion_id", ""),
        "description": criterion.get("description", ""),
        "evaluation_method": method,
        "max_points": _max_points(criterion),
        "min_points": _min_points(criterion),
        "points": 0.0,
        "grade": None,
        "reasoning": "",
    }

    if method == "automatic":
        _score_automatic(criterion, gold_standard, prediction, result)
    elif method == "llm_judge":
        _score_judge(criterion, gold_standard, prediction, judge, prompt_renderer, result)
    elif method == "hybrid":
        auto = dict(result)
        _score_automatic(criterion, gold_standard, prediction, auto)
        jud = dict(result)
        _score_judge(criterion, gold_standard, prediction, judge, prompt_renderer, jud)
        # Conservative: take the lower of the two point awards.
        if auto["points"] <= jud["points"]:
            result.update(auto)
        else:
            result.update(jud)
        result["hybrid_detail"] = {"automatic": auto, "judge": jud}
    else:
        result["reasoning"] = f"unknown evaluation_method {method!r}"

    return result


def _score_automatic(criterion, gold, pred, result) -> None:
    metric_name = criterion.get("metric")
    if not metric_name:
        result["reasoning"] = "automatic criterion has no metric name"
        return
    adapter = get_metric(metric_name)
    if adapter is None:
        result["reasoning"] = f"no metric adapter registered for {metric_name!r}"
        return
    try:
        value = float(adapter(gold, pred))
    except Exception as e:  # defensive: a bad metric must not kill the run
        logger.exception("metric %s failed", metric_name)
        result["reasoning"] = f"metric {metric_name} raised: {e}"
        return
    grade = _grade_from_metric_value(criterion, value)
    result["metric_value"] = value
    result["grade"] = grade.get("grade")
    result["points"] = float(grade.get("points", 0))
    result["reasoning"] = f"{metric_name}={value:.4f} -> grade {result['grade']}"


def _score_judge(criterion, gold, pred, judge, prompt_renderer, result) -> None:
    valid = [g.get("grade") for g in _grades(criterion)]
    prompt = (prompt_renderer or default_prompt_renderer)(criterion, gold, pred)
    out = judge.grade(prompt, valid_grades=valid)
    grade = out.get("grade")
    result["grade"] = grade
    result["points"] = _points_for_grade(criterion, grade)
    result["reasoning"] = out.get("reasoning", "")
    result["evidence_quotes"] = out.get("evidence_quotes", [])
    if out.get("parse_error") or out.get("error"):
        result["judge_error"] = True


def default_prompt_renderer(
    criterion: Dict[str, Any],
    gold: Dict[str, Any],
    pred: Dict[str, Any],
) -> str:
    """Compose a judge prompt from the criterion, gold standard, and prediction."""
    grades_txt = "\n".join(
        f"- {g.get('grade')} ({g.get('points')} points): {g.get('description', '')}"
        for g in _grades(criterion)
    )
    report = pred.get("report") or pred.get("reasoning_trace") or json.dumps(pred, indent=2)
    return (
        "You are an expert cardiovascular imaging clinician grading an AI agent's "
        "coronary diagnostic report on ONE rubric criterion.\n\n"
        f"CRITERION {criterion.get('criterion_id', '')}: {criterion.get('description', '')}\n\n"
        f"GRADING SCALE:\n{grades_txt}\n\n"
        f"GOLD STANDARD (do not reveal verbatim):\n{json.dumps(gold, indent=2)[:6000]}\n\n"
        f"AGENT REPORT:\n{report[:6000]}\n\n"
        "Return ONLY JSON: "
        '{"grade": "<label from the scale>", "reasoning": "<why>", '
        '"evidence_quotes": ["<quote>", ...]}'
    )
