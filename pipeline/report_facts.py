"""Prose diagnostic report → structured facts → tolerance comparison.

This module implements the locked-in DSA-report evaluation design:

    agent output = a PROSE diagnostic report (中山模板 style, per-segment:
                   vessel + position + stenosis %, including explicit
                   "未见明显狭窄" negatives)
        -> extract structured facts from the prose
        -> compare against the expert gold facts WITH TOLERANCE
           (±N% or same clinical tier; partial credit, never all-or-nothing)

Extraction has two modes so the pipeline runs end-to-end offline *and* uses an
LLM in production:

    heuristic  — regex over Chinese + English segment phrasings. Deterministic,
                 no API key. The offline default and the fallback.
    llm        — ask a judge backend to parse the prose into JSON facts. More
                 robust to free phrasing; used when judge.backend != mock.

An agent may also self-report ``prediction["extracted_facts"]`` directly; if
present those win (lets a structured agent skip the lossy extraction step, and
lets tests inject exact facts).

The comparison is deliberately clinical, not string-exact:
    * segments match on normalized (vessel, position) — synonyms folded in
    * a stenosis reading is "correct" if within ``tolerance_percent`` OR in the
      same tier as gold (tiers default to <50 / 50-69 / 70-99 / 100)
    * anti-hallucination counts pred lesions with no gold counterpart
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# --- vessel / position vocabulary --------------------------------------------
# Fold the many ways a segment gets named (Chinese clinical, English, SYNTAX id)
# down to a canonical (vessel, position). Only anatomy the four-view DSA task
# cares about — extend as more cases surface new branch names.

_VESSEL_SYNONYMS: Dict[str, str] = {
    # Left main
    "左主干": "LM", "lm": "LM", "left main": "LM", "lmca": "LM",
    # LAD
    "前降支": "LAD", "lad": "LAD", "左前降支": "LAD",
    "left anterior descending": "LAD",
    # LCX
    "回旋支": "LCX", "lcx": "LCX", "左回旋支": "LCX",
    "left circumflex": "LCX", "circumflex": "LCX",
    # RCA
    "右冠": "RCA", "右冠状动脉": "RCA", "rca": "RCA",
    "right coronary": "RCA", "right coronary artery": "RCA",
    # Named branches (treated as their own "vessel" for matching)
    "对角支": "D", "第一对角支": "D1", "第二对角支": "D2",
    "diagonal": "D", "d1": "D1", "d2": "D2",
    "钝缘支": "OM", "obtuse marginal": "OM", "om": "OM", "om1": "OM1",
    "后降支": "PDA", "pda": "PDA", "posterior descending": "PDA",
    "左室后支": "PLB", "plb": "PLB", "posterolateral": "PLB",
    "中间支": "RI", "ramus": "RI", "ri": "RI",
}

_POSITION_SYNONYMS: Dict[str, str] = {
    "近段": "proximal", "近端": "proximal", "proximal": "proximal", "prox": "proximal",
    "中段": "mid", "中端": "mid", "mid": "mid", "middle": "mid",
    "远段": "distal", "远端": "distal", "distal": "distal",
    "开口": "ostial", "开口部": "ostial", "ostial": "ostial", "ostium": "ostial",
    "体部": "mid", "分叉": "bifurcation", "bifurcation": "bifurcation",
}

# SYNTAX id -> canonical (vessel, position). Lets gold/pred reference either the
# numeric SYNTAX segment or the descriptive name and still match.
_SYNTAX_ID_MAP: Dict[str, Tuple[str, str]] = {
    "1": ("RCA", "proximal"), "RCA_1": ("RCA", "proximal"),
    "2": ("RCA", "mid"), "RCA_2": ("RCA", "mid"),
    "3": ("RCA", "distal"), "RCA_3": ("RCA", "distal"),
    "4": ("PDA", ""), "RCA_4": ("PDA", ""), "RCA_16": ("PDA", ""),
    "5": ("LM", ""), "LM_5": ("LM", ""),
    "6": ("LAD", "proximal"), "LAD_6": ("LAD", "proximal"),
    "7": ("LAD", "mid"), "LAD_7": ("LAD", "mid"),
    "8": ("LAD", "distal"), "LAD_8": ("LAD", "distal"),
    "9": ("D1", ""), "LAD_9": ("D1", ""),
    "10": ("D2", ""), "LAD_10": ("D2", ""),
    "11": ("LCX", "proximal"), "LCX_11": ("LCX", "proximal"),
    "12": ("OM", ""), "LCX_12": ("OM", ""),
    "13": ("LCX", "distal"), "LCX_13": ("LCX", "distal"),
    "14": ("PLB", ""), "LCX_14": ("PLB", ""),
    "15": ("PDA", ""), "LCX_15": ("PDA", ""),
}


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def canonical_segment(seg: Dict[str, Any]) -> Tuple[str, str]:
    """Fold a fact dict to a canonical (vessel, position) key for matching.

    Accepts any of: ``segment_id`` (SYNTAX), ``vessel`` + ``position``, or a raw
    ``name``. Unknown vessels pass through uppercased so novel branches still
    match themselves across gold/pred.

    When both `segment_id` and an explicit `position` field are present, the
    explicit position wins (allows gold to say "LCX_11 but actually no position
    stated in the report" by setting `segment_id: "LCX_11", position: ""`).
    """
    sid = str(seg.get("segment_id") or "").strip()
    # If an explicit position is given, honor it even if segment_id would imply one.
    explicit_position = seg.get("position")
    has_explicit_position = (explicit_position is not None) and ("position" in seg)

    if sid and not has_explicit_position:
        key = _SYNTAX_ID_MAP.get(sid) or _SYNTAX_ID_MAP.get(sid.upper())
        if key:
            return key

    vessel_raw = _norm(seg.get("vessel")) or _norm(seg.get("name"))
    position_raw = _norm(explicit_position if has_explicit_position else seg.get("position"))

    vessel = _VESSEL_SYNONYMS.get(vessel_raw, vessel_raw.upper() if vessel_raw else "")
    # A raw name may pack vessel+position ("前降支近段"): scan it for both.
    if not position_raw and vessel_raw:
        for token, canon in _POSITION_SYNONYMS.items():
            if token in vessel_raw:
                position_raw = token
                break
    position = _POSITION_SYNONYMS.get(position_raw, position_raw)
    return (vessel, position)


# --- clinical tiers -----------------------------------------------------------

def stenosis_tier(pct: Optional[float], boundaries: Tuple[int, ...] = (50, 70, 100)) -> str:
    """Map a stenosis % to a clinical tier label.

    Default boundaries (from the alignment meeting): <50 / 50-69 / 70-99 / 100.
    None -> "unknown" so a missing reading never silently counts as 0%.
    """
    if pct is None:
        return "unknown"
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return "unknown"
    if v >= boundaries[2]:
        return "occluded"
    if v >= boundaries[1]:
        return "severe"
    if v >= boundaries[0]:
        return "moderate"
    return "none_mild"


# --- heuristic extraction (offline default) -----------------------------------

# Match "<segment phrase> ... <pct>%" and bare "未见明显狭窄" negatives.
_PCT_RE = re.compile(r"(\d{1,3})\s*%")
_NEG_RE = re.compile(r"(未见明显狭窄|未见狭窄|无明显狭窄|no significant stenosis|no stenosis|patent)")


def _find_vessel_position(text: str) -> Tuple[str, str]:
    """Scan a text fragment for the first vessel + position tokens it contains."""
    low = text.lower()
    vessel = ""
    # Longest synonym first so "左前降支" beats "前降支" etc.
    for token in sorted(_VESSEL_SYNONYMS, key=len, reverse=True):
        if token in low or token in text:
            vessel = _VESSEL_SYNONYMS[token]
            break
    position = ""
    for token in sorted(_POSITION_SYNONYMS, key=len, reverse=True):
        if token in low or token in text:
            position = _POSITION_SYNONYMS[token]
            break
    return vessel, position


def extract_facts_heuristic(report_text: str) -> Dict[str, Any]:
    """Parse a prose report into structured facts with regex heuristics.

    Splits on newlines / semicolons / Chinese punctuation into segment-level
    fragments, then reads a vessel, position and stenosis % (or negative) from
    each. Deterministic and API-free — the offline baseline and the fallback
    when LLM extraction is unavailable. It is intentionally conservative: a
    fragment with no recognizable vessel is skipped rather than guessed.
    """
    facts: Dict[str, Any] = {"dominance": "", "segments": []}
    if not report_text:
        return facts

    low = report_text.lower()
    if "右优势" in report_text or "right dominant" in low or "right-dominant" in low:
        facts["dominance"] = "right"
    elif "左优势" in report_text or "left dominant" in low or "left-dominant" in low:
        facts["dominance"] = "left"
    elif "均衡型" in report_text or "co-dominant" in low or "balanced" in low:
        facts["dominance"] = "balanced"

    # Split on line breaks, semicolons, periods AND commas: a single line often
    # packs two segments ("前降支:近端狭窄60%,第一对角支未见狭窄"). Without the
    # comma split, longest-vessel-match would grab 第一对角支 and misattribute the
    # 60% to it. Commas (、，,) are safe separators — a fragment with no vessel is
    # simply skipped.
    fragments = re.split(r"[\n;；。,，、]+", report_text)
    seen: set = set()
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        vessel, position = _find_vessel_position(frag)
        if not vessel:
            continue
        key = (vessel, position)
        pct_match = _PCT_RE.search(frag)
        if pct_match:
            pct: Optional[float] = float(pct_match.group(1))
        elif _NEG_RE.search(frag):
            pct = 0.0
        else:
            continue  # a vessel mention with neither % nor an explicit negative
        # First mention of a segment wins (reports often re-reference distally).
        if key in seen:
            continue
        seen.add(key)
        facts["segments"].append({
            "vessel": vessel,
            "position": position,
            "stenosis_percent": pct,
            "source_text": frag[:120],
        })
    return facts


# --- LLM extraction (production) ----------------------------------------------

_EXTRACTION_PROMPT = """You extract structured facts from a coronary angiography (DSA) diagnostic report.

Read the report below and output ONLY JSON with this shape:
{{
  "dominance": "right" | "left" | "balanced" | "",
  "segments": [
    {{"vessel": "<LM|LAD|LCX|RCA|D1|OM|PDA|...>",
      "position": "<proximal|mid|distal|ostial|>",
      "stenosis_percent": <integer 0-100>}}
  ]
}}

Rules:
- One entry per vessel segment the report describes, INCLUDING segments stated
  as "未见明显狭窄"/"no significant stenosis" (use stenosis_percent 0).
- "完全闭塞"/"total occlusion" -> 100. If a range is given, use the midpoint.
- Do NOT invent segments the report does not mention.

REPORT:
{report}
"""


def extract_facts_llm(report_text: str, judge: Any) -> Optional[Dict[str, Any]]:
    """Extract facts by asking a judge backend to parse the prose into JSON.

    ``judge`` is any object exposing ``.grade(prompt, valid_grades=None)`` or a
    raw ``.complete(prompt)``; we reuse the judge's model plumbing so no new
    client is needed. Returns None on any failure so callers fall back to the
    heuristic extractor.
    """
    if not report_text or judge is None:
        return None
    prompt = _EXTRACTION_PROMPT.format(report=report_text[:8000])
    text = None
    try:
        if hasattr(judge, "complete"):
            text = judge.complete(prompt)
        elif hasattr(judge, "grade"):
            out = judge.grade(prompt, valid_grades=None)
            text = out.get("raw") or out.get("reasoning") or ""
    except Exception:
        logger.exception("LLM fact extraction failed")
        return None
    if not text:
        return None
    return _parse_facts_json(text)


def _parse_facts_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM response (tolerates fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fenced.group(1) if fenced else None
    if blob is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        blob = text[start:end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("dominance", "")
    data.setdefault("segments", [])
    return data


def extract_facts(
    prediction: Dict[str, Any],
    judge: Any = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Resolve structured facts for a prediction.

    Precedence:
        1. ``prediction["extracted_facts"]`` if the agent self-reported them
        2. LLM extraction from ``prediction["report"]`` (mode "llm"/"auto"+judge)
        3. heuristic extraction from the prose (always available)

    ``mode``: "auto" (facts -> llm-if-judge -> heuristic), "llm", "heuristic".
    """
    self_reported = prediction.get("extracted_facts")
    if isinstance(self_reported, dict) and self_reported.get("segments") is not None:
        return self_reported

    report = prediction.get("report") or prediction.get("reasoning_trace") or ""

    if mode in ("auto", "llm") and judge is not None:
        llm_facts = extract_facts_llm(report, judge)
        if llm_facts is not None:
            return llm_facts
        if mode == "llm":
            logger.warning("LLM extraction returned nothing; falling back to heuristic")

    return extract_facts_heuristic(report)


# --- tolerance comparison -----------------------------------------------------

def _index(segments: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for seg in segments or []:
        idx[canonical_segment(seg)] = seg
    return idx


def compare_facts(
    gold_facts: Dict[str, Any],
    pred_facts: Dict[str, Any],
    tolerance_percent: float = 10.0,
    tier_boundaries: Tuple[int, ...] = (50, 70, 100),
    significance_threshold: float = 50.0,
) -> Dict[str, Any]:
    """Compare predicted facts against gold with clinical tolerance.

    Returns a dict of sub-scores in [0, 1] (anti_hallucination excepted, which
    is a 0-1 cleanliness score), each with the counts behind it:

        coverage_recall     — gold segments the report also covered (incl. negatives)
        naming_accuracy     — of covered segments, fraction with exact SYNTAX naming
        stenosis_accuracy   — of covered segments, fraction within ±tol or same tier
        dominance_correct   — 1.0 if dominance matches, else 0.0 (unknown gold -> 1.0)
        anti_hallucination  — 1 - fabricated_significant / max(1, pred_significant)

    A matched pair is one where gold and pred fold to the same (vessel, position).
    """
    gold_segs = gold_facts.get("segments", []) or []
    pred_segs = pred_facts.get("segments", []) or []
    gold_idx = _index(gold_segs)
    pred_idx = _index(pred_segs)

    # -- coverage: which gold segments did the report mention at all? ---------
    matched_keys = [k for k in gold_idx if k in pred_idx]
    coverage_recall = (len(matched_keys) / len(gold_idx)) if gold_idx else 1.0

    # -- naming: of matched, how many carry the exact SYNTAX id gold expects? --
    naming_hits = 0
    naming_total = 0
    for k in matched_keys:
        g = gold_idx[k]
        p = pred_idx[k]
        g_id = str(g.get("segment_id") or "").upper()
        if not g_id:
            continue  # gold has no canonical id to check against
        naming_total += 1
        p_id = str(p.get("segment_id") or "").upper()
        # Credit an exact id match, or a name that folds to the same canonical key.
        if p_id == g_id or canonical_segment(p) == canonical_segment(g):
            naming_hits += 1
    naming_accuracy = (naming_hits / naming_total) if naming_total else 1.0

    # -- stenosis: within ±tol OR same clinical tier --------------------------
    sten_hits = 0
    sten_detail: List[Dict[str, Any]] = []
    for k in matched_keys:
        g_pct = _as_float(gold_idx[k].get("stenosis_percent"))
        p_pct = _as_float(pred_idx[k].get("stenosis_percent"))
        ok = False
        if g_pct is not None and p_pct is not None:
            within = abs(g_pct - p_pct) <= tolerance_percent
            same_tier = stenosis_tier(g_pct, tier_boundaries) == stenosis_tier(p_pct, tier_boundaries)
            ok = within or same_tier
        if ok:
            sten_hits += 1
        sten_detail.append({
            "segment": "_".join(k), "gold": g_pct, "pred": p_pct, "correct": ok,
        })
    stenosis_accuracy = (sten_hits / len(matched_keys)) if matched_keys else 1.0

    # -- dominance ------------------------------------------------------------
    g_dom = _norm(gold_facts.get("dominance"))
    p_dom = _norm(pred_facts.get("dominance"))
    dominance_correct = 1.0 if (not g_dom or g_dom == p_dom) else 0.0

    # -- anti-hallucination: pred significant lesions with no gold basis ------
    fabricated: List[str] = []
    pred_significant = 0
    for k, p in pred_idx.items():
        p_pct = _as_float(p.get("stenosis_percent"))
        if p_pct is None or p_pct < significance_threshold:
            continue
        pred_significant += 1
        g = gold_idx.get(k)
        g_pct = _as_float(g.get("stenosis_percent")) if g else None
        # Fabricated if no gold segment there, or gold there is well below
        # significance and outside tolerance (a real lesion invented from noise).
        if g is None or (
            g_pct is not None
            and g_pct < significance_threshold
            and abs(g_pct - p_pct) > tolerance_percent
        ):
            fabricated.append("_".join(k))
    anti_hallucination = 1.0 - (len(fabricated) / max(1, pred_significant))

    return {
        "coverage_recall": round(coverage_recall, 4),
        "naming_accuracy": round(naming_accuracy, 4),
        "stenosis_accuracy": round(stenosis_accuracy, 4),
        "dominance_correct": dominance_correct,
        "anti_hallucination": round(anti_hallucination, 4),
        "counts": {
            "gold_segments": len(gold_idx),
            "pred_segments": len(pred_idx),
            "matched": len(matched_keys),
            "stenosis_correct": sten_hits,
            "fabricated_significant": fabricated,
            "pred_significant": pred_significant,
        },
        "stenosis_detail": sten_detail,
    }


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
