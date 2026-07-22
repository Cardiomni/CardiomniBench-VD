"""Tests for prose-report fact extraction + tolerance comparison.

Anchored on the user's own locked-in gold example (中山-style prose report).
"""

from __future__ import annotations

import pytest

from pipeline import report_facts as rf


# The exact shape from the locked-in design discussion.
GOLD_REPORT = """冠脉分布为右优势型。
1. 左冠状动脉:
   (a) 左主干:未见狭窄
   (b) 前降支:近端狭窄60%,第一对角支未见狭窄
   (c) 回旋支:未见明显狭窄
2. 右冠状动脉:近段狭窄75%
"""

GOLD_FACTS = {
    "dominance": "right",
    "segments": [
        {"segment_id": "LM_5", "vessel": "LM", "position": "", "stenosis_percent": 0},
        {"segment_id": "LAD_6", "vessel": "LAD", "position": "proximal", "stenosis_percent": 60},
        {"segment_id": "LAD_9", "vessel": "D1", "position": "", "stenosis_percent": 0},
        {"segment_id": "LCX_11", "vessel": "LCX", "position": "", "stenosis_percent": 0},  # no position in report text
        {"segment_id": "RCA_1", "vessel": "RCA", "position": "proximal", "stenosis_percent": 75},
    ],
}


def test_heuristic_extracts_all_segments_from_gold_prose():
    facts = rf.extract_facts_heuristic(GOLD_REPORT)
    assert facts["dominance"] == "right"
    keys = {rf.canonical_segment(s) for s in facts["segments"]}
    # LM, LAD-prox, D1, LCX, RCA-prox all present.
    assert ("LM", "") in keys
    assert ("LAD", "proximal") in keys
    assert ("D1", "") in keys
    assert ("LCX", "") in keys or ("LCX", "proximal") in keys
    assert ("RCA", "proximal") in keys


def test_comma_split_attributes_60pct_to_lad_not_d1():
    """Regression: '前降支:近端狭窄60%,第一对角支未见狭窄' must give LAD=60, D1=0."""
    facts = rf.extract_facts_heuristic(GOLD_REPORT)
    by_key = {rf.canonical_segment(s): s for s in facts["segments"]}
    assert by_key[("LAD", "proximal")]["stenosis_percent"] == 60.0
    assert by_key[("D1", "")]["stenosis_percent"] == 0.0


def test_perfect_report_scores_full_marks():
    pred = {"report": GOLD_REPORT}
    facts = rf.extract_facts(pred, mode="heuristic")
    cmp = rf.compare_facts(GOLD_FACTS, facts)
    assert cmp["coverage_recall"] == 1.0
    assert cmp["stenosis_accuracy"] == 1.0
    assert cmp["dominance_correct"] == 1.0
    assert cmp["anti_hallucination"] == 1.0


def test_stenosis_tolerance_within_10pct():
    pred = {"report": "右优势型。右冠近段狭窄68%。"}
    facts = rf.extract_facts(pred, mode="heuristic")
    # gold RCA=75, pred=68 -> within ±10 AND same tier (severe). Correct.
    cmp = rf.compare_facts({"dominance": "right",
                            "segments": [{"segment_id": "RCA_1", "vessel": "RCA",
                                          "position": "proximal", "stenosis_percent": 75}]},
                           facts)
    assert cmp["stenosis_accuracy"] == 1.0


def test_stenosis_tier_boundary_counts_wrong():
    pred = {"report": "右冠近段狭窄45%。"}
    facts = rf.extract_facts(pred, mode="heuristic")
    # gold 75 (severe) vs pred 45 (none_mild): >10 apart AND different tier -> wrong.
    cmp = rf.compare_facts({"segments": [{"segment_id": "RCA_1", "vessel": "RCA",
                                          "position": "proximal", "stenosis_percent": 75}]},
                           facts)
    assert cmp["stenosis_accuracy"] == 0.0


def test_partial_coverage_gives_partial_credit():
    # Report only mentions 3 of the 5 gold segments.
    pred = {"report": "右优势型。左主干未见狭窄。前降支近段狭窄60%。右冠近段狭窄75%。"}
    facts = rf.extract_facts(pred, mode="heuristic")
    cmp = rf.compare_facts(GOLD_FACTS, facts)
    assert 0.5 <= cmp["coverage_recall"] < 1.0
    assert cmp["counts"]["matched"] == 3


def test_hallucination_penalized():
    # Report invents a severe LCX lesion gold says is clean.
    pred = {"report": "右优势型。回旋支近段狭窄80%。"}
    facts = rf.extract_facts(pred, mode="heuristic")
    cmp = rf.compare_facts(GOLD_FACTS, facts)
    assert cmp["anti_hallucination"] < 1.0
    assert "LCX_proximal" in cmp["counts"]["fabricated_significant"] or \
           any("LCX" in f for f in cmp["counts"]["fabricated_significant"])


def test_occlusion_maps_to_100_tier():
    assert rf.stenosis_tier(100) == "occluded"
    assert rf.stenosis_tier(85) == "severe"
    assert rf.stenosis_tier(55) == "moderate"
    assert rf.stenosis_tier(20) == "none_mild"
    assert rf.stenosis_tier(None) == "unknown"


def test_self_reported_facts_take_precedence():
    pred = {"report": "garbage that would not parse",
            "extracted_facts": {"dominance": "right", "segments": [
                {"segment_id": "RCA_1", "stenosis_percent": 75}]}}
    facts = rf.extract_facts(pred, mode="heuristic")
    assert facts["segments"][0]["segment_id"] == "RCA_1"


def test_syntax_id_and_name_fold_to_same_key():
    a = rf.canonical_segment({"segment_id": "LAD_6"})
    b = rf.canonical_segment({"vessel": "前降支", "position": "近段"})
    assert a == b == ("LAD", "proximal")
