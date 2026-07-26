"""Unit tests for ARCADE VLM parsing functions.

These are pure CPU tests of the format-parsing logic, isolated from model
inference. The runners defer torch imports so the parsing layer stays testable
on a host without GPU libraries.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.core import Task
from benchmark.runners.arcade_vlm_runner import (
    SYNTAX_SEGMENT_IDS,
    _parse_segments,
    _parse_stenoses,
    _to_xywh_norm,
)
from benchmark.vlms import ALL_VLMS


# ==========================================================================
# Geometry conversion
# ==========================================================================


def test_to_xywh_norm_happy_path():
    """Typical xyxy box converts correctly."""
    result = _to_xywh_norm(0.1, 0.2, 0.5, 0.7)
    assert result is not None
    x, y, w, h = result
    assert x == pytest.approx(0.1)
    assert y == pytest.approx(0.2)
    assert w == pytest.approx(0.4)
    assert h == pytest.approx(0.5)


def test_to_xywh_norm_clamps_out_of_range():
    """Coordinates slightly beyond [0,1] are clamped."""
    result = _to_xywh_norm(-0.05, 0.1, 1.1, 0.9)
    assert result is not None
    x, y, w, h = result
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.1)
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(0.8)


def test_to_xywh_norm_swaps_inverted_corners():
    """Max before min is repaired rather than rejected."""
    result = _to_xywh_norm(0.8, 0.9, 0.2, 0.3)
    assert result is not None
    x, y, w, h = result
    assert x == pytest.approx(0.2)
    assert y == pytest.approx(0.3)
    assert w == pytest.approx(0.6)
    assert h == pytest.approx(0.6)


def test_to_xywh_norm_rejects_zero_area():
    """A box with no extent is dropped rather than scored."""
    assert _to_xywh_norm(0.5, 0.5, 0.5, 0.5) is None
    assert _to_xywh_norm(0.3, 0.4, 0.3, 0.6) is None
    assert _to_xywh_norm(0.2, 0.5, 0.8, 0.5) is None


def test_to_xywh_norm_clamp_then_zero_check():
    """Out-of-range coords clamped to [0,1] may become degenerate."""
    # After clamp: x_min=1.0, x_max=1.0 → width=0 → None
    assert _to_xywh_norm(1.1, 0.2, 1.5, 0.8) is None


# ==========================================================================
# Segment parsing
# ==========================================================================


def test_parse_segments_normal_output():
    """Multiple valid lines each produce one instance."""
    text = """
SEGMENT: 1 [0.21, 0.17, 0.40, 0.34]
SEGMENT: 2 [0.24, 0.42, 0.47, 0.63]
SEGMENT: 12a [0.1, 0.2, 0.3, 0.5]
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 3
    assert rejects["rejected_unknown_label"] == 0
    assert rejects["rejected_degenerate_box"] == 0
    
    assert instances[0]["label"] == "1"
    assert len(instances[0]["bbox_xywh_norm"]) == 4
    assert instances[1]["label"] == "2"
    assert instances[2]["label"] == "12a"


def test_parse_segments_ignores_prose():
    """Text outside the format is skipped without error."""
    text = """
I see several coronary segments. Let me identify them:
SEGMENT: 5 [0.1, 0.1, 0.3, 0.3]
The left main is clearly visible.
SEGMENT: 6 [0.2, 0.4, 0.5, 0.7]
No other segments are visible in this view.
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 2
    assert instances[0]["label"] == "5"
    assert instances[1]["label"] == "6"


def test_parse_segments_rejects_unknown_label():
    """Non-SYNTAX ids are counted but not included."""
    text = """
SEGMENT: 1 [0.1, 0.1, 0.3, 0.3]
SEGMENT: 99 [0.2, 0.2, 0.4, 0.4]
SEGMENT: LAD [0.3, 0.3, 0.5, 0.5]
SEGMENT: 7 [0.4, 0.4, 0.6, 0.6]
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 2
    assert instances[0]["label"] == "1"
    assert instances[1]["label"] == "7"
    assert rejects["rejected_unknown_label"] == 2


def test_parse_segments_rejects_degenerate_box():
    """Zero-area boxes are counted but not included."""
    text = """
SEGMENT: 1 [0.5, 0.5, 0.5, 0.5]
SEGMENT: 2 [0.1, 0.1, 0.4, 0.4]
SEGMENT: 3 [0.2, 0.3, 0.2, 0.8]
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 1
    assert instances[0]["label"] == "2"
    assert rejects["rejected_degenerate_box"] == 2


def test_parse_segments_deduplicates_repeated_lines():
    """The same (label, box) repeated is counted once."""
    text = """
SEGMENT: 1 [0.1, 0.1, 0.3, 0.3]
SEGMENT: 1 [0.1, 0.1, 0.3, 0.3]
SEGMENT: 2 [0.2, 0.2, 0.4, 0.4]
SEGMENT: 1 [0.1, 0.1, 0.3, 0.3]
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 2
    assert instances[0]["label"] == "1"
    assert instances[1]["label"] == "2"


def test_parse_segments_same_box_different_label_distinct():
    """Same geometry, different id → two instances (ambiguous case)."""
    text = """
SEGMENT: 1 [0.2, 0.2, 0.5, 0.5]
SEGMENT: 2 [0.2, 0.2, 0.5, 0.5]
    """
    instances, rejects = _parse_segments(text)
    assert len(instances) == 2


def test_parse_segments_empty_or_refusal():
    """No valid output returns empty list, not an exception."""
    assert _parse_segments("I cannot identify segments in this image.") == ([], {"rejected_unknown_label": 0, "rejected_degenerate_box": 0})
    assert _parse_segments("") == ([], {"rejected_unknown_label": 0, "rejected_degenerate_box": 0})


def test_parse_segments_case_insensitive_label():
    """Segment ids are lowercased for matching."""
    text = "SEGMENT: 12A [0.1, 0.1, 0.3, 0.3]"
    instances, _ = _parse_segments(text)
    assert len(instances) == 1
    assert instances[0]["label"] == "12a"


# ==========================================================================
# Stenosis parsing
# ==========================================================================


def test_parse_stenoses_normal_output():
    """Multiple valid stenosis lines parse correctly."""
    text = """
STENOSIS: [0.22, 0.38, 0.31, 0.62]
STENOSIS: [0.34, 0.31, 0.50, 0.60]
    """
    instances, rejects = _parse_stenoses(text)
    assert len(instances) == 2
    assert rejects["rejected_degenerate_box"] == 0
    assert all(i["label"] == "stenosis" for i in instances)


def test_parse_stenoses_rejects_degenerate():
    """Zero-area stenosis boxes are dropped."""
    text = """
STENOSIS: [0.3, 0.3, 0.3, 0.3]
STENOSIS: [0.1, 0.1, 0.4, 0.4]
    """
    instances, rejects = _parse_stenoses(text)
    assert len(instances) == 1
    assert rejects["rejected_degenerate_box"] == 1


def test_parse_stenoses_deduplicates():
    """Repeated stenosis boxes are counted once."""
    text = """
STENOSIS: [0.2, 0.2, 0.4, 0.4]
STENOSIS: [0.2, 0.2, 0.4, 0.4]
    """
    instances, rejects = _parse_stenoses(text)
    assert len(instances) == 1


def test_parse_stenoses_empty_or_refusal():
    """No valid output returns empty list."""
    assert _parse_stenoses("I see no stenoses.") == ([], {"rejected_degenerate_box": 0})
    assert _parse_stenoses("") == ([], {"rejected_degenerate_box": 0})


# ==========================================================================
# Label space completeness
# ==========================================================================


def test_syntax_segment_ids_complete():
    """The parser accepts every id ARCADE gold uses."""
    # From the actual gold in data/tasks/arcade_segmentation
    gold_labels = {
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "9a",
        "10", "11", "12", "12a", "12b", "13", "14", "14a", "14b",
        "15", "16", "16a", "16b", "16c",
    }
    assert gold_labels.issubset(SYNTAX_SEGMENT_IDS), \
        f"Parser missing: {sorted(gold_labels - SYNTAX_SEGMENT_IDS)}"


# ==========================================================================
# VLM method wiring
# ==========================================================================


def test_every_vlm_declares_arcade_tasks():
    """All VLMs must list the two ARCADE tasks in their task tuple."""
    for vlm in ALL_VLMS:
        assert Task.ARCADE_SEGMENTATION in vlm.tasks, \
            f"{vlm.name} missing ARCADE_SEGMENTATION"
        assert Task.ARCADE_STENOSIS in vlm.tasks, \
            f"{vlm.name} missing ARCADE_STENOSIS"


def _stub_runners(monkeypatch):
    """Replace both runner modules with recording stubs.

    ``VLMMethod.predict`` does ``from benchmark.runners import ...``, which
    resolves the attribute on the package, so the attribute is what must be
    patched -- a ``sys.modules`` entry is ignored once the real submodule has
    been imported. Stubbing is necessary because ``vlm_runner`` imports torch
    eagerly and the real ``predict`` would try to load a checkpoint.
    """
    import types

    from benchmark import runners

    reached: dict[str, str] = {}

    def make(module_name: str):
        module = types.ModuleType(module_name)

        def predict(method, case, output_dir, device):
            reached["runner"] = module_name
            return "called"

        module.predict = predict
        return module

    for name in ("vlm_runner", "arcade_vlm_runner"):
        monkeypatch.setattr(runners, name, make(name), raising=False)
    return reached


def _dummy_vlm(tasks):
    from benchmark.core import DomainRelation, Provenance
    from benchmark.vlms import VLMMethod

    return VLMMethod(
        name="test_vlm",
        tasks=tasks,
        repo_id="mock/model",
        provenance=Provenance(
            source="test",
            trained_on="test data",
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a",
            limitations="test only",
        ),
    )


class _FakeCase:
    """Minimal stand-in: dispatch only reads ``task``."""

    def __init__(self, task):
        self.task = task
        self.case_id = "test_case"


def test_vlm_predict_dispatch_routes_each_declared_task(monkeypatch):
    """Every declared task reaches a runner, and the correct one.

    Guards against adding a task to ``_VLM_TASKS`` without wiring the dispatch
    dict: that would surface as an exception on case 12 of 42 rather than here.
    """
    reached = _stub_runners(monkeypatch)
    vlm = _dummy_vlm(
        (
            Task.CARDIOSYNTAX_SCORING,
            Task.ARCADE_SEGMENTATION,
            Task.ARCADE_STENOSIS,
        )
    )

    expected = {
        Task.CARDIOSYNTAX_SCORING: "vlm_runner",
        Task.ARCADE_SEGMENTATION: "arcade_vlm_runner",
        Task.ARCADE_STENOSIS: "arcade_vlm_runner",
    }

    for task in vlm.tasks:
        reached.clear()
        result = vlm.predict(_FakeCase(task), Path("/tmp"), "cpu")
        assert result == "called", f"no runner reached for {task.value}"
        assert reached["runner"] == expected[task], (
            f"{task.value} routed to {reached['runner']}, "
            f"expected {expected[task]}"
        )


def test_vlm_predict_rejects_unwired_task(monkeypatch):
    """An unsupported task raises rather than silently falling back.

    Negative control for the test above: without it, that test would still pass
    if dispatch routed every task to a single runner.
    """
    _stub_runners(monkeypatch)
    vlm = _dummy_vlm((Task.CCA_SEGMENTATION,))

    with pytest.raises(ValueError, match="no VLM runner for task"):
        vlm.predict(_FakeCase(Task.CCA_SEGMENTATION), Path("/tmp"), "cpu")
