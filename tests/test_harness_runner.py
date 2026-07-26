"""Unit tests for harness runner orchestration loop.

These are pure CPU tests that validate the loop logic without running real models
or tool implementations. The runner uses `use_mock_tools=True` to avoid GPU
dependencies, and the model generation is stubbed out to return fixed text so the
parsing and control flow can be verified.
"""

from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmark.core import Task
from benchmark.harnesses import HarnessMethod
from benchmark.io_spec import CaseInput
from benchmark.runners.harness_runner import (
    _build_prompt,
    _execute_tool,
    _parse_final_answer,
    _parse_tool_calls,
)


# ==========================================================================
# SOP isolation
# ==========================================================================
#
# The unguided baseline must not learn the staged structure it is the control for.
# The leak these tests guard against was real: the tool layer's
# `capability_boundary` string says "Suitable for Stage 4 geometry/quantification,
# not Stage 2 naming", and that text is copied verbatim into the prompt because
# tool limitations are disclosed honestly. The effect was to tell the control group
# that a staged pipeline exists and which stage each tool belongs to, which biases
# the comparison toward *under*-stating what the SOP contributes.
#
# `_build_prompt` strips the stage numbers as a temporary patch. The real fix is
# for tool metadata to stop referencing the caller's orchestration at all; these
# tests keep the leak from returning in the meantime, and they still pass once the
# upstream text is cleaned up.


def _real_tool_metadata() -> dict:
    """Tool metadata as `predict` actually assembles it, not a hand-written stub.

    Using the real `segmentation_metadata()` / `detection_metadata()` output is the
    point: the mock dicts used elsewhere in this file never contained the staging
    vocabulary, which is why the leak survived the first round of tests.
    """
    from algorithms.tools.stenosis_detection import detection_metadata, is_available
    from algorithms.tools.vessel_segmentation import segmentation_metadata

    seg = segmentation_metadata()
    det_available, _ = is_available()
    det = detection_metadata()

    return {
        "segment_vessels": {
            "signature": "segment_vessels(image_path, device='cuda:1')",
            "description": "Segment coronary vessels in a 2D XCA frame.",
            "capability_boundary": seg.get("capability_boundary", ""),
            "available": True,
        },
        "detect_stenosis": {
            "signature": "detect_stenosis(image_path, device='cuda:1')",
            "description": "Detect stenoses automatically.",
            "available": det_available,
            "blocker": det.get("blocker", ""),
            "alternatives": det.get("alternatives", []),
        },
    }


@pytest.mark.parametrize(
    "task",
    [Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS, Task.CARDIOSYNTAX_SCORING],
)
def test_prompt_leaks_no_stage_numbers(task):
    """No prompt may carry SOP stage numbers, whatever the tool metadata says."""
    case = CaseInput(task=task, case_id="x", case_dir=Path("/tmp"), spec={})
    prompt = _build_prompt(case, _real_tool_metadata())

    assert not re.search(r"\bstage \d+\b", prompt, re.IGNORECASE), (
        f"SOP stage number leaked into the {task.value} prompt: {prompt!r}"
    )


def test_prompt_leaks_no_sop_vocabulary():
    """Staging vocabulary beyond bare numbers stays out of the baseline prompt."""
    case = CaseInput(
        task=Task.ARCADE_SEGMENTATION, case_id="x", case_dir=Path("/tmp"), spec={}
    )
    lowered = _build_prompt(case, _real_tool_metadata()).lower()

    for banned in ("four-stage", "four stage", "dominance", "dominant", "sop"):
        assert banned not in lowered, f"SOP hint {banned!r} leaked into the prompt"


def test_sanitizer_preserves_capability_facts():
    """Stripping stage numbers must not remove the honest capability disclosure.

    Over-filtering would be its own bug: if the prompt stopped saying that the
    segmenter cannot produce SYNTAX ids, the harness would be measured without
    being told what its tools can do, which is a different unfair comparison.
    """
    case = CaseInput(
        task=Task.ARCADE_SEGMENTATION, case_id="x", case_dir=Path("/tmp"), spec={}
    )
    prompt = _build_prompt(case, _real_tool_metadata())

    assert "Cannot produce SYNTAX segment ids" in prompt
    assert "WARNING" in prompt  # the unavailable tool is still flagged
    assert "diameter_qca" in prompt  # its alternatives are still offered


def test_sanitizer_handles_stage_variants():
    """The filter covers case and number variants, not just the current wording."""
    case = CaseInput(
        task=Task.ARCADE_SEGMENTATION, case_id="x", case_dir=Path("/tmp"), spec={}
    )
    meta = {
        "t": {
            "signature": "t()",
            "description": "d",
            "capability_boundary": "Use in STAGE 3, never stage 12 or Stage 4.",
            "available": True,
        }
    }
    prompt = _build_prompt(case, meta)

    assert not re.search(r"\bstage \d+\b", prompt, re.IGNORECASE)


# ==========================================================================
# Tool call parsing
# ==========================================================================


def test_parse_tool_calls_single():
    """A single TOOL: line is extracted."""
    text = "Let me segment the vessels.\nTOOL: segment_vessels(image_path=auto, device=cuda:1)"
    calls = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "segment_vessels"
    assert calls[0]["args"]["image_path"] == "auto"
    assert calls[0]["args"]["device"] == "cuda:1"


def test_parse_tool_calls_multiple():
    """Multiple TOOL: lines in one output are all extracted."""
    text = """
I'll segment first, then quantify.
TOOL: segment_vessels(image_path=auto, device=auto)
TOOL: quantify_stenosis(image_path=auto, roi_coords=100,200,300,400)
"""
    calls = _parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "segment_vessels"
    assert calls[1]["name"] == "quantify_stenosis"


def test_parse_tool_calls_no_args():
    """A tool with no arguments is handled."""
    text = "TOOL: some_tool()"
    calls = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "some_tool"
    assert calls[0]["args"] == {}
    # Empty parentheses are a genuine no-arg call, not a format failure.
    assert calls[0]["arg_parse_failed"] is False


def test_parse_tool_calls_positional_args_flagged():
    """Positional arguments yield no name=value pairs and are flagged.

    A model that writes `segment_vessels("image.png")` has called the tool in a
    form the executor cannot act on. That is different from calling it with no
    arguments, and the difference is a measurable harness behaviour.
    """
    calls = _parse_tool_calls('TOOL: segment_vessels("/path/to/image.png")')
    assert len(calls) == 1
    assert calls[0]["args"] == {}
    assert calls[0]["arg_parse_failed"] is True


def test_parse_tool_calls_none():
    """Text without TOOL: markers returns empty list."""
    text = "I think we should look at the LAD. No tools needed yet."
    calls = _parse_tool_calls(text)
    assert calls == []


def test_parse_tool_calls_case_insensitive():
    """TOOL: matching is case-insensitive."""
    text = "tool: segment_vessels(image_path=auto)"
    calls = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "segment_vessels"


# ==========================================================================
# Mock tool execution
# ==========================================================================


def test_execute_tool_mock_segment():
    """Mock segment_vessels returns a dict with expected keys."""
    case = Mock()
    case.case_dir = Path("/tmp/mock")
    
    result = _execute_tool(
        "segment_vessels",
        {"image_path": "auto", "device": "cuda:1"},
        {},
        case,
        "cuda:1",
        use_mock=True,
    )
    
    assert result["tool_source"] == "mock"
    assert "result" in result
    assert "mask_shape" in result["result"]
    assert "foreground_fraction" in result["result"]


def test_execute_tool_mock_quantify():
    """Mock quantify_stenosis returns stenosis measurements."""
    case = Mock()
    case.case_dir = Path("/tmp/mock")
    
    result = _execute_tool(
        "quantify_stenosis",
        {"image_path": "auto", "roi_coords": "100,200,300,400"},
        {},
        case,
        "cuda:1",
        use_mock=True,
    )
    
    assert result["tool_source"] == "mock"
    assert "result" in result
    assert "percent_stenosis" in result["result"]
    assert "severity_class" in result["result"]


def test_execute_tool_mock_detect_raises():
    """Mock detect_stenosis always fails, as the real one does."""
    case = Mock()
    case.case_dir = Path("/tmp/mock")
    
    result = _execute_tool(
        "detect_stenosis",
        {"image_path": "auto"},
        {},
        case,
        "cuda:1",
        use_mock=True,
    )
    
    assert result["tool_source"] == "mock"
    assert "error" in result
    assert "NotImplementedError" in result["error"]


def test_execute_tool_unknown():
    """An unknown tool name produces an error."""
    case = Mock()
    case.case_dir = Path("/tmp/mock")
    
    result = _execute_tool(
        "nonexistent_tool",
        {},
        {},
        case,
        "cuda:1",
        use_mock=True,
    )
    
    assert "error" in result
    assert "unknown tool" in result["error"]


# ==========================================================================
# Final answer parsing
# ==========================================================================


def test_parse_final_answer_scoring():
    """ANSWER: with a number is extracted for scoring tasks."""
    text = "Based on the analysis, ANSWER: 23.5"
    answer = _parse_final_answer(text, Task.CARDIOSYNTAX_SCORING)
    
    assert answer is not None
    assert "score" in answer
    assert answer["score"] == 23.5
    # A recovered number is a successful structured parse.
    assert answer["parsed"] == 23.5


def test_parse_final_answer_scoring_no_number_is_unparseable():
    """Prose with no number is an answer, but not a scoreable one.

    This is the distinction the results table needs: the harness did respond, so
    this is not a budget failure, but nothing scoreable came back. Both outcomes
    score zero and must not look identical in the record.
    """
    answer = _parse_final_answer(
        "ANSWER: the complexity appears elevated but I cannot quantify it",
        Task.CARDIOSYNTAX_SCORING,
    )

    assert answer is not None
    assert answer["parsed"] is None
    assert "score" not in answer


def test_parse_final_answer_arcade_never_claims_parsed():
    """ARCADE answers carry raw text only, with parsed left explicitly None.

    The runner must not rescue an unstructured answer into instances: doing the
    harness's work for it would stop the unguided baseline from being unguided.
    """
    for task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        answer = _parse_final_answer(
            "ANSWER: SEGMENT: 1 [0.1, 0.1, 0.3, 0.3]", task
        )
        assert answer is not None
        assert answer["parsed"] is None
        assert "raw_output" in answer


def test_parse_final_answer_always_has_parsed_key():
    """Every non-None return carries `parsed`, so .get() cannot be ambiguous."""
    cases = [
        ("ANSWER: 12", Task.CARDIOSYNTAX_SCORING),
        ("ANSWER: no number here", Task.CARDIOSYNTAX_SCORING),
        ("ANSWER: something", Task.ARCADE_SEGMENTATION),
        ("ANSWER: something", Task.ARCADE_STENOSIS),
    ]
    for text, task in cases:
        answer = _parse_final_answer(text, task)
        assert answer is not None
        assert "parsed" in answer, f"missing 'parsed' for {task}"


def test_parse_final_answer_segmentation():
    """ANSWER: for segmentation tasks captures the raw text."""
    text = "ANSWER: I found 3 segments: LAD proximal, mid, and distal."
    answer = _parse_final_answer(text, Task.ARCADE_SEGMENTATION)
    
    assert answer is not None
    assert "raw_output" in answer


def test_parse_final_answer_none():
    """Text without ANSWER: returns None."""
    text = "Still analyzing the vessels, need more information."
    answer = _parse_final_answer(text, Task.ARCADE_SEGMENTATION)
    
    assert answer is None


def test_parse_final_answer_case_insensitive():
    """ANSWER: matching is case-insensitive."""
    text = "answer: 15"
    answer = _parse_final_answer(text, Task.CARDIOSYNTAX_SCORING)
    
    assert answer is not None
    assert answer["score"] == 15.0


# ==========================================================================
# Prompt construction
# ==========================================================================


def test_build_prompt_includes_task():
    """The prompt contains the task description."""
    case = Mock()
    case.task = Task.ARCADE_SEGMENTATION
    
    prompt = _build_prompt(case, {})
    
    assert "SYNTAX" in prompt
    assert "segment" in prompt.lower()


def test_build_prompt_includes_tools():
    """Tool signatures and descriptions appear in the prompt."""
    case = Mock()
    case.task = Task.ARCADE_STENOSIS
    
    tools = {
        "segment_vessels": {
            "signature": "segment_vessels(image_path, device)",
            "description": "Segment the vessel tree",
            "available": True,
        }
    }
    
    prompt = _build_prompt(case, tools)
    
    assert "segment_vessels" in prompt
    assert "Segment the vessel tree" in prompt


def test_build_prompt_marks_unavailable():
    """Unavailable tools are flagged with WARNING."""
    case = Mock()
    case.task = Task.ARCADE_STENOSIS
    
    tools = {
        "detect_stenosis": {
            "signature": "detect_stenosis(image_path)",
            "description": "Detect stenoses",
            "available": False,
            "blocker": "weights missing",
            "alternatives": ["use segment + quantify instead"],
        }
    }
    
    prompt = _build_prompt(case, tools)
    
    assert "WARNING" in prompt
    assert "weights missing" in prompt
    assert "Alternatives" in prompt


def test_build_prompt_includes_capability_boundary():
    """Tool limitations are stated in the prompt."""
    case = Mock()
    case.task = Task.ARCADE_SEGMENTATION
    
    tools = {
        "segment_vessels": {
            "signature": "segment_vessels(image_path)",
            "available": True,
            "capability_boundary": "Cannot name SYNTAX segments",
        }
    }
    
    prompt = _build_prompt(case, tools)
    
    assert "Cannot name SYNTAX segments" in prompt


# ==========================================================================
# End-to-end loop structure (with mocks)
# ==========================================================================


def test_predict_completes_with_mock_tools():
    """The full predict function runs without real tools or model."""
    from benchmark.harnesses import HarnessMethod
    from benchmark.core import Provenance, DomainRelation
    
    method = HarnessMethod(
        name="test_harness",
        tasks=(Task.CARDIOSYNTAX_SCORING,),
        base_model="llava_16_mistral_7b",
        tools=("segment_vessels",),
        max_turns=3,
        use_mock_tools=True,
        provenance=Provenance(
            source="test",
            trained_on="test",
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a",
            limitations="test only",
        ),
    )
    
    case = Mock()
    case.case_id = "test_case"
    case.task = Task.CARDIOSYNTAX_SCORING
    case.case_dir = Path(tempfile.mkdtemp())
    
    # Create a dummy image
    img_path = case.case_dir / "image.png"
    img_path.touch()
    
    try:
        from benchmark.runners.harness_runner import predict
        
        pred = predict(method, case, Path("/tmp"), "cuda:1")
        
        assert pred.case_id == "test_case"
        assert pred.task == Task.CARDIOSYNTAX_SCORING
        assert "trace" in pred.diagnostics
        assert "harness" in pred.diagnostics
        assert pred.diagnostics["harness"] == "test_harness"
        assert pred.diagnostics["tool_source"] == "mock"
        
    finally:
        # Cleanup
        img_path.unlink()
        case.case_dir.rmdir()


def test_predict_records_trace():
    """Every turn is recorded in diagnostics["trace"]."""
    from benchmark.harnesses import HarnessMethod
    from benchmark.core import Provenance, DomainRelation
    
    method = HarnessMethod(
        name="test_harness",
        tasks=(Task.ARCADE_STENOSIS,),
        base_model="llava_16_mistral_7b",
        tools=("segment_vessels", "detect_stenosis"),
        max_turns=2,
        use_mock_tools=True,
        provenance=Provenance(
            source="test",
            trained_on="test",
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a",
            limitations="test",
        ),
    )
    
    case = Mock()
    case.case_id = "test_trace"
    case.task = Task.ARCADE_STENOSIS
    case.case_dir = Path(tempfile.mkdtemp())
    
    img_path = case.case_dir / "image.png"
    img_path.touch()
    
    try:
        from benchmark.runners.harness_runner import predict
        
        pred = predict(method, case, Path("/tmp"), "cuda:1")
        
        assert "trace" in pred.diagnostics
        trace = pred.diagnostics["trace"]
        assert isinstance(trace, list)
        assert len(trace) > 0
        
        # First turn should have turn number and model output
        assert "turn" in trace[0]
        assert "model_output" in trace[0]
        
    finally:
        img_path.unlink()
        case.case_dir.rmdir()


def test_predict_respects_max_turns():
    """The loop stops at max_turns even without a final answer."""
    from benchmark.harnesses import HarnessMethod
    from benchmark.core import Provenance, DomainRelation
    
    method = HarnessMethod(
        name="test_budget",
        tasks=(Task.ARCADE_SEGMENTATION,),
        base_model="llava_16_mistral_7b",
        tools=("segment_vessels",),
        max_turns=2,
        use_mock_tools=True,
        provenance=Provenance(
            source="test",
            trained_on="test",
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a",
            limitations="test",
        ),
    )
    
    case = Mock()
    case.case_id = "test_budget"
    case.task = Task.ARCADE_SEGMENTATION
    case.case_dir = Path(tempfile.mkdtemp())
    
    img_path = case.case_dir / "image.png"
    img_path.touch()
    
    try:
        from benchmark.runners.harness_runner import predict
        
        pred = predict(method, case, Path("/tmp"), "cuda:1")
        
        assert pred.diagnostics["turns_used"] <= method.max_turns
        assert pred.diagnostics["max_turns"] == 2
        
    finally:
        img_path.unlink()
        case.case_dir.rmdir()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="patches vlm_runner.generate_turn, and importing it requires torch",
)
def test_generation_error_yields_empty_prediction():
    """When generation raises, the case produces an empty prediction, not a crash.

    The bug this guards against was real: if the VLM call raised, `final_answer`
    stayed None, and `final_answer.get(...)` at Prediction construction crashed with
    AttributeError. That masked the real error (which was in diagnostics but never
    reached the caller) and made the failure look like a code bug rather than a
    model issue.
    """
    from benchmark.harnesses import HarnessMethod
    from benchmark.core import Provenance, DomainRelation
    
    method = HarnessMethod(
        name="test_gen_fail", tasks=(Task.ARCADE_SEGMENTATION,),
        base_model="llava_16_mistral_7b", tools=("segment_vessels",),
        max_turns=2, use_mock_tools=False,
        provenance=Provenance(source="t", trained_on="n/a",
            domain_relation=DomainRelation.NOT_TRAINED, reported_metric="n/a",
            limitations="t"),
    )
    
    case = Mock()
    case.case_id = "fail_test"
    case.task = Task.ARCADE_SEGMENTATION
    case.case_dir = Path(tempfile.mkdtemp())
    (case.case_dir / "image.png").touch()
    
    try:
        # `generate_turn` is imported inside `predict`, so the patch has to land on
        # the source module rather than on harness_runner's namespace.
        import benchmark.runners.vlm_runner as vr
        orig = vr.generate_turn

        def fail(*a, **k):
            raise RuntimeError("forced generation failure")

        vr.generate_turn = fail

        from benchmark.runners.harness_runner import predict
        pred = predict(method, case, Path(tempfile.mkdtemp()), "cuda:1")
        
        # Should not crash, should return empty with error in diagnostics
        assert pred.case_id == "fail_test"
        assert pred.diagnostics["generation_error"] is not None
        assert "RuntimeError" in pred.diagnostics["generation_error"]
        assert pred.diagnostics["budget_exhausted"] is False
        assert pred.raw_output == ""  # empty, not a fake answer
        
        # Mock tools were False, so unless something was already created, this is empty
        assert len(pred.instances or []) == 0
        
    finally:
        vr.generate_turn = orig
        (case.case_dir / "image.png").unlink()
        case.case_dir.rmdir()


def test_placeholder_image_path_counts_as_override():
    """A syntactically valid call carrying a useless path is still a semantic miss.

    `_execute_tool` resolves `image_path` itself, so the case succeeds even when the
    model emits '/path/to/image'. That repair is deliberate, but it must be visible:
    without this counter a model that never learns where the data lives scores the
    same as one that always gets it right, and the first real run showed the VLM
    emitting exactly such a placeholder.
    """
    case_dir = Path(tempfile.mkdtemp())
    (case_dir / "image.png").touch()
    case = Mock()
    case.case_dir = case_dir
    case.task = Task.ARCADE_SEGMENTATION

    result = _execute_tool(
        "segment_vessels",
        {"image_path": "/path/to/xca_image"},
        {},
        case,
        "cpu",
        use_mock=False,
    )

    overrides = result.get("args_overridden", [])
    assert len(overrides) == 1
    assert overrides[0]["arg"] == "image_path"
    assert overrides[0]["model_value"] == "/path/to/xca_image"
    assert overrides[0]["used_value"].endswith("image.png")


def test_correct_image_path_is_not_an_override():
    """Resolving to the same path the model gave must not inflate the counter."""
    case_dir = Path(tempfile.mkdtemp())
    image = case_dir / "image.png"
    image.touch()
    case = Mock()
    case.case_dir = case_dir
    case.task = Task.ARCADE_SEGMENTATION

    result = _execute_tool(
        "segment_vessels",
        {"image_path": str(image)},
        {},
        case,
        "cpu",
        use_mock=False,
    )

    assert "args_overridden" not in result
