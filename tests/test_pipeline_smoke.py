"""End-to-end gray-box tests for the CardiomniBench-VD pipeline.

These run fully offline (mock agent + mock judge, synthetic fixture case) and
assert the whole flow is wired: config -> discovery -> agent -> scoring ->
aggregate summary. No docker, no API keys, no real DICOM data required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.config import PipelineConfig
from pipeline.metric_registry import get_metric, list_metrics
from pipeline.orchestrator import Orchestrator
from pipeline.runner import build_task_spec, run_agent
from pipeline.judge_backends import make_judge, parse_grade_json

REPO = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO / "configs" / "smoke.yaml"


def test_config_loads_and_validates():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    assert cfg.run_name == "smoke"
    assert cfg.agent.backend == "mock"
    assert cfg.judge.backend == "mock"


def test_config_rejects_bad_backend():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.agent.backend = "nonsense"
    with pytest.raises(ValueError):
        cfg.validate()


def test_docker_backend_requires_image():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.agent.backend = "docker"
    cfg.agent.image = None
    cfg.agent.command = "run"
    with pytest.raises(ValueError):
        cfg.validate()


def test_discovers_smoke_case():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cases = Orchestrator(cfg).discover_cases()
    assert [c.name for c in cases] == ["case_smoke"]


def test_task_spec_strips_gold_standard():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    orch = Orchestrator(cfg)
    case_dir = orch.discover_cases()[0]
    case = orch._load_case(case_dir)
    spec = build_task_spec("case_smoke", case)
    assert "gold_standard" not in spec
    assert spec["input"]["clinical_context"]["age"] == 60
    assert "cta" in spec["input"]


def test_mock_agent_writes_prediction(tmp_path):
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    orch = Orchestrator(cfg)
    case_dir = orch.discover_cases()[0]
    case = orch._load_case(case_dir)
    res = run_agent(cfg.agent, "case_smoke", case, case_dir, tmp_path)
    assert res.ok
    assert (tmp_path / "prediction.json").exists()
    assert (tmp_path / "task_spec.json").exists()
    assert res.prediction["case_id"] == "case_smoke"


def test_full_run_produces_summary(tmp_path):
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.output.root = str(tmp_path)
    summary = Orchestrator(cfg).run()
    assert summary["num_cases"] == 1
    assert summary["reruns"] == 2
    # mock judge grades A + presence-based metrics -> a positive score.
    assert summary["overall_mean"] > 0
    assert summary["overall_sd"] == 0.0  # deterministic across reruns
    assert (tmp_path / "smoke" / "summary.json").exists()
    # Every rubric dimension should appear in the per-dimension breakdown.
    assert set(summary["per_dimension_mean"]) == {
        "data_handling", "perception_accuracy", "fusion_reasoning",
        "clinical_interpretation", "scientific_reasoning", "source_reliability",
    }


def test_summary_json_is_wellformed(tmp_path):
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.output.root = str(tmp_path)
    Orchestrator(cfg).run()
    with open(tmp_path / "smoke" / "summary.json") as f:
        data = json.load(f)
    assert data["agent"]["backend"] == "mock"
    assert data["judge"]["backend"] == "mock"


def test_metric_registry_has_core_metrics():
    names = list_metrics()
    for expected in ("segment_f1_score", "stenosis_mae", "syntax_score_mae"):
        assert expected in names
        assert get_metric(expected) is not None


def test_mock_judge_respects_scale():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    judge = make_judge(cfg.judge)
    out = judge.grade("prompt", valid_grades=["A", "B", "C"])
    assert out["grade"] in ("A", "B", "C")


def test_parse_grade_json_tolerates_fences():
    out = parse_grade_json('```json\n{"grade": "B", "reasoning": "ok"}\n```', ["A", "B", "C"])
    assert out["grade"] == "B"
    out2 = parse_grade_json("garbage not json", ["A", "B", "C"])
    assert out2["grade"] is None and out2["parse_error"]


def test_local_backend_runs_subprocess(tmp_path):
    """local backend: a shell command that writes prediction.json is honored."""
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.agent.backend = "local"
    # Command template writes a minimal valid prediction into {output_dir}.
    cfg.agent.command = (
        "printf '%s' '{\"case_id\": \"case_smoke\", \"report\": \"x\"}' "
        "> {output_dir}/prediction.json"
    )
    cfg.validate()
    orch = Orchestrator(cfg)
    case_dir = orch.discover_cases()[0]
    case = orch._load_case(case_dir)
    res = run_agent(cfg.agent, "case_smoke", case, case_dir, tmp_path)
    assert res.ok, res.error
    assert res.prediction["case_id"] == "case_smoke"


def test_registry_lists_agents():
    from pipeline.registry import list_agents
    names = list_agents(str(REPO / "benchmark.toml"))
    assert "mock" in names
    assert "cardiomni" in names


def test_registry_loads_mock_agent():
    from pipeline.registry import load_registry
    cfg = load_registry(str(REPO / "benchmark.toml"), "mock")
    assert cfg.agent.backend == "mock"
    assert cfg.judge.backend == "mock"
    assert cfg.tasks.source == "data/cases"


def test_registry_agent_inherits_environment():
    """A docker agent inherits the shared [environment] (image/gpu/budgets)."""
    from pipeline.registry import load_registry
    cfg = load_registry(str(REPO / "benchmark.toml"), "cardiomni")
    assert cfg.agent.backend == "docker"
    assert cfg.agent.image == "cardiomni:latest"   # inherited from [environment]
    assert cfg.agent.gpu is True                    # inherited
    assert cfg.agent.cpus == 4                       # inherited budget
    assert cfg.agent.model == "anthropic/claude-opus-4-8"
    assert "{model}" not in cfg.agent.command or "cardiomni.run" in cfg.agent.command


def test_registry_unknown_agent_raises():
    from pipeline.registry import load_registry
    with pytest.raises(KeyError):
        load_registry(str(REPO / "benchmark.toml"), "does_not_exist")


def test_registry_local_agent_overrides_backend():
    from pipeline.registry import load_registry
    cfg = load_registry(str(REPO / "benchmark.toml"), "local_script")
    assert cfg.agent.backend == "local"


def test_binary_scale_grades_on_correctness():
    """Binary criteria (no threshold ranges) must award the top grade when the
    metric signals correct (>=0.5), not silently fall through to the bottom."""
    from pipeline.scoring import _grade_from_metric_value
    binary = {
        "grading_scale": {"type": "binary", "grades": [
            {"grade": "A", "points": 10, "description": "ok"},
            {"grade": "C", "points": 0, "description": "no"},
        ]}
    }
    assert _grade_from_metric_value(binary, 1.0)["grade"] == "A"
    assert _grade_from_metric_value(binary, 0.0)["grade"] == "C"


def test_filter_by_fusion_category():
    cfg = PipelineConfig.load(str(SMOKE_CONFIG))
    cfg.tasks.filter = {"fusion_category": "fusion_required"}
    assert len(Orchestrator(cfg).discover_cases()) == 1
    cfg.tasks.filter = {"fusion_category": "cta_only"}
    assert len(Orchestrator(cfg).discover_cases()) == 0
