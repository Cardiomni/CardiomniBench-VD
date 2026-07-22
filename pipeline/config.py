"""Pipeline configuration — the single source of the four swap axes.

A run is fully described by one YAML file. Load it with ``PipelineConfig.load``.
Every axis has a safe default so a minimal config still runs end-to-end with
mock backends (no docker, no API keys, no real data).

Example (see configs/smoke.yaml)::

    run_name: smoke
    tasks:
      source: tests/fixtures/tasks      # directory of task dirs
      split: null                        # optional splits.yaml split name
      filter: {fusion_category: null}    # optional metadata filters
    agent:
      backend: mock                      # mock | local | docker
      model: mock-model
      command: "echo"                    # template; {task_dir}/{model}/... expanded
    judge:
      backend: mock                      # mock | llm | cli
      model: mock-judge
    rubric:
      dimensions_file: rubrics/rubric_dimensions.yaml
      default_case_rubric: rubrics/examples/case_001_rubric.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand(value: Any) -> Any:
    """Expand ``${ENV_VAR}`` references in strings, recursively in containers."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _subset(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are declared fields of the dataclass ``cls``."""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass
class TasksConfig:
    """Which cases to run — the '换任务' axis."""

    source: str = "data/cases"
    split: Optional[str] = None            # name of a split in splits_file
    splits_file: Optional[str] = None      # e.g. data/splits.yaml
    filter: Dict[str, Any] = field(default_factory=dict)  # metadata equality filters
    case_glob: str = "case_*"              # directory name pattern under source
    limit: Optional[int] = None            # cap number of cases (gray-box testing)


@dataclass
class AgentConfig:
    """The system under test — the '换基座' + '换 agent' axes.

    ``backend`` picks the runner:
        mock   — return a canned prediction (no execution; gray-box path)
        local  — run ``command`` as a subprocess on this host
        docker — run ``command`` inside ``image`` (adds --gpus when gpu=true)

    ``command`` is a template. Available placeholders:
        {task_dir}    absolute path to the case directory
        {task_spec}   path to the generated task_spec.json (gold-standard stripped)
        {output_dir}  where the agent must write prediction.json
        {model}       the base model id (the '换基座' knob)
        {extra_args}  joined agent.extra_args

    This mirrors BiomniBench's ``harbor run --path <task> --agent <H> --model <M>``:
    model and harness are orthogonal — swap either without touching the other.
    """

    backend: str = "mock"                  # mock | local | docker
    name: str = "mock-agent"               # label for reports
    model: str = "mock-model"              # base model id ('换基座')
    command: str = ""                      # command template ('换 agent')
    image: Optional[str] = None            # docker image (backend=docker)
    gpu: bool = False                      # request GPUs (backend=docker)
    gpu_device: str = "all"                # value for docker --gpus
    workdir: str = "/workspace"            # cwd inside container
    env: Dict[str, str] = field(default_factory=dict)   # extra env passed through
    extra_args: List[str] = field(default_factory=list)
    timeout_s: int = 3600                  # per-case wall-clock budget
    mock_prediction: Optional[str] = None  # path to canned prediction (backend=mock)


@dataclass
class JudgeConfig:
    """Rubric scoring for subjective criteria — the '换 rubric CLI/LLM' axis.

    ``backend`` picks how llm_judge / cli criteria are graded:
        mock — deterministic canned grade (offline, gray-box path)
        llm  — call an LLM API (needs api_key_env set in the environment)
        cli  — shell out to an external judge command (see command template)
    """

    backend: str = "mock"                  # mock | llm | cli
    model: str = "mock-judge"
    temperature: float = 0.0
    max_tokens: int = 4000
    api_key_env: str = "ANTHROPIC_API_KEY"
    num_judges: int = 1                    # reruns for kappa (validation)
    command: str = ""                      # cli backend template ({prompt_file})
    prompt_template: Optional[str] = None  # path to judge prompt template
    mock_grade: str = "B"                  # default grade for backend=mock


@dataclass
class RubricConfig:
    """Where dimension weights and per-case criteria live."""

    dimensions_file: str = "rubrics/rubric_dimensions.yaml"
    default_case_rubric: Optional[str] = "rubrics/examples/case_001_rubric.yaml"
    # If a case dir has its own rubric.yaml it wins over default_case_rubric.
    case_rubric_name: str = "rubric.yaml"


@dataclass
class OutputConfig:
    root: str = "runs"                     # results written under root/run_name/
    reruns: int = 1                        # independent reruns (mean ± SD)
    save_predictions: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "default"
    seed: int = 42
    tasks: TasksConfig = field(default_factory=TasksConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    rubric: RubricConfig = field(default_factory=RubricConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Directory the config was loaded from; relative paths resolve against it.
    base_dir: str = "."

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load and validate a pipeline config from a YAML file."""
        cfg_path = Path(path).resolve()
        with open(cfg_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        raw = _expand(raw)

        cfg = cls(
            run_name=raw.get("run_name", "default"),
            seed=raw.get("seed", 42),
            tasks=TasksConfig(**_subset(TasksConfig, raw.get("tasks", {}))),
            agent=AgentConfig(**_subset(AgentConfig, raw.get("agent", {}))),
            judge=JudgeConfig(**_subset(JudgeConfig, raw.get("judge", {}))),
            rubric=RubricConfig(**_subset(RubricConfig, raw.get("rubric", {}))),
            output=OutputConfig(**_subset(OutputConfig, raw.get("output", {}))),
            base_dir=str(cfg_path.parent),
        )
        cfg.validate()
        return cfg

    def resolve(self, maybe_relative: Optional[str]) -> Optional[Path]:
        """Resolve a config path against base_dir (absolute paths pass through)."""
        if maybe_relative is None:
            return None
        p = Path(maybe_relative)
        if p.is_absolute():
            return p
        return (Path(self.base_dir) / p).resolve()

    def validate(self) -> None:
        """Fail fast on invalid enum choices; leave path existence to runtime."""
        if self.agent.backend not in ("mock", "local", "docker"):
            raise ValueError(f"agent.backend must be mock|local|docker, got {self.agent.backend!r}")
        if self.judge.backend not in ("mock", "llm", "cli"):
            raise ValueError(f"judge.backend must be mock|llm|cli, got {self.judge.backend!r}")
        if self.agent.backend == "docker" and not self.agent.image:
            raise ValueError("agent.backend=docker requires agent.image")
        if self.agent.backend in ("local", "docker") and not self.agent.command:
            raise ValueError(f"agent.backend={self.agent.backend} requires agent.command")
        if self.judge.backend == "cli" and not self.judge.command:
            raise ValueError("judge.backend=cli requires judge.command")
        if self.output.reruns < 1:
            raise ValueError("output.reruns must be >= 1")


def load_config(path: str) -> PipelineConfig:
    """Module-level convenience wrapper."""
    return PipelineConfig.load(path)
