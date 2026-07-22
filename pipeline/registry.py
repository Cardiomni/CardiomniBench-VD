"""Unified TOML registry — one benchmark.toml registers everything.

BiomniBench-DA ships one `task.toml` *per task* (see the reference under
`references/`), which is verbose when the environment and judge are identical
across tasks. CardiomniBench-VD collapses that into a SINGLE `benchmark.toml`
at the repo root that registers, in one place:

    [environment]      shared Docker image / GPU / resource budgets
    [judge]            the rubric scorer (mock | llm | cli)
    [tasks]            where cases live + how to filter them
    [rubric]           dimension weights + default case rubric
    [output]           run output dir + reruns
    [agents.<name>]    one table per agent under test (base model + command)

You then run one agent by name:

    python -m pipeline.cli run --toml benchmark.toml --agent cardiomni

`load_registry` merges the shared sections with the chosen agent table and
returns a normal ``PipelineConfig`` — the rest of the pipeline is unchanged.
Parsed with the stdlib ``tomllib`` (Python 3.11+); falls back to ``tomli`` if
present on older interpreters.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

from .config import (
    AgentConfig,
    JudgeConfig,
    OutputConfig,
    PipelineConfig,
    RubricConfig,
    TasksConfig,
)

try:
    import tomllib as _toml  # Python 3.11+
    _TOML_BINARY = True
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    import tomli as _toml  # type: ignore
    _TOML_BINARY = True


def _expand(value: Any) -> Any:
    """Expand ``${ENV_VAR}`` in strings, recursively in containers."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _subset(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def load_toml(path: str) -> Dict[str, Any]:
    """Read and env-expand a TOML file."""
    with open(path, "rb") as f:
        raw = _toml.load(f)
    return _expand(raw)


def list_agents(path: str) -> List[str]:
    """Return the names of all agents registered in the benchmark.toml."""
    raw = load_toml(path)
    return sorted((raw.get("agents", {}) or {}).keys())


def load_registry(path: str, agent: str) -> PipelineConfig:
    """Build a PipelineConfig from benchmark.toml for one named agent.

    The chosen ``[agents.<agent>]`` table is merged over the shared
    ``[environment]`` defaults so every agent inherits the same Docker image,
    GPU, and budgets unless it overrides them.
    """
    toml_path = Path(path).resolve()
    raw = load_toml(str(toml_path))

    agents = raw.get("agents", {}) or {}
    if agent not in agents:
        available = ", ".join(sorted(agents)) or "(none registered)"
        raise KeyError(f"agent {agent!r} not in {toml_path.name}; available: {available}")

    # Shared environment defaults every agent inherits (Docker/GPU/budgets).
    env = raw.get("environment", {}) or {}
    agent_raw = dict(agents[agent])
    merged_agent = {**env, **agent_raw}          # agent table wins over shared env
    merged_agent.setdefault("name", agent)

    cfg = PipelineConfig(
        run_name=agent_raw.get("run_name", raw.get("run_name", agent)),
        seed=raw.get("seed", 42),
        tasks=TasksConfig(**_subset(TasksConfig, raw.get("tasks", {}))),
        agent=AgentConfig(**_subset(AgentConfig, merged_agent)),
        judge=JudgeConfig(**_subset(JudgeConfig, raw.get("judge", {}))),
        rubric=RubricConfig(**_subset(RubricConfig, raw.get("rubric", {}))),
        output=OutputConfig(**_subset(OutputConfig, raw.get("output", {}))),
        base_dir=str(toml_path.parent),
    )
    cfg.validate()
    return cfg
