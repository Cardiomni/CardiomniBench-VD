"""CardiomniBench-VD pipeline CLI.

Two ways to describe a run:

  A) unified TOML registry (recommended — one file registers every agent):
       python -m pipeline.cli run  --toml benchmark.toml --agent cardiomni
       python -m pipeline.cli agents --toml benchmark.toml   # list registered agents

  B) a single YAML config (one config per run):
       python -m pipeline.cli run  --config configs/smoke.yaml

Other subcommands:
    python -m pipeline.cli list     (--toml … --agent … | --config …)   # discovered cases
    python -m pipeline.cli validate (--toml … --agent … | --config …)   # config check only
    python -m pipeline.cli metrics                                      # registered metrics

CLI flags override the loaded config so the four swap axes can be changed
without editing any file, e.g.:
    python -m pipeline.cli run --toml benchmark.toml --agent cardiomni \
        --model anthropic/claude-opus-4-8 --judge-backend llm --gpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import PipelineConfig
from .metric_registry import list_metrics
from .orchestrator import Orchestrator
from .registry import list_agents, load_registry


def _apply_overrides(cfg: PipelineConfig, args) -> None:
    if args.model:
        cfg.agent.model = args.model
    if args.agent_backend:
        cfg.agent.backend = args.agent_backend
    if args.agent_image:
        cfg.agent.image = args.agent_image
    if args.agent_command:
        cfg.agent.command = args.agent_command
    if args.gpu:
        cfg.agent.gpu = True
    if args.judge_backend:
        cfg.judge.backend = args.judge_backend
    if args.judge_model:
        cfg.judge.model = args.judge_model
    if args.run_name:
        cfg.run_name = args.run_name
    if args.reruns is not None:
        cfg.output.reruns = args.reruns
    cfg.validate()


def _add_source_flags(p: argparse.ArgumentParser) -> None:
    """A run source is either a unified TOML (+agent) or a single YAML config."""
    p.add_argument("--toml", help="path to a unified benchmark.toml registry")
    p.add_argument("--agent", help="agent name to select from the TOML registry")
    p.add_argument("--config", help="path to a single YAML pipeline config")


def _add_override_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-name")
    p.add_argument("--model", help="base model id (换基座)")
    p.add_argument("--agent-backend", choices=["mock", "local", "docker"])
    p.add_argument("--agent-image")
    p.add_argument("--agent-command")
    p.add_argument("--gpu", action="store_true", help="request GPUs (docker backend)")
    p.add_argument("--judge-backend", choices=["mock", "llm", "cli"])
    p.add_argument("--judge-model")
    p.add_argument("--reruns", type=int)


def _load_source(args) -> PipelineConfig:
    """Build a PipelineConfig from --toml/--agent or --config."""
    if getattr(args, "toml", None):
        if not args.agent:
            raise SystemExit("--toml requires --agent <name> (see: agents --toml <file>)")
        return load_registry(args.toml, args.agent)
    if getattr(args, "config", None):
        return PipelineConfig.load(args.config)
    raise SystemExit("provide either --toml <file> --agent <name>, or --config <file>")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="cardiomni-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    _add_source_flags(p_run)
    _add_override_flags(p_run)

    p_list = sub.add_parser("list", help="list discovered cases")
    _add_source_flags(p_list)
    _add_override_flags(p_list)

    p_val = sub.add_parser("validate", help="validate a config without running")
    _add_source_flags(p_val)
    _add_override_flags(p_val)

    p_agents = sub.add_parser("agents", help="list agents registered in a benchmark.toml")
    p_agents.add_argument("--toml", required=True)

    sub.add_parser("metrics", help="list registered objective metrics")

    args = parser.parse_args(argv)

    if args.cmd == "metrics":
        print("\n".join(list_metrics()))
        return 0

    if args.cmd == "agents":
        for name in list_agents(args.toml):
            print(name)
        return 0

    cfg = _load_source(args)
    _apply_overrides(cfg, args)

    if args.cmd == "validate":
        print(f"OK: config valid (agent={cfg.agent.backend}, judge={cfg.judge.backend})")
        return 0

    orch = Orchestrator(cfg)

    if args.cmd == "list":
        for d in orch.discover_cases():
            print(d.name)
        return 0

    if args.cmd == "run":
        summary = orch.run()
        print(json.dumps({
            "run_name": summary["run_name"],
            "num_cases": summary["num_cases"],
            "overall_mean": round(summary["overall_mean"], 2),
            "overall_sd": round(summary["overall_sd"], 2),
            "per_dimension_mean": {k: round(v, 1) for k, v in summary["per_dimension_mean"].items()},
        }, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
