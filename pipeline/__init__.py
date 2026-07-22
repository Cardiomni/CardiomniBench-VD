"""CardiomniBench-VD evaluation pipeline.

Config-driven orchestration mirroring BiomniBench-DA / Harbor:
    task directory -> agent (base model + harness) -> prediction
        -> rubric evaluation (objective metrics + judge) -> aggregate scores.

Four independent swap axes, all set from a single YAML config:
    * base model   (agent.model)
    * agent harness(agent.backend / agent.command / agent.image)
    * judge        (judge.backend: mock | llm | cli)
    * tasks        (tasks.source / split / filter)
"""

__all__ = ["config", "runner", "judge_backends", "metric_registry", "scoring", "orchestrator"]
