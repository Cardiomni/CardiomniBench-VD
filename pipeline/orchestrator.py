"""Orchestrator — the end-to-end pipeline: tasks -> agent -> score -> aggregate.

Flow (mirrors BiomniBench's per-task evaluation loop):

    load config
      -> discover cases (tasks.source / split / filter / limit)
      -> for each rerun, for each case:
            run_agent(...)                 # mock | local | docker
            -> prediction.json
            score all rubric criteria      # metric adapters + judge backend
            -> per-dimension + overall case score
      -> aggregate: mean ± SD over reruns, per-dimension, fusion-lift hook

Everything runs offline with backend=mock: no docker, no API keys, no real data.
Results are written under output.root/run_name/.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import PipelineConfig
from .judge_backends import make_judge
from .runner import run_agent
from .scoring import score_criterion

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.dimensions = self._load_dimensions()
        self.judge = make_judge(cfg.judge)
        # Resolve to an absolute path (relative to the config file, like other
        # config paths). Must be absolute: the local/docker backends run with a
        # different cwd, so a relative output dir would resolve against the wrong
        # base and the agent's prediction.json would land outside the run dir.
        self.run_dir = self.cfg.resolve(cfg.output.root) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # -- setup -----------------------------------------------------------------

    def _load_dimensions(self) -> List[Dict[str, Any]]:
        path = self.cfg.resolve(self.cfg.rubric.dimensions_file)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data["dimensions"]

    def _dimension_weight(self, name: str) -> float:
        for d in self.dimensions:
            if d["name"] == name:
                return float(d.get("weight", 0.0))
        return 0.0

    # -- task discovery --------------------------------------------------------

    def discover_cases(self) -> List[Path]:
        """Return case directories per tasks.source / split / filter / limit."""
        source = self.cfg.resolve(self.cfg.tasks.source)
        if source is None or not source.exists():
            logger.warning("tasks.source %s does not exist; no cases", source)
            return []

        case_dirs = sorted(
            d for d in source.glob(self.cfg.tasks.case_glob) if d.is_dir()
        )

        # Restrict to a named split, if configured.
        allowed = self._split_case_ids()
        if allowed is not None:
            case_dirs = [d for d in case_dirs if d.name in allowed]

        # Metadata equality filters (e.g. fusion_category: fusion_required).
        if self.cfg.tasks.filter:
            case_dirs = [d for d in case_dirs if self._passes_filter(d)]

        if self.cfg.tasks.limit is not None:
            case_dirs = case_dirs[: self.cfg.tasks.limit]

        logger.info("discovered %d case(s)", len(case_dirs))
        return case_dirs

    def _split_case_ids(self) -> Optional[set]:
        if not self.cfg.tasks.split or not self.cfg.tasks.splits_file:
            return None
        path = self.cfg.resolve(self.cfg.tasks.splits_file)
        if path is None or not path.exists():
            logger.warning("splits_file %s missing; ignoring split", path)
            return None
        with open(path, "r") as f:
            splits = yaml.safe_load(f) or {}
        ids = (splits.get(self.cfg.tasks.split, {}) or {}).get("case_ids", []) or []
        return set(ids)

    def _passes_filter(self, case_dir: Path) -> bool:
        case = self._load_case(case_dir)
        meta = case.get("case_metadata", {}) or {}
        for key, want in self.cfg.tasks.filter.items():
            if want is None:
                continue
            if meta.get(key) != want:
                return False
        return True

    # -- case loading ----------------------------------------------------------

    def _load_case(self, case_dir: Path) -> Dict[str, Any]:
        """Load a case's task spec + gold standard from its YAML file(s)."""
        # Preferred layout: a single task YAML (task_template shape).
        for candidate in ("task.yaml", "gold_standard.yaml", "case.yaml"):
            p = case_dir / candidate
            if p.exists():
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
        raise FileNotFoundError(f"no task.yaml/gold_standard.yaml in {case_dir}")

    def _load_case_rubric(self, case_dir: Path) -> Optional[Dict[str, Any]]:
        """Per-case rubric wins; else the config's default_case_rubric."""
        local = case_dir / self.cfg.rubric.case_rubric_name
        if local.exists():
            with open(local, "r") as f:
                return yaml.safe_load(f)
        default = self.cfg.resolve(self.cfg.rubric.default_case_rubric)
        if default and default.exists():
            with open(default, "r") as f:
                return yaml.safe_load(f)
        return None

    # -- evaluation ------------------------------------------------------------

    def evaluate_case(
        self,
        case_dir: Path,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Score one prediction against one case's rubric; weighted 0-100."""
        case = self._load_case(case_dir)
        gold = case.get("gold_standard", {}) or {}
        rubric = self._load_case_rubric(case_dir)

        out: Dict[str, Any] = {
            "case_id": case_dir.name,
            "dimensions": {},
            "overall_score": 0.0,
        }
        if not rubric:
            out["error"] = "no rubric available"
            return out

        # Pre-extract facts from prose report if needed (for DSA-report tolerance metrics).
        # If the agent self-reported extracted_facts, those win. Otherwise, if the judge
        # backend is LLM/CLI (not mock), use it to parse the prose. Falls back to heuristic.
        if "extracted_facts" not in prediction and "report" in prediction:
            from . import report_facts as rf
            mode = "auto" if self.cfg.judge.backend != "mock" else "heuristic"
            prediction["extracted_facts"] = rf.extract_facts(prediction, self.judge, mode=mode)

        weighted_sum = 0.0
        weight_total = 0.0
        for dim in rubric.get("dimensions", []):
            dim_name = dim["dimension_name"]
            weight = self._dimension_weight(dim_name) or float(dim.get("weight", 0.0))
            crit_results = []
            earned = 0.0
            possible = 0.0
            for criterion in dim.get("criteria", []):
                r = score_criterion(criterion, gold, prediction, self.judge)
                crit_results.append(r)
                earned += r["points"]
                possible += r["max_points"]
            pct = (earned / possible * 100.0) if possible > 0 else 0.0
            out["dimensions"][dim_name] = {
                "weight": weight,
                "earned": earned,
                "possible": possible,
                "percent": pct,
                "criteria": crit_results,
            }
            weighted_sum += pct * weight
            weight_total += weight

        out["overall_score"] = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        return out

    # -- run loop --------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and write results; return the summary."""
        cases = self.discover_cases()
        mock_pred = self.cfg.resolve(self.cfg.agent.mock_prediction)

        per_rerun: List[Dict[str, Any]] = []
        for rerun in range(self.cfg.output.reruns):
            rerun_dir = self.run_dir / f"rerun_{rerun}"
            case_scores: List[Dict[str, Any]] = []

            for case_dir in cases:
                case = self._load_case(case_dir)
                out_dir = rerun_dir / case_dir.name
                agent_res = run_agent(
                    self.cfg.agent, case_dir.name, case, case_dir, out_dir,
                    mock_prediction_path=mock_pred,
                )
                if not agent_res.ok or agent_res.prediction is None:
                    logger.warning("[%s] agent failed: %s", case_dir.name, agent_res.error)
                    case_scores.append({
                        "case_id": case_dir.name,
                        "overall_score": 0.0,
                        "agent_ok": False,
                        "agent_error": agent_res.error,
                    })
                    continue

                ev = self.evaluate_case(case_dir, agent_res.prediction)
                ev["agent_ok"] = True
                ev["fusion_category"] = (case.get("case_metadata", {}) or {}).get("fusion_category")
                case_scores.append(ev)

                with open(out_dir / "evaluation.json", "w") as f:
                    json.dump(ev, f, indent=2)

            per_rerun.append({"rerun": rerun, "cases": case_scores})

        summary = self._summarize(per_rerun)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("run complete: %s", self.run_dir / "summary.json")
        return summary

    def _summarize(self, per_rerun: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate over reruns: mean ± SD overall and per dimension."""
        rerun_means: List[float] = []
        for rr in per_rerun:
            scores = [c["overall_score"] for c in rr["cases"]]
            rerun_means.append(statistics.mean(scores) if scores else 0.0)

        # Per-dimension mean over the first rerun's cases (representative).
        dim_means: Dict[str, float] = {}
        if per_rerun:
            first = per_rerun[0]["cases"]
            dim_names = set()
            for c in first:
                dim_names.update((c.get("dimensions") or {}).keys())
            for name in dim_names:
                vals = [
                    c["dimensions"][name]["percent"]
                    for c in first
                    if c.get("dimensions", {}).get(name)
                ]
                dim_means[name] = statistics.mean(vals) if vals else 0.0

        return {
            "run_name": self.cfg.run_name,
            "agent": {"backend": self.cfg.agent.backend, "name": self.cfg.agent.name,
                      "model": self.cfg.agent.model},
            "judge": {"backend": self.cfg.judge.backend, "model": self.cfg.judge.model},
            "num_cases": len(per_rerun[0]["cases"]) if per_rerun else 0,
            "reruns": self.cfg.output.reruns,
            "overall_mean": statistics.mean(rerun_means) if rerun_means else 0.0,
            "overall_sd": statistics.pstdev(rerun_means) if len(rerun_means) > 1 else 0.0,
            "per_dimension_mean": dim_means,
            "per_rerun_mean": rerun_means,
            "cases": per_rerun[0]["cases"] if per_rerun else [],
        }


def run_pipeline(config_path: str) -> Dict[str, Any]:
    """Load a config and run the full pipeline."""
    cfg = PipelineConfig.load(config_path)
    return Orchestrator(cfg).run()
