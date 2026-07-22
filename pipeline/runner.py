"""Agent runners — execute the system under test to produce a prediction.

Three backends behind one ``run_agent`` entrypoint, selected by ``agent.backend``:

    mock    canned prediction, no execution      (offline gray-box path)
    local   subprocess on this host              (tests command-template wiring)
    docker  subprocess inside a container image   (adds --gpus when gpu=true)

The contract mirrors BiomniBench/Harbor: the agent is handed a task directory
and a gold-standard-stripped ``task_spec.json``, and must write ``prediction.json``
into ``output_dir``. Because the output schema is fixed here (not by the harness),
any agent — GPT-4V, a Claude-Code-style agent, Cardiomni — is scored uniformly.

Nothing in this module requires docker to be installed: the docker backend only
shells out when actually invoked with ``agent.backend=docker``.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AgentConfig

logger = logging.getLogger(__name__)

PREDICTION_FILENAME = "prediction.json"
TASK_SPEC_FILENAME = "task_spec.json"


@dataclass
class AgentResult:
    """Outcome of one agent invocation on one case."""

    case_id: str
    prediction: Optional[Dict[str, Any]]
    ok: bool
    backend: str
    command: Optional[str] = None
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


def build_task_spec(case_id: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Build the agent-facing task spec: inputs + clinical context, NO gold standard.

    Keeps the agent blind to the answer (BiomniBench forbids reading the source),
    while giving it everything a clinician would have: the DICOM file paths, the
    clinical context, and the required output sections.
    """
    inp = case.get("input", {}) or {}
    expected = case.get("expected_output", {}) or {}
    return {
        "case_id": case_id,
        "input": {
            "cta": inp.get("cta", {}),
            "dsa": inp.get("dsa", {}),
            "clinical_context": inp.get("clinical_context", {}),
            "prohibited_resources": inp.get("prohibited_resources", {}),
        },
        "expected_output": {
            "format": expected.get("format", "structured_json"),
            "required_sections": expected.get("required_sections", []),
        },
        "instructions": (
            "Produce a structured coronary diagnostic report as prediction.json. "
            "Populate every required section. Do NOT fabricate FFR/IVUS/OCT, "
            "perfusion, viability, lab, or history data that is not provided — "
            "state explicitly when such data would be needed."
        ),
    }


def _render_command(template: str, mapping: Dict[str, str]) -> str:
    """Fill a command template by replacing only the known ``{placeholder}`` tokens.

    Uses literal string replacement (not str.format) so command templates may
    freely contain other braces — JSON payloads, shell ``${VAR}``, awk ``{print}``
    — without being misparsed as format fields.
    """
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def run_agent(
    cfg: AgentConfig,
    case_id: str,
    case: Dict[str, Any],
    case_dir: Path,
    output_dir: Path,
    mock_prediction_path: Optional[Path] = None,
) -> AgentResult:
    """Dispatch to the configured backend and return an AgentResult."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always write the task spec so local/docker agents have a stable input file.
    spec = build_task_spec(case_id, case)
    spec_path = output_dir / TASK_SPEC_FILENAME
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    if cfg.backend == "mock":
        return _run_mock(cfg, case_id, output_dir, mock_prediction_path)
    if cfg.backend == "local":
        return _run_subprocess(cfg, case_id, case_dir, output_dir, spec_path, docker=False)
    if cfg.backend == "docker":
        return _run_subprocess(cfg, case_id, case_dir, output_dir, spec_path, docker=True)
    return AgentResult(case_id, None, ok=False, backend=cfg.backend,
                       error=f"unknown backend {cfg.backend!r}")


def _run_mock(
    cfg: AgentConfig,
    case_id: str,
    output_dir: Path,
    mock_prediction_path: Optional[Path],
) -> AgentResult:
    """Return a canned prediction — the offline path used for gray-box testing."""
    if mock_prediction_path and Path(mock_prediction_path).exists():
        with open(mock_prediction_path, "r") as f:
            prediction = json.load(f)
    else:
        prediction = _default_mock_prediction(case_id)

    pred_path = output_dir / PREDICTION_FILENAME
    with open(pred_path, "w") as f:
        json.dump(prediction, f, indent=2)

    return AgentResult(case_id, prediction, ok=True, backend="mock",
                       command=None, returncode=0)


def _run_subprocess(
    cfg: AgentConfig,
    case_id: str,
    case_dir: Path,
    output_dir: Path,
    spec_path: Path,
    docker: bool,
) -> AgentResult:
    """Run the agent command locally or inside a docker container."""
    mapping = {
        "task_dir": str(case_dir),
        "task_spec": str(spec_path),
        "output_dir": str(output_dir),
        "model": cfg.model,
        "extra_args": " ".join(cfg.extra_args),
    }
    inner = _render_command(cfg.command, mapping)

    if docker:
        cmd = _wrap_docker(cfg, inner, case_dir, output_dir)
    else:
        cmd = inner

    logger.info("[%s] running (%s): %s", case_id, cfg.backend, cmd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
            cwd=str(case_dir) if not docker else None,
        )
    except subprocess.TimeoutExpired as e:
        return AgentResult(case_id, None, ok=False, backend=cfg.backend,
                           command=cmd, error=f"timeout after {cfg.timeout_s}s: {e}")
    except OSError as e:
        return AgentResult(case_id, None, ok=False, backend=cfg.backend,
                           command=cmd, error=f"exec error: {e}")

    prediction, load_err = _load_prediction(output_dir / PREDICTION_FILENAME)
    ok = proc.returncode == 0 and prediction is not None
    return AgentResult(
        case_id, prediction, ok=ok, backend=cfg.backend, command=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout[-4000:], stderr=proc.stderr[-4000:],
        error=load_err if not ok else None,
    )


def _wrap_docker(cfg: AgentConfig, inner: str, case_dir: Path, output_dir: Path) -> str:
    """Compose a ``docker run`` command around the agent's inner command.

    GPU is a first-class knob: ``agent.gpu=true`` adds ``--gpus <device>`` so the
    same config works on a GPU box (the target server) without code changes.
    """
    parts: List[str] = ["docker", "run", "--rm"]
    if cfg.gpu:
        parts += ["--gpus", cfg.gpu_device]
    # Resource budgets (0 = unset), mirroring BiomniBench's task.toml [environment].
    if cfg.cpus:
        parts += ["--cpus", str(cfg.cpus)]
    if cfg.memory_mb:
        parts += ["--memory", f"{cfg.memory_mb}m"]
    # Mount the case dir (read-only) and the output dir (writable).
    parts += ["-v", f"{shlex.quote(str(case_dir))}:{cfg.workdir}/task:ro"]
    parts += ["-v", f"{shlex.quote(str(output_dir))}:{cfg.workdir}/out"]
    parts += ["-w", cfg.workdir]
    for key, val in cfg.env.items():
        parts += ["-e", f"{key}={val}"]
    parts += [cfg.image, "bash", "-lc", shlex.quote(inner)]
    return " ".join(parts)


def _load_prediction(path: Path):
    """Load prediction.json, returning (data, error_message)."""
    if not path.exists():
        return None, f"agent did not write {path.name}"
    try:
        with open(path, "r") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON in {path.name}: {e}"


def _default_mock_prediction(case_id: str) -> Dict[str, Any]:
    """A schema-complete placeholder report so downstream scoring has real shape."""
    return {
        "case_id": case_id,
        "anatomical_localization": {"dominance": "right", "segments_identified": []},
        "cta_findings": {"segments": [], "calcium_score": {}, "high_risk_plaque_features": {}},
        "dsa_findings": {"segments": [], "timi_flow": [], "collaterals": {}},
        "fusion_analysis": {"blooming_correction": {}, "cto_assessment": {}, "culprit_lesion": {}},
        "comprehensive_scoring": {"syntax_score": {}, "cadrads_per_patient": ""},
        "clinical_decision": {"recommendation": "", "rationale": ""},
        "capability_boundary_statement": (
            "Mock prediction — no imaging analysis performed."
        ),
        "reasoning_trace": "Mock reasoning trace for pipeline gray-box testing.",
        "report": "Mock diagnostic report generated by the mock agent backend.",
    }
