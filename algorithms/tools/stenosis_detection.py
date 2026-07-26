"""Stenosis detection tool for the Cardiomni agent — BLOCKED, weights unavailable.

This is an honest stub. It raises instead of returning a plausible-looking empty
result, because a tool that silently returns "no stenoses found" is
indistinguishable from a working detector on a normal case, and that ambiguity is
how a broken pipeline reports good numbers.

What is missing
---------------
PROPOSAL.md §2.6 names DeepCORO-CLIP as an optional second-opinion stenosis
detector. Its weights are not on this host and cannot currently be obtained:

    specialist_models/deepcoro_clip/weights/deepcoro_clip_stenosis/   (empty)
    specialist_models/deepcoro_clip/weights/VasoVision/               (empty)

A full-tree search for ``*.pth``, ``*.pt``, ``*.ckpt`` and ``*.safetensors``
under both ``deepcoro/`` and ``deepcoro_clip/`` returns nothing.

Per the repo's own ``deepcoro_clip/ACCESS_PENDING.md`` (dated 2026-07-23), the
upstream HuggingFace repositories are gated and require manual approval from the
authors:

    heartwise/deepcoro_clip            HTTP 401  unauthorized
    heartwise/deepcoro_clip_stenosis   HTTP 307  redirect to _generic
    heartwise/deepcoro_clip_generic    HTTP 200/403  exists, needs review
    heartwise/VasoVision               HTTP 401  unauthorized

The host also cannot reach huggingface.co directly (see the mirror comment in
``env.sh``), so even an approved token would need the mirror to carry the repo.

Also unavailable: StenUNet
--------------------------
``data/tasks/AGENT_SPEC.md`` lists a "stenosis detector (StenUNet-style)" as the
tool serving ``arcade_stenosis``. There is no StenUNet checkpoint in
``specialist_models/weights/``; ``stenosis_detection/`` and
``vessel_segmentation/`` under ``specialist_models/`` are both empty directories.

What works instead, today
-------------------------
For ``arcade_stenosis``, CM-UNet via ``benchmark/runners/cm_unet_runner.py`` is a
usable detector: the task's gold label space is the single class ``"stenosis"``,
so a binary segmenter's output is directly comparable (unlike
``arcade_segmentation``, where 25 SYNTAX ids make label-aware F1 ~0). Runs exist
under ``runs/arcade_stenosis/``.

For lesion quantification once an ROI is known, use
``algorithms.tools.diameter_qca.quantify_stenosis``.

To unblock this tool
--------------------
1. Obtain author approval for ``heartwise/deepcoro_clip_generic``, or
2. Mirror the weights to ``specialist_models/deepcoro_clip/weights/``, then
3. Implement ``detect_stenosis`` against the upstream loader in
   ``deepcoro_clip/models/`` and add a smoke test on a real case.

Do not implement this by returning mock predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# Where the weights would live if they existed. Checked at call time so the error
# message reflects reality rather than a stale assumption.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WEIGHTS_DIR = (
    _REPO_ROOT
    / "algorithms"
    / "specialist_models"
    / "deepcoro_clip"
    / "weights"
    / "deepcoro_clip_stenosis"
)

_BLOCKER_MESSAGE = """\
DeepCORO-CLIP stenosis detection is unavailable: no weights on this host.

  expected: {weights_dir}
  found:    {found}

Upstream repositories are gated (HTTP 401/403) and require manual approval from
the authors; see algorithms/specialist_models/deepcoro_clip/ACCESS_PENDING.md.

Working alternatives:
  - arcade_stenosis detection: benchmark/runners/cm_unet_runner.py (CM-UNet;
    the task's gold label space is the single class "stenosis")
  - lesion quantification given an ROI: algorithms.tools.diameter_qca

This tool raises rather than returning an empty result, because an empty
stenosis list is indistinguishable from a correct reading of a normal artery."""


def is_available() -> tuple[bool, str]:
    """Report availability without raising.

    Mirrors the ``check_available`` convention in ``benchmark/runners/`` so an
    agent can probe its tool suite and plan around a missing tool instead of
    failing mid-pipeline.
    """
    if not _WEIGHTS_DIR.is_dir():
        return False, f"weights directory does not exist: {_WEIGHTS_DIR}"
    contents = list(_WEIGHTS_DIR.iterdir())
    if not contents:
        return False, f"weights directory is empty: {_WEIGHTS_DIR}"
    checkpoints = [
        p for p in contents if p.suffix in {".pth", ".pt", ".ckpt", ".safetensors"}
    ]
    if not checkpoints:
        return False, (
            f"no checkpoint files in {_WEIGHTS_DIR} "
            f"(found {len(contents)} non-checkpoint entries)"
        )
    return True, ""


def detect_stenosis(
    image_path: str | Path,
    device: str = "cuda:1",
) -> List[Dict[str, Any]]:
    """Detect stenoses in an XCA frame or cine.

    Not implemented: the DeepCORO-CLIP weights are unavailable on this host.

    Args:
        image_path: XCA frame or cine clip.
        device: Torch device string.

    Returns:
        Would return one dict per detected stenosis with bounding box, segment
        attribution and confidence.

    Raises:
        NotImplementedError: always, until weights are obtained. The message
            names the exact missing paths and the working alternatives.
    """
    available, reason = is_available()
    found = reason if not available else "weights present"
    raise NotImplementedError(
        _BLOCKER_MESSAGE.format(weights_dir=_WEIGHTS_DIR, found=found)
    )


def detection_metadata() -> Dict[str, Any]:
    """Report tool status for the agent's reasoning_trace / capability statement."""
    available, reason = is_available()
    return {
        "tool": "stenosis_detection",
        "backend": "DeepCORO-CLIP",
        "available": available,
        "blocker": reason if not available else None,
        "upstream_access": "gated; HTTP 401/403, needs author approval",
        "alternatives": [
            "benchmark/runners/cm_unet_runner.py for arcade_stenosis",
            "algorithms.tools.diameter_qca for ROI quantification",
        ],
    }
