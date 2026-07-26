"""2D coronary vessel segmentation tool for the Cardiomni agent.

Why CM-UNet and not SAM-VMNet
-----------------------------
PROPOSAL.md §2.6 names SAM-VMNet as the vessel-segmentation tool, but its weights
do not exist on this host: the two files in
``specialist_models/sam_vmnet/pre_trained_weights/`` are 132-byte unresolved
git-lfs pointer texts, not tensors, and the architecture additionally needs
``mamba_ssm`` + ``causal_conv1d`` CUDA extensions that no environment here has.

CM-UNet is the only 2D XCA vessel segmenter on disk with a real checkpoint
(124MB ``CM-UNet_weights.pth``, verified loadable) and a verified inference path.
It is therefore what this tool wraps. When SAM-VMNet weights become available,
add a ``backend=`` parameter rather than replacing this implementation.

Preprocessing comes from the TOML, not from here
------------------------------------------------
Every parameter (pad_to, model_input, normalize, unsharp) is read at call time
via ``load_method_config``, and inference reuses the helpers in
``benchmark/runners/cm_unet_runner.py`` verbatim. This is deliberate: a second,
independent copy of the preprocessing chain is exactly what produced the two
silent-failure bugs this repo has already hit (ARCADE Dice 0.726 -> 0.000 from a
missing unsharp/normalize step; CCA Dice 0.536 -> 0.048 from missing resampling).
One reviewable source of truth, one code path.

Capability boundary (read before using in Stage 2)
--------------------------------------------------
CM-UNet is **binary**: it separates vessel from background and cannot name
segments. It does not and cannot produce SYNTAX segment ids. On
``arcade_segmentation`` a label-aware F1 against 25 SYNTAX classes is ~0.0 by
construction, because no predicted label ever equals a gold id.

Consequences for the agent's four-stage SOP:

- **Stage 2 (systematic segment scan)** requires naming every segment. This tool
  cannot satisfy that requirement. It provides vessel *geometry* only; segment
  naming needs a different mechanism (VLM reasoning over the mask, an atlas, or
  a labelled segmenter that does not currently exist in this repo).
- **Stage 4 (lesion assessment)** can use this mask productively: it is a valid
  input to ``diameter_qca.quantify_stenosis`` for MLD / %DS measurement, and its
  vessel-detection quality is measurable via label-agnostic metrics
  (``mean_matched_iou``, or F1 recomputed ignoring labels).

Usage
-----
    from algorithms.tools import segment_vessels

    mask = segment_vessels("path/to/image.png", device="cuda:1")
    # mask.shape == original frame shape, dtype bool
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo root: .../CardiomniBench-VD
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default variant. ``coronary_cm_unet_native`` uses pad_to=0, which resizes
# 512 -> 256 directly instead of padding to 1536 first. The padded reading
# reproduces upstream literally but leaves anatomy occupying ~85x85 pixels of a
# 256x256 input; see the cm_unet_runner docstring for why both exist.
DEFAULT_METHOD = "coronary_cm_unet_native"

# Model cache keyed by (method_name, device). Loading a 124MB checkpoint per call
# would dominate runtime when the agent segments many frames in one session.
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(method_name: str, device: str, weights_path: Path, out_classes: int):
    """Load and cache the CM-UNet model.

    Uses ``strict=True`` on ``load_state_dict`` for the same reason the runner
    does: a silently-tolerated key mismatch scores a partly-random network, which
    looks like a weak model rather than a loading bug.
    """
    key = (method_name, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch

    from benchmark.runners.cm_unet_runner import _load_architecture

    UNet = _load_architecture()
    model = UNet(out_classes=out_classes)
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    _MODEL_CACHE[key] = model
    return model


def segment_vessels(
    image_path: str | Path,
    device: str = "cuda:1",
    method_name: str = DEFAULT_METHOD,
    threshold: float | None = None,
) -> np.ndarray:
    """Segment coronary vessels in a 2D XCA frame.

    Args:
        image_path: XCA frame (PNG, grayscale or convertible to grayscale).
        device: Torch device string, e.g. ``"cuda:1"`` or ``"cpu"``. Avoid
            ``cuda:0`` on this shared host.
        method_name: Which ``methods/<name>.toml`` to read parameters from.
            Defaults to the pad_to=0 variant.
        threshold: Foreground probability cut. ``None`` uses the TOML's
            ``[decision]`` rule (``argmax`` is equivalent to 0.5 for a 2-class
            head).

    Returns:
        Boolean mask with the **same shape as the input frame**, ``True`` where
        vessel. Padding and resizing are undone before returning.

    Raises:
        FileNotFoundError: image or checkpoint missing.
        RuntimeError: the TOML lacks the ``[preprocess]`` block this tool needs.

    Note:
        The returned mask is binary vessel/background. It carries no segment
        identity; see the module docstring on Stage 2 limitations.
    """
    import torch

    from benchmark.method_config import load_method_config
    from benchmark.runners.cm_unet_runner import _preprocess, _read_frame, _postprocess_to_original

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    config = load_method_config(method_name)
    if not config.image2d_preprocess:
        raise RuntimeError(
            f"{method_name}: vessel_segmentation needs a [preprocess] block with "
            "model_input; without it the preprocessing would have to be guessed, "
            "which is how this repo's two silent scoring bugs happened."
        )

    img_cfg = config.image2d_preprocess
    weights_path = Path(config.weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{method_name}: checkpoint not found at {weights_path}"
        )

    # Resolve the decision threshold from config unless the caller overrides it.
    if threshold is None:
        decision_cfg = config.decision or {}
        rule = str(decision_cfg.get("rule", "argmax"))
        if rule == "argmax":
            threshold = 0.5
        elif rule == "threshold":
            threshold = float(decision_cfg.get("threshold", 0.5))
        else:
            raise ValueError(
                f"{method_name}: unknown decision rule {rule!r} "
                "(expected 'argmax' or 'threshold')"
            )

    frame = _read_frame(image_path)
    tensor_input, geometry = _preprocess(
        frame,
        pad_to=img_cfg.pad_to,
        model_input=img_cfg.model_input,
        normalize=img_cfg.normalize,
        unsharp_radius=img_cfg.unsharp_radius,
        unsharp_amount=img_cfg.unsharp_amount,
    )

    out_classes = int(config.architecture.get("out_classes", 2))
    model = _get_model(method_name, device, weights_path, out_classes)

    with torch.no_grad():
        batch = torch.from_numpy(tensor_input).unsqueeze(0).to(device)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        foreground = probabilities[0, 1].cpu().numpy()

    mask_model_res = (foreground >= threshold).astype(np.uint8)
    # Map back through the resize+pad geometry so the mask aligns with the input.
    mask = _postprocess_to_original(mask_model_res, geometry)
    return mask.astype(bool)


def segmentation_metadata(method_name: str = DEFAULT_METHOD) -> dict:
    """Report what this tool is and what it cannot do.

    Intended for the agent's ``reasoning_trace``: grounding a claim in a tool
    call is only meaningful if the tool's capability boundary travels with it.
    """
    from benchmark.method_config import load_method_config

    config = load_method_config(method_name)
    return {
        "tool": "vessel_segmentation",
        "backend": "CM-UNet",
        "method_config": method_name,
        "weights": str(config.weights_path),
        "label_space": "binary {0=background, 1=vessel}",
        "can_name_segments": False,
        "capability_boundary": (
            "Binary vessel detector. Cannot produce SYNTAX segment ids; "
            "label-aware F1 on arcade_segmentation is ~0 by construction. "
            "Suitable for Stage 4 geometry/quantification, not Stage 2 naming."
        ),
        "sam_vmnet_status": (
            "unavailable: pre_trained_weights are 132-byte git-lfs pointers, "
            "and mamba_ssm/causal_conv1d are not installed"
        ),
    }
