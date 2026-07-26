"""Runner for CM-UNet: 2D X-ray angiography vessel segmentation.

CM-UNet (Challier et al., arXiv:2507.17779, Apache-2.0) is the only genuine 2D
XCA model among the downloaded weights, which makes it the natural first
baseline for the ARCADE tasks. Everything else on disk is 3D CTA or 2D+t cine.

What the checkpoint actually is, verified by loading it
------------------------------------------------------
``CM-UNet_weights.pth`` is a plain ``state_dict`` for the ``UNet`` in the
bundled ``CM-UNet/model.py``: first conv ``down_conv1...0.weight`` of shape
``(64, 1, 3, 3)`` (4-dim, so Conv2d) and ``conv_last.weight`` of shape
``(2, 64, 1, 1)``. One input channel, two output logits, no built-in softmax.
``forward`` calls ``x.unsqueeze(1)`` itself, so it expects ``(B, H, W)``.

Preprocessing follows the upstream code, not a guess
----------------------------------------------------
``dataset.py`` is explicit about inference-time handling:

- ``get_validation_augmentation`` pads to 1536x1536 with ``border_mode=0``
  (zero padding), and does nothing else.
- ``__getitem__`` then resizes to **256x256 with BICUBIC**.
- There is **no intensity normalisation** anywhere; raw ``uint8`` values go in.

That pad-then-shrink order matters. Padding a 512x512 ARCADE frame to 1536 and
then resizing to 256 leaves the actual anatomy occupying only ~85x85 pixels,
with the rest zeros - a 3x resolution loss on vessels that are a few pixels wide
to begin with. Upstream presumably worked with natively ~1536px angiograms where
padding is nearly a no-op.

Both readings are therefore selectable via ``preprocess.pad_to`` in
``methods/<name>.toml``, because which one is right is an empirical question, not
something to settle by assertion:

- ``pad_to = 1536`` reproduces upstream literally.
- ``pad_to = 0`` (or omitted) resizes 512 -> 256 directly, preserving vessel
  scale at the cost of deviating from the published pipeline.

Instances, and the honest limitation
------------------------------------
ARCADE segmentation wants a *labelled* instance list over 25 SYNTAX segment ids.
CM-UNet is **binary**: vessel vs background. It cannot name segments. This runner
therefore does connected-component decomposition and emits every component as an
instance labelled ``"vessel"``.

The consequence is stated plainly rather than buried: on
``arcade_segmentation`` this scores ~0 F1 under label-aware matching, because no
predicted label ever equals a gold SYNTAX id. It is still worth running, for two
reasons. It gives a real vessel-detection upper bound via the label-agnostic
metrics (``mean_matched_iou``, and F1 recomputed ignoring labels), and on
``arcade_stenosis`` the gold label space is the single class ``"stenosis"``, so
the mismatch is one rename away from being a genuine comparison - though a
vessel detector answering a stenosis question is still the wrong question, and
the numbers should be read as "can it find vessel-like structure at all".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.core import Prediction, Task
from benchmark.io_spec import CaseInput

#: Where the vendored upstream architecture lives.
_CM_UNET_SRC = (
    Path(__file__).resolve().parents[2]
    / "algorithms"
    / "specialist_models"
    / "weights"
    / "CM-UNet"
    / "CM-UNet"
)

#: Upstream inference resolution, from dataset.py __getitem__.
_MODEL_INPUT = 256


def _load_architecture():
    """Import the upstream ``UNet`` class from the vendored source.

    The directory is added to ``sys.path`` rather than copied so there is exactly
    one definition of the architecture and it stays traceable to the release.
    """
    if not (_CM_UNET_SRC / "model.py").exists():
        raise FileNotFoundError(
            f"CM-UNet architecture not found at {_CM_UNET_SRC / 'model.py'}"
        )
    if str(_CM_UNET_SRC) not in sys.path:
        sys.path.insert(0, str(_CM_UNET_SRC))
    from model import UNet  # type: ignore[import-not-found]

    return UNet


def _read_frame(image_path: Path) -> np.ndarray:
    """Load a single grayscale angiography frame as a 2D uint8 array."""
    from PIL import Image

    with Image.open(image_path) as handle:
        frame = handle.convert("L")
        return np.asarray(frame, dtype=np.uint8)


def _preprocess(
    frame: np.ndarray,
    pad_to: int,
    model_input: int = _MODEL_INPUT,
    normalize: str = "none",
    unsharp_radius: float = 0.0,
    unsharp_amount: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the upstream pad-then-resize pipeline.

    Returns the model input and the geometry needed to map a prediction back to
    the original frame. No intensity scaling is applied, matching upstream.
    """
    from PIL import Image

    original_h, original_w = frame.shape
    geometry: dict[str, Any] = {
        "original_shape": (int(original_h), int(original_w)),
        "pad_to": int(pad_to),
    }

    if pad_to and (pad_to > original_h or pad_to > original_w):
        # album.PadIfNeeded centres the image, so replicate that placement.
        canvas = np.zeros((pad_to, pad_to), dtype=frame.dtype)
        top = (pad_to - original_h) // 2
        left = (pad_to - original_w) // 2
        canvas[top : top + original_h, left : left + original_w] = frame
        geometry["pad_offset"] = (int(top), int(left))
        geometry["padded_shape"] = (int(pad_to), int(pad_to))
        working = canvas
    else:
        geometry["pad_offset"] = (0, 0)
        geometry["padded_shape"] = (int(original_h), int(original_w))
        working = frame

    with Image.fromarray(working) as img:
        resized = img.resize((model_input, model_input), resample=Image.BICUBIC)
        tensor_input = np.asarray(resized, dtype=np.float32)

    tensor_input = _apply_intensity(
        tensor_input, normalize, unsharp_radius, unsharp_amount
    )

    return tensor_input, geometry


def _apply_intensity(
    array: np.ndarray,
    normalize: str,
    unsharp_radius: float,
    unsharp_amount: float,
) -> np.ndarray:
    """Reproduce upstream offline intensity steps at inference time.

    Upstream data_processing.ipynb runs Unsharper then Intensity_normalizer
    before .npy files the training dataset reads. Skipping this measured
    Dice 0.000 on ARCADE vs 0.726 with it, on identical weights.

    z-score is per-image, matching upstream: it fits mean/std per sample
    rather than over a corpus, so it needs no dataset statistics and
    transfers to a new dataset unchanged.
    """
    working = array.astype(np.float32, copy=True)

    if unsharp_radius > 0 and unsharp_amount != 0:
        from skimage.filters import unsharp_mask

        working = unsharp_mask(
            working,
            radius=float(unsharp_radius),
            amount=float(unsharp_amount),
            preserve_range=True,
        ).astype(np.float32)

    if normalize == "none":
        return working
    if normalize == "divide255":
        return working / 255.0
    if normalize == "minmax":
        low, high = float(working.min()), float(working.max())
        span = high - low
        return (working - low) / span if span else np.zeros_like(working)
    if normalize == "zscore":
        std = float(working.std())
        return (working - float(working.mean())) / (std if std else 1.0)
    raise ValueError(f"unknown normalize mode {normalize!r}")


def _postprocess_to_original(
    mask_256: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    """Undo resize and padding so the mask aligns with the original frame."""
    from PIL import Image

    padded_h, padded_w = geometry["padded_shape"]
    original_h, original_w = geometry["original_shape"]
    top, left = geometry["pad_offset"]

    with Image.fromarray(mask_256.astype(np.uint8)) as img:
        upscaled = img.resize((padded_w, padded_h), resample=Image.NEAREST)
        padded_mask = np.asarray(upscaled, dtype=np.uint8)

    return padded_mask[top : top + original_h, left : left + original_w]


def _mask_to_instances(
    mask: np.ndarray,
    label: str,
    min_area: int,
) -> list[dict[str, Any]]:
    """Decompose a binary mask into per-component instances.

    Components smaller than ``min_area`` pixels are dropped: a segmentation of
    thin vessels always leaves a scatter of few-pixel specks, and emitting them
    would inflate the false-positive count with noise rather than with real
    predictions the method is committing to.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - SciPy is a declared dep
        raise RuntimeError("connected components need SciPy") from exc

    labelled, count = ndimage.label(mask > 0)
    height, width = mask.shape
    instances: list[dict[str, Any]] = []

    for component in range(1, count + 1):
        component_mask = labelled == component
        area = int(component_mask.sum())
        if area < min_area:
            continue

        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        y0, y1 = int(np.argmax(rows)), int(height - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(width - np.argmax(cols[::-1]))

        instances.append(
            {
                "label": label,
                "bbox_xywh_norm": [
                    x0 / width,
                    y0 / height,
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                ],
                "mask": component_mask[y0:y1, x0:x1].astype(np.uint8),
                "area_px": area,
            }
        )

    # Largest first, so a consumer truncating the list keeps the main vessels.
    instances.sort(key=lambda inst: inst["area_px"], reverse=True)
    return instances


def check_available(method: Any) -> tuple[bool, str]:
    """Preflight: the vendored architecture must be importable.

    The checkpoint is a bare ``state_dict``, so it cannot be instantiated without
    the upstream ``UNet`` class. Report unavailable here rather than failing every
    case with the same ImportError.
    """
    if not (_CM_UNET_SRC / "model.py").exists():
        return False, f"cm_unet runner needs vendored model.py at {_CM_UNET_SRC}"
    return True, ""


def predict(
    method: Any,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Segment one XCA frame and return it as an instance list."""
    import torch

    from benchmark.method_config import load_method_config

    config = load_method_config(method.name)
    if not config.image2d_preprocess:
        raise RuntimeError(
            f"{method.name}: cm_unet runner needs [preprocess] with model_input"
        )
    if not config.instances:
        raise RuntimeError(
            f"{method.name}: cm_unet runner needs [instances] for decomposition"
        )

    img_cfg = config.image2d_preprocess
    inst_cfg = config.instances
    decision_cfg = config.decision

    pad_to = img_cfg.pad_to
    model_input = img_cfg.model_input
    min_area = inst_cfg.min_component_pixels
    # Per-task label override: the label a binary mask should carry depends on
    # what the task's gold calls its single class. arcade_stenosis uses "stenosis",
    # and getting this wrong scores 0 for a reason unrelated to the model.
    instance_label = inst_cfg.per_task.get(case.task.value, inst_cfg.label)
    decision_rule = str(decision_cfg.get("rule", "argmax"))

    image_path = case.case_dir / "image.png"
    if not image_path.exists():
        raise FileNotFoundError(f"{case.case_id}: no image.png in {case.case_dir}")

    frame = _read_frame(image_path)
    tensor_input, geometry = _preprocess(
        frame,
        pad_to=pad_to,
        model_input=model_input,
        normalize=img_cfg.normalize,
        unsharp_radius=img_cfg.unsharp_radius,
        unsharp_amount=img_cfg.unsharp_amount,
    )

    UNet = _load_architecture()
    # Constructor arguments come from [architecture] in the TOML, transcribed from
    # the upstream model definition, so a wrong value is wrong in one reviewable
    # place rather than hidden in this runner.
    model = UNet(out_classes=int(config.architecture.get("out_classes", 2)))
    state = torch.load(str(method.weights_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strict=True: a silent key mismatch would score a partly random network.
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    with torch.no_grad():
        batch = torch.from_numpy(tensor_input).unsqueeze(0).to(device)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        foreground = probabilities[0, 1].cpu().numpy()

    # For a 2-class head argmax is exactly p_foreground >= 0.5, verified
    # bit-identical on the CCA task. Both spellings are honoured so the TOML can
    # say what it means, and a threshold other than 0.5 is a real variant.
    if decision_rule == "argmax":
        threshold = 0.5
    elif decision_rule == "threshold":
        threshold = float(decision_cfg.get("threshold", 0.5))
    else:
        raise ValueError(
            f"{method.name}: unknown decision rule {decision_rule!r} "
            f"(expected 'argmax' or 'threshold')"
        )
    mask_256 = (foreground >= threshold).astype(np.uint8)
    mask = _postprocess_to_original(mask_256, geometry)
    instances = _mask_to_instances(mask, label=instance_label, min_area=min_area)

    return Prediction(
        case_id=case.case_id,
        task=case.task,
        instances=instances,
        diagnostics={
            "method": method.name,
            "pad_to": pad_to,
            "model_input": model_input,
            "decision_rule": decision_rule,
            "threshold": threshold,
            "n_instances": len(instances),
            "foreground_fraction": float(mask.mean()),
            "min_component_area": min_area,
            "instance_label": instance_label,
            # Recorded because it is the interpretive crux: a binary detector
            # cannot produce SYNTAX ids, so label-aware F1 on
            # arcade_segmentation is expected to be ~0 by construction.
            "label_space_note": (
                "binary vessel detector; labels are not SYNTAX segment ids"
            ),
        },
    )
