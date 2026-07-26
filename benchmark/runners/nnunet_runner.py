"""
nnU-Net runner: plans-driven inference for nnU-Net / UMambaBot checkpoints.

Why this exists separately from monai_unet_runner
-------------------------------------------------
These checkpoints are ``PlainConvUNet`` from ``dynamic_network_architectures``
(292 tensors, ``encoder.stages.*`` keys), not MONAI ``UNet``. Nothing about them
is compatible: architecture, patch shape, spacing, and class count all differ.
Verified by loading the checkpoint into the plans-derived network, which gives
0 missing / 0 unexpected keys and 30.8M parameters.

Everything here is read from the sidecar ``plans.json`` + ``dataset.json`` rather
than hardcoded, which is how nnU-Net is designed to be reproduced. The two
supported checkpoints share one pipeline and differ only in encoder.

Three details that are easy to get wrong, and are wrong silently
---------------------------------------------------------------
1. **3 classes, binary gold.** The head emits background / lca / rca
   (``decoder.seg_layers.0.weight`` is ``(3, 320, 1, 1, 1)``), while our gold is
   ``binary {0=background, 1=coronary}``. Foreground is the *union* of lca and
   rca: ``p(lca) + p(rca) > p(background)``. Using ``argmax(...) >= 1`` is
   incorrect because it treats the two vessel classes as mutually exclusive,
   dropping vessel wherever they split probability.
2. **Spacing.** Cases are isotropic 0.5mm; plans want ``[0.5, 0.35, 0.35]``.
   Skipping the resample feeds the network a body it never saw in training.
3. **Deep supervision.** The network is built with ``deep_supervision=True`` to
   match the checkpoint, so the forward pass returns a *list* of logits at
   descending resolutions. Index 0 is the full-resolution head; the rest are
   training aids and must be discarded.

Prediction geometry is restored to the input grid before scoring, because Dice is
computed against gold on the original grid.

No operating threshold
----------------------
Decision is the network's native argmax. The 20 CCA cases are the test set and
there is no separate tuning split, so a threshold selected against them would be
a fitted hyperparameter reported as a zero-shot result.
"""

from __future__ import annotations

import json
import pydoc
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch

if TYPE_CHECKING:
    from benchmark.specialists import CoronaryCTASegmenter

from benchmark.core import Prediction
from benchmark.io_spec import CaseInput, write_mask

#: nnU-Net's CTNormalization clips to the training-set foreground percentiles
#: recorded in plans, then z-scores with the stored mean/std. Both come from
#: ``foreground_intensity_properties_per_channel``; nothing is invented here.
_CONFIG = "3d_fullres"


def _fit_shape(arr: np.ndarray, target: tuple[int, ...]) -> np.ndarray:
    """Force ``arr`` to exactly ``target`` shape.

    ``scipy.ndimage.zoom`` computes output size by rounding ``input * factor``,
    which can land one voxel off the intended shape. Cropping or zero-padding the
    edge keeps the arrays alignable for a voxelwise comparison.
    """
    if arr.shape == tuple(target):
        return arr
    fitted = np.zeros(target, dtype=np.float32)
    region = tuple(slice(0, min(a, t)) for a, t in zip(arr.shape, target))
    fitted[region] = arr[region]
    return fitted


def _arch_from_legacy_plans(cfg: dict) -> dict:
    """Translate a pre-v2.2 nnU-Net plans config into ``architecture``-style kwargs.

    Older plans (as shipped with the UMambaBot checkpoint) flatten the network
    description across the configuration dict instead of nesting it under an
    ``architecture`` key::

        UNet_class_name: PlainConvUNet      -> network_class_name
        UNet_base_num_features: 32          -> features_per_stage[0]
        unet_max_num_features: 320          -> cap on features_per_stage
        pool_op_kernel_sizes: [...]         -> strides
        conv_kernel_sizes: [...]            -> kernel_sizes
        n_conv_per_stage_encoder: [...]     -> n_conv_per_stage
        n_conv_per_stage_decoder: [...]     -> n_conv_per_stage_decoder

    ``features_per_stage`` doubles per stage from the base width and saturates at
    ``unet_max_num_features``, which is how nnU-Net derived it before the field
    was serialized explicitly.
    """
    n_stages = len(cfg["pool_op_kernel_sizes"])
    base = cfg["UNet_base_num_features"]
    cap = cfg.get("unet_max_num_features", base * 2 ** (n_stages - 1))
    features = [min(base * 2**i, cap) for i in range(n_stages)]

    class_name = cfg["UNet_class_name"]
    module = {
        "PlainConvUNet": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "ResidualEncoderUNet": (
            "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"
        ),
    }.get(class_name)
    if module is None:
        raise ValueError(f"unsupported legacy UNet_class_name: {class_name}")

    return {
        "network_class_name": module,
        "arch_kwargs": {
            "n_stages": n_stages,
            "features_per_stage": features,
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": cfg["conv_kernel_sizes"],
            "strides": cfg["pool_op_kernel_sizes"],
            "n_conv_per_stage": cfg["n_conv_per_stage_encoder"],
            "n_conv_per_stage_decoder": cfg["n_conv_per_stage_decoder"],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        "_kw_requires_import": [
            "conv_op",
            "norm_op",
            "dropout_op",
            "nonlin",
        ],
    }


def _build_network(weights_dir: Path, num_input_channels: int, num_classes: int):
    """Instantiate the network exactly as serialized in plans.json.

    Handles both plans schemas: the current one nests everything under
    ``architecture``, while older exports flatten it across the config dict.
    """
    plans = json.loads((weights_dir / "plans.json").read_text())
    cfg = plans["configurations"][_CONFIG]
    arch = cfg.get("architecture") or _arch_from_legacy_plans(cfg)

    kwargs = dict(arch["arch_kwargs"])
    # plans.json stores class paths as strings for these keys.
    for key in arch.get("_kw_requires_import", ()):
        kwargs[key] = pydoc.locate(kwargs[key]) if kwargs[key] else None

    net_cls = pydoc.locate(arch["network_class_name"])
    if net_cls is None:
        raise ImportError(f"cannot locate {arch['network_class_name']}")

    net = net_cls(
        input_channels=num_input_channels,
        num_classes=num_classes,
        deep_supervision=True,
        **kwargs,
    )
    return net, plans, cfg, arch


def _ct_normalize(data: np.ndarray, plans: dict) -> np.ndarray:
    """Apply nnU-Net CTNormalization using the statistics stored in plans."""
    props = plans.get("foreground_intensity_properties_per_channel", {}).get("0", {})
    lower = props.get("percentile_00_5")
    upper = props.get("percentile_99_5")
    mean = props.get("mean")
    std = props.get("std")

    if None in (lower, upper, mean, std):
        # Missing stats would mean guessing the intensity scale. Refuse instead:
        # a wrong window silently degrades the mask rather than erroring.
        raise ValueError(
            "plans.json lacks foreground_intensity_properties_per_channel['0']; "
            "cannot apply CTNormalization faithfully"
        )

    clipped = np.clip(data, lower, upper)
    return ((clipped - mean) / (std + 1e-8)).astype(np.float32)


def check_available(method: CoronaryCTASegmenter) -> tuple[bool, str]:
    """Preflight for nnU-Net checkpoints.

    nnU-Net is plans-driven: target spacing, patch size, normalisation scheme and
    class count all come from ``plans.json`` beside the checkpoint, with the label
    names in ``dataset.json``. Without them the checkpoint cannot be configured
    at all, so this reports unavailable rather than letting every case fail with
    the same missing-file error.
    """
    for sidecar in ("plans.json", "dataset.json"):
        if not (method.weights_path.parent / sidecar).exists():
            return False, f"nnunet runner needs {sidecar} beside the checkpoint"
    return True, ""


def predict(
    method: CoronaryCTASegmenter,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Run plans-driven nnU-Net inference and return a Prediction."""
    import monai.inferers
    from scipy.ndimage import zoom

    volume_path = case.require_volume()
    nii = nib.load(str(volume_path))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    original_shape = data.shape
    original_spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])

    weights_dir = method.weights_path.parent
    dataset = json.loads((weights_dir / "dataset.json").read_text())
    num_classes = len(dataset["labels"])
    net, plans, cfg, arch = _build_network(
        weights_dir,
        num_input_channels=len(dataset["channel_names"]),
        num_classes=num_classes,
    )

    checkpoint = torch.load(
        str(method.weights_path), map_location="cpu", weights_only=False
    )
    state = checkpoint.get("network_weights", checkpoint)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        # The plans and the checkpoint disagree. Scoring a partially initialised
        # network would produce a number that looks like a result.
        raise RuntimeError(
            f"{method.name}: checkpoint/plans mismatch "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys)"
        )
    net = net.to(device).eval()

    normalized = _ct_normalize(data, plans)

    # Resample onto the spacing the network was trained at.
    target_spacing = tuple(float(s) for s in cfg["spacing"])
    resampled = normalized
    if not np.allclose(original_spacing, target_spacing, atol=1e-3):
        factors = [o / t for o, t in zip(original_spacing, target_spacing)]
        resampled = zoom(normalized, factors, order=3)

    volume_tensor = torch.from_numpy(resampled).float().unsqueeze(0).unsqueeze(0)

    torch_device = torch.device(device)

    # Deep supervision makes forward() return a list of logits, one per decoder
    # scale. MONAI's stitcher needs a single tensor, so the auxiliary heads are
    # dropped inside the predictor rather than after the inferer.
    def _full_resolution(x: torch.Tensor) -> torch.Tensor:
        out = net(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    # The resampled volume is much larger than the input (0.5mm isotropic ->
    # [0.5, 0.35, 0.35] roughly doubles the voxel count), and the head is
    # 3-class, so a GPU-resident accumulator costs ~10GB before overlap
    # buffers. Stitching on CPU keeps peak GPU memory at patch scale.
    with torch.no_grad():
        logits = monai.inferers.sliding_window_inference(
            inputs=volume_tensor.to(torch_device),
            roi_size=tuple(cfg["patch_size"]),
            sw_batch_size=method.sw_batch_size,
            predictor=_full_resolution,
            overlap=0.5,
            device=torch.device("cpu"),
            sw_device=torch_device,
        )

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # Foreground is the union of the two vessel classes, since gold is binary.
    vessel_prob = probs[1:].sum(axis=0) if num_classes > 2 else probs[1]
    background_prob = probs[0]

    # Resample to the input grid before deciding. Coronary vessels are only a few
    # voxels across, so nearest-neighbor resampling of an already-binarized mask
    # drops thin branches outright; interpolating probabilities first does not.
    if vessel_prob.shape != original_shape:
        factors = [o / m for o, m in zip(original_shape, vessel_prob.shape)]
        vessel_prob = _fit_shape(zoom(vessel_prob, factors, order=3), original_shape)
        background_prob = _fit_shape(
            zoom(background_prob, factors, order=3), original_shape
        )

    # Native argmax, no operating threshold. The 20 CCA cases are the test set
    # and there is no tuning split, so any threshold selected against them would
    # be a fitted hyperparameter reported as zero-shot.
    mask = (vessel_prob > background_prob).astype(np.uint8)

    mask_path = write_mask(mask, output_dir, volume_path)

    diagnostics = {
        "config": _CONFIG,
        "architecture": arch["network_class_name"].rsplit(".", 1)[-1],
        "plans_schema": "architecture" if "architecture" in cfg else "legacy_flat",
        "num_classes": num_classes,
        "class_merge": "lca+rca union" if num_classes > 2 else "binary",
        "decision_rule": "argmax (vessel union vs background)",
        "target_spacing": list(target_spacing),
        "original_spacing": list(original_spacing),
        "resampled_shape": list(resampled.shape),
        "prob_max": float(vessel_prob.max()),
        "prob_mean": float(vessel_prob.mean()),
        "foreground_fraction": float(mask.mean()),
    }

    return Prediction(
        case_id=case.case_id,
        task=case.task,
        mask_path=mask_path,
        diagnostics=diagnostics,
    )
