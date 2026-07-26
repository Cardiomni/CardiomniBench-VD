"""MONAI U-Net runner for coronary CTA segmentation.

Pipeline provenance
-------------------
Every transform below is transcribed from the upstream project that released
these checkpoints, https://github.com/noahschuetz/coronary-artery-segmentation:

* ``src/data/transforms.py::get_val_transforms`` - the deterministic chain
  (orient, resample, body crop, HU window, scale, optional TV denoise, optional
  Frangi channel) and its default parameters.
* ``scripts/inference.py::run_inference`` - sliding window with gaussian
  blending under AMP, then ``softmax`` and a probability threshold.

An earlier version of this runner improvised its own preprocessing: a -200..300
HU window, whole-volume z-score, no resampling, and ``argmax`` on the logits.
That scored Dice 0.021 against an upstream-reported 0.788, because CCA volumes
are 0.5 mm isotropic while the network was trained at 1.0 mm, and because
whole-volume statistics are dominated by the ~99.9% of voxels that are not
vessel. Preprocessing mismatch, not domain shift, produced that number.

Concrete parameters are not written here. They come from
``methods/<name>.toml`` via :mod:`benchmark.method_config`, so the values that
must match upstream live in one auditable place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import nibabel as nib
import numpy as np
import torch

if TYPE_CHECKING:
    from benchmark.specialists import CoronaryCTASegmenter

from benchmark.core import Prediction
from benchmark.io_spec import CaseInput, write_mask
from benchmark.method_config import InferenceConfig, PostprocessConfig, PreprocessConfig, load_method_config


def _build_val_transforms(pre: PreprocessConfig):
    """Recreate upstream ``get_val_transforms`` for a single unlabelled image.

    Upstream operates on ``{"image", "label"}`` dictionaries during validation.
    Inference has no label, so the label key is dropped; the image path through
    the chain is otherwise identical, including transform order.
    """
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        ScaleIntensityRanged,
        Spacingd,
    )

    keys = ["image"]
    transforms: list[Any] = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=pre.orientation),
        # Only the image is resampled here, so a single interpolation mode.
        Spacingd(keys=keys, pixdim=pre.pixdim, mode="bilinear"),
    ]

    if pre.body_crop:
        # Threshold on raw HU: this must run before intensity scaling, exactly as
        # upstream orders it, or the -500 HU air cutoff loses its meaning.
        threshold = pre.body_threshold_hu
        transforms.append(
            CropForegroundd(
                keys=keys,
                source_key="image",
                select_fn=lambda x, _t=threshold: x > _t,
                margin=pre.body_margin,
            )
        )

    transforms.append(
        ScaleIntensityRanged(
            keys=keys,
            a_min=pre.window_a_min,
            a_max=pre.window_a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    )

    if pre.normalize_mode in ("zscore", "minmax+zscore"):
        transforms.append(
            NormalizeIntensityd(
                keys=keys,
                nonzero=pre.normalize_nonzero,
                channel_wise=pre.normalize_channel_wise,
            )
        )

    if pre.denoise_tv:
        transforms.append(_TVDenoise(keys=keys, weight=pre.denoise_weight, iters=pre.denoise_iter))

    if pre.vesselness:
        transforms.append(_FrangiChannel(keys=keys, pre=pre))

    transforms.append(EnsureTyped(keys=keys, data_type="tensor"))
    return Compose(transforms)


class _TVDenoise:
    """TV-Chambolle denoise, mirroring upstream ``TVDenoised``.

    Upstream applies this after intensity scaling, so the weight is calibrated
    for data already in [0, 1]. Implemented as a plain callable rather than a
    ``MapTransform`` subclass to avoid depending on MONAI's transform internals.
    """

    def __init__(self, keys: list[str], weight: float, iters: int) -> None:
        self.keys = keys
        self.weight = weight
        self.iters = iters

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from skimage.restoration import denoise_tv_chambolle

        out = dict(data)
        for key in self.keys:
            arr = np.asarray(out[key], dtype=np.float32)
            # Channel-first: denoise each channel volume independently.
            denoised = np.stack(
                [
                    denoise_tv_chambolle(
                        channel, weight=self.weight, max_num_iter=self.iters
                    )
                    for channel in arr
                ]
            ).astype(np.float32)
            out[key] = _preserve_meta(out[key], denoised)
        return out


class _FrangiChannel:
    """Append Frangi vesselness as a second channel (upstream 2-channel input).

    Only ``att_mamba2_unet`` uses this. Sigmas, alpha, beta and gamma come from
    the TOML, which records upstream's documented defaults; ``black_ridges`` is
    False because contrast-filled vessels are bright.
    """

    def __init__(self, keys: list[str], pre: PreprocessConfig) -> None:
        self.keys = keys
        self.pre = pre

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        from skimage.filters import frangi

        out = dict(data)
        pre = self.pre
        for key in self.keys:
            arr = np.asarray(out[key], dtype=np.float32)
            base = arr[0]
            vesselness = frangi(
                base,
                sigmas=pre.vesselness_sigmas,
                alpha=pre.vesselness_alpha,
                beta=pre.vesselness_beta,
                gamma=pre.vesselness_gamma,
                black_ridges=False,
            ).astype(np.float32)
            stacked = (
                np.stack([base, vesselness])
                if pre.vesselness_keep_original
                else vesselness[None]
            )
            out[key] = _preserve_meta(out[key], stacked)
        return out


def _preserve_meta(original: Any, array: np.ndarray) -> Any:
    """Return ``array`` as a MetaTensor when the input carried metadata.

    The affine and spatial history must survive to the end of the chain: the
    predicted mask is inverted back to the original grid using that metadata.
    """
    try:
        from monai.data import MetaTensor
    except ImportError:  # pragma: no cover - MONAI is a hard dependency here
        return torch.from_numpy(array)

    tensor = torch.from_numpy(array)
    if isinstance(original, MetaTensor):
        return MetaTensor(tensor, meta=original.meta, applied_operations=original.applied_operations)
    return tensor


def _resample_to_reference(
    mask: np.ndarray,
    source_affine: np.ndarray,
    reference: nib.Nifti1Image,
) -> np.ndarray:
    """Map a mask from the preprocessed grid back onto the reference grid.

    Preprocessing resamples to 1 mm and crops away air, so the network output
    lives on a different grid than the case volume. Scoring happens on the
    original grid, so the mask has to be carried back with nearest-neighbour
    sampling. Using ``scipy.ndimage.affine_transform`` keeps this exact rather
    than assuming the crop was centred.
    """
    from scipy.ndimage import affine_transform

    ref_shape = reference.shape[:3]
    # Voxel-to-voxel transform: reference index -> source index.
    transform = np.linalg.inv(source_affine) @ reference.affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]

    return affine_transform(
        mask.astype(np.uint8),
        matrix,
        offset=offset,
        output_shape=ref_shape,
        order=0,  # nearest neighbour: labels must not be interpolated
        mode="constant",
        cval=0,
    ).astype(np.uint8)


def _apply_postprocessing(mask: np.ndarray, post: PostprocessConfig) -> np.ndarray:
    """Drop connected components below the configured size."""
    if post.min_component_size <= 0:
        return mask

    from scipy import ndimage as ndi

    if post.connectivity == 6:
        structure = ndi.generate_binary_structure(3, 1)
    elif post.connectivity == 18:
        structure = ndi.generate_binary_structure(3, 2)
    else:
        structure = ndi.generate_binary_structure(3, 3)

    labelled, count = ndi.label(mask, structure=structure)
    if count == 0:
        return mask

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # never keep background as a component
    keep = np.where(sizes >= post.min_component_size)[0]
    return np.isin(labelled, keep).astype(np.uint8)


def _load_network(
    weights_path: Path,
    in_channels: int,
    device: torch.device,
    architecture: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, int]:
    """Build the architecture this checkpoint was trained with and load it strictly.

    Two architectures ship under ``coronary-seg-unet/`` and they are told apart
    by their key layout, not by a config field, so a mislabelled TOML cannot
    cause a wrong-architecture load:

    * MONAI ``UNet`` (``baseline_unet.pth``): keys like
      ``model.0.conv.unit0.conv.weight``. Channel widths and class count are
      read from the conv shapes, so a differently sized sibling still loads.
    * ``AttMamba2UNet`` (``att_mamba2_unet.pth``): keys like ``stem.0.weight``
      plus ``*.att_mamba.mamba.A_log``, built from the vendored upstream module
      in ``algorithms/specialist_models/att_mamba2/``.

    Either way the load is strict: a silent partial load would produce a
    plausible mask from partly random weights, which is worse than a hard
    failure.
    """
    import monai.networks.nets

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}

    if "stem.0.weight" in state and any(".att_mamba.mamba." in k for k in state):
        return _load_att_mamba2(state, weights_path, in_channels, device)

    first_conv = state.get("model.0.conv.unit0.conv.weight")
    if first_conv is None:
        raise ValueError(
            f"{weights_path.name} is not a MONAI UNet state dict "
            "(missing model.0.conv.unit0.conv.weight)"
        )
    if first_conv.shape[1] != in_channels:
        raise ValueError(
            f"{weights_path.name} expects {first_conv.shape[1]} input channel(s) but the "
            f"configured preprocessing produces {in_channels}. Check 'vesselness' in the "
            "method TOML."
        )

    channels = _infer_channels(state)
    out_channels = _infer_out_channels(state)

    # Constructor arguments come from the TOML, not from the tensors. Inferring
    # widths gave a 4-stage network for this 5-stage checkpoint, and norm=
    # "instance" cannot be inferred at all: it changes key names, not shapes,
    # so a wrong choice fails as a confusing set of missing keys.
    if architecture:
        channels = tuple(int(c) for c in architecture["channels"])
        strides = tuple(int(s) for s in architecture["strides"])
        out_channels = int(architecture.get("out_channels", out_channels))
        num_res_units = int(architecture.get("num_res_units", 2))
        norm = str(architecture.get("norm", "instance"))
        spatial_dims = int(architecture.get("spatial_dims", 3))
    else:
        raise ValueError(
            f"{weights_path.name} needs an [architecture] table in its method TOML "
            "(channels, strides, num_res_units, norm) transcribed from the upstream "
            "model definition; these are not recoverable from the state dict alone."
        )

    network = monai.networks.nets.UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        norm=norm,
    )
    network.load_state_dict(state, strict=True)
    return network.to(device).eval(), out_channels


def _load_att_mamba2(
    state: dict[str, torch.Tensor],
    weights_path: Path,
    in_channels: int,
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    """Build ``AttMamba2UNet`` sized from the checkpoint and load it strictly.

    Widths come from the stem and the three downsample convs, so the encoder
    progression is read off the tensors instead of trusted from the TOML.
    """
    from algorithms.specialist_models.att_mamba2 import MAMBA2_IMPL, AttMamba2UNet

    stem = state["stem.0.weight"]
    ckpt_in = int(stem.shape[1])
    if ckpt_in != in_channels:
        raise ValueError(
            f"{weights_path.name} expects {ckpt_in} input channel(s) but the configured "
            f"preprocessing produces {in_channels}. Check 'vesselness' in the method TOML "
            "(this checkpoint needs CT + Frangi, so vesselness must be true)."
        )

    features = [int(stem.shape[0])]
    for i in (1, 2, 3):
        key = f"down{i}.0.weight"
        if key not in state:
            raise ValueError(f"{weights_path.name} is missing {key}; unexpected layout")
        features.append(int(state[key].shape[0]))

    out_channels = int(state["out.conv.conv.weight"].shape[0])

    network = AttMamba2UNet(
        in_channels=ckpt_in,
        out_channels=out_channels,
        features=tuple(features),
        patch_size=(96, 96, 96),
    )
    network.load_state_dict(state, strict=True)

    if MAMBA2_IMPL != "mamba_ssm":
        # Worth surfacing: the scan is a sequential PyTorch loop here, so a full
        # volume takes minutes rather than seconds.
        print(
            f"[monai_unet_runner] {weights_path.name}: Mamba2 backend='{MAMBA2_IMPL}' "
            "(mamba-ssm unavailable, using kernel-free reimplementation)",
            flush=True,
        )
    return network.to(device).eval(), out_channels


def _infer_channels(state: dict[str, torch.Tensor]) -> tuple[int, ...]:
    """Recover the encoder widths from the checkpoint's conv shapes."""
    widths = [int(state["model.0.conv.unit0.conv.weight"].shape[0])]
    prefix = "model.1.submodule."
    while True:
        key = f"{prefix}0.conv.unit0.conv.weight"
        if key not in state:
            break
        widths.append(int(state[key].shape[0]))
        prefix += "1.submodule."
    return tuple(widths)


def _infer_out_channels(state: dict[str, torch.Tensor]) -> int:
    """Read the class count from the last decoder convolution."""
    candidates = [k for k in state if k.endswith("conv.weight") and ".conv." in k]
    # The output head is the deepest 'model.2...' path; fall back to 2 classes.
    heads = sorted(k for k in candidates if k.startswith("model.2"))
    if heads:
        return int(state[heads[-1]].shape[0])
    return 2


def check_available(method: CoronaryCTASegmenter) -> tuple[bool, str]:
    """Preflight for MONAI UNet checkpoints.

    Architecture compatibility is a property of the weights, so it is checked
    here rather than in the method registry: this runner builds either a MONAI
    ``UNet`` or the vendored ``AttMamba2UNet``, both of which take 1 or 2 input
    channels. Anything else means the wrong runner was declared.
    """
    if method.in_channels not in (1, 2):
        return (
            False,
            f"monai_unet runner cannot serve {method.in_channels}-channel weights",
        )
    return True, ""


def predict(
    method: CoronaryCTASegmenter,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Segment one CCA volume using the method's TOML-declared pipeline."""
    from monai.inferers import sliding_window_inference

    from benchmark.method_config import load_method_config

    config = load_method_config(method.name)
    pre = config.preprocess
    inf: InferenceConfig = config.inference
    post: PostprocessConfig = config.postprocess
    if pre is None:
        raise ValueError(
            f"methods/{method.name}.toml has no [preprocess] table; this runner "
            "needs an explicit transform chain."
        )

    volume_path = case.require_volume()
    reference = nib.load(str(volume_path))
    torch_device = torch.device(device)

    transforms = _build_val_transforms(pre)
    prepared = transforms({"image": str(volume_path)})["image"]

    network, out_channels = _load_network(
        Path(method.weights_path), pre.in_channels, torch_device, config.architecture
    )

    image = prepared.unsqueeze(0).to(torch_device).float()
    use_amp = inf.amp and torch_device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=inf.roi_size,
                    sw_batch_size=inf.sw_batch_size,
                    predictor=network,
                    overlap=inf.overlap,
                    mode=inf.mode,
                )
        else:
            logits = sliding_window_inference(
                inputs=image,
                roi_size=inf.roi_size,
                sw_batch_size=inf.sw_batch_size,
                predictor=network,
                overlap=inf.overlap,
                mode=inf.mode,
            )

        logits = logits.float()
        if inf.decision_rule == "threshold":
            # Upstream inference.py: softmax over classes, then threshold the
            # foreground channel. Not argmax - the threshold is tunable and the
            # two differ whenever foreground probability sits below 0.5 while
            # still exceeding every other single class.
            probabilities = torch.softmax(logits, dim=1)[0, inf.foreground_class]
            probability_map = probabilities.cpu().numpy()
            mask_prepared = (probability_map >= inf.threshold).astype(np.uint8)
        elif inf.decision_rule == "argmax":
            predicted = torch.argmax(logits, dim=1)[0]
            probability_map = torch.softmax(logits, dim=1)[0, inf.foreground_class].cpu().numpy()
            mask_prepared = (predicted == inf.foreground_class).cpu().numpy().astype(np.uint8)
        else:
            raise ValueError(f"unsupported decision_rule: {inf.decision_rule}")

    source_affine = np.asarray(prepared.meta["affine"], dtype=np.float64)
    mask = _resample_to_reference(mask_prepared, source_affine, reference)
    mask = _apply_postprocessing(mask, post)

    # write_mask copies the affine and header from the reference volume itself,
    # so it takes the path rather than a bare affine: a mask written with only a
    # matching shape but a fresh affine scores near zero on every overlap metric.
    mask = mask.astype(post.output_dtype, copy=False)
    mask_path = write_mask(mask, output_dir, volume_path)

    return Prediction(
        case_id=case.case_id,
        task=case.task,
        mask_path=mask_path,
        diagnostics={
            "decision_rule": inf.decision_rule,
            "threshold": inf.threshold if inf.decision_rule == "threshold" else None,
            "out_channels": out_channels,
            "in_channels": pre.in_channels,
            "pixdim": list(pre.pixdim),
            "hu_window": [pre.window_a_min, pre.window_a_max],
            "normalize_mode": pre.normalize_mode,
            "denoise_tv": pre.denoise_tv,
            "prepared_shape": list(mask_prepared.shape),
            "reference_shape": list(reference.shape[:3]),
            "prob_max": float(probability_map.max()),
            "prob_mean": float(probability_map.mean()),
            "foreground_fraction": float(mask.mean()),
        },
    )
