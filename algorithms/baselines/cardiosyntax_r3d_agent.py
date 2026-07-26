#!/usr/bin/env python
"""CardioSyntax v2 (R3D-18 + LSTM) baseline agent for task ``cardiosyntax_scoring``.

This is the *official* baseline shipped with the CardioSYNTAX dataset
(HuggingFace ``MesserMMP/coronary-syntax-prediction``): a R3D-18 backbone scores
each projection independently, and an LSTM head aggregates the per-view features
into a study-level SYNTAX contribution. Left and right coronary arteries use
separate models, and predictions are averaged over a 5-fold ensemble with an
optional per-fold ``a*x+b`` calibration.

Why a dedicated adapter is needed
---------------------------------
The upstream inference code (``full_model/rnn_dataset.py``) reads each view with
``pydicom.dcmread(path).pixel_array``. Our benchmark cases store the *same*
array as ``.npy`` (public CardioSYNTAX ships no DICOM), shaped
``[frames, 512, 512]`` uint8. Everything downstream of that single read is
format-agnostic, so this agent reimplements just the loading + preprocessing and
reuses the upstream model definition verbatim.

Agent contract (see docs/PIPELINE_API.md)
-----------------------------------------
Reads ``<case>/task.yaml``, writes ``<out>/prediction.json`` with
``syntax_score`` / ``syntax_left`` / ``syntax_right`` / ``dominance``.

The gold ``dominance`` label is absent for 49/60 cases upstream, so this model
does not predict it (it has no dominance head); the field is emitted as ``None``
and should be excluded from scoring rather than counted as a miss.

Usage
-----
    python cardiosyntax_r3d_agent.py --case-dir <case> --output-dir <out> \
        [--weights-dir DIR] [--device cuda:4] [--folds 0,1,2,3,4] [--scaling]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Repo root: .../CardiomniBench-VD
REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = (
    REPO_ROOT
    / "algorithms"
    / "specialist_models"
    / "weights"
    / "coronary-syntax-prediction"
)

# Upstream preprocessing constants (full_model/rnn_dataset.py + inference/rnn_apply.py)
FRAMES_PER_CLIP = 32
VIDEO_SIZE = (256, 256)
# ImageNet statistics: upstream uses these for both training and inference
# (rnn_train.py line 314-315, inference/rnn_apply.py line 268-269)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def _fail(message: str, *, code: int = 1) -> None:
    print(f"[cardiosyntax_r3d] ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


# --------------------------------------------------------------------------
# Preprocessing: mirrors rnn_dataset.SyntaxDataset.__getitem__ for .npy input
# --------------------------------------------------------------------------
def load_view(path: Path, length: int = FRAMES_PER_CLIP) -> "np.ndarray":
    """Load one projection as a ``[length, H, W]`` uint8 clip.

    Reproduces the upstream DICOM path: channel-last fix, uint16 rescale,
    repeat-pad short clips, then take the centre ``length`` frames
    (deterministic, matching ``train=False``).
    """
    video = np.load(path)

    if video.ndim != 3:
        raise ValueError(f"expected a 3D [frames,H,W] clip, got {video.shape} in {path}")

    # Upstream guard for channel-last arrays.
    if video.shape[0] > 128 and video.shape[-1] <= 128:
        video = np.moveaxis(video, -1, 0)

    if video.dtype == np.uint16:
        vmax = int(video.max())
        if vmax <= 0:
            raise ValueError(f"invalid vmax={vmax} in {path}")
        video = (video.astype(np.float32) * (255.0 / vmax)).clip(0, 255).astype(np.uint8)
    else:
        video = video.astype(np.uint8)

    # Repeat-pad clips shorter than the window.
    while video.shape[0] < length:
        video = np.concatenate([video, video], axis=0)

    begin = (video.shape[0] - length) // 2
    return video[begin : begin + length]


def clip_to_tensor(clip: "np.ndarray") -> "torch.Tensor":
    """``[T,H,W]`` uint8 -> normalised ``[C,T,H,W]`` float tensor."""
    import torch
    import torch.nn.functional as F

    # Grey -> 3 channels, as upstream does before ToTensorVideo.
    stacked = np.stack([clip, clip, clip], axis=-1)  # [T,H,W,C]
    tensor = torch.from_numpy(stacked)

    # ToTensorVideo: [T,H,W,C] uint8 -> [C,T,H,W] float in [0,1]
    tensor = tensor.permute(3, 0, 1, 2).to(torch.float32) / 255.0

    if tensor.shape[-2:] != VIDEO_SIZE:
        tensor = F.interpolate(
            tensor, size=VIDEO_SIZE, mode="bilinear", align_corners=False, antialias=True
        )

    mean = torch.tensor(NORM_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1, 1)
    return (tensor - mean) / std


# --------------------------------------------------------------------------
# Task spec parsing
# --------------------------------------------------------------------------
def split_views_by_artery(task: dict) -> dict[str, list[Path]]:
    """Group view paths into ``left`` / ``right`` using the task spec.

    Each view in ``input.views`` carries an ``artery`` field (LCA/RCA) produced
    during conversion, which is exactly the split the upstream models expect.
    """
    case_dir = Path(task["__case_dir__"])
    grouped: dict[str, list[Path]] = {"left": [], "right": []}

    for view in task.get("input", {}).get("views", []) or []:
        rel = view.get("file_path")
        if not rel:
            continue
        path = (case_dir / rel).resolve()
        if not path.exists():
            print(f"[cardiosyntax_r3d] WARN: missing view {path}", file=sys.stderr)
            continue

        artery = str(view.get("artery", "")).strip().upper()
        if artery in ("LCA", "LEFT", "L"):
            grouped["left"].append(path)
        elif artery in ("RCA", "RIGHT", "R"):
            grouped["right"].append(path)
        else:
            print(
                f"[cardiosyntax_r3d] WARN: unknown artery {artery!r}, skipping {path.name}",
                file=sys.stderr,
            )

    return grouped


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def discover_checkpoints(weights_dir: Path) -> dict[str, dict[int, Path]]:
    """Map ``{'left'|'right': {fold: path}}`` from full-model checkpoints.

    Filenames look like ``LeftBinSyntax_R3D_fold02_lstm_mean_post_best.pt``.

    Git-LFS pointer files (~134 bytes of ``version https://git-lfs...``) are
    skipped: the upstream repo ships both ``full_model/`` (real tensors) and
    ``full_model_weights/`` (pointers only), and loading a pointer fails with
    ``invalid load key, 'v'``.
    """
    found: dict[str, dict[int, Path]] = {"left": {}, "right": {}}
    skipped_pointers = 0
    for path in sorted(weights_dir.glob("*.pt")):
        name = path.name
        side = "left" if name.lower().startswith("left") else (
            "right" if name.lower().startswith("right") else None
        )
        if side is None:
            continue
        match = re.search(r"fold0*(\d+)", name, re.IGNORECASE)
        if not match:
            continue
        if path.stat().st_size < 100_000:  # LFS pointer, not a checkpoint
            skipped_pointers += 1
            continue
        found[side][int(match.group(1))] = path

    if skipped_pointers:
        print(
            f"[cardiosyntax_r3d] WARN: skipped {skipped_pointers} Git-LFS pointer file(s) "
            f"in {weights_dir} (run 'git lfs pull' there, or point --weights-dir at the "
            f"directory holding real tensors)",
            file=sys.stderr,
        )
    return found


#: Loaded models, keyed by (checkpoint path, device). Building a model reads a
#: ~250MB checkpoint and rebuilds an R3D-18 backbone; without this cache an
#: ensemble run reloads the same 10 checkpoints for every case, which dominates
#: total runtime and changes no result.
_MODEL_CACHE: dict[tuple[str, str], "torch.nn.Module"] = {}


def build_model(checkpoint: Path, device: "torch.device") -> "torch.nn.Module":
    """Instantiate the upstream LSTM-head module and load one checkpoint.

    ``SyntaxLightningModule.__init__`` requests pretrained torchvision weights,
    which would need network access; our checkpoint fully overwrites them, so we
    neutralise that download first to keep the agent offline-safe.

    Results are cached: repeated calls for the same (checkpoint, device) return
    the already-loaded model rather than reloading from disk.
    """
    import torch
    import torchvision.models.video as tvmv

    key = (str(checkpoint), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))

    original_r3d_18 = tvmv.r3d_18
    tvmv.r3d_18 = lambda *a, **kw: original_r3d_18(weights=None)  # offline-safe
    try:
        from full_model.rnn_model import SyntaxLightningModule

        model = SyntaxLightningModule(
            num_classes=2,
            lr=0.0,
            variant="lstm_mean",
            pt_weights_format=True,
        )
    finally:
        tvmv.r3d_18 = original_r3d_18

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[cardiosyntax_r3d] WARN: {len(missing)} missing keys in {checkpoint.name}",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"[cardiosyntax_r3d] WARN: {len(unexpected)} unexpected keys in {checkpoint.name}",
            file=sys.stderr,
        )

    model = model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def load_scaling(weights_root: Path) -> dict[str, dict[str, float]]:
    path = weights_root / "scaling_coeffs" / "scaling_coeffs.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[cardiosyntax_r3d] WARN: unreadable scaling coeffs: {exc}", file=sys.stderr)
        return {}


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def predict_side(
    view_paths: list[Path],
    checkpoints: dict[int, Path],
    folds: list[int],
    device: "torch.device",
) -> dict[int, float]:
    """Return per-fold SYNTAX contribution for one artery (before scaling).

    Scaling is applied **after** summing left + right, not before.
    """
    import torch

    if not view_paths:
        return {}

    clips = []
    for path in view_paths:
        try:
            clips.append(clip_to_tensor(load_view(path)))
        except Exception as exc:
            print(f"[cardiosyntax_r3d] WARN: skipping {path.name}: {exc}", file=sys.stderr)
    if not clips:
        return {}

    # [1, num_views, C, T, H, W] — the LSTM consumes the view axis.
    batch = torch.stack(clips, dim=0).unsqueeze(0).to(device)

    fold_preds: dict[int, float] = {}
    for fold in folds:
        checkpoint = checkpoints.get(fold)
        if checkpoint is None:
            continue
        try:
            model = build_model(checkpoint, device)
            with torch.no_grad():
                output = model(batch)
            # The head is two-way (see upstream ``predict_step``): index 0 is a
            # classification logit for SYNTAX>22, index 1 is the regression
            # value. ``proj_size=2`` in the LSTM enforces this shape.
            #
            # Training regresses ``log1p(score)`` (rnn_dataset.py line 157), so
            # invert with ``expm1`` to get back to the SYNTAX scale.
            flat = output.flatten()
            raw = float(flat[1].item() if flat.numel() >= 2 else flat[0].item())
            value = float(np.expm1(raw))
            fold_preds[fold] = value
            # The model stays resident on purpose: build_model caches it, and the
            # next case reuses it. Ten R3D-18 models are a few GB, which fits
            # alongside the activations for one case.
        except Exception as exc:
            print(
                f"[cardiosyntax_r3d] WARN: fold {fold} failed ({exc})",
                file=sys.stderr,
            )

    return fold_preds


def aggregate_folds(
    left_folds: dict[int, float],
    right_folds: dict[int, float],
    folds: list[int],
    scaling: dict[str, dict[str, float]],
    use_scaling: bool,
) -> tuple[float | None, float | None, float | None]:
    """Combine per-fold side predictions into ``(total, left, right)``.

    Mirrors upstream ``inference/rnn_apply.py``: for each fold, sum the left and
    right contributions, apply that fold's ``a*x+b`` calibration to the **sum**,
    then average over folds and clamp negatives to zero. Per-side values are
    reported uncalibrated, since the coefficients are fitted on the total only.

    Inputs are already back on the SYNTAX scale (``expm1`` applied per fold).
    """
    totals: list[float] = []
    for fold in folds:
        left_value = left_folds.get(fold)
        right_value = right_folds.get(fold)
        if left_value is None and right_value is None:
            continue
        combined = (left_value or 0.0) + (right_value or 0.0)
        if use_scaling:
            coeffs = scaling.get(f"fold{fold}")
            if coeffs:
                combined = coeffs.get("a", 1.0) * combined + coeffs.get("b", 0.0)
        totals.append(combined)

    total = max(0.0, float(np.mean(totals))) if totals else None
    left = float(np.mean(list(left_folds.values()))) if left_folds else None
    right = float(np.mean(list(right_folds.values()))) if right_folds else None
    return total, left, right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=UPSTREAM_DIR / "full_model",
        help=(
            "Directory of full-model .pt checkpoints. Defaults to 'full_model/', "
            "which holds the real tensors; 'full_model_weights/' contains only "
            "Git-LFS pointers in this checkout."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:4",
        help="Torch device; GPU 4 is typically the idle card on this host.",
    )
    parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Comma-separated fold ids to ensemble.",
    )
    parser.add_argument(
        "--scaling",
        action="store_true",
        help=(
            "Apply the per-fold a*x+b calibration. OFF by default: the upstream "
            "coefficients (a=1.11-1.65) are fitted to maximise mean_recall for the "
            "SYNTAX>22 *binary* task, so they inflate scores and worsen regression "
            "error. Measured on 3 cases: MAE 1.45 uncalibrated vs 7.97 calibrated. "
            "Enable only when reporting the >22 classification metric."
        ),
    )
    args = parser.parse_args()

    task_path = args.case_dir / "task.yaml"
    if not task_path.exists():
        _fail(f"task.yaml not found in {args.case_dir}")

    task = yaml.safe_load(task_path.read_text())
    task["__case_dir__"] = str(args.case_dir.resolve())
    case_id = task.get("case_id", args.case_dir.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "prediction.json"

    def write(payload: dict[str, Any]) -> None:
        prediction_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    try:
        import torch
    except ImportError:
        _fail(
            "PyTorch is required. Use an env that has it, e.g. "
            "/opt/anaconda3/envs/gkp-gsa/bin/python"
        )

    if not args.weights_dir.exists():
        _fail(f"weights dir not found: {args.weights_dir}")

    checkpoints = discover_checkpoints(args.weights_dir)
    if not checkpoints["left"] and not checkpoints["right"]:
        _fail(f"no Left*/Right* fold checkpoints under {args.weights_dir}")

    folds = [int(f) for f in str(args.folds).split(",") if str(f).strip() != ""]
    device = torch.device(
        args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    )
    scaling = load_scaling(UPSTREAM_DIR)
    use_scaling = bool(args.scaling)

    grouped = split_views_by_artery(task)
    print(
        f"[cardiosyntax_r3d] {case_id}: "
        f"{len(grouped['left'])} left / {len(grouped['right'])} right views, "
        f"device={device}, folds={folds}"
    )

    try:
        left_folds = predict_side(grouped["left"], checkpoints["left"], folds, device)
        right_folds = predict_side(grouped["right"], checkpoints["right"], folds, device)
    except Exception:
        traceback.print_exc()
        write(
            {
                "case_id": case_id,
                "task_type": "cardiosyntax_scoring",
                "agent": "cardiosyntax_r3d_lstm",
                "error": "inference failed",
                "syntax_score": None,
                "syntax_left": None,
                "syntax_right": None,
                "dominance": None,
            }
        )
        return 1

    # Aggregate exactly as upstream ``inference/rnn_apply.py`` does: sum the two
    # arteries *within* each fold, calibrate that sum with the fold's (a, b),
    # then average across folds and clamp at zero. Scaling per-side before the
    # sum would apply the intercept ``b`` twice.
    total, left, right = aggregate_folds(left_folds, right_folds, folds, scaling, use_scaling)

    write(
        {
            "case_id": case_id,
            "task_type": "cardiosyntax_scoring",
            "agent": "cardiosyntax_r3d_lstm",
            "model": {
                "backbone": "R3D-18",
                "head": "LSTM (lstm_mean)",
                "ensemble_folds": folds,
                "calibrated": use_scaling,
                "source": "MesserMMP/coronary-syntax-prediction (CardioSyntax v2)",
            },
            "syntax_score": total,
            "syntax_left": left,
            "syntax_right": right,
            # No dominance head upstream; gold is null for 49/60 cases anyway.
            "dominance": None,
            "num_views": {
                "left": len(grouped["left"]),
                "right": len(grouped["right"]),
            },
        }
    )

    print(
        f"[cardiosyntax_r3d] {case_id}: total={total} left={left} right={right} "
        f"-> {prediction_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
