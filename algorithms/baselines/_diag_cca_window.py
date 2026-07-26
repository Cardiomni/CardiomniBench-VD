"""Sweep HU windowing / normalization schemes on one CTA patch to find the
preprocessing the checkpoint was actually trained with.

Runs the model on a small crop centred on gold vessels, so each variant costs
seconds instead of a minute. Reports how vessel-like the response is.

Usage:
    python _diag_cca_window.py [case_dir] [checkpoint]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.networks.nets import UNet

DEFAULT_CASE = (
    "/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/data/tasks/"
    "cca_segmentation/cases/case_cca_0001_0"
)
DEFAULT_CKPT = (
    "/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/"
    "weights/coronary-seg-unet/baseline_unet.pth"
)
PATCH = 96


def build(checkpoint: str, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def variants(patch: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate normalizations, all mapping raw HU to a network-ready range."""
    out: dict[str, np.ndarray] = {}

    # Fixed CT windows scaled to [0,1] (MONAI ScaleIntensityRange style).
    for lo, hi, name in (
        (-200, 300, "win[-200,300]->01"),
        (-100, 700, "win[-100,700]->01"),
        (0, 400, "win[0,400]->01"),
        (-1024, 1024, "win[-1024,1024]->01"),
        (100, 500, "win[100,500]->01"),
    ):
        out[name] = np.clip((patch - lo) / (hi - lo), 0, 1)

    # Same windows mapped to [-1,1].
    for lo, hi, name in ((-200, 300, "win[-200,300]->pm1"), (-100, 700, "win[-100,700]->pm1")):
        out[name] = np.clip((patch - lo) / (hi - lo), 0, 1) * 2 - 1

    # Z-score, the other common nnU-Net/MONAI choice.
    clipped = np.clip(patch, -200, 300)
    out["zscore(clip-200,300)"] = (clipped - clipped.mean()) / (clipped.std() + 1e-8)
    out["zscore(raw)"] = (patch - patch.mean()) / (patch.std() + 1e-8)

    # Raw HU, to confirm it is wrong rather than assume it.
    out["raw HU"] = patch.astype(np.float32)
    return out


def main() -> int:
    case = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE)
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CKPT

    task = yaml.safe_load((case / "task.yaml").read_text())
    image = nib.load(str(case / "image.nii.gz")).get_fdata()
    gold = nib.load(task["gold_standard"]["label_file"]).get_fdata() > 0.5

    # Pick the patch with the MOST gold voxels. The centroid is useless here:
    # the coronary tree is hollow and branching, so its centre of mass falls in
    # empty space between vessels (verified - a centroid patch had 0 gold voxels).
    half = PATCH // 2
    coarse = 16  # stride for the search, in voxels
    best = (-1, None)
    limits = [image.shape[i] - PATCH for i in range(3)]
    idx = np.array(np.nonzero(gold))
    lo_bound = idx.min(axis=1)
    hi_bound = idx.max(axis=1)
    for x in range(max(0, lo_bound[0] - half), min(limits[0], hi_bound[0]), coarse):
        for y in range(max(0, lo_bound[1] - half), min(limits[1], hi_bound[1]), coarse):
            for z in range(max(0, lo_bound[2] - half), min(limits[2], hi_bound[2]), coarse):
                count = int(
                    gold[x : x + PATCH, y : y + PATCH, z : z + PATCH].sum()
                )
                if count > best[0]:
                    best = (count, (x, y, z))
    origin = best[1]
    slices = tuple(slice(origin[i], origin[i] + PATCH) for i in range(3))
    patch = image[slices]
    gold_patch = gold[slices]
    print(f"patch {patch.shape} at origin {origin} (densest gold region)")
    print(f"gold voxels in patch: {int(gold_patch.sum())} ({100*gold_patch.mean():.3f}%)")
    if not gold_patch.any():
        print("ERROR: still no gold voxels; aborting", file=sys.stderr)
        return 1
    print(f"patch HU: min={patch.min():.0f} max={patch.max():.0f} mean={patch.mean():.0f}\n")

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model = build(checkpoint, device)

    print(f"{'variant':26} {'fg%':>7} {'Dice':>7} {'prec':>6} {'rec':>6}")
    print("-" * 56)
    rows = []
    for name, arr in variants(patch).items():
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        pred = probs > 0.5
        inter = np.logical_and(pred, gold_patch).sum()
        dice = 2 * inter / max(1, pred.sum() + gold_patch.sum())
        prec = inter / max(1, pred.sum())
        rec = inter / max(1, gold_patch.sum())
        # Threshold-free check: is the vessel probability higher inside gold than
        # outside? If even this fails, the model is not responding to vessels.
        auc_like = float(probs[gold_patch].mean() - probs[~gold_patch].mean())
        rows.append((dice, auc_like, name))
        print(
            f"{name:26} {100*pred.mean():6.3f}% {dice:7.4f} {prec:6.3f} {rec:6.3f}"
            f"   dProb={auc_like:+.4f}"
        )

    rows.sort(reverse=True)
    print(f"\nbest by Dice: {rows[0][2]}  (Dice {rows[0][0]:.4f})")
    by_prob = sorted(rows, key=lambda r: r[1], reverse=True)[0]
    print(f"best by dProb: {by_prob[2]}  (dProb {by_prob[1]:+.4f})")
    print(
        "\ndProb = mean P(vessel) inside gold minus outside. Positive means the "
        "model does respond to vessels and only the threshold is off; ~0 means "
        "it does not see them at all."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
