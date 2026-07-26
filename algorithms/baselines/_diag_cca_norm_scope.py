"""Does the checkpoint expect per-volume or per-patch z-score statistics?

Training used 96^3 patches, so per-patch statistics are plausible and can differ
a lot from whole-volume ones (a volume is mostly air/lung, a cardiac patch is
mostly soft tissue). Also sweeps the decision threshold, since a weak-but-real
response may just need a lower cut than 0.5.

Usage:
    python _diag_cca_norm_scope.py [case_dir] [checkpoint]
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


def densest_origin(gold: np.ndarray, shape: tuple[int, ...]) -> tuple[int, int, int]:
    idx = np.array(np.nonzero(gold))
    lo, hi = idx.min(axis=1), idx.max(axis=1)
    limits = [shape[i] - PATCH for i in range(3)]
    best = (-1, (0, 0, 0))
    for x in range(max(0, lo[0] - PATCH // 2), min(limits[0], hi[0]), 16):
        for y in range(max(0, lo[1] - PATCH // 2), min(limits[1], hi[1]), 16):
            for z in range(max(0, lo[2] - PATCH // 2), min(limits[2], hi[2]), 16):
                count = int(gold[x : x + PATCH, y : y + PATCH, z : z + PATCH].sum())
                if count > best[0]:
                    best = (count, (x, y, z))
    return best[1]


def main() -> int:
    case = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE)
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CKPT

    task = yaml.safe_load((case / "task.yaml").read_text())
    image = nib.load(str(case / "image.nii.gz")).get_fdata()
    gold = nib.load(task["gold_standard"]["label_file"]).get_fdata() > 0.5

    origin = densest_origin(gold, image.shape)
    slices = tuple(slice(origin[i], origin[i] + PATCH) for i in range(3))
    patch = image[slices]
    gold_patch = gold[slices]

    # Whole-volume statistics, computed once on the clipped volume.
    clipped_volume = np.clip(image, -200, 300)
    volume_mean = float(clipped_volume.mean())
    volume_std = float(clipped_volume.std())
    clipped_patch = np.clip(patch, -200, 300)
    patch_mean = float(clipped_patch.mean())
    patch_std = float(clipped_patch.std())

    print(f"patch origin {origin}  gold {int(gold_patch.sum())} voxels "
          f"({100*gold_patch.mean():.3f}%)")
    print(f"volume stats: mean={volume_mean:8.2f} std={volume_std:7.2f}")
    print(f"patch  stats: mean={patch_mean:8.2f} std={patch_std:7.2f}\n")

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model = build(checkpoint, device)

    schemes = {
        "per-volume zscore": (clipped_patch - volume_mean) / (volume_std + 1e-8),
        "per-patch zscore": (clipped_patch - patch_mean) / (patch_std + 1e-8),
    }

    for name, arr in schemes.items():
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()

        inside = float(probs[gold_patch].mean())
        outside = float(probs[~gold_patch].mean())
        print(f"=== {name} ===")
        print(f"  P(vessel): inside gold {inside:.4f}  outside {outside:.4f}  "
              f"dProb {inside - outside:+.4f}")
        print(f"  prob range [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"  {'thresh':>8} {'fg%':>7} {'Dice':>7} {'prec':>6} {'rec':>6}")
        for threshold in (0.5, 0.3, 0.1, 0.05, 0.02, 0.01):
            pred = probs > threshold
            inter = np.logical_and(pred, gold_patch).sum()
            dice = 2 * inter / max(1, pred.sum() + gold_patch.sum())
            prec = inter / max(1, pred.sum())
            rec = inter / max(1, gold_patch.sum())
            print(f"  {threshold:8.2f} {100*pred.mean():6.2f}% {dice:7.4f} "
                  f"{prec:6.3f} {rec:6.3f}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
