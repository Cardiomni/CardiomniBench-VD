"""Locate where predicted vs gold voxels sit, to tell an orientation bug apart
from genuine model failure.

Usage:
    python _diag_cca_geometry.py <case_dir> <pred_mask.nii.gz>
"""

from __future__ import annotations

import sys

import nibabel as nib
import numpy as np
import yaml

DEFAULT_CASE = (
    "/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/data/tasks/"
    "cca_segmentation/cases/case_cca_0001_0"
)


def main() -> int:
    case = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE
    pred_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cca_test/mask.nii.gz"

    with open(f"{case}/task.yaml") as handle:
        task = yaml.safe_load(handle)

    img = nib.load(f"{case}/image.nii.gz")
    gold = nib.load(task["gold_standard"]["label_file"])
    pred = nib.load(pred_path)

    print("--- headers ---")
    for name, nii in (("image", img), ("gold", gold), ("pred", pred)):
        zooms = tuple(round(float(z), 3) for z in nii.header.get_zooms())
        print(f"{name:6} shape={nii.shape} zooms={zooms} axcodes={nib.aff2axcodes(nii.affine)}")

    g = gold.get_fdata() > 0.5
    p = pred.get_fdata() > 0.5

    print("\n--- bounding boxes ---")
    for name, mask in (("gold", g), ("pred", p)):
        if not mask.any():
            print(f"{name}: EMPTY")
            continue
        idx = np.array(np.nonzero(mask))
        box = [(int(idx[i].min()), int(idx[i].max())) for i in range(3)]
        com = idx.mean(axis=1)
        print(f"{name}: bbox={box} centroid=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f})")

    print("\n--- does a flip or transpose recover overlap? ---")
    print(f"identity              inter={int(np.logical_and(g, p).sum())}")
    for axis in (0, 1, 2):
        inter = int(np.logical_and(g, np.flip(p, axis=axis)).sum())
        print(f"flip axis {axis}           inter={inter}")
    for perm in ((0, 2, 1), (1, 0, 2), (2, 1, 0)):
        permuted_shape = tuple(np.array(p.shape)[list(perm)])
        if permuted_shape == g.shape:
            inter = int(np.logical_and(g, np.transpose(p, perm)).sum())
            print(f"transpose {perm}     inter={inter}")

    print("\n--- extent along each axis ---")
    for name, mask in (("gold", g), ("pred", p)):
        for axis in range(3):
            other = tuple(i for i in range(3) if i != axis)
            counts = mask.sum(axis=other)
            nz = np.nonzero(counts)[0]
            span = f"{int(nz.min())}..{int(nz.max())}" if nz.size else "none"
            print(
                f"  {name} axis{axis}: span={span} "
                f"peak_at={int(counts.argmax())} peak={int(counts.max())}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
