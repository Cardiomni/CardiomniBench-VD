#!/usr/bin/env python
"""Generate CCA case folders (3D coronary CTA vessel segmentation), 20 cases.

Each case:
  - image.nii.gz   symlink -> Datasets/CCA/train/images/<n>.nii.gz
  - task.yaml      input (volume path + shape/spacing) + gold_standard
  - gold label mask is referenced (kept in the source tree; it IS the answer,
    so task.yaml.gold_standard.label_file points at it and gets stripped from
    the agent-facing task_spec).

CCA is CTA (not DSA); per the pivot it is out of the DSA-only main line and
serves as a vessel-anatomy / tool-training source. Included for completeness.
"""
import json
from pathlib import Path
import yaml

DS = Path("/mnt/aliyunsb/Cardiomni/Datasets/CCA")
ROOT = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD")

EXPECTED = {
    "format": "binary_volume_mask",
    "target": "coronary_artery_tree (whole)",
    "label_space": "binary {0=background, 1=coronary}",
    "metric": "Dice (aux: clDice, Hausdorff)",
    "output_note": "Write a 3D mask NIfTI same shape as input to prediction dir.",
}


def relsym(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def main():
    import nibabel as nib
    task_dir = ROOT / "data" / "tasks" / "cca_segmentation"
    cases_dir = task_dir / "cases"
    entries = json.load(open(DS / "train" / "train.json"))
    manifest = []
    for i, e in enumerate(entries, 1):
        img_rel = e["image"]           # train/images/N.nii.gz
        lab_rel = e["label"]
        n = Path(img_rel).stem.replace(".nii", "")
        case_id = f"case_cca_{i:04d}_{n}"
        cdir = cases_dir / case_id
        cdir.mkdir(parents=True, exist_ok=True)
        relsym(DS / img_rel, cdir / "image.nii.gz")

        hdr = nib.load(str(DS / img_rel))
        shape = [int(x) for x in hdr.shape]
        zooms = [round(float(x), 4) for x in hdr.header.get_zooms()]

        case = {
            "task_version": "1.0.0",
            "case_id": case_id,
            "case_metadata": {
                "task_type": "cca_segmentation",
                "source_dataset": "CCA",
                "source_split": "train_public",  # only 20 public
                "modality_note": "CTA (out of DSA-only main line; anatomy/tool source)",
                "difficulty_level": "medium",
            },
            "input": {
                "modality": "CTA",
                "volume": {
                    "file_path": "image.nii.gz",
                    "shape": shape,
                    "spacing_mm": zooms,
                    "note": "3D isotropic 0.5mm coronary CT angiography volume.",
                },
            },
            "expected_output": EXPECTED,
            "gold_standard": {
                "label_file": str((DS / lab_rel)),
                "label_space": "binary 0/1",
                "note": "3D voxel mask, same shape as input; foreground ~0.1%.",
            },
        }
        with open(cdir / "task.yaml", "w") as f:
            yaml.safe_dump(case, f, allow_unicode=True, sort_keys=False)
        manifest.append(case_id)
    json.dump({"task": "cca_segmentation", "n": len(manifest), "cases": manifest},
              open(task_dir / "_cases.json", "w"), indent=2)
    print(f"[cca_segmentation] wrote {len(manifest)} cases -> {cases_dir}")


if __name__ == "__main__":
    main()
