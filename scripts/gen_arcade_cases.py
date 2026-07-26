#!/usr/bin/env python
"""Generate ARCADE case folders (segmentation + stenosis) from locked subsets.

Reads the subset manifests under Datasets/ARCADE_FO/ and the FiftyOne
samples.json, and writes one case folder per image under
data/tasks/<task>/cases/case_*/ with:
  - image.png            symlink -> Datasets/ARCADE_FO/data/<file>
  - task.yaml            case_metadata + input + gold_standard (labels+bboxes)
  - ../../gold/<case>/masks.npz   per-instance binary masks (bbox-local)

Input -> output only; no rubric. Gold masks kept OUT of the case dir.
"""
import base64, io, json, os, zlib
from pathlib import Path
import numpy as np
import yaml

DS = Path("/mnt/aliyunsb/Cardiomni/Datasets/ARCADE_FO")
ROOT = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD")
SAMPLES = json.load(open(DS / "samples.json"))["samples"]
BYFILE = {}  # (task_tag, filename) -> sample
for s in SAMPLES:
    fn = s["filepath"].split("/")[-1]
    tags = set(s["tags"])
    if "test_case_seg" in tags:
        BYFILE[("seg", fn)] = s
    elif "test_case_sten" in tags:
        BYFILE[("sten", fn)] = s

RCA = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}


def decode_mask(det):
    raw = base64.b64decode(det["mask"]["$binary"]["base64"])
    return np.load(io.BytesIO(zlib.decompress(raw))).astype(np.uint8)


def relsym(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)  # absolute target = stable on the NAS


def build_task(task_tag, subset_file, task_name, task_type, difficulty_fn):
    subset = json.load(open(DS / subset_file))["files"]
    task_dir = ROOT / "data" / "tasks" / task_name
    cases_dir = task_dir / "cases"
    gold_dir = task_dir / "gold"
    prefix = "arcade_seg" if task_tag == "seg" else "arcade_sten"

    manifest = []
    for i, fn in enumerate(sorted(subset), 1):
        s = BYFILE[(task_tag, fn)]
        dets = s["segmentations"]["detections"]
        stem = fn.replace(".png", "")
        case_id = f"case_{prefix}_{i:04d}_{stem}"
        cdir = cases_dir / case_id
        cdir.mkdir(parents=True, exist_ok=True)

        # input: symlink the image into the case folder
        relsym(DS / "data" / fn, cdir / "image.png")

        # Frame size, read once: it defines both the input spec below and the
        # pixel boxes recorded per instance.
        W = int(s["metadata"].get("width", 512))
        H = int(s["metadata"].get("height", 512))

        # gold: labels + normalized bboxes inline; masks -> gold sidecar
        instances, masks = [], {}
        for j, d in enumerate(dets):
            mask = decode_mask(d)
            bx, by, bw, bh = (float(v) for v in d["bounding_box"])
            inst = {
                "instance_id": j,
                "label": str(d["label"]),
                "bbox_xywh_norm": [round(float(x), 6) for x in d["bounding_box"]],
                # Pixel box, stored explicitly because the mask's own shape is the
                # authority for its extent. Recovering it from the rounded
                # normalised box does not work: the mask is cropped in source
                # pixel coordinates while bbox_xywh_norm is rounded to 6 decimals,
                # so the two disagree by +/-1 px on ~7% of instances and no
                # rounding convention reproduces all of them. Scoring code that
                # reconstructed the box was silently resizing gold masks.
                "bbox_xywh_px": [
                    int(round(bx * W)),
                    int(round(by * H)),
                    int(mask.shape[1]),
                    int(mask.shape[0]),
                ],
            }
            instances.append(inst)
            masks[f"inst_{j}"] = mask
        gcase = gold_dir / case_id
        gcase.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(gcase / "masks.npz", **masks)

        labels = sorted({str(d["label"]) for d in dets})
        system = "RCA" if all(l in RCA for l in labels) else (
            "LCA" if not any(l in RCA for l in labels) else "MIXED")
        diff = difficulty_fn(dets, labels)

        case = {
            "task_version": "1.0.0",
            "case_id": case_id,
            "case_metadata": {
                "task_type": task_type,
                "source_dataset": "ARCADE",
                "source_split": "test",
                "source_file": fn,
                "coco_id": s.get("coco_id"),
                "difficulty_level": diff,
                "coronary_system": system,
            },
            "input": {
                "modality": "XCA",  # X-ray coronary angiography, single frame
                "image": {
                    "file_path": "image.png",
                    "width": W,
                    "height": H,
                    "note": "Single 2D angiography frame, grayscale.",
                },
            },
            "expected_output": EXPECTED[task_type],
            "gold_standard": {
                "instances": instances,
                "num_instances": len(instances),
                "masks_file": f"../../gold/{case_id}/masks.npz",
                "mask_note": "bbox-local binary masks, one array per instance_id",
            },
        }
        with open(cdir / "task.yaml", "w") as f:
            yaml.safe_dump(case, f, allow_unicode=True, sort_keys=False)
        manifest.append(case_id)

    json.dump({"task": task_name, "n": len(manifest), "cases": manifest},
              open(task_dir / "_cases.json", "w"), indent=2)
    print(f"[{task_name}] wrote {len(manifest)} cases -> {cases_dir}")
    return len(manifest)


EXPECTED = {
    "arcade_segmentation": {
        "format": "instance_list",
        "target": "coronary_artery_segments",
        "label_space": "SYNTAX segment ids (1..16 + a/b/c subsegments), 25 classes",
        "fields_per_instance": ["label", "bbox_xywh_norm", "mask"],
        "metric": "mean_F1_per_image (ARCADE official)",
    },
    "arcade_stenosis": {
        "format": "instance_list",
        "target": "stenosis_regions",
        "label_space": 'single class "stenosis" (location only, no percent)',
        "fields_per_instance": ["label", "bbox_xywh_norm", "mask"],
        "metric": "mean_F1_per_image (ARCADE official)",
    },
}


def seg_difficulty(dets, labels):
    n = len(dets)
    rare = {"12", "10", "15"}
    if any(l in rare for l in labels) or n >= 8:
        return "hard"
    if n >= 6:
        return "medium"
    return "easy"


def sten_difficulty(dets, labels):
    n = len(dets)
    if n >= 4:
        return "hard"
    if n >= 3:
        return "medium"
    return "medium"  # subset is all multi-lesion (>=2)


if __name__ == "__main__":
    total = 0
    total += build_task("seg", "_subset_segmentation.json",
                        "arcade_segmentation", "arcade_segmentation", seg_difficulty)
    total += build_task("sten", "_subset_stenosis_multilesion.json",
                        "arcade_stenosis", "arcade_stenosis", sten_difficulty)
    print(f"TOTAL ARCADE cases: {total}")
