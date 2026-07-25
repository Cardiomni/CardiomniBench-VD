#!/usr/bin/env python
"""Generate CardioSYNTAX case folders, 44 studies (the on-disk Part 9 subset).

Each case = one patient study with N multi-view angiography cine videos (.npy):
  - videos/<vid>.npy    symlink -> Datasets/CardioSYNTAX/9/<uid>/<vid>.npy
  - task.yaml           input (per-video artery + projection angles) + gold
                        (syntax score + left/right + dominance + bypass)

This is the ONLY dataset whose input matches Cardiomni's real input format
(multi-view DSA cines + projection angles). SYNTAX score is a future-work label
per the pivot; included as a regression/ dominance task for now.
"""
import json, os
from pathlib import Path
import yaml

DS = Path("/mnt/aliyunsb/Cardiomni/Datasets/CardioSYNTAX")
ROOT = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD")

EXPECTED = {
    "format": "structured_json",
    "targets": {
        "syntax_score": "float (total anatomical SYNTAX score)",
        "syntax_left": "float", "syntax_right": "float",
        "dominance": 'one of {"right","left"} (SYNTAX has no co-dominant)',
    },
    "metric": "MAE / RMSE / R2 for scores; accuracy for dominance",
    "note": "Study-level prediction from all views combined.",
}


def relsym(target: Path, link: Path):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def band(s):
    if s is None:
        return None
    if s <= 22:
        return "low"
    if s <= 32:
        return "intermediate"
    return "high"


def main():
    studies = json.load(open(DS / "part9.json"))
    # keep only studies whose video dir exists on disk
    ondisk = {d for d in os.listdir(DS / "9") if (DS / "9" / d).is_dir()}
    studies = [s for s in studies if s["study_uid"] in ondisk]

    task_dir = ROOT / "data" / "tasks" / "cardiosyntax_scoring"
    cases_dir = task_dir / "cases"
    manifest = []
    for i, s in enumerate(sorted(studies, key=lambda x: x["study_uid"]), 1):
        uid = s["study_uid"]
        short = uid.split(".")[-1][-8:]
        case_id = f"case_csyn_{i:04d}_{short}"
        cdir = cases_dir / case_id
        (cdir / "videos").mkdir(parents=True, exist_ok=True)

        views = []
        for v in s.get("videos", []):
            vp = DS / v["path"]                # 9/<uid>/<vid>.npy
            if not vp.exists():
                continue
            vname = Path(v["path"]).name
            relsym(vp, cdir / "videos" / vname)
            views.append({
                "file_path": f"videos/{vname}",
                "artery": v.get("artery"),
                "positioner_primary_angle": v.get("PositionerPrimaryAngle"),
                "positioner_secondary_angle": v.get("PositionerSecondaryAngle"),
                "shape": v.get("shape"),
            })

        case = {
            "task_version": "1.0.0",
            "case_id": case_id,
            "case_metadata": {
                "task_type": "cardiosyntax_scoring",
                "source_dataset": "CardioSYNTAX",
                "source_split": "part9_ondisk",
                "study_uid": uid,
                "num_views": len(views),
                "difficulty_level": band(s.get("syntax")) or "low",
            },
            "input": {
                "modality": "XCA_cine",  # multi-view X-ray angiography videos
                "views": views,
                "note": ("Multiple projection cine videos (.npy, frames x 512 x "
                         "512, uint8). Combine views for a study-level readout."),
            },
            "expected_output": EXPECTED,
            "gold_standard": {
                "syntax_score": s.get("syntax"),
                "syntax_left": s.get("syntax_left"),
                "syntax_right": s.get("syntax_right"),
                "dominance": s.get("dominance"),
                "bypass": s.get("bypass"),
                "risk_band": band(s.get("syntax")),
            },
        }
        with open(cdir / "task.yaml", "w") as f:
            yaml.safe_dump(case, f, allow_unicode=True, sort_keys=False)
        manifest.append(case_id)
    json.dump({"task": "cardiosyntax_scoring", "n": len(manifest), "cases": manifest},
              open(task_dir / "_cases.json", "w"), indent=2)
    print(f"[cardiosyntax_scoring] wrote {len(manifest)} cases -> {cases_dir}")


if __name__ == "__main__":
    main()
