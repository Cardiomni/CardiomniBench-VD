#!/usr/bin/env python
"""Generate CardioSYNTAX case folders — the 60 THREE-EXPERT annotated studies.

Gold = official SYNTAX score (all.json, aligns with the CardioSYNTAX paper /
end-to-end baseline) PLUS the three independent expert scores (reliability band:
median / mean / min-max) — the latter is what makes this set special and feeds a
"within-expert-range" tolerance metric later.

Sources:
  - three_experts.json : {study_uid: {expert0_score, expert1_score, expert2_score}}  (60)
  - all.json           : per-study {syntax, syntax_left, syntax_right, dominance,
                         bypass, videos[{path, artery, angles, shape}]}  (1844, keyed by uid)
  - videos on disk     : Datasets/CardioSYNTAX/datasets/<uid>/<vid>.npy (flat, no part prefix)

Each case:
  videos/<vid>.npy   symlink -> Datasets/CardioSYNTAX/datasets/<uid>/<vid>.npy
  task.yaml          case_metadata + input(views+angles) + gold_standard
"""
import json, os, statistics
from pathlib import Path
import yaml

CS = Path("/mnt/aliyunsb/Cardiomni/Datasets/CardioSYNTAX")
RAW = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/.raw_data/CardioSyntax")
ROOT = Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD")
VIDROOT = CS / "datasets"

EXPECTED = {
    "format": "structured_json",
    "targets": {
        "syntax_score": "float (total anatomical SYNTAX score)",
        "syntax_left": "float", "syntax_right": "float",
        "dominance": 'one of {"right","left"} (SYNTAX has no co-dominant)',
    },
    "metric": "MAE / RMSE / R2 for scores; accuracy for dominance",
    "aux_metric": "fraction of predictions within the 3-expert [min,max] band",
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
    te = json.load(open(RAW / "three_experts.json"))          # 60 uids -> 3 scores
    allj = {s["study_uid"]: s for s in json.load(open(RAW / "all.json"))}

    task_dir = ROOT / "data" / "tasks" / "cardiosyntax_scoring"
    cases_dir = task_dir / "cases"
    manifest = []
    skipped = []

    for i, uid in enumerate(sorted(te.keys()), 1):
        meta = allj.get(uid)
        vdir = VIDROOT / uid
        if meta is None or not vdir.is_dir():
            skipped.append(uid)
            continue

        short = uid.split(".")[-1][-8:]
        case_id = f"case_csyn_{i:04d}_{short}"
        cdir = cases_dir / case_id
        (cdir / "videos").mkdir(parents=True, exist_ok=True)

        disk = set(os.listdir(vdir))
        views = []
        for v in meta.get("videos", []):
            bn = os.path.basename(v["path"])
            if bn not in disk:
                continue
            relsym(vdir / bn, cdir / "videos" / bn)
            views.append({
                "file_path": f"videos/{bn}",
                "artery": v.get("artery"),
                "artery_prob": v.get("artery_prob"),
                "positioner_primary_angle": v.get("PositionerPrimaryAngle"),
                "positioner_secondary_angle": v.get("PositionerSecondaryAngle"),
                "shape": v.get("shape"),
            })

        trio = [te[uid]["expert0_score"], te[uid]["expert1_score"], te[uid]["expert2_score"]]
        official = meta.get("syntax")

        case = {
            "task_version": "1.0.0",
            "case_id": case_id,
            "case_metadata": {
                "task_type": "cardiosyntax_scoring",
                "source_dataset": "CardioSYNTAX",
                "source_subset": "three_expert_60",
                "study_uid": uid,
                "num_views": len(views),
                "has_dominance_label": meta.get("dominance") is not None,
                "difficulty_level": band(official) or "low",
            },
            "input": {
                "modality": "XCA_cine",
                "views": views,
                "note": ("Multiple projection cine videos (.npy, frames x 512 x "
                         "512, uint8). Combine views for a study-level readout."),
            },
            "expected_output": EXPECTED,
            "gold_standard": {
                # primary label (official CardioSYNTAX, paper-aligned)
                "syntax_score": official,
                "syntax_left": meta.get("syntax_left"),
                "syntax_right": meta.get("syntax_right"),
                "dominance": meta.get("dominance"),
                "bypass": meta.get("bypass"),
                "risk_band": band(official),
                # reliability band from 3 independent expert reads
                "expert_scores": trio,
                "expert_median": round(statistics.median(trio), 2),
                "expert_mean": round(statistics.mean(trio), 2),
                "expert_min": min(trio),
                "expert_max": max(trio),
                "expert_spread": round(max(trio) - min(trio), 2),
            },
        }
        with open(cdir / "task.yaml", "w") as f:
            yaml.safe_dump(case, f, allow_unicode=True, sort_keys=False)
        manifest.append(case_id)

    json.dump({"task": "cardiosyntax_scoring", "n": len(manifest),
               "subset": "three_expert_60", "cases": manifest},
              open(task_dir / "_cases.json", "w"), indent=2)
    print(f"[cardiosyntax_scoring] wrote {len(manifest)} cases -> {cases_dir}")
    if skipped:
        print(f"  SKIPPED {len(skipped)} (no meta/videos): {skipped[:3]}...")


if __name__ == "__main__":
    main()
