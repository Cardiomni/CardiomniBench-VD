#!/usr/bin/env python
"""Batch-run the CardioSyntax R3D+LSTM baseline over the whole task and score it.

Loads each fold's model once and reuses it across all cases, instead of the
per-case reload the single-case CLI does (that would be 60 x 10 x 128MB of
redundant I/O).

Reports the regression metrics our rubric uses (MAE / RMSE / Pearson r) plus the
SYNTAX>22 balanced accuracy that the upstream paper leads with, so the numbers
are comparable to both.

Usage:
    python run_cardiosyntax_batch.py [--device cuda:4] [--limit N] [--scaling]
                                     [--out runs/cardiosyntax_r3d]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardiosyntax_r3d_agent import (  # noqa: E402
    UPSTREAM_DIR,
    aggregate_folds,
    build_model,
    clip_to_tensor,
    discover_checkpoints,
    load_scaling,
    load_view,
    split_views_by_artery,
)

TASK_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "tasks" / "cardiosyntax_scoring" / "cases"
)
HIGH_COMPLEXITY_THRESHOLD = 22.0


def score_case(models_by_side, view_paths_by_side, device) -> dict[str, dict[int, float]]:
    """Run every preloaded fold model on one case."""
    import torch

    out: dict[str, dict[int, float]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        paths = view_paths_by_side.get(side) or []
        if not paths:
            continue

        clips = []
        for path in paths:
            try:
                clips.append(clip_to_tensor(load_view(path)))
            except Exception as exc:
                print(f"    WARN skip {path.name}: {exc}", file=sys.stderr)
        if not clips:
            continue

        batch = torch.stack(clips, dim=0).unsqueeze(0).to(device)
        for fold, model in models_by_side[side].items():
            with torch.no_grad():
                flat = model(batch).flatten()
            raw = float(flat[1].item() if flat.numel() >= 2 else flat[0].item())
            out[side][fold] = float(np.expm1(raw))  # training target is log1p
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("runs/cardiosyntax_r3d_baseline")
    )
    args = parser.parse_args()

    import torch

    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    checkpoints = discover_checkpoints(UPSTREAM_DIR / "full_model")
    scaling = load_scaling(UPSTREAM_DIR)

    print(f"Loading {len(folds)} folds x 2 sides on {device} ...", flush=True)
    models_by_side: dict[str, dict[int, object]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for fold in folds:
            checkpoint = checkpoints[side].get(fold)
            if checkpoint is None:
                print(f"  WARN no checkpoint for {side} fold{fold}", file=sys.stderr)
                continue
            models_by_side[side][fold] = build_model(checkpoint, device)
    print("Models ready.", flush=True)

    case_dirs = sorted(d for d in TASK_DIR.iterdir() if d.is_dir())
    if args.limit:
        case_dirs = case_dirs[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = time.time()

    for index, case_dir in enumerate(case_dirs, 1):
        task_path = case_dir / "task.yaml"
        if not task_path.exists():
            continue
        task = yaml.safe_load(task_path.read_text())
        task["__case_dir__"] = str(case_dir.resolve())
        case_id = task.get("case_id", case_dir.name)
        gold = task.get("gold_standard", {}) or {}

        grouped = split_views_by_artery(task)
        try:
            per_fold = score_case(models_by_side, grouped, device)
        except Exception as exc:
            print(f"[{index}/{len(case_dirs)}] {case_id}: FAILED {exc}", file=sys.stderr)
            continue

        total, left, right = aggregate_folds(
            per_fold["left"], per_fold["right"], folds, scaling, args.scaling
        )
        record = {
            "case_id": case_id,
            "pred_total": total,
            "pred_left": left,
            "pred_right": right,
            "gold_total": gold.get("syntax_score"),
            "gold_left": gold.get("syntax_left"),
            "gold_right": gold.get("syntax_right"),
            "expert_min": gold.get("expert_min"),
            "expert_max": gold.get("expert_max"),
            "n_views": {"left": len(grouped["left"]), "right": len(grouped["right"])},
        }
        records.append(record)
        (args.out / f"{case_id}.json").write_text(json.dumps(record, indent=2))

        elapsed = time.time() - started
        print(
            f"[{index}/{len(case_dirs)}] {case_id}: "
            f"pred={total if total is None else round(total, 2)} "
            f"gold={record['gold_total']}  ({elapsed:.0f}s)",
            flush=True,
        )
        print(f"@@progress {index / max(1, len(case_dirs)):.4f}", flush=True)

    # ---- aggregate metrics -------------------------------------------------
    paired = [
        (r["pred_total"], float(r["gold_total"]))
        for r in records
        if r["pred_total"] is not None and r["gold_total"] is not None
    ]
    summary: dict = {
        "agent": "cardiosyntax_r3d_lstm",
        "source": "MesserMMP/coronary-syntax-prediction (CardioSyntax v2)",
        "folds": folds,
        "calibrated": bool(args.scaling),
        "n_cases": len(records),
        "n_scored": len(paired),
    }

    if paired:
        preds = np.array([p for p, _ in paired])
        golds = np.array([g for _, g in paired])
        errors = np.abs(preds - golds)
        summary["mae"] = float(errors.mean())
        summary["rmse"] = float(np.sqrt(((preds - golds) ** 2).mean()))
        summary["median_ae"] = float(np.median(errors))
        if len(paired) > 1 and preds.std() > 0 and golds.std() > 0:
            summary["pearson_r"] = float(np.corrcoef(preds, golds)[0, 1])

        # SYNTAX>22 balanced accuracy, the upstream paper's headline metric.
        pred_high = preds > HIGH_COMPLEXITY_THRESHOLD
        gold_high = golds > HIGH_COMPLEXITY_THRESHOLD
        tp = int((pred_high & gold_high).sum())
        tn = int((~pred_high & ~gold_high).sum())
        fp = int((pred_high & ~gold_high).sum())
        fn = int((~pred_high & gold_high).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        summary["high_complexity"] = {
            "threshold": HIGH_COMPLEXITY_THRESHOLD,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        }

        # Agreement with the expert spread, where available.
        in_band = [
            r for r in records
            if r["pred_total"] is not None
            and r.get("expert_min") is not None
            and r.get("expert_max") is not None
        ]
        if in_band:
            hits = sum(
                1 for r in in_band
                if float(r["expert_min"]) <= r["pred_total"] <= float(r["expert_max"])
            )
            summary["within_expert_band"] = {
                "n": len(in_band),
                "hits": hits,
                "rate": hits / len(in_band),
            }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    for key in ("n_cases", "n_scored", "calibrated", "mae", "rmse", "median_ae", "pearson_r"):
        if key in summary:
            value = summary[key]
            print(f"  {key:12} {value:.4f}" if isinstance(value, float) else f"  {key:12} {value}")
    if "high_complexity" in summary:
        hc = summary["high_complexity"]
        print(
            f"  SYNTAX>22    balanced_acc={hc['balanced_accuracy']:.4f} "
            f"(sens={hc['sensitivity']:.3f} spec={hc['specificity']:.3f})"
        )
    if "within_expert_band" in summary:
        band = summary["within_expert_band"]
        print(f"  expert band  {band['hits']}/{band['n']} = {band['rate']:.3f}")
    print(f"\nWrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
