"""
Gold-standard loading and scoring for CardiomniBench-VD.

All comparison against ground truth happens here. Keeping it in one module means
a reviewer can verify the scoring logic without reading any inference code, and
no runner can influence its own score.

Two levels of metric, kept deliberately separate
------------------------------------------------
per-case   Computable from one (gold, prediction) pair: absolute error, Dice.
           Averaged across cases at reporting time.

set-level  Only defined over the whole case set: Pearson correlation, decision
           sensitivity/specificity, expert-agreement rate. A correlation over 60
           cases is not the mean of 60 per-case correlations, so these are
           computed once from the collected pairs rather than averaged.

Gold location differs by task, which is a real trap: CardioSYNTAX gold is inline
in task.yaml, while CCA gold is an out-of-tree NIfTI referenced by path. Both are
handled by load_gold so nothing downstream needs to know the difference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.core import Task


@dataclass
class Gold:
    """Ground truth for one case, uniform across tasks."""

    case_id: str
    task: Task

    # Scalar tasks
    score: float | None = None
    components: dict[str, float] = field(default_factory=dict)

    #: Individual expert readings, when the dataset provides them. These define
    #: what "expert-level" means for this case: three cardiologists scoring the
    #: same angiogram spanned 9 SYNTAX points on the first case in our set, so a
    #: model error must be read against that spread, not against zero.
    expert_scores: list[float] = field(default_factory=list)
    expert_min: float | None = None
    expert_max: float | None = None

    # Volume tasks
    mask: np.ndarray | None = None
    spacing: tuple[float, ...] | None = None

    # Instance tasks (ARCADE). Each instance is a dict with 'label',
    # 'bbox_xywh_norm', and (once masks.npz is loaded) 'mask' as a bbox-local
    # array. Image size is kept because bboxes are normalised, so scoring cannot
    # place them without it.
    instances: list[dict[str, Any]] = field(default_factory=list)
    width: int = 512
    height: int = 512


def load_gold(case_dir: Path) -> Gold:
    """Load ground truth for a case, handling both storage conventions."""
    import yaml

    with (case_dir / "task.yaml").open() as handle:
        spec = yaml.safe_load(handle)

    task = Task(spec["case_metadata"]["task_type"])
    block = spec.get("gold_standard") or {}
    gold = Gold(case_id=spec["case_id"], task=task)

    if task is Task.CARDIOSYNTAX_SCORING:
        # Inline in task.yaml, not a separate file.
        gold.score = float(block["syntax_score"])
        for key in ("syntax_left", "syntax_right"):
            if block.get(key) is not None:
                gold.components[key] = float(block[key])
        experts = [float(s) for s in (block.get("expert_scores") or [])]
        gold.expert_scores = experts
        gold.expert_min = (
            float(block["expert_min"]) if block.get("expert_min") is not None
            else (min(experts) if experts else None)
        )
        gold.expert_max = (
            float(block["expert_max"]) if block.get("expert_max") is not None
            else (max(experts) if experts else None)
        )

    elif task is Task.CCA_SEGMENTATION:
        # Out-of-tree NIfTI referenced by absolute path.
        import nibabel as nib

        label_path = Path(block["label_file"])
        if not label_path.exists():
            raise FileNotFoundError(f"{gold.case_id}: gold mask missing {label_path}")
        nii = nib.load(str(label_path))
        gold.mask = (nii.get_fdata() > 0.5).astype(np.uint8)
        gold.spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])

    elif task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        image_spec = ((spec.get("input") or {}).get("image")) or {}
        gold.width = int(image_spec.get("width", 512))
        gold.height = int(image_spec.get("height", 512))

        instances = [dict(inst) for inst in (block.get("instances") or [])]

        # Masks live in a sibling .npz keyed by 'inst_<instance_id>', with each
        # array cropped to its own bbox. The path in task.yaml is relative to the
        # case directory. A missing file is not fatal: bboxes alone still yield a
        # (weaker) detection score, and evaluate_instances records which was used.
        masks_rel = block.get("masks_file")
        if masks_rel:
            masks_path = (case_dir / masks_rel).resolve()
            if masks_path.exists():
                with np.load(masks_path) as archive:
                    for inst in instances:
                        key = f"inst_{inst.get('instance_id')}"
                        if key in archive:
                            inst["mask"] = archive[key]
                            # Marks this mask as the reference, so the metrics
                            # refuse to resample it onto a mismatched box instead
                            # of silently distorting gold.
                            inst["_is_gold"] = True
            else:
                raise FileNotFoundError(
                    f"{gold.case_id}: masks_file declared but missing: {masks_path}"
                )

        gold.instances = instances

    else:
        raise ValueError(f"unsupported task: {task}")

    return gold


# ==========================================================================
# Per-case scoring
# ==========================================================================


def score_case(gold: Gold, prediction, output_dir: Path) -> dict[str, float]:
    """Compute per-case metrics for one prediction.

    Values needed later for set-level metrics (the raw gold/pred pair, the expert
    envelope) are included so aggregation never has to reload gold from disk.
    """
    if gold.task is Task.CARDIOSYNTAX_SCORING:
        return _score_syntax(gold, prediction)
    if gold.task is Task.CCA_SEGMENTATION:
        return _score_segmentation(gold, prediction)
    if gold.task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        return _score_instances(gold, prediction)
    raise ValueError(f"unsupported task: {gold.task}")


def _score_syntax(gold: Gold, prediction) -> dict[str, float]:
    """Per-case SYNTAX metrics."""
    pred = prediction.score
    if pred is None or pred != pred:
        return {}

    gold_score = float(gold.score)
    error = pred - gold_score

    metrics: dict[str, float] = {
        "mae": abs(error),
        "signed_error": error,
        "squared_error": error * error,
        # Carried for set-level correlation and decision metrics.
        "gold_score": gold_score,
        "pred_score": float(pred),
    }

    # Risk tertile agreement: low <=22, intermediate 23-32, high >=33.
    def tier(value: float) -> int:
        return 0 if value <= 22 else (1 if value < 33 else 2)

    metrics["tier_correct"] = float(tier(gold_score) == tier(pred))

    # The >22 threshold is the one that changes clinical management, so it gets
    # its own per-case record for set-level sensitivity/specificity.
    metrics["gold_gt22"] = float(gold_score > 22)
    metrics["pred_gt22"] = float(pred > 22)

    # Expert envelope, when available.
    if gold.expert_min is not None and gold.expert_max is not None:
        metrics["expert_min"] = gold.expert_min
        metrics["expert_max"] = gold.expert_max
        metrics["within_expert_range"] = float(
            gold.expert_min <= pred <= gold.expert_max
        )
        metrics["expert_spread"] = gold.expert_max - gold.expert_min

    return metrics


def _score_segmentation(gold: Gold, prediction) -> dict[str, float]:
    """Per-case segmentation metrics."""
    import nibabel as nib

    from evaluation.metrics.segmentation_metrics import evaluate_segmentation

    if prediction.mask_path is None or not prediction.mask_path.exists():
        return {}

    pred_nii = nib.load(str(prediction.mask_path))
    pred_mask = (pred_nii.get_fdata() > 0.5).astype(np.uint8)

    if pred_mask.shape != gold.mask.shape:
        # A shape mismatch means the prediction is unusable; report it rather
        # than silently resampling, which would fabricate agreement.
        return {
            "shape_mismatch": 1.0,
            "dice": 0.0,
        }

    # clDice and Hausdorff each cost minutes on a 832x832x576 CTA (the CCA cases
    # are all this size) and the morphological skeleton allocates several float32
    # copies of the volume. CARDIOMNI_FAST_METRICS=1 drops them so a full sweep
    # can report Dice first; rerun without it for the headline topology numbers.
    fast = os.environ.get("CARDIOMNI_FAST_METRICS", "") == "1"

    return evaluate_segmentation(
        gold.mask,
        pred_mask,
        spacing=gold.spacing or (1.0, 1.0, 1.0),
        include_hausdorff=not fast,
        include_cldice=not fast,
    )


def _score_instances(gold: Gold, prediction) -> dict[str, float]:
    """Per-case instance-segmentation metrics for ARCADE."""
    from evaluation.metrics.instance_metrics import evaluate_instances

    # Gold is loaded from task.yaml, which carries width/height in input.image.
    # When it is missing fall back to ARCADE's standard 512.
    width = getattr(gold, "width", 512)
    height = getattr(gold, "height", 512)

    gold_instances = getattr(gold, "instances", None) or []
    pred_instances = prediction.instances or []

    return evaluate_instances(
        gold_instances,
        pred_instances,
        width=width,
        height=height,
    )


# ==========================================================================
# Set-level scoring
# ==========================================================================


def set_level_metrics(task: Task, per_case: list[dict[str, float]]) -> dict[str, float]:
    """Metrics defined only over the whole case set.

    `per_case` is the list of metric dicts returned by score_case, in any order.
    """
    if task is not Task.CARDIOSYNTAX_SCORING:
        return {}

    from evaluation.metrics.syntax_scoring_metrics import (
        compute_decision_metrics,
        compute_point_metrics,
        compute_severity_breakdown,
        compute_tier_accuracy,
    )

    paired = [
        m for m in per_case if "gold_score" in m and "pred_score" in m
    ]
    if len(paired) < 2:
        return {}

    gold_scores = [m["gold_score"] for m in paired]
    pred_scores = [m["pred_score"] for m in paired]

    out: dict[str, float] = {}
    out.update(compute_point_metrics(gold_scores, pred_scores))
    out.update(compute_decision_metrics(gold_scores, pred_scores))
    out.update(compute_tier_accuracy(gold_scores, pred_scores))

    # Severity breakdown catches a regressor that has learned the mean: it will
    # look acceptable overall while failing entirely on the severe cases.
    breakdown = compute_severity_breakdown(gold_scores, pred_scores)
    for band, stats in breakdown.items():
        for key, value in stats.items():
            out[f"{band}_{key}"] = value

    # Expert agreement, using the envelope carried on each case.
    with_experts = [
        m for m in paired if "expert_min" in m and "expert_max" in m
    ]
    if with_experts:
        inside = sum(m["within_expert_range"] for m in with_experts)
        out["within_expert_range"] = inside / len(with_experts)
        out["n_with_experts"] = float(len(with_experts))
        out["mean_expert_spread"] = float(
            np.mean([m["expert_spread"] for m in with_experts])
        )

    return out
