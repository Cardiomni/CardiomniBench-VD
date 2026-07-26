"""Instance-level metrics for the ARCADE tasks.

ARCADE asks for a *labelled instance list* per image, not a single binary mask:
each coronary segment is its own instance carrying a SYNTAX segment id (25
classes: ``1 2 3 4 5 6 7 8 9 9a 10 10a 11 12 12a 12b 13 14 14a 14b 15 16 16a
16b 16c``). Our locked subset has 4-9 instances per image, so both "found the
vessel but called it segment 7 instead of 6" and "merged two segments into one"
are real failure modes that a plain Dice would hide.

Scoring therefore works in two stages:

1. **Match** predicted instances to gold instances. Matching is *label-aware*:
   an instance may only match gold of the same SYNTAX id, because relabelling a
   correctly traced vessel is a clinical error, not a near-miss. Within a label,
   the optimal one-to-one assignment maximises total IoU (Hungarian algorithm),
   so a method is never penalised for the order it happened to emit instances.
2. **Score** the matching. A match counts as a true positive when its IoU clears
   a threshold; unmatched predictions are false positives and unmatched gold are
   false negatives. That gives precision / recall / F1 per image, which is what
   ARCADE's official ``mean_F1_per_image`` averages.

The IoU threshold is explicit rather than hidden: ``0.5`` is the ARCADE/COCO
convention, but thin distal vessels are only a few pixels wide, where a
one-pixel centreline offset costs far more IoU than it does on a thick proximal
segment. ``f1_at_iou`` is therefore reported across several thresholds so a
single strict cutoff cannot make a usable method look broken.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

#: The 25 SYNTAX segment ids ARCADE labels, in the order used by the dataset.
SYNTAX_SEGMENT_LABELS: tuple[str, ...] = (
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "9a",
    "10", "10a", "11", "12", "12a", "12b", "13", "14", "14a", "14b",
    "15", "16", "16a", "16b", "16c",
)

#: Thresholds reported by :func:`evaluate_instances`. 0.5 is the headline.
DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75)


# --- geometry ----------------------------------------------------------------


def _bbox_to_xyxy(
    instance_or_bbox: Any, width: int, height: int
) -> tuple[int, int, int, int]:
    """Resolve an instance's pixel box as ``(x0, y0, x1, y1)``.

    ``bbox_xywh_px`` is preferred when present, because reconstructing the box
    from the rounded normalised one is lossy: gold masks are cropped in source
    pixel coordinates while ``bbox_xywh_norm`` is rounded to 6 decimals, so the
    two disagree by +/-1 px on ~7% of ARCADE instances and no rounding convention
    reproduces all of them. Trusting the normalised box there silently resized
    gold itself, which stayed hidden because gold-vs-gold distorts both sides
    identically and still scores 1.0.

    A bare sequence is still accepted so predictions, which only ever carry the
    normalised box, keep working unchanged.
    """
    px = (
        instance_or_bbox.get("bbox_xywh_px")
        if isinstance(instance_or_bbox, dict)
        else None
    )
    if px is not None:
        x0, y0, bw, bh = (int(round(float(v))) for v in px)
        x1, y1 = x0 + bw, y0 + bh
    else:
        bbox = (
            instance_or_bbox["bbox_xywh_norm"]
            if isinstance(instance_or_bbox, dict)
            else instance_or_bbox
        )
        x, y, w, h = (float(v) for v in bbox)
        x0 = int(round(x * width))
        y0 = int(round(y * height))
        x1 = int(round((x + w) * width))
        y1 = int(round((y + h) * height))

    # Clip to the image so a slightly out-of-range prediction degrades
    # gracefully instead of raising.
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(x0, min(x1, width))
    y1 = max(y0, min(y1, height))
    return x0, y0, x1, y1


def _instance_to_full_mask(
    instance: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    """Paint one instance onto a full-image boolean mask.

    Gold stores masks bbox-local (``masks.npz`` holds one array per instance,
    shaped like its box), so the box is what places the mask in the image. When
    an instance carries no mask the filled box is used instead: that keeps
    box-only methods (e.g. detection LoRAs) scoreable, and it is recorded in the
    per-case diagnostics as ``mask_source="bbox"`` rather than passed off as a
    real segmentation.
    """
    canvas = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = _bbox_to_xyxy(instance, width, height)
    if x1 <= x0 or y1 <= y0:
        return canvas

    mask = instance.get("mask")
    if mask is None:
        canvas[y0:y1, x0:x1] = True
        return canvas

    local = np.asarray(mask)
    if local.ndim != 2:
        raise ValueError(
            f"instance mask must be 2D, got shape {local.shape}"
        )
    local = local.astype(bool)

    box_h, box_w = y1 - y0, x1 - x0
    if local.shape != (box_h, box_w):
        # A gold mask must never be resampled: its stored shape is the authority
        # for its extent, so a mismatch means the box and the mask disagree and
        # the numbers would be scored against distorted gold. Predictions are
        # resized instead of rejected, since a method that emits a mask at its own
        # working resolution is doing something reasonable.
        if instance.get("_is_gold"):
            raise ValueError(
                f"gold instance mask shape {local.shape} does not match its box "
                f"{(box_h, box_w)}. Resampling gold would distort the reference; "
                "regenerate the cases so bbox_xywh_px is present and consistent "
                "(scripts/gen_arcade_cases.py)."
            )
        # Nearest-neighbour resize onto the box. Interpolation is deliberately
        # avoided: it would invent intermediate values on a binary mask.
        rows = (np.linspace(0, local.shape[0] - 1, box_h)).round().astype(int)
        cols = (np.linspace(0, local.shape[1] - 1, box_w)).round().astype(int)
        local = local[np.ix_(rows, cols)]

    canvas[y0:y1, x0:x1] = local
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


# --- matching ----------------------------------------------------------------


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Optimal assignment on a square-or-rectangular cost matrix.

    Uses ``scipy.optimize.linear_sum_assignment`` when SciPy is importable and
    falls back to a greedy pass otherwise. The fallback exists so the metric
    never becomes the reason a run cannot be scored; it is noted in the returned
    diagnostics because greedy can be worse than optimal on ambiguous overlaps.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - SciPy is a declared dependency
        return _greedy_assignment(cost)

    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def _greedy_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Descending-score greedy matching, used only if SciPy is unavailable."""
    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    order = np.argsort(cost, axis=None)
    for flat in order:
        r, c = np.unravel_index(flat, cost.shape)
        r, c = int(r), int(c)
        if r in used_rows or c in used_cols:
            continue
        pairs.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
    return pairs


def match_instances(
    gold_instances: Sequence[dict[str, Any]],
    pred_instances: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Label-aware optimal matching between gold and predicted instances.

    Returns the matched pairs with their IoU, plus the unmatched indices on each
    side. Matching is done independently per SYNTAX label, which enforces that a
    prediction can only be credited when it names the segment correctly.
    """
    return _match_instances_impl(
        gold_instances, pred_instances, width, height, label_aware=True
    )


def match_instances_ignoring_label(
    gold_instances: Sequence[dict[str, Any]],
    pred_instances: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Label-agnostic optimal matching for binary segmenters.

    Identical to :func:`match_instances` but collapses all labels to a single
    bucket, so a binary vessel detector is not penalised for failing to produce
    SYNTAX ids it was never trained to emit. The geometry-only metrics derived
    from this (precision/recall/IoU) measure whether the method finds vessels at
    all, while the label-aware ones measure whether it can also name the segment.
    Separating the two isolates "cannot find the vessel" from "finds it but
    mislabels the segment", which need different fixes.
    """
    return _match_instances_impl(
        gold_instances, pred_instances, width, height, label_aware=False
    )


def _match_instances_impl(
    gold_instances: Sequence[dict[str, Any]],
    pred_instances: Sequence[dict[str, Any]],
    width: int,
    height: int,
    label_aware: bool,
) -> dict[str, Any]:
    """Core matching logic, with label-aware as a switch."""
    gold_masks = [_instance_to_full_mask(g, width, height) for g in gold_instances]
    pred_masks = [_instance_to_full_mask(p, width, height) for p in pred_instances]

    # A single shared bucket makes matching label-agnostic; keying by label
    # confines each assignment to instances that agree on the segment name.
    def bucket_of(inst: dict[str, Any]) -> str:
        return str(inst["label"]) if label_aware else ""

    gold_by_label: dict[str, list[int]] = {}
    for i, inst in enumerate(gold_instances):
        gold_by_label.setdefault(bucket_of(inst), []).append(i)

    pred_by_label: dict[str, list[int]] = {}
    for i, inst in enumerate(pred_instances):
        pred_by_label.setdefault(bucket_of(inst), []).append(i)

    matches: list[dict[str, Any]] = []
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for label, g_idx in gold_by_label.items():
        p_idx = pred_by_label.get(label)
        if not p_idx:
            continue

        iou_matrix = np.zeros((len(g_idx), len(p_idx)), dtype=float)
        for a, gi in enumerate(g_idx):
            for b, pi in enumerate(p_idx):
                iou_matrix[a, b] = _iou(gold_masks[gi], pred_masks[pi])

        # linear_sum_assignment minimises, so negate to maximise IoU.
        for a, b in _hungarian(-iou_matrix):
            if a >= len(g_idx) or b >= len(p_idx):
                continue
            iou = float(iou_matrix[a, b])
            if iou <= 0.0:
                # A zero-overlap pairing is an artefact of assignment needing to
                # fill the matrix, not a detection. Leave both sides unmatched.
                continue
            gi, pi = g_idx[a], p_idx[b]
            matches.append({"gold": gi, "pred": pi, "label": label, "iou": iou})
            matched_gold.add(gi)
            matched_pred.add(pi)

    return {
        "matches": matches,
        "unmatched_gold": sorted(set(range(len(gold_instances))) - matched_gold),
        "unmatched_pred": sorted(set(range(len(pred_instances))) - matched_pred),
    }


# --- scoring -----------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 from counts, with the degenerate cases pinned.

    When there is no gold and no prediction the image is perfectly handled, so
    F1 is 1.0; that keeps a method from being punished for correctly emitting
    nothing. Any other empty side scores 0.0.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _pixel_dice(
    gold_instances: Sequence[dict[str, Any]],
    pred_instances: Sequence[dict[str, Any]],
    width: int,
    height: int,
) -> float:
    """Dice over the union of all instances, ignoring instance identity.

    Flattening every instance into one foreground mask discards the instance
    structure on purpose: it is the only comparison that works when the method
    and the gold disagree about what counts as one object, and it is the number
    directly comparable to a segmentation paper's reported Dice. Two empty masks
    score 1.0, matching the convention used by the volumetric metrics.
    """
    gold_union = np.zeros((height, width), dtype=bool)
    for inst in gold_instances:
        gold_union |= _instance_to_full_mask(inst, width, height)

    pred_union = np.zeros((height, width), dtype=bool)
    for inst in pred_instances:
        pred_union |= _instance_to_full_mask(inst, width, height)

    gold_sum = int(gold_union.sum())
    pred_sum = int(pred_union.sum())
    if gold_sum == 0 and pred_sum == 0:
        return 1.0
    inter = int(np.logical_and(gold_union, pred_union).sum())
    denom = gold_sum + pred_sum
    return (2.0 * inter / denom) if denom else 0.0


def evaluate_instances(
    gold_instances: Sequence[dict[str, Any]],
    pred_instances: Sequence[dict[str, Any]],
    width: int = 512,
    height: int = 512,
    iou_thresholds: Iterable[float] = DEFAULT_IOU_THRESHOLDS,
    headline_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one image's instance list against gold.

    The headline ``f1`` is at ``headline_threshold`` (0.5, the ARCADE/COCO
    convention). ``f1_at_iou_*`` covers the other thresholds so threshold
    sensitivity is visible, and ``mean_matched_iou`` shows delineation quality
    among the instances that were found at all - a method can have good F1 with
    sloppy boundaries, or precise boundaries on the few segments it detects.

    Three families of numbers come back, and which one to read depends on what
    the method claims to do:

    - label-aware (``f1``, ``precision``, ``recall``, ``mean_matched_iou``): the
      headline. A prediction is credited only when it also names the segment.
    - label-agnostic (``*_label_agnostic``): the same geometry with labels
      collapsed. This is what a binary vessel segmenter should be read on, since
      it cannot emit SYNTAX ids and would otherwise report 0 across the board.
    - ``pixel_dice``: matching-free overlap of all foreground, comparable to a
      segmentation paper's own reported Dice.
    """
    matching = match_instances(gold_instances, pred_instances, width, height)
    matches = matching["matches"]

    results: dict[str, Any] = {
        "n_gold_instances": float(len(gold_instances)),
        "n_pred_instances": float(len(pred_instances)),
    }

    thresholds = sorted({float(t) for t in iou_thresholds} | {float(headline_threshold)})
    for thr in thresholds:
        tp = sum(1 for m in matches if m["iou"] >= thr)
        fp = len(pred_instances) - tp
        fn = len(gold_instances) - tp
        precision, recall, f1 = _prf(tp, fp, fn)

        suffix = f"{thr:g}".replace(".", "")
        results[f"precision_at_iou{suffix}"] = precision
        results[f"recall_at_iou{suffix}"] = recall
        results[f"f1_at_iou{suffix}"] = f1

        if thr == float(headline_threshold):
            results["precision"] = precision
            results["recall"] = recall
            results["f1"] = f1
            results["true_positives"] = float(tp)
            results["false_positives"] = float(fp)
            results["false_negatives"] = float(fn)

    ious = [m["iou"] for m in matches]
    results["mean_matched_iou"] = float(np.mean(ious)) if ious else 0.0

    # Label-agnostic geometry, matched again with labels collapsed. Without this a
    # binary vessel segmenter reports 0 everywhere and the run carries no signal
    # at all, which cannot distinguish "segmented the vessels well but cannot name
    # segments" from "found nothing". These keys are the honest way to report such
    # a method on arcade_segmentation; the label-aware keys above remain the
    # headline, since naming the segment is part of the task.
    agnostic = match_instances_ignoring_label(
        gold_instances, pred_instances, width, height
    )
    agnostic_matches = agnostic["matches"]
    tp_a = sum(1 for m in agnostic_matches if m["iou"] >= float(headline_threshold))
    precision_a, recall_a, f1_a = _prf(
        tp_a, len(pred_instances) - tp_a, len(gold_instances) - tp_a
    )
    results["precision_label_agnostic"] = precision_a
    results["recall_label_agnostic"] = recall_a
    results["f1_label_agnostic"] = f1_a
    ious_a = [m["iou"] for m in agnostic_matches]
    results["mean_matched_iou_label_agnostic"] = (
        float(np.mean(ious_a)) if ious_a else 0.0
    )

    # Pixel-level agreement over the union of all instances, which needs no
    # matching at all. This is the one number a binary segmenter can be compared
    # on directly against its own published Dice.
    results["pixel_dice"] = _pixel_dice(gold_instances, pred_instances, width, height)

    # Label accuracy independent of geometry: of the segments gold contains, how
    # many did the method name at all? This separates "cannot find the vessel"
    # from "finds it but mislabels the segment", which need different fixes.
    gold_labels = {str(g["label"]) for g in gold_instances}
    pred_labels = {str(p["label"]) for p in pred_instances}
    if gold_labels:
        results["label_set_recall"] = len(gold_labels & pred_labels) / len(gold_labels)
    else:
        results["label_set_recall"] = 1.0 if not pred_labels else 0.0
    if pred_labels:
        results["label_set_precision"] = len(gold_labels & pred_labels) / len(pred_labels)
    else:
        results["label_set_precision"] = 1.0 if not gold_labels else 0.0

    return results


# --- registry adapters -------------------------------------------------------



