"""Regression tests for ARCADE instance-list scoring.

These pin behaviour that was verified by hand during review but had no coverage,
including two bugs that were live: gold masks being silently resampled onto a
reconstructed box, and a binary segmenter having no reportable number at all
because every label-aware metric is 0 by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics.instance_metrics import (
    _bbox_to_xyxy,
    evaluate_instances,
    match_instances,
    match_instances_ignoring_label,
)

W = H = 64


def _inst(label, x0, y0, w, h, mask=None, *, gold=False):
    """Build an instance with both box conventions, mirroring gold on disk."""
    inst = {
        "label": label,
        "bbox_xywh_norm": [x0 / W, y0 / H, w / W, h / H],
        "bbox_xywh_px": [x0, y0, w, h],
    }
    if mask is not None:
        inst["mask"] = mask
        if gold:
            inst["_is_gold"] = True
    return inst


def _solid(w, h):
    return np.ones((h, w), dtype=np.uint8)


# --- box resolution ---------------------------------------------------------


def test_pixel_box_is_preferred_over_normalised():
    """The stored pixel box wins, since the normalised one is lossy.

    Here the two disagree on purpose: the normalised box would round to a
    different width, and trusting it is what silently resized gold masks.
    """
    inst = {
        "bbox_xywh_norm": [0.1004, 0.2, 0.3006, 0.4],  # would round differently
        "bbox_xywh_px": [6, 13, 20, 26],
    }
    assert _bbox_to_xyxy(inst, W, H) == (6, 13, 26, 39)


def test_falls_back_to_normalised_box_when_no_pixel_box():
    """Predictions only carry the normalised box and must still work."""
    inst = {"bbox_xywh_norm": [0.25, 0.25, 0.5, 0.5]}
    assert _bbox_to_xyxy(inst, W, H) == (16, 16, 48, 48)


def test_bare_sequence_still_accepted():
    assert _bbox_to_xyxy([0.0, 0.0, 1.0, 1.0], W, H) == (0, 0, W, H)


def test_box_is_clipped_to_the_image():
    """An out-of-range prediction degrades instead of raising."""
    inst = {"bbox_xywh_norm": [0.9, 0.9, 0.5, 0.5]}
    x0, y0, x1, y1 = _bbox_to_xyxy(inst, W, H)
    assert (x1, y1) == (W, H) and x0 < x1 and y0 < y1


# --- gold is never resampled ------------------------------------------------


def test_gold_mask_shape_mismatch_is_fatal():
    """Resampling gold would distort the reference, so it must raise.

    This is the bug that hid behind a passing gold-vs-gold check: both sides were
    distorted identically and still scored 1.0.
    """
    bad = _inst("1", 0, 0, 10, 10, mask=_solid(9, 9), gold=True)
    with pytest.raises(ValueError, match="gold instance mask shape"):
        evaluate_instances([bad], [], width=W, height=H)


def test_prediction_mask_shape_mismatch_is_resized_not_fatal():
    """A method emitting a mask at its own resolution is reasonable."""
    gold = _inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True)
    pred = _inst("1", 0, 0, 10, 10, mask=_solid(5, 5))  # coarser, same box
    result = evaluate_instances([gold], [pred], width=W, height=H)
    assert result["f1"] == 1.0


# --- matching ---------------------------------------------------------------


def test_perfect_copy_scores_one():
    gold = [
        _inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True),
        _inst("6", 20, 20, 8, 8, mask=_solid(8, 8), gold=True),
    ]
    pred = [
        _inst("1", 0, 0, 10, 10, mask=_solid(10, 10)),
        _inst("6", 20, 20, 8, 8, mask=_solid(8, 8)),
    ]
    result = evaluate_instances(gold, pred, width=W, height=H)
    assert result["f1"] == 1.0
    assert result["mean_matched_iou"] == 1.0
    assert result["pixel_dice"] == 1.0


def test_one_wrong_label_costs_exactly_one_instance():
    """4 gold, one mislabelled -> F1 0.75, the hand-checked value."""
    gold = [_inst(str(i), i * 12, 0, 10, 10, mask=_solid(10, 10), gold=True) for i in range(1, 5)]
    pred = [_inst(str(i), i * 12, 0, 10, 10, mask=_solid(10, 10)) for i in range(1, 5)]
    pred[0]["label"] = "9"  # was "1"
    assert evaluate_instances(gold, pred, width=W, height=H)["f1"] == 0.75


def test_zero_overlap_same_label_is_not_matched():
    """Disjoint boxes must not be paired just because the label agrees."""
    gold = [_inst("1", 0, 0, 8, 8, mask=_solid(8, 8), gold=True)]
    pred = [_inst("1", 40, 40, 8, 8, mask=_solid(8, 8))]
    assert match_instances(gold, pred, W, H)["matches"] == []


def test_duplicate_predictions_are_punished_by_precision():
    """One-to-one matching means spamming boxes cannot inflate recall."""
    gold = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True)]
    pred = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10)) for _ in range(10)]
    result = evaluate_instances(gold, pred, width=W, height=H)
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(0.1)


# --- label-agnostic reporting ----------------------------------------------


def test_binary_segmenter_scores_zero_label_aware_but_nonzero_agnostic():
    """The reason label-agnostic keys exist.

    A geometrically perfect prediction that cannot name SYNTAX segments must
    still produce a readable number, otherwise the run carries no signal and
    "segmented well but cannot name" is indistinguishable from "found nothing".
    """
    gold = [
        _inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True),
        _inst("6", 20, 20, 8, 8, mask=_solid(8, 8), gold=True),
    ]
    pred = [
        _inst("vessel", 0, 0, 10, 10, mask=_solid(10, 10)),
        _inst("vessel", 20, 20, 8, 8, mask=_solid(8, 8)),
    ]
    result = evaluate_instances(gold, pred, width=W, height=H)
    assert result["f1"] == 0.0, "label-aware must still reflect the naming failure"
    assert result["f1_label_agnostic"] == 1.0
    assert result["mean_matched_iou_label_agnostic"] == 1.0
    assert result["pixel_dice"] == 1.0


def test_label_agnostic_matching_crosses_labels():
    gold = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True)]
    pred = [_inst("stenosis", 0, 0, 10, 10, mask=_solid(10, 10))]
    assert match_instances(gold, pred, W, H)["matches"] == []
    assert len(match_instances_ignoring_label(gold, pred, W, H)["matches"]) == 1


# --- degenerate inputs ------------------------------------------------------


def test_empty_both_sides_scores_one():
    """Nothing to find and nothing claimed is agreement, not failure."""
    result = evaluate_instances([], [], width=W, height=H)
    assert result["f1"] == 1.0
    assert result["pixel_dice"] == 1.0


def test_gold_only_scores_zero():
    gold = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True)]
    result = evaluate_instances(gold, [], width=W, height=H)
    assert result["f1"] == 0.0
    assert result["false_negatives"] == 1.0


def test_pred_only_scores_zero():
    pred = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10))]
    result = evaluate_instances([], pred, width=W, height=H)
    assert result["f1"] == 0.0
    assert result["false_positives"] == 1.0


def test_missing_mask_falls_back_to_filled_box():
    """Box-only predictions stay scoreable; the box is treated as solid."""
    gold = [_inst("1", 0, 0, 10, 10, mask=_solid(10, 10), gold=True)]
    pred = [_inst("1", 0, 0, 10, 10)]  # no mask
    assert evaluate_instances(gold, pred, width=W, height=H)["f1"] == 1.0
