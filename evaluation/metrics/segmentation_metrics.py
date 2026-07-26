"""
Volumetric segmentation metrics for CardiomniBench-VD.

Used by the CCA 3D coronary-tree segmentation task. Coronary trees are thin,
sparse (~0.1% of voxels) and topologically important, so plain Dice is not
enough:

- Dice          overlap; the headline number, but insensitive to broken branches
- clDice        centreline Dice; penalises topology breaks that Dice forgives
- Hausdorff95   boundary distance at the 95th percentile; robust to outliers
- volume ratio  a sanity check that separates "found the wrong voxels" from
                "found no voxels" (a model can match the volume fraction and
                still have near-zero overlap)

All functions take binary masks as numpy arrays and return plain floats so the
metric registry can consume them directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_binary(mask: np.ndarray) -> np.ndarray:
    """Coerce a mask to a boolean array."""
    return np.asarray(mask) > 0.5


def compute_dice(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Dice coefficient plus the precision/recall that produced it.

    Reporting precision and recall alongside Dice is deliberate: a model that
    over-segments and one that mis-locates can share a Dice score, and only the
    split tells them apart.
    """
    g = _as_binary(gold)
    p = _as_binary(pred)

    intersection = float(np.logical_and(g, p).sum())
    gold_size = float(g.sum())
    pred_size = float(p.sum())

    if gold_size == 0 and pred_size == 0:
        # Empty gold and empty prediction: perfect agreement by convention.
        return {"dice": 1.0, "precision": 1.0, "recall": 1.0, "volume_ratio": 1.0}

    dice = 2.0 * intersection / (gold_size + pred_size) if (gold_size + pred_size) else 0.0
    precision = intersection / pred_size if pred_size else 0.0
    recall = intersection / gold_size if gold_size else 0.0
    volume_ratio = pred_size / gold_size if gold_size else float("inf")

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "volume_ratio": volume_ratio,
    }


def _soft_skeletonize(mask: np.ndarray, iterations: int = 10) -> np.ndarray:
    """Morphological skeleton via iterative thinning.

    Uses scipy's grey erosion/dilation so no extra dependency is needed beyond
    scipy, which the pipeline already requires.
    """
    from scipy import ndimage

    img = _as_binary(mask).astype(np.float32)
    skeleton = np.zeros_like(img)
    eroded = img.copy()

    for _ in range(iterations):
        if eroded.sum() == 0:
            break
        opened = ndimage.grey_dilation(
            ndimage.grey_erosion(eroded, size=(3, 3, 3)), size=(3, 3, 3)
        )
        skeleton = np.maximum(skeleton, np.clip(eroded - opened, 0, 1))
        eroded = ndimage.grey_erosion(eroded, size=(3, 3, 3))

    return skeleton > 0


def compute_cldice(
    gold: np.ndarray, pred: np.ndarray, iterations: int = 10
) -> dict[str, float]:
    """Centreline Dice (clDice).

    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens) where
        Tprec = |skeleton(pred) inside gold| / |skeleton(pred)|
        Tsens = |skeleton(gold) inside pred| / |skeleton(gold)|

    A tubular structure can score reasonable Dice while being fragmented; clDice
    drops sharply in that case, which is why it is the aux metric of record for
    vessel trees.
    """
    g = _as_binary(gold)
    p = _as_binary(pred)

    if g.sum() == 0 and p.sum() == 0:
        return {"cldice": 1.0, "topology_precision": 1.0, "topology_sensitivity": 1.0}
    if g.sum() == 0 or p.sum() == 0:
        return {"cldice": 0.0, "topology_precision": 0.0, "topology_sensitivity": 0.0}

    skeleton_pred = _soft_skeletonize(p, iterations)
    skeleton_gold = _soft_skeletonize(g, iterations)

    pred_skeleton_size = float(skeleton_pred.sum())
    gold_skeleton_size = float(skeleton_gold.sum())

    topology_precision = (
        float(np.logical_and(skeleton_pred, g).sum()) / pred_skeleton_size
        if pred_skeleton_size
        else 0.0
    )
    topology_sensitivity = (
        float(np.logical_and(skeleton_gold, p).sum()) / gold_skeleton_size
        if gold_skeleton_size
        else 0.0
    )

    denominator = topology_precision + topology_sensitivity
    cldice = (
        2.0 * topology_precision * topology_sensitivity / denominator
        if denominator
        else 0.0
    )

    return {
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
    }


def compute_hausdorff(
    gold: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    percentile: float = 95.0,
) -> dict[str, float]:
    """Hausdorff distance in millimetres, at the given percentile and at max.

    The 95th percentile is reported as the headline value because a single
    stray voxel dominates the true maximum.
    """
    from scipy import ndimage

    g = _as_binary(gold)
    p = _as_binary(pred)

    if g.sum() == 0 or p.sum() == 0:
        return {
            f"hausdorff{int(percentile)}_mm": float("inf"),
            "hausdorff_max_mm": float("inf"),
            "mean_surface_distance_mm": float("inf"),
        }

    # Distance from every voxel to the nearest gold / pred voxel.
    distance_to_gold = ndimage.distance_transform_edt(~g, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~p, sampling=spacing)

    # Symmetric surface distances.
    pred_to_gold = distance_to_gold[p]
    gold_to_pred = distance_to_pred[g]
    all_distances = np.concatenate([pred_to_gold, gold_to_pred])

    return {
        f"hausdorff{int(percentile)}_mm": float(np.percentile(all_distances, percentile)),
        "hausdorff_max_mm": float(all_distances.max()),
        "mean_surface_distance_mm": float(all_distances.mean()),
    }


def evaluate_segmentation(
    gold: np.ndarray,
    pred: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_hausdorff: bool = True,
    include_cldice: bool = True,
) -> dict[str, Any]:
    """Run the full segmentation metric suite on one case.

    Hausdorff and clDice are optional because both are expensive on large
    volumes (a 832x832x576 CTA takes tens of seconds each).
    """
    results: dict[str, Any] = {}
    results.update(compute_dice(gold, pred))

    if include_cldice:
        results.update(compute_cldice(gold, pred))
    if include_hausdorff:
        results.update(compute_hausdorff(gold, pred, spacing=spacing))

    g = _as_binary(gold)
    p = _as_binary(pred)
    results["gold_voxels"] = int(g.sum())
    results["pred_voxels"] = int(p.sum())
    results["gold_fraction"] = float(g.mean())
    results["pred_fraction"] = float(p.mean())

    return results


# --- registry adapters -------------------------------------------------------
# The metric registry calls single-value functions with (gold, pred) dicts, so
# each headline metric gets a thin wrapper.


def dice_score(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    """Registry adapter: expects {'mask': ndarray} in both dicts."""
    return compute_dice(gold["mask"], pred["mask"])["dice"]


def cldice_score(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    """Registry adapter for clDice."""
    return compute_cldice(gold["mask"], pred["mask"])["cldice"]


def hausdorff95_mm(gold: dict[str, Any], pred: dict[str, Any]) -> float:
    """Registry adapter for the 95th-percentile Hausdorff distance."""
    spacing = tuple(gold.get("spacing", (1.0, 1.0, 1.0)))
    return compute_hausdorff(gold["mask"], pred["mask"], spacing=spacing)["hausdorff95_mm"]
