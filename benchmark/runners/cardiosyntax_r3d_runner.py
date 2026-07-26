"""
CardioSYNTAX R3D+LSTM runner.

Delegates the checkpoint-specific details to the verified upstream helpers in
algorithms/baselines/cardiosyntax_r3d_agent.py rather than reimplementing them.
Those helpers encode two non-obvious facts that are easy to get wrong:

  - the regression head emits log1p(score), so expm1 is mandatory on output
  - the per-fold linear calibration applies to the left+right SUM, not to each
    artery separately

This wrapper adapts them to the typed Method/Prediction interface and supports
evaluating either the full ensemble or a single fold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from benchmark.specialists import CardioSYNTAXRegressor

from benchmark.core import Prediction
from benchmark.io_spec import CaseInput

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = _REPO_ROOT / "algorithms" / "baselines"
if str(_BASELINES) not in sys.path:
    sys.path.insert(0, str(_BASELINES))


def predict(
    method: CardioSYNTAXRegressor,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Run SYNTAX score regression and return a Prediction."""
    from cardiosyntax_r3d_agent import (
        aggregate_folds,
        discover_checkpoints,
        load_scaling,
        predict_side,
    )

    checkpoints = discover_checkpoints(method.weights_path)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {method.weights_path}")

    # Either one fold or the released ensemble.
    if method.single_fold is not None:
        folds = [method.single_fold]
    else:
        folds = list(method.folds or ())
    if not folds:
        raise ValueError(f"{method.name}: no folds selected")

    scaling = (
        load_scaling(method.weights_path.parent) if method.calibrated else {}
    )
    torch_device = torch.device(device)

    left_folds = predict_side(
        case.views_by_artery.get("left", []),
        checkpoints.get("left", {}),
        folds,
        torch_device,
    )
    right_folds = predict_side(
        case.views_by_artery.get("right", []),
        checkpoints.get("right", {}),
        folds,
        torch_device,
    )

    total, left, right = aggregate_folds(
        left_folds, right_folds, folds, scaling, method.calibrated
    )

    if total is None:
        raise RuntimeError(
            f"{case.case_id}: every fold failed "
            f"(left={len(left_folds)}, right={len(right_folds)} succeeded)"
        )

    components = {}
    if left is not None:
        components["syntax_left"] = float(left)
    if right is not None:
        components["syntax_right"] = float(right)

    # Per-fold spread is worth keeping: it separates a confident ensemble from
    # one whose folds disagree, which a single averaged number hides.
    fold_totals = [
        (left_folds.get(f) or 0.0) + (right_folds.get(f) or 0.0) for f in folds
    ]
    diagnostics = {
        "folds_used": folds,
        "folds_succeeded_left": sorted(left_folds),
        "folds_succeeded_right": sorted(right_folds),
        "calibrated": method.calibrated,
        "fold_spread": float(np.std(fold_totals)) if len(fold_totals) > 1 else 0.0,
        "n_views_left": len(case.views_by_artery.get("left", [])),
        "n_views_right": len(case.views_by_artery.get("right", [])),
    }

    return Prediction(
        case_id=case.case_id,
        task=case.task,
        score=float(total),
        components=components,
        diagnostics=diagnostics,
    )
