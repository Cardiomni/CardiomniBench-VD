"""Regression tests for 2D XCA intensity preprocessing.

These pin the fix for a silent failure: CM-UNet was configured with
``normalize = false``, transcribed faithfully from ``Finetuning/dataset.py``,
which applies no intensity step. But that dataset reads ``.npy`` files already
put through unsharp masking and per-image z-score by upstream's offline
notebook. Raw uint8 input produced an empty mask and a clean Dice of 0.000,
indistinguishable from a checkpoint that simply does not transfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.method_config import NORMALIZE_MODES, Image2DPreprocessConfig
from benchmark.runners.cm_unet_runner import _apply_intensity


# --- config parsing ---------------------------------------------------------


def test_normalize_defaults_to_none() -> None:
    """Absent means absent; the runner must not invent a scheme."""
    config = Image2DPreprocessConfig.from_toml({"model_input": 256}, "test")
    assert config.normalize == "none"
    assert config.unsharp_radius == 0.0


def test_named_modes_round_trip() -> None:
    for mode in NORMALIZE_MODES:
        config = Image2DPreprocessConfig.from_toml(
            {"model_input": 256, "normalize": mode}, "test"
        )
        assert config.normalize == mode


def test_unknown_mode_raises_rather_than_falling_back() -> None:
    """A typo must fail loudly; silently ignoring it is how this bug survived."""
    with pytest.raises(ValueError, match="normalize="):
        Image2DPreprocessConfig.from_toml(
            {"model_input": 256, "normalize": "zscore_v2"}, "test"
        )


def test_legacy_booleans_still_load() -> None:
    """Older TOMLs used booleans; true maps to upstream's actual scheme."""
    assert (
        Image2DPreprocessConfig.from_toml(
            {"model_input": 256, "normalize": True}, "test"
        ).normalize
        == "zscore"
    )
    assert (
        Image2DPreprocessConfig.from_toml(
            {"model_input": 256, "normalize": False}, "test"
        ).normalize
        == "none"
    )


def test_unsharp_parameters_are_read() -> None:
    config = Image2DPreprocessConfig.from_toml(
        {"model_input": 256, "unsharp_radius": 60, "unsharp_amount": 3}, "test"
    )
    assert (config.unsharp_radius, config.unsharp_amount) == (60.0, 3.0)


# --- intensity transforms ---------------------------------------------------


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 32)).astype(np.float32)


def test_none_is_a_true_passthrough(frame: np.ndarray) -> None:
    assert np.array_equal(_apply_intensity(frame, "none", 0, 0), frame)


def test_divide255_maps_into_unit_range(frame: np.ndarray) -> None:
    out = _apply_intensity(frame, "divide255", 0, 0)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert out.max() == pytest.approx(frame.max() / 255.0)


def test_zscore_gives_zero_mean_unit_variance(frame: np.ndarray) -> None:
    """The property the network was trained to expect."""
    out = _apply_intensity(frame, "zscore", 0, 0)
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-5)


def test_zscore_is_per_image_not_corpus(frame: np.ndarray) -> None:
    """Upstream fits mean/std per sample, so a rescaled copy normalises alike.

    This is what lets the scheme transfer to ARCADE without dataset statistics.
    """
    a = _apply_intensity(frame, "zscore", 0, 0)
    b = _apply_intensity(frame * 3.0 + 10.0, "zscore", 0, 0)
    assert np.allclose(a, b, atol=1e-4)


def test_minmax_spans_exactly_zero_to_one(frame: np.ndarray) -> None:
    out = _apply_intensity(frame, "minmax", 0, 0)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_constant_frame_does_not_produce_nans() -> None:
    """Zero variance must not divide by zero and poison the network input."""
    flat = np.full((8, 8), 42.0, dtype=np.float32)
    for mode in ("minmax", "zscore"):
        out = _apply_intensity(flat, mode, 0, 0)
        assert np.isfinite(out).all(), mode


def test_unsharp_runs_before_normalisation(frame: np.ndarray) -> None:
    """Order matters: sharpening then standardising is upstream's order.

    If normalisation ran first the result would differ, so this asserts the two
    orders are distinguishable and that we produce the upstream one.
    """
    ours = _apply_intensity(frame, "zscore", 60, 3)
    reversed_order = _apply_intensity(
        _apply_intensity(frame, "zscore", 0, 0), "none", 60, 3
    )
    assert not np.allclose(ours, reversed_order, atol=1e-3)
    assert ours.mean() == pytest.approx(0.0, abs=1e-5)


def test_unsharp_is_skipped_when_amount_is_zero(frame: np.ndarray) -> None:
    """radius without amount is a no-op, so it must not silently sharpen."""
    assert np.array_equal(_apply_intensity(frame, "none", 60, 0), frame)


def test_unknown_mode_raises_in_the_transform_too() -> None:
    with pytest.raises(ValueError, match="unknown normalize mode"):
        _apply_intensity(np.zeros((4, 4), dtype=np.float32), "bogus", 0, 0)


def test_input_is_not_mutated(frame: np.ndarray) -> None:
    """The caller's array is reused for other variants in ablations."""
    original = frame.copy()
    _apply_intensity(frame, "zscore", 60, 3)
    assert np.array_equal(frame, original)


# --- the shipped configuration ---------------------------------------------


def test_shipped_cm_unet_tomls_declare_upstream_intensity_pipeline() -> None:
    """Guard against a revert to raw uint8, which scores 0.000 on ARCADE."""
    from benchmark.method_config import load_method_config

    for name in ("coronary_cm_unet", "coronary_cm_unet_native"):
        config = load_method_config(name)
        assert config.image2d_preprocess is not None, name
        assert config.image2d_preprocess.normalize == "zscore", name
        assert config.image2d_preprocess.unsharp_radius == 60, name
        assert config.image2d_preprocess.unsharp_amount == 3, name
