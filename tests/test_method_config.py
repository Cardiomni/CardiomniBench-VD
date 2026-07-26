"""Tests for TOML-backed method configuration.

The point of these tests is to protect the parameters that decide whether a
checkpoint produces a usable mask. A silent default is what let coronary_unet
run with a -200..300 HU window at the wrong voxel spacing and report Dice 0.021
against an upstream-reported 0.788, so the loader is expected to raise on a
missing pipeline parameter rather than substitute a plausible value.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from benchmark.method_config import (
    MethodConfigError,
    list_method_configs,
    load_method_config,
)

# Values transcribed from the upstream release, restated here so a drifting TOML
# fails a test instead of quietly changing what the benchmark measures. Source:
# github.com/noahschuetz/coronary-artery-segmentation src/data/transforms.py
UPSTREAM_UNET = {
    "pixdim": (1.0, 1.0, 1.0),
    "window": (-260.0, 760.0),
    "normalize_mode": "minmax",
    "body_threshold_hu": -500.0,
    "roi_size": (96, 96, 96),
    "threshold": 0.5,
    "overlap": 0.5,
}


def test_shipped_methods_are_discoverable() -> None:
    names = list_method_configs()
    assert "coronary_unet" in names
    assert "coronary_nnunet" in names


def test_coronary_unet_matches_upstream_pipeline() -> None:
    config = load_method_config("coronary_unet")
    pre = config.preprocess
    assert pre is not None, "coronary_unet needs an explicit [preprocess] table"

    assert pre.pixdim == UPSTREAM_UNET["pixdim"]
    assert (pre.window_a_min, pre.window_a_max) == UPSTREAM_UNET["window"]
    assert pre.normalize_mode == UPSTREAM_UNET["normalize_mode"]
    assert pre.body_crop is True
    assert pre.body_threshold_hu == UPSTREAM_UNET["body_threshold_hu"]
    # Upstream leaves TV denoising on by default in the validation chain.
    assert pre.denoise_tv is True

    inference = config.inference
    assert inference.roi_size == UPSTREAM_UNET["roi_size"]
    assert inference.overlap == UPSTREAM_UNET["overlap"]
    # Upstream thresholds the foreground probability; argmax is a different rule.
    assert inference.decision_rule == "threshold"
    assert inference.threshold == UPSTREAM_UNET["threshold"]


def test_baseline_unet_is_single_channel() -> None:
    """baseline_unet takes CT only; the Frangi channel belongs to att_mamba2."""
    pre = load_method_config("coronary_unet").preprocess
    assert pre is not None
    assert pre.vesselness is False
    assert pre.in_channels == 1


def test_nnunet_defers_preprocessing_to_plans_json() -> None:
    """plans.json is already a machine-readable contract, so it stays the source."""
    config = load_method_config("coronary_nnunet")
    assert config.preprocess is None, (
        "nnU-Net preprocessing must come from plans.json, not be duplicated in TOML"
    )
    assert "plans_path" in config.extra_paths
    assert config.extra_paths["plans_path"].name == "plans.json"
    # The reference patch size is still recorded so the TOML documents the model.
    assert config.inference.roi_size == (96, 160, 160)
    assert config.inference.decision_rule == "argmax"


def test_weights_path_resolves_against_repo_root() -> None:
    config = load_method_config("coronary_unet")
    assert config.weights_path is not None
    assert config.weights_path.is_absolute()
    assert config.weights_path.name == "baseline_unet.pth"


def test_unknown_method_names_the_expected_file(tmp_path: Path) -> None:
    with pytest.raises(MethodConfigError) as excinfo:
        load_method_config("no_such_method", methods_dir=tmp_path)
    assert "no_such_method.toml" in str(excinfo.value)


def test_missing_pipeline_parameter_raises(tmp_path: Path) -> None:
    """A partial [preprocess] table must fail loudly, not fall back to a guess."""
    (tmp_path / "partial.toml").write_text(
        textwrap.dedent(
            """
            [method]
            name = "partial"
            family = "specialist"
            tasks = ["cca_segmentation"]
            runner = "monai_unet"

            [preprocess]
            pixdim = [1.0, 1.0, 1.0]
            orientation = "RAS"
            # window_a_min / window_a_max / normalize_mode deliberately absent

            [inference]
            roi_size = [96, 96, 96]
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(MethodConfigError) as excinfo:
        load_method_config("partial", methods_dir=tmp_path)
    assert "window_a_min" in str(excinfo.value)


def test_inference_table_is_required(tmp_path: Path) -> None:
    (tmp_path / "no_inference.toml").write_text(
        textwrap.dedent(
            """
            [method]
            name = "no_inference"
            family = "specialist"
            tasks = ["cca_segmentation"]
            runner = "monai_unet"

            [preprocess]
            pixdim = [1.0, 1.0, 1.0]
            orientation = "RAS"
            window_a_min = -260.0
            window_a_max = 760.0
            normalize_mode = "minmax"
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(MethodConfigError) as excinfo:
        load_method_config("no_inference", methods_dir=tmp_path)
    assert "[inference]" in str(excinfo.value)


def test_vesselness_implies_two_input_channels(tmp_path: Path) -> None:
    """att_mamba2 stacks Frangi vesselness onto the CT channel."""
    (tmp_path / "two_channel.toml").write_text(
        textwrap.dedent(
            """
            [method]
            name = "two_channel"
            family = "specialist"
            tasks = ["cca_segmentation"]
            runner = "monai_unet"

            [preprocess]
            pixdim = [1.0, 1.0, 1.0]
            orientation = "RAS"
            window_a_min = -260.0
            window_a_max = 760.0
            normalize_mode = "minmax"
            vesselness = true
            vesselness_keep_original = true

            [inference]
            roi_size = [96, 96, 96]
            """
        ),
        encoding="utf-8",
    )
    pre = load_method_config("two_channel", methods_dir=tmp_path).preprocess
    assert pre is not None
    assert pre.in_channels == 2
