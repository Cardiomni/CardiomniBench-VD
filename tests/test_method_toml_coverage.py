"""Every runnable method must declare its pipeline in a TOML.

A method with a runner but no TOML is the exact shape of the bug that made
coronary_unet report Dice 0.021 against an upstream-reported 0.788: the runner
carried its own preprocessing literals, and nothing tied them back to the
release that produced the weights. These tests fail on that arrangement so it
cannot come back quietly through a newly added method.

Methods with ``runner=None`` are declared but not wired up. They are allowed to
have no TOML, since nothing will execute them.
"""

from __future__ import annotations

import pytest

from benchmark.method_config import MethodConfigError, list_method_configs, load_method_config
from benchmark.specialists import ALL_SPECIALISTS


def _runnable_methods() -> list:
    """Specialists that declare a runner, i.e. ones that will actually execute."""
    return [m for m in ALL_SPECIALISTS if getattr(m, "runner", None)]


def test_at_least_one_runnable_method_exists() -> None:
    """Guards against the check below passing because the list came back empty."""
    assert _runnable_methods(), "no specialist declares a runner; the coverage test is vacuous"


@pytest.mark.parametrize("method", _runnable_methods(), ids=lambda m: m.name)
def test_runnable_method_has_toml(method) -> None:
    available = list_method_configs()
    assert method.name in available, (
        f"'{method.name}' declares runner='{method.runner}' but has no "
        f"methods/{method.name}.toml. Its preprocessing would live as literals "
        "inside the runner, which is how the coronary_unet spacing and HU-window "
        "mismatch went unnoticed. Transcribe the upstream pipeline into a TOML."
    )


@pytest.mark.parametrize("method", _runnable_methods(), ids=lambda m: m.name)
def test_runnable_method_toml_loads(method) -> None:
    """The TOML must parse and carry an [inference] table."""
    try:
        config = load_method_config(method.name)
    except MethodConfigError as exc:  # pragma: no cover - failure path
        pytest.fail(f"methods/{method.name}.toml does not load: {exc}")

    assert config.runner == method.runner, (
        f"methods/{method.name}.toml declares runner='{config.runner}' but the "
        f"Python definition says '{method.runner}'; they must agree"
    )
    # 2D single-frame methods have no sliding window: the whole frame is one
    # forward pass, so they declare [decision] instead of [inference].
    if config.image2d_preprocess is not None:
        assert config.decision.get("rule"), (
            f"{method.name}: 2D method must declare [decision] rule"
        )
    else:
        # load_method_config already requires [inference]; assert the parsed shape.
        assert all(v > 0 for v in config.inference.roi_size), (
            f"{method.name}: [inference] roi_size must be positive, got "
            f"{config.inference.roi_size}"
        )


def test_task_literal_matches_task_enum() -> None:
    """benchmark.methods.Task must list exactly the canonical Task enum values.

    Two task vocabularies exist: the Task enum in benchmark.core and the older
    string Literal in benchmark.methods. When they drifted, argparse silently
    refused --tasks arcade_segmentation while the enum, the case data and the
    metrics all supported it, and the failure looked like a bad CLI argument
    rather than a stale list.
    """
    import typing

    from benchmark.core import Task as TaskEnum
    from benchmark.methods import Task as TaskLiteral

    literal_values = set(typing.get_args(TaskLiteral))
    enum_values = {t.value for t in TaskEnum}
    assert literal_values == enum_values, (
        "benchmark.methods.Task and benchmark.core.Task disagree.\n"
        f"  only in Literal: {sorted(literal_values - enum_values)}\n"
        f"  only in enum:    {sorted(enum_values - literal_values)}"
    )


def test_cli_task_choices_cover_every_task() -> None:
    """The runner CLI must offer every task the harness can score."""
    from benchmark.core import Task as TaskEnum
    from benchmark.run_unified import build_parser

    for action in build_parser()._actions:
        if action.dest == "tasks":
            assert set(action.choices) == {t.value for t in TaskEnum}, (
                f"--tasks offers {sorted(action.choices)}, but the harness "
                f"defines {sorted(t.value for t in TaskEnum)}"
            )
            return
    pytest.fail("run_unified has no --tasks argument")


@pytest.mark.parametrize("method", _runnable_methods(), ids=lambda m: m.name)
def test_declared_runner_is_importable_by_name(method) -> None:
    """Every declared runner name must resolve to a real module.

    Dispatch is by name, so a typo in ``runner=`` is not a syntax error: it
    surfaces mid-run as an unavailable method. Resolution is checked here without
    requiring the inference stack, since a missing dependency is a different
    problem from a missing runner.
    """
    from benchmark.specialists import _load_runner_module

    try:
        _load_runner_module(method.runner)
    except ImportError as exc:
        message = str(exc)
        if "not found" in message:
            pytest.fail(f"{method.name}: runner '{method.runner}' does not exist ({exc})")
        # "needs 'torch'" means the module was located; the dependency is absent
        # from this interpreter, which this test does not require.
        pytest.skip(f"runner '{method.runner}' present but needs a missing dependency")


@pytest.mark.parametrize("name", list_method_configs())
def test_shipped_toml_declares_a_provenance_source(name: str) -> None:
    """A pipeline is only auditable if the TOML says where it was copied from."""
    config = load_method_config(name)
    source = config.provenance.get("source")
    assert source, (
        f"methods/{name}.toml has no [method.provenance] source. Without it the "
        "parameters cannot be checked against the release that trained the weights."
    )


@pytest.mark.parametrize("name", list_method_configs())
def test_weights_path_is_declared_and_exists(name: str) -> None:
    """Catch a TOML pointing at a checkpoint that is not on disk."""
    config = load_method_config(name)
    if config.weights_path is None:
        pytest.skip(f"{name} declares no weights_path")
    assert config.weights_path.exists(), (
        f"methods/{name}.toml points at {config.weights_path}, which does not exist"
    )
