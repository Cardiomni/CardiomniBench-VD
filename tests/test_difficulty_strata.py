"""Tests for difficulty-stratified reporting.

The property under test is that difficulty is an *analysis* dimension: it is read
from the case directory when results are summarised, and never reaches a method.
A model told that a case is hard is not being measured on the same task as one
that was not told, so a leak here would silently invalidate every stratified
number in the paper.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from benchmark.results import (
    CaseResult,
    case_difficulty,
    stratify_by_difficulty,
)

TASK_ROOT = Path(__file__).resolve().parents[1] / "data" / "tasks"


def _write_case(directory: Path, case_id: str, difficulty: str | None) -> None:
    """Create a minimal task.yaml, optionally carrying a difficulty label."""
    directory.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {"task_type": "arcade_segmentation"}
    if difficulty is not None:
        metadata["difficulty_level"] = difficulty
    payload = {"case_id": case_id, "case_metadata": metadata}
    with (directory / "task.yaml").open("w") as handle:
        yaml.safe_dump(payload, handle)


def test_reads_difficulty_from_case_metadata(tmp_path: Path) -> None:
    _write_case(tmp_path / "case_a", "case_a", "hard")
    assert case_difficulty(tmp_path / "case_a") == "hard"


def test_missing_difficulty_returns_none_rather_than_a_bucket(tmp_path: Path) -> None:
    """An unlabelled case must not be invented into a bucket."""
    _write_case(tmp_path / "case_b", "case_b", None)
    assert case_difficulty(tmp_path / "case_b") is None


def test_missing_task_yaml_returns_none(tmp_path: Path) -> None:
    assert case_difficulty(tmp_path / "does_not_exist") is None


def test_unlabelled_cases_are_excluded_not_lumped_together() -> None:
    """Cases without a label are dropped from strata, not pooled into one."""
    results = [
        CaseResult("m", "arcade_segmentation", "c1", "ok", {"dice": 0.8}),
        CaseResult("m", "arcade_segmentation", "c2", "ok", {"dice": 0.6}),
        CaseResult("m", "arcade_segmentation", "c3", "ok", {"dice": 0.4}),
    ]
    strata = stratify_by_difficulty(
        results, "m", "arcade_segmentation", {"c1": "easy", "c2": "easy"}
    )
    assert set(strata) == {"easy"}
    assert strata["easy"].metrics["dice"]["n"] == 2


def test_each_bucket_aggregates_only_its_own_cases() -> None:
    results = [
        CaseResult("m", "arcade_segmentation", "c1", "ok", {"dice": 1.0}),
        CaseResult("m", "arcade_segmentation", "c2", "ok", {"dice": 0.0}),
    ]
    strata = stratify_by_difficulty(
        results, "m", "arcade_segmentation", {"c1": "easy", "c2": "hard"}
    )
    assert strata["easy"].metrics["dice"]["mean"] == pytest.approx(1.0)
    assert strata["hard"].metrics["dice"]["mean"] == pytest.approx(0.0)


def test_small_cells_are_reported_with_their_n_not_suppressed() -> None:
    """An n=1 cell must still appear: hiding it would hide the sample size."""
    results = [CaseResult("m", "arcade_segmentation", "c1", "ok", {"dice": 0.5})]
    strata = stratify_by_difficulty(
        results, "m", "arcade_segmentation", {"c1": "hard"}
    )
    assert strata["hard"].metrics["dice"]["n"] == 1
    assert strata["hard"].metrics["dice"]["sd"] == 0.0


def test_other_methods_and_tasks_do_not_leak_into_a_stratum() -> None:
    results = [
        CaseResult("m", "arcade_segmentation", "c1", "ok", {"dice": 1.0}),
        CaseResult("other", "arcade_segmentation", "c1", "ok", {"dice": 0.0}),
        CaseResult("m", "arcade_stenosis", "c1", "ok", {"dice": 0.0}),
    ]
    strata = stratify_by_difficulty(
        results, "m", "arcade_segmentation", {"c1": "easy"}
    )
    assert strata["easy"].metrics["dice"]["n"] == 1
    assert strata["easy"].metrics["dice"]["mean"] == pytest.approx(1.0)


def test_failed_cases_counted_but_excluded_from_metric_means() -> None:
    results = [
        CaseResult("m", "arcade_segmentation", "c1", "ok", {"dice": 0.9}),
        CaseResult("m", "arcade_segmentation", "c2", "failed", {}, error="boom"),
    ]
    strata = stratify_by_difficulty(
        results, "m", "arcade_segmentation", {"c1": "easy", "c2": "easy"}
    )
    summary = strata["easy"]
    assert (summary.n_total, summary.n_ok, summary.n_failed) == (2, 1, 1)
    assert summary.metrics["dice"]["n"] == 1


@pytest.mark.parametrize(
    "task,expected",
    [
        ("arcade_segmentation", {"easy", "medium", "hard"}),
        ("cardiosyntax_scoring", {"low", "intermediate", "high"}),
    ],
)
def test_real_data_uses_per_task_vocabularies(task: str, expected: set[str]) -> None:
    """The two label vocabularies are distinct and are not mapped onto one axis.

    This is the reason stratification reports each task separately: asserting the
    real labels here pins that they genuinely differ, so a future change that
    silently unified them would fail rather than produce a plausible table.
    """
    cases = TASK_ROOT / task / "cases"
    if not cases.is_dir():
        pytest.skip(f"{task} cases not present")
    labels = {
        label
        for case in cases.iterdir()
        if case.is_dir() and (label := case_difficulty(case)) is not None
    }
    assert labels, f"no difficulty labels found for {task}"
    assert labels <= expected, f"unexpected labels for {task}: {labels - expected}"


def test_difficulty_is_absent_from_every_prompt_sent_to_a_method() -> None:
    """The prompts must never mention difficulty.

    Stratification is only meaningful if the label did not influence the answer.
    The VLM prompts are the text that reaches a model on these tasks, so they are
    the place where a leak would appear.
    """
    prompt_sources = {
        "arcade_vlm_runner": ("SEGMENTATION_PROMPT", "STENOSIS_PROMPT"),
        "vlm_runner": ("SYNTAX_PROMPT",),
    }
    runners = Path(__file__).resolve().parents[1] / "benchmark" / "runners"
    banned = ("difficulty", "easy case", "hard case", "difficulty_level")

    checked = 0
    for module, constants in prompt_sources.items():
        # Parsed as source rather than imported: these modules import torch at
        # module level and the rest of the suite runs without it. The prompt is a
        # module-level string literal, so the AST yields the exact text a model
        # receives.
        tree = ast.parse((runners / f"{module}.py").read_text())
        literals = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets
            if isinstance(target, ast.Name) and isinstance(node.value.value, str)
        }
        for name in constants:
            assert name in literals, f"{module}.{name} is not a string literal"
            lowered = literals[name].lower()
            for term in banned:
                assert term not in lowered, (
                    f"{module}.{name} leaks difficulty via {term!r}"
                )
            checked += 1

    assert checked == 3, f"expected to check 3 prompts, checked {checked}"
