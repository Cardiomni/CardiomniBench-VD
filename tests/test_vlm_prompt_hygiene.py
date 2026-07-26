"""Guards against literal coordinate values leaking into VLM prompts.

The ARCADE prompts originally illustrated their output format with concrete
boxes::

    SEGMENT: 1 [0.21, 0.17, 0.40, 0.34]
    SEGMENT: 2 [0.24, 0.42, 0.47, 0.63]

Two unrelated models (llava_16_mistral_7b, llama3_llava_next_8b) echoed those
numbers back verbatim for every image. An ablation that removed the examples
produced image-dependent boxes, confirming the prompt was supplying the answer
and confounding the measurement.

A format example is therefore only safe if it contains no parseable coordinate.
These tests read the prompts as source constants so they fail on the literal text
rather than on a model's behaviour, which keeps them offline and deterministic.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "benchmark" / "runners" / "arcade_vlm_runner.py"

#: Any decimal in [0,1) is a plausible normalised coordinate a model could copy.
DECIMAL = re.compile(r"\b0\.\d+\b")


def _prompt_constants() -> dict[str, str]:
    """Extract module-level string constants whose name contains PROMPT."""
    tree = ast.parse(RUNNER.read_text())
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for target in node.targets:
            name = getattr(target, "id", "")
            if "PROMPT" in name:
                found[name] = node.value.value
    return found


def test_prompt_constants_are_discoverable() -> None:
    """If this fails the other tests are vacuous, so assert it separately."""
    prompts = _prompt_constants()
    assert "SEGMENTATION_PROMPT" in prompts
    assert "STENOSIS_PROMPT" in prompts


@pytest.mark.parametrize("name", ["SEGMENTATION_PROMPT", "STENOSIS_PROMPT"])
def test_prompt_contains_no_literal_coordinates(name: str) -> None:
    """No copyable decimal may appear anywhere in a localisation prompt."""
    text = _prompt_constants()[name]
    leaked = DECIMAL.findall(text)
    assert not leaked, (
        f"{name} contains literal coordinate values {leaked}; models copy these "
        "verbatim instead of reading the image. Use placeholders such as "
        "[x_min, y_min, x_max, y_max] instead."
    )


@pytest.mark.parametrize("name", ["SEGMENTATION_PROMPT", "STENOSIS_PROMPT"])
def test_prompt_still_specifies_the_output_schema(name: str) -> None:
    """Removing examples must not remove the format contract the parser needs."""
    text = _prompt_constants()[name]
    assert "x_min" in text and "y_max" in text, (
        f"{name} no longer states the coordinate schema; the output parser "
        "depends on the model knowing the field order."
    )


@pytest.mark.parametrize(
    ("name", "keyword"),
    [("SEGMENTATION_PROMPT", "SEGMENT:"), ("STENOSIS_PROMPT", "STENOSIS:")],
)
def test_prompt_retains_its_line_prefix(name: str, keyword: str) -> None:
    """The per-line prefix is what the regex parser keys on."""
    assert keyword in _prompt_constants()[name]


@pytest.mark.parametrize("name", ["SEGMENTATION_PROMPT", "STENOSIS_PROMPT"])
def test_prompt_forbids_reusing_one_box_for_everything(name: str) -> None:
    """The observed degenerate mode is one box repeated for every item.

    The prompt should at least ask for distinct boxes, so that a model emitting
    identical coordinates is disobeying an explicit instruction rather than
    filling a gap the prompt left open.
    """
    text = _prompt_constants()[name].lower()
    assert "same coordinates" in text or "its own box" in text, (
        f"{name} does not tell the model that distinct findings need distinct "
        "boxes; without it, one repeated box is an unpenalised reading."
    )
