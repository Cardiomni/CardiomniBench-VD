"""Zero-shot VLM runner for the ARCADE tasks.

Why bbox and not masks
----------------------
A VLM cannot emit a segmentation mask from text output. Wiring one to SAM would
measure the pair, not the VLM. ARCADE gold carries ``bbox_xywh_px`` for every
instance, and ``evaluation/metrics/instance_metrics.py`` falls back to the filled
box when an instance has no mask, so a box-only prediction is scoreable as-is and
is recorded as ``mask_source="bbox"`` rather than passed off as segmentation.

What this measures that CM-UNet cannot
--------------------------------------
CM-UNet emits one binary vessel class, so every label-aware metric on
arcade_segmentation is 0 by construction and no model on disk provides an upper
bound for *naming* SYNTAX segments. A prompted VLM does name them, which is the
capability Cardiomni Stage 2 is built around.

Model loading, dtype and the loader class are reused from ``vlm_runner`` so the
two runners cannot drift apart on how a checkpoint is instantiated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.vlms import VLMMethod

from benchmark.core import Prediction, Task
from benchmark.io_spec import CaseInput

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
# One prompt per task, shared by every model, for the same reason vlm_runner
# holds its prompt fixed: otherwise the table compares prompt engineering.

SEGMENTATION_PROMPT = """You are shown one frame from an invasive coronary angiogram (XCA).

Identify every coronary artery segment that is visible and give its location.

Use the SYNTAX segment numbering:
- RCA: 1 proximal, 2 mid, 3 distal, 4 PDA, 16/16a/16b/16c posterolateral
- Left main: 5
- LAD: 6 proximal, 7 mid, 8 distal, 9/9a diagonal, 10/10a septal
- LCX: 11 proximal, 12/12a/12b intermediate and obtuse marginal, 13 distal, \
14/14a/14b posterolateral, 15 PDA when left-dominant

Only one coronary side is injected in a given frame, so report only the segments \
you can actually see. Do not list segments from the other side.

Give the location as a bounding box in normalised coordinates, where (0,0) is the \
top-left corner of the image and (1,1) is the bottom-right.

Report one segment per line, in exactly this format, and write nothing else:
SEGMENT: <id> [x_min, y_min, x_max, y_max]

Each of the four numbers is a decimal between 0 and 1 that you must measure from \
this image. Give each segment its own box: two segments that occupy different parts \
of the vessel tree must not share the same coordinates."""

STENOSIS_PROMPT = """You are shown one frame from an invasive coronary angiogram (XCA).

Find every stenosis: a segment of artery that is visibly narrowed compared with \
the healthy vessel on either side of it. This frame contains at least two.

Give the location of each stenosis as a bounding box in normalised coordinates, \
where (0,0) is the top-left corner of the image and (1,1) is the bottom-right. Box \
the narrowed part of the vessel, not the whole artery.

Report one stenosis per line, in exactly this format, and write nothing else:
STENOSIS: [x_min, y_min, x_max, y_max]

Each of the four numbers is a decimal between 0 and 1 that you must measure from \
this image. Give each stenosis its own box: two stenoses at different points along \
the vessel must not share the same coordinates."""

#: Label space ARCADE grades against. A model may emit "LAD" or "7." or "segment
#: 7"; only ids in this set are kept, so a hallucinated label becomes a missing
#: instance rather than a silent false match against a name gold never uses.
SYNTAX_SEGMENT_IDS = frozenset(
    [
        "1", "2", "3", "4", "5", "6", "7", "8",
        "9", "9a", "10", "10a", "11", "12", "12a", "12b",
        "13", "14", "14a", "14b", "15", "16", "16a", "16b", "16c",
    ]
)

_BOX = r"\[\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\]"
# The id group accepts any short token, not just well-formed SYNTAX ids, so that a
# model answering "SEGMENT: LAD [...]" or "SEGMENT: 99 [...]" is counted as a
# rejected label instead of silently failing to match: a run with 0 instances and
# 0 rejects means something different from one with 0 instances and 8 rejects.
_SEGMENT_LINE = re.compile(r"SEGMENT\s*:\s*([A-Za-z0-9]{1,6})\s*" + _BOX, re.I)
_STENOSIS_LINE = re.compile(r"STENOSIS\s*:\s*" + _BOX, re.I)


def _to_xywh_norm(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> list[float] | None:
    """Convert a clamped xyxy box to xywh, or None if it is degenerate.

    Models sometimes emit coordinates slightly outside [0,1], or with the corners
    swapped. Both are recoverable and are repaired here; a box with no area is
    dropped, because scoring it would credit or penalise a non-answer.
    """
    x_min, x_max = sorted((x_min, x_max))
    y_min, y_max = sorted((y_min, y_max))
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(1.0, x_max), min(1.0, y_max)
    width, height = x_max - x_min, y_max - y_min
    if width <= 0.0 or height <= 0.0:
        return None
    return [x_min, y_min, width, height]


def _parse_segments(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse ``SEGMENT: <id> [box]`` lines into gradeable instances."""
    instances: list[dict[str, Any]] = []
    rejected_label = 0
    rejected_box = 0
    seen: set[tuple[str, tuple[float, ...]]] = set()

    for match in _SEGMENT_LINE.finditer(text):
        label = match.group(1).lower()
        if label not in SYNTAX_SEGMENT_IDS:
            rejected_label += 1
            continue
        box = _to_xywh_norm(*(float(match.group(i)) for i in range(2, 6)))
        if box is None:
            rejected_box += 1
            continue
        # A repeated identical line is one answer stated twice, not two findings.
        key = (label, tuple(round(v, 4) for v in box))
        if key in seen:
            continue
        seen.add(key)
        instances.append({"label": label, "bbox_xywh_norm": box})

    return instances, {
        "rejected_unknown_label": rejected_label,
        "rejected_degenerate_box": rejected_box,
    }


def _parse_stenoses(text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse ``STENOSIS: [box]`` lines into gradeable instances."""
    instances: list[dict[str, Any]] = []
    rejected_box = 0
    seen: set[tuple[float, ...]] = set()

    for match in _STENOSIS_LINE.finditer(text):
        box = _to_xywh_norm(*(float(match.group(i)) for i in range(1, 5)))
        if box is None:
            rejected_box += 1
            continue
        key = tuple(round(v, 4) for v in box)
        if key in seen:
            continue
        seen.add(key)
        instances.append({"label": "stenosis", "bbox_xywh_norm": box})

    return instances, {"rejected_degenerate_box": rejected_box}


def predict(
    method: VLMMethod,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Prompt a VLM to localise segments or stenoses in one XCA frame."""
    from PIL import Image

    # Imported here, not at module scope: vlm_runner imports torch eagerly, and a
    # module-level import would make the parsers below untestable on a machine
    # without torch installed. The parsing logic is the part that needs unit tests.
    from benchmark.runners.vlm_runner import generate_turn

    image_path = case.case_dir / "image.png"
    if not image_path.exists():
        raise FileNotFoundError(f"{case.case_id}: no image.png in {case.case_dir}")

    if case.task is Task.ARCADE_SEGMENTATION:
        prompt = SEGMENTATION_PROMPT
    elif case.task is Task.ARCADE_STENOSIS:
        prompt = STENOSIS_PROMPT
    else:
        raise ValueError(f"{method.name}: arcade_vlm runner cannot serve {case.task}")

    with Image.open(image_path) as handle:
        # Greyscale XCA to RGB: the processors expect 3 channels, and replicating
        # the single channel adds no information the model did not have.
        image = handle.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        reply = generate_turn(method, device, messages, [image])

    if case.task is Task.ARCADE_SEGMENTATION:
        instances, rejects = _parse_segments(reply)
    else:
        instances, rejects = _parse_stenoses(reply)

    diagnostics: dict[str, Any] = {
        "n_instances": len(instances),
        "mask_source": "bbox",
        **rejects,
    }
    if instances and case.task is Task.ARCADE_SEGMENTATION:
        diagnostics["labels_predicted"] = sorted({i["label"] for i in instances})

    # Raw output kept in full: a zero-instance case is usually a format failure
    # rather than the model claiming a normal coronary tree, and only the text
    # says which. The Prediction.raw_output field exists for exactly this.
    return Prediction(
        case_id=case.case_id,
        task=case.task,
        instances=instances,
        diagnostics=diagnostics,
        raw_output=reply,
    )
