"""
Zero-shot VLM runner for SYNTAX scoring.

One prompt and one parser are shared by every VLM. That is a deliberate
constraint: if each model got a tuned prompt, the table would compare prompt
engineering effort rather than models. The prompt states the task, the anatomy
convention, and the required output format, and nothing else.

Model loading is cached across cases. Reloading a 16GB checkpoint for each of 60
cases would dominate runtime and change nothing about the results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from benchmark.vlms import VLMMethod

from benchmark.core import Prediction
from benchmark.io_spec import (
    VLM_FRAMES_PER_VIEW,
    CaseInput,
    frames_to_pil,
    sample_cine_frames,
)

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
# Held identical for every model. It supplies the clinical framing a cardiologist
# would have, then constrains the output so parsing is unambiguous.

SYNTAX_PROMPT = """You are shown frames from an invasive coronary angiogram of one patient. \
Multiple projections of the left and right coronary arteries are included.

Assess the coronary anatomy and estimate the SYNTAX score for this patient.

The SYNTAX score quantifies the complexity of coronary artery disease. It sums \
weighted contributions from every significant lesion (>=50% diameter stenosis in \
a vessel >=1.5mm), weighting by lesion location, and by adverse features such as \
bifurcation involvement, total occlusion, tortuosity, calcification, and lesion \
length. Typical values: 0 for normal coronaries, 1-22 for low complexity, \
23-32 for intermediate, 33 or above for high complexity.

Report your estimate as a single number on the final line, in exactly this format:

SYNTAX_SCORE: <number>"""


# Cache of loaded models, keyed by (repo_id, device). Loading dominates runtime
# otherwise.
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def _load(method: VLMMethod, device: str) -> tuple[Any, Any]:
    """Load and cache a model + processor."""
    key = (method.repo_id, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import transformers

    snapshot = method._snapshot_dir()
    if snapshot is None:
        raise FileNotFoundError(f"{method.repo_id} not in cache")

    dtype = getattr(torch, method.dtype)
    loader = getattr(transformers, method.loader)

    processor = transformers.AutoProcessor.from_pretrained(
        str(snapshot), trust_remote_code=method.trust_remote_code
    )
    model = loader.from_pretrained(
        str(snapshot),
        dtype=dtype,
        trust_remote_code=method.trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    _MODEL_CACHE[key] = (model, processor)
    return model, processor


def generate_turn(
    method: VLMMethod,
    device: str,
    messages: list[dict[str, Any]],
    images: list[Any] | None = None,
) -> str:
    """Run one generation step and return the decoded continuation.

    Single entry point for every VLM generation in the benchmark, whether the
    caller wants one shot (VLM baselines) or one step of a loop (harnesses).
    Sharing it is a correctness requirement, not tidiness: this repository has
    twice produced silently wrong numbers from duplicated preprocessing
    (CM-UNet 0.726 -> 0.000, CCA 0.536 -> 0.048). A second copy of the
    chat-template / processor / decode sequence would be a third instance of the
    same failure, and a harder one to catch, because harness output is *expected*
    to differ from baseline output, so a divergence would not show up as a
    suspicious number.

    ``messages`` follows the HF chat format. ``images`` must contain one PIL
    image per ``{"type": "image"}`` placeholder in ``messages``; pass None for a
    text-only turn.

    Only the continuation is decoded. The echoed prompt carries the worked
    example used in the ARCADE prompts, and parsing that back would score the
    example rather than the model's answer.
    """
    model, processor = _load(method, device)

    prompt_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    if images:
        inputs = processor(text=[prompt_text], images=images, return_tensors="pt")
    else:
        inputs = processor(text=[prompt_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=method.max_new_tokens,
            do_sample=method.do_sample,
        )

    prompt_length = inputs["input_ids"].shape[1]
    return processor.decode(
        generated[0][prompt_length:], skip_special_tokens=True
    ).strip()


def _parse_score(text: str) -> tuple[float | None, str]:
    """Extract the SYNTAX score from model output.

    Returns (score, parse_note). The note records how the number was found, so a
    result obtained by loose fallback is distinguishable from one the model
    reported in the requested format.
    """
    # Preferred: the requested format.
    match = re.search(r"SYNTAX[_\s]*SCORE\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if match:
        return float(match.group(1)), "explicit"

    # Fallback: last number in the output. Recorded as a weaker parse because it
    # may pick up an unrelated figure.
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1]), "fallback_last_number"

    return None, "no_number_found"


def predict(
    method: VLMMethod,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Prompt a VLM for a SYNTAX score."""
    # Render frames from every available view, left then right.
    images = []
    view_counts = {}
    for artery in ("left", "right"):
        paths = case.views_by_artery.get(artery, [])
        view_counts[artery] = len(paths)
        for path in paths:
            frames = sample_cine_frames(path, VLM_FRAMES_PER_VIEW)
            images.extend(frames_to_pil(frames))

    if not images:
        raise RuntimeError(f"{case.case_id}: no frames could be rendered")

    # Chat template with one image placeholder per rendered frame.
    content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": SYNTAX_PROMPT})
    messages = [{"role": "user", "content": content}]

    # One-shot generation is a single turn through the shared entry point, so the
    # baseline and the harness loop cannot drift apart.
    text = generate_turn(method, device, messages, images)

    score, parse_note = _parse_score(text)
    if score is None:
        raise RuntimeError(
            f"{case.case_id}: no number in model output: {text[:200]!r}"
        )

    # Clamp to the valid SYNTAX range. Out-of-range output is a parsing or
    # reasoning failure; recording the raw value alongside makes it auditable.
    clamped = max(0.0, min(float(score), 200.0))

    return Prediction(
        case_id=case.case_id,
        task=case.task,
        score=clamped,
        diagnostics={
            "n_images": len(images),
            "n_views_left": view_counts.get("left", 0),
            "n_views_right": view_counts.get("right", 0),
            "parse_note": parse_note,
            "raw_score": float(score),
            "clamped": clamped != float(score),
        },
        raw_output=text,
    )
