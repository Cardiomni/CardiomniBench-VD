"""
Vision-language model definitions for CardiomniBench-VD.

A VLM here is a general-purpose or medical multimodal model prompted zero-shot.
None of them were trained for SYNTAX scoring or coronary segmentation, so these
rows measure something different from the specialist rows: whether a model that
has read about coronary anatomy can apply that knowledge to images it has never
been trained on.

That distinction is why `domain_relation` is NOT_TRAINED for every entry, and why
the results table separates these from specialists rather than ranking them in
one list.

What a VLM can and cannot do here
---------------------------------
SYNTAX scoring is answerable in principle: the model sees angiographic frames and
returns a number. It will be bad at it, and that is a finding.

The 2D ARCADE tasks are answerable as *localisation*: the reply is a list of boxes,
for segmentation with a SYNTAX segment id attached. That is weaker than a mask and
is scored as such (see runners/arcade_vlm_runner.py), but it is the only thing on
disk that names segments at all - CM-UNet emits one binary vessel class - so it is
the only available reference for the naming step.

Dense 3D segmentation is still not answerable by a text-output model. Rather than
fabricate a mask, cca_segmentation is simply not offered to VLMs; the `tasks`
tuple omits it, so the harness never asks. A blank row in the results table is
more honest than a zero produced by a model that was never asked a well-posed
question.

Cache state
-----------
Weights live in the shared HF cache. Several repos are partially downloaded, so
`check_available` verifies every shard in the weight map actually resolves rather
than trusting the directory's existence. A partially-fetched model that loads
half its layers would silently produce noise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from benchmark.core import (
    DomainRelation,
    Family,
    Method,
    Prediction,
    Provenance,
    Task,
)
from benchmark.io_spec import CaseInput

HF_CACHE = Path(
    os.environ.get("HF_HOME", "/mnt/aliyunsb/Cardiomni/hf_cache")
) / "hub"


@dataclass
class VLMMethod(Method):
    """A prompted vision-language model.

    No training, no fine-tuning: the model sees rendered images plus a text
    instruction and its reply is parsed. Prompt and parser are shared across all
    VLMs (see runners/vlm_runner.py for scoring, runners/arcade_vlm_runner.py for
    the ARCADE tasks) so differences between rows reflect the models rather than
    per-model prompt engineering.
    """

    family: ClassVar[Family] = Family.VLM

    repo_id: str = ""
    """HuggingFace repo, also the cache key."""

    revision: str = ""
    """Pinned upstream commit SHA of the weights actually evaluated.

    HF repos are mutable: upstream can re-upload weights under the same name, so
    an unpinned row is not reproducible. This records the snapshot the reported
    numbers came from. Empty means unpinned, which ``check_available`` reports so
    it cannot pass unnoticed.
    """

    #: Loader class name in transformers. AutoModelForImageTextToText covers all
    #: currently-usable candidates; kept explicit so an exception is visible.
    loader: str = "AutoModelForImageTextToText"

    trust_remote_code: bool = False

    #: Generation is deterministic: temperature 0 equivalent. A benchmark number
    #: that changes between runs is not a measurement.
    max_new_tokens: int = 512
    do_sample: bool = False

    #: dtype for loading. bfloat16 fits a 7-8B model comfortably on one H20.
    dtype: str = "bfloat16"

    def _snapshot_dir(self) -> Path | None:
        """Locate the cached snapshot directory, if present."""
        repo_dir = HF_CACHE / f"models--{self.repo_id.replace('/', '--')}"
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            return None
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
        if not candidates:
            return None
        if self.revision:
            # Prefer the pinned revision; fall back so a mismatch is reported by
            # check_available rather than silently looking absent.
            pinned = snapshots / self.revision
            if pinned.is_dir():
                return pinned
        return candidates[0]

    def check_available(self) -> tuple[bool, str]:
        """Verify the config loads and every weight shard resolves.

        Checking the shard index rather than just the directory is deliberate:
        an interrupted download leaves a complete-looking snapshot whose tensors
        are missing, and that failure would otherwise appear as garbage output
        rather than an error.
        """
        snapshot = self._snapshot_dir()
        if snapshot is None:
            return False, f"not in cache: {self.repo_id}"

        config = snapshot / "config.json"
        if not config.exists():
            return False, "config.json missing"

        # Resolve the shard list from whichever index format is present.
        index_names = (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        shards: set[str] = set()
        declared_total: int | None = None
        for index_name in index_names:
            index_path = snapshot / index_name
            if index_path.exists():
                with index_path.open() as handle:
                    index = json.load(handle)
                weight_map = index.get("weight_map", {})
                shards.update(weight_map.values())
                size = (index.get("metadata") or {}).get("total_size")
                if isinstance(size, int):
                    declared_total = size
                break
        else:
            # Single-file model: accept either format.
            for single in ("model.safetensors", "pytorch_model.bin"):
                if (snapshot / single).exists():
                    shards.add(single)
                    break
            else:
                return False, "no weight files or shard index found"

        missing = [s for s in sorted(shards) if not (snapshot / s).exists()]
        if missing:
            return (
                False,
                f"{len(missing)}/{len(shards)} shards missing "
                f"(first: {missing[0]})",
            )

        # An .incomplete blob means a download is still in flight.
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_dir():
            incomplete = list(blobs.glob("*.incomplete"))
            if incomplete:
                return (
                    False,
                    f"{len(incomplete)} blob(s) still downloading",
                )

        total = sum(
            os.path.getsize(os.path.realpath(snapshot / s))
            for s in shards
            if (snapshot / s).exists()
        )

        # Compare against the size the index declares. Every shard can exist and
        # still be short: an interrupted download leaves a truncated file whose
        # safetensors header will not deserialise, which surfaces only when the
        # model is loaded. Lingshu-7B passed the existence check and then failed
        # all 171 cases with "Error while deserializing header: incomplete
        # metadata", so presence is verified by byte count rather than by stat.
        # The tolerance absorbs the metadata bytes the index does not count.
        if declared_total is not None:
            shortfall = declared_total - total
            if shortfall > 50 * 1024 * 1024:
                return (
                    False,
                    f"truncated download: {total / 1e9:.1f}GB on disk vs "
                    f"{declared_total / 1e9:.1f}GB declared "
                    f"(short {shortfall / 1e9:.1f}GB)",
                )

        # Report the snapshot actually on disk against the pin, so an unpinned
        # or drifted checkpoint is visible in run logs instead of being assumed.
        if not self.revision:
            rev_note = ", UNPINNED revision"
        elif snapshot.name != self.revision:
            rev_note = f", revision MISMATCH: on disk {snapshot.name[:12]}"
        else:
            rev_note = f", rev {self.revision[:12]}"

        return True, f"{len(shards)} shard(s), {total / 1e9:.1f}GB{rev_note}"

    def predict(
        self, case: CaseInput, output_dir: Path, device: str
    ) -> Prediction:
        """Dispatch to the runner that owns this task's prompt and parser.

        The two runners differ in prompt, output grammar and prediction shape, so
        the mapping is explicit and a task with no entry raises. Falling back to
        the scoring runner would silently ask a localisation case for a SYNTAX
        number and record the mismatch as a model failure.
        """
        from benchmark.runners import arcade_vlm_runner, vlm_runner

        runners = {
            Task.CARDIOSYNTAX_SCORING: vlm_runner.predict,
            Task.ARCADE_SEGMENTATION: arcade_vlm_runner.predict,
            Task.ARCADE_STENOSIS: arcade_vlm_runner.predict,
        }
        runner = runners.get(case.task)
        if runner is None:
            raise ValueError(
                f"{self.name}: no VLM runner for task {case.task.value}; "
                f"supported: {', '.join(sorted(t.value for t in runners))}"
            )
        return runner(self, case, output_dir, device)


# ==========================================================================
# Concrete methods
# ==========================================================================
#
# SYNTAX scoring plus the two 2D ARCADE localisation tasks. CCA segmentation stays
# withheld: see the module docstring on why a dense 3D mask is not a well-posed
# question for a text-output model.

_VLM_TASKS = (
    Task.CARDIOSYNTAX_SCORING,
    Task.ARCADE_SEGMENTATION,
    Task.ARCADE_STENOSIS,
)



#: Appended to every VLM's limitations. The ARCADE caveats are the same for all of
#: them because the constraint is the output channel, not the checkpoint.
_ARCADE_CAVEAT = (
    " On the ARCADE tasks the output is a bounding box, not a segmentation "
    "mask, and is scored as the filled rectangle (diagnostics record "
    "mask_source=bbox); the ceiling is therefore below a mask-producing "
    "method's. Segment ids come from cardiology text in the pretraining "
    "corpus rather than from any learned correspondence to annotated "
    "angiograms."
)


def _general_vlm(
    name: str, repo_id: str, description: str, revision: str = ""
) -> VLMMethod:
    """A general-purpose VLM with no medical training claim."""
    return VLMMethod(
        name=name,
        tasks=_VLM_TASKS,
        repo_id=repo_id,
        revision=revision,
        provenance=Provenance(
            source=f"{repo_id}@{revision}" if revision else repo_id,
            trained_on="general web-scale image-text data",
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a - not evaluated on coronary angiography upstream",
            limitations=(
                f"{description} Never trained on angiography. Any SYNTAX number "
                "it produces comes from general visual reasoning plus whatever "
                "cardiology text was in its pretraining corpus, not from a "
                "learned scoring function." + _ARCADE_CAVEAT
            ),
        ),
    )


def _medical_vlm(
    name: str,
    repo_id: str,
    trained_on: str,
    limitations: str,
    revision: str = "",
) -> VLMMethod:
    """A VLM with medical-domain instruction tuning."""
    return VLMMethod(
        name=name,
        tasks=_VLM_TASKS,
        repo_id=repo_id,
        revision=revision,
        provenance=Provenance(
            source=f"{repo_id}@{revision}" if revision else repo_id,
            trained_on=trained_on,
            domain_relation=DomainRelation.NOT_TRAINED,
            reported_metric="n/a - not evaluated on SYNTAX scoring upstream",
            limitations=limitations + _ARCADE_CAVEAT,
        ),
    )


ALL_VLMS: list[VLMMethod] = [
    # Revisions are the upstream commits actually evaluated, read back from the
    # local HF cache. HF repos are mutable, so a name alone does not identify a
    # checkpoint; these pins are what makes a reported number reproducible.
    # --- medical-tuned ---
    _medical_vlm(
        name="lingshu_7b",
        repo_id="lingshu-medical-mllm/Lingshu-7B",
        revision="b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
        trained_on="medical multimodal corpus (Qwen2.5-VL base)",
        limitations=(
            "Medical instruction tuning covers radiology reports and VQA, but "
            "the training mix is not documented to include invasive coronary "
            "angiography or SYNTAX scoring. Best-positioned medical VLM here, "
            "still a zero-shot transfer."
        ),
    ),
    # --- general-purpose ---
    _general_vlm(
        "qwen25_vl_7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Strong general VLM with native video input support, which suits cine loops.",
        revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
    ),
    _general_vlm(
        "qwen3_vl_8b",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Newest Qwen generation.",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    ),
    _general_vlm(
        "llava_16_mistral_7b",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "LLaVA-NeXT lineage; the closest available stand-in for the LLaVA-Med "
        "baseline used in prior agent papers, whose checkpoint is not HF-loadable.",
        revision="2424fdd47412fccc66d91719126b420e9fbd7065",
    ),
    _general_vlm(
        "llama3_llava_next_8b",
        "llava-hf/llama3-llava-next-8b-hf",
        "LLaVA-NeXT on a Llama-3 backbone.",
        revision="b041c0d0ea0dd0196d147206c210c8d1752fc2da",
    ),
    _general_vlm(
        "llava_onevision_7b",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "Trained for multi-image and video input, relevant for multi-view angiography.",
        revision="0d50680527681998e456c7b78950205bedd8a068",
    ),
]

BY_NAME: dict[str, VLMMethod] = {m.name: m for m in ALL_VLMS}
