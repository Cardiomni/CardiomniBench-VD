"""
Harness definitions for CardiomniBench-VD.

A harness is an *orchestration strategy*, not a model. It is handed a fixed base
model and a fixed tool interface, and the only thing that varies between rows in
this family is how it sequences its calls. That is what makes this the one family
whose members are the research object: specialists are tools and upper-bound
references, VLMs measure the base model unaided, and a harness measures what
orchestration adds on top of an unchanged base model.

Why the weakest baseline matters
--------------------------------
`naive_tool_caller` is deliberately unguided: it receives the tool signatures and
the task, and nothing else. No staging, no clinical protocol, no hint that
dominance should be established before segments are named. It exists so that
Cardiomni's four-stage SOP has something to be measured against. If an unguided
loop matches the SOP, the SOP contributes nothing, and that result has to be
reachable for the comparison to mean anything.

This is why the prompt here must not be improved. Prompt quality is the
independent variable of the experiment one axis over (base model swapping);
inside this family, letting it drift would confound the attribution.

Fair comparison and honest tools
--------------------------------
Every harness sees the same tools, including the broken one. `detect_stenosis`
raises unconditionally because the DeepCORO-CLIP weights were never obtainable,
and it is offered anyway: how a harness reacts to an unavailable tool is itself a
result. A weak loop tends to call it, take the exception, and lose the case; a
planning harness can read `detection_metadata()["alternatives"]` and route around
it. The trace records which happened.

Tool limitations are stated in the prompt rather than hidden. `segment_vessels`
advertises `can_name_segments=False`, so a harness is told, truthfully, that
segmentation cannot supply SYNTAX ids. Concealing that would not make the
comparison harder, it would make it uninterpretable.
"""

from __future__ import annotations

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

#: Tasks a harness is asked to attempt. The two ARCADE tasks are the ones where
#: orchestration is observable: they need localisation plus, for segmentation,
#: naming, which is exactly the split between what the tools can do and what the
#: base model has to contribute. cardiosyntax_scoring is included because a
#: single number is the cheapest end-to-end check that a loop terminates and
#: produces something scoreable.
_HARNESS_TASKS = (
    Task.ARCADE_SEGMENTATION,
    Task.ARCADE_STENOSIS,
    Task.CARDIOSYNTAX_SCORING,
)


@dataclass
class HarnessMethod(Method):
    """An orchestration strategy wrapped around a fixed base model and toolset.

    The base model is named rather than embedded so the same harness row can be
    re-run against a different base model without touching this file: that is the
    second swap axis in PROPOSAL.md §4, and it only works if the harness holds a
    reference instead of a checkpoint.
    """

    family: ClassVar[Family] = Family.HARNESS

    base_model: str = ""
    """Name of the VLM in `benchmark.vlms.BY_NAME` used as the reasoning engine."""

    tools: tuple[str, ...] = ()
    """Tool identifiers this harness may call, resolved by the runner."""

    #: Hard turn budget. A fixed budget is part of the evaluation protocol: an
    #: unbounded loop would make the comparison one of patience rather than
    #: orchestration, and a runaway case would stall a 42-case sweep.
    max_turns: int = 8

    #: Set when the loop should run without real tool implementations, for
    #: pipeline verification. Results produced this way are marked in the
    #: per-case diagnostics and must never be reported as measurements.
    use_mock_tools: bool = False

    def check_available(self) -> tuple[bool, str]:
        """Availability is the base model's availability plus tool imports.

        The broken tool is deliberately not treated as a blocker. Its failure is
        an observation this experiment wants, so gating on it would remove the
        very behaviour the harness axis is meant to expose.
        """
        from benchmark.vlms import BY_NAME

        base = BY_NAME.get(self.base_model)
        if base is None:
            return False, f"unknown base model: {self.base_model}"

        if self.use_mock_tools:
            return True, f"{self.base_model} + mock tools (not a measurement)"

        ok, detail = base.check_available()
        if not ok:
            return False, f"base model {self.base_model}: {detail}"

        # Import-check the tool layer so a missing module is reported here rather
        # than as a per-case failure on every case in the sweep.
        try:
            import algorithms.tools  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, f"tool layer unimportable: {type(exc).__name__}: {exc}"

        return True, f"base {self.base_model} ready, {len(self.tools)} tool(s)"

    def predict(
        self, case: CaseInput, output_dir: Path, device: str
    ) -> Prediction:
        """Run the orchestration loop for one case."""
        from benchmark.runners import harness_runner

        return harness_runner.predict(self, case, output_dir, device)


# ==========================================================================
# Concrete harnesses
# ==========================================================================
#
# Only the unguided baseline is wired up. Cardiomni is implemented separately by
# the author of the SOP; the external agent harnesses (Claude Code, Codex) need a
# sandbox contract that does not exist yet. They are listed as declared-but-absent
# rather than silently missing, so the gap in the results table is visible.

#: The shared tool set. Identical across harnesses on purpose: if a harness could
#: pick its own tools, a difference in results would no longer isolate
#: orchestration.
SHARED_TOOLS = ("segment_vessels", "quantify_stenosis", "detect_stenosis")


# DEPRECATED (2026-07-26). This row conflates two families and must not appear in
# the paper. A VLM baseline answers from the image alone; handing it tool
# signatures and a multi-turn loop measures neither the base model's vision nor an
# agent's orchestration. The unaided-base-model control already exists as
# Family.VLM (benchmark/vlms.py + runners/vlm_runner.py +
# runners/arcade_vlm_runner.py), which prompts for structured output with no tools.
# Kept only so runs/harness_smoke{,2}/ stay interpretable; those 4 cases are void.
# When Cardiomni lands, its control is the VLM family, not this.
NAIVE_TOOL_CALLER = HarnessMethod(
    name="naive_tool_caller",
    tasks=_HARNESS_TASKS,
    base_model="llava_16_mistral_7b",
    tools=SHARED_TOOLS,
    max_turns=8,
    provenance=Provenance(
        source="this repository (benchmark/runners/harness_runner.py)",
        trained_on="n/a - no training; a prompting loop over a frozen base model",
        domain_relation=DomainRelation.NOT_TRAINED,
        reported_metric="n/a - baseline defined here, no upstream number exists",
        limitations=(
            "Intentionally unguided: the base model receives the tool "
            "signatures and the task, with no staged protocol and no clinical "
            "prior. Weakness is the design goal, since this row is the control "
            "that Cardiomni's SOP orchestration is measured against. Inherits "
            "every limitation of its base model, which was never trained on "
            "angiography, and of the tools, which cannot name SYNTAX segments. "
            "One offered tool (detect_stenosis) always raises because its "
            "weights are unobtainable; how the loop copes is recorded in the "
            "trace rather than hidden."
        ),
    ),
)


# ==========================================================================
# Placeholder for the Cardiomni harness
# ==========================================================================
#
# The SOP-guided harness is implemented by the project author. It is registered
# here once the implementation exists, so it appears in the results table
# alongside the unguided baseline. The gap in the table before that happens is
# intentional: the core contribution is not yet built.
#
# Implementation guidance for the four-stage SOP:
#
# Stage 1 (Dominance): Identify whether the coronary tree is left-dominant,
# right-dominant, or co-dominant. This is a visual + anatomical judgement the
# base VLM can make from the rendered frames. No tool calls in this stage; the
# model looks at the images and states a conclusion.
#
# Stage 2 (Segment naming): Systematically scan every SYNTAX segment visible in
# the views. Call `segment_vessels` to localise the tree, but *do not rely on it
# for segment ids*: `segmentation_metadata()["can_name_segments"]` is False, and
# real data confirms this — CM-UNet's label_set_precision and label_set_recall
# are both 0.0000 on 222 ARCADE cases. The SOP must use the base VLM to look at
# the mask overlay and assign segment names, or use an alternative approach that
# does not depend on the segmentation tool producing labels it cannot produce.
#
# Stage 3 (View selection): Choose the projections that show each lesion most
# clearly. For quantification, perpendicular views are preferred; for counting
# lesions, avoid double-counting the same stenosis visible in overlapping
# projections. The base VLM decides which frames to analyse in Stage 4.
#
# Stage 4 (Lesion assessment): For each identified lesion, call
# `quantify_stenosis` with the ROI coords. The Stage 2 output tells you *where*
# each segment is (from the segmentation mask or from the VLM's spatial
# reasoning); Stage 4 measures *how narrow* it is. Do not call `detect_stenosis`
# — check `stenosis_detection.is_available()` first and read the "alternatives"
# in `detection_metadata()` when it returns False.
#
# Each stage's output informs the next, and the trace records the four-stage
# structure explicitly so the contribution of staging can be measured. The
# harness returns a Prediction with the final answer (SYNTAX score for scoring
# tasks, instance list for ARCADE), and diagnostics["trace"] carries the
# structured log that shows where each piece of information came from.
#
# CARDIOMNI = HarnessMethod(
#     name="cardiomni",
#     tasks=_HARNESS_TASKS,
#     base_model="llava_16_mistral_7b",  # or qwen / llama3_llava for base-swap
#     tools=SHARED_TOOLS,
#     max_turns=16,  # SOP needs more budget than the unguided loop
#     provenance=Provenance(
#         source="this repository (benchmark/runners/cardiomni_runner.py)",
#         trained_on="n/a - orchestration over a frozen base model + tools",
#         domain_relation=DomainRelation.NOT_TRAINED,
#         reported_metric="n/a",
#         limitations=(
#             "Four-stage SOP (dominance → segment naming → view selection → "
#             "quantification) over llava_16_mistral_7b + shared tools. The SOP "
#             "explicitly routes around detect_stenosis unavailability and uses "
#             "the VLM for segment naming because segmentation_metadata() states "
#             "that the tool cannot produce SYNTAX ids. Inherits base model "
#             "limitations (no angiography training) and tool limitations (binary "
#             "segmentation only)."
#         ),
#     ),
# )


ALL_HARNESSES: list[HarnessMethod] = [
    NAIVE_TOOL_CALLER,
    # CARDIOMNI,  # uncomment when implemented
]

BY_NAME: dict[str, HarnessMethod] = {m.name: m for m in ALL_HARNESSES}
