"""
Core abstractions for CardiomniBench-VD.

This module defines the contracts every evaluated method obeys. It is written to
be read: a reviewer should be able to open this one file and understand what a
"method" is, what it is handed, what it must return, and where responsibility
sits, without tracing string identifiers through the codebase.

Structure
---------
    Prediction          what a method returns (typed, validated)
    Method              abstract base: a thing that can be evaluated
      +- SpecialistMethod   a trained model with checkpoint weights
      |    +- VolumeSegmenter    3D mask output
      |    +- ScoreRegressor     scalar output
      +- VLMMethod          a prompted vision-language model

Design rules this file enforces
-------------------------------
1. No string-based indirection. A method holds a real callable/class, not a
   module path resolved at runtime. If a runner is missing, that is an import
   error at load time, not a mid-run failure on case 47 of 60.

2. Configuration is typed. Required fields are constructor arguments; optional
   ones have declared defaults. A missing required field fails immediately with
   a clear message rather than surfacing as a KeyError deep in inference.

3. Shared behaviour lives in a base class, not in copied dict literals. The four
   coronary CTA segmentation checkpoints share one preprocessing definition
   because they inherit it, so it cannot drift between them.

4. Every method declares its own provenance and known limitations. These are
   printed with results, so a reader of the output table sees the caveats
   attached to the number rather than having to find them in a paper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from benchmark.io_spec import CaseInput


# ==========================================================================
# Task and family taxonomy
# ==========================================================================


class Task(str, Enum):
    """The evaluated tasks. Values match the directory names under data/tasks/."""

    CARDIOSYNTAX_SCORING = "cardiosyntax_scoring"
    CCA_SEGMENTATION = "cca_segmentation"
    ARCADE_SEGMENTATION = "arcade_segmentation"
    ARCADE_STENOSIS = "arcade_stenosis"

    @property
    def output_kind(self) -> OutputKind:
        """What shape of answer this task expects."""
        return {
            Task.CARDIOSYNTAX_SCORING: OutputKind.SCALAR,
            Task.CCA_SEGMENTATION: OutputKind.VOLUME_MASK,
            Task.ARCADE_SEGMENTATION: OutputKind.INSTANCE_LIST,
            Task.ARCADE_STENOSIS: OutputKind.INSTANCE_LIST,
        }[self]


class OutputKind(str, Enum):
    """The form of a prediction, which determines how it is stored and scored."""

    SCALAR = "scalar"
    VOLUME_MASK = "volume_mask"
    INSTANCE_LIST = "instance_list"  # List of instances, each with label/bbox/mask


class Family(str, Enum):
    """How a method arrives at its answer.

    This distinction matters for interpretation: a specialist was trained for
    something, a VLM is being asked cold, and a harness is an orchestration
    strategy wrapped around some base model. They are not competitors on equal
    terms and the results table labels them so.

    Only the harness axis is the object of study. Specialists are tools and
    upper-bound references; VLMs establish what the base model does unaided. A
    harness comparison holds the base model and the tool interface fixed and
    varies only the orchestration, so a difference between two harness rows is
    attributable to the orchestration itself.
    """

    SPECIALIST = "specialist"
    VLM = "vlm"
    HARNESS = "harness"


class DomainRelation(str, Enum):
    """Relationship between a method's training data and our evaluation data.

    This is the single most important qualifier on any number in this benchmark.
    An in-domain result and a transfer result are different claims, and mixing
    them unlabelled in one column would be misleading.
    """

    IN_DOMAIN = "in_domain"
    """Trained on the same dataset family we evaluate on."""

    CROSS_DATASET = "cross_dataset"
    """Same imaging modality and same anatomy, different dataset.

    Kept distinct from :attr:`CROSS_DOMAIN` because it changes how a weak result
    should be read. A coronary CTA network evaluated on a different coronary CTA
    cohort has no modality gap to blame, so a low Dice points at the inference
    implementation, preprocessing, or annotation protocol rather than at domain
    shift.
    """

    CROSS_DOMAIN = "cross_domain"
    """Different imaging modality or anatomy; this is a transfer measurement."""

    NOT_TRAINED = "not_trained"
    """Never trained for this task at all, e.g. a prompted general VLM."""

    UNKNOWN = "unknown"
    """Training corpus not documented upstream. Treated as cross-domain, and
    labelled honestly rather than assumed."""


# ==========================================================================
# Prediction
# ==========================================================================


@dataclass
class Prediction:
    """What a method returns for one case.

    Exactly one of `score`, `mask_path`, or `instances` carries the answer,
    matching the task's OutputKind. `diagnostics` holds anything that helps
    interpret the result later - confidence statistics, view counts, parse
    failures - and is persisted verbatim.
    """

    case_id: str
    task: Task

    score: float | None = None
    mask_path: Path | None = None

    #: Instance-segmentation answer for the ARCADE tasks. One dict per detected
    #: segment; keys mirror the gold in ``data/tasks/arcade_*/cases/*/task.yaml``:
    #: ``label`` (SYNTAX segment id as a string), ``bbox_xywh_norm`` (4 floats in
    #: [0,1]), and optionally ``mask`` (bbox-local binary array). Masks stay
    #: out-of-band rather than inline in prediction.json when they are large; the
    #: scorer accepts either an array or a path recorded here.
    instances: list[dict[str, Any]] | None = None

    # Optional decomposition, reported when a method provides it.
    components: dict[str, float] = field(default_factory=dict)

    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Raw model output, kept for VLMs so a parse can be re-examined without
    # re-running inference.
    raw_output: str | None = None

    def validate(self) -> None:
        """Check the prediction matches its task's expected output kind."""
        kind = self.task.output_kind
        if kind is OutputKind.SCALAR:
            if self.score is None:
                raise ValueError(
                    f"{self.case_id}: task {self.task.value} needs a score, got None"
                )
        elif kind is OutputKind.VOLUME_MASK:
            if self.mask_path is None:
                raise ValueError(
                    f"{self.case_id}: task {self.task.value} needs a mask_path"
                )
            if not self.mask_path.exists():
                raise FileNotFoundError(
                    f"{self.case_id}: mask_path does not exist: {self.mask_path}"
                )
        elif kind is OutputKind.INSTANCE_LIST:
            # An empty list is a legitimate answer ("no segments found") and must
            # stay distinguishable from None ("the method never produced one"),
            # so this checks for None rather than falsiness.
            if self.instances is None:
                raise ValueError(
                    f"{self.case_id}: task {self.task.value} needs instances"
                )
            for i, inst in enumerate(self.instances):
                if "label" not in inst:
                    raise ValueError(
                        f"{self.case_id}: instance {i} has no 'label'"
                    )
                bbox = inst.get("bbox_xywh_norm")
                if bbox is None or len(bbox) != 4:
                    raise ValueError(
                        f"{self.case_id}: instance {i} needs bbox_xywh_norm "
                        f"of length 4, got {bbox!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for persistence."""
        return {
            "case_id": self.case_id,
            "task": self.task.value,
            "score": self.score,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "instances": self.instances,
            "components": self.components,
            "diagnostics": self.diagnostics,
            "raw_output": self.raw_output,
        }


# ==========================================================================
# Provenance
# ==========================================================================


@dataclass(frozen=True)
class Provenance:
    """Where a method came from and what its authors claim about it.

    Carried alongside every result so a reader can weigh the number without
    leaving the output.
    """

    source: str
    """Repository or publication the weights come from."""

    trained_on: str
    """Dataset the weights were fitted to, or "n/a" for an untrained VLM."""

    domain_relation: DomainRelation

    reported_metric: str = "not stated"
    """What the authors report on their own test set, verbatim."""

    limitations: str = ""
    """Known caveats that affect how the number should be read."""


# ==========================================================================
# Method hierarchy
# ==========================================================================


@dataclass
class Method(ABC):
    """A thing that can be evaluated on a task.

    Subclasses implement `predict`. Everything else - weight checking, naming,
    provenance reporting - is handled here so every method behaves consistently.
    """

    name: str
    """Unique identifier, used in result tables and on the command line."""

    tasks: tuple[Task, ...]
    """Which tasks this method can attempt."""

    provenance: Provenance
    """Origin and author claims, reported with every result."""

    #: Set by each subclass; not a constructor argument.
    family: ClassVar[Family]

    def supports(self, task: Task) -> bool:
        return task in self.tasks

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        """Report whether this method can actually run.

        Called before any inference. Returning False keeps the method out of the
        results table with a stated reason, rather than letting it fail
        per-case or, worse, emit a fabricated number.
        """

    @abstractmethod
    def predict(
        self, case: CaseInput, output_dir: Path, device: str
    ) -> Prediction:
        """Run inference on one case."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"
