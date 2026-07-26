"""
Specialist model definitions for CardiomniBench-VD.

A specialist is a trained model with published checkpoint weights. It knows one
task or a narrow family of tasks, and was fitted to a particular dataset.

This module defines the concrete specialist methods under evaluation. Each is a
small dataclass declaring its checkpoint path, architecture hints, and any
quirks the runner needs to know. Shared preprocessing (the z-score clipping for
coronary CTA, the log1p transform for SYNTAX scores) lives in the base class so
it cannot drift between checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from benchmark.core import (
    DomainRelation,
    Family,
    Method,
    Prediction,
    Provenance,
    Task,
)
from benchmark.io_spec import CaseInput

WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "algorithms" / "specialist_models" / "weights"


def _load_runner_module(runner_name: str):
    """Import a runner by short name or full module path.

    Short names ("monai_unet", "nnunet", "sam_med3d", "cm_unet") resolve under
    ``benchmark.runners``. Full paths ("benchmark.runners.cardiosyntax_r3d") are
    imported directly. This lets ``methods.py`` and ``specialists.py`` use the
    same protocol without rewriting every registration.
    """
    import importlib

    if "." in runner_name:
        module_path = runner_name
    else:
        module_path = f"benchmark.runners.{runner_name}_runner"

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        # Distinguish "the runner file does not exist" from "the runner exists but
        # one of its dependencies is missing". The second is by far the more
        # common case (torch is absent from the config-only environment) and
        # reporting it as a missing runner sends debugging in the wrong direction.
        missing = exc.name or ""
        if missing == module_path or module_path.endswith(f".{missing}"):
            raise ImportError(
                f"runner {runner_name!r} not found at {module_path}"
            ) from exc
        raise ImportError(
            f"runner {runner_name!r} needs {missing!r}, which is not installed "
            f"in this interpreter"
        ) from exc


# ==========================================================================
# Base classes
# ==========================================================================


@dataclass
class SpecialistMethod(Method):
    """A trained model with published checkpoint weights."""

    family: ClassVar[Family] = Family.SPECIALIST

    weights_path: Path = None  # type: ignore[assignment]
    """Checkpoint file or directory. Required; the None default exists only so
    dataclass inheritance can add fields in subclasses."""

    def check_available(self) -> tuple[bool, str]:
        """Weights exist and are not Git-LFS pointer stubs."""
        if self.weights_path is None:
            return False, "no weights_path declared"
        if not self.weights_path.exists():
            return False, f"missing: {self.weights_path}"
        if self.weights_path.is_file():
            size = self.weights_path.stat().st_size
            # A pointer stub is a few hundred bytes and looks like a checkpoint
            # until torch.load fails. Catch it here instead.
            if size < 100_000:
                return (
                    False,
                    f"{self.weights_path.name} is {size}B, likely a Git-LFS pointer",
                )
            return True, f"{self.weights_path} ({size / 1e6:.0f}MB)"
        return True, str(self.weights_path)

    def _runner_check(self) -> tuple[bool, str]:
        """Weights check plus the runner's own architecture preflight.

        Every specialist reaches its model the same way: a ``runner`` name is
        resolved to a module, and that module owns whatever else must be true
        before inference can start. Subclasses declare the runner; none of them
        need to know how dispatch works.
        """
        if self.runner is None:
            return False, "no runner implemented for this architecture"
        ok, msg = SpecialistMethod.check_available(self)
        if not ok:
            return ok, msg

        # Importing the runner pulls in torch and friends. In a config-only
        # interpreter that fails, and the right answer is "unavailable here",
        # not a crash: listing methods must work without the inference stack.
        try:
            module = _load_runner_module(self.runner)
        except ImportError as exc:
            return False, str(exc)
        hook = getattr(module, "check_available", None)
        if hook is not None:
            return hook(self)
        return True, msg

    def predict(
        self, case: CaseInput, output_dir: Path, device: str
    ) -> Prediction:
        """Dispatch to the declared runner module."""
        module = _load_runner_module(self.runner)
        return module.predict(self, case, output_dir, device)


@dataclass
class CoronaryCTASegmenter(SpecialistMethod):
    """3D coronary artery segmentation on CTA using a **MONAI UNet** checkpoint.

    Scope, verified against the checkpoints rather than assumed
    ----------------------------------------------------------
    This class covers only weights that are genuinely a MONAI ``UNet`` state
    dict with 1 input channel. That was confirmed by inspecting keys:
    ``baseline_unet.pth`` starts with ``model.0.conv.unit0.conv.weight`` of shape
    ``(32, 1, 3, 3, 3)``, which matches.

    Three other coronary checkpoints in this repo do **not** belong here, and
    forcing them through this runner would be wrong rather than merely
    suboptimal:

    - ``att_mamba2_unet.pth``: custom architecture (``stem.0.weight``, shape
      ``(32, 2, 7, 7, 7)``). Two input channels, the second being a Frangi
      vesselness map per the model card. Needs the upstream repo's
      ``src.models.model_factory.get_model``.
    - nnU-Net / UMambaBot ``checkpoint_final.pth``: ``PlainConvUNet`` from
      ``dynamic_network_architectures`` (292 tensors, ``encoder.stages.*`` keys),
      anisotropic ``[96, 160, 160]`` patches, resampling to ``[0.5, 0.35, 0.35]``
      mm, and a **3-class** head (background/lca/rca, seg layer shape
      ``(3, 320, 1, 1, 1)``) rather than binary.
    - ``CM-UNet_weights.pth``: a **2D X-ray angiography** model, not 3D CTA at
      all. Different task entirely.

    Because ``load_state_dict`` here is ``strict=True``, a mismatch fails loudly
    instead of silently scoring a randomly initialised network. Keep it that way.

    Preprocessing is z-score after clipping to the soft-tissue window. This is
    not a hyperparameter we tuned; the empirical sweep that determined it is
    documented in evaluation/metrics/segmentation_metrics.py.
    """

    #: MONAI UNet channel widths, 5-level. Inferred from checkpoint if None.
    channels: tuple[int, ...] | None = (32, 64, 128, 256, 512)
    num_res_units: int = 2

    #: Input channel count the checkpoint expects.
    in_channels: int = 1

    #: Which runner implements this checkpoint's architecture. Compatibility is
    #: a property of the weights, so it is declared per method rather than
    #: guessed at load time.
    #:   "monai_unet" - MONAI UNet state dict, or the vendored AttMamba2UNet;
    #:                  the runner picks the architecture from the state-dict keys
    #:   "nnunet"     - PlainConvUNet/UMambaBot via sidecar plans.json
    #:   None         - no runner implemented yet; method reports unavailable
    runner: str | None = "monai_unet"

    #: Preprocessing, sliding-window and decision-rule parameters are NOT
    #: declared here. They live in ``methods/<name>.toml`` and are loaded by
    #: :mod:`benchmark.method_config`.
    #:
    #: They used to be dataclass defaults (a -200..300 HU window with a
    #: whole-volume z-score, and argmax decisions). Those values did not match
    #: the upstream release that produced these checkpoints, and the mismatch
    #: was invisible because nothing pointed back at the source repository.
    #: coronary_unet scored Dice 0.021 against an upstream-reported 0.788,
    #: mostly because CCA is 0.5 mm isotropic while training ran at 1.0 mm.
    #:
    #: The TOML files transcribe each upstream pipeline with file-level
    #: citations, so a wrong parameter is wrong in one reviewable place.

    def __post_init__(self) -> None:
        if not self.supports(Task.CCA_SEGMENTATION):
            raise ValueError(f"{self.name}: must support cca_segmentation")

    def check_available(self) -> tuple[bool, str]:
        """Weights must exist *and* have a runner that matches their architecture.

        Reporting a method as available when no runner can load its weights just
        moves the failure to the middle of a benchmark run.
        """
        return self._runner_check()


@dataclass
class XCASegmenter(SpecialistMethod):
    """2D vessel segmentation on a single X-ray angiography frame.

    This is a different shape of problem from :class:`CoronaryCTASegmenter`, not a
    variant of it: the input is one 2D projection rather than a volume, there is
    no spacing or orientation to resample, and the output feeds an instance list
    rather than a volume mask.

    The binary-vs-labelled gap is the thing to keep in view. ARCADE asks for
    instances labelled with SYNTAX segment ids; a binary vessel segmenter cannot
    produce those, so it is scored honestly as a detector (see
    ``benchmark/runners/cm_unet_runner.py`` for how components become instances
    and what that costs under label-aware matching). Such a model is a useful
    upper bound on "can the vessel be found at all", and a plausible tool for the
    Cardiomni agent to call, but it is not a competitor on the labelled task.
    """

    #: Output logit count. 2 means (background, vessel) with no built-in softmax.
    out_classes: int = 2

    #: Single-channel grayscale input.
    in_channels: int = 1

    runner: str | None = "cm_unet"

    #: Preprocessing, decision rule and instance extraction are declared in
    #: ``methods/<name>.toml``, never here. For this family the padding choice is
    #: load-bearing: see methods/README.md on why both the upstream-faithful and
    #: the resolution-preserving configurations are registered and reported.

    def check_available(self) -> tuple[bool, str]:
        return self._runner_check()


@dataclass
class CardioSYNTAXRegressor(SpecialistMethod):
    """R3D+LSTM SYNTAX score regressor.

    Trained on log1p(score) and emits log1p(score), so expm1 is mandatory on
    the way out. The official release is a 5-fold × 2-artery ensemble (10
    checkpoints); the fold variants are also evaluated individually to show how
    much variance sits under the ensemble mean.
    """

    #: Ensemble parameters. None means single-fold.
    folds: tuple[int, ...] | None = (0, 1, 2, 3, 4)
    single_fold: int | None = None

    #: Whether to apply the per-fold linear calibration fitted by the authors.
    calibrated: bool = False

    #: Fixed architecture.
    backbone: str = "r3d_18"
    num_classes: int = 2  # dual-head: classification + regression
    target_transform: str = "log1p"

    def __post_init__(self) -> None:
        if not self.supports(Task.CARDIOSYNTAX_SCORING):
            raise ValueError(f"{self.name}: must support cardiosyntax_scoring")
        if (self.single_fold is not None) and (self.folds is not None):
            raise ValueError("single_fold and folds are mutually exclusive")

    def predict(
        self, case: CaseInput, output_dir: Path, device: str
    ) -> Prediction:
        from benchmark.runners import cardiosyntax_r3d_runner

        return cardiosyntax_r3d_runner.predict(self, case, output_dir, device)


# ==========================================================================
# Concrete methods
# ==========================================================================

# Shorthand constructors for the two evaluated domains.
_IMAGECAS_CTA_PROVENANCE = Provenance(
    source="noahschuetz/coronary-segmentation (trained on ImageCAS CTA)",
    trained_on="ImageCAS Challenge dataset",
    # ImageCAS and CCA are both coronary CTA, so there is no modality or anatomy
    # gap here - only a cohort/annotation-protocol difference.
    domain_relation=DomainRelation.CROSS_DATASET,
    reported_metric="Dice 0.788 / clDice 0.864 on ImageCAS hold-out",
    limitations=(
        "Severely under-confident on out-of-distribution data (mean vessel "
        "probability ~0.023 inside true vessels), so native argmax recovers "
        "little of the vessel tree. Reported zero-shot at argmax without a "
        "threshold fitted on the test cases."
    ),
)

_CARDIOSYNTAX_DSA_PROVENANCE = Provenance(
    source="MesserMMP/CardioSYNTAX (official checkpoint)",
    trained_on="CardioSYNTAX training split",
    domain_relation=DomainRelation.IN_DOMAIN,
    reported_metric="Pearson r ~0.81 on the 1844-case in-domain hold-out",
    limitations=(
        "Systematic low-bias on high-risk cases (gold ≥33): prediction ceiling "
        "at ~34 while gold reaches 58. This shifts the PCI-vs-CABG decision "
        "boundary and is the main failure mode."
    ),
)


CORONARY_UNet = CoronaryCTASegmenter(
    name="coronary_unet",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=_IMAGECAS_CTA_PROVENANCE,
    weights_path=WEIGHTS_ROOT / "coronary-seg-unet" / "baseline_unet.pth",
)

#: Diagnostic twin of :data:`CORONARY_UNet`: same checkpoint, same preprocessing,
#: but ``decision_rule = "argmax"`` in ``methods/coronary_unet_argmax.toml``.
#: coronary_unet reproduces upstream's softmax-then-threshold(0.5) rule and comes
#: out precision-heavy (0.775) but recall-poor (0.410), predicting about half the
#: gold vessel volume. Threshold 0.5 on a 2-class head is strictly stricter than
#: argmax, so this variant separates "the decision rule is the bottleneck" from
#: "preprocessing is still misaligned". It is a diagnostic, not a headline
#: baseline: reporting it as the method's score would deviate from upstream.
CORONARY_UNET_ARGMAX = CoronaryCTASegmenter(
    name="coronary_unet_argmax",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=_IMAGECAS_CTA_PROVENANCE,
    weights_path=WEIGHTS_ROOT / "coronary-seg-unet" / "baseline_unet.pth",
)

CORONARY_ATT_MAMBA2 = CoronaryCTASegmenter(
    name="coronary_att_mamba2",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=Provenance(
        source="noahschuetz/coronary-segmentation (Att-Mamba2 variant)",
        trained_on="ImageCAS Challenge dataset",
        domain_relation=DomainRelation.CROSS_DATASET,
        reported_metric="Dice 0.791 on the upstream ImageCAS validation split",
        limitations=(
            "Custom architecture (stem.0.weight, shape (32, 2, 7, 7, 7)) taking 2 "
            "input channels: CT + Frangi vesselness. Not a MONAI UNet, so the "
            "runner builds the vendored upstream AttMamba2UNet in "
            "algorithms/specialist_models/att_mamba2/ and loads the checkpoint "
            "strictly (193/193 tensors). mamba-ssm needs nvcc, which this host "
            "lacks, so Mamba2 runs through a kernel-free PyTorch "
            "reimplementation: same parameters and recurrence, but a sequential "
            "scan, hence far slower than baseline_unet per volume. "
            "Cross-dataset caveat is the same as coronary_unet: ImageCAS labels "
            "the full lumen while CCA labels the internal diameter."
        ),
    ),
    weights_path=WEIGHTS_ROOT / "coronary-seg-unet" / "att_mamba2_unet.pth",
    in_channels=2,
    runner="monai_unet",
)

CORONARY_NNUNET = CoronaryCTASegmenter(
    name="coronary_nnunet",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=Provenance(
        source="mhyu222/coronary-segmentation-nnunet-umamba (nnU-Net)",
        trained_on="Dataset101_Corornary (per plans.json), source cohort not stated",
        # plans.json shows CTNormalization and coronary CTA spacing, so the
        # modality is certain even though the cohort is not documented.
        domain_relation=DomainRelation.CROSS_DATASET,
        reported_metric="not stated",
        limitations=(
            "PlainConvUNet from dynamic_network_architectures (292 tensors, "
            "encoder.stages.* keys), not MONAI. 3-class head "
            "(background/lca/rca, seg layer (3, 320, 1, 1, 1)) rather than "
            "binary; anisotropic [96,160,160] patches at [0.5,0.35,0.35] mm "
            "spacing with CTNormalization. Now served via nnunet_runner."
        ),
    ),
    weights_path=WEIGHTS_ROOT / "coronary-seg-nnunet" / "nnUNet" / "checkpoint_final.pth",
    runner="nnunet",
)

CORONARY_UMAMBA = CoronaryCTASegmenter(
    name="coronary_umamba",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=Provenance(
        source="mhyu222/coronary-segmentation-nnunet-umamba (UMambaBot)",
        trained_on="Dataset101_Corornary (per plans.json), source cohort not stated",
        domain_relation=DomainRelation.CROSS_DATASET,
        reported_metric="not stated",
        limitations=(
            "Cannot be evaluated in this environment. Despite plans.json listing "
            "UNet_class_name=PlainConvUNet, the checkpoint's encoder is built from "
            "Mamba SSM blocks (mamba_layer.mamba.A_log, .D, .dt_proj etc.), so "
            "loading it into a PlainConvUNet gives 282 missing / 387 unexpected "
            "keys. The real network class needs the mamba_ssm and causal_conv1d "
            "CUDA extensions, neither of which is installed."
        ),
    ),
    weights_path=WEIGHTS_ROOT / "coronary-seg-nnunet" / "UMambaBot" / "checkpoint_final.pth",
    runner=None,  # blocked: needs mamba_ssm + causal_conv1d
)

# CM-UNet is the only genuine 2D XCA segmenter among the downloaded weights, so
# it is the natural first baseline on the ARCADE tasks. It is registered twice:
# the preprocessing question (follow the model card's 1536 padding, or preserve
# resolution on our 512 frames) has a real effect on thin vessels, and picking a
# winner after seeing the scores would turn the benchmark into a report on our
# tuning. Both are evaluated and both are reported - see methods/README.md.
_CM_UNET_PROVENANCE = Provenance(
    source="CamilleChallier/CM-UNet (arXiv:2507.17779)",
    trained_on="X-ray angiography, self-supervised pretraining then transfer",
    domain_relation=DomainRelation.CROSS_DOMAIN,
    reported_metric="reported on the authors' own XCA test split",
    limitations=(
        "Binary vessel/background segmenter: it has no notion of SYNTAX segment "
        "identity, so on arcade_segmentation it cannot score under label-aware "
        "matching by construction. Read it as a vessel-detection bound via "
        "mean_matched_iou and the label-agnostic metrics. Trained on angiograms "
        "padded to 1536 and resized to 256; our ARCADE frames are 512, which is "
        "why the padding configuration is treated as an open question rather than "
        "a fixed part of the method."
    ),
)

CORONARY_CM_UNET = XCASegmenter(
    name="coronary_cm_unet",
    tasks=(Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS),
    provenance=_CM_UNET_PROVENANCE,
    weights_path=WEIGHTS_ROOT / "CM-UNet" / "CM-UNet_weights.pth",
)

CORONARY_CM_UNET_NATIVE = XCASegmenter(
    name="coronary_cm_unet_native",
    tasks=(Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS),
    provenance=Provenance(
        source=_CM_UNET_PROVENANCE.source + " (native-resolution preprocessing)",
        trained_on=_CM_UNET_PROVENANCE.trained_on,
        domain_relation=_CM_UNET_PROVENANCE.domain_relation,
        reported_metric="no upstream number applies: the pipeline is modified",
        limitations=(
            _CM_UNET_PROVENANCE.limitations
            + " This variant skips the 1536 padding and resizes 512 to 256 "
            "directly, so its scores are not a claim about CM-UNet as published."
        ),
    ),
    weights_path=WEIGHTS_ROOT / "CM-UNet" / "CM-UNet_weights.pth",
)

CARDIOSYNTAX_R3D = CardioSYNTAXRegressor(
    name="cardiosyntax_r3d",
    tasks=(Task.CARDIOSYNTAX_SCORING,),
    provenance=_CARDIOSYNTAX_DSA_PROVENANCE,
    weights_path=WEIGHTS_ROOT / "coronary-syntax-prediction" / "full_model",
)

CARDIOSYNTAX_R3D_CALIBRATED = CardioSYNTAXRegressor(
    name="cardiosyntax_r3d_calibrated",
    tasks=(Task.CARDIOSYNTAX_SCORING,),
    provenance=Provenance(
        source=_CARDIOSYNTAX_DSA_PROVENANCE.source + " (with calibration)",
        trained_on=_CARDIOSYNTAX_DSA_PROVENANCE.trained_on,
        domain_relation=_CARDIOSYNTAX_DSA_PROVENANCE.domain_relation,
        reported_metric=_CARDIOSYNTAX_DSA_PROVENANCE.reported_metric,
        limitations=_CARDIOSYNTAX_DSA_PROVENANCE.limitations
        + " Uses the per-fold linear coefficients shipped by the authors in "
        "scaling_coeffs.json, fitted on their own training data. This is part "
        "of the released checkpoint, not a threshold tuned on our test cases, "
        "so it remains zero-shot with respect to this benchmark.",
    ),
    weights_path=WEIGHTS_ROOT / "coronary-syntax-prediction" / "full_model",
    calibrated=True,
)


# Folds as separate rows. The ensemble is kept; the folds show what spread sits
# underneath the reported ensemble number.
def _expand_folds() -> list[CardioSYNTAXRegressor]:
    variants = [CARDIOSYNTAX_R3D, CARDIOSYNTAX_R3D_CALIBRATED]
    for base in [CARDIOSYNTAX_R3D, CARDIOSYNTAX_R3D_CALIBRATED]:
        for fold in range(5):
            variants.append(
                CardioSYNTAXRegressor(
                    name=f"{base.name}_fold{fold}",
                    tasks=base.tasks,
                    provenance=Provenance(
                        source=base.provenance.source + f" (fold {fold} only)",
                        trained_on=base.provenance.trained_on,
                        domain_relation=base.provenance.domain_relation,
                        reported_metric=base.provenance.reported_metric,
                        limitations=base.provenance.limitations,
                    ),
                    weights_path=base.weights_path,
                    folds=None,
                    single_fold=fold,
                    calibrated=base.calibrated,
                )
            )
    return variants


SAM_MED3D = CoronaryCTASegmenter(
    name="sam_med3d",
    tasks=(Task.CCA_SEGMENTATION,),
    provenance=Provenance(
        source="blueyo0/SAM-Med3D (SAM-Med3D-turbo, arXiv:2310.15161)",
        trained_on="SA-Med3D-140K (143K 3D masks, 245 organ categories)",
        domain_relation=DomainRelation.CROSS_DOMAIN,
        reported_metric="Dice reported on 16 multi-organ datasets (not coronary-specific)",
        limitations=(
            "Foundation model for 3D medical segmentation. Uses sparse point prompts "
            "sampled from gold mask (not zero-shot). Processes 128³ patches at 1.5mm "
            "spacing with gold-guided prompts per patch. Designed for organ-level "
            "segmentation; coronary arteries are thin tubular structures that may be "
            "outside its primary training distribution."
        ),
    ),
    weights_path=WEIGHTS_ROOT / "SAM-Med3D" / "sam_med3d_turbo.pth",
    runner="sam_med3d",
)

ALL_SPECIALISTS: list[SpecialistMethod] = [
    CORONARY_UNet,
    CORONARY_UNET_ARGMAX,
    CORONARY_ATT_MAMBA2,
    CORONARY_NNUNET,
    CORONARY_UMAMBA,
    CORONARY_CM_UNET,
    CORONARY_CM_UNET_NATIVE,
    SAM_MED3D,
    *_expand_folds(),
]
