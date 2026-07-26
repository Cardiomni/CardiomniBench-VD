"""
Method registry for CardiomniBench-VD.

One declarative table of every zero-shot method under evaluation. Adding a
method means adding a row here; the runner, the result tables and the paper
numbers all follow from it. Nothing else needs to change.

Two families:

  SPECIALIST  purpose-built models with published weights. They define the
              practical upper bound for their own task and expose where that
              bound sits.
  VLM         general vision-language models prompted zero-shot. No fine-tuning,
              no task-specific heads.

Every method declares which tasks it supports. The runner skips unsupported
pairs instead of pretending a segmentation UNet can score SYNTAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = REPO_ROOT / "algorithms" / "specialist_models" / "weights"
HF_CACHE = Path("/mnt/aliyunsb/Cardiomni/hf_cache")

# Task names as plain strings, kept in step with the canonical Task enum in
# benchmark.core rather than spelled out again. This module predates the enum and
# is still string-based; deriving the Literal from one source stops the two from
# drifting, which is how --tasks silently lost the ARCADE options.
Task = Literal[
    "cardiosyntax_scoring",
    "cca_segmentation",
    "arcade_segmentation",
    "arcade_stenosis",
]
Family = Literal["specialist", "vlm"]


@dataclass(frozen=True)
class Method:
    """A single evaluable method."""

    name: str
    family: Family
    tasks: tuple[Task, ...]
    # The runner module that knows how to drive this model. Two spellings are
    # accepted and resolve identically (benchmark.specialists._load_runner_module):
    # a short name like "monai_unet" resolves to benchmark.runners.monai_unet_runner,
    # and a full dotted path like "benchmark.runners.cardiosyntax_r3d" is imported
    # as given. A runner exports predict(method, case, output_dir, device) and may
    # export check_available(method) for architecture preflight, so adding a model
    # never means editing dispatch code.
    runner: str
    # Free-form knobs handed to the runner (checkpoint path, arch hints, ...).
    config: dict = field(default_factory=dict)
    # Human-readable provenance, printed in result tables.
    source: str = ""
    # What the authors report on their own in-domain test set, for reference.
    reported: str = ""
    notes: str = ""

    @property
    def is_zero_shot_transfer(self) -> bool:
        """True when the weights were trained on a different dataset than we test on."""
        return bool(self.config.get("cross_domain", False))


# --------------------------------------------------------------------------
# SPECIALIST MODELS
# --------------------------------------------------------------------------

SPECIALISTS: list[Method] = [
    Method(
        name="cardiosyntax_r3d",
        family="specialist",
        tasks=("cardiosyntax_scoring",),
        runner="benchmark.runners.cardiosyntax_r3d",
        config={
            "weights": str(
                WEIGHTS_ROOT / "coronary-syntax-prediction" / "full_model"
            ),
            "arch": "r3d_18+lstm",
            "num_classes": 2,
            "target_transform": "log1p",
            "cross_domain": False,
            # 5-fold ensemble; also evaluated fold-by-fold via expand_variants.
            "folds": (0, 1, 2, 3, 4),
        },
        source="MesserMMP/CardioSYNTAX (official weights)",
        reported="Pearson r ~0.81 on the 1844-case in-domain test set",
        notes=(
            "In-domain: trained and evaluated on CardioSYNTAX splits. Regression "
            "head predicts log1p(score); expm1 on the way out is mandatory. "
            "Released as a 5-fold x 2-artery ensemble (10 checkpoints); the "
            "folds are also evaluated individually."
        ),
    ),
    Method(
        name="coronary_unet",
        family="specialist",
        tasks=("cca_segmentation",),
        runner="benchmark.runners.monai_unet",
        config={
            "weights": str(WEIGHTS_ROOT / "coronary-seg-unet" / "baseline_unet.pth"),
            "arch": "monai_unet",
            "channels": (32, 64, 128, 256, 512),
            "num_res_units": 2,
            "normalization": "zscore_clip",
            "clip_range": (-200, 300),
            "threshold": 0.02,
            "cross_domain": True,
        },
        source="noahschuetz/coronary-segmentation (trained on ImageCAS)",
        reported="Dice 0.788 / clDice 0.864 on the ImageCAS hold-out",
        notes=(
            "Cross-domain: ImageCAS weights applied to CCA. Severely "
            "under-confident out of distribution, so the operating threshold is "
            "0.02 rather than argmax; this is reported, not tuned away."
        ),
    ),
    Method(
        name="coronary_att_mamba2",
        family="specialist",
        tasks=("cca_segmentation",),
        runner="benchmark.runners.monai_unet",
        config={
            "weights": str(WEIGHTS_ROOT / "coronary-seg-unet" / "att_mamba2_unet.pth"),
            "arch": "att_mamba2_unet",
            "normalization": "zscore_clip",
            "clip_range": (-200, 300),
            "threshold": 0.02,
            "cross_domain": True,
        },
        source="noahschuetz/coronary-segmentation (Att-Mamba2 variant)",
        reported="same repo as coronary_unet; per-variant numbers not broken out",
        notes="Architecture is resolved from the checkpoint at load time.",
    ),
    Method(
        name="coronary_nnunet",
        family="specialist",
        tasks=("cca_segmentation",),
        runner="benchmark.runners.nnunet",
        config={
            "weights": str(
                WEIGHTS_ROOT / "coronary-seg-nnunet" / "nnUNet" / "checkpoint_final.pth"
            ),
            "arch": "nnunet",
            "cross_domain": True,
        },
        source="mhyu222/coronary-segmentation-nnunet-umamba (nnU-Net)",
        reported="not stated by the authors",
        notes="Training corpus undocumented upstream; treated as cross-domain.",
    ),
    Method(
        name="coronary_umamba",
        family="specialist",
        tasks=("cca_segmentation",),
        runner="benchmark.runners.nnunet",
        config={
            "weights": str(
                WEIGHTS_ROOT / "coronary-seg-nnunet" / "UMambaBot" / "checkpoint_final.pth"
            ),
            "arch": "umamba_bot",
            "cross_domain": True,
        },
        source="mhyu222/coronary-segmentation-nnunet-umamba (UMambaBot)",
        reported="not stated by the authors",
        notes="Training corpus undocumented upstream; treated as cross-domain.",
    ),
    Method(
        name="coronary_cm_unet",
        family="specialist",
        tasks=("cca_segmentation",),
        runner="benchmark.runners.monai_unet",
        config={
            "weights": str(WEIGHTS_ROOT / "CM-UNet" / "CM-UNet_weights.pth"),
            "arch": "cm_unet",
            "normalization": "zscore_clip",
            "clip_range": (-200, 300),
            "threshold": 0.02,
            "cross_domain": True,
        },
        source="CM-UNet (Mamba-CNN hybrid segmentation)",
        reported="not stated for coronary CTA",
        notes="Architecture resolved from the checkpoint at load time.",
    ),
]


# --------------------------------------------------------------------------
# VISION-LANGUAGE MODELS
# --------------------------------------------------------------------------
#
# Intentionally empty. The VLM registry lives in benchmark/vlms.py, which is what
# run_unified.py, harnesses.py and harness_runner.py import.
#
# A second list of VLMs used to live here and had drifted out of agreement with
# vlms.py in three ways that would each corrupt a comparison:
#   * no pinned revisions, so a row did not identify the weights it scored with;
#   * different names for the same checkpoint (llava_next_mistral vs
#     llava_16_mistral_7b, lingshu vs lingshu_7b), which double-counts a model;
#   * cca_segmentation in the task tuple, which vlms.py deliberately withholds
#     because a dense 3D mask is not answerable by a text-output model.
# It also listed checkpoints that are not loadable here (llava_med, janus_pro,
# huatuo_vision). Keeping one registry is what makes "swap only the harness"
# true. Add VLMs in benchmark/vlms.py.

VLMS: list[Method] = []


ALL_METHODS: list[Method] = SPECIALISTS + VLMS
BY_NAME: dict[str, Method] = {m.name: m for m in ALL_METHODS}

TASKS: tuple[Task, ...] = (
    "cardiosyntax_scoring",
    "cca_segmentation",
    "arcade_segmentation",
    "arcade_stenosis",
)


# --------------------------------------------------------------------------
# CHECKPOINT VARIANTS
# --------------------------------------------------------------------------
# A released model is often a set of weights rather than one file: k-fold
# checkpoints, architecture variants, or multiple training scales. Each is a
# legitimate separate evaluation row, and the spread across them is itself a
# result - it shows how much of a reported number depends on which checkpoint
# the authors happened to publish.
#
# expand_variants() turns one Method into one row per checkpoint. The ensemble
# row is kept alongside the individual folds, since the ensemble is what the
# authors actually released as their model.


def expand_variants(method: Method) -> list[Method]:
    """Expand a method into per-checkpoint variants where that is meaningful.

    Returns [method] unchanged when the method has a single checkpoint.
    """
    variants: list[Method] = [method]

    fold_spec = method.config.get("folds")
    if not fold_spec:
        return variants

    for fold in fold_spec:
        variants.append(
            Method(
                name=f"{method.name}_fold{fold:02d}",
                family=method.family,
                tasks=method.tasks,
                runner=method.runner,
                config={**method.config, "single_fold": fold, "folds": None},
                source=method.source,
                reported=method.reported,
                notes=f"Single fold {fold} of the {method.name} ensemble.",
            )
        )
    return variants


def with_variants() -> list[Method]:
    """The full method list including per-checkpoint variants."""
    expanded: list[Method] = []
    for method in ALL_METHODS:
        expanded.extend(expand_variants(method))
    return expanded


# --------------------------------------------------------------------------
# WEIGHT AVAILABILITY
# --------------------------------------------------------------------------
# Everything here is zero-shot inference from published weights. A method with
# no locally resolvable weights is not evaluable and must never silently degrade
# into a mock or an API call - that would put a fabricated row in a results
# table. weights_status() is the gate, and the runner refuses to proceed without
# it.


def _hf_snapshot_dir(repo: str) -> Path | None:
    """Locate a fully materialised snapshot for a HF repo in the local cache.

    A snapshot counts as usable when it has a config.json and every weight file
    it references resolves to a real blob. Partial blobs elsewhere in the same
    repo cache are ignored: an unrelated in-flight download must not invalidate
    an otherwise complete snapshot.
    """
    cache_dir = HF_CACHE / "hub" / f"models--{repo.replace('/', '--')}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.is_dir():
        return None

    for candidate in sorted(snapshots.iterdir(), reverse=True):
        if not candidate.is_dir() or not (candidate / "config.json").exists():
            continue

        weight_files = list(candidate.glob("*.safetensors")) + list(
            candidate.glob("*.bin")
        )
        if not weight_files:
            continue
        # Snapshot entries are symlinks into blobs/; a dangling link or a
        # sibling .incomplete means that shard is still downloading.
        if any(
            not f.resolve().exists() or f.resolve().with_suffix(".incomplete").exists()
            for f in weight_files
        ):
            continue
        return candidate
    return None


def weights_status(method: Method) -> tuple[bool, str]:
    """Return (available, detail) for a method's weights.

    Checked before any inference so an unavailable method is reported as SKIPPED
    with a reason, never as a score.
    """
    if method.family == "vlm":
        repo = method.config["repo"]
        snapshot = _hf_snapshot_dir(repo)
        if snapshot is None:
            return False, f"no complete snapshot for {repo} in {HF_CACHE}"
        return True, str(snapshot)

    weights = Path(method.config["weights"])
    if not weights.exists():
        return False, f"missing weights: {weights}"
    if weights.is_file():
        size = weights.stat().st_size
        # Git-LFS pointer files are a few hundred bytes and masquerade as
        # weights; this bit us once already on the CardioSYNTAX checkpoint.
        if size < 100_000:
            return False, f"{weights.name} is {size}B, likely a Git-LFS pointer"
        return True, f"{weights} ({size / 1e6:.0f}MB)"
    return True, str(weights)


def available_methods(task: Task | None = None) -> tuple[list[Method], list[tuple[Method, str]]]:
    """Split methods into (evaluable, skipped-with-reason)."""
    pool = [m for m in ALL_METHODS if task is None or task in m.tasks]
    ready: list[Method] = []
    skipped: list[tuple[Method, str]] = []
    for method in pool:
        ok, detail = weights_status(method)
        (ready if ok else skipped).append(method if ok else (method, detail))
    return ready, skipped


def methods_for(task: Task, family: Family | None = None) -> list[Method]:
    """Every method that can run the given task."""
    return [
        m
        for m in ALL_METHODS
        if task in m.tasks and (family is None or m.family == family)
    ]


def resolve(names: list[str] | None, task: Task | None = None) -> list[Method]:
    """Resolve method names to Method objects, validating as we go.

    None means "everything applicable", which is what the one-command run uses.
    """
    if names:
        unknown = [n for n in names if n not in BY_NAME]
        if unknown:
            raise KeyError(
                f"unknown method(s): {', '.join(unknown)}. "
                f"available: {', '.join(sorted(BY_NAME))}"
            )
        selected = [BY_NAME[n] for n in names]
    else:
        selected = list(ALL_METHODS)

    if task:
        selected = [m for m in selected if task in m.tasks]
    return selected
