"""TOML-backed method configuration.

Why this module exists
----------------------
Preprocessing parameters decide whether a checkpoint produces a usable mask or
noise. The ImageCAS U-Net dropped from a reported Dice of 0.788 to 0.021 on CCA
purely because the runner guessed its own HU window, skipped resampling, and used
argmax where upstream uses a sigmoid threshold. Nothing in the code flagged the
mismatch, because the parameters lived as literals next to the inference call.

So each method now declares its pipeline in ``methods/<name>.toml``, transcribed
from the upstream repository with the source noted in comments. Runners read the
TOML; they do not carry defaults of their own. A parameter that is wrong is then
wrong in one visible place, and a parameter that is missing raises instead of
silently falling back to a plausible-looking guess.

nnU-Net style checkpoints are the exception: their ``plans.json`` already is a
machine-readable contract, so the TOML records those values as documentation and
the runner keeps reading ``plans.json`` itself. Duplicating them here as live
inputs would create two sources of truth that can drift apart.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Valid strings for [preprocess].normalize in 2D methods.
NORMALIZE_MODES = frozenset(["none", "divide255", "minmax", "zscore"])

# tomllib is stdlib from Python 3.11. The inference environment for these
# checkpoints (conda env gkp-gsa) is 3.10, so fall back to the tomli backport
# and finally to a clear error rather than failing inside a runner mid-case.
if sys.version_info >= (3, 11):  # pragma: no cover - version dependent
    import tomllib as _toml
else:  # pragma: no cover - version dependent
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError as _exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "reading method TOMLs on Python < 3.11 needs the 'tomli' backport: "
            "pip install tomli"
        ) from _exc

#: Method TOMLs live here, one file per method, named after the method.
METHODS_DIR = Path(__file__).resolve().parent.parent / "methods"


class MethodConfigError(RuntimeError):
    """Raised when a method TOML is missing, malformed, or incomplete."""


def _require(table: dict[str, Any], key: str, where: str) -> Any:
    """Fetch ``key`` or explain exactly which file and table lacked it."""
    if key not in table:
        raise MethodConfigError(f"{where} is missing required key '{key}'")
    return table[key]


@dataclass(frozen=True)
class PreprocessConfig:
    """Deterministic image preparation, in the order the transforms apply."""

    pixdim: tuple[float, float, float]
    orientation: str
    window_a_min: float
    window_a_max: float
    normalize_mode: str

    body_crop: bool = True
    body_threshold_hu: float = -500.0
    body_margin: tuple[int, int, int] = (8, 8, 8)

    normalize_nonzero: bool = True
    normalize_channel_wise: bool = True

    denoise_tv: bool = False
    denoise_weight: float = 0.06
    denoise_iter: int = 20

    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_kernel: tuple[int, int] = (64, 64)
    clahe_axis: int = 2

    vesselness: bool = False
    vesselness_sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
    vesselness_alpha: float = 0.5
    vesselness_beta: float = 0.5
    vesselness_gamma: float = 15.0
    vesselness_keep_original: bool = True

    @property
    def in_channels(self) -> int:
        """Input channel count implied by the vesselness settings."""
        if not self.vesselness:
            return 1
        return 2 if self.vesselness_keep_original else 1

    @classmethod
    def from_toml(cls, table: dict[str, Any], where: str) -> "PreprocessConfig":
        if not table:
            raise MethodConfigError(f"{where} has no [preprocess] table")
        return cls(
            pixdim=tuple(float(v) for v in _require(table, "pixdim", where)),  # type: ignore[arg-type]
            orientation=str(_require(table, "orientation", where)),
            window_a_min=float(_require(table, "window_a_min", where)),
            window_a_max=float(_require(table, "window_a_max", where)),
            normalize_mode=str(_require(table, "normalize_mode", where)),
            body_crop=bool(table.get("body_crop", True)),
            body_threshold_hu=float(table.get("body_threshold_hu", -500.0)),
            body_margin=tuple(int(v) for v in table.get("body_margin", (8, 8, 8))),  # type: ignore[arg-type]
            normalize_nonzero=bool(table.get("normalize_nonzero", True)),
            normalize_channel_wise=bool(table.get("normalize_channel_wise", True)),
            denoise_tv=bool(table.get("denoise_tv", False)),
            denoise_weight=float(table.get("denoise_weight", 0.06)),
            denoise_iter=int(table.get("denoise_iter", 20)),
            clahe=bool(table.get("clahe", False)),
            clahe_clip=float(table.get("clahe_clip", 2.0)),
            clahe_kernel=tuple(int(v) for v in table.get("clahe_kernel", (64, 64))),  # type: ignore[arg-type]
            clahe_axis=int(table.get("clahe_axis", 2)),
            vesselness=bool(table.get("vesselness", False)),
            vesselness_sigmas=tuple(
                float(v) for v in table.get("vesselness_sigmas", (1.0, 2.0, 3.0, 4.0))
            ),
            vesselness_alpha=float(table.get("vesselness_alpha", 0.5)),
            vesselness_beta=float(table.get("vesselness_beta", 0.5)),
            vesselness_gamma=float(table.get("vesselness_gamma", 15.0)),
            vesselness_keep_original=bool(table.get("vesselness_keep_original", True)),
        )


@dataclass(frozen=True)
class VideoPreprocessConfig:
    """Frame sampling and normalisation for cine-video methods.

    Video regression models share none of the CT segmentation knobs: there is no
    voxel spacing, no HU window, no anatomical orientation. Forcing them through
    :class:`PreprocessConfig` would mean inventing meaningless values for
    ``pixdim`` and ``window_a_min``, so they get their own schema and the loader
    dispatches on the presence of ``frames_per_clip``.
    """

    frames_per_clip: int
    output_size: tuple[int, int]
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]

    frame_sampling: str = "center"
    padding_strategy: str = "repeat"
    input_size: tuple[int, int] = (512, 512)
    resize_mode: str = "bilinear_antialias"
    rescale_uint16: bool = True
    rescale_method: str = "per_video_max"
    channel_replication: int = 3

    @classmethod
    def from_toml(cls, table: dict[str, Any], where: str) -> "VideoPreprocessConfig":
        return cls(
            frames_per_clip=int(_require(table, "frames_per_clip", where)),
            output_size=tuple(int(v) for v in _require(table, "output_size", where)),  # type: ignore[arg-type]
            normalize_mean=tuple(float(v) for v in _require(table, "normalize_mean", where)),
            normalize_std=tuple(float(v) for v in _require(table, "normalize_std", where)),
            frame_sampling=str(table.get("frame_sampling", "center")),
            padding_strategy=str(table.get("padding_strategy", "repeat")),
            input_size=tuple(int(v) for v in table.get("input_size", (512, 512))),  # type: ignore[arg-type]
            resize_mode=str(table.get("resize_mode", "bilinear_antialias")),
            rescale_uint16=bool(table.get("rescale_uint16", True)),
            rescale_method=str(table.get("rescale_method", "per_video_max")),
            channel_replication=int(table.get("channel_replication", 3)),
        )


@dataclass(frozen=True)
class Image2DPreprocessConfig:
    """Padding and resizing for single-frame 2D methods (X-ray angiography).

    Third modality alongside :class:`PreprocessConfig` (3D volumes) and
    :class:`VideoPreprocessConfig` (cine clips), and separate for the same reason:
    a 2D projection has no voxel spacing, no HU window and no anatomical
    orientation, so routing it through the CT schema would mean inventing values
    that mean nothing. The loader dispatches on the presence of ``model_input``.

    ``pad_to`` is the load-bearing knob. Upstream CM-UNet pads to 1536 before
    resizing to 256, which is close to a no-op on natively large angiograms but
    costs 3x resolution on a 512 frame. Setting it to 0 disables padding. Both
    settings are registered as separate methods rather than one being chosen for
    us; see methods/README.md.

    ``normalize`` is the other load-bearing knob, and it was originally set to
    false by reading only ``Finetuning/dataset.py``, which indeed applies no
    intensity scaling. That was the wrong layer to read: the dataset loads ``.npy``
    files that ``data_processing/data_processing.ipynb`` already put through
    ``Unsharper(radius=60, amount=3)`` and ``Intensity_normalizer`` (per-image
    z-score). Feeding raw uint8 instead measured Dice 0.000 on ARCADE, versus
    0.709 with z-score and 0.726 with unsharp+z-score, on the same weights.
    """

    model_input: int

    #: Square canvas to zero-pad onto before resizing. 0 disables padding.
    pad_to: int = 0
    #: Intensity normalisation applied after resizing. See NORMALIZE_MODES.
    #: "none" reproduces what the dataset layer does; "zscore" reproduces what the
    #: weights were actually trained on. Accepts legacy booleans.
    normalize: str = "none"
    #: skimage unsharp_mask radius, applied before normalisation. 0 disables.
    unsharp_radius: float = 0.0
    #: skimage unsharp_mask amount. Only used when unsharp_radius > 0.
    unsharp_amount: float = 0.0

    @classmethod
    def from_toml(cls, table: dict[str, Any], where: str) -> "Image2DPreprocessConfig":
        raw_normalize = table.get("normalize", "none")
        # Booleans are accepted so an older TOML keeps loading, mapping true to
        # the upstream scheme rather than to an arbitrary one.
        if isinstance(raw_normalize, bool):
            normalize = "zscore" if raw_normalize else "none"
        else:
            normalize = str(raw_normalize).lower()
        if normalize not in NORMALIZE_MODES:
            raise ValueError(
                f"{where}: normalize={raw_normalize!r} is not one of "
                f"{sorted(NORMALIZE_MODES)}"
            )
        return cls(
            model_input=int(_require(table, "model_input", where)),
            pad_to=int(table.get("pad_to", 0)),
            normalize=normalize,
            unsharp_radius=float(table.get("unsharp_radius", 0.0)),
            unsharp_amount=float(table.get("unsharp_amount", 0.0)),
        )


@dataclass(frozen=True)
class InstanceConfig:
    """How a binary mask becomes a labelled instance list.

    Instance-list tasks (ARCADE) need connected-component decomposition and a
    label per component. A binary segmenter has no label to give, so ``label`` is
    the placeholder it stamps on every instance, and the resulting mismatch
    against gold SYNTAX ids is a documented limitation rather than a bug.

    The label a binary mask should carry depends on what the task's gold calls its
    single class: ``arcade_stenosis`` uses ``"stenosis"``, while
    ``arcade_segmentation`` uses 25 distinct ids. A wrong label scores 0 under
    label-aware matching for a reason that has nothing to do with the model, so
    ``per_task`` is an opt-in override rather than forcing every method to know
    every task's vocabulary.
    """

    label: str = "vessel"
    min_component_pixels: int = 20
    per_task: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, table: dict[str, Any]) -> "InstanceConfig":
        per_task_raw = table.get("per_task") or {}
        return cls(
            label=str(table.get("label", "vessel")),
            min_component_pixels=int(table.get("min_component_pixels", 20)),
            per_task={str(k): str(v) for k, v in per_task_raw.items()},
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Sliding-window settings and how logits become a decision."""

    roi_size: tuple[int, int, int]
    sw_batch_size: int = 4
    overlap: float = 0.5
    mode: str = "gaussian"
    amp: bool = True

    decision_rule: str = "threshold"
    foreground_class: int = 1
    threshold: float = 0.5

    @classmethod
    def from_toml(
        cls, table: dict[str, Any], where: str, *, require_roi: bool = True
    ) -> "InferenceConfig":
        if not table:
            raise MethodConfigError(f"{where} has no [inference] table")
        # nnU-Net methods keep their patch size in plans.json; the TOML only
        # documents it under a *_ref name so the two cannot silently disagree.
        roi = table.get("roi_size") or table.get("roi_size_ref")
        if roi is None:
            if require_roi:
                raise MethodConfigError(
                    f"{where} [inference] needs roi_size (or roi_size_ref for "
                    "plans.json-driven methods)"
                )
            # Video methods have no spatial patch; a sentinel keeps the field
            # total while still failing loudly if a runner tries to slide a
            # window with it.
            roi = (0, 0, 0)
        return cls(
            roi_size=tuple(int(v) for v in roi),  # type: ignore[arg-type]
            sw_batch_size=int(table.get("sw_batch_size", 4)),
            overlap=float(table.get("overlap", 0.5)),
            mode=str(table.get("mode", "gaussian")),
            amp=bool(table.get("amp", True)),
            decision_rule=str(table.get("decision_rule", "threshold")),
            foreground_class=int(table.get("foreground_class", 1)),
            threshold=float(table.get("threshold", 0.5)),
        )


@dataclass(frozen=True)
class PostprocessConfig:
    """Mask cleanup applied after the decision rule."""

    min_component_size: int = 0
    connectivity: int = 26
    merge_classes: bool = False
    output_dtype: str = "uint8"

    @classmethod
    def from_toml(cls, table: dict[str, Any]) -> "PostprocessConfig":
        table = table or {}
        return cls(
            min_component_size=int(table.get("min_component_size", 0)),
            connectivity=int(table.get("connectivity", 26)),
            merge_classes=bool(table.get("merge_classes", False)),
            output_dtype=str(table.get("output_dtype", "uint8")),
        )


@dataclass(frozen=True)
class MethodConfig:
    """One method's full contract, as transcribed from its upstream source."""

    name: str
    family: str
    tasks: tuple[str, ...]
    runner: str | None
    weights_path: Path | None
    provenance: dict[str, Any] = field(default_factory=dict)
    preprocess: PreprocessConfig | None = None
    #: Set instead of :attr:`preprocess` for cine-video methods. Exactly one of
    #: the three (preprocess / video_preprocess / image2d_preprocess) is
    #: populated, decided by the shape of the TOML's [preprocess].
    video_preprocess: VideoPreprocessConfig | None = None
    #: Set instead of :attr:`preprocess` for single-frame 2D image methods.
    image2d_preprocess: Image2DPreprocessConfig | None = None
    #: For methods that output instance lists: how to decompose and label.
    instances: InstanceConfig | None = None
    #: For methods that output scalar decisions: how probabilities become classes.
    decision: dict[str, Any] = field(default_factory=dict)
    #: Always present for 3D/video methods: [inference] is required by the loader.
    #: 2D image methods that work on a full frame may leave this as the default.
    inference: InferenceConfig = field(default_factory=lambda: InferenceConfig(roi_size=(96, 96, 96)))
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    #: Raw ``[architecture]`` table: constructor arguments for the network, as
    #: transcribed from the upstream model definition. Kept as a plain dict
    #: because each runner knows its own architecture's signature.
    architecture: dict[str, Any] = field(default_factory=dict)
    extra_paths: dict[str, Path] = field(default_factory=dict)
    source_file: Path | None = None

    def require_preprocess(self) -> PreprocessConfig:
        if self.preprocess is None:
            raise MethodConfigError(
                f"method '{self.name}' has no [preprocess] table; its runner "
                "needs one to avoid guessing intensity and spacing settings"
            )
        return self.preprocess

    def require_inference(self) -> InferenceConfig:
        """Kept for symmetry with :meth:`require_preprocess`; never raises."""
        return self.inference


def load_method_config(
    name: str, methods_dir: str | Path | None = None, repo_root: str | Path | None = None
) -> MethodConfig:
    """Load ``methods/<name>.toml``.

    Relative paths inside the TOML resolve against the repository root, so a
    config stays valid regardless of the process working directory.
    """
    methods_dir = Path(methods_dir) if methods_dir else METHODS_DIR
    root = Path(repo_root) if repo_root else methods_dir.parent
    path = methods_dir / f"{name}.toml"
    if not path.is_file():
        raise MethodConfigError(
            f"no method config at {path}. Every runner-backed method needs one "
            "so its preprocessing is reviewable instead of hardcoded."
        )

    try:
        doc = _toml.loads(path.read_text(encoding="utf-8"))
    except _toml.TOMLDecodeError as exc:
        raise MethodConfigError(f"{path} is not valid TOML: {exc}") from exc

    method = doc.get("method") or {}
    if not method:
        raise MethodConfigError(f"{path} has no [method] table")

    declared = str(_require(method, "name", str(path)))
    if declared != name:
        raise MethodConfigError(
            f"{path} declares name='{declared}' but the file is named '{name}.toml'"
        )

    def _resolve(value: Any) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else (root / p)

    weights = method.get("weights_path")
    extra = {
        key: _resolve(value)
        for key, value in method.items()
        if key.endswith("_path") and key != "weights_path"
    }

    preprocess_table = doc.get("preprocess") or {}
    # A leading-underscore key marks a documentation-only table (nnU-Net style,
    # where plans.json remains authoritative), so it is not parsed as live input.
    documentation_only = any(k.startswith("_") for k in preprocess_table)
    # Video methods are recognised by frame sampling rather than by a family
    # string, so a mislabelled [method] family cannot route a clip through the
    # CT transform chain.
    is_video = "frames_per_clip" in preprocess_table
    # Single-frame 2D methods (X-ray angiography) are recognised by model_input.
    # They have no sliding window: the whole frame is one forward pass.
    is_image2d = "model_input" in preprocess_table

    # [inference] is mandatory for volumetric and video methods. A method with no
    # declared sliding-window and decision rule cannot be run reproducibly, and
    # defaulting the decision rule is exactly how argmax silently replaced
    # upstream's 0.5 threshold before. 2D single-frame methods are exempt: there
    # is no window to declare, and their decision rule lives in [decision].
    inference_table = doc.get("inference") or {}
    if not inference_table and not is_image2d:
        raise MethodConfigError(
            f"{path} has no [inference] table; declare roi_size and the "
            "decision rule instead of relying on runner defaults"
        )
    if is_image2d and not (doc.get("decision") or {}):
        raise MethodConfigError(
            f"{path} is a 2D method with no [decision] table; declare the rule "
            "that turns logits into a mask instead of relying on runner defaults"
        )

    return MethodConfig(
        name=declared,
        family=str(method.get("family", "specialist")),
        tasks=tuple(str(t) for t in method.get("tasks", ())),
        runner=method.get("runner"),
        weights_path=_resolve(weights) if weights else None,
        provenance=dict(method.get("provenance") or doc.get("method", {}).get("provenance") or {}),
        preprocess=(
            None
            if documentation_only or is_video or is_image2d
            else PreprocessConfig.from_toml(preprocess_table, str(path))
        ),
        video_preprocess=(
            VideoPreprocessConfig.from_toml(preprocess_table, str(path)) if is_video else None
        ),
        image2d_preprocess=(
            Image2DPreprocessConfig.from_toml(preprocess_table, str(path))
            if is_image2d
            else None
        ),
        instances=(
            InstanceConfig.from_toml(doc["instances"]) if doc.get("instances") else None
        ),
        decision=dict(doc.get("decision") or {}),
        inference=(
            InferenceConfig.from_toml(
                inference_table, str(path), require_roi=not is_video
            )
            if inference_table
            else InferenceConfig(roi_size=(0, 0, 0))
        ),
        postprocess=PostprocessConfig.from_toml(doc.get("postprocess") or {}),
        architecture=dict(doc.get("architecture") or {}),
        extra_paths=extra,
        source_file=path,
    )


def list_method_configs(methods_dir: str | Path | None = None) -> list[str]:
    """Names of every method that ships a TOML, sorted."""
    methods_dir = Path(methods_dir) if methods_dir else METHODS_DIR
    if not methods_dir.is_dir():
        return []
    return sorted(p.stem for p in methods_dir.glob("*.toml"))
