"""
Input/output contract for CardiomniBench-VD.

This module is the single definition of what a method receives and what it must
return. Every runner goes through it, so no runner reinvents frame sampling,
volume normalization, or mask geometry handling - and no runner can quietly
disagree with another about what "the input" means.

Why this layer exists
---------------------
The two tasks have incompatible native formats, and the two method families need
different views of them:

                    native input                specialist wants   VLM wants
  cardiosyntax      8x .npy cine [T,512,512]    the raw stacks     a few frames
  cca_segmentation  .nii.gz volume [832,832,576] the raw volume    2D renderings

A 3D CTA volume cannot be fed to a 2D vision-language model, and a text model
cannot emit 400k voxels. Rather than let each runner improvise, the conversions
are defined once here and used by all of them. Two runs of the same method are
then comparable, and two different methods see identical input.

Output contract
---------------
Scalar tasks return the number in the prediction dict. Volume tasks write a
NIfTI next to their prediction and reference it by filename; masks are never
inlined, because a 400k-voxel array in JSON is unusable.

Geometry is preserved on write by reusing the input's affine and header. A mask
saved with an identity affine would score as catastrophically misaligned even if
every voxel were correct, so `write_mask` refuses to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Task is defined once, in core. Importing it here rather than redeclaring a
# parallel Literal keeps a single source of truth for the task taxonomy.
from benchmark.core import Task

# --------------------------------------------------------------------------
# Fixed rendering parameters
# --------------------------------------------------------------------------
# These are constants, not tunables. Every VLM sees exactly this view of the
# data, so differences between VLMs reflect the models rather than the
# preprocessing.

# Frames sampled per cine loop for VLM input, evenly spaced across the loop.
# Angiographic contrast filling is progressive, so evenly spaced frames span
# early filling through full opacification.
VLM_FRAMES_PER_VIEW = 4

# Soft-tissue window for CTA. Coronary lumen with contrast sits well inside it.
CTA_WINDOW = (-200.0, 300.0)

# Slab thickness in voxels for maximum-intensity projection. Thick enough that a
# vessel stays continuous across the slab, thin enough not to flatten the tree
# into a single blob.
MIP_SLAB_VOXELS = 40


@dataclass
class CaseInput:
    """Everything a method may read for one case.

    Runners take what they need and ignore the rest. Carrying both the native
    handles and the rendered views in one object keeps the runner signature
    identical across tasks and families.
    """

    task: Task
    case_id: str
    case_dir: Path
    spec: dict[str, Any]

    # cardiosyntax: cine views grouped by artery, as file paths
    views_by_artery: dict[str, list[Path]] = field(default_factory=dict)

    # cca: the volume file and its geometry
    volume_path: Path | None = None
    volume_shape: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None

    # arcade: the single 2D XCA frame and its pixel size. No spacing: a
    # projection has no physical voxel extent, which is why ARCADE metrics are
    # pixel-based while CCA metrics are in millimetres.
    image_path: Path | None = None
    image_shape: tuple[int, ...] | None = None

    def require_volume(self) -> Path:
        """Return the volume path, failing loudly if this is not a volume task."""
        if self.volume_path is None:
            raise ValueError(f"{self.case_id}: no volume for task {self.task}")
        return self.volume_path

    def require_image(self) -> Path:
        """Return the 2D frame path, failing loudly if this is not an image task."""
        if self.image_path is None:
            raise ValueError(f"{self.case_id}: no 2D image for task {self.task}")
        return self.image_path


def load_case_input(case_dir: Path) -> CaseInput:
    """Build the uniform CaseInput for a case directory."""
    import yaml

    with (case_dir / "task.yaml").open() as handle:
        spec = yaml.safe_load(handle)

    task = Task(spec["case_metadata"]["task_type"])
    case_input = CaseInput(
        task=task,
        case_id=spec["case_id"],
        case_dir=case_dir,
        spec=spec,
    )

    if task is Task.CARDIOSYNTAX_SCORING:
        grouped: dict[str, list[Path]] = {"left": [], "right": []}
        for view in spec["input"].get("views") or []:
            relative = view.get("file_path")
            if not relative:
                continue
            path = (case_dir / relative).resolve()
            if not path.exists():
                continue
            artery = str(view.get("artery", "")).strip().upper()
            if artery in ("LCA", "LEFT"):
                grouped["left"].append(path)
            elif artery in ("RCA", "RIGHT"):
                grouped["right"].append(path)
        case_input.views_by_artery = grouped

    elif task is Task.CCA_SEGMENTATION:
        volume = spec["input"]["volume"]
        path = (case_dir / volume["file_path"]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{spec['case_id']}: missing volume {path}")
        case_input.volume_path = path
        case_input.volume_shape = tuple(volume["shape"])
        case_input.spacing = tuple(volume.get("spacing_mm") or ())

    elif task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
        # 2D single-frame XCA: one PNG per case.
        image_rel = spec["input"]["image"]["file_path"]
        image_path = (case_dir / image_rel).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"{spec['case_id']}: missing image {image_path}")
        case_input.image_path = image_path
        img_meta = spec["input"]["image"]
        case_input.image_shape = (img_meta["height"], img_meta["width"])

    else:
        raise ValueError(f"unsupported task_type: {task}")

    return case_input


# --------------------------------------------------------------------------
# Renderings for 2D vision-language models
# --------------------------------------------------------------------------


def sample_cine_frames(
    video_path: Path, n_frames: int = VLM_FRAMES_PER_VIEW
) -> list[np.ndarray]:
    """Evenly sample frames from a cine loop, returned as uint8 grayscale.

    Even spacing is deliberate: it covers the contrast bolus from early filling
    to full opacification, so a model is not handed only pre-contrast frames.
    """
    video = np.load(str(video_path))
    if video.ndim != 3:
        raise ValueError(f"{video_path.name}: expected [T,H,W], got {video.shape}")

    total = video.shape[0]
    count = min(n_frames, total)
    indices = np.linspace(0, total - 1, count).round().astype(int)

    frames = []
    for index in indices:
        frame = video[index].astype(np.float32)
        low, high = np.percentile(frame, [1, 99])
        if high <= low:
            scaled = np.zeros_like(frame)
        else:
            scaled = (np.clip(frame, low, high) - low) / (high - low)
        frames.append((scaled * 255).astype(np.uint8))
    return frames


def volume_to_mip_slabs(
    volume_path: Path,
    axis: int = 2,
    slab_voxels: int = MIP_SLAB_VOXELS,
    max_slabs: int = 12,
) -> list[np.ndarray]:
    """Render a CTA volume as maximum-intensity-projection slabs.

    MIP rather than single slices because a coronary artery crosses a 0.5 mm
    slice in a couple of voxels; a single axial slice shows disconnected dots,
    while a slab shows a continuous vessel a model can actually describe.
    """
    import nibabel as nib

    nii = nib.load(str(volume_path))
    data = nii.get_fdata()

    windowed = np.clip(data, *CTA_WINDOW)
    windowed = (windowed - CTA_WINDOW[0]) / (CTA_WINDOW[1] - CTA_WINDOW[0])

    depth = windowed.shape[axis]
    starts = list(range(0, depth, slab_voxels))
    if len(starts) > max_slabs:
        keep = np.linspace(0, len(starts) - 1, max_slabs).round().astype(int)
        starts = [starts[i] for i in keep]

    slabs = []
    for start in starts:
        stop = min(start + slab_voxels, depth)
        slab = windowed.take(indices=range(start, stop), axis=axis).max(axis=axis)
        slabs.append((slab * 255).astype(np.uint8))
    return slabs


def frames_to_pil(frames: list[np.ndarray]):
    """Convert grayscale arrays to RGB PIL images, as image encoders expect."""
    from PIL import Image

    return [Image.fromarray(frame).convert("RGB") for frame in frames]


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------

# Keys every prediction must carry, plus the per-task payload key.
REQUIRED_KEYS: dict[Task, tuple[str, ...]] = {
    Task.CARDIOSYNTAX_SCORING: ("case_id", "task_type", "syntax_score"),
    Task.CCA_SEGMENTATION: ("case_id", "task_type", "mask_file"),
}

MASK_FILENAME = "mask.nii.gz"


def write_mask(
    mask: np.ndarray,
    output_dir: Path,
    reference_volume: Path,
    filename: str = MASK_FILENAME,
) -> Path:
    """Write a binary mask as NIfTI, inheriting the input's geometry.

    The affine and header are copied from the reference volume. Writing a mask
    with a fresh identity affine is the classic way to produce a perfectly
    shaped, perfectly worthless result: shape matches, world coordinates do not,
    and every overlap metric collapses.
    """
    import nibabel as nib

    reference = nib.load(str(reference_volume))
    if mask.shape != reference.shape:
        raise ValueError(
            f"mask shape {mask.shape} != reference {reference.shape}; "
            "a resampled mask must be restored to input geometry before writing"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    image = nib.Nifti1Image(
        mask.astype(np.uint8), affine=reference.affine, header=reference.header
    )
    image.set_data_dtype(np.uint8)
    nib.save(image, str(path))
    return path


def validate_prediction(prediction: dict[str, Any], task: Task) -> None:
    """Raise if a prediction does not satisfy the output contract.

    Called before scoring so a malformed prediction is reported as a failure
    with a reason, instead of silently scoring as zero and looking like a real
    (bad) result.
    """
    missing = [k for k in REQUIRED_KEYS[task] if k not in prediction]
    if missing:
        raise ValueError(f"prediction missing required keys: {missing}")

    if task is Task.CARDIOSYNTAX_SCORING:
        score = prediction["syntax_score"]
        if score is None:
            raise ValueError("syntax_score is None")
        value = float(score)
        if value != value:
            raise ValueError("syntax_score is NaN")
        # The SYNTAX scale has no negative values and no realistic score above
        # ~80; anything outside that is a parsing failure, not a prediction.
        if not -0.001 <= value <= 200.0:
            raise ValueError(f"syntax_score {value} outside plausible range")

    elif task is Task.CCA_SEGMENTATION:
        if not prediction["mask_file"]:
            raise ValueError("mask_file is empty")
