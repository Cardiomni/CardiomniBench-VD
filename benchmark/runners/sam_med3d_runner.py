"""
SAM-Med3D runner for coronary CTA segmentation.

Critical Limitation
-------------------
SAM-Med3D was designed for organ-scale segmentation (liver, kidney, spleen) and
operates on 128³ voxel crops at 1.5mm spacing (192mm physical cube). Coronary
arteries span 416×416×288mm cardiac volumes, so a single centered crop captures
<25% of the vessel tree.

This runner uses **gold-mask-derived point prompts**, making evaluation **not
zero-shot**. The official SAM-Med3D README explicitly states:
    "Ground-truth labels are required to generate prompt points."

Strategy: 128³ sliding patches with gold-guided prompts
--------------------------------------------------------
1. Resample input volume to 1.5mm isotropic (SAM-Med3D's native spacing)
2. Divide resampled volume into non-overlapping 128³ patches
3. For each patch containing gold foreground:
   - Sample N=5 positive point prompts from gold mask within that patch
   - Run SAM-Med3D inference with those prompts
4. Assemble per-patch predictions into full-volume mask
5. Resample back to original grid and write output

This is **not a fair comparison** to zero-shot VLMs or specialists that see the
full volume in one pass. It is included for completeness but should be interpreted
as "gold-guided foundation model upper bound" rather than a true baseline.

Architecture Note
-----------------
PromptEncoder3D only supports point embeddings (indices 0,1 for negative/positive).
The `_embed_boxes()` method exists but references `self.point_embeddings[2]` and
`[3]` which are never created, so bounding box prompts will raise IndexError.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from benchmark.specialists import CoronaryCTASegmenter

from benchmark.core import Prediction
from benchmark.io_spec import CaseInput, write_mask
from benchmark.scoring import load_gold

# Add SAM-Med3D source to path
SAM_SRC = Path(__file__).parents[2] / "algorithms" / "specialist_models" / "sam_med3d_src"
if SAM_SRC.exists():
    sys.path.insert(0, str(SAM_SRC))

from segment_anything.build_sam3D import build_sam3D_vit_b_ori


def _resample_volume(
    volume: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Resample volume to target spacing using trilinear interpolation."""
    factors = [o / t for o, t in zip(original_spacing, target_spacing)]
    
    # torch.nn.functional.interpolate expects (N, C, D, H, W)
    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    
    new_shape = tuple(int(s * f) for s, f in zip(volume.shape, factors))
    resampled = F.interpolate(
        volume_tensor,
        size=new_shape,
        mode='trilinear',
        align_corners=False,
    )
    
    return resampled[0, 0].numpy()


def _sample_points_from_mask(
    mask_patch: np.ndarray,
    n_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N positive point prompts from a binary mask patch.
    
    Returns:
        coords: (1, N, 3) array of [x, y, z] coordinates
        labels: (1, N) array of point labels (all 1 = positive)
    """
    fg_coords = np.argwhere(mask_patch > 0)  # (M, 3) in [z, y, x] order
    
    if len(fg_coords) == 0:
        # No foreground; return center point as fallback
        center = np.array([mask_patch.shape[2] // 2,
                          mask_patch.shape[1] // 2,
                          mask_patch.shape[0] // 2])
        coords = center.reshape(1, 1, 3)
        labels = np.ones((1, 1), dtype=np.int64)
        return coords, labels
    
    n_actual = min(n_points, len(fg_coords))
    sampled_idx = np.random.choice(len(fg_coords), n_actual, replace=False)
    sampled = fg_coords[sampled_idx]  # (N, 3) in [z, y, x]
    
    # SAM expects [x, y, z] order
    coords = sampled[:, [2, 1, 0]].reshape(1, n_actual, 3)
    labels = np.ones((1, n_actual), dtype=np.int64)
    
    return coords, labels


def _znormalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalization, optionally masked to foreground region."""
    if mask is not None and mask.sum() > 0:
        fg = volume[mask > 0]
        mean, std = fg.mean(), fg.std()
    else:
        mean, std = volume.mean(), volume.std()
    
    return (volume - mean) / (std + 1e-8)


def check_available(method: CoronaryCTASegmenter) -> tuple[bool, str]:
    """Preflight for SAM-Med3D.

    The checkpoint is useless without the vendored ``segment_anything`` source,
    since the architecture is built by ``build_sam3D`` rather than being
    reconstructible from the state dict alone.
    """
    source_root = (
        Path(__file__).resolve().parents[2]
        / "algorithms" / "specialist_models" / "sam_med3d_src"
    )
    if not (source_root / "segment_anything" / "build_sam3D.py").exists():
        return False, "sam_med3d runner needs vendored segment_anything source"
    return True, ""


def predict(
    method: CoronaryCTASegmenter,
    case: CaseInput,
    output_dir: Path,
    device: str,
) -> Prediction:
    """Run SAM-Med3D with 128³ sliding patches and gold-guided point prompts."""
    
    # Load input volume
    volume_path = case.require_volume()
    nii = nib.load(str(volume_path))
    volume = nii.get_fdata().astype(np.float32)
    original_shape = volume.shape
    original_spacing = tuple(float(z) for z in nii.header.get_zooms()[:3])
    
    # Load gold mask for prompt generation (required by SAM-Med3D design)
    gold = load_gold(case.case_dir)
    gold_mask = gold.mask  # same shape as volume
    
    # Resample to 1.5mm isotropic (SAM-Med3D's native spacing)
    target_spacing = (1.5, 1.5, 1.5)
    volume_resampled = _resample_volume(volume, original_spacing, target_spacing)
    gold_resampled = _resample_volume(
        gold_mask.astype(np.float32),
        original_spacing,
        target_spacing
    )
    gold_resampled = (gold_resampled > 0.5).astype(np.uint8)
    
    # Z-normalize using foreground statistics
    volume_normalized = _znormalize(volume_resampled, gold_resampled)
    
    # Load SAM-Med3D model
    # The checkpoint wraps state dict in 'model_state_dict' key; build_sam3D
    # expects unwrapped dict, so we load and unwrap manually.
    checkpoint = torch.load(str(method.weights_path), map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # build_sam3D_vit_b_ori uses 128³ input (confirmed by pos_embed shape (1,8,8,8,768))
    model = build_sam3D_vit_b_ori(checkpoint=None)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    
    # Divide into 128³ patches
    patch_size = 128
    d, h, w = volume_normalized.shape
    
    # Pad to make evenly divisible by patch_size
    pad_d = (patch_size - d % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    volume_padded = np.pad(
        volume_normalized,
        ((0, pad_d), (0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0,
    )
    gold_padded = np.pad(
        gold_resampled,
        ((0, pad_d), (0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0,
    )
    
    d_pad, h_pad, w_pad = volume_padded.shape
    prediction_full = np.zeros(volume_padded.shape, dtype=np.uint8)
    
    n_patches_total = 0
    n_patches_processed = 0
    
    # Process each 128³ patch
    for z in range(0, d_pad, patch_size):
        for y in range(0, h_pad, patch_size):
            for x in range(0, w_pad, patch_size):
                n_patches_total += 1
                
                patch_vol = volume_padded[
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size,
                ]
                patch_gold = gold_padded[
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size,
                ]
                
                # Skip patches with no gold foreground
                if patch_gold.sum() == 0:
                    continue
                
                n_patches_processed += 1
                
                # Sample point prompts from gold mask
                point_coords, point_labels = _sample_points_from_mask(patch_gold, n_points=5)
                
                # Prepare input tensor (1, 1, D, H, W)
                input_tensor = torch.from_numpy(patch_vol).float().unsqueeze(0).unsqueeze(0)
                input_tensor = input_tensor.to(device)
                
                # Inference
                with torch.no_grad():
                    image_embeddings = model.image_encoder(input_tensor)
                    
                    points_coords_t = torch.from_numpy(point_coords).float().to(device)
                    points_labels_t = torch.from_numpy(point_labels).long().to(device)
                    
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=[points_coords_t, points_labels_t],
                        boxes=None,
                        masks=None,
                    )
                    
                    low_res_masks, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    
                    # Upsample to patch resolution
                    masks_hr = F.interpolate(
                        low_res_masks,
                        size=patch_vol.shape,
                        mode='trilinear',
                        align_corners=False,
                    )
                    
                    mask_patch = (torch.sigmoid(masks_hr) > 0.5).cpu().numpy()[0, 0]
                
                # Write patch prediction into full volume
                prediction_full[
                    z:z+patch_size,
                    y:y+patch_size,
                    x:x+patch_size,
                ] = mask_patch.astype(np.uint8)
    
    # Remove padding
    prediction_resampled = prediction_full[:d, :h, :w]
    
    # Resample back to original grid
    prediction_original = _resample_volume(
        prediction_resampled.astype(np.float32),
        target_spacing,
        original_spacing,
    )
    prediction_original = (prediction_original > 0.5).astype(np.uint8)
    
    # Crop/pad to exact original shape (resampling may introduce rounding)
    if prediction_original.shape != original_shape:
        fitted = np.zeros(original_shape, dtype=np.uint8)
        region = tuple(slice(0, min(a, o)) for a, o in zip(prediction_original.shape, original_shape))
        fitted[region] = prediction_original[region]
        prediction_original = fitted
    
    # Write mask
    mask_path = write_mask(prediction_original, output_dir, volume_path)
    
    diagnostics = {
        "prompt_strategy": "gold-derived points per patch",
        "n_points_per_patch": 5,
        "patch_size": patch_size,
        "target_spacing_mm": list(target_spacing),
        "original_spacing_mm": list(original_spacing),
        "resampled_shape": list(volume_normalized.shape),
        "n_patches_total": n_patches_total,
        "n_patches_processed": n_patches_processed,
        "n_patches_skipped": n_patches_total - n_patches_processed,
        "foreground_fraction": float(prediction_original.mean()),
    }
    
    return Prediction(
        case_id=case.case_id,
        task=case.task,
        mask_path=mask_path,
        diagnostics=diagnostics,
    )
