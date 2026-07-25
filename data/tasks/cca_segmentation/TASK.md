# Task: CCA — 3D Coronary Vessel Segmentation (CTA)

> **One line**: Given a 3D coronary CT angiography volume, segment the whole
> coronary artery tree at the voxel level.

## Task type
`cca_segmentation` — 3D binary semantic segmentation.

## Source
CCA dataset (Yang et al. 2023, arXiv:2305.04208), public **training** split
(only 20 of 200 cases are public). All 20 used. Source: `Datasets/CCA/train/`.

## Input (what the agent reads)
One 3D CTA volume.

- `input.modality`: `"CTA"`
- `input.volume.file_path`: `"image.nii.gz"` (relative to case folder)
- `input.volume.shape`: e.g. `[832, 832, 576]`
- `input.volume.spacing_mm`: e.g. `[0.5, 0.5, 0.5]` (isotropic)

## Output (what the agent writes → `prediction.json` + a mask file)
A binary 3D mask, same shape as the input volume:

- `1` = coronary artery voxel, `0` = background.
- Written as a NIfTI to the prediction directory; `prediction.json` references it.
- Foreground is sparse (~0.1% of voxels) — thin-vessel segmentation.

**Not in this task**: segment naming, stenosis, or grading. Binary vessel/background only.

## Gold standard (stripped from `task_spec.json`)
`task.yaml.gold_standard.label_file` → the source `train/labels/<n>.nii.gz`
(3-radiologist consensus + 1 arbiter). Binary 0/1, same shape as input.

## Metric (deferred)
**Dice coefficient** (primary), aux **clDice** (centerline/connectivity) and
**Hausdorff distance**.

## Status note (important)
CCA is **CTA, not DSA**. Per the project pivot (DSA-only main line), it is **out
of the primary benchmark** and serves as a coronary-anatomy prior / segmentation
tool-training source (e.g. for a vessel-segmentation tool). Kept here for
completeness and future work; do not report it as a DSA result.

## Cardiomni relevance
Anatomy grounding only. A vessel-segmentation tool trained/validated here could be
exposed to the agent, but the CTA modality itself is not part of the DSA diagnosis
task.
