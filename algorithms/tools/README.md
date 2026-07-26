# Cardiomni Agent Tool Suite

Callable specialist tools for the Cardiomni agent's four-stage SOP. These are
**tools the agent calls**, not baselines the harness scores (PROPOSAL.md §2.6, §5).

Every number in this document was measured on real data with
`/opt/anaconda3/envs/gkp-gsa/bin/python`, not estimated.

```python
from algorithms.tools import segment_vessels, quantify_stenosis, detect_stenosis
```

Requires torch + MONAI + scipy. On this host use
`/opt/anaconda3/envs/gkp-gsa/bin/python` (`.venv` was still syncing at the time of
writing). GPU is shared: pin an idle card, never `cuda:0`.

---

## `segment_vessels` — 2D XCA vessel segmentation

```python
segment_vessels(
    image_path: str | Path,
    device: str = "cuda:1",
    method_name: str = "coronary_cm_unet_native",
    threshold: float | None = None,
) -> np.ndarray  # (H, W) bool, same shape as the input frame
```

Measured on `data/tasks/arcade_segmentation/cases/case_arcade_seg_0001_101-5`:

```
mask shape: (512, 512)   dtype: bool
foreground: 7172 / 262144 = 2.7359%
```

Gold vessel occupancy on the same case is ~2.54%, so the mask is anatomically
plausible in scale — neither empty nor saturated.

### What it can do
Separate vessel from background, at the input frame's own resolution. Padding and
resizing are undone before returning, so the mask indexes the original image
directly. Useful as geometry input for **Stage 4** quantification.

### What it cannot do
**It cannot name SYNTAX segments.** CM-UNet is a binary detector; its label space
is `{0=background, 1=vessel}` and contains no segment vocabulary at all. Measured
across the full 222-case ARCADE run:

```
label-aware F1        0.0000
label_set_precision   0.0000
label_set_recall      0.0000
```

Precision and recall are both zero, not merely low — the predicted label set and
the gold label set are disjoint. This is by construction, not a tuning failure.

**Consequence for Stage 2 (systematic segment scan):** this tool does not satisfy
the naming requirement. Segment identity has to come from somewhere else (VLM
reasoning over the mask, an anatomical atlas, or a labelled segmenter — none of
which currently exists in this repo).

### Capability boundary, machine-readable
```python
from algorithms.tools.vessel_segmentation import segmentation_metadata
segmentation_metadata()["can_name_segments"]   # False
segmentation_metadata()["capability_boundary"] # prose statement
```
Worth copying into the agent's `reasoning_trace`: it makes "I know this tool
cannot name segments" auditable evidence rather than an implicit assumption.

### Variants
`method_name` selects a `methods/*.toml`. Default `coronary_cm_unet_native`
(`pad_to = 0`, resize 512→256 directly). The alternative `coronary_cm_unet`
reproduces upstream's pad-to-1536 first, which leaves anatomy occupying ~85×85 px
of a 256×256 input. All parameters (pad, normalize, unsharp, decision rule) are
read from the TOML at call time; nothing is hardcoded here.

---

## `quantify_stenosis` — QCA-style diameter measurement

```python
quantify_stenosis(
    image_path: str | Path,
    roi_coords: tuple[int, int, int, int],  # (x1, y1, x2, y2) — REQUIRED
    pixel_spacing: float = 0.2,             # mm per pixel
    backend: str = "auto",                  # "auto" | "cv2" | "scipy"
) -> dict
```

Measured on `case_arcade_sten_0001_104-6` with ROI `(200, 200, 320, 320)`:

```
reference_diameter_mm : 21.0
mld_mm                : 2.4
mld_position          : 117
percent_stenosis      : 88.57
mean_diameter_mm      : 17.96
severity_class        : "70-99%"
backend_used          : "scipy"
clinical_interpretation: "Obstructive; revascularization (PCI/CABG) indicated ..."
```

### Precondition: the ROI is yours to supply
This is the classical algorithm (Reiber 1984, Janssen 1991), not a detector. It
measures whatever window you hand it and **cannot autonomously locate a stenosis**.
Stage 4 must decide where to look first — via `segment_vessels` geometry, a
detector's bounding box, or explicit annotation. Garbage ROI, garbage %DS.

`pixel_spacing` is a calibration input. The 0.2 mm default is a typical XCA value;
absolute millimetre outputs are only as trustworthy as this number. Percent
stenosis is a ratio and therefore spacing-invariant.

### Backends
`cv2` mirrors upstream `classical_qca.py`; `scipy` reimplements the same steps
with `scipy.ndimage.gaussian_filter` plus a hand-written Sobel convolution, for
environments without OpenCV. `auto` prefers cv2 when importable.

**Verification status, stated honestly:** the scipy path is measured working (the
numbers above). The two paths have **not** been numerically compared, because no
environment on this host currently has cv2 (`opencv-python-headless` is in
pyproject's `specialist` extra, pending `.venv` sync). The comparison harness is
written and ready:

```python
from algorithms.tools.diameter_qca import test_qca_consistency
test_qca_consistency(image_path, roi_coords)  # asserts agreement within 1e-3
```

It skips cleanly when cv2 is absent. Run it once cv2 lands; if it fails, that is a
real algorithmic divergence worth reporting, not a tolerance to loosen.

---

## `detect_stenosis` — unavailable, raises by design

```python
detect_stenosis(image_path, device="cuda:1") -> NoReturn  # always raises
```

```
NotImplementedError: DeepCORO-CLIP stenosis detection is unavailable:
no weights on this host.
```

It raises rather than returning `[]`, because an empty stenosis list is
indistinguishable from a correct reading of a healthy artery. A silent empty
result would let a broken pipeline report plausible numbers.

### Probe before calling
```python
from algorithms.tools.stenosis_detection import is_available, detection_metadata

ok, reason = is_available()
# (False, "weights directory is empty: .../weights/deepcoro_clip_stenosis")

detection_metadata()["alternatives"]
# ['benchmark/runners/cm_unet_runner.py for arcade_stenosis',
#  'algorithms.tools.diameter_qca for ROI quantification']
```

An agent should call `is_available()` during planning and route around the gap
instead of failing mid-pipeline: CM-UNet for `arcade_stenosis` detection (that
task's gold label space is the single class `"stenosis"`, so a binary segmenter is
directly comparable), and `quantify_stenosis` for measurement once an ROI exists.

---

## Working end-to-end example (Stage 4 minimum viable chain)

Segment → derive ROI from the mask → measure. Verified run, real output:

```python
import numpy as np
from scipy import ndimage
from algorithms.tools import segment_vessels, quantify_stenosis

IMG = "data/tasks/arcade_stenosis/cases/case_arcade_sten_0001_104-6/image.png"

# 1. binary vessel mask
mask = segment_vessels(IMG, device="cuda:7")

# 2. ROI = bounding box of the largest connected vessel component
labeled, n = ndimage.label(mask)
sizes = ndimage.sum(mask, labeled, range(1, n + 1))
biggest = int(np.argmax(sizes)) + 1
ys, xs = np.where(labeled == biggest)
pad = 5
roi = (
    max(0, xs.min() - pad), max(0, ys.min() - pad),
    min(mask.shape[1], xs.max() + pad), min(mask.shape[0], ys.max() + pad),
)

# 3. QCA on that ROI
result = quantify_stenosis(IMG, roi_coords=tuple(int(v) for v in roi))
```

```
STEP1 mask shape=(512, 512) dtype=bool foreground=8820 (3.3646%)
STEP2 connected components=5
STEP2 largest component=6116 px, roi=(79,65,216,462) size=137x397
STEP3 backend=scipy
STEP3 reference_diameter_mm=25.68
STEP3 mld_mm=1.4
STEP3 mld_position=15
STEP3 percent_stenosis=94.55
STEP3 severity_class=70-99%
STEP3 interpretation=Obstructive; revascularization (PCI/CABG) indicated if
      symptomatic or ischemia documented.
```

Read this as a working chain, not a validated reading. Step 2's
largest-component heuristic picked a 137×397 box spanning most of one vessel
tree, so the "reference diameter" is drawn from that whole span rather than a
normal segment adjacent to the lesion. Clinical QCA localises the ROI to the
lesion and its immediate reference. A real Stage 4 should narrow the ROI — that
narrowing decision is agent logic, and it is exactly what these tools leave to
the agent.

---

## Substitutions from PROPOSAL.md §2.6

The suite is "shared and swappable" by design, so substituting a backend is
compliant. What was swapped, and why:

| §2.6 names | Implemented with | Why |
|---|---|---|
| SAM-VMNet (segmentation) | **CM-UNet** | SAM-VMNet weights on this host are 132-byte unresolved git-lfs pointer *text*, not tensors (`medsam_vit_b.pth`, `vmamba_tiny_e292.pth`). The architecture also needs `mamba_ssm` + `causal_conv1d` CUDA extensions, absent from every environment here. Remaining upstream weights are on Google Drive / Baidu, and this host cannot reach huggingface.co. CM-UNet's 124 MB checkpoint is real, loads with `strict=True`, and has a verified inference path. |
| DeepCORO-CLIP (second opinion) | **nothing — raises** | Both weight directories (`deepcoro_clip_stenosis/`, `VasoVision/`) are empty; a full-tree search for `*.pth/*.pt/*.ckpt/*.safetensors` finds none. Per the vendored `ACCESS_PENDING.md`, upstream returns HTTP 401/403 and needs author approval. |
| QCA-style diameter | **classical QCA, as specified** | Not a substitution. Upstream `classical_qca.py` used verbatim, with a scipy edge-detection path added for environments lacking cv2. |

Also absent: no StenUNet checkpoint exists in `specialist_models/weights/`, and
`specialist_models/stenosis_detection/` and `.../vessel_segmentation/` are empty
directories, despite what some inventory docs claim.

When SAM-VMNet or DeepCORO weights arrive, add a `backend=` parameter rather than
replacing these implementations, so results stay comparable across the swap.

---

## Design note: one source of truth for preprocessing

`vessel_segmentation` imports `_load_architecture`, `_read_frame`, `_preprocess`,
`_apply_intensity` and `_postprocess_to_original` from
`benchmark/runners/cm_unet_runner.py` instead of reimplementing them, and reads
every parameter through `load_method_config`.

This is deliberate. Two independent copies of a preprocessing chain is what
produced both silent-failure bugs this repo has hit: ARCADE Dice 0.726 → 0.000
from a skipped unsharp/normalize step, and CCA Dice 0.536 → 0.048 from missing
resampling. Both failed with exit code 0 and plausible-looking foreground ratios.
Adding a second copy here would reintroduce that class of bug.
