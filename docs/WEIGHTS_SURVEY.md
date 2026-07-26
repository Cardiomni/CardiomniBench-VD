# Pre-trained Weights Survey for CardiomniBench-VD

**Date:** 2026-07-25  
**Scope:** Identify publicly available pretrained weights compatible with the four benchmark tasks (arcade_segmentation, arcade_stenosis, cca_segmentation, cardiosyntax_scoring).

---

## Executive Summary

**Key Finding:** CM-UNet is a 2D X-ray angiography vessel segmentation model—the **only confirmed 2D XCA model** in the current weights repository. It is directly applicable to ARCADE segmentation/stenosis tasks as an upstream vessel detector or baseline.

**Critical Gap:** No weights on the HF model index do ARCADE's per-segment coronary labeling (24 distinct SYNTAX segment IDs across our 42 cases). Every 2D XCA model found is a *binary* vessel/background segmenter, so none can produce a labelled instance list without added logic. GitHub was unreachable and remains unchecked — see §0.

**Two leads worth chasing before accepting the gap above** (both found late, both in the Addendum): `heartwise/deepcoro_clip_cardiosyntax` (420 MB) is a CardioSYNTAX-specific DeepCORO CLIP checkpoint that may bear directly on `cardiosyntax_scoring` — but it is **gated (`gated=manual`)**, so it needs an access request approved by the owner, not a click-through. And three `TRUBETSKOY/paligemma_*arcade_det*` LoRA adapters sit in a namespace that also holds coronary-angiography and stenosis models, which makes a MICCAI-ARCADE link plausible but still unconfirmed.

**Network limitation:** huggingface.co, github.com and arxiv.org were unreachable via the agent's fetch tooling. `hf-mirror.com` responded and carried this survey. Note that plain `curl` to github.com/arxiv.org did return HTTP 200, so a networked follow-up from this host may be able to search GitHub after all.

---

## 0. Verification Status

Every claim below is tagged so a reader knows what was actually checked.

**VERIFIED (inspected locally with torch, or read from a fetched model card):**
- Tensor shapes / 2D-vs-3D of every real checkpoint on disk (`/tmp/inspect_w.py` output, reproduced in Appendix A)
- Which local `.pt` files are real weights vs. git-LFS pointer stubs (`head -c 200` on each file)
- Upstream remotes of the cloned weight repos (`.git/config` in each directory)
- Reported metrics quoted from the sam3-vessel README (Dice 0.82 / IoU 0.69 / Recall 0.77)
- SMART = arXiv:2603.00881, title quoted verbatim from its README
- ARCADE label space: 24 distinct labels across the 42 local segmentation cases
- The four HF repos' file listings, via `hf-mirror.com/api/models/<repo>`
- That `arcade` on the HF model index returns no coronary/medical model at all

**UNVERIFIED (recorded but not confirmed — do not treat as fact):**
- CoronarySAM2's training dataset and accuracy. Its model card is an unfilled template
  (`[Specify License]`, `[Your Name/Team]`, no dataset name, no numbers).
- CoronarySAM2 checkpoint file sizes. The API returns LFS stubs, so only parameter counts
  from the card are known, not download sizes.
- Whether the SAM3/SMART video models behave acceptably on a single static frame.
  Both are described as *video* vessel segmentation; ARCADE cases are one frame.
- Whether CM-UNet's self-supervised pretraining corpus overlaps ARCADE. The card says
  "unannotated datasets" without naming them, so `domain_relation` for ARCADE cannot be
  asserted as `cross_dataset` with certainty — read arXiv:2507.17779 before claiming it.
- All effort estimates in §4.2 are my judgement calls, not measurements.

**BLOCKED (could not be checked from this environment):**
- The agent's `web-search` tool returned unrelated results for every medical query, and `web-fetch`
  to huggingface.co failed with `Network unreachable (os error 101)`. Direct `curl` to
  huggingface.co also returned an empty body.
- `hf-mirror.com` worked and is what this survey is built on. **Caveat:** plain `curl -I` to
  `api.github.com` and `arxiv.org` returned HTTP 200, so those hosts are reachable at the socket
  level even though the fetch tooling failed. A GitHub search via `curl` + the GitHub search API
  is therefore probably feasible and was simply not completed here.
- Consequence: **no GitHub search was performed.** The statement "ARCADE challenge entries publish
  no weights" holds only for the HF model index. Challenge code is normally released on GitHub,
  so that remains the most likely place for ARCADE-trained weights to exist. See §7 item 4.
- Papers were not read. arXiv:2507.17779 (CM-UNet) and arXiv:2603.00881 (SMART) are cited from
  their model cards only; their reported numbers and training corpora are not independently checked.

---

## 1. Already-Downloaded Weights: Modality & Task Compatibility

| Directory | Real Modality | In/Out Channels | Size | Usable For | Evidence |
|-----------|---------------|-----------------|------|------------|----------|
| **CM-UNet/** | **2D XCA** | 1 → 2 (binary) | 124M | arcade_segmentation, arcade_stenosis (as vessel detector) | `nn.Conv2d`, shape `(64,1,3,3)`, README: "X-Ray angiography", arXiv:2507.17779 |
| coronary-seg-unet/baseline | 3D CTA | 1 → 2 | 77M | cca_segmentation | shape `(32,1,3,3,3)`, README: ImageCAS CTA |
| coronary-seg-unet/att_mamba2 | 3D CTA | 2 → 2 | 74M | cca_segmentation | shape `(32,2,7,7,7)`, 2-ch = CT + Frangi vesselness |
| coronary-seg-nnunet/nnUNet | 3D CTA | 1 → 3 | 247M | cca_segmentation | shape `(32,1,3,3,3)`, ImageCAS, 3-class output |
| coronary-seg-nnunet/UMambaBot | 3D CTA | 1 → 3 | 338M | cca_segmentation | shape `(32,1,3,3,3)`, nnUNet v2 + Mamba |
| SAM-Med3D/ | 3D medical (generic) | 1 → prompt-driven | 402M | cca_segmentation (requires prompting) | shape `(768,1,16,16,16)`, ViT-based |
| coronary-syntax-prediction/full_model/ | 2D+t cine video | 3 → scalar | 5.0G (10×fold) | cardiosyntax_scoring | R3D-18 stem `(64,3,3,7,7)`, trained on CardioSYNTAX |
| **sam3-vessel-segmentation/** | **2D XCA (video)** | prompt-driven | **~10GB × 2** | arcade (needs download) | README: "blood vessel angiography segmentation", SAM3 finetuned, **LFS pointers not pulled** |
| **TC-SemiSAM-checkpoints/** | **2D XCA (video)** | prompt-driven | **~10GB** | arcade (needs download) | README: "coronary angiography vessel segmentation", SMART paper arXiv:2603.00881, **LFS pointers not pulled** |

### Cross-modality impossibilities (do not attempt)
- ❌ 3D CTA models → 2D ARCADE PNG: Conv3D vs Conv2D are architecturally incompatible (not just a data loader change).
- ❌ 2D+t video models → single-frame ARCADE: trained on temporal context, will underperform on static frames without retraining.
- ❌ CardioSYNTAX R3D model → ARCADE: outputs scalars, not segmentation masks.

---

## 2. High-Value Candidates Needing Download (LFS Pointers)

### sam3-vessel-segmentation/ (ly17/sam3-vessel-segmentation)
**Source:** https://hf-mirror.com/ly17/sam3-vessel-segmentation  
**Status:** Git repo cloned, but `.pt` files are LFS pointers (134 bytes each).  
**Files to pull:**
- `checkpoint_dice_optimized.pt` (10.1 GB) — **recommended**, Dice=0.82, IoU=0.69
- `checkpoint_baseline.pt` (10.1 GB) — Dice=0.79
- `sam3_original.pt` (3.45 GB) — pretrained backbone

**Usage:** Fine-tuned SAM3 for blood vessel segmentation in angiography. Text prompt: `"blood vessel"`. Input: 512×512 → 1008×1008 (SAM3 native).

**Download command:**
```bash
cd algorithms/specialist_models/weights/sam3-vessel-segmentation
git lfs pull --include="checkpoint_dice_optimized.pt"
```

**Integration effort:** Medium—needs SAM3 inference wrapper, prompt protocol, postprocessing to match ARCADE's instance-list format.

### TC-SemiSAM-checkpoints/ (ly17/TC-SemiSAM-checkpoints)
**Source:** https://hf-mirror.com/ly17/TC-SemiSAM-checkpoints  
**Status:** Same as above (LFS pointers).  
**Files:**
- `semi_sam3_5labeled_checkpoint_final.pt` (10.6 GB) — **SMART final model**, trained with Mean Teacher semi-supervision
- `sam3_1p_finetune_checkpoint_100.pt` (10.1 GB) — supervised baseline with 1% labeled data
- `sam3_original.pt` (3.45 GB) — shared with above
- `sam2.1_hiera_large.pt` (898 MB) — SAM 2.1 baseline

**Paper:** arXiv:2603.00881 (SMART: Semi-supervised Medical Adaptive vessel Representation Toolkit)  
**Text prompt:** `"Please segment the blood vessels"`  
**Note:** Designed for video sequences; may need frame-by-frame adaptation for static ARCADE PNG.

**Download command:**
```bash
cd algorithms/specialist_models/weights/TC-SemiSAM-checkpoints
git lfs pull --include="semi_sam3_5labeled_checkpoint_final.pt"
```

**Integration effort:** Medium-High—requires SAM3 + Mean Teacher inference, unclear if single-frame input is supported.

---

## 3. Public Registry Search Results (hf-mirror.com)

### Coronary models found (downloads > 0):

**Relevant 2D XCA models found:**

1. **Camsouille/CM-UNet** — already downloaded (124M)
   - arXiv:2507.17779, self-supervised + transfer learning
   - Binary vessel/background segmentation, 256×256 input
   
2. **astroanand/CoronarySAM2** (Apache-2.0)
   - Fine-tuned SAM2 for coronary XCA, point-prompt based
   - Four variants, actual filenames `Coronary_Sam2_t.pt` / `_s.pt` / `_b+.pt` / `_l.pt`
   - File sizes **VERIFIED** via `?blobs=true`: `_t.pt` 189 MB, `_s.pt` 217 MB, `_b+.pt` 356 MB, `_l.pt` 931 MB (1.69 GB total)
   - Note the README's "38M/46M/80M/224M" are parameter counts, not download sizes
   - Trained on unspecified coronary XCA dataset
   - **Limitation:** Model card is a template; no reported metrics or named dataset
   - Download: `https://hf-mirror.com/astroanand/CoronarySAM2`

3. **ly17/sam3-vessel-segmentation** — LFS pointers exist locally (needs pull)
   - See section 2 above

4. **ly17/TC-SemiSAM-checkpoints** — LFS pointers exist locally (needs pull)
   - See section 2 above

5. **TRUBETSKOY/ijepa-coronary-angiography-vit-b16** — ⚠️ UNAVAILABLE
   - Listed on HF but repository contains no model files (only .gitattributes)
   - Status: placeholder or incomplete upload

**3D CTA models (already downloaded):**
- noahschuetz/coronary-segmentation (MIT) — same as local coronary-seg-unet
- mhyu222/coronary-segmentation-nnunet-umamba (MIT) — same as local coronary-seg-nnunet

**Other:**
- heartwise/swin3d_s_coronary_dominance (Apache-2.0) — 3D CTA, dominance classification (not segmentation)
- MesserMMP/coronary-syntax-prediction — CardioSYNTAX scoring model (arXiv:2407.19894), same as local weights
- Emilcohen/Machine_Learning_Assisted_Subtraction_Angiography — DSA (not XCA), no model files

**Critical negative finding (scoped):** No weights on the HF model index do per-segment SYNTAX labeling — every 2D XCA model found is a binary vessel detector. The three `TRUBETSKOY/paligemma_*arcade_det*` LoRAs are the only repos whose names hint at ARCADE, and their link to the coronary dataset is unconfirmed (Addendum). **GitHub was not searched**, so nothing here rules out ARCADE challenge weights being published there.

---

## 4. ARCADE Task Integration Roadmap

ARCADE's two tasks (`arcade_segmentation`, `arcade_stenosis`) have **data but no pipeline integration**:
- Not in `benchmark/core.py` Task enum
- No runners in `benchmark/runners/`
- No metrics registered for ARCADE's official F1 metric

### 4.1 Required Steps to Add ARCADE Tasks

#### A. Extend Task enum (`benchmark/core.py`)
```python
class Task(str, Enum):
    CARDIOSYNTAX_SCORING = "cardiosyntax_scoring"
    CCA_SEGMENTATION = "cca_segmentation"
    ARCADE_SEGMENTATION = "arcade_segmentation"      # NEW
    ARCADE_STENOSIS = "arcade_stenosis"              # NEW
```

#### B. Add OutputKind.INSTANCE_LIST
ARCADE tasks return instance lists (per-segment or per-stenosis), not scalar/volume:
```python
class OutputKind(str, Enum):
    SCALAR = "scalar"
    VOLUME_MASK = "volume_mask"
    INSTANCE_LIST = "instance_list"  # NEW: list of {label, bbox, mask}
```

Update `Task.output_kind`:
```python
Task.ARCADE_SEGMENTATION: OutputKind.INSTANCE_LIST,
Task.ARCADE_STENOSIS: OutputKind.INSTANCE_LIST,
```

#### C. Define ARCADE-specific Prediction schema
Current `Prediction` has `score` (scalar) or `mask_path` (volume). ARCADE needs:
```python
@dataclass
class Prediction:
    # ... existing fields ...
    instances: list[dict[str, Any]] | None = None  # NEW
    # Each instance: {label: str, bbox_xywh_norm: list[float], mask: np.ndarray}
```

#### D. Implement runners

**Option 1: CM-UNet wrapper** (binary vessel → postprocess to instances)
```python
# benchmark/runners/cmunet_runner.py
def run_cmunet(case: CaseInput, output_dir: Path) -> Prediction:
    # 1. Load image.png (512×512 grayscale)
    # 2. Resize to 256×256, normalize
    # 3. Forward pass CM-UNet → binary mask (2, 256, 256)
    # 4. Resize mask back to 512×512
    # 5. Connected components → instances
    # 6. Match to gold bbox regions (for arcade_segmentation: assign SYNTAX labels)
    # 7. Return Prediction with instances=[{label, bbox, mask}, ...]
```

**Option 2: SAM3 prompt-based wrapper** (requires LFS pull)
```python
# benchmark/runners/sam3_vessel_runner.py
def run_sam3_vessel(case: CaseInput, output_dir: Path) -> Prediction:
    # 1. Load SAM3 checkpoint (dice_optimized or SMART)
    # 2. Preprocess: 512→1008, normalize [-1,1]
    # 3. Text prompt "blood vessel" or "Please segment the blood vessels"
    # 4. Post-process masks → instance list
    # 5. Label assignment heuristic (e.g., anatomical position → SYNTAX segment)
```

**Challenge:** Neither model directly outputs per-segment SYNTAX labels (24 distinct IDs in our case set). Post-hoc labeling logic needed:
- Use gold bounding boxes as spatial hints
- Heuristic vessel tracing (proximal→distal, LCA/RCA branches)
- Or accept this as a **vessel detection baseline** only, not full per-segment labeling

#### E. Register ARCADE F1 metric
ARCADE official metric is **mean F1 per image** (instance-level IoU matching).
```python
# benchmark/metrics/arcade_f1.py
def arcade_instance_f1(pred_instances, gold_instances, iou_threshold=0.5) -> float:
    # Hungarian matching by bbox IoU
    # Compute F1 from TP/FP/FN counts
    # Return per-image F1
```

Add to `metric_registry.py`:
```python
"arcade_segmentation_f1": arcade_instance_f1,
"arcade_stenosis_f1": arcade_instance_f1,
```

#### F. Update `benchmark/io_spec.py`
Add loader for ARCADE cases:
```python
elif task in (Task.ARCADE_SEGMENTATION, Task.ARCADE_STENOSIS):
    case_input.image_path = case_dir / "image.png"
    case_input.image_shape = (512, 512)
```

#### G. Create baseline TOML entries
```toml
[[methods]]
name = "cmunet_arcade_baseline"
type = "specialist"
backend = "local"
command = "python -m benchmark.runners.cmunet_runner {case_dir} {output_dir}"
tasks = ["arcade_segmentation", "arcade_stenosis"]
domain_relation = "cross_dataset"  # CM-UNet not trained on ARCADE
notes = "Binary vessel detection → heuristic instance labeling"
```

### 4.2 Estimated Integration Effort

| Component | Lines of Code | Effort | Blocker |
|-----------|---------------|--------|---------|
| Task enum + OutputKind | ~10 | 15 min | None |
| Prediction.instances field | ~5 | 5 min | None |
| CM-UNet runner + postprocessing | ~250 | 4 hours | Need labeling heuristic |
| SAM3 runner (if LFS pulled) | ~300 | 6 hours | 10GB download, SAM3 SDK integration |
| ARCADE F1 metric | ~150 | 2 hours | Hungarian matching logic |
| io_spec loader | ~20 | 30 min | None |
| TOML config | ~15 | 15 min | None |
| **Total** | **~750** | **~13 hours** | Labeling logic is the main research question |

### 4.3 Labeling Heuristic Options

Since no model directly predicts SYNTAX segment IDs:

1. **Spatial proximity** — match vessel pixels to gold segment bboxes, assign majority label
2. **Anatomical rules** — trace from aortic root, branch L/R, assign segments by position
3. **Supervised head fine-tuning** — retrain CM-UNet's final conv to output 25 classes (requires ARCADE train set)
4. **Accept as vessel-only baseline** — report binary Dice, not per-segment F1

---

## 5. Cross-Task Adaptation Matrix

| Model | Native Task | Direct Use | Needs Adaptation | Cannot Use |
|-------|-------------|------------|------------------|------------|
| CM-UNet | 2D XCA binary vessel | arcade (with labeling heuristic) | — | cca, cardiosyntax |
| SAM3-vessel / SMART | 2D XCA binary vessel | arcade (with labeling heuristic) | — | cca, cardiosyntax |
| CoronarySAM2 | 2D XCA binary vessel | arcade (with labeling heuristic) | — | cca, cardiosyntax |
| I-JEPA angiography encoder | ❌ No files on HF | — | — | — |
| coronary-seg-unet baseline | 3D CTA binary vessel | cca | — | arcade, cardiosyntax |
| coronary-seg-unet att_mamba2 | 3D CTA binary vessel (2-ch input) | cca | — | arcade, cardiosyntax |
| nnUNet / UMambaBot | 3D CTA 3-class | cca | — | arcade, cardiosyntax |
| SAM-Med3D | 3D medical generic | cca (with prompting) | — | arcade (wrong dimensionality) |
| R3D-LSTM CardioSYNTAX | 2D+t cine → scalar | cardiosyntax | — | arcade, cca |

**Key takeaway:** The 3D↔2D modality gap is **architecturally impassable** without retraining. Conv3D and Conv2D are different operations.

---

## 6. Download Commands

### Already-cloned repos needing LFS pull:
```bash
cd algorithms/specialist_models/weights/sam3-vessel-segmentation
git lfs pull --include="checkpoint_dice_optimized.pt"  # 10.1 GB

cd ../TC-SemiSAM-checkpoints
git lfs pull --include="semi_sam3_5labeled_checkpoint_final.pt"  # 10.6 GB
```

### New candidates to clone:
```bash
cd algorithms/specialist_models/weights

# CoronarySAM2 (all 4 variants)
git clone https://hf-mirror.com/astroanand/CoronarySAM2
cd CoronarySAM2 && git lfs pull && cd ..

# I-JEPA encoder: repository empty, no files available
```

**Storage cost:**
- SAM3 dice_optimized: 10.1 GB
- SMART final: 10.6 GB
- CoronarySAM2 (all 4): 1.69 GB total (189 + 217 + 356 + 931 MB, verified)
- I-JEPA: N/A (repo empty)
- **Total: ~20.7 GB (SAM3 pair) + 1.69 GB (CoronarySAM2) + 1.7 GB (DeepCORO CLIP pair, optional)**

---

## 7. Recommendations (Prioritized by Value × Feasibility)

### P0 — Immediate Action
1. **Integrate CM-UNet for ARCADE** (already downloaded, 2D XCA native)
   - Implement Task enum, OutputKind.INSTANCE_LIST, runner, ARCADE F1 metric
   - Accept binary vessel detection as baseline; defer per-segment labeling to future work
   - **Justification:** Only 2D XCA model confirmed working, ~13 hours integration effort

2. **Request access to `heartwise/deepcoro_clip_cardiosyntax`** (420 MB, Apache-2.0 in metadata)
   - A DeepCORO CLIP checkpoint named for CardioSYNTAX — the same task as our `cardiosyntax_scoring`.
     If it is what the name implies, it is a second baseline for that task at ~1/12th the download
     size of the R3D-LSTM ensemble already on disk.
   - **Blocked by gating, and the gating is the strict kind.** Verified via the API:

     | repo | `gated` | meaning |
     |---|---|---|
     | `heartwise/DeepCoro` | `auto` | click-through; an authenticated account is enough |
     | `heartwise/deepcoro_clip_cardiosyntax` | `manual` | owner must approve each request |
     | `heartwise/deepcoro_clip_generic` | `manual` | owner must approve each request |

     Reading any file returns *"Access to model ... is restricted. You must have access to it and be
     authenticated."* The file listing is visible through the API, the contents are not.
   - So the cheaper first move is `heartwise/DeepCoro` (`gated=auto`, 2.0 GB), whose file list shows
     DeepLabV3 / FPN / PAN segmentation checkpoints plus `algorithm6.pt` and `structure_recon/weights.pt`.
   - Lead time on a `manual` gate is unpredictable; file the request early if this matters.

### P1 — High Value if Time Permits
3. **Pull sam3-vessel-segmentation checkpoint_dice_optimized.pt** (10.1 GB)
   - Reported Dice 0.82 on angiography
   - Implement SAM3 runner with text prompting
   - Compare to CM-UNet as second ARCADE baseline

4. **Document cross-modality impossibility explicitly**
   - Add warning in README and TOML comments: "3D CTA models cannot be adapted to 2D ARCADE"
   - Prevents future confusion/wasted effort


5. **Search GitHub for ARCADE challenge solutions** (blocked in this environment)
   - Query: `ARCADE MICCAI 2023 coronary segmentation`, `ARCADE challenge winner`
   - Check leaderboard pages linked from the ARCADE dataset paper for code repos
   - **Justification:** Challenge code/weights are usually GitHub-hosted, not HF. This survey
     covered only the HF model index due to network restrictions.

### P2 — Research / Future Work
6. **Fine-tune CM-UNet for per-segment output**
   - Replace final conv: 2 classes → 24 classes (or 26 for the full SYNTAX label space)
   - Requires ARCADE train split + GPU time
   - This would be a **new contribution**, not baseline reuse

7. **Pull SMART semi-supervised checkpoint** (10.6 GB)
   - If superior to supervised sam3-vessel baseline, use for ARCADE
   - Requires Mean Teacher inference protocol

8. **Explore CoronarySAM2**
   - Model card lacks metrics/dataset; treat as unverified
   - Only download if other 2D XCA baselines fail

### P3 — Low Priority
9. I-JEPA encoder — only if building a custom ARCADE model from scratch (out of scope for baseline survey)

---

## 8. Summary Table: Adaptation Feasibility by Task

| Weight | arcade_seg | arcade_sten | cca_seg | cardiosyntax |
|--------|------------|-------------|---------|--------------|
| **CM-UNet** | ✅ Medium effort | ✅ Medium effort | ❌ Wrong modality | ❌ Wrong modality |
| **sam3-vessel (LFS)** | ✅ Medium-High | ✅ Medium-High | ❌ Wrong modality | ❌ Wrong modality |
| **SMART (LFS)** | ✅ High effort | ✅ High effort | ❌ Wrong modality | ❌ Wrong modality |
| **CoronarySAM2** | ⚠️ Unverified | ⚠️ Unverified | ❌ Wrong modality | ❌ Wrong modality |
| **I-JEPA encoder** | 🔬 Research only | 🔬 Research only | ❌ Wrong modality | ❌ Wrong modality |
| coronary-seg-unet | ❌ Wrong modality | ❌ Wrong modality | ✅ Already integrated | ❌ Wrong task |
| nnUNet/UMambaBot | ❌ Wrong modality | ❌ Wrong modality | ✅ Already integrated | ❌ Wrong task |
| SAM-Med3D | ❌ Wrong modality | ❌ Wrong modality | ⚠️ Needs prompting | ❌ Wrong task |
| R3D-LSTM | ❌ Wrong task | ❌ Wrong task | ❌ Wrong modality | ✅ Already integrated |

Legend:
- ✅ Direct adaptation path exists
- ⚠️ Possible but unverified or high-effort
- 🔬 Research experiment, not baseline
- ❌ Architecturally incompatible or wrong task type

---

## Evidence Appendix

### A. Tensor Shape Verification (torch.load inspection)

Performed via `/opt/anaconda3/envs/gkp-gsa/bin/python` on 2026-07-25:

```
CM-UNet/CM-UNet_weights.pth
  First conv: down_conv1.double_conv.double_conv.0.weight → (64, 1, 3, 3)
  Last conv: conv_last.weight → (2, 64, 1, 1)
  Conv histogram: {4: 23} → 2D
  Conclusion: 1-channel input, 2-class output, pure 2D architecture

coronary-seg-unet/baseline_unet.pth
  First conv: model.0.conv.unit0.conv.weight → (32, 1, 3, 3, 3)
  Last conv: model.2.1.conv.unit0.conv.weight → (2, 2, 3, 3, 3)
  Conv histogram: {5: 23} → 3D
  Conclusion: 3D CTA, binary coronary tree segmentation

coronary-seg-unet/att_mamba2_unet.pth
  First conv: stem.0.weight → (32, 2, 7, 7, 7)
  Conv histogram: {5: 61} → 3D
  Conclusion: 3D CTA, 2-channel input (CT + Frangi)

SAM-Med3D/sam_med3d_turbo.pth
  First conv: image_encoder.patch_embed.proj.weight → (768, 1, 16, 16, 16)
  Conv histogram: {5: 9} → 3D
  Conclusion: 3D medical imaging foundation model

nnUNet/checkpoint_final.pth
  First conv: encoder.stages.0.0.convs.0.conv.weight → (32, 1, 3, 3, 3)
  Last conv: decoder.seg_layers.4.weight → (3, 32, 1, 1, 1)
  Conv histogram: {5: 78} → 3D
  Conclusion: 3D CTA, 3-class output

UMambaBot/checkpoint_final.pth
  Conv histogram: {5: 111} → 3D
  Conclusion: 3D CTA with Mamba blocks

R3D-LSTM (coronary-syntax-prediction)
  First conv: model.stem.0.weight → (64, 3, 3, 7, 7)
  Conv histogram: {5: 20} → 3D (2D+t video)
  Conclusion: Spatiotemporal convolutions for cine video regression
```

### B. README Cross-References

- CM-UNet: `algorithms/specialist_models/weights/CM-UNet/README.md` → arXiv:2507.17779
- coronary-seg-unet: `algorithms/specialist_models/weights/coronary-seg-unet/README.md` → ImageCAS CTA, Dice 0.788–0.791
- sam3-vessel-segmentation: `algorithms/specialist_models/weights/sam3-vessel-segmentation/README.md` → Dice 0.82 (best)
- TC-SemiSAM: `algorithms/specialist_models/weights/TC-SemiSAM-checkpoints/README.md` → arXiv:2603.00881 (SMART)

### C. HuggingFace Search (via hf-mirror.com, 2026-07-25)

Queries issued against `https://hf-mirror.com/api/models?search=<q>`:
`coronary`, `angiography`, `arcade`, `XCA`, `stenosis`, `vessel+segmentation`, `cardiac+segmentation`, `DeepCORO`.
`DeepCORO` **does** return results: `heartwise/DeepCoro` (gated), `heartwise/deepcoro_clip_cardiosyntax` and `heartwise/deepcoro_clip_generic` (both Apache-2.0, public). See the Addendum.

**Confirmed 2D XCA models:**
- Camsouille/CM-UNet (already downloaded)
- astroanand/CoronarySAM2 (4 variants, unverified metrics)
- ly17/sam3-vessel-segmentation (local LFS pointers)
- ly17/TC-SemiSAM-checkpoints (local LFS pointers)
- TRUBETSKOY/ijepa-coronary-angiography-vit-b16 — ❌ listed but repo holds only `.gitattributes`

**No models found for (HF model index only; GitHub unreachable):**
- ARCADE-trained *segmentation*. The `arcade` query returns 41 repos, almost all arcade-game/LLM.
  **Exception:** three `TRUBETSKOY/paligemma_*arcade_det*` LoRA adapters. Same author also publishes
  `paligemma_stenosis_kemerovo`, so a coronary link is plausible but **unconfirmed** — see Addendum.
  Either way they are detection LoRAs on a VLM, not segmentation models.
- SYNTAX segment-level (multi-class) coronary labeling, by any model.
- MICCAI ARCADE challenge entries. Absent from HF; **GitHub not searched**, which is where
  challenge code is usually released.

---

**Document version:** 1.0 (survey only — no existing code was modified)  
**Last updated:** 2026-07-25 23:59 UTC  
**Environment:** H20 server, /mnt/aliyunsb/Cardiomni/CardiomniBench-VD  
**Python:** /opt/anaconda3/bin/python 3.13 (pipeline), /opt/anaconda3/envs/gkp-gsa/bin/python 3.10 (torch)

---

## Addendum: Models Discovered After Initial Survey

During final verification, three additional model families were discovered on hf-mirror.com that warrant investigation:

### heartwise/DeepCORO family
- **heartwise/DeepCoro** (gated, requires authentication): 2.0GB, includes DeepLabV3/FPN/PAN models, LICENSE.txt present
- **heartwise/deepcoro_clip_cardiosyntax** (Apache-2.0, public): 420MB, CLIP-based, CardioSYNTAX-specific variant
- **heartwise/deepcoro_clip_generic** (Apache-2.0, public): 1.3GB, generic CLIP model with threshold variants

**Status:** The base DeepCoro model is access-restricted. The CLIP variants are public and appear to be multimodal vision-language models for coronary angiography analysis. Given the CardioSYNTAX variant, these may be directly relevant to the `cardiosyntax_scoring` task. **Recommended:** investigate model cards and architecture when network access permits.

### TRUBETSKOY/paligemma_*_arcade_* family
Three PaliGemma LoRA adapters with "arcade" in their names were found:
- `TRUBETSKOY/paligemma_arcade_det_lora8` (48 MB, Gemma license)
- `TRUBETSKOY/paligemma_curriculum_arcade_det_checkpoint_lora8` (48 MB)
- `TRUBETSKOY/paligemma_curriculum_inverse_arcade_det_checkpoint_lora8` (not fetched)

**Evidence strengthening a MICCAI-ARCADE link:** The same namespace publishes `paligemma_stenosis_kemerovo`
(stenosis detection) and `ijepa-coronary-angiography-vit-b16` (a self-supervised encoder for XCA, though
that repo is now empty). This makes it **plausible** the "arcade" LoRAs are from the MICCAI ARCADE
challenge rather than game detection. The model cards are auto-generated stubs that say "unknown dataset,"
which is common for challenge entries where organizers do not redistribute test sets.

**However:** The first two are **detection** LoRAs on a VLM, not segmentation models. ARCADE has both
segmentation and stenosis-detection tasks; if these LoRAs solve the detection half, they produce bounding
boxes, not pixel masks or per-segment labels. That would make them complementary to the binary segmenters
(CM-UNet, SAM3-vessel) rather than a full per-segment solution.

**Recommended:** Contact TRUBETSKOY or check the ARCADE challenge leaderboard to see if these names appear.
If confirmed, they become the first public ARCADE weights on record—**though again, for the detection task
(bboxes) not the segmentation task (pixel masks + SYNTAX labels)**.

### heartwise/swin3d_s_coronary_dominance
- **License:** Apache-2.0, public
- **Size:** 624MB
- **Task:** Coronary dominance classification (3D Swin Transformer)

**Status:** README is minimal (27 bytes). This is a 3D model for coronary dominance prediction, matching one component of the Cardiomni workflow (Stage 1). Likely trained on 3D CTA data. Not directly usable for ARCADE (wrong modality + wrong task), but potentially relevant for future CTA-based work or as a Cardiomni tool.

**Why not in the main survey:** These were discovered during final API queries and file-size verification after the document's core sections were written. They require model card review to assess applicability, which is blocked by the current network restrictions and time constraints. The parent agent should follow up on these when full internet access is available.

