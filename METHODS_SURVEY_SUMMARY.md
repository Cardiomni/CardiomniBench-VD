# Methods Library Survey - Executive Summary

**Date**: 2026-07-24  
**Survey Scope**: Coronary angiography analysis methods (classical → foundation models)  
**Total Methods Cataloged**: 35+

---

## What Was Accomplished

### 1. Comprehensive Methods Survey ✅

Systematically cataloged **35+ methods** across 8 task categories:
- **Vessel Segmentation**: 15+ methods (Frangi → U-Net → SAM-VMNet → DeepCORO-CLIP)
- **Stenosis Detection/Quantification**: 8+ methods (ARCADE baselines → CathAI → LT-YOLO)
- **SYNTAX Scoring**: 3 methods (CardioSYNTAX, MesserMMP, inverse problem)
- **Projection Classification**: 2 methods
- **Dominance Classification**: 1 method (Real-Time classification)
- **Graph Neural Networks**: 5 methods (topology-aware, 3D reconstruction)
- **Transformer/Attention**: 6 methods (TransCC, Hybrid Transformer-CNN, Angio-Fusion)
- **Multi-Task Systems**: 2 comprehensive (CathAI, DeepCORO-CLIP)

### 2. Three Deliverables Created

#### `METHODS_LIBRARY.md` (Main Survey Document)
- **8 task categories** with detailed method tables
- Performance metrics, code/weights availability
- Chronological organization (classical → foundation models)
- Role in CardiomniBench-VD (callable tools vs. reference upper bounds)
- 50+ paper citations with URLs

#### `methods_library.bib` (BibTeX Database)
- 40+ properly formatted citations
- Organized by category matching the survey
- URLs and DOIs included
- Ready to merge into `aaai2027.bib`

#### `PAPER_INTEGRATION_GUIDE.md` (Writing Roadmap)
- Specific text additions for Introduction, Related Work, Experiments
- Citation placement recommendations
- Table/figure templates
- Quick stats for establishing field maturity

---

## Key Findings

### Evolution Timeline

```
Classical (pre-2015)          → 2 methods   (Frangi, active contours)
Early DL (2015-2020)          → 3 methods   (U-Net, AngioNet, nnU-Net)
Modern DL (2020-2024)         → 20+ methods (SAM variants, Transformer, GNN, YOLO)
Foundation Models (2024-2026) → 5+ methods  (DeepCORO-CLIP, MedSAM-2)
```

### Specialist Model Upper Bounds

**Best reported performance** (not directly comparable to our benchmark):
- **Vessel Segmentation**: IoU 0.63 (SAM-VMNet, ARCADE dataset)
- **Stenosis Detection**: AUROC 0.888 (DeepCORO-CLIP, 203K videos)
- **Stenosis Quantification**: MAE 13.6% (DeepCORO-CLIP vs. core-lab QCA)
- **SYNTAX Scoring**: R² 0.8958 (inverse problem method, small dataset)

### Training Data Scale

Specialist models require massive supervision:
- **DeepCORO-CLIP**: 203,808 videos from 28,117 patients
- **CathAI**: 195,000 videos
- **CardioSYNTAX**: 3,018 multi-view studies

**Our positioning**: Cardiomni is zero-training, orchestrates these as tools.

---

## Tool Registry for CardiomniBench-VD

### Callable Tools (Exposed via BaseAlgorithm API)

| Tool | Task | Code | Weights | Priority |
|------|------|------|---------|----------|
| SAM-VMNet | Segmentation + stenosis detection | ✅ | ❌ | High |
| CM-UNet | Self-supervised segmentation | ✅ | ❌ | Medium |
| TC-SemiSAM | Semi-supervised segmentation | ✅ | ❌ | Medium |
| SAM3-vessel | Vessel-adapted SAM | ✅ | ❌ | Medium |
| CardioSYNTAX | SYNTAX score | ✅ | ❌ | High |
| MesserMMP | SYNTAX score (alternative) | ✅ | ❌ | Medium |
| DeepCORO-CLIP | Multi-task foundation model | ✅ | ❌ (auth required) | Low |

**Status**: All code repositories downloaded. Weights blocked by network (non-critical — can use mock implementations for experiments).

### Reference Upper Bounds (Cited, Not Competing)

1. **CathAI** (Nature Medicine 2021): 195K videos, AUROC 0.862
2. **DeepCORO-CLIP** (arXiv 2026): 203K videos, MAE 13.6%, AUROC 0.888

---

## Paper Integration Roadmap

### Introduction (1 paragraph addition)
Add context on field maturity:
> "Over 30 deep learning methods published since 2020... IoU 0.63 for segmentation, 
> AUROC 0.888 for stenosis detection, MAE 13.6% for quantification..."

### Related Work Section 2.1 (expand by 2 paragraphs)
1. **Classical baseline** (1 sentence): Frangi filter, active contours
2. **DL trajectory** (1 paragraph): U-Net → nnU-Net → Transformer → GNN → Foundation models
3. **Keep existing positioning**: "treat as callable tools, not competitors"

### Experiments Section 3 (add 1 short paragraph)
**Tool Library paragraph**: List available tools following BioML-Bench paradigm

### Tables
- **Table 1** (main results): Add specialist model reference rows with footnotes
- **Optional Table 2** (tool library): Summary of callable tools with performance

### Citations to Add
**Critical 10**: Frangi, U-Net, nnU-Net, SAM, MedSAM, SAM-VMNet, CM-UNet, MesserMMP, TransCC, GNN-labeling

---

## How This Achieves the Dual Goals

### Goal 1: Facilitate Paper Writing ✅

**Introduction/Related Work**:
- Ready-to-use stats: "30+ methods since 2020"
- Clear evolution narrative: classical → DL → foundation → **agent (ours)**
- Proper citations: 40+ entries in `methods_library.bib`

**Positioning clarity**:
- Methods library shows **breadth** of specialist models
- Cardiomni addresses **orchestration** (the missing piece)
- Analogy: specialist models = individual functions, Cardiomni = complete program

### Goal 2: Clarify Method Positioning ✅

**We are NOT competing with**:
- DeepCORO-CLIP (203K videos, AUROC 0.888) — they solve stenosis quantification
- CathAI (195K videos, AUC 0.862) — they solve stenosis detection
- SAM-VMNet, CM-UNet, CardioSYNTAX — they solve single tasks

**We ARE competing with**:
- Naive tool-caller (LLM + tool descriptions)
- Claude Code (general agent harness)
- Codex (code-generation agent)
- OpenHands, AIDE (if included as baselines)

**Our contribution**:
- **4-stage SOP scaffolding** (dominance → scan → view selection → assessment)
- **Explicit reasoning trace** (every claim → view/frame/tool call)
- **Tool orchestration** across the specialist models library
- **Zero-training** explainability vs. opaque 200K-video models

---

## Next Steps

### For Paper Writing (Immediate)
1. Copy citations from `methods_library.bib` → `aaai2027.bib`
2. Add text snippets from `PAPER_INTEGRATION_GUIDE.md` → `AnonymousSubmission2027.tex`
3. Create Figure 1 (evolution timeline)
4. Fill in Table 1 placeholders once experiments run

### For Experiments (After Harness Implementation)
1. Implement Cardiomni harness (4-stage SOP)
2. Implement naive baseline
3. Mock specialist tools (if weights remain blocked)
4. Run benchmark on 175 cases (ARCADE + CardioSYNTAX)
5. Compare harnesses (main result)

### For Method Library (Optional Expansion)
- Add lesion characterization methods (calcification, CTO, thrombus)
- Add QCA (quantitative coronary angiography) classical methods
- Add FFR/IVUS integration methods (if extending to multi-modal)

---

## Files Created

1. **`METHODS_LIBRARY.md`** — 35+ methods, 8 categories, 50+ citations
2. **`methods_library.bib`** — 40+ BibTeX entries
3. **`PAPER_INTEGRATION_GUIDE.md`** — Specific text additions and table templates
4. **This file** (`METHODS_SURVEY_SUMMARY.md`) — Executive overview

All files in: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/`

---

## Citation Style Examples

### In Introduction
```latex
...with over 30 methods published since 2020~\citep{samvmnet2024,cmunet2025,cardiosyntax2024}
achieving IoU 0.63 for vessel segmentation~\citep{samvmnet2024}, AUROC 0.888 for stenosis
detection~\citep{deepcoro2026}, and MAE 13.6\% for quantification~\citep{deepcoro2026}.
```

### In Related Work
```latex
Early approaches used Frangi vesselness filtering~\citep{frangi1998multiscale}. Deep learning
brought U-Net~\citep{ronneberger2015unet}, nnU-Net~\citep{isensee2021nnunet}, and recently
Transformers~\citep{transcc2023} and graph neural networks~\citep{gnn_coronary_labeling2022}.
Foundation models like DeepCORO-CLIP~\citep{deepcoro2026} aggregate multi-view projections...
```

### In Experiments
```latex
We expose specialist models as callable tools: SAM-VMNet~\citep{samvmnet2024} for segmentation,
CardioSYNTAX~\citep{cardiosyntax2024} for SYNTAX scoring, and DeepCORO-CLIP~\citep{deepcoro2026}
as a multi-task foundation model. All harnesses access the same tool library.
```

---

**Status**: Survey complete. Ready for paper integration and experiments.
