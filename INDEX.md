# Coronary Angiography Methods Library - Documentation Index

**Purpose**: Quick navigation for paper writing and benchmark development  
**Created**: 2026-07-24  
**Status**: Survey complete, 35+ methods cataloged

---

## 📚 Main Documents

### 1. **METHODS_LIBRARY.md** — The Complete Survey
**Path**: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/METHODS_LIBRARY.md`

**What's inside**:
- 35+ methods organized by 8 task categories
- Evolution timeline: Classical (1998) → Foundation Models (2026)
- Performance metrics, code/weights availability
- Role in CardiomniBench-VD (tools vs. upper bounds)
- 50+ paper citations with clickable URLs

**Use when**: Writing Related Work section, understanding field landscape

---

### 2. **methods_library.bib** — Citation Database
**Path**: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/methods_library.bib`

**What's inside**:
- 40+ BibTeX entries properly formatted
- Organized by category (vessel seg, stenosis, SYNTAX, GNN, etc.)
- URLs and DOIs included

**Use when**: Adding citations to `aaai2027.bib`

**Critical entries to add to paper**:
```bibtex
@article{samvmnet2024, ...}      % SAM-VMNet
@article{cmunet2025, ...}        % CM-UNet
@article{cardiosyntax2024, ...}  % CardioSYNTAX
@article{deepcoro2026, ...}      % DeepCORO-CLIP
@article{frangi1998multiscale, ...} % Classical baseline
@article{ronneberger2015unet, ...}  % U-Net
@article{isensee2021nnunet, ...}    % nnU-Net
```

---

### 3. **PAPER_INTEGRATION_GUIDE.md** — Writing Roadmap
**Path**: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/PAPER_INTEGRATION_GUIDE.md`

**What's inside**:
- Specific text additions for Introduction, Related Work, Experiments
- LaTeX code snippets ready to copy-paste
- Table/figure templates with footnotes
- Citation placement recommendations

**Use when**: Actually writing/editing the paper

**Key sections**:
- Quick stats for Introduction: "30+ methods since 2020, AUROC 0.888..."
- Related Work additions: classical baseline + DL trajectory
- Experiments table template with specialist model reference rows

---

### 4. **METHODS_SURVEY_SUMMARY.md** — Executive Summary
**Path**: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/METHODS_SURVEY_SUMMARY.md`

**What's inside**:
- High-level findings and key stats
- Tool registry (7 callable tools identified)
- Dual-goal achievement (paper writing + method positioning)
- Next steps roadmap

**Use when**: Explaining what was done, quick reference for stats

---

## 🎯 Quick Reference: Key Numbers for Paper

### Field Maturity
- **35+ methods cataloged** (1998-2026)
- **20+ methods** published 2020-2024 alone
- **8 task categories** covered

### Specialist Model Upper Bounds
- **Vessel Segmentation**: IoU 0.63 (SAM-VMNet, ARCADE)
- **Stenosis Detection**: AUROC 0.888 (DeepCORO-CLIP, 203K videos)
- **Stenosis Quantification**: MAE 13.6% vs core-lab QCA (DeepCORO-CLIP)
- **Training Scale**: 195K-203K videos (CathAI, DeepCORO-CLIP)

### Our Positioning
- **Zero-training** agent vs. 200K-video specialist models
- **Orchestration** across tool library, not single-task prediction
- **Explainable** reasoning trace vs. opaque embeddings
- **Competitors**: Naive tool-caller, Claude Code, Codex (NOT the specialist models)

---

## 🔧 Tool Registry Status

### Available in `algorithms/specialist_models/`

| Tool | Task | Code | Weights | Priority |
|------|------|------|---------|----------|
| **SAM-VMNet** | Segmentation + stenosis | ✅ Downloaded | ❌ Network blocked | High |
| **CM-UNet** | Self-supervised seg | ✅ Downloaded | ❌ Network blocked | Medium |
| **CardioSYNTAX** | SYNTAX score | ✅ Downloaded | ❌ Network blocked | High |
| **MesserMMP** | SYNTAX (alternative) | ✅ Downloaded | ❌ Network blocked | Medium |
| **TC-SemiSAM** | Semi-supervised seg | ✅ Downloaded | ❌ Network blocked | Medium |
| **SAM3-vessel** | Vessel-adapted SAM | ✅ Downloaded | ❌ Network blocked | Medium |
| **DeepCORO-CLIP** | Multi-task foundation | ✅ Downloaded | ❌ Auth required | Low |

**Solution**: Use mock implementations (rule-based or simple heuristics) for experiments. Weights are non-critical — paper focuses on orchestration, not specialist model accuracy.

---

## 📝 How to Use This for Paper Writing

### Step 1: Introduction
Copy stats from **PAPER_INTEGRATION_GUIDE.md** Section "Quick Stats":
```latex
"Over 30 deep learning methods published since 2020... IoU 0.63, AUROC 0.888, MAE 13.6%..."
```

### Step 2: Related Work Section 2.1
1. Add 1 sentence on classical methods (Frangi filter)
2. Add 1 paragraph on DL trajectory (U-Net → nnU-Net → Transformer → GNN → Foundation)
3. Keep existing specialist-model positioning (already perfect)

### Step 3: Add Citations
Open `methods_library.bib`, copy these 10 critical entries to `aaai2027.bib`:
- `frangi1998multiscale`
- `ronneberger2015unet`
- `isensee2021nnunet`
- `kirillov2023sam`
- `samvmnet2024`
- `cmunet2025`
- `cardiosyntax2024`
- `messermmp2024`
- `deepcoro2026`
- `gnn_coronary_labeling2022`

### Step 4: Experiments Section
Add 1 short paragraph listing callable tools (template in **PAPER_INTEGRATION_GUIDE.md**)

### Step 5: Main Results Table
Use template from **PAPER_INTEGRATION_GUIDE.md** Section "Table 1" — includes:
- Agent harnesses (our comparison group)
- Specialist models as reference rows with footnotes

---

## 🎓 Method Positioning Clarity

### What Specialist Models Do
- **Single tasks**: Segmentation OR stenosis OR SYNTAX (not integrated workflow)
- **Opaque**: End-to-end learned, no reasoning trace
- **Data-hungry**: 195K-203K videos, domain-specific training

### What Cardiomni Does
- **Complete workflow**: 4-stage SOP (dominance → scan → view selection → assessment)
- **Explainable**: Every conclusion → specific view/frame/tool call
- **Zero-training**: Orchestrates specialist models as tools, no retraining
- **Higher-level**: Analogous to "program" vs. "function"

### Competition Axis
- ❌ **NOT competing with**: DeepCORO-CLIP, CathAI, SAM-VMNet (they are tools)
- ✅ **Competing with**: Naive tool-caller, Claude Code, Codex, OpenHands (harnesses)

---

## 🚀 Next Actions

### For Paper (Immediate)
1. ✅ Survey complete — 35+ methods cataloged
2. ⏳ Copy 10 critical citations to `aaai2027.bib`
3. ⏳ Add Introduction stats (2 sentences)
4. ⏳ Expand Related Work Section 2.1 (2 paragraphs)
5. ⏳ Add tool library paragraph to Experiments
6. ⏳ Update main results table template

### For Benchmark (Non-blocking)
1. ⏳ Implement Cardiomni agent harness (4-stage SOP)
2. ⏳ Implement Naive baseline
3. ⏳ Create mock specialist model implementations (if weights unavailable)
4. ⏳ Implement BaseAlgorithm wrappers for 7 tools

---

## 📊 File Locations Summary

```
/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/
├── METHODS_LIBRARY.md              # Main survey (35+ methods, 8 categories)
├── methods_library.bib             # 40+ BibTeX citations
├── PAPER_INTEGRATION_GUIDE.md      # LaTeX snippets for paper writing
├── METHODS_SURVEY_SUMMARY.md       # Executive summary
└── INDEX.md                        # This file

algorithms/specialist_models/       # 7 tools, code ✅, weights ❌
├── sam_vmnet/
├── cm_unet/
├── cardiosyntax/
├── deepcoro/
├── deepcoro_clip/
└── weights/                        # Empty (network blocked)
```

---

**Ready to use!** All documents are complete and cross-referenced.
