# Coronary Angiography Analysis Methods Library

**Purpose**: Comprehensive survey of methods for coronary angiography analysis, organized by task type. Serves as (1) foundation for paper Related Work section, (2) tool registry for agent orchestration, and (3) method positioning framework.

**Scope**: Classical CV (pre-2015) → Early DL (2015-2019) → Modern DL (2020-2023) → Foundation Models (2024-2026)

**Last Updated**: 2026-07-24

---

## Task Taxonomy

Our benchmark addresses the complete clinical workflow across 8 task categories:

1. **Vessel Segmentation** — pixel-level coronary artery delineation
2. **Stenosis Detection** — lesion localization (bounding boxes)
3. **Stenosis Quantification** — severity estimation (% stenosis)
4. **SYNTAX Scoring** — holistic disease burden assessment
5. **Projection Classification** — view angle recognition (RAO/LAO/CRA/CAU)
6. **Dominance Classification** — left vs. right vs. co-dominant
7. **Lesion Characterization** — calcification, thrombus, CTO
8. **Multi-View Fusion** — cross-projection aggregation

---

## 1. Vessel Segmentation Methods

### 1.1 Classical Methods (Pre-2015)

| Method | Year | Venue | Approach | Performance | Code | Role in Our Work |
|--------|------|-------|----------|-------------|------|------------------|
| **Frangi Vesselness Filter** | 1998 | TMI | Hessian-based multi-scale filtering | Baseline | ✅ scipy | Classical baseline |
| **Improved Vesselness** (Beyond Frangi) | 2015 | ResearchGate | Ratio of multiscale Hessian eigenvalues | Better than Frangi | ⏳ | Enhanced classical baseline |

**Key papers**:
- Frangi et al., "Multiscale vessel enhancement filtering" [[ResearchGate](https://www.researchgate.net/publication/2388170_Multiscale_Vessel_Enhancement_Filtering)]
- "Beyond Frangi: An improved multiscale vesselness filter" [[ResearchGate](https://www.researchgate.net/publication/283558933_Beyond_Frangi_An_improved_multiscale_vesselness_filter)]

### 1.2 Early Deep Learning (2015-2020)

| Method | Year | Venue | Architecture | Dataset | Performance | Code | Weights | Role |
|--------|------|-------|--------------|---------|-------------|------|---------|------|
| **U-Net** | 2015 | MICCAI | CNN encoder-decoder | Medical imaging | Foundation architecture | ✅ | ✅ | Architecture baseline |
| **AngioNet** | 2021 | Nature Sci. Rep. | CNN | Proprietary | State-of-art 2021 | ✅ | ⏳ | Comparison |

**Key papers**:
- AngioNet: "a convolutional neural network for vessel segmentation in X-ray angiography" [[Nature](https://link.springer.com/10.1038/s41598-021-97355-8)]

### 1.3 Modern Deep Learning (2020-2024)

| Method | Year | Venue | Architecture | Dataset | Performance (Dice/IoU) | Code | Weights | Role |
|--------|------|-------|--------------|---------|------------------------|------|---------|------|
| **nnU-Net** | 2020 | Nature Methods | Self-configuring U-Net | Multi-domain | 0.903 (MRCA coronary) | ✅ | ✅ | Strong baseline |
| **SAM-VMNet** | 2024 | arXiv | SAM + VMamba hybrid | ARCADE, DCA1, GH | IoU 0.6308, Sens 0.9772 | ✅ | ❌ | Callable tool |
| **CM-UNet** | 2025 | arXiv | Self-supervised contrastive U-Net | ARCADE | +48.7% Dice with 18 images vs 500 | ✅ | ❌ | Callable tool |
| **TC-SemiSAM** | 2024 | HuggingFace | Semi-supervised SAM variant | — | — | ✅ | ❌ | Callable tool |
| **VM-CAGSeg** | 2025 | Frontiers Medicine | Vessel structure-aware state space model | — | — | ⏳ | ⏳ | Citation |
| **Angio-Fusion Net** | 2026 | Frontiers CardioVasc Med | Dual-stream VGG16 + Attention U-Net | — | — | ⏳ | ⏳ | Citation |
| **YOLO variants** | 2025 | Electronics MDPI | YOLOv8/v9/v11 | ARCADE | YOLOv9-E F1 0.4524 (seg), YOLOv11-X F1 0.7826 (stenosis) | ⏳ | ⏳ | Citation |

**Key papers**:
- nnU-Net: "a self-configuring method for deep learning-based biomedical image segmentation" [[Nature Methods](https://www.nature.com/articles/s41592-020-01008-z)]
- SAM-VMNet: "A Deep Learning Model for Coronary Artery Segmentation and Quantitative Stenosis Detection" [[arXiv:2406.00492](https://arxiv.org/abs/2406.00492)]
- CM-UNet: "A Self-Supervised Learning-based model for Coronary Artery Segmentation in X-Ray Angiography" [[arXiv:2507.17779](https://arxiv.org/abs/2507.17779)]
- nnU-Net for MRCA: "Fast and automatic coronary artery segmentation using nnU-Net" [[Springer](https://link.springer.com/article/10.1007/s10554-025-03408-8)]
- VM-CAGSeg: [[Frontiers](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1661680/full)]
- Angio-Fusion Net: [[Frontiers](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2026.1748962/full)]
- YOLO Assessment: [[Electronics MDPI](https://www.mdpi.com/2079-9292/14/13/2683)]

### 1.4 Foundation Models (2023-2026)

| Method | Year | Venue | Architecture | Training Data | Performance | Code | Weights | Role |
|--------|------|-------|--------------|---------------|-------------|------|---------|------|
| **SAM (Segment Anything)** | 2023 | ICCV | Vision Transformer | 11M images, 1B masks | Zero-shot segmentation | ✅ | ✅ | Foundation baseline |
| **MedSAM** | 2023 | arXiv | SAM fine-tuned on medical images | Medical imaging | Medical domain adaptation | ✅ | ✅ | Medical baseline |
| **MedSAM-2** | 2024 | arXiv | SAM 2 for 3D medical video | Medical 3D volumes | 3D tracking | ✅ | ✅ | 3D baseline |
| **SAM3-vessel** | 2024 | HuggingFace | Vessel-specific SAM | — | — | ✅ | ❌ | Callable tool |

**Key papers**:
- SAM review: "Segment Anything Model for Medical Images?" [[arXiv:2304.14660](https://arxiv.org/html/2304.14660v6)]
- MedSAM-2: "Segment Anything in 3D Medical Images and Videos" [[arXiv:2504.03600](https://arxiv.org/html/2504.03600)]
- Medical SAM 2: "Segment medical images as video via Segment Anything Model 2" [[arXiv:2408.00874](https://arxiv.org/abs/2408.00874)]
- SAM medical review: "A Review of Deep Learning Approaches Based on Segment Anything Model" [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12729286/)]

### 1.5 Graph Neural Networks

| Method | Year | Venue | Architecture | Task | Code | Role |
|--------|------|-------|--------------|------|------|------|
| **GCN for vessel extraction** | 2019 | arXiv | Graph Convolutional Networks | Coronary surface mesh extraction | ⏳ | Citation |
| **G2ViT** | 2024 | PubMed | GNN + Vision Transformer | Retinal & coronary vessel segmentation | ⏳ | Citation |
| **3D GCN Reconstruction** | 2023 | arXiv | Graph CNN | 3D coronary vessel from bi-plane | ⏳ | Citation |
| **Automated Coronary Labeling** | 2022 | arXiv | Message passing GNN | Topology-aware vessel labeling | ⏳ | Citation |

**Key papers**:
- GCN mesh extraction: [[arXiv:1908.05343](https://www.arxiv.org/pdf/1908.05343)]
- G2ViT: "Graph Neural Network-Guided Vision Transformer Enhanced Network" [[PubMed](https://pubmed.ncbi.nlm.nih.gov/38723311/)]
- 3D GCN: "3D Coronary Vessel Reconstruction from Bi-Plane Angiography using Graph Convolutional Networks" [[arXiv:2302.14795](https://arxiv.org/abs/2302.14795)]
- Automated labeling: "Automated Coronary Arteries Labeling Via Geometric Deep Learning" [[arXiv:2212.00386](https://ar5iv.labs.arxiv.org/html/2212.00386)]

---

## 2. Stenosis Detection & Quantification

### 2.1 Detection (Localization)

| Method | Year | Venue | Architecture | Dataset | Performance | Code | Weights | Role |
|--------|------|-------|--------------|---------|-------------|------|---------|------|
| **CathAI** | 2021 | Nature Medicine | 4-stage cascade (195K videos) | Proprietary (195K) | AUC 0.862 (≥70% stenosis) | ❌ | ❌ | Reference upper bound |
| **ARCADE Challenge Methods** | 2023 | MICCAI | Various (YOLO, DINO, etc.) | ARCADE (1500 images) | See leaderboard | ✅ | ⏳ | Benchmark comparison |
| **LT-YOLO** | 2024 | PMC | Long-term temporal YOLO | ICA videos | — | ⏳ | ⏳ | Citation |
| **Automatic Stenosis Detection** | 2023 | arXiv | — | ARCADE | Challenge baseline | ⏳ | ⏳ | Citation |
| **Transfer Learning** | 2020 | Mathematics MDPI | VGG16, ResNet50, Inception-v3 | XCA images | — | ⏳ | ⏳ | Early DL baseline |

**Key papers**:
- CathAI: "fully automated coronary angiography interpretation and stenosis estimation" [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10421915/)]
- ARCADE Challenge: [[Grand Challenge](https://arcade.grand-challenge.org/)]
- ARCADE Dataset: "Dataset for Automatic Region-based Coronary Artery Disease Diagnostics" [[Nature Sci Data](https://www.nature.com/articles/s41597-023-02871-z)]
- LT-YOLO: "long-term temporal enhanced YOLO for stenosis detection" [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12001240/)]
- Automatic Detection: [[arXiv:2310.14961](https://arxiv.org/abs/2310.14961)]
- Transfer Learning: [[Mathematics MDPI](https://www.mdpi.com/2227-7390/8/9/1510)]
- Evaluation on ARCADE: "Evaluating Stenosis Detection with Grounding DINO, YOLO, and DINO-DETR" [[arXiv:2503.01601](https://arxiv.org/html/2503.01601v1)]

### 2.2 Quantification (Severity Estimation)

| Method | Year | Venue | Training Data | Metric | Performance | Code | Weights | Role |
|--------|------|-------|---------------|--------|-------------|------|---------|------|
| **DeepCORO-CLIP** | 2026 | arXiv | 203,808 videos, 28,117 patients | Stenosis MAE vs QCA | 13.6% (vs clinical 19.0%) | ✅ | ❌ | Reference upper bound |
| **SAM-VMNet** | 2024 | arXiv | ARCADE, DCA1, GH | TPR / PPV | 0.5867 / 0.5911 | ✅ | ❌ | Callable tool |

**Key papers**:
- DeepCORO-CLIP: "A Multi-View Foundation Model for Comprehensive Coronary Angiography Video-Text Analysis" [[arXiv:2603.17675](https://arxiv.org/abs/2603.17675)] *(Training: 203K videos, AUROC 0.888 internal / 0.89 external)*

---

## 3. SYNTAX Score Prediction

| Method | Year | Venue | Architecture | Training Data | Performance | Code | Weights | Role |
|--------|------|-------|--------------|---------------|-------------|------|---------|------|
| **CardioSYNTAX** | 2024 | arXiv / Zenodo | R3D backbone + LSTM sequence model | 3,018 multi-view DSA studies | End-to-end regression | ✅ | ❌ | Callable tool |
| **MesserMMP-SYNTAX** | 2024 | HuggingFace | R3D + sequence model | Multi-view DSA | Automated prediction | ✅ | ❌ | Callable tool |
| **Inverse Problem Algorithm** | 2022 | Diagnostics MDPI | AI assessment technique | — | r² = 0.8958 | ⏳ | ⏳ | Citation |

**Key papers**:
- CardioSYNTAX: "End-to-end SYNTAX score prediction – dataset, benchmark and method" [[arXiv:2407.19894](https://arxiv.org/abs/2407.19894v2)]
- Dataset paper: "X-ray Coronary Angiogram images and SYNTAX score" [[Nature Sci Data](https://www.nature.com/articles/s41597-025-04727-0)]
- MesserMMP: [[HuggingFace](https://huggingface.co/MesserMMP/coronary-syntax-prediction)]
- Inverse Problem: "Quantitative Prediction of SYNTAX Score" [[Diagnostics MDPI](https://www.mdpi.com/2075-4418/12/12/3180)]

---

## 4. Projection Classification & View Recognition

| Method | Year | Venue | Architecture | Task | Performance | Code | Role |
|--------|------|-------|--------------|------|-------------|------|------|
| **CathAI Stage 1** | 2021 | Nature Medicine | CNN (part of 4-stage) | Projection angle classification | Part of cascade | ❌ | Reference |
| **Automated Detection (view classifier)** | 2021 | arXiv | DNN | LCA/RCA view | 0.97 accuracy | ⏳ | Citation |

**Key papers**:
- Automated Detection: [[arXiv:2103.02969](https://arxiv.org/abs/2103.02969)]

---

## 5. Dominance Classification

| Method | Year | Venue | Architecture | Task | Performance | Code | Role |
|--------|------|-------|--------------|------|-------------|------|------|
| **Real-Time Dominance Classification** | 2025 | Diagnostics MDPI | Advanced deep video architectures | Left/Right/Co-dominant | Real-time | ⏳ | Citation |

**Key papers**:
- Real-Time Classification: [[Diagnostics MDPI](https://www.mdpi.com/2075-4418/15/10/1186)]

---

## 6. Multi-View & Transformer-Based Methods

| Method | Year | Venue | Architecture | Task | Code | Role |
|--------|------|-------|--------------|------|------|------|
| **TransCC** | 2023 | arXiv | Transformer + self-attention | CCTA coronary segmentation | Dice 0.730, IoU 0.582 | ⏳ | Citation |
| **Hybrid Transformer-CNN** | 2026 | arXiv | MS-GLWA + SGFA | DSA vessel segmentation | Boundary-aware | ⏳ | Citation |
| **Federated Graph-Transformer** | 2024 | MDPI Biomed | Graph + Transformer | CAD severity grading | Federated learning | ⏳ | Citation |

**Key papers**:
- TransCC: "Transformer Network for Coronary Artery CCTA Segmentation" [[arXiv:2310.04779](https://arxiv.org/abs/2310.04779)]
- Hybrid Transformer-CNN: "Self-Guided Attention and Boundary-Weighted Adaptive Loss" [[arXiv:2606.29744](https://arxiv.org/abs/2606.29744)]
- Federated Graph-Transformer: [[MDPI Biomed](https://www.mdpi.com/2504-4990/8/7/187)]

---

## 7. Physics-Informed & Biomechanics Methods

| Method | Year | Venue | Architecture | Task | Code | Role |
|--------|------|-------|--------------|------|------|------|
| **PI-GNN for WSS** | 2026 | Nature Sci. Rep. | Physics-informed GNN | Wall shear stress prediction in stenotic arteries | Real-time hemodynamics | ⏳ | Citation |

**Key papers**:
- PI-GNN: "Physics-informed graph neural networks for real-time prediction of wall shear stress" [[Nature](https://www.nature.com/articles/s41598-026-47410-z)]

---

## 8. Multi-Task & Integrated Systems

| Method | Year | Venue | Architecture | Tasks Covered | Training Data | Role |
|--------|------|-------|--------------|---------------|---------------|------|
| **DeepCORO-CLIP** | 2026 | arXiv | mVIT + BioMedBERT + CLIP | Stenosis, calcification, CTO, thrombus, narrative generation | 203K videos | Reference upper bound |
| **CathAI** | 2021 | Nature Medicine | 4-stage cascade | Projection → artery ID → localization → severity | 195K videos | Reference upper bound |

---

## Summary Statistics

### By Era
- **Classical (pre-2015)**: 2 methods (Frangi baseline)
- **Early DL (2015-2020)**: 3 methods (U-Net, AngioNet, nnU-Net)
- **Modern DL (2020-2024)**: 20+ methods (SAM variants, Transformer, GNN, YOLO)
- **Foundation Models (2024-2026)**: 5+ methods (DeepCORO-CLIP, MedSAM-2, SAM3)

### By Task
- **Vessel Segmentation**: 15+ methods
- **Stenosis Detection/Quantification**: 8+ methods
- **SYNTAX Scoring**: 3 methods
- **Multi-view/Multi-task**: 2 comprehensive systems (DeepCORO-CLIP, CathAI)

### Code Availability
- **✅ Fully available**: 10 methods (nnU-Net, SAM variants, CardioSYNTAX, CM-UNet, DeepCORO-CLIP, etc.)
- **⏳ Partially available or pending**: 15+ methods
- **❌ Proprietary**: 2 methods (CathAI, internal datasets)

### Weights Availability
- **✅ Public weights**: 5 methods (nnU-Net, SAM, MedSAM family)
- **❌ Blocked by network/auth**: 8 methods (in `algorithms/specialist_models/weights/`)
- **❌ Not released**: CathAI, several recent methods

---

## Role in CardiomniBench-VD

### Callable Tools (exposed via BaseAlgorithm API)
1. SAM-VMNet — vessel segmentation + stenosis detection
2. CM-UNet — self-supervised vessel segmentation
3. TC-SemiSAM — semi-supervised segmentation
4. SAM3-vessel — vessel-adapted SAM
5. CardioSYNTAX — SYNTAX score prediction
6. MesserMMP-SYNTAX — alternative SYNTAX predictor
7. DeepCORO-CLIP — multi-task foundation model *(if weights become available)*

### Reference Upper Bounds (reported in paper, not competing agents)
1. **CathAI** (Nature Medicine 2021): 195K videos, AUC 0.862
2. **DeepCORO-CLIP** (arXiv 2026): 203K videos, MAE 13.6%, AUROC 0.888

### Citations (Related Work only)
- All Transformer, GNN, YOLO variants, classical methods
- ARCADE challenge methods
- nnU-Net as architecture baseline

### Positioning Statement (for paper)

> While specialist deep learning models (CathAI, DeepCORO-CLIP) achieve strong single-metric performance after training on 195K–203K videos, they remain opaque, data-hungry, and task-specific. Each new clinical requirement (e.g., adding calcification detection to a stenosis model) necessitates retraining. In contrast, **agent-based orchestration** treats these models as **composable tools** within an explainable workflow, enabling zero-training adaptation to new task combinations and providing verifiable reasoning traces that specialist models cannot offer.

---

## Data Synthesis for Paper

### Introduction Statistics
- "Coronary angiography analysis has been addressed by **30+ deep learning methods** published 2020–2026"
- "Methods span **8 distinct task types** from vessel segmentation to SYNTAX scoring"
- "Foundation models trained on **~200K videos** (CathAI, DeepCORO-CLIP) set specialist upper bounds"

### Related Work Timeline
```
1998: Frangi filter (classical CV baseline)
2015: U-Net (architecture foundation)
2020: nnU-Net (self-configuring segmentation)
2021: CathAI (4-stage cascade, 195K videos)
2023: SAM medical adaptations (foundation models)
2024: CardioSYNTAX (end-to-end SYNTAX prediction)
2026: DeepCORO-CLIP (video-text foundation, 203K videos)
2026: Cardiomni (this work) — agent orchestration over tool library
```

### Comparison Table for Paper (Section 3)
```
Method Type          | Training Req | Adaptability | Interpretability | Tool Use
---------------------|--------------|--------------|------------------|----------
Specialist Models    | 195K–203K    | Task-specific | Opaque          | N/A
  (CathAI, DeepCORO) | videos       | (retrain)     | embeddings      |
Agent Harnesses      | Zero-shot    | Workflow-level| Explicit trace  | Orchestrate
  (Cardiomni, ours)  | (tools only) | (SOP config)  | + tool calls    | specialists
```

---

## Next Steps

1. **Add missing citations to aaai2027.bib** (15+ papers identified above)
2. **Implement BaseAlgorithm wrappers** for 7 callable tools
3. **Update Related Work section** with comprehensive taxonomy
4. **Create comparison figure**: Evolution timeline (classical → DL → foundation → agents)
5. **Verify ARCADE leaderboard** for latest benchmark numbers

---

## Sources

All hyperlinks to papers included inline. Key repositories:
- ARCADE Challenge: https://arcade.grand-challenge.org/
- CardioSYNTAX Dataset: https://zenodo.org/records/14005818
- DeepCORO-CLIP: https://github.com/HeartWise-AI/DeepCORO_CLIP
- SAM-VMNet: https://github.com/qimingfan10/SAM-VMNet
- CM-UNet: https://github.com/CamilleChallier/Contrastive-Masked-UNet
