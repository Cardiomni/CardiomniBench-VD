# Algorithms Inventory for CardiomniBench-VD

**Purpose**: Organize available methods as (1) paper citations and (2) benchmark infrastructure (following BioML-Bench model).

---

## Core Positioning

- **Not competitors**: Specialist models (DeepCORO-CLIP, SAM-VMNet, etc.) are **tools** and **reference upper bounds**
- **Actual competitors**: Agent harnesses (naive tool-caller, Claude Code, Codex, OpenHands)
- **Contribution**: Cardiomni's **orchestration** (4-stage SOP, tool routing, reasoning trace), not end-to-end model training

---

## 1. Specialist Models (Tools Library)

Exposed via unified `BaseAlgorithm` interface. Agents call them like BioML-Bench's model zoo.

### Vessel Segmentation
- **SAM-VMNet** (2024): MedSAM + VMamba hybrid for coronary segmentation [arXiv:2406.00492]
- **CM-UNet** (2025): Self-supervised learning for XCA segmentation [arXiv:2507.17779]
- **TC-SemiSAM** (2024): Semi-supervised vessel segmentation
- **SAM3-vessel** (2024): Vessel-adapted SAM variant

### Multi-View Foundation Models  
- **DeepCORO-CLIP** (2026): Video-text model on 203K videos, segment-level stenosis MAE 13.6% [arXiv:2603.17675]
- **CathAI** (2021): 4-stage cascade on 195K videos, stenosis AUC 0.862 [Nature Medicine]

### SYNTAX Score Prediction
- **CardioSYNTAX** (2024): End-to-end SYNTAX regression from multi-view DSA [Zenodo]
- **MesserMMP-SYNTAX** (2024): R3D backbone + sequence model [HuggingFace]

**Status**: Code ✅ all downloaded | Weights ❌ blocked by network (non-critical for paper)

---

## 2. Agent Baselines (Competitors)

Fixed: base model (Claude Opus 4.8), task set (ARCADE + CardioSYNTAX), tool API  
Variable: **harness only**

| Baseline | Description |
|----------|-------------|
| **Naive Tool-Caller** | LLM + tool descriptions, no SOP scaffolding |
| **Claude Code** | Anthropic's official agent harness |
| **Codex** | OpenAI's code-generation agent |
| **Cardiomni (ours)** | 4-stage SOP pipeline with explicit reasoning trace |

**Status**: ⏳ to implement all 4 harnesses

---

## 3. Benchmark Tasks (Evaluation Data)

| Dataset | Task | Cases | Gold Standard |
|---------|------|-------|---------------|
| **ARCADE** | Vessel segmentation | 42 | Pixel-level masks |
| **ARCADE** | Stenosis detection | 69 | Lesion bboxes + SYNTAX class |
| **CCA** | 3D CTA segmentation | 20 | Volumetric masks |
| **CardioSYNTAX** | SYNTAX scoring | 60 (3-expert) | Multi-expert scores with inter-reader variability |

**Status**: ✅ all ingested, pipeline runs 175 cases end-to-end

---

## 4. Paper Structure Mapping

### Section 2: Related Work
**Already written**:
- Specialist models: CathAI, DeepCORO-CLIP, CardioSYNTAX, ARCADE/SAM-VMNet
- Agent benchmarks: SWE-bench, MLE-bench (the paradigm we follow)

**To add** (if used in experiments):
- CM-UNet, MesserMMP-SYNTAX, TC-SemiSAM

### Section 3: Experiments
**Main result table** (harness comparison):
```
Harness              | Stenosis MAE | Coverage | Naming | Trace | Overall
---------------------|--------------|----------|--------|-------|--------
Naive Tool-Caller    | [TBD]        | ...      | ...    | ...   | ...
Claude Code          | [TBD]        | ...      | ...    | ...   | ...
Codex                | [TBD]        | ...      | ...    | ...   | ...
Cardiomni (ours)     | [TBD]        | ...      | ...    | ...   | ...
---------------------|--------------|----------|--------|-------|--------
DeepCORO-CLIP*       | 13.6%†       | —        | —      | —     | —
*Reference upper bound (specialist model trained on 203K videos)
†Reported on different test set
```

**Base model swap table** (optional):
```
Model (Cardiomni harness) | Stenosis MAE | Coverage | Overall
--------------------------|--------------|----------|--------
Claude Opus 4.8           | [TBD]        | ...      | ...
GPT-4o                    | [TBD]        | ...      | ...
Gemini Pro 2.0            | [TBD]        | ...      | ...
```

---

## 5. Implementation Priorities

### Critical (for paper experiments)
1. **Cardiomni harness** — the main contribution (4-stage SOP in Python)
2. **Naive baseline** — simplest tool-caller (for comparison)
3. **Mock specialist models** — placeholder responses for 4 task types (if weights unavailable)

### Nice-to-have
- Claude Code / Codex adapters (stronger baselines)
- Real specialist model weights (network issues — can run experiments with mocks + note limitations)
- OpenHands / AIDE (extra baselines)

---

## 6. Key Insight for Paper

**Title direction**: "CardiomniBench-VD: Evaluating Agent Harnesses on Coronary Angiography via Tool Orchestration"

**Core claim**: 
- Holding the base model and specialist models fixed, **harness design** (SOP scaffolding, reasoning trace, tool routing) makes the difference
- Cardiomni's 4-stage SOP beats naive tool-calling by [X]% on [Y] metric
- Zero-training explainable agents can approach specialist-model accuracy while offering interpretability

**BioML-Bench parallel**:
- BioML-Bench: standardized ML models as tools
- CardiomniBench-VD: standardized coronary analysis models as tools
- Both: evaluate agents on **tool orchestration**, not model training
