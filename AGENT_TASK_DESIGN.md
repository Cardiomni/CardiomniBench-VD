# Cardiomni Agent Task Design

**Reference Analysis**: EchoAgent paradigm + ARCADE/CardioSYNTAX datasets → Cardiomni task specification

**Date**: 2026-07-24

---

## 1. EchoAgent "Eyes-Hands-Minds" Paradigm Summary

### 1.1 Core Architecture

EchoAgent implements a three-layer workflow:

1. **"Eyes" (Perceptual Layer)**: Raw video → view identification, anatomical structure segmentation, quantitative measurement
2. **"Hands" (Operational Layer)**: Execute specialist tools (EchoPrime for view recognition, MemSAM/EchoONE for segmentation, USFM for parameter calculation)
3. **"Minds" (Cognitive Layer)**: Expertise-driven cognition engine (EDC) → knowledge repository → orchestrated reasoning hub (OR Hub) → dynamic multimodal reasoning graph

### 1.2 Key Design Principles

**Input/Output Contract**:
- **Input**: Raw Echo videos (multiple views, 20-50 frames/sequence) + diagnostic query Q
- **Output**: Structured diagnostic conclusion with reasoning traces + confidence scores

**Evaluation Metrics** (CAMUS + MIMIC-EchoQA datasets):
- **Single-structure task (EF grading)**: Acc, G-mean, AUROC (binary classification at three thresholds: Normal/Mildly reduced/Considerably reduced)
- **Multi-structure task (EchoQA)**: Per-structure Acc across 14 anatomical categories (Pericardium, Aortic valve, Mitral valve, Ventricles, Atria, Vessels, Others)
- **Clinical reliability metric**: "Eyes-hands-minds" coordination achieves **88.0% Acc** on EF grading, **80.16% G-mean**, **84.15% average Acc** on multi-structure task

**Competition Axis**:
- NOT vs. specialist models (MemSAM, EchoONE, H2former → these are tools/"hands")
- **vs. general-purpose MLLMs**: GPT-4, LLaVA-Med, Qwen2.5-7B-VL, DeepSeek-VL2 (30% Acc drop without domain "minds")
- **vs. enhanced GPT-5\***: GPT-5 + hierarchical collaboration toolkit still underperforms EchoAgent by 4-5% (shows orchestration value)

---

## 2. ARCADE/CardioSYNTAX Dataset Task Structure

### 2.1 ARCADE Task 1: Vessel Segment Segmentation

**Input**:
```json
{
  "image": "512×512 DSA PNG",
  "phase": "phase_1 | final_phase",
  "task": "segmentation"
}
```

**Output**:
```json
{
  "segmentations": {
    "detections": [
      {
        "label": "1-16",  // AHA 16-segment model
        "bounding_box": [x_center, y_center, width, height],  // normalized [0-1]
        "mask": {"$binary": "base64_encoded"},
        "iscrowd": 0
      }
    ]
  }
}
```

**Metrics**:
- mAP@IoU=0.50, mAP@IoU=0.75, mAP@IoU=0.50:0.95
- Dice coefficient per segment
- Segment accuracy, Detection recall

### 2.2 ARCADE Task 2: Stenosis Detection

**Input**: Same as Task 1

**Output**:
```json
{
  "segmentations": {
    "detections": [
      {
        "label": "stenosis",
        "bounding_box": [...],
        "mask": {"$binary": "..."}
      }
      // Multiple stenosis possible (69/1500 images have multi-lesion)
    ]
  }
}
```

**Metrics**:
- Precision, Recall, F1 @ IoU=0.50
- False positive rate (critical for clinical deployment)
- Multi-lesion detection rate

**Clinical Context**: Stenosis grading (0-25% normal, 25-50% mild, 50-70% moderate, 70-99% severe, 100% occlusion) — but ARCADE only provides **location**, not **severity percentage**.

### 2.3 CardioSYNTAX Task: SYNTAX Score Regression

**Input**:
```json
{
  "videos": [
    {"view": "RAO_CAUDAL", "frames": 20-50},
    {"view": "LAO_CRANIAL", "frames": 20-50},
    // 5-10 multi-angle sequences
  ],
  "task": "syntax_score"
}
```

**Output**:
```json
{
  "syntax_total": 0-67,  // Integer score
  "left_system_score": 0-X,
  "right_system_score": 0-Y
}
```

**Metrics**:
- MAE (Mean Absolute Error)
- Pearson correlation coefficient
- Inter-observer agreement (60 cases have 3 expert annotations → Fleiss' kappa)

**Scale**: 1844 cases (CardioSYNTAX dataset)

---

## 3. Cardiomni Agent Task Specification (Aligned with SWE-bench/MLE-bench)

### 3.1 Task Paradigm: DSA Diagnostic Report Generation

**Core Insight**: Unlike ARCADE (single-task outputs) or EchoAgent (QA-style answers), Cardiomni targets **end-to-end structured diagnostic report** (中山模板 style) from raw DSA DICOM series.

**Why this matters**:
- ARCADE tasks are **functions** (vessel segmentation, stenosis detection)
- Cardiomni task is a **program** (orchestrate 4-stage SOP → integrate multi-view → generate clinical report)
- Competition axis: Cardiomni harness vs. naive tool-caller vs. Claude Code vs. Codex (NOT vs. SAM-VMNet/DeepCORO-CLIP)

### 3.2 Task Contract (CardiomniBench-VD)

#### Task 1: Multi-View DSA Diagnostic Report

**Input** (`task_spec.json`):
```json
{
  "case_id": "case_chxc_001",
  "modality": "DSA",
  "dicom_series": [
    {"view": "RAO_30_CAUDAL_20", "path": "data/case_*/dsa/series_01/", "frames": 30},
    {"view": "LAO_45_CRANIAL_25", "path": "data/case_*/dsa/series_02/", "frames": 28},
    // 5-10 views typical
  ],
  "task_type": "dsa_diagnostic_report",
  "patient_info": {
    "age": 65,
    "gender": "M",
    "indication": "Suspected CAD"
  }
}
```

**Output** (`prediction.json`):
```json
{
  "dominance": "right | left | co-dominant",
  "segments": [
    {
      "segment_id": "1",  // AHA 16-segment
      "segment_name": "Proximal RCA",
      "stenosis_present": true,
      "stenosis_severity": "70-99%",  // Or "未见明显狭窄"
      "stenosis_position": "proximal third",
      "lesion_characteristics": {
        "length_mm": 12.5,
        "calcification": "moderate",
        "eccentricity": "eccentric"
      },
      "reference_diameter_mm": 3.2,
      "minimal_lumen_diameter_mm": 0.9
    }
    // All 16 segments must be explicitly reported (including negatives)
  ],
  "syntax_score": {
    "total": 24,
    "left_system": 16,
    "right_system": 8
  },
  "recommendations": {
    "treatment": "PCI | CABG | medical",
    "target_vessels": ["RCA proximal", "LAD mid"],
    "urgency": "elective | urgent | emergent"
  },
  "reasoning_trace": [
    {"stage": "dominance", "tool": "view_classifier", "result": "Right dominant via PDA origin"},
    {"stage": "systematic_scan", "tool": "sam_vmnet", "result": "Segment 1-3 identified"},
    {"stage": "view_selection", "tool": "projection_selector", "result": "RAO_30 optimal for RCA proximal"},
    {"stage": "lesion_assessment", "tool": "qca_diameter", "result": "70% stenosis confirmed"}
  ]
}
```

**Clinical Tolerance Matching** (per `report_facts.py`):
- Stenosis severity: ±10% absolute OR same clinical tier (<50 / 50-69 / 70-99 / 100)
- Segment coverage: All 16 segments must have explicit statement (presence or "未见明显狭窄")
- Dominance: Exact match required

#### Task 2: Single-View Stenosis Quantification (Ablation Study)

**Input**:
```json
{
  "case_id": "arcade_001",
  "image": "512×512 PNG",
  "phase": "phase_1",
  "task_type": "stenosis_quantification"
}
```

**Output**:
```json
{
  "stenosis_detections": [
    {
      "bbox": [x, y, w, h],  // Normalized
      "severity_percentage": 75.0,
      "confidence": 0.92,
      "segment_id": "6"  // Optional
    }
  ]
}
```

**Metrics**: Precision, Recall, F1 @ IoU=0.50, MAE for severity percentage

#### Task 3: Vessel Segmentation (Tool Validation)

**Input**: Same as Task 2

**Output**:
```json
{
  "segments": [
    {
      "segment_id": "1-16",
      "mask": "base64_encoded_binary",
      "bbox": [...]
    }
  ]
}
```

**Metrics**: mAP@IoU=0.50:0.95, Dice coefficient, Segment accuracy

#### Task 4: SYNTAX Score Prediction (Treatment Decision)

**Input**: Multi-view DSA series (same as Task 1)

**Output**:
```json
{
  "syntax_total": 24,
  "left_system_score": 16,
  "right_system_score": 8
}
```

**Metrics**: MAE, Pearson r, Clinical decision agreement (PCI vs. CABG threshold at SYNTAX=23)

### 3.3 Evaluation Metrics (Hierarchical Rubric)

Following PROPOSAL.md §3, the DSA-only seven-axis rubric:

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| **Stenosis Accuracy** | 0.25 | ±10% or same tier match with gold standard |
| **Segment Coverage** | 0.20 | All 16 segments explicitly mentioned (partial credit for missed negatives) |
| **Anatomical Naming** | 0.10 | AHA nomenclature adherence |
| **Dominance Determination** | 0.10 | Exact match (right/left/co-dominant) |
| **Reasoning Traceability** | 0.15 | reasoning_trace field presence + logical coherence |
| **Tool Orchestration** | 0.10 | Appropriate tool calls in 4-stage SOP order |
| **Anti-Hallucination** | 0.10 | No fabricated segments/views/measurements |

**Aggregation**: Weighted sum → case score → mean ± SD across benchmark (175 cases from ARCADE, or 60 expert-annotated cases if available)

---

## 4. Agent Implementation Requirements (4-Stage SOP)

### Stage 1: Dominance Determination
- **Input**: All DICOM series
- **Tool calls**: 
  - `view_classifier.identify_views()` → RAO/LAO/cranial/caudal angles
  - `dominance_classifier.predict(views)` → right/left/co-dominant based on PDA/PLV origin
- **Output**: `dominance: str`

### Stage 2: Systematic Segment Scan
- **Input**: Selected views per dominance
- **Tool calls**:
  - `sam_vmnet.segment_vessels(dicom)` → vessel masks for all visible segments
  - `segment_classifier.map_to_aha16(masks)` → assign 1-16 labels
- **Output**: `segments: List[Segment]` (may be incomplete if occluded in single view)

### Stage 3: View/Projection Selection
- **Input**: All views + preliminary segment list
- **Logic**: For each segment, select optimal view minimizing foreshortening/overlap
- **Tool calls**: `projection_selector.rank_views(segment_id, available_views)`
- **Output**: `optimal_views: Dict[segment_id, view]`

### Stage 4: Lesion Assessment
- **Input**: Per-segment optimal views
- **Tool calls**:
  - `stenosis_detector.detect(view)` → candidate lesion bboxes
  - `qca_tool.measure_diameter(lesion_region)` → reference_diameter, MLD → % stenosis
  - `deepcoro_clip.second_opinion(lesion)` (optional) → confidence boost
- **Output**: Per-segment stenosis report + reasoning trace

### Orchestration Logic (OR Hub analog)
- **Adaptive branching**: If Stage 2 coverage < 70%, trigger additional view acquisition prompt
- **Confidence gating**: If Stage 4 stenosis severity straddles clinical threshold (e.g., 68-72%), invoke DeepCORO-CLIP second opinion
- **Reasoning graph**: Build DAG of (tool_call, result, confidence) → final diagnosis with traceable evidence chain

---

## 5. Baseline Harnesses (Competition Comparators)

### 5.1 Naive Tool-Caller (Baseline 1)
- **Behavior**: Call all tools on all views sequentially, no reasoning/selection
- **Pseudocode**:
  ```python
  dominance = call_tool("dominance_classifier", all_views)
  segments = []
  for view in all_views:
      segments += call_tool("sam_vmnet", view)
  stenosis = []
  for seg in segments:
      stenosis += call_tool("qca_tool", seg)
  return assemble_report(dominance, segments, stenosis)
  ```
- **Expected weakness**: Over-segmentation, missed lesions in occluded views, no evidence synthesis

### 5.2 Claude Code Adapter (Baseline 2)
- **Behavior**: Invoke Claude Code with full DICOM context + task description, let it orchestrate tools via natural language
- **Pseudocode**:
  ```python
  prompt = f"You have {len(views)} DSA views. Use tools to diagnose CAD and generate report."
  response = claude_code_api(prompt, context=dicom_metadata, tools=all_tools)
  return parse_response(response)
  ```
- **Expected weakness**: No domain-specific SOP, may hallucinate segment names, tool call order arbitrary

### 5.3 Codex Adapter (Baseline 3)
- **Behavior**: GPT-4 Code Interpreter with Python tool wrappers
- **Similar to Baseline 2** but likely stronger code generation, weaker medical reasoning

### 5.4 Cardiomni Agent (Proposed)
- **Behavior**: Implements 4-stage SOP, adaptive reasoning hub, clinical guideline grounding
- **Expected advantage**: +10-15% stenosis accuracy, +20% reasoning traceability over Baseline 1-3

---

## 6. Dataset Construction Plan

### 6.1 Public Benchmark (Phase 1)
- **Source**: ARCADE (3000 images → 175 cases reconstructed as multi-view sequences via metadata)
- **Gold standard**: ARCADE ground truth (segment masks + stenosis bboxes) + manual SYNTAX score annotation (subset of 60 cases)
- **Splits**: 60 train (for tool calibration) / 40 val / 75 test

### 6.2 Expert-Annotated Benchmark (Phase 2, "in preparation")
- **Source**: Private hospital data with 3-expert consensus reports
- **Scale**: 60-100 cases
- **Gold standard**: Full diagnostic reports (中山模板) with stenosis %, SYNTAX scores, treatment recommendations
- **Status**: Data collection in progress, not claimed in AAAI 2027 paper

---

## 7. Implementation Roadmap

### P0 (Required for AAAI 2027 submission)
1. ✅ Pipeline harness complete (29/29 tests passing)
2. ⬜ **Cardiomni agent implementation** (`docker/agent/` Dockerfile + 4-stage SOP Python code)
3. ⬜ **Baseline 1-3 harnesses** (mock implementations initially, full integration if time permits)
4. ⬜ **ARCADE dataset preprocessing** (3000 images → 175 reconstructed cases with task_spec.json)
5. ⬜ **Rubric alignment** (purge fusion_reasoning dimension, implement DSA-only 7-axis rubric)
6. ⬜ **Tool wrappers** (YOLOv11-X, YOLOv9c, placeholder for SAM-VMNet/CM-UNet/CardioSYNTAX if weights unavailable)

### P1 (Post-submission enhancements)
- Full specialist model integration (if weights become accessible)
- Expert-annotated dataset experiments
- Ablation studies (E/H/M components per EchoAgent Table 4)

---

## 8. Key Differences: EchoAgent → Cardiomni

| Aspect | EchoAgent | Cardiomni |
|--------|-----------|-----------|
| **Modality** | Echo (ultrasound, 2D+Doppler) | DSA (X-ray angiography, 2D projection) |
| **Input** | Video clips + diagnostic query Q | Multi-view DICOM series (no explicit query) |
| **Output** | QA-style answer + confidence | Structured diagnostic report (16 segments + SYNTAX) |
| **Tasks** | 14 anatomical structures (binary classification) | 4 tasks (segmentation, detection, quantification, SYNTAX) |
| **Specialist tools** | EchoPrime, MemSAM, EchoONE, USFM (4 tools) | SAM-VMNet, YOLOv11-X, QCA, DeepCORO-CLIP, CardioSYNTAX (5-7 tools) |
| **Reasoning hub** | OR Hub (multimodal reasoning graph, adaptive branching) | 4-stage SOP (dominance → scan → view selection → assessment) |
| **Evaluation** | CAMUS (1000 videos), MIMIC-EchoQA (48 views × 622 cases) | ARCADE (3000 images → 175 cases), CardioSYNTAX (1844 cases, subset) |
| **Baseline competitors** | LLaVA-Med, GPT-4, Qwen2.5-7B-VL, GPT-5\* | Naive tool-caller, Claude Code, Codex |
| **Primary metric** | Acc (88.0% EF grading, 84.15% multi-structure) | Stenosis accuracy (weighted 0.25), Coverage (0.20) |
| **Clinical decision** | EF grading (normal/reduced) → heart failure management | SYNTAX score → PCI vs. CABG decision |

**Core shared principle**: "Eyes-hands-minds" workflow → specialist models are tools ("hands"), multimodal LLMs provide reasoning ("minds"), domain knowledge grounds decisions ("EDC engine" analog).

---

## 9. Critical Citations to Add (from EchoAgent)

These demonstrate the agent-vs-MLLM framing:

1. **Wang et al. 2026** - EchoAgent: Our reference architecture
2. **Christensen et al. 2024** - MediRAG: MLLM-based medical reasoning
3. **Achiam et al. 2023** - GPT-4 technical report (baseline competitor)
4. **Bai et al. 2023/2024** - Qwen2-VL series (baseline competitor)
5. **Wu et al. 2024** - DeepSeek-VL2 (baseline competitor)
6. **Liu et al. 2024** - LLaVA-Med (baseline competitor)

---

## 10. Next Steps

1. **Validate task contracts** with existing `pipeline/report_facts.py` scoring logic
2. **Design mock tool outputs** for YOLOv11-X, YOLOv9c (since we have weights), placeholder responses for blocked models
3. **Implement Cardiomni agent skeleton** (4-stage SOP orchestration without real tool calls initially)
4. **Create baseline harness mocks** (Naive/Claude Code/Codex simulators)
5. **Run smoke test** on `case_chxc_001` with mock backends to verify end-to-end flow

**Critical path**: Agent implementation (P0.2) is the bottleneck. All other components (pipeline, dataset, rubric) are infrastructure supporting this core contribution.
