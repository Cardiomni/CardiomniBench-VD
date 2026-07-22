# DSA Diagnostic Report Schema (Prose + Facts)

**Status**: Locked-in design per 2026-07-22 alignment discussion  
**Task**: Narrow DSA-only diagnostic report evaluation (not the full CTA+DSA fusion task)

---

## Overview

The **narrowed DSA task** evaluates an agent's ability to produce a **prose diagnostic report** from multi-view coronary angiography (DSA) cines, matching the format used in clinical practice (e.g., 中山医院 template). The report describes stenosis by vessel segment, including explicit "未见明显狭窄" statements for non-stenosed segments.

**Key design locked points** (from the alignment discussion):

1. **Output format**: PROSE report (narrative, 中山模板 style), NOT a structured YAML form  
2. **Required content (text)**: vessel segment name + stenosis % (including negatives)  
3. **Optional (visual, bonus)**: lesion location marked on the angiogram image (arrow/point) — "锦上添花"  
4. **Evaluation method**: LLM judge extracts structured facts from the prose → compares against expert gold standard using **tolerance** (±10% or same clinical tier) + **partial credit** (6/8 segments correct → partial score, not all-or-nothing)

This schema defines the gold-standard structure for this task. It is **narrower** than the full `task_template.yaml` (which covers CTA+DSA fusion, SYNTAX scoring, treatment decisions). This narrow task focuses exclusively on: **Input = N-view DSA cines → Output = prose diagnostic report → Evaluation = fact extraction + tolerance scoring**.

---

## Gold Standard Structure

### Top-level (case file: `data/cases/<case_id>/task.yaml`)

```yaml
task_version: "1.0.0"
case_id: "case_chxc_001"  # 陈秀川-DSA real case example

case_metadata:
  creation_date: "2026-07-22"
  annotator_primary: "张冠兆"
  difficulty_level: "medium"  # easy / medium / hard
  pathology_tags: ["multi-vessel"]  # optional
  fusion_category: "dsa_only"  # this task uses DSA only, no CTA
  task_type: "dsa_report"  # marker for the narrow task

# =============================================================================
# INPUT DATA
# =============================================================================
input:
  dsa:
    # List of DICOM cine sequences (one per view/acquisition).
    # The agent receives all views; gold_standard annotates which vessels are
    # best seen in which views (informational, not scored directly).
    views:
      - file_path: "data/cases/case_chxc_001/dsa/IM000000.dcm"
        modality: "XA"
        num_frames: 26
        positioner_primary_angle: -31.5   # RAO31
        positioner_secondary_angle: -21.3  # CAU21
        view_label: "RAO31/CAU21"
        target_vessels: ["LCX", "LAD_proximal"]  # informational
      - file_path: "data/cases/case_chxc_001/dsa/IM000001.dcm"
        modality: "XA"
        num_frames: 16
        positioner_primary_angle: 45.5    # LAO45
        positioner_secondary_angle: -19.2  # CAU19
        view_label: "LAO45/CAU19 (spider)"
        target_vessels: ["LM", "LAD", "LCX"]  # bifurcation view
      # ... additional views ...

  clinical_context:
    age: null
    sex: ""
    chief_complaint: ""
    risk_factors: []
    # Minimal context for the narrow task; full context in the CTA+DSA fusion task.

  prohibited_resources:
    items: ["FFR/iFR", "IVUS/OCT", "Stress test", "Prior reports"]
    note: "Agent must recognize when additional data is needed for treatment decisions"

# =============================================================================
# EXPECTED OUTPUT
# =============================================================================
expected_output:
  format: "prose_report"  # NOT structured JSON (the locked-in change)
  required_sections:
    - "dominance"  # 冠脉分布为右优势型 / 左优势型
    - "per_segment_findings"  # vessel + position + stenosis % or "未见明显狭窄"
  
  # OPTIONAL (bonus): visual localization on the angiogram frames
  optional_sections:
    - "lesion_locations"  # {segment_id, file_path, frame_index, x, y}

  # Example prose report (the target output format):
  example_report: |
    冠脉分布为右优势型。
    1. 左冠状动脉:
       (a) 左主干:未见狭窄
       (b) 前降支:近端狭窄60%,第一对角支未见狭窄
       (c) 回旋支:未见明显狭窄
    2. 右冠状动脉:近段狭窄75%

# =============================================================================
# GOLD STANDARD (Expert Annotation)
# =============================================================================
gold_standard:
  # -------------------------------------------------------------------------
  # DSA Report Facts (structured ground truth for tolerance comparison)
  # -------------------------------------------------------------------------
  dsa_report_facts:
    dominance: "right"  # right / left / balanced

    segments:
      # One entry per vessel segment visible in the DSA study, INCLUDING
      # segments with no stenosis (must be explicitly annotated to test
      # completeness / anti-omission).
      #
      # Fields:
      #   segment_id   — SYNTAX id (e.g., "LM_5", "LAD_6", "RCA_1"). Optional
      #                   if vessel + position are sufficient; required if the
      #                   rubric checks SYNTAX naming accuracy.
      #   vessel       — canonical short name (LM / LAD / LCX / RCA / D1 / OM / PDA)
      #   position     — proximal / mid / distal / "" (empty if not applicable,
      #                   e.g., LM, D1, or when the report text doesn't specify).
      #                   When both segment_id and position are present, the
      #                   explicit position wins (allows overriding SYNTAX default).
      #   stenosis_percent — 0-100. Use 0 for "未见明显狭窄". Use 100 for total
      #                      occlusion. Expert uncertainty range can be noted in
      #                      stenosis_percent_range (not directly scored; informational).
      #   stenosis_percent_range — [min, max] expert confidence interval (optional)
      #   notes        — free-form annotation notes (optional)

      - segment_id: "LM_5"
        vessel: "LM"
        position: ""
        stenosis_percent: 0
        notes: "未见狭窄"

      - segment_id: "LAD_6"
        vessel: "LAD"
        position: "proximal"
        stenosis_percent: 60
        stenosis_percent_range: [55, 65]
        notes: "近端狭窄，best seen in LAO45/CAU19 spider view"

      - segment_id: "LAD_9"
        vessel: "D1"
        position: ""
        stenosis_percent: 0
        notes: "第一对角支未见狭窄"

      - segment_id: "LCX_11"
        vessel: "LCX"
        position: ""  # override SYNTAX_11 default "proximal" because report just says "回旋支"
        stenosis_percent: 0
        notes: "未见明显狭窄"

      - segment_id: "RCA_1"
        vessel: "RCA"
        position: "proximal"
        stenosis_percent: 75
        stenosis_percent_range: [70, 80]
        notes: "近段狭窄，best seen in RAO30 view"

    # -------------------------------------------------------------------------
    # Clinical Tiers (informational, for threshold validation)
    # -------------------------------------------------------------------------
    # The tolerance scorer uses these tier boundaries:
    #   <50%       — none_mild (no intervention)
    #   50-69%     — moderate (may need FFR)
    #   70-99%     — severe (likely intervention)
    #   100%       — occluded (CTO)
    # A prediction is "correct" if within ±10% OR in the same tier as gold.
    clinical_tiers:
      none_mild: [0, 50)
      moderate: [50, 70)
      severe: [70, 100)
      occluded: 100

  # -------------------------------------------------------------------------
  # Visual Localization (OPTIONAL gold, for the bonus criterion)
  # -------------------------------------------------------------------------
  lesion_locations:
    # If the expert provides lesion coordinates on the angiogram frames, the
    # rubric's visual_localization criterion can score them. Omit this block
    # entirely if visual localization is not annotated for this case.
    - segment_id: "LAD_6"
      file_path: "data/cases/case_chxc_001/dsa/IM000001.dcm"
      frame_index: 3  # 0-indexed frame in the cine
      x: 256
      y: 180
      description: "LAD proximal stenosis, spider view peak opacification"

    - segment_id: "RCA_1"
      file_path: "data/cases/case_chxc_001/dsa/IM000006.dcm"
      frame_index: 6
      x: 300
      y: 200
      description: "RCA proximal stenosis"

  # -------------------------------------------------------------------------
  # Capability Boundary (what the agent SHOULD acknowledge as unknown)
  # -------------------------------------------------------------------------
  capability_boundary:
    ffr_needed: false  # For this narrow task, FFR is out of scope
    ivus_oct_needed: false
    notes: |
      The narrow DSA-report task does NOT require the agent to make treatment
      decisions (PCI vs. CABG vs. medical therapy). The agent's report should
      state stenosis findings; a full decision would need FFR for 50-69% lesions
      and guideline concordance checks (covered in the full fusion task).

# =============================================================================
# RUBRIC REFERENCE
# =============================================================================
rubric:
  rubric_file: "rubrics/dsa_report_rubric.yaml"  # the narrowed 6-dimension rubric
  rubric_version: "1.0.0"

# =============================================================================
# METADATA
# =============================================================================
metadata:
  data_source: "Hospital case, anonymized"
  anonymization_verified: true
  expert_consensus: true  # ≥2 experts reviewed
  solvability_verified: true  # all required views present, diagnostic quality
  tags: ["dsa_only", "teaching_case"]
```

---

## Prose Report Output Contract

The agent writes a **prose diagnostic report** to `prediction.json` with this structure:

```json
{
  "case_id": "case_chxc_001",
  "report": "<prose text, 中山模板 style>",
  "extracted_facts": {
    "dominance": "right",
    "segments": [
      {"vessel": "LM", "position": "", "stenosis_percent": 0},
      {"vessel": "LAD", "position": "proximal", "stenosis_percent": 60},
      ...
    ]
  },
  "lesion_locations": [
    {"segment_id": "LAD_6", "file_path": "...", "frame_index": 3, "x": 256, "y": 180}
  ],
  "reasoning_trace": "<optional internal reasoning>"
}
```

**Key fields**:

- **`report`** (required): The prose diagnostic report. This is the PRIMARY output. The judge extracts structured facts from this text for scoring.
- **`extracted_facts`** (optional): If the agent self-reports structured facts, the pipeline uses them directly (skips extraction). Useful for structured agents and for testing the tolerance scorer offline.
- **`lesion_locations`** (optional): Visual localization annotations (bonus criterion).
- **`reasoning_trace`** (optional): Internal reasoning, not scored but useful for debugging.

---

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Agent produces prose report (prediction["report"])              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Fact extraction:                                                 │
│    • Use prediction["extracted_facts"] if present                   │
│    • Else: LLM judge parses the prose → structured facts            │
│    • Fallback: heuristic regex extraction (offline, deterministic)  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Tolerance comparison (deterministic, automatic metrics):         │
│    • Segment coverage recall (gold segments the report covered)     │
│    • Naming accuracy (SYNTAX id or canonical vessel+position match) │
│    • Stenosis accuracy (±10% OR same clinical tier)                 │
│    • Dominance correctness (right/left/balanced)                    │
│    • Anti-hallucination (pred significant lesions w/ no gold basis) │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Rubric scoring (see rubrics/dsa_report_rubric.yaml):            │
│    • Weighted aggregation across 6 dimensions                       │
│    • Partial credit (e.g., 6/8 segments correct → 75% on that dim) │
│    • Visual localization is an optional bonus dimension             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Comparison to Full `task_template.yaml`

| Aspect | Full CTA+DSA Fusion Task | Narrow DSA-Report Task |
|--------|-------------------------|------------------------|
| **Input** | Paired CTA (3D) + DSA (2D cines) | DSA cines only |
| **Output format** | Structured JSON (8 sections) | Prose report + optional facts |
| **Gold standard** | `stage0_anatomy`, `stage1a_cta`, `stage1b_dsa`, `stage2_fusion`, `stage3_scoring`, `stage4_decision` | `dsa_report_facts` (one block) |
| **Evaluation** | 16 automatic metrics + 12 LLM-judge criteria across 6 dimensions | 5 automatic tolerance metrics + optional visual bonus |
| **Rubric** | `rubrics/examples/case_001_rubric.yaml` (654 lines, 24 criteria) | `rubrics/dsa_report_rubric.yaml` (~6 criteria) |
| **Core novelty** | Fusion reasoning (blooming correction, CTO, vulnerable plaque) | Completeness + tolerance (anti-omission, ±% or tier) |
| **Use case** | Research benchmark (paper contribution) | Clinical validation, teaching cases, real-case smoke test |

Both tasks share the same **pipeline infrastructure** (orchestrator, judge backends, metric registry, config). The narrow task is a **proper subset** — it reuses `pipeline/report_facts.py` for extraction + comparison, and registers its 5 tolerance metrics in `pipeline/metric_registry.py` alongside the full task's 16 perception metrics. A case can be scored under either rubric by swapping `rubric.default_case_rubric` in the config.

---

## Next Steps (Implementation Roadmap)

1. ✅ **Fact extraction + tolerance comparison** (`pipeline/report_facts.py`) — DONE, 10/10 tests passing
2. ✅ **Metric registry integration** (`pipeline/metric_registry.py`) — DONE, 5 new adapters registered
3. ⏳ **Narrowed rubric** (`rubrics/dsa_report_rubric.yaml`) — IN PROGRESS (task 12)
4. ⏳ **Real-case worked example** (`data/cases/case_chxc_001/`) — skeleton ready, awaiting expert values from 张冠兆
5. ⏳ **End-to-end offline test** — run the pipeline on the worked example with mock judge + heuristic extraction (deterministic, no API key)
6. ⏳ **LLM extraction validation** — compare heuristic vs. LLM fact extraction on 5-10 varied prose reports, measure agreement
7. ⏳ **Visual localization scorer** — if expert provides coordinates, implement the distance-based bonus criterion

---

## Open Questions / Decisions Needed

- **Tier boundaries**: Current default `(50, 70, 100)` matches the alignment meeting. Should this be configurable per case, or fixed?
- **Tolerance %**: Default `±10%` for stenosis. Should moderate lesions (50-69%) use tighter tolerance since they drive FFR decisions?
- **Negative points**: Should anti-hallucination allow NEGATIVE dimension scores (like `source_reliability` in the full rubric), or just reduce the score to 0?
- **Visual localization distance threshold**: If expert annotates lesion at (x, y) and agent at (x', y'), how close is "correct"? Propose: ≤20 pixels at 512×512 resolution (~ 1 vessel diameter).
- **Multi-expert agreement**: For the worked example, do we need ≥2 experts to agree on stenosis %, or is one senior interventionalist (张冠兆) sufficient for the teaching case?

---

**Document Status**: Living spec. Update as real cases surface edge cases (e.g., CTO with retrograde fill, anomalous anatomy, severe motion artifact rendering a view non-diagnostic).

**Last updated**: 2026-07-22  
**Author**: Jiaming Ma + Claude (Kiro)  
**Reviewer**: 张冠兆 (pending)
