# CardiomniBench-VD Annotation Protocol

## Overview

CardiomniBench-VD cases are expert-authored diagnostic puzzles for evaluating autonomous cardiovascular AI agents. Each case consists of:
- **Paired DICOM imaging**: CTA (3D volume) + DSA (cine sequence)
- **Clinical context**: Patient demographics, symptoms, risk factors, reason for imaging
- **Gold standard**: Multi-stage expert diagnostic reasoning trace + structured outputs (anatomy, perception, fusion, scoring, decision, capability boundary)
- **Rubric**: Per-case evaluation criteria across 6 dimensions, anchored to clinical grading standards

This protocol ensures consistency, clinical validity, and adversarial robustness (mirroring DrugDiscoveryBench and BiomniBench-DA annotation best practices).

---

## 4-Stage Annotation Workflow

### Stage 1: Expert A Authors the Case (5-8 hours per case)

**Who:** Interventional cardiologist or cardiovascular imaging specialist (≥5 years experience, board-certified).

**Steps:**
1. **Select imaging pair**: Real de-identified CTA+DSA from a teaching file or consented research cohort. Inclusion criteria:
   - Diagnostic quality (motion artifacts acceptable if clinically interpretable)
   - Multi-vessel disease or complex anatomy (single-vessel mild stenosis = too easy)
   - CTA-DSA pairing within 30 days, same patient, similar clinical state
   - At least one diagnostic challenge: heavy calcification requiring DSA correction, CTO ambiguity, culprit-vessel determination, or capability-boundary declaration needed (e.g., requires FFR for intermediate lesion)

2. **Anonymize DICOM**: Strip all PHI (PatientName, PatientID → CASE_XXX, dates shifted). Verify with pydicom or DICOM anonymizer.

3. **Draft gold_standard.yaml**: Perform a full clinical read and document your reasoning across all 5 stages:
   - `stage0_anatomy`: Dominance, 16 SYNTAX segments present/absent, anomalies
   - `stage1a_cta`: Per-segment findings (stenosis %, plaque nature, Agatston, high-risk features)
   - `stage1b_dsa`: Per-segment TIMI flow, lesion morphology, thrombus grade, collaterals (Rentrop)
   - `stage2_fusion`: CTA-DSA concordance/discordance, blooming correction, CTO判断, culprit vessel
   - `stage3_scoring`: SYNTAX score (segment-by-segment breakdown), CAD-RADS per vessel, Agatston category, risk tier
   - `stage4_decision`: Revascularization strategy (medical/PCI/CABG), rationale
   - `capability_boundary`: What this case CANNOT determine from imaging alone (e.g., "functional significance of LAD 60% stenosis requires FFR", "myocardial viability requires CMR/PET")

4. **Author case-specific rubric** (`rubrics/examples/case_XXX_rubric.yaml`): Instantiate the 6 rubric dimensions with ~15-25 criteria tailored to this case. For each criterion:
   - A/B/C descriptions specific to this case's diagnostic challenges
   - Point allocation (A=full, B=partial, C=miss/error)
   - Evaluation method (automatic metric | llm_judge | hybrid)
   - At least 2 anti-hallucination criteria in `source_reliability` dimension (e.g., "Agent must NOT fabricate FFR value", points=-10 if violated)

5. **Write clinical context** (`metadata.yaml`): Age/sex (anonymized), presenting symptoms, risk factors, prior history, reason for imaging, prohibited resources (e.g., "no stress test available", "patient refused nuclear scan").

**Deliverables from Stage 1:**
- `data/cases/case_XXX/cta.dcm` (series or .tar of slices)
- `data/cases/case_XXX/dsa.dcm` (multi-frame XA)
- `data/cases/case_XXX/metadata.yaml`
- `data/cases/case_XXX/gold_standard.yaml`
- `rubrics/examples/case_XXX_rubric.yaml`
- Authoring time log

---

### Stage 2: Expert B Reviews (2-3 hours per case)

**Who:** Second interventional cardiologist or imaging specialist (different from Expert A; same credentials).

**Steps:**
1. **Blind read**: Perform an independent diagnostic read of the CTA+DSA pair using only the clinical context. DO NOT look at Expert A's gold_standard.yaml yet.
2. **Compare**: After completing your read, compare your findings to Expert A's gold_standard.yaml. Flag discrepancies (stenosis % >10% difference, CAD-RADS grade mismatch, SYNTAX segment naming conflict, capability-boundary disagreement).
3. **Adjudicate**: Discuss discrepancies with Expert A. Reach consensus or escalate to a third tiebreaker expert. Update gold_standard.yaml to consensus version.
4. **Review rubric**: Check that the case-specific rubric criteria are:
   - Clinically sound (A/B/C boundaries match real clinical decision thresholds)
   - Unambiguous (two judges would assign the same grade given the same agent output)
   - Exhaustive across the 5 stages (anatomy → decision)
   - Adversarially robust (anti-hallucination traps set)

**Deliverables from Stage 2:**
- Review report with inter-rater agreement metrics (Cohen's κ for categorical, ICC for continuous)
- Consensus gold_standard.yaml (if changed)
- Approved rubric

---

### Stage 3: Automated Solvability Check (10 minutes per case)

**Who:** QC script (run by dataset curator).

**Steps:**
1. **DICOM integrity**: Load CTA series and DSA cine with pydicom, verify pixel_array extractable, check required tags (Modality, SliceThickness, PixelSpacing).
2. **Gold standard schema validation**: Parse gold_standard.yaml against the task_template.yaml schema. All required fields must be non-empty (anatomy, perception, fusion, scoring, decision, capability_boundary).
3. **Rubric completeness**: Each of the 6 dimensions must have ≥2 criteria. At least one criterion must be automatic, at least one llm_judge. At least one negative-point anti-hallucination criterion.
4. **Clinical sanity checks**:
   - SYNTAX score ∈ [0, 48] (theoretical max ~47)
   - CAD-RADS grades ∈ {0, 1, 2, 3, 4, 5} + modifiers
   - Stenosis % ∈ [0, 100]
   - Agatston score ≥ 0
   - TIMI flow/thrombus/Rentrop grades within defined ranges

**Deliverables from Stage 3:**
- Pass/fail report per case
- If fail: return to Stage 1 or 2 for fix

---

### Stage 4: QC Signoff (30 minutes per case)

**Who:** Dataset PI or senior cardiovascular imaging faculty.

**Steps:**
1. **Review all Stage 1-3 artifacts**: Read the clinical context, view the DICOM pair (with a DICOM viewer), read the gold_standard.yaml, spot-check the rubric.
2. **Clinical realism**: Does this case reflect a real diagnostic challenge encountered in clinical practice? Is it too easy (single-vessel, obvious lesion) or impossibly hard (motion artifact obliterates all vessels)?
3. **Ethical check**: PHI fully anonymized? Consent documentation on file? No re-identifiable anatomical variants or rare diseases that could breach anonymity?
4. **Benchmark integrity**: Does the capability_boundary declaration prevent the case from being a "gotcha" (asking for functional significance without FFR is fair; asking for next year's plaque rupture is not)?

**Deliverables from Stage 4:**
- Signoff document (case_XXX_approved.txt with PI signature and date)
- Case added to `data/splits.yaml` train/val/test split

---

## Quality Metrics

Track these across the full benchmark:
- **Inter-rater agreement** (Stage 2): Cohen's κ ≥ 0.75 for CAD-RADS grading, ICC ≥ 0.80 for SYNTAX score, 100% consensus on capability_boundary declarations after adjudication
- **Difficulty distribution**: 40% intermediate, 40% hard, 20% very hard (subjective PI rating based on number of diagnostic challenges per case)
- **Anti-hallucination trap rate**: ≥30% of cases include at least one tempting-but-unprovable finding (e.g., clinical context mentions "prior MI" but asks agent to assess myocardial viability from CTA/DSA alone — agent must declare "requires CMR/PET" not fabricate a viability claim)

---

## Timeline

- **Per case**: 8-12 expert-hours total (5-8h Stage 1, 2-3h Stage 2, 10min Stage 3, 30min Stage 4)
- **20-case pilot**: ~200 expert-hours (~5 weeks with 2 experts working part-time)
- **50-case full benchmark**: ~500 expert-hours (~3 months with rotating expert pool)

---

## References

- DrugDiscoveryBench annotation protocol (4-stage with inter-rater adjudication)
- BiomniBench-DA process-level rubric design (6 dimensions + anti-hallucination)
- CardiomniBench-VD PROPOSAL.md §3 (5-stage task decomposition) and §4 (rubric anchor standards)
