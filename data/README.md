# CardiomniBench-VD Data Directory

This directory will contain individual case data for the benchmark. Each case consists of paired CTA and DSA DICOM files along with metadata and expert-annotated gold standard.

**IMPORTANT**: Currently, this directory is EMPTY (placeholder only). Real DICOM data will be added after expert annotation is complete, following the annotation protocol in `docs/annotation_protocol.md`.

---

## Directory Structure

```
data/
├── README.md                  # This file
├── splits.yaml               # Train/val/test splits (empty template)
└── cases/                    # Individual case folders
    ├── case_001/
    │   ├── cta.dcm          # CTA DICOM file
    │   ├── dsa.dcm          # DSA DICOM file (multi-frame XA)
    │   ├── metadata.yaml    # Case metadata (de-identification info, acquisition params)
    │   └── gold_standard.yaml  # Expert-annotated ground truth (from task_template.yaml)
    ├── case_002/
    │   └── ...
    └── ...
```

---

## Case Folder Contents

Each `case_XXX/` folder contains:

### 1. `cta.dcm`
- **Modality**: CT (Computed Tomography)
- **Type**: 3D volume (multi-slice DICOM)
- **Content**: Coronary CT Angiography with arterial phase contrast
- **Key metadata**: Slice thickness, pixel spacing, reconstruction kernel, HU calibration
- **Requirements**: De-identified per HIPAA/GDPR, all burned-in PHI removed

### 2. `dsa.dcm`
- **Modality**: XA (X-Ray Angiography)
- **Type**: Multi-frame 2D cine loop (dynamic contrast sequence)
- **Content**: Digital Subtraction Angiography showing coronary vessel opacification
- **Key metadata**: Number of frames, frame rate, projection angle, pixel spacing
- **Requirements**: De-identified, may include multiple projections per case

### 3. `metadata.yaml`
Case-level metadata including:
- De-identification record (original study date → offset date)
- Acquisition parameters (scanner model, protocol)
- Clinical context (age, sex, risk factors, chief complaint)
- Annotation metadata (primary annotator, reviewer, date, version)
- Quality flags (motion artifacts, contrast adequacy, calcium artifacts)

### 4. `gold_standard.yaml`
Expert-annotated ground truth following the structure in `tasks/task_template.yaml`:
- **Stage 0**: Dominance, segment identification
- **Stage 1a**: CTA findings (stenosis %, CAD-RADS, plaque, calcium score)
- **Stage 1b**: DSA findings (stenosis %, TIMI flow, collaterals, lesion morphology)
- **Stage 2**: Fusion reasoning (blooming correction, CTO assessment, culprit lesion)
- **Stage 3**: SYNTAX Score, risk stratification
- **Stage 4**: Clinical decision (PCI vs CABG, guideline-concordant recommendation)
- **Capability boundary**: Cases where FFR/IVUS/perfusion imaging would be needed

---

## Data Splits

Splits are defined in `splits.yaml` following a stratified design to ensure:
- Balanced representation of difficulty levels (easy/medium/hard)
- Coverage of key pathology types (multi-vessel, CTO, heavy calcium, bifurcation)
- Balanced CAD-RADS and SYNTAX Score distributions

**Proposed split ratios** (to be finalized after case collection):
- **Train**: 60% (for agent development and few-shot learning contexts)
- **Val**: 20% (for hyperparameter tuning, judge calibration)
- **Test**: 20% (held-out for final evaluation)

---

## Data Acquisition Status

**Current status**: EMPTY — pending expert annotation

**Next steps**:
1. Identify candidate CTA+DSA paired studies from clinical archive
2. Apply inclusion/exclusion criteria (see annotation protocol)
3. De-identify DICOM files (remove PHI, shift dates, anonymize UIDs)
4. Run data quality checks (motion artifacts, contrast adequacy, paired study verification)
5. Expert annotation following 4-step protocol (Author → Review → Solvability → QC)
6. Populate case folders with DICOM + metadata + gold_standard

**Timeline**: TBD based on IRB approval and expert availability

---

## Data Compliance

All data will be:
- **De-identified** per HIPAA Safe Harbor method (18 identifiers removed)
- **IRB-approved** with appropriate waiver or consent
- **Multi-center** to ensure generalizability (target: 3+ institutions)
- **Quality-controlled** via automated checks + expert review

**Storage**: 
- Raw DICOM files stored securely with access control
- Only de-identified data distributed in benchmark release
- Data use agreement required for benchmark access

---

## Adding a New Case

To add a new case to the benchmark:

1. **Prepare DICOM files**
   ```bash
   # De-identify using your preferred tool (e.g., DICOM Anonymizer, CTP, pydicom)
   # Verify modality tags: CT for CTA, XA for DSA
   # Ensure pixel data is present and parseable
   ```

2. **Create case folder**
   ```bash
   mkdir -p data/cases/case_XXX
   cp /path/to/deidentified_cta.dcm data/cases/case_XXX/cta.dcm
   cp /path/to/deidentified_dsa.dcm data/cases/case_XXX/dsa.dcm
   ```

3. **Fill metadata**
   ```bash
   cp tasks/task_template.yaml data/cases/case_XXX/gold_standard.yaml
   # Edit gold_standard.yaml with expert annotations
   ```

4. **Create metadata.yaml** with acquisition details and de-identification record

5. **Update splits.yaml** to assign case to train/val/test

6. **Run validation**
   ```bash
   python evaluation/validate_case.py data/cases/case_XXX
   ```

---

## Quality Criteria

Cases are included only if they meet:
- **Technical quality**: Motion artifacts < grade 2, adequate contrast opacification
- **Pairing**: CTA and DSA from same patient within 90 days
- **Diagnostic relevance**: At least one vessel with ≥50% stenosis or significant pathology
- **Annotatability**: Experts can reach consensus on gold standard within 2 review rounds
- **Diversity**: Contributes to balanced representation of pathology types and difficulty levels

Cases are EXCLUDED if:
- Prior CABG with complete revascularization (no native vessels to evaluate)
- Image quality non-diagnostic (cannot determine stenosis with confidence)
- Congenital coronary anomalies (non-standard anatomy)
- Experts disagree on gold standard after 2 review rounds

---

## Citation

When using CardiomniBench-VD data, please cite:

```
[Citation TBD after paper publication]
```

---

For questions about data acquisition, annotation protocol, or IRB compliance, contact: [TBD]
