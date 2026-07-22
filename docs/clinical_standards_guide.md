# Clinical Standards Guide for CardiomniBench-VD

This guide defines the anchor standards used in gold_standard.yaml annotation and rubric grading. These are the authoritative clinical classification systems that agent outputs are judged against.

---

## 1. CAD-RADS (Coronary Artery Disease Reporting and Data System)

**Purpose:** Standardized per-vessel stenosis severity grading for CTA reports.

**Grades:**
- **CAD-RADS 0**: No plaque, no stenosis (0%)
- **CAD-RADS 1**: Minimal non-obstructive plaque (1-24% stenosis)
- **CAD-RADS 2**: Mild non-obstructive stenosis (25-49%)
- **CAD-RADS 3**: Moderate stenosis (50-69%)
- **CAD-RADS 4**: Severe stenosis (70-99%)
  - **4A**: Single-vessel severe stenosis
  - **4B**: Two-vessel severe stenosis
  - **4C**: Three-vessel or left-main severe stenosis
- **CAD-RADS 5**: Total occlusion (100%)

**Modifiers:**
- **N**: Non-diagnostic (motion/artifact)
- **S**: Stent present
- **G**: Graft present
- **V**: Vulnerability features (positive remodeling, low-attenuation plaque, spotty calcification, napkin-ring sign)

**Clinical implications:**
- CAD-RADS 0-2: Medical therapy, risk factor modification
- CAD-RADS 3: Consider functional testing (stress test, FFR-CT) if symptomatic
- CAD-RADS 4-5: Candidate for invasive angiography ± revascularization

**Reference:** Cury RC, et al. CAD-RADS 2.0 - 2022 Coronary Artery Disease Reporting and Data System. *Radiology: Cardiothoracic Imaging*. 2022;4(5):e220183.

---

## 2. SYNTAX Score (SYNergy between PCI with TAXUS and Cardiac Surgery)

**Purpose:** Quantify anatomical complexity of coronary artery disease to guide revascularization strategy (PCI vs CABG).

**Scoring:**
- Each lesion ≥50% stenosis in vessels ≥1.5mm diameter scores points based on:
  - **Segment location** (1-5 points per segment; proximal LAD = 5, distal branches = 1)
  - **Lesion characteristics**: total occlusion (+5), trifurcation (+1), bifurcation (+1), aorto-ostial (+1), severe tortuosity (+1), length >20mm (+1), heavy calcification (+1), thrombus (+1)
- Sum all lesion points → SYNTAX score

**16 SYNTAX Segments** (modified AHA classification):
1. RCA proximal  
2. RCA mid  
3. RCA distal  
4. Posterior descending artery (PDA)  
5. Left main (LM)  
6. LAD proximal  
7. LAD mid  
8. LAD distal  
9. First diagonal (D1)  
10. Second diagonal (D2)  
11. LCx proximal  
12. LCx mid / obtuse marginal (OM)  
13. LCx distal  
14. Posterolateral branch (PLB)  
15. Posterior descending from LCx (if left-dominant)  
16. Ramus intermedius (if present)

**Risk Tiers:**
- **Low** (0-22): PCI reasonable
- **Intermediate** (23-32): PCI vs CABG equipoise, discuss with Heart Team
- **High** (≥33): CABG preferred for complex 3-vessel / left-main disease

**Online calculator:** [syntaxscore.org](http://www.syntaxscore.org/)

**Reference:** Sianos G, et al. The SYNTAX Score: an angiographic tool grading the complexity of coronary artery disease. *EuroIntervention*. 2005;1(2):219-227.

---

## 3. TIMI Flow Grade

**Purpose:** Assess antegrade coronary flow on invasive angiography (DSA).

**Grades:**
- **TIMI 0**: No perfusion (complete occlusion, no antegrade flow)
- **TIMI 1**: Penetration without perfusion (contrast passes occlusion but doesn't fill distal bed)
- **TIMI 2**: Partial perfusion (slow flow, delayed distal filling/clearance)
- **TIMI 3**: Complete perfusion (normal flow, brisk filling and clearance)

**Clinical note:** TIMI 0-1 indicates acute total occlusion; TIMI 2 suggests critical stenosis or microvascular dysfunction.

---

## 4. TIMI Thrombus Grade

**Purpose:** Quantify intracoronary thrombus burden (acute coronary syndrome context).

**Grades:**
- **Grade 0**: No thrombus
- **Grade 1**: Possible thrombus (angiographic haziness, reduced contrast density)
- **Grade 2**: Definite thrombus, small (≤1/2 vessel diameter)
- **Grade 3**: Definite thrombus, moderate (>1/2 to <2× vessel diameter)
- **Grade 4**: Definite thrombus, large (≥2× vessel diameter)
- **Grade 5**: Total occlusion (no flow; cannot assess thrombus size, but occlusion present)

---

## 5. Rentrop Collateral Grade

**Purpose:** Assess collateral circulation to a chronically occluded territory (CTO evaluation).

**Grades:**
- **Grade 0**: No visible collaterals
- **Grade 1**: Collaterals fill side branches of occluded artery but not epicardial trunk
- **Grade 2**: Collaterals partially fill epicardial trunk
- **Grade 3**: Collaterals completely fill epicardial trunk (robust collateralization)

**Clinical note:** Rentrop 2-3 suggests viable myocardium supplied by collaterals; favorable for CTO-PCI.

**Reference:** Rentrop KP, et al. Changes in collateral channel filling immediately after controlled coronary artery occlusion by an angioplasty balloon in human subjects. *JACC*. 1985;5(3):587-592.

---

## 6. ACC/AHA Lesion Classification

**Purpose:** Predict PCI success and complication risk based on lesion morphology.

**Types:**
- **Type A** (success >85%, low risk): discrete (<10mm), concentric, readily accessible, non-angulated, smooth contour, no thrombus, no calcification
- **Type B1** (success 60-85%, moderate risk): 1 B feature: tubular 10-20mm, eccentric, moderate tortuosity, moderately angulated, irregular contour, moderate/heavy calcification, ostial, bifurcation
- **Type B2** (success 60-85%, moderate risk): ≥2 B features
- **Type C** (success <60%, high risk): diffuse >20mm, excessive tortuosity, extremely angulated, total occlusion >3 months, inability to protect major side branch, degenerated vein graft with friable lesion

**Reference:** Ryan TJ, et al. Guidelines for percutaneous transluminal coronary angioplasty. *Circulation*. 1993;88(6):2987-3007.

---

## 7. Agatston Score (Coronary Artery Calcium)

**Purpose:** Quantify calcified plaque burden on non-contrast cardiac CT (or CTA).

**Calculation:** Sum of (lesion area × density factor [1-4 based on peak HU]) across all slices.

**Categories:**
- **0**: No calcification (very low risk)
- **1-10**: Minimal (low risk)
- **11-100**: Mild (mild risk)
- **101-400**: Moderate (moderate risk)
- **>400**: Severe (high risk; ≥15% 10-year event risk)

**Clinical note:** Agatston >400 predicts high cardiovascular event risk but does NOT directly indicate stenosis severity (heavy calcification can occur without obstructive CAD, and vice versa). On CTA, calcification causes blooming artifact → DSA better for quantifying stenosis in heavily calcified lesions.

**Reference:** Agatston AS, et al. Quantification of coronary artery calcium using ultrafast computed tomography. *JACC*. 1990;15(4):827-832.

---

## 8. High-Risk Plaque Features (Vulnerability Markers)

**Purpose:** Identify plaques at high risk for rupture (acute coronary syndrome).

**CTA features:**
- **Low-attenuation plaque** (<30 HU): lipid-rich necrotic core
- **Positive remodeling** (remodeling index ≥1.1): outward vessel expansion
- **Spotty calcification**: small calcified nodules <3mm
- **Napkin-ring sign**: low-attenuation core with rim of higher attenuation (thin fibrous cap)

**Clinical note:** Presence of ≥2 features → "vulnerable plaque", higher risk for future events even if stenosis <50%.

**Reference:** Motoyama S, et al. Multislice computed tomographic characteristics of coronary lesions in acute coronary syndromes. *JACC*. 2007;50(4):319-326.

---

## 9. Coronary Dominance

**Definition:** Which artery (RCA or LCx) gives rise to the posterior descending artery (PDA) and supplies the inferior wall of the left ventricle.

- **Right dominance** (~70%): PDA from RCA
- **Left dominance** (~10%): PDA from LCx
- **Co-dominance** (~20%): PDA from both RCA and LCx

**Clinical relevance:** Left-dominant anatomy means LCx occlusion threatens a larger myocardial territory (inferior + lateral walls); critical for SYNTAX segment naming (segment 4 PDA comes from RCA in right-dominant, from LCx in left-dominant).

---

## Applying These Standards in CardiomniBench-VD

**For annotators (gold_standard.yaml):**
- Use exact grade/score terminology (e.g., "CAD-RADS 4A", "SYNTAX score 28, intermediate risk", "TIMI 3 flow")
- Cite the standard when ambiguous (e.g., "LAD 65% → CAD-RADS 3 per 50-69% definition")
- Document WHY a grade was chosen (e.g., "Agatston 450 → severe category per >400 threshold")

**For LLM judges (rubric grading):**
- An agent's output that says "70% LAD stenosis" should be graded as CAD-RADS 4, not 3 (70% crosses the severe threshold)
- SYNTAX segment numbering must match the 16-segment map exactly; wrong segment name = perception error
- Capability boundaries: asking for "functional significance" of a 60% lesion is reasonable; fabricating an FFR value without actual FFR-CT or catheter FFR = hallucination (negative points)

---

**Last updated:** 2026-07-22  
**Curator:** CardiomniBench-VD team
