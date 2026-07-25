# Agent-Facing Spec — How an Agent Solves CardiomniBench-VD

This is the contract to design the Cardiomni agent (and any baseline) against.
For every task the agent receives a `task_spec.json` and must write a
`prediction.json`. Nothing else is scored. The harness is fixed; only the
agent/base-model changes (SWE-bench / MLE-bench paradigm).

## Universal I/O contract

**Reads** (in `output_dir`): `task_spec.json` =
`{case_id, task_type, input, expected_output}` — **no gold**. Input file paths in
`input` are relative to the **case folder** (`{task_dir}` placeholder in the run
command). The agent may open the referenced image/volume/video files.

**Writes**: `prediction.json` in `output_dir`. Schema depends on `task_type`
(below). The agent must key its behavior off `task_spec.task_type`.

**Must not**: read `task.yaml`, the `gold/` tree, or any `*label*` file.

---

## Per-task read/write

### `arcade_segmentation`
- Read: `input.image.file_path` (512² PNG).
- Write:
  ```json
  {"instances": [{"label": "6", "bbox_xywh_norm": [x,y,w,h], "mask": <rle|array>}]}
  ```
  `label` ∈ 25 SYNTAX ids. Emit one instance per visible segment.

### `arcade_stenosis`
- Read: `input.image.file_path` (512² PNG).
- Write:
  ```json
  {"instances": [{"label": "stenosis", "bbox_xywh_norm": [x,y,w,h], "mask": <...>}]}
  ```
  One instance per stenosis. No percent, no segment id.

### `cca_segmentation`
- Read: `input.volume.file_path` (3D `.nii.gz`), use `shape`/`spacing_mm`.
- Write: a NIfTI binary mask (same shape) to `output_dir`, plus
  `{"mask_file": "pred_mask.nii.gz"}` in `prediction.json`.

### `cardiosyntax_scoring`
- Read: all `input.views[]` (each `file_path` `.npy` cine + `artery` + angles).
- Write:
  ```json
  {"syntax_score": 12.0, "syntax_left": 7.0, "syntax_right": 5.0, "dominance": "right"}
  ```
  Integrate evidence across all views (study-level answer).

---

## How the tasks map to the Cardiomni four-stage SOP

Cardiomni encodes the interventional-cardiology reading SOP as an explicit,
tool-grounded pipeline. Each public task exercises a slice of it, with a
gold-checkable output:

| SOP stage | What it does | Exercised / checked by |
|---|---|---|
| **1. Dominance** | right/left from PDA origin | `cardiosyntax_scoring.dominance` |
| **2. Systematic segment scan** | name every segment, no omission | `arcade_segmentation` (per-segment instances) |
| **3. View selection** | pick the projection that best shows a segment (no foreshorten/overlap) | `cardiosyntax_scoring` (multi-view integration) |
| **4. Lesion assessment** | locate + (later) grade stenosis | `arcade_stenosis` (locate); grading = future gold |

The full DSA-report task (`data/cases/case_chxc_001`) is where all four stages
compose into a prose report (dominance + per-segment stenosis %). The public tasks
are the decomposed, individually-verifiable sub-skills.

## Shared tool suite (exposed to every harness → contribution is orchestration)

Per the MLE-bench paradigm, specialist models are **shared tools**, not
competitors — every harness may call them, so performance differences are
attributed to *orchestration*, not tool ownership. Repo location:
`algorithms/specialist_models/` (integration status tracked in
`algorithms/INTEGRATION_STATUS.md`).

| Tool | Input → Output | Serves task/stage |
|---|---|---|
| **SAM-VMNet** (vessel seg) | XCA/CTA image → vessel mask → centerline → diameter | seg tasks; Stage 2/4 |
| **QCA-style diameter** | image + lesion ROI → MLD, reference diameter, %DS, length | Stage 4 quantify |
| **stenosis detector** (StenUNet-style) | XCA frame → stenosis regions | `arcade_stenosis`; Stage 4 locate |
| **dominance classifier** | RCA cine → {right,left} | `cardiosyntax`; Stage 1 |
| **DeepCoro / DeepCORO-CLIP** (video) | cine video(s) → stenosis box + segment + % ; study-level 2nd opinion | Stage 4; multi-view integration |

> The exact CLI/python call and weight paths for each are in each model's README
> under `algorithms/specialist_models/<name>/`; confirm integration status there
> before wiring a tool call. Tools are optional — an agent may reason directly
> from pixels — but tool-grounded evidence is what the "tool-orchestration" and
> "reasoning-traceability" axes will reward.

## Design guidance (from EchoAgent lessons, staged clinical agents)
1. **Stage the pipeline with explicit intermediate state** (dominance → scan →
   view-select → lesion), passing a structured state object between stages.
2. **Ground every quantitative claim in a tool call or an image region** — no
   free-floating numbers (anti-hallucination).
3. **Enumerate expected segments systematically** to avoid omission (Stage 2).
4. **Separate perception from interpretation** — detect/segment first, grade/
   integrate second.
5. **Reserve `reasoning_trace` / evidence fields** in `prediction.json` linking
   each conclusion to a view/frame/tool output (for later traceability scoring).

## What is NOT here yet (deferred)
- Rubric / scoring adapters per task (F1, Dice, MAE) — infra exists, wiring TBD.
- Stenosis **percent** and **segment attribution** gold — no public label; needs
  expert annotation (中山 template) for the full report task.

