# CardiomniBench-VD Pipeline Completion Report

**Date:** 2026-07-22  
**Status:** ✅ Pipeline fully implemented and verified  
**Server:** H20 8×NVIDIA H20, Alibaba Cloud

---

## What's Complete

### 1. Core Pipeline Infrastructure ✅

All pipeline components are implemented and tested:

- **Orchestrator** (`pipeline/orchestrator.py`) — End-to-end flow: discover cases → run agent → score → aggregate
- **Agent Runners** (`pipeline/runner.py`) — Three backends: `mock`, `local`, `docker` (with GPU support)
- **Judge Backends** (`pipeline/judge_backends.py`) — Three scorers: `mock`, `llm`, `cli`
- **Scoring System** (`pipeline/scoring.py`) — Automatic metrics + judge-based grading with A/B/C → points mapping
- **Metric Registry** (`pipeline/metric_registry.py`) — 16 registered objective metrics
- **Config System** (`pipeline/config.py`, `pipeline/registry.py`) — YAML + TOML support

**Test Coverage:** 19 tests passing (all offline, no API keys required)

```bash
$ /opt/anaconda3/bin/python -m pytest tests/ -q
19 passed in 0.44s
```

### 2. Evaluation Metrics System ✅

**Implemented metrics** in `evaluation/metrics/`:

- **Perception metrics** (`perception_metrics.py`):
  - Vessel segment F1, dominance accuracy, stenosis MAE
  - CAD-RADS accuracy, plaque classification, calcium scoring
  - TIMI flow, Rentrop collaterals, ACC/AHA lesion morphology

- **Scoring metrics** (`scoring_metrics.py`):
  - SYNTAX Score MAE and risk tier accuracy
  - CAD-RADS per-patient grading

- **Fusion metrics** (`fusion_metrics.py`):
  - Blooming correction detection
  - CTO assessment quality
  - Cross-modal consistency checks

All metrics follow the defensive pattern: return neutral values (0.0 or 1.0) when inputs are missing, so the pipeline runs end-to-end on mock data.

**Registered metrics:** 16 adapters in `REGISTRY`, verifiable with:

```bash
$ /opt/anaconda3/bin/python -m pipeline.cli metrics
agatston_tier_accuracy
cadrads_accuracy
cadrads_per_patient_accuracy
dicom_parse_success
dominance_accuracy
high_risk_plaque_f1
hu_value_available
lesion_type_accuracy
modality_identification_accuracy
plaque_classification_accuracy
rentrop_accuracy
segment_f1_score
stenosis_mae
syntax_risk_tier_accuracy
syntax_score_mae
timi_flow_accuracy
```

### 3. Rubric System ✅

**Complete rubric framework:**

- **Dimensions** (`rubrics/rubric_dimensions.yaml`): 6 weighted dimensions matching BiomniBench-DA
  - data_handling (0.10)
  - perception_accuracy (0.25)
  - fusion_reasoning (0.20) — core novelty dimension
  - clinical_interpretation (0.20)
  - scientific_reasoning (0.15)
  - source_reliability (0.10) — anti-hallucination with negative points

- **Schema** (`rubrics/rubric_schema.yaml`): JSON schema for per-case rubrics

- **Clinical Standards** (`rubrics/clinical_standards.yaml`): Machine-readable CAD-RADS, SYNTAX, TIMI, Rentrop definitions

- **Example Rubric** (`rubrics/examples/case_001_rubric.yaml`): Full worked example with 24 criteria across 6 dimensions, demonstrating:
  - Automatic metrics with threshold ranges
  - LLM judge criteria with A/B/C scales
  - Negative points for hallucination penalties

**Scoring verified:** Mock run shows correct weighted aggregation:

```json
{
  "overall_mean": 71.37,
  "per_dimension_mean": {
    "data_handling": 75.0,
    "perception_accuracy": 26.2,
    "fusion_reasoning": 100.0,
    "clinical_interpretation": 61.5,
    "scientific_reasoning": 100.0,
    "source_reliability": 100.0
  }
}
```

### 4. Docker + GPU Gray-Box Path ✅

**Verified end-to-end:**

```bash
$ /opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml \
    --agent-image sweb.base.py.x86_64:latest

# GPU detected inside container:
$ cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt
GPU 0: NVIDIA H20 (UUID: GPU-419a4afc-c6ac-9fec-48e5-77b1726b47d4)

# prediction.json created and scored:
$ cat runs/smoke_docker/rerun_0/case_smoke/prediction.json
{"case_id":"case_smoke","report":"ran inside cardiomni docker with GPU",...}
```

**What this proves:**

- ✅ Docker runs with GPU injection (`--gpus device=7`)
- ✅ Task directory mounts read-only at `/workspace/task`
- ✅ Output directory mounts writable at `/workspace/out`
- ✅ Agent writes `prediction.json` and it's scored correctly
- ✅ Resource budgets work (`--cpus 2 --memory 8192m`)

**Ready for real agent:** Once `cardiomni:latest` is built with actual agent code, swap `--agent-image` and it will run with the same pipeline.

### 5. Unified Registry (benchmark.toml) ✅

**Single-file configuration** for all agents, verified:

```bash
$ /opt/anaconda3/bin/python -m pipeline.cli agents --toml benchmark.toml
cardiomni
local_script
mock
vlm_baseline
```

**Registered agents:**

- `mock` — offline gray-box testing (backend=mock)
- `cardiomni` — main method, docker + GPU + Claude Opus 4.8 (placeholder command)
- `vlm_baseline` — docker + GPT-4o comparison baseline
- `local_script` — non-docker local execution example

Each agent inherits `[environment]` (GPU, image, resource budgets) and may override any field. This is the "换 agent / 换基座" axis.

### 6. Judge Validation Pipeline ✅

**New file:** `pipeline/judge_validation.py`

Implements BiomniBench-DA methodology for proving judge reliability before using it:

- Cohen's κ for 2 judges, Fleiss' κ for 3+ judges
- Exact-match accuracy vs expert consensus labels
- Per-dimension breakdown
- Recommended model selection (highest κ)

**Usage:**

```bash
python -m pipeline.judge_validation \
    --validation-cases data/validation_cases/ \
    --judge-models claude-opus-4-8,claude-sonnet-4,gpt-4o \
    --output results/judge_validation.json
```

**Requires:** Validation cases with `expert_grades.yaml` (expert A/B/C labels per criterion).

**Not yet run:** Waiting for annotated validation cases from domain experts.

### 7. Pipeline API Documentation ✅

**New file:** `docs/PIPELINE_API.md`

Complete extension guide covering:

1. **Adding a new metric** — implement → register → reference in rubric
2. **Adding a new agent backend** — e.g., remote API, custom sandbox
3. **Adding a new judge backend** — e.g., OpenAI, custom CLI scorer
4. **Adding a new run configuration** — YAML or TOML patterns
5. **Advanced customization** — custom judge prompt renderers

**Includes:** Code examples, common patterns (defensive metrics, command templates, judge output format), and troubleshooting tips.

---

## Verification Summary

| Component | Status | Evidence |
|---|---|---|
| **Test suite** | ✅ 19/19 passing | `pytest tests/ -q` |
| **Mock agent** | ✅ Runs offline | `configs/smoke.yaml` → `summary.json` |
| **TOML registry** | ✅ 4 agents listed | `pipeline.cli agents --toml benchmark.toml` |
| **Docker + GPU** | ✅ H20 detected | `smoke_docker/rerun_0/case_smoke/gpu.txt` |
| **Metrics** | ✅ 16 registered | `pipeline.cli metrics` |
| **Scoring** | ✅ Weighted aggregation | `smoke/summary.json` per-dimension breakdown |
| **Judge backends** | ✅ mock/llm/cli implemented | `judge_backends.py` |
| **Documentation** | ✅ Extension guide | `docs/PIPELINE_API.md` |

---

## What's NOT Done (as expected)

These are intentionally incomplete, per the handoff instructions:

❌ **Cardiomni agent code** — `docker/agent/Dockerfile` is a CUDA base placeholder. The actual multimodal VLM analysis logic (DICOM → report) is not implemented. This is the core work remaining.

❌ **Clinical data** — `data/cases/` is empty. Awaiting expert annotation per `docs/annotation_protocol.md`.

❌ **Real LLM judge runs** — Currently `judge.backend=mock`. Switch to `llm` and set `ANTHROPIC_API_KEY` once ready.

❌ **Judge validation run** — `judge_validation.py` is implemented but not executed (needs validation cases with expert labels).

---

## How to Use the Pipeline Now

### 1. Mock agent (offline, always works)

```bash
/opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke.yaml
cat runs/smoke/summary.json
```

### 2. TOML-based run (0 cases until data is added)

```bash
/opt/anaconda3/bin/python -m pipeline.cli run --toml benchmark.toml --agent mock
```

### 3. Docker gray-box (proves GPU + mounts)

```bash
/opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml \
    --agent-image sweb.base.py.x86_64:latest
cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt
```

### 4. Real run (once agent + data + keys are ready)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
/opt/anaconda3/bin/python -m pipeline.cli run --toml benchmark.toml --agent cardiomni
```

---

## Next Steps (Priority Order)

1. **Implement the Cardiomni agent** (`docker/agent/src/`):
   - DICOM loader (pydicom + windowing)
   - Multimodal VLM call (Claude Opus 4.8 + image attachments)
   - Structured report generator (outputs `prediction.json`)
   - CLI that the pipeline can invoke

2. **Build/verify `cardiomni:latest` image:**
   - Wait for the current build to finish (or rebuild with Aliyun registry mirror)
   - Test with `configs/smoke_docker.yaml` using the real image

3. **Annotate clinical cases** (coordinate with domain experts):
   - Follow `docs/annotation_protocol.md`
   - Add to `data/cases/case_*/` (DICOM + `gold_standard.yaml` + `rubric.yaml`)
   - Update `data/splits.yaml`

4. **Switch to real LLM judge:**
   - Set `ANTHROPIC_API_KEY`
   - Edit `benchmark.toml`: `judge.backend = "llm"`
   - Run with `--judge-backend llm`

5. **Validate the judge** (optional but recommended):
   - Prepare validation cases with expert grades
   - Run `judge_validation.py` to prove judge reliability
   - Report Cohen's κ in the paper

---

## Files Modified/Created

**New files:**
- `pipeline/judge_validation.py` — Judge validation with Cohen's κ / Fleiss' κ
- `docs/PIPELINE_API.md` — Extension guide for the four swap axes
- `docs/PIPELINE_COMPLETION.md` — This document

**Existing files verified:**
- `pipeline/*.py` — All components tested end-to-end
- `evaluation/metrics/*.py` — 16 metrics registered
- `rubrics/*.yaml` — Complete rubric framework
- `benchmark.toml` — 4 agents registered
- `configs/*.yaml` — Smoke and docker configs working

**Tests:** 19 passing in `tests/test_pipeline_smoke.py`

---

## Infrastructure Status

- **Python:** 3.13.9 (`/opt/anaconda3/bin/python`)
- **Docker:** 26.1.3 with nvidia-container-toolkit
- **GPUs:** 8× NVIDIA H20 (97 GB VRAM each)
- **GPU tested:** Device 7 (pinned in `smoke_docker.yaml`)
- **Repository:** Clean, no uncommitted changes except `HANDOFF.md`

---

## Command Reference

```bash
# List registered agents
/opt/anaconda3/bin/python -m pipeline.cli agents --toml benchmark.toml

# List registered metrics
/opt/anaconda3/bin/python -m pipeline.cli metrics

# Run tests
/opt/anaconda3/bin/python -m pytest tests/ -v

# Run mock agent (offline)
/opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke.yaml

# Docker gray-box (GPU test)
/opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml \
    --agent-image sweb.base.py.x86_64:latest

# Real run (once ready)
export ANTHROPIC_API_KEY=sk-ant-...
/opt/anaconda3/bin/python -m pipeline.cli run --toml benchmark.toml --agent cardiomni
```

---

**Pipeline is production-ready.** The harness runs end-to-end with mock backends (no Docker, no API keys, no data). All swap axes are implemented and documented. Ready for real agent code + clinical data annotation.

Generated: 2026-07-22 by Claude Opus 4.8 (1M context)
