# CardiomniBench-VD

**Process-level benchmark + evaluation pipeline for autonomous cardiovascular diagnosis agents.**

CardiomniBench-VD (**VD = Vascular Diagnosis**) is the companion benchmark to
**Cardiomni**, an autonomous agent that reads paired coronary **CTA (3D) + DSA (2D)**
DICOM studies and produces a structured diagnostic report. It is modeled on
**BiomniBench-DA** (Biomni's process-level evaluation companion): we score the
*reasoning process*, not just the final answer, across six weighted clinical dimensions.

The core novelty is **fusion reasoning** — knowing which modality to trust when CTA and
DSA disagree (e.g. calcium-blooming correction) — and **capability-boundary honesty**
(declaring "needs FFR/IVUS" instead of fabricating it).

> **Status:** the full pipeline runs end-to-end today with **mock backends** — no Docker,
> no API keys, no real data required. Clinical cases are intentionally empty pending
> expert annotation. Everything is config-driven so you can drop it on a GPU server and
> swap in a real agent, base model, judge, and dataset without touching code.

---

## Quickstart

```bash
pip install -r requirements.txt

# End-to-end gray-box run (mock agent + mock judge, one synthetic fixture case):
python -m pipeline.cli run --config configs/smoke.yaml

# Run the test suite (14 tests, fully offline):
python -m pytest tests/ -q

# Inspect what's available:
python -m pipeline.cli metrics                       # registered objective metrics
python -m pipeline.cli list  --config configs/smoke.yaml   # discovered cases
python -m pipeline.cli validate --config configs/default.yaml
```

A run writes results under `runs/<run_name>/`: per-case `prediction.json` +
`evaluation.json`, and a top-level `summary.json` (overall mean ± SD, per-dimension
breakdown).

---

## The four swap axes

Everything you'd want to change lives in one YAML config. Nothing is hard-coded.
This mirrors BiomniBench/Harbor's `harbor run --path <task> --agent <H> --model <M>`,
where the agent harness and the base model are independent axes.

| You want to swap… | Config knob | Notes |
|---|---|---|
| **基座 (base model)** | `agent.model` | Substituted into the agent command as `{model}` |
| **agent (harness)** | `agent.backend` + `agent.command` + `agent.image` | `mock` \| `local` \| `docker` |
| **rubric scorer** | `judge.backend` | `mock` \| `llm` \| `cli` |
| **tasks (dataset)** | `tasks.source` / `split` / `filter` / `limit` | Filter e.g. by `fusion_category` |

CLI flags override the config for one-off runs:

```bash
python -m pipeline.cli run --config configs/smoke.yaml \
    --agent-backend docker --agent-image cardiomni:latest \
    --model anthropic/claude-opus-4-8 --gpu \
    --judge-backend llm --judge-model claude-opus-4-8
```

### Agent contract

Any agent is plugged in the same way (so results are comparable across agents):

- **Input:** a case directory + a generated `task_spec.json` (clinical context and DICOM
  paths, **gold standard stripped out** — the agent stays blind to the answer).
- **Output:** the agent must write `prediction.json` into its output dir, following the
  section schema in `tasks/task_template.yaml` (`anatomical_localization`, `cta_findings`,
  `dsa_findings`, `fusion_analysis`, `comprehensive_scoring`, `clinical_decision`,
  `capability_boundary_statement`, plus a `reasoning_trace` / `report`).

### GPU

GPU is a first-class config flag: `agent.gpu: true` makes the docker backend add
`--gpus <device>`. See `configs/example_docker_gpu.yaml` for the deploy shape and
`docker/agent/Dockerfile` for the CUDA base image.

---

## Repository layout

```
CardiomniBench-VD/
├── README.md                 # this file — start here
├── requirements.txt          # runtime + test dependencies
├── conftest.py               # makes the repo root importable for pytest
│
├── pipeline/                 # ← the evaluation harness (the live code)
│   ├── config.py             #   loads/validates the run YAML (the 4 swap axes)
│   ├── runner.py             #   agent backends: mock | local | docker (+ --gpus)
│   ├── judge_backends.py     #   rubric scorers: mock | llm | cli
│   ├── metric_registry.py    #   maps rubric metric names → evaluation/metrics fns
│   ├── scoring.py            #   grade → points, per criterion
│   ├── orchestrator.py       #   discover → run → score → aggregate (mean ± SD)
│   └── cli.py                #   `python -m pipeline.cli {run,list,metrics,validate}`
│
├── configs/                  # run configurations
│   ├── default.yaml          #   real-data shape (0 cases until data is added)
│   ├── smoke.yaml            #   offline gray-box run (mock everything)
│   └── example_docker_gpu.yaml  # GPU-server deploy shape (docker + LLM judge)
│
├── rubrics/                  # evaluation rubric definitions
│   ├── rubric_dimensions.yaml   # the 6 weighted dimensions
│   ├── rubric_schema.yaml       # JSON-schema for per-case rubrics
│   ├── clinical_standards.yaml  # machine-readable CAD-RADS / SYNTAX / TIMI / …
│   └── examples/case_001_rubric.yaml  # worked example (illustrative)
│
├── tasks/
│   └── task_template.yaml    # the gold-standard schema — clone per new case
│
├── data/                     # clinical cases (EMPTY pending annotation)
│   ├── README.md             #   case folder layout + how to add one
│   ├── splits.yaml           #   train/val/test split template
│   └── cases/                #   case_XXX/{cta.dcm,dsa.dcm,gold_standard.yaml,rubric.yaml}
│
├── evaluation/
│   ├── metrics/              # objective metric functions (used by the pipeline)
│   ├── prompts/              # judge prompt template asset
│   └── legacy/               # ⚠ superseded pre-pipeline code, kept for reference only
│
├── tests/
│   ├── test_pipeline_smoke.py   # end-to-end offline tests
│   └── fixtures/tasks/case_smoke/  # synthetic case (NOT real clinical data)
│
├── docker/agent/Dockerfile   # CUDA base image for the agent (build on the server)
│
├── docs/                     # design + annotation documentation
│   ├── PROPOSAL.md           #   full design rationale + clinical citations
│   ├── annotation_protocol.md   # 4-stage expert annotation workflow
│   └── clinical_standards_guide.md  # human-readable guide to the standards
│
├── references/               # background material (not part of the pipeline)
│   ├── Biomni_Huang_et_al.pdf
│   ├── BiomniBench-DA_Qu_et_al.pdf
│   └── biomni_biomnibench_claims.md  # extracted claim library for the paper
│
└── paper/                    # the LaTeX manuscript (self-contained; its own git repo)
```

---

## Evaluation model

Each case is scored across **6 weighted dimensions** (defined in
`rubrics/rubric_dimensions.yaml`), mirroring BiomniBench-DA:

| Dimension | Weight | What it measures |
|---|---|---|
| `data_handling` | 0.10 | DICOM parsing, modality ID, HU extraction |
| `perception_accuracy` | 0.25 | Segment ID, stenosis %, CAD-RADS, plaque, calcium |
| `fusion_reasoning` | 0.20 | **CTA-DSA integration, blooming correction, CTO** (core novelty) |
| `clinical_interpretation` | 0.20 | SYNTAX score, risk tier, guideline-concordant decisions |
| `scientific_reasoning` | 0.15 | Reasoning coherence, evidence citation, explainability |
| `source_reliability` | 0.10 | Anti-hallucination (**negative points** for fabricated FFR/IVUS/labs) |

Each rubric criterion is scored **A/B/C** (like BiomniBench). Objective criteria use a
metric function and threshold ranges; subjective criteria use the judge backend, which
emits *only a grade label* — points come from the rubric's fixed grade→points table,
computed in code.

---

## Adding a real case

See `docs/annotation_protocol.md` for the full 4-stage expert workflow. In short:

1. De-identify a paired CTA+DSA study → `data/cases/case_XXX/{cta.dcm,dsa.dcm}`.
2. Author `gold_standard.yaml` from `tasks/task_template.yaml` (Expert A), review (Expert B).
3. Author `rubric.yaml` in the case folder (or reuse a default).
4. Run the solvability/QC checks and add the case to `data/splits.yaml`.

Then run `python -m pipeline.cli run --config configs/default.yaml`.

---

## Extending the pipeline

- **New objective metric:** add one adapter entry to `pipeline/metric_registry.py`
  (`REGISTRY`), then reference its name from a rubric criterion's `metric:` field.
- **New agent backend / judge backend:** add a branch in `pipeline/runner.py::run_agent`
  or `pipeline/judge_backends.py::make_judge`.
- **New run:** copy a config in `configs/` and change the four swap axes.

### Planned (not yet implemented)

- **Judge validation** (`pipeline/judge_validation.py`): before trusting the LLM judge,
  validate it against expert A/B/C labels — run multiple judges, report Cohen's κ and
  exact-match accuracy, pick the most reliable judge model. This mirrors BiomniBench-DA's
  core methodology ("prove the ruler is accurate before measuring with it").

---

## References

- **Biomni** (Huang et al., Stanford) — general-purpose biomedical agent. `references/`
- **BiomniBench-DA** (Qu et al.) — process-level evaluation companion. `references/`
- Design rationale and clinical citations: `docs/PROPOSAL.md`
