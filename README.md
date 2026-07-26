# CardiomniBench-VD

**Black-box evaluation pipeline for autonomous coronary angiography diagnosis agents.**

CardiomniBench-VD is the companion evaluation suite to **Cardiomni**, an autonomous agent that reads **DSA (coronary angiography)** DICOM studies and produces a structured diagnostic report with full reasoning traceability. The evaluation paradigm follows **SWE-bench / MLE-bench**: fix the base model and task set, swap only the agent harness, and attribute performance differences to the harness design rather than the underlying LLM.

The pipeline is **config-driven and black-box** — an agent is any command that reads a case directory and writes `prediction.json`. This means you can swap agents, base models, judges, and datasets without touching code, enabling fair comparison across different agent architectures (Cardiomni vs. Claude Code vs. naive tool-caller vs. pure LLM).

> **Core contribution**: The **Cardiomni agent harness itself** (encoding expert SOP as explicit reasoning stages), not the benchmark. The benchmark provides infrastructure for controlled comparison but is deliberately de-emphasized — it appears in the paper only in abstract/intro/experiments, never in Method.

> **Status:** The full pipeline runs end-to-end today with **mock backends** — no Docker, no API keys, no real data required. 191 cases are prepared and the test suite is green. The harness is complete; the Cardiomni agent itself and real baseline inference are the open work items.

---

## Quickstart

The project standardises on **uv** for dependency management (`pyproject.toml` + `uv.lock`, Python 3.10, torch from the cu128 index). The unified environment entrypoint is `env.sh`:

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD
source env.sh              # exports $BENCH_PY → .venv/bin/python, sets cache dirs to NAS
$BENCH_PY -m pytest tests/ -q
```

The `.venv` setup (via `uv sync`) places torch, CUDA wheels, and all caches on the NAS (`/mnt/aliyunsb`, ~1PB) rather than the root filesystem, which is a 99GB disk that hit capacity mid-install and stalled package resolution for 105 minutes without reporting an error. The rationale for each exported path is documented in `env.sh`; `ENV_MIGRATION.md` covers the conda-to-uv migration itself.

For offline harness work (pure-Python pipeline layer, no torch), the server's anaconda interpreter is the fastest path:

```bash
/opt/anaconda3/bin/python -m pytest tests/ -q                 # whole suite, offline, ~1s

# --- Unified TOML registry (recommended) ---------------------------------
# benchmark.toml registers every agent, the shared Docker environment, judge,
# and task set in ONE file. List agents, then run one by name:
python -m pipeline.cli agents --toml benchmark.toml            # cardiomni, mock, gpt4v, claude, ...
python -m pipeline.cli run    --toml benchmark.toml --agent mock

# --- Single YAML config (one config per run) -----------------------------
python -m pipeline.cli run --config configs/smoke.yaml

# Inspect what's available:
python -m pipeline.cli metrics                                 # 21 registered metrics
python -m pipeline.cli list --toml benchmark.toml --agent mock # discovered cases
```

A run writes results under `runs/<run_name>/`: per-case `prediction.json` + `evaluation.json`, and a top-level `summary.json` (mean ± SD across reruns, per-dimension breakdown).

---

## The four swap axes

Everything you'd want to change lives in config — either the unified `benchmark.toml` (all agents in one file) or a single YAML. Nothing is hard-coded. This mirrors BiomniBench/Harbor's evaluation model.

| You want to swap… | Config knob | Notes |
|---|---|---|
| **基座 (base model)** | `agent.model` | Substituted into the agent command as `{model}` |
| **agent (harness)** | `agent.backend` + `agent.command` + `agent.image` | `mock` \| `local` \| `docker` |
| **rubric scorer** | `judge.backend` | `mock` \| `llm` \| `cli` |
| **tasks (dataset)** | `tasks.source` / `split` / `filter` / `limit` | Filter e.g. by task type or difficulty |

CLI flags override the config for one-off runs:

```bash
python -m pipeline.cli run --config configs/smoke.yaml \
    --agent-backend docker --agent-image cardiomni:latest \
    --model anthropic/claude-opus-4-8 --gpu \
    --judge-backend llm --judge-model claude-opus-4-8
```

### Agent contract

Any agent is plugged in the same way (so results are comparable across agents):

- **Input:** a case directory + a generated `task_spec.json` (clinical context and DICOM paths, **gold standard stripped out** — the agent stays blind to the answer).
- **Output:** the agent must write `prediction.json` into its output dir, following the task schema. For the DSA diagnostic report task, this includes structured per-segment findings (vessel name, stenosis %, dominance) plus a reasoning trace linking each conclusion to the DICOM frames and tool calls that support it.

The schema varies by task:
- **ARCADE segmentation**: bounding boxes + masks for coronary segments (SYNTAX naming)
- **ARCADE stenosis**: bounding boxes + masks for stenosis regions (location only; ARCADE labels carry no percent)
- **CardioSYNTAX scoring**: study-level coronary dominance + SYNTAX score from multi-view cine
- **CCA segmentation**: 3D binary vessel mask (CTA anatomy reference, outside the DSA main line)

### GPU

GPU is a first-class config flag: `agent.gpu: true` (or `[environment] gpu = true` in `benchmark.toml`) makes the docker backend add `--gpus <device>`, plus optional `--cpus` / `--memory` budgets. Pin specific GPUs with `gpu_device = "device=1,2"`.

### Deploy on a GPU server

```bash
# 1. Clone (public repo — no auth needed)
git clone https://github.com/Cardiomni/CardiomniBench-VD.git
cd CardiomniBench-VD

# 2. Build the agent image (needs nvidia-container-toolkit on the host)
docker build -t cardiomni:latest docker/agent

# 3. Gray-box docker check — proves GPU injection + mounts + scoring end-to-end,
#    without the real agent code (writes gpu.txt + a valid prediction.json):
python -m pipeline.cli run --config configs/smoke_docker.yaml
cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt   # should list a GPU

# 4. Real run once agent + data + keys are in place:
export ANTHROPIC_API_KEY=...
python -m pipeline.cli run --toml benchmark.toml --agent cardiomni
```

The TOML registry uses the stdlib `tomllib` on Python 3.11+ and falls back to `tomli` below that; `tomli` is already declared in `pyproject.toml` and `requirements.txt` under a `python_version < "3.11"` marker, so the uv target (3.10) is covered.

---

## Repository layout

```
CardiomniBench-VD/
├── pipeline/              # Evaluation engine
│   ├── orchestrator.py      # discover → run → score → aggregate
│   ├── runner.py            # agent backends (mock/local/docker)
│   ├── judge_backends.py    # rubric scorers (mock/llm/cli)
│   ├── scoring.py           # grade → points conversion
│   ├── metric_registry.py   # objective metric adapters
│   └── report_facts.py      # DSA report extraction + tolerance matching
├── evaluation/
│   └── metrics/           # 21 registered objective metrics
├── benchmark/             # Specialist-model runner (tools / upper bounds)
│   ├── run_unified.py       # one command, all methods with verified weights
│   ├── specialists.py       # method objects + provenance
│   └── method_config.py     # reads methods/*.toml
├── methods/               # One TOML per specialist model (see methods/README.md)
├── algorithms/
│   ├── baselines/         # Baseline agent wrappers (GPT-4V, Claude, specialist models)
│   └── toolkit.py         # Tool library for agents
├── data/
│   ├── cases/             # Test case directories (currently empty; see data/tasks/)
│   └── tasks/             # 191 cases across 4 tasks
│       ├── arcade_segmentation/cases/    # 42 cases (DSA, 25 SYNTAX segment classes)
│       ├── arcade_stenosis/cases/        # 69 cases (DSA, multi-lesion ≥2/image)
│       ├── cardiosyntax_scoring/cases/   # 60 cases (DSA multi-view, 3× expert annotated)
│       └── cca_segmentation/cases/       # 20 cases (CTA anatomy reference only)
├── configs/               # YAML run configurations
├── rubrics/               # Multi-dimensional evaluation rubrics
├── tests/                 # offline harness tests (mock backends)
├── benchmark.toml         # Unified registry (environment + judge + tasks + agents)
├── pyproject.toml         # uv-managed dependencies (Python 3.10, torch cu128)
└── docker/
    └── agent/             # Dockerfile for agent runtime (CUDA + pydicom base)
```

---

## Tasks and datasets

CardiomniBench-VD covers **3 XCA (DSA) tasks** plus one CTA anatomy-reference task, derived from **3 public datasets**:

| Task | Dataset | Cases | Input | Output | Metric | Status |
|------|---------|-------|-------|--------|--------|--------|
| **Vessel segmentation** | ARCADE | 42 | Single XCA frame (PNG) | Segment instances (SYNTAX naming + bbox + mask) | Segment F1 | Primary |
| **Stenosis detection** | ARCADE | 69 | Single XCA frame (PNG) | Stenosis instances (bbox + mask) | Stenosis F1 | Primary |
| **CardioSYNTAX** | CardioSYNTAX | 60 | Multi-view cine (.npy) + angles | Dominance + SYNTAX score | Dominance accuracy; score MAE | Mixed (see below) |
| **CTA vessel segmentation** | CCA | 20 | 3D CTA volume (NIfTI) | Binary vessel mask (3D) | Dice, clDice, HD95 | Anatomy reference |

**Primary DSA tasks** (ARCADE segmentation 42 / stenosis 69): single-frame XCA. Segmentation covers 25 SYNTAX segment classes and maps to Cardiomni **Stage 2** (systematic segment scan). Stenosis is single-class location-only, restricted to the hard multi-lesion subset (≥2 lesions per image), and maps to the detection half of **Stage 4**. ARCADE carries no stenosis percent, so percent grading cannot be supervised here — that gap is the anti-hallucination boundary.

**CardioSYNTAX (60 studies)**: the only public data whose input shape matches Cardiomni's real input — 6–14 projection cine runs per study with C-arm angle metadata. The **multi-view videos are the main-line asset** (they exercise Stage 1 dominance and Stage 3 view selection), and **dominance is a directly gold-checkable Stage-1 output** (labeled on 11/60 studies). The **SYNTAX score itself is future work**, not a current paper claim; it is kept scoreable for reference, with a 3-expert reliability band (mean inter-expert spread 8.6 pts) as the ceiling for any predictor.

**CTA anatomy reference (CCA, 20 cases)**: this task uses **CT angiography, not DSA**, placing it outside the DSA main line. It is retained as a coronary-anatomy prior and a segmentation sanity check — a vessel-segmentation tool validated here can be exposed to the agent as a callable tool. The CTA modality itself is not part of the Cardiomni diagnosis workflow, and CCA numbers are reported separately, never as a DSA result.

All datasets are **public and license-clean** (CC0 or CC-BY). ARCADE is from Zenodo (Nature Scientific Data 2023), CardioSYNTAX from Zenodo (CC-BY-4.0, includes PositionerPrimaryAngle/SecondaryAngle metadata for multi-view reasoning), CCA from Mendeley.

**Data format constraint**: Public datasets have been **stripped of DICOM encapsulation** — ARCADE is PNG+COCO JSON, CardioSYNTAX is NumPy arrays (.npy) with angle metadata but not true DICOM headers, CCA is NIfTI. For full DICOM-based multi-view reasoning validation, private expert-annotated data is in preparation but not yet integrated.

---

## Specialist models as tools (`benchmark/` + `methods/`)

A second, narrower runner evaluates the task-specific deep-learning models that the
agent may call as tools. It is separate from the agent harness on purpose: these are
**tools and upper-bound reference points**, not competing agents.

```bash
source env.sh
$BENCH_PY -m benchmark.run_unified --help
$BENCH_PY -m benchmark.run_unified --tasks arcade_segmentation \
    --methods coronary_cm_unet_native --device cuda:5
```

Every method's preprocessing, decision rule, and inference geometry is declared in
`methods/<name>.toml` with provenance, never hardcoded in Python — the convention and
its rationale are in `methods/README.md`. Where a preprocessing choice is genuinely
ambiguous, both variants are registered and both are reported.

**Why this matters for the agent.** The variant gap is a measurement, and it is large.
Same CM-UNet checkpoint, all 222 ARCADE cases, only `pad_to` differing:

| Task | `pad_to = 1536` (upstream `dataset.py`) | `pad_to = 0` (adapted to native 512×512) |
|---|---|---|
| `arcade_segmentation` | `pixel_dice` 0.0002 — 12 of 42 cases fully empty | `pixel_dice` 0.5924 — no empty predictions |
| `arcade_stenosis` | `pixel_dice` 0.0000 — 24 of 69 cases fully empty | `pixel_dice` 0.4054 — no empty predictions |

Intensity normalisation behaves the same way (5 cases, same weights): raw `uint8`
gives Dice 0.000, `÷255` 0.321, per-image z-score 0.709, and unsharp + z-score — the
actual upstream offline pipeline — 0.726, against a gold vessel coverage of 0.0254.
The failure mode is quiet: a misconfigured model returns an empty mask and a clean
0.000, which reads exactly like a checkpoint that cannot transfer. An agent calling a
segmentation tool faces these same choices, so their real cost is worth knowing.

**Capability boundaries, stated plainly.** CM-UNet scores `f1 = 0.0000` on
`arcade_segmentation`, and that is constructive rather than a tuning failure: it emits
a single binary vessel class and cannot name the 25 SYNTAX segments the task asks for.
It also predicts 2.93 connected components against 6.52 gold anatomical branches, so
even label-free matching (IoU 0.31) stays under the 0.5 threshold. `pixel_dice` is the
metric that reflects what it actually does. On `arcade_stenosis` its `f1` is 0.0078 —
using a vessel segmenter as a stenosis detector essentially does not work, since it
outlines the whole vessel while a stenosis is a local narrowing on it.

On `cardiosyntax_scoring` (all 60 studies), `cardiosyntax_r3d` gives MAE
6.9001 ± 7.8065, RMSE 10.37, Pearson r 0.788, tier accuracy 0.75 — against a 3-expert
reliability band whose mean inter-expert spread is 8.6 points.

Both results support the same positioning: these models are **callable tools and
upper-bound reference points**, and the gap between pixel-level extraction and
named-segment clinical reasoning is the work left to the agent.

---

## Evaluation rubric

The rubric is **multi-dimensional** — we do not reduce to a single scalar. For the DSA diagnostic report task, the dimensions are:

1. **Stenosis accuracy** — continuous MAE + tier classification (<50 / 50–69 / 70–99 / 100), with ±10% clinical tolerance
2. **Segment coverage** — recall over all coronary segments, giving credit for explicit "no significant stenosis" findings
3. **Naming accuracy** — correct segment identification (SYNTAX 26-segment model)
4. **Dominance** — correct determination of right/left/co-dominant coronary anatomy
5. **Reasoning traceability** — does the agent link each conclusion to specific DICOM frames and tool outputs?
6. **Tool orchestration** — appropriate use of segmentation/quantification tools when needed
7. **Anti-hallucination** — does the agent avoid fabricating findings or inventing unsupported measurements?

Each dimension is scored independently by objective metrics (where gold standard exists) or LLM judge (for trace quality). The judge extracts structured facts from prose reports using heuristic regex (offline mock) or LLM (real runs), then applies tolerance-based comparison rather than exact match.

For other tasks (segmentation, stenosis detection, SYNTAX scoring), standard computer vision metrics apply (F1, Dice, MAE).

---

## Extending the pipeline

The pattern is always **implement → register → configure → test**:

1. **New metric**: write a function in `evaluation/metrics/<category>.py`, decorate with `@register_metric("metric_name")`, reference from rubric YAML `metric: metric_name`
2. **New agent**: write a wrapper in `algorithms/baselines/<name>.py` or a full Docker image, register in `benchmark.toml` under `[agents.<name>]`, run with `--agent <name>`
3. **New task**: add cases under `data/tasks/<task_name>/cases/case_*/`, define schema in `tasks/task_template.yaml`, register in `benchmark.toml` under `[tasks]`
4. **New judge backend**: implement in `pipeline/judge_backends.py`, configure with `judge.backend = <name>`

See `docs/PIPELINE_API.md` for complete extension documentation.

---

## Testing

```bash
# Full suite — runs in about a second
python -m pytest tests/ -q

# Specific test modules
python -m pytest tests/test_pipeline_smoke.py -v      # pipeline integration
python -m pytest tests/test_report_facts.py -v        # DSA report scoring
python -m pytest tests/test_instance_metrics.py -v    # instance / pixel metrics
python -m pytest tests/test_method_toml_coverage.py -v  # every methods/*.toml is complete

# Single test
python -m pytest tests/test_report_facts.py::test_perfect_report_scores_full_marks -v
```

The harness tests run **fully offline** with mock backends — no API keys, Docker, or real
data. Under `/opt/anaconda3/bin/python` six tests skip because torch is absent (one
Mamba2 test plus the `monai_unet` / `nnunet` / `sam_med3d` runner-dependency checks);
`$BENCH_PY -m pytest tests/ -q` exercises those too.

---

## Baseline agents (EchoAgent-style evaluation)

Following **EchoAgent (Wang et al. 2026)**, we evaluate in three tiers:

**Tier 1: Task-specific specialist models (upper bound)**
- SAM-VMNet (vessel segmentation)
- DeepCORO-CLIP (stenosis detection)
- SYNTAX Calculator (rule-based scoring)

**Tier 2: General-purpose multimodal LLMs (fair comparison)**
- GPT-4o (OpenAI)
- Claude 3 Opus (Anthropic)
- Gemini 1.5 Pro (optional)

**Tier 3: Our method**
- Cardiomni (full agent with 4-stage SOP + tool orchestration)

Specialist models serve as **tools** that any agent can call (via `algorithms/toolkit.py`) and as **performance upper bounds** for reference, not as direct competitors. The core comparison axis is **harness design** (Tier 2 vs. Tier 3), keeping the base model fixed.

Baseline agent wrappers are registered in `benchmark.toml` as `gpt4v`, `claude`, `deepcoro`, `sam_vmnet`, etc. Currently they return mock predictions; connecting to real inference (API calls or local weights) is open work for generating EchoAgent-style comparison tables.

---

## Paper integration

CardiomniBench-VD is the companion evaluation suite to the **Cardiomni** agent, submitted to AAAI 2027. The paper's positioning:

- **Core contribution**: the Cardiomni agent harness itself (encoding expert SOP as explicit reasoning stages), not the benchmark
- **Single modality**: DSA (coronary angiography) only. CTA, cross-modality fusion, FFR, and full CAD-RADS grading are future work and not current claims
- **Benchmark role**: deliberately de-emphasized — appears only in abstract/intro/experiments, never in Method section
- **Evaluation claim**: small public-data suite demonstrating the SWE-bench evaluation paradigm; private expert-annotated set mentioned as "in preparation"
- **Specialist models**: cited as tools and upper-bound references, not as competitors

The benchmark provides the infrastructure for controlled agent comparison but is not positioned as a novel dataset contribution.

---

## Environment

Developed and tested on H20 server (8× NVIDIA H20, Alibaba Cloud). GPU is shared — pin an idle card (`nvidia-smi`; avoid device 0). Docker gray-box path (GPU + mounts + scoring, no agent code) is verified via `configs/smoke_docker.yaml`.

`source env.sh` is the single entrypoint for GPU work: it exports `$BENCH_PY` (the uv-managed `.venv` interpreter) and redirects `UV_CACHE_DIR` / `PIP_CACHE_DIR` / `HF_HOME` / `TORCH_HOME` onto the NAS. This is a hard constraint, not a preference — the root filesystem is 99GB and cannot hold the torch + CUDA wheel set (~15GB) alongside 92GB of model weights. Filling it once stalled a `uv sync` for 105 minutes without an error.

Offline pipeline work (tests, `--help`, mock backends) needs no torch and runs under `/opt/anaconda3/bin/python` (3.13.9).

---

## License

[To be determined — likely Apache 2.0 for code, CC-BY for cases]

## Citation

```
[AAAI 2027 citation pending]
```
