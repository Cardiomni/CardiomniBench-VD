# CardiomniBench-VD Pipeline API

**Extension points for customizing the evaluation pipeline**

This document describes how to extend CardiomniBench-VD without modifying core pipeline code. The four main extension points correspond to the four swap axes documented in the main README.

---

## 1. Adding a New Metric (换 rubric metric)

**When:** You need a new objective metric to score a rubric criterion automatically.

**Where:** `pipeline/metric_registry.py` + `evaluation/metrics/`

### Steps

1. **Implement the metric function** in one of the `evaluation/metrics/*.py` files (or create a new module):

```python
# evaluation/metrics/my_metrics.py

def compute_myocardial_perfusion_accuracy(gold_segments, pred_segments):
    """
    Compare predicted perfusion defects against gold standard.
    
    Args:
        gold_segments: list of dicts with 'segment_id' and 'perfusion_defect'
        pred_segments: list of dicts with 'segment_id' and 'perfusion_defect'
    
    Returns:
        float in [0, 1]: accuracy across matched segments
    """
    # Your implementation here
    matches = 0
    total = 0
    for gold_seg in gold_segments:
        seg_id = gold_seg['segment_id']
        pred_seg = next((s for s in pred_segments if s['segment_id'] == seg_id), None)
        if pred_seg:
            if gold_seg.get('perfusion_defect') == pred_seg.get('perfusion_defect'):
                matches += 1
            total += 1
    return matches / total if total > 0 else 0.0
```

2. **Register the metric** in `pipeline/metric_registry.py`:

```python
# Import your new metric
from evaluation.metrics import my_metrics as mm

# Add an adapter to REGISTRY
def _perfusion_accuracy(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_segs = (gold.get("stage1a_cta", {}) or {}).get("segments", []) or []
    pred_segs = (pred.get("cta_findings", {}) or {}).get("segments", []) or []
    if not gold_segs:
        return 0.0
    return mm.compute_myocardial_perfusion_accuracy(gold_segs, pred_segs)

REGISTRY: Dict[str, Adapter] = {
    # ... existing metrics ...
    "perfusion_accuracy": _perfusion_accuracy,
}
```

**The adapter's job:** Extract the relevant fields from the full `gold_standard` and `prediction` dicts, call your metric function, and return a single `float` in `[0, 1]`.

3. **Reference the metric in a rubric**:

```yaml
# rubrics/examples/case_002_rubric.yaml
- criterion_id: "C070"
  description: "Myocardial perfusion defect detection accuracy"
  evaluation_method: "automatic"
  metric: "perfusion_accuracy"  # <-- matches REGISTRY key
  grading_scale:
    type: "continuous"
    grades:
      - grade: "A"
        points: 10
        threshold: {min: 0.85, max: 1.0}
      - grade: "B"
        points: 6
        threshold: {min: 0.70, max: 0.85}
      - grade: "C"
        points: 2
        threshold: {min: 0.0, max: 0.70}
```

4. **Verify:** Run `python -m pipeline.cli metrics` to confirm your metric appears in the list.

---

## 2. Adding a New Agent Backend (换 agent)

**When:** You want to run agents in a new execution environment (e.g., a remote API, a sandbox, etc.).

**Where:** `pipeline/runner.py`

### Steps

1. **Add a new backend case** in `run_agent()`:

```python
# pipeline/runner.py

def run_agent(
    cfg: AgentConfig,
    case_id: str,
    case: Dict[str, Any],
    case_dir: Path,
    output_dir: Path,
    mock_prediction_path: Optional[Path] = None,
) -> AgentResult:
    # ... existing code ...
    
    if cfg.backend == "remote_api":
        return _run_remote_api(cfg, case_id, case_dir, output_dir, spec_path)
    
    # ... existing backends ...
```

2. **Implement the backend function**:

```python
def _run_remote_api(
    cfg: AgentConfig,
    case_id: str,
    case_dir: Path,
    output_dir: Path,
    spec_path: Path,
) -> AgentResult:
    """
    Call a remote agent API, poll for results, download prediction.json.
    """
    import requests
    
    # Upload task_spec.json
    with open(spec_path, "rb") as f:
        resp = requests.post(
            cfg.command,  # treat command as API endpoint
            files={"task_spec": f},
            headers={"Authorization": f"Bearer {cfg.env.get('API_KEY')}"}
        )
    
    if resp.status_code != 200:
        return AgentResult(case_id, None, ok=False, backend="remote_api",
                          error=f"API returned {resp.status_code}")
    
    # Poll for completion, download prediction.json
    # ... your polling logic ...
    
    prediction, load_err = _load_prediction(output_dir / PREDICTION_FILENAME)
    return AgentResult(case_id, prediction, ok=True, backend="remote_api")
```

3. **Configure in benchmark.toml** or a YAML config:

```toml
[agents.my_remote_agent]
backend = "remote_api"
model = "anthropic/claude-opus-4-8"
command = "https://api.example.com/v1/evaluate"  # endpoint URL
[agents.my_remote_agent.env]
API_KEY = "${MY_API_KEY}"
```

4. **Test:** `python -m pipeline.cli run --toml benchmark.toml --agent my_remote_agent`

---

## 3. Adding a New Judge Backend (换 rubric scorer)

**When:** You want to use a different LLM provider, or a custom scoring CLI tool.

**Where:** `pipeline/judge_backends.py`

### Steps

1. **Add a new judge class**:

```python
# pipeline/judge_backends.py

class OpenAIJudge(Judge):
    """Grade via OpenAI chat API."""
    
    def grade(self, prompt: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"judge.backend=openai needs {self.cfg.api_key_env} in env")
        
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("judge.backend=openai requires 'openai' package") from e
        
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        text = resp.choices[0].message.content
        return parse_grade_json(text, valid_grades)
```

2. **Register in the factory**:

```python
def make_judge(cfg: JudgeConfig) -> "Judge":
    if cfg.backend == "mock":
        return MockJudge(cfg)
    if cfg.backend == "llm":
        return LLMJudgeBackend(cfg)
    if cfg.backend == "openai":
        return OpenAIJudge(cfg)
    if cfg.backend == "cli":
        return CLIJudge(cfg)
    raise ValueError(f"unknown judge.backend {cfg.backend!r}")
```

3. **Configure**:

```yaml
# configs/my_run.yaml
judge:
  backend: openai
  model: gpt-4o
  temperature: 0.0
  api_key_env: OPENAI_API_KEY
```

4. **Test:** `export OPENAI_API_KEY=...` and run with `--judge-backend openai`

---

## 4. Adding a New Run Configuration (换任务/dataset)

**When:** You want to test on a different case subset, or change agent/judge/rubric settings.

**Where:** `configs/*.yaml` or `benchmark.toml`

### Option A: Single-run YAML config

Create a new file in `configs/`:

```yaml
# configs/my_experiment.yaml
run_name: my_experiment
seed: 42

tasks:
  source: data/cases
  case_glob: "case_*"
  split: test  # use the 'test' split from data/splits.yaml
  filter:
    fusion_category: "fusion_required"  # only cases requiring CTA-DSA fusion
  limit: 20  # cap at 20 cases

agent:
  backend: docker
  name: cardiomni_v2
  model: anthropic/claude-opus-4-8
  image: cardiomni:v2
  gpu: true
  gpu_device: "device=3"
  command: "python -m cardiomni.run --task-spec /workspace/out/task_spec.json --output /workspace/out/prediction.json --model {model}"

judge:
  backend: llm
  model: claude-opus-4-8
  temperature: 0.0
  api_key_env: ANTHROPIC_API_KEY

rubric:
  dimensions_file: rubrics/rubric_dimensions.yaml
  default_case_rubric: rubrics/examples/case_001_rubric.yaml

output:
  root: runs
  reruns: 3  # run 3 times for mean ± SD
```

Run: `python -m pipeline.cli run --config configs/my_experiment.yaml`

### Option B: Add an agent to `benchmark.toml`

```toml
# benchmark.toml

[agents.my_agent]
backend = "docker"
model = "gpt-4o"
command = "python -m baselines.gpt4o_agent --task-spec /workspace/out/task_spec.json --output /workspace/out/prediction.json"
[agents.my_agent.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
```

Run: `python -m pipeline.cli run --toml benchmark.toml --agent my_agent`

---

## 5. Advanced: Custom Prompt Renderer for Judge

**When:** You want to customize how the judge prompt is constructed.

**Where:** `pipeline/scoring.py`

The default prompt renderer is `default_prompt_renderer()`. To override:

```python
# my_custom_scoring.py
from pipeline.scoring import score_criterion

def my_prompt_renderer(criterion, gold, pred):
    # Build your custom prompt
    return f"Grade this criterion: {criterion['description']}\n\nPrediction: {pred.get('report')}"

# Use it:
result = score_criterion(criterion, gold, pred, judge, prompt_renderer=my_prompt_renderer)
```

Or patch it globally in `orchestrator.py` before calling `score_criterion`.

---

## 6. Validation and Testing

After adding any extension:

1. **Run the test suite:** `python -m pytest tests/ -v`
2. **Smoke test with mock backend:** `python -m pipeline.cli run --config configs/smoke.yaml`
3. **Check your extension is registered:**
   - Metrics: `python -m pipeline.cli metrics`
   - Agents: `python -m pipeline.cli agents --toml benchmark.toml`

---

## Common Patterns

### Defensive metric adapters

Metrics should return a neutral value (0.0 or 1.0) when input is missing, so the pipeline runs end-to-end on incomplete mock data:

```python
def _my_metric(gold: Dict[str, Any], pred: Dict[str, Any]) -> float:
    gold_val = gold.get("field")
    pred_val = pred.get("field")
    if not gold_val or not pred_val:
        return 0.0  # neutral/fail grade
    return compute_accuracy(gold_val, pred_val)
```

### Agent command templates

Use `{placeholder}` tokens in `agent.command`; the runner fills them:

- `{task_dir}` — absolute path to the case directory (read-only in docker)
- `{task_spec}` — absolute path to the generated `task_spec.json`
- `{output_dir}` — absolute path where the agent must write `prediction.json`
- `{model}` — value of `agent.model` (for multi-model agents)
- `{extra_args}` — joined `agent.extra_args`

Docker backend automatically mounts `{task_dir}` at `/workspace/task:ro` and `{output_dir}` at `/workspace/out`.

### Judge output format

Every judge backend must return:

```python
{
    "grade": "A" | "B" | "C" | None,  # categorical grade from the rubric scale
    "reasoning": str,                  # explanation
    "evidence_quotes": [str],          # optional: quotes from the prediction
    # optional error flags:
    "parse_error": bool,
    "grade_out_of_scale": bool,
    "error": str,
}
```

Points are assigned by `scoring.py` from the rubric's `grades[].points` table — the judge never decides points directly.

---

## Summary

| Extension Point | File | Steps |
|---|---|---|
| **New metric** | `pipeline/metric_registry.py` | 1. Implement function in `evaluation/metrics/` <br> 2. Add adapter to `REGISTRY` <br> 3. Reference in rubric YAML |
| **New agent backend** | `pipeline/runner.py` | 1. Add case in `run_agent()` <br> 2. Implement `_run_<backend>()` <br> 3. Configure in `benchmark.toml` or YAML |
| **New judge backend** | `pipeline/judge_backends.py` | 1. Implement `Judge` subclass <br> 2. Register in `make_judge()` <br> 3. Configure `judge.backend` |
| **New run config** | `configs/` or `benchmark.toml` | 1. Create YAML or add `[agents.<name>]` <br> 2. Set tasks/agent/judge/rubric <br> 3. Run with `--config` or `--agent` |

All extensions follow the same pattern: **implement → register → configure → test**. The pipeline never hard-codes agent logic, judge models, or metric implementations — everything is swappable via config.
