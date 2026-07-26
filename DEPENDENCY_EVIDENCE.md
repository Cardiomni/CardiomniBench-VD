# Dependency Evidence Trail

**Generated**: 2025-07-25  
**Environment**: gkp-gsa (conda, Python 3.10.20)  
**Method**: AST scan of first-party code + version check via importlib

This document records the evidence for every dependency in `pyproject.toml`.

---

## Core Dependencies (24 packages)

### Deep Learning Framework

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **torch** | 2.11.0+cu128 | 10 files | baselines, runners (cardiosyntax, cca_unet, monai_unet, nnunet, sam_med3d) |
| **torchvision** | 0.26.0+cu128 | 1 file | `baselines/cardiosyntax_r3d_agent.py` (R3D video model) |

**Files importing torch**:
- `algorithms/baselines/_diag_cca_norm_scope.py`
- `algorithms/baselines/_diag_cca_window.py`
- `algorithms/baselines/cardiosyntax_r3d_agent.py`
- `algorithms/baselines/cca_unet_agent.py`
- `algorithms/baselines/run_cardiosyntax_batch.py`
- `benchmark/runners/cardiosyntax_r3d_runner.py`
- `benchmark/runners/monai_unet_runner.py`
- `benchmark/runners/nnunet_runner.py`
- `benchmark/runners/sam_med3d_runner.py`
- `benchmark/runners/vlm_runner.py`

### Medical Imaging

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **monai** | 1.6.0 | 5 files | MONAI UNet runner, nnU-Net runner, CCA baselines |
| **nibabel** | 5.4.2 | 11 files | NIfTI I/O (benchmark.io_spec, baselines, runners, scripts) |
| **pydicom** | 3.0.2 | 2 files | `algorithms/toolkit.py`, `scripts/parse_dsa_metadata.py` |
| **SimpleITK** | 2.5.5 | — | Transitive via monai (ITK image resampling); not directly imported in first-party code but present in gkp-gsa |

**Files importing monai**:
- `algorithms/baselines/_diag_cca_norm_scope.py`
- `algorithms/baselines/_diag_cca_window.py`
- `algorithms/baselines/cca_unet_agent.py`
- `benchmark/runners/monai_unet_runner.py`
- `benchmark/runners/nnunet_runner.py`

**Files importing nibabel**:
- `algorithms/baselines/_diag_cca_geometry.py`
- `algorithms/baselines/_diag_cca_norm_scope.py`
- `algorithms/baselines/_diag_cca_window.py`
- `algorithms/baselines/cca_unet_agent.py`
- `benchmark/io_spec.py` (core I/O)
- `benchmark/run_all.py`
- `scripts/gen_cca_cases.py`
- ... (5 more files)

### Numerical/Scientific

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **numpy** | 2.2.6 | 21 files | Ubiquitous (arrays, metrics, runners, baselines) |
| **scipy** | 1.15.3 | 3 files | `evaluation/metrics/segmentation_metrics.py`, monai/nnunet runners |
| **scikit-image** | 0.25.2 (skimage) | 1 file | `benchmark/runners/monai_unet_runner.py` |
| **scikit-learn** | 1.7.2 (sklearn) | — | Not imported in first-party code but present in gkp-gsa; vendored models use it |
| **pandas** | 2.3.3 | — | Not imported in first-party code but present in gkp-gsa; scripts may use it |

**Files importing numpy** (sample):
- `algorithms/baselines/_diag_cca_geometry.py`
- `algorithms/baselines/cardiosyntax_r3d_agent.py`
- `algorithms/baselines/cca_unet_agent.py`
- `benchmark/io_spec.py`
- `evaluation/metrics/segmentation_metrics.py`
- ... (16 more files)

**Files importing scipy**:
- `benchmark/runners/monai_unet_runner.py`
- `benchmark/runners/nnunet_runner.py`
- `evaluation/metrics/segmentation_metrics.py`

### Vision Models

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **timm** | 1.0.22 | — | Present in gkp-gsa; vendored specialist models (CM-UNet, Spark) use it |
| **einops** | 0.8.2 | — | Present in gkp-gsa; likely for tensor ops in vision models |

### Visualization

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **matplotlib** | 3.10.9 | — | Present in gkp-gsa; standard for plotting |
| **Pillow** | 12.3.0 (PIL) | 1 file | `benchmark/io_spec.py` (frames_to_pil) |
| **tqdm** | 4.68.3 | — | Present in gkp-gsa; progress bars in runners |

**Files importing PIL**:
- `benchmark/io_spec.py`

### Configuration/Serialization

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **pyyaml** | 6.0.3 (yaml) | 24 files | YAML configs (pipeline, baselines, scripts) |
| **tomli** | 2.4.1 | 2 files | `benchmark/method_config.py`, `pipeline/registry.py` (Python <3.11 fallback) |

**Files importing yaml** (sample):
- `algorithms/baselines/_diag_cca_geometry.py`
- `algorithms/baselines/cardiosyntax_r3d_agent.py`
- `algorithms/baselines/cca_unet_agent.py`
- `pipeline/config.py`
- `pipeline/orchestrator.py`
- `scripts/convert_arcade.py`
- ... (18 more files)

**Files importing tomli**:
- `benchmark/method_config.py`
- `pipeline/registry.py`

### nnU-Net Dependencies

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **nnunetv2** | 2.8.1 (no `__version__`) | 1 runner | `benchmark/runners/nnunet_runner.py` |
| **acvl-utils** | 0.2.6 | — | nnU-Net utility library (transitive) |
| **batchgenerators** | 0.25.3 | — | nnU-Net data augmentation (transitive) |
| **connected-components-3d** | 4.0.0 | — | CC3D for postprocessing (transitive) |
| **graphviz** | 0.21 | — | nnU-Net planning visualization (transitive) |

**Files importing nnunetv2**: Not directly imported; runner invokes CLI via subprocess.

### Testing

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **pytest** | 9.1.1 | 5 files | All test files |

**Files importing pytest**:
- `tests/test_mamba2_torch.py`
- `tests/test_method_config.py`
- `tests/test_method_toml_coverage.py`
- `tests/test_pipeline_smoke.py`
- `tests/test_report_facts.py`

**Note**: pytest in core dependencies is unusual (typically a dev tool) but matches `requirements.txt`.

---

## Optional Dependencies

### [llm] — LLM API Clients (2 packages)

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **anthropic** | MISSING | 2 files | `algorithms/baselines/claude_agent.py`, `pipeline/judge_backends.py` |
| **openai** | 2.44.0 | 2 files | `algorithms/baselines/gpt4v_agent.py`, `algorithms/baselines/pure_llm/pure_llm.py` |

**Files importing anthropic**:
- `algorithms/baselines/claude_agent.py`
- `pipeline/judge_backends.py`

**Files importing openai**:
- `algorithms/baselines/gpt4v_agent.py`
- `algorithms/baselines/pure_llm/pure_llm.py`

### [vlm] — Vision-Language Models (4 packages)

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **transformers** | MISSING | 1 file | `benchmark/runners/vlm_runner.py` (AutoProcessor, AutoModelForImageTextToText) |
| **accelerate** | MISSING | — | HF model loading optimization (inferred) |
| **huggingface-hub** | 1.24.0 | 3 files | `algorithms/baselines/download_vlm_*.py`, `scripts/download_weights_direct.py` |
| **safetensors** | 0.8.0 | — | HF checkpoint format (transitive) |

**Files importing transformers**:
- `benchmark/runners/vlm_runner.py` (line 75: `transformers.AutoProcessor.from_pretrained`)

**Files importing huggingface_hub**:
- `algorithms/baselines/download_vlm_parallel.py`
- `algorithms/baselines/download_vlm_weights.py`
- `scripts/download_weights_direct.py`

**Note**: transformers floor of `>=4.51.0` is UNVERIFIED — inferred from `benchmark/vlms.py` listing Qwen3-VL-8B (newest model). Confirm on first `uv sync --extra vlm`.

### [lightning] — PyTorch Lightning (3 packages)

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **pytorch-lightning** | 2.6.5 | — | Vendored specialist models (CM-UNet MoCo pretraining) |
| **torchmetrics** | 1.9.0 | — | Lightning metrics (transitive) |
| **tensorboardX** | 2.6.5 | — | TensorBoard logging (transitive) |

**Note**: Not imported in first-party code; vendored models in `algorithms/specialist_models/cm_unet/Pretraining/MoCo/` use them.

### [specialist] — Specialist Model Toolkits (1 package + 4 commented)

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **opencv-python-headless** | MISSING (cv2) | — | Vendored models use cv2 for image ops |
| ~~ultralytics~~ | MISSING | 1 file (commented) | `algorithms/toolkit.py` (YOLO models; HF blocked per docs) |
| ~~wandb~~ | MISSING | — | Experiment tracking (vendored DeepCORO-CLIP) |
| ~~nltk~~ | MISSING | — | NLP for DeepCORO-CLIP report parsing |
| ~~rouge-score~~ | MISSING | — | Text metrics for DeepCORO-CLIP |

**Note**: Commented packages are MISSING in gkp-gsa and have known issues (HF blocked, manual setup required).

### [dev] — Development Tools (3 packages)

| Package | Version (gkp-gsa) | Evidence | Usage |
|---------|-------------------|----------|-------|
| **pytest-cov** | MISSING | — | Coverage reporting (dev convenience) |
| **ruff** | MISSING | — | Linting + formatting (dev convenience) |
| **mypy** | MISSING | — | Type checking (dev convenience) |

**Note**: Not present in gkp-gsa; added for development workflow.

---

## Vendored Dependencies (NOT in pyproject.toml)

The following packages are used by vendored specialist models under `algorithms/specialist_models/` but are NOT declared in `pyproject.toml` because those repos manage their own dependencies:

| Package | Vendored Repo | Purpose |
|---------|---------------|---------|
| **mmcv** | CM-UNet, ARCADE-stenosis | OpenMMLab computer vision |
| **mmengine** | CM-UNet | OpenMMLab engine |
| **pytorchvideo** | CardioSYNTAX | Video model datasets |
| **seaborn** | CM-UNet, DeepCORO-CLIP | Statistical plotting |
| **segment_anything** (local) | sam_med3d_src/ | SAM3 3D segmentation (NOT PyPI package) |
| **selective_scan** | SAM-VMNet | Mamba SSM |
| **thop** | SAM-VMNet | FLOPs counting |
| **albumentations** | FRNet | Data augmentation |
| **yacs** | FRNet, StenUNet | YACS config system |

**Installation**: These are either:
1. **Bundled in the vendored repo** (`algorithms/specialist_models/*/` with own setup.py)
2. **Injected into sys.path** (sam_med3d_src/)
3. **Require manual setup** per `MODEL_INVENTORY.md` and `WEIGHTS_INVENTORY.md`

Do NOT add these to `pyproject.toml` — it would conflict with the vendored copies.

---

## Version Source: gkp-gsa Freeze

The following command was used to extract versions:

```bash
/opt/anaconda3/envs/gkp-gsa/bin/python -c "
import importlib
packs = ['torch', 'torchvision', 'monai', 'nibabel', 'numpy', 'scipy', 
         'skimage', 'pydicom', 'einops', 'transformers', 'yaml', 'pandas', 
         'PIL', 'matplotlib', 'tqdm', 'sklearn', 'timm', 'pytest', 'tomli', 
         'anthropic', 'openai', 'pytorch_lightning', 'tensorboardX', 
         'torchmetrics', 'huggingface_hub', 'safetensors', 'SimpleITK', 
         'nnunetv2', 'acvl_utils', 'batchgenerators', 'connected_components_3d', 
         'graphviz']
for m in packs:
    try:
        mod = importlib.import_module(m)
        print(f'{m}: {getattr(mod, \"__version__\", \"?\")}')
    except Exception:
        print(f'{m}: MISSING')
"
```

**Output** (2025-07-25):
```
torch: 2.11.0+cu128
torchvision: 0.26.0+cu128
monai: 1.6.0
nibabel: 5.4.2
numpy: 2.2.6
scipy: 1.15.3
skimage: 0.25.2
pydicom: 3.0.2
einops: 0.8.2
transformers: MISSING
yaml: 6.0.3
pandas: 2.3.3
PIL: 12.3.0
matplotlib: 3.10.9
tqdm: 4.68.3
sklearn: 1.7.2
timm: 1.0.22
pytest: 9.1.1
tomli: 2.4.1
anthropic: MISSING
openai: 2.44.0
pytorch_lightning: 2.6.5
tensorboardX: 2.6.5
torchmetrics: 1.9.0
huggingface_hub: 1.24.0
safetensors: 0.8.0
SimpleITK: 2.5.5
nnunetv2: ?
acvl_utils: 0.2.6
batchgenerators: 0.25.3
connected_components_3d: 4.0.0
graphviz: 0.21
```

---

## AST Scan Methodology

To avoid false positives from vendored code, the scan was restricted to first-party directories:

```bash
roots=['benchmark', 'pipeline', 'evaluation', 'tests', 'scripts', 
       'algorithms/baselines', 'algorithms/toolkit.py', 
       'algorithms/base.py', 'algorithms/test_toolkit_complete.py', 
       'conftest.py', 'docker']
```

Excluded:
- `algorithms/specialist_models/` (9.6 GB of vendored repos)
- `runs/` (pipeline output)
- `data/` (datasets)
- `github_repos/` (cloned external repos)

This ensures `pyproject.toml` declares only what the harness and baselines need, not the full transitive closure of vendored dependencies.

---

## Changelog

| Date | Change |
|------|--------|
| 2025-07-25 | Initial evidence collection from gkp-gsa (Python 3.10.20) |

---

**Maintainer**: Cardiomni Team  
**See Also**: `pyproject.toml`, `ENV_MIGRATION.md`
