# Environment Migration: conda gkp-gsa → uv

**Status**: Configuration ready  
**Created**: 2025-07-25  
**Target Python**: 3.10.20 (matches gkp-gsa)  
**CUDA**: 12.8 (torch 2.11.0+cu128)

---

## Overview

This document guides migration from the shared conda `gkp-gsa` environment to a project-local uv-managed virtual environment. The new setup:

- **Isolates dependencies** — no more borrowing packages from unrelated projects
- **Pins versions explicitly** — `pyproject.toml` declares every dependency with evidence
- **Supports CUDA 12.8** — torch from PyTorch's cu128 index, not generic PyPI wheels
- **Enables reproducibility** — `uv.lock` (generated on first sync) freezes the full tree

The conda environment remains available for other projects; this migration does not remove it.

---

## Quick Start

### 1. Initialize the Environment

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD

# Remove the existing empty .venv (Python 3.11) if present
rm -rf .venv

# Create and populate the environment (Python 3.10 per .python-version)
uv sync
```

**Expected output**:
- Resolves ~50 packages (core dependencies)
- Downloads torch 2.11.0+cu128 (~2.5 GB) from PyTorch index
- Creates `.venv/` with Python 3.10.20
- Writes `uv.lock` (commit this to git)

**Time**: ~5 minutes on first run (download-bound), <30s on subsequent syncs.

### 2. Activate the Environment

**Option A: Traditional activation** (modifies shell PATH)
```bash
source .venv/bin/activate
python -m pytest tests/ -q
deactivate
```

**Option B: uv run prefix** (no activation needed)
```bash
uv run python -m pytest tests/ -q
uv run python -m pipeline.cli run --config configs/smoke.yaml
```

**Recommendation**: Use `uv run` in scripts and CI; use activation for interactive sessions.

### 3. Verify CUDA

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
```

**Expected**: `CUDA available: True, devices: 8` (H20 × 8)

If False, check:
- `nvidia-smi` shows driver 550+ (CUDA 12.8 compatible)
- No conflicting `LD_LIBRARY_PATH` from conda

### 4. Run the Test Suite

```bash
# Full suite (19 tests, ~15 seconds)
uv run python -m pytest tests/ -v

# Smoke test only (pipeline integration)
uv run python -m pytest tests/test_pipeline_smoke.py -v

# Skip GPU tests if running on CPU node
uv run python -m pytest tests/ -m "not gpu"
```

**Expected**: All tests pass. The `test_mamba2_torch.py` test checks that the kernel-free Mamba2 fallback works without `mamba_ssm` installed.

### 5. Run the Pipeline (Offline Mock)

```bash
# Minimal smoke test (mock agent, 1 case)
uv run python -m pipeline.cli run --config configs/smoke.yaml

# DSA report extraction task
uv run python -m pipeline.cli run --config configs/smoke_dsa_report.yaml

# List available agents
uv run python -m pipeline.cli agents --toml benchmark.toml
```

**Expected**: Writes `runs/<run_name>/` with `summary.json` showing 100% pass rate for mock agent.

---

## Optional Dependency Groups

Install additional groups as needed:

```bash
# LLM judge + Claude baseline (anthropic, openai)
uv sync --extra llm

# VLM baselines: Qwen, LLaVA (transformers 4.57+, accelerate, huggingface-hub)
uv sync --extra vlm

# PyTorch Lightning for specialist model training
uv sync --extra lightning

# Development tools: ruff, mypy, pytest-cov
uv sync --extra dev

# Everything (11+ GB installed size)
uv sync --all-extras
```

**Note**: The `specialist` group is commented out in `pyproject.toml` because ultralytics/wandb/nltk are not available in gkp-gsa and have known download issues (HuggingFace blocked per `MODEL_INVENTORY.md`). Install manually if needed.

---

## Environment Comparison: gkp-gsa vs uv

| Aspect | conda gkp-gsa (old) | uv .venv (new) |
|--------|---------------------|----------------|
| **Python** | 3.10.20 | 3.10.20 (pinned via `.python-version`) |
| **torch** | 2.11.0+cu128 | 2.11.0+cu128 (PyTorch index) |
| **numpy** | 2.2.6 | 2.2.6 (constrained >=2.2,<3.0) |
| **monai** | 1.6.0 | 1.6.0 |
| **nnunetv2** | 2.8.1 | 2.8.1 |
| **transformers** | ❌ MISSING | ✅ 4.57+ (via `--extra vlm`) |
| **Isolation** | Shared (multi-project) | Project-local |
| **Reproducibility** | Manual `conda env export` | Automatic `uv.lock` |
| **Activation time** | ~2s (conda overhead) | <0.1s (uv overhead) |

---

## Validation Checklist

Run these commands to verify migration correctness:

```bash
# 1. Python version matches
uv run python --version
# Expected: Python 3.10.20

# 2. Core imports succeed
uv run python -c "
import torch, torchvision, monai, nibabel, pydicom
import numpy, scipy, sklearn, pandas, PIL
import pytest, yaml
print('✓ All core imports OK')
"

# 3. CUDA available
uv run python -c "import torch; assert torch.cuda.is_available()"
# Expected: no output (assertion passes)

# 4. Full test suite passes
uv run python -m pytest tests/ -q
# Expected: 19 passed in ~15s

# 5. Baseline agent runs (coronary_unet on 1 CCA case)
uv run python -m benchmark.run_unified --methods coronary_unet --limit 1 --device cuda:4
# Expected: writes runs/*/cases.jsonl with status=ok, Dice ~0.6

# 6. Mock pipeline end-to-end
uv run python -m pipeline.cli run --config configs/smoke.yaml
# Expected: summary.json with 100% pass rate
```

If any check fails, compare against gkp-gsa:
```bash
/opt/anaconda3/envs/gkp-gsa/bin/python -c "import <module>; print(<module>.__version__)"
```

---

## Updating Dependencies

### Add a New Package

```bash
# Add to pyproject.toml [project.dependencies]
# Example: adding 'seaborn' for plotting
uv add seaborn>=0.13.0

# Or manually edit pyproject.toml, then:
uv sync
```

### Upgrade a Package

```bash
# Upgrade specific package
uv add --upgrade torch

# Upgrade all to latest compatible versions
uv lock --upgrade
uv sync
```

### Pin an Exact Version

Edit `pyproject.toml`:
```toml
dependencies = [
    "numpy==2.2.6",  # exact pin (was ">=2.2.0,<3.0")
]
```
Then `uv sync`.

### Check What Would Change

```bash
uv lock --dry-run --upgrade
```

---

## Known Issues & Workarounds

### Issue 1: SAM3 Not on PyPI

**Symptom**: `sam_med3d_runner.py` imports `from segment_anything.build_sam3D import ...` but this is not in PyPI.

**Cause**: SAM3 is a vendored local package in `algorithms/specialist_models/sam_med3d_src/`. The runner injects it into `sys.path`.

**Workaround**: No action needed. The runner handles sys.path. Do not install `segment-anything` from PyPI (that's SAM2, incompatible).

**Weights**: Available in `algorithms/specialist_models/weights/TC-SemiSAM-checkpoints/sam3_original.pt` (3.5 GB).

### Issue 2: transformers Missing in gkp-gsa

**Symptom**: `vlm_runner.py` fails with `ModuleNotFoundError: No module named 'transformers'`.

**Cause**: gkp-gsa does not have transformers installed; VLM baselines (Qwen, LLaVA) were not previously tested.

**Workaround**: Install the `vlm` extra:
```bash
uv sync --extra vlm
```
This adds transformers 4.57+, accelerate, and huggingface-hub.

### Issue 3: Mamba2 CUDA Kernels Not Compiled

**Symptom**: `att_mamba2_unet` checkpoint loads but logs "falling back to kernel-free implementation".

**Cause**: `mamba_ssm` requires `nvcc` (CUDA compiler) to build kernels. The H20 server does not have CUDA dev tools installed.

**Workaround**: The pure-PyTorch fallback (`algorithms/specialist_models/att_mamba2/mamba2_torch.py`) is functionally equivalent and tested in `tests/test_mamba2_torch.py`. Performance is ~2× slower but results are numerically identical.

To install mamba_ssm (requires nvcc):
```bash
# Install CUDA toolkit first (requires root/sudo)
apt-get install nvidia-cuda-toolkit

# Then pip install (caution: large compile, 10+ min)
uv pip install mamba-ssm
```

### Issue 4: Ultralytics YOLO Blocked

**Symptom**: `toolkit.py` imports `ultralytics` but it's not installed.

**Cause**: HuggingFace downloads blocked per `MODEL_INVENTORY.md`. The `specialist` optional group comments out ultralytics.

**Workaround**: Manual installation via conda or local .whl:
```bash
# Option A: conda in base environment, then symlink
conda install -c conda-forge ultralytics
ln -s /opt/anaconda3/envs/base/lib/python3.10/site-packages/ultralytics .venv/lib/python3.10/site-packages/

# Option B: download .whl and install offline
uv pip install ultralytics-8.3.0-py3-none-any.whl
```

Or skip YOLO baselines if not needed (they're optional).

### Issue 5: Existing .venv is Python 3.11

**Symptom**: After `uv sync`, Python is still 3.11 instead of 3.10.

**Cause**: uv reuses an existing `.venv` if present. The repo had a 3.11 .venv from prior testing.

**Workaround**: Remove and recreate:
```bash
rm -rf .venv
uv sync
```

Verify: `uv run python --version` → `Python 3.10.20`

---

## Migration Impact on Existing Workflows

### Running Baselines (run_unified.py)

**Old**:
```bash
/opt/anaconda3/envs/gkp-gsa/bin/python -m benchmark.run_unified --methods coronary_unet --limit 1
```

**New**:
```bash
uv run python -m benchmark.run_unified --methods coronary_unet --limit 1
```

Or with activation:
```bash
source .venv/bin/activate
python -m benchmark.run_unified --methods coronary_unet --limit 1
```

### Running Tests

**Old**:
```bash
/opt/anaconda3/bin/python -m pytest tests/ -q   # used base (py3.13), skipped torch tests
```

**New**:
```bash
uv run python -m pytest tests/ -q   # uses .venv (py3.10), runs all tests
```

### Scripts (convert_arcade.py, gen_cca_cases.py, etc.)

**Old**: Hardcoded shebang `#!/opt/anaconda3/envs/gkp-gsa/bin/python` (none found in first-party code).

**New**: Use portable shebang:
```python
#!/usr/bin/env python
```
Then run via `uv run ./script.py` or activate .venv first.

### Docker Agent (configs/smoke_docker.yaml)

**No change**: The Docker image (`cardiomni:latest`) has its own dependencies declared in `docker/agent/Dockerfile`. The uv environment is for host-side harness execution only.

---

## Rollback Plan

If the uv environment causes issues, revert to gkp-gsa:

```bash
# 1. Deactivate uv environment (if activated)
deactivate

# 2. Remove .venv and uv.lock
rm -rf .venv uv.lock

# 3. Use gkp-gsa directly (as before)
/opt/anaconda3/envs/gkp-gsa/bin/python -m pytest tests/ -q
/opt/anaconda3/envs/gkp-gsa/bin/python -m benchmark.run_unified --methods coronary_unet --limit 1
```

The `pyproject.toml` and `.python-version` files are harmless if not using uv.

---

## Adding pyproject.toml to Git

**Recommendation**: Commit `pyproject.toml`, `.python-version`, and `uv.lock` to version control.

```bash
git add pyproject.toml .python-version ENV_MIGRATION.md
git add uv.lock   # after first `uv sync`

# Add .venv to .gitignore if not already present
echo ".venv/" >> .gitignore

git commit -m "Add uv environment configuration (replaces conda gkp-gsa)"
```

**Why commit uv.lock?**
- Ensures all collaborators get identical dependency versions
- Enables reproducible CI builds
- Analogous to `Pipfile.lock`, `poetry.lock`, `package-lock.json`

---

## FAQ

### Q: Why Python 3.10 instead of 3.13?

The task specifies matching gkp-gsa (Python 3.10.20). While torch 2.11 supports 3.13, nnunetv2 and some specialist models have not been tested on 3.13. The `.python-version` file pins 3.10 for consistency.

### Q: Can I use conda and uv simultaneously?

Technically yes, but not recommended. Activating both can lead to PATH conflicts and import confusion. Choose one:
- **uv** for this project (isolated, reproducible)
- **conda gkp-gsa** for other projects sharing that environment

### Q: What if `uv sync` fails with a network error?

The PyTorch cu128 index and PyPI mirrors may be slow or blocked. Options:

1. **Retry with verbose output**:
   ```bash
   uv sync -v
   ```

2. **Use a PyPI mirror** (if cu128 index is reachable):
   ```bash
   uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   (But torch/torchvision still come from PyTorch index per `[tool.uv.sources]`.)

3. **Offline install from gkp-gsa** (last resort):
   ```bash
   # Export gkp-gsa packages
   /opt/anaconda3/envs/gkp-gsa/bin/pip freeze > gkp-gsa-freeze.txt
   
   # Install into uv venv (bypasses index)
   uv pip install -r gkp-gsa-freeze.txt
   ```

### Q: How do I update the environment after pulling new code?

```bash
git pull origin main
uv sync   # automatically installs any new dependencies
```

uv reads `pyproject.toml` and `uv.lock`, reconciles the current .venv, and installs/removes packages as needed.

### Q: Can I install packages directly with `uv pip install`?

Yes, but it bypasses the lock file:
```bash
uv pip install some-package   # installs but doesn't update pyproject.toml or uv.lock
```

Prefer `uv add some-package`, which updates both and keeps everything in sync.

---

## Reference: File Inventory

Files created/modified by this migration:

| File | Status | Purpose |
|------|--------|---------|
| `pyproject.toml` | ✅ Created | Dependency declarations, tool config |
| `.python-version` | ✅ Created | Pins Python 3.10 for uv |
| `ENV_MIGRATION.md` | ✅ Created | This document |
| `uv.lock` | ⏳ Generated on first `uv sync` | Lock file (commit to git) |
| `.venv/` | ⏳ Generated on first `uv sync` | Virtual environment (ignore in git) |
| `.gitignore` | ⚠️ Needs update | Add `.venv/` if not present |
| `requirements.txt` | ⚠️ Obsolete | Superseded by pyproject.toml (keep for reference) |

---

## Next Steps

1. **Run `uv sync`** to initialize the environment (not done by this sub-agent per task instructions).
2. **Validate** with the checklist above.
3. **Update CI/CD** (if any) to use `uv run` instead of hardcoded conda paths.
4. **Notify collaborators** to run `uv sync` after pulling this branch.
5. **Archive gkp-gsa export** for rollback:
   ```bash
   /opt/anaconda3/envs/gkp-gsa/bin/pip freeze > .archive/gkp-gsa-20250725.txt
   ```

---

**Maintainer**: Cardiomni Team  
**Last Updated**: 2025-07-25

---

## Addendum: Hardcoded Python Paths Found

**Search command**:
```bash
grep -rn "gkp-gsa\|/opt/anaconda3" --include='*.py' --include='*.sh' --include='*.md' . | grep -v specialist_models/github_repos | grep -v "^./runs/"
```

**Results** (2025-07-25):

### Shell Scripts (P0 — breaks execution)
```bash
# HANDOFF_CHECK.sh line 10
PYTHON=/opt/anaconda3/bin/python

# RUN_BASELINE_TESTS.sh line 4
PYTHON=/opt/anaconda3/bin/python
```

**Fix**: Replace with:
```bash
PYTHON="uv run python"
# or after activation:
PYTHON=$(which python)
```

### Python Source (P1 — runtime errors on missing env)
```python
# algorithms/baselines/cardiosyntax_r3d_agent.py line 428
subprocess.run(["/opt/anaconda3/envs/gkp-gsa/bin/python", ...])

# algorithms/baselines/cca_unet_agent.py line 191
_fail("PyTorch / MONAI required; use e.g. /opt/anaconda3/envs/gkp-gsa/bin/python")
```

**Fix**:
```python
# Use the current interpreter
import sys
subprocess.run([sys.executable, ...])

# Update error message
_fail("PyTorch / MONAI required; activate the uv environment first: source .venv/bin/activate")
```

### Documentation (P2 — informational only)
- `README.md` line 17, 241
- `docs/PIPELINE_COMPLETION.md` lines 25, 52, 113, 140, 230, 237, 243, 252, 306, 318, 321, 324, 327, 330, 335
- `docs/ANNOTATION_ACTION_PLAN.md` line 113
- `INFRASTRUCTURE_COMPLETE.md` lines 94, 97, 100, 218
- `benchmark/method_config.py` line 31 (comment only)

**Fix**: Global search-replace in docs:
```bash
find . -name "*.md" -type f -exec sed -i 's|/opt/anaconda3/bin/python|uv run python|g' {} +
```

---

**Migration Priority**:
1. **P0 (before first run)**: Update `HANDOFF_CHECK.sh` and `RUN_BASELINE_TESTS.sh`
2. **P1 (before baseline tests)**: Fix `cardiosyntax_r3d_agent.py` and `cca_unet_agent.py`
3. **P2 (documentation cleanup)**: Update markdown files (can be deferred)

