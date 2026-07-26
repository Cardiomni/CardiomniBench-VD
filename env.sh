# Shared environment for every CardiomniBench-VD command.
#
#   source env.sh
#
# Why this file exists
# --------------------
# The root filesystem is a 99GB disk that hit 100% full mid-install and stalled a
# `uv sync` for 105 minutes without failing: uv kept reporting "Downloading ..."
# while writing to a full disk. Every cache that can grow without bound therefore
# lives on the NAS (/mnt/aliyunsb, ~1PB) instead of $HOME.
#
# The NAS is also the only place with room for the torch+CUDA wheel set (~15GB
# unpacked) plus the 92GB of model weights, so .venv itself is on the NAS too.

CARDIOMNI_ROOT=/mnt/aliyunsb/Cardiomni
BENCH_ROOT="$CARDIOMNI_ROOT/CardiomniBench-VD"

# --- package manager caches (NAS, not $HOME) -------------------------------
export UV_CACHE_DIR="$CARDIOMNI_ROOT/.cache/uv"
export UV_HTTP_TIMEOUT=300  # Default 30s insufficient for 132MB nvidia-nvshmem-cu12 over NAS
export PIP_CACHE_DIR="$CARDIOMNI_ROOT/.cache/pip"
# uv hardlinks from cache into .venv by default. Cache and venv are both on the
# same NAS mount, but hardlinks across NFS are unreliable, so copy instead.
export UV_LINK_MODE=copy

# --- model weight caches ---------------------------------------------------
# 92GB of VLM/LLM checkpoints already live here; check_available() in
# benchmark/vlms.py resolves snapshots relative to HF_HOME.
export HF_HOME="$CARDIOMNI_ROOT/hf_cache"
export TORCH_HOME="$CARDIOMNI_ROOT/.cache/torch"
# huggingface.co is unreachable from this host; the mirror serves the same repos.
export HF_ENDPOINT=https://hf-mirror.com
# The mirror's Xet CAS backend returns 401, so the classic resolve path is used.
export HF_HUB_DISABLE_XET=1

# --- the one interpreter ---------------------------------------------------
# All four tasks, all baselines, and the test suite run under this venv. Conda
# envs are not used: gkp-gsa has torch but no transformers, which silently failed
# 180 VLM cases with ModuleNotFoundError before this was standardised.
export BENCH_PY="$BENCH_ROOT/.venv/bin/python"

# --- GPU policy ------------------------------------------------------------
# 8x H20, shared with other users. Device 0 is reserved for others by convention;
# pass --device explicitly and prefer an idle card from `nvidia-smi`.
export CARDIOMNI_DEFAULT_DEVICE=cuda:5

if [ -n "${BASH_VERSION:-}" ] && [ -t 1 ]; then
    printf 'CardiomniBench-VD env: %s\n' "$BENCH_PY"
    printf '  UV_CACHE_DIR=%s\n  HF_HOME=%s\n' "$UV_CACHE_DIR" "$HF_HOME"
fi
