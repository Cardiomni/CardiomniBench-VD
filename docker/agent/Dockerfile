# Cardiomni agent runtime — GPU-enabled base image.
# Built on the GPU server, NOT this laptop. The pipeline invokes it via
# `docker run --gpus ...` when agent.backend=docker (see configs/example_docker_gpu.yaml).
#
#   docker build -t cardiomni:latest docker/agent
#
# CUDA runtime base so segmentation/vision tools (TotalSegmentator, nnU-Net, etc.)
# and any local LLM can use the GPU. Pin the CUDA tag to match the server driver.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- Python deps -------------------------------------------------------------
# DICOM I/O + compressed transfer syntax support (proposal §7.1: pydicom direct read).
# Add your agent framework + model client here (換 agent / 換基座 happen at run
# time via {model}, so the image only needs the harness, not a specific model).
RUN python3 -m pip install --upgrade pip && python3 -m pip install \
        pydicom \
        pylibjpeg pylibjpeg-libjpeg gdcm \
        numpy \
        pyyaml \
        anthropic

# Place the Cardiomni agent package here when it exists:
# COPY cardiomni/ /workspace/cardiomni/
# RUN python3 -m pip install -e /workspace/cardiomni

# Contract (matches pipeline/runner.py):
#   input : /workspace/task        (case dir, read-only mount)
#           /workspace/out/task_spec.json  (gold-standard-stripped spec)
#   output: /workspace/out/prediction.json (agent must write this)
#
# The actual entrypoint command comes from agent.command in the config, so this
# image stays agent-agnostic. Default is a no-op that documents the contract.
CMD ["bash", "-lc", "echo 'Set agent.command in your pipeline config to run the agent.'"]
