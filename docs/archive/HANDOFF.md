> ⚠️ **定位已更新 (2026-07-22)**：本文件的"定位/主张"部分已作废（旧 CTA-DSA 融合框架）。当前权威规划见 `/mnt/aliyunsb/Cardiomni/PROPOSAL.md`。本文件的**工程实现描述仍然有效**，可继续复用。

---

# CardiomniBench-VD — Server Handoff

**Location:** `/mnt/aliyunsb/CardiomniBench-VD` (H20 8×NVIDIA H20 server, Alibaba Cloud)  
**GitHub:** `https://github.com/Cardiomni/CardiomniBench-VD` (public)  
**Last sync:** 2025-07-22 (commit `37a66c5` — server deployment status added)

---

## Current Status (已完成)

✅ **Repo structure reorganized** — clean handoff layout: `pipeline/` (harness), `benchmark.toml`
   (unified registry), `configs/`, `rubrics/`, `tasks/`, `data/`, `docker/`, `tests/`, `docs/`,
   `references/`. `paper/` excluded (managed in its own Overleaf repo).

✅ **Unified TOML registry** — `benchmark.toml` registers the shared Docker environment, judge,
   task set, and every agent in ONE file (simplified from BiomniBench-DA's one-task.toml-per-task).
   Run agents by name: `python -m pipeline.cli run --toml benchmark.toml --agent <name>`.

✅ **Pipeline fully verified on this server:**
   - Python: 3.13.9 (`/opt/anaconda3/bin/python`)
   - Tests: 19 passed
   - Docker + GPU injection: verified end-to-end (container sees NVIDIA H20)
   - Mock run: works offline without data/keys

✅ **Docker gray-box path tested:**
   ```bash
   /opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml
   cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt   # lists NVIDIA H20
   ```

---

## What's NOT Done (待实现)

❌ **Cardiomni agent code** — `docker/agent/Dockerfile` is a CUDA base placeholder. The actual
   agent logic (DICOM ingestion → multimodal VLM analysis → structured report generation) is
   not implemented yet. This is the core work.

❌ **Clinical data** — `data/cases/` is empty. Awaiting expert annotation per `docs/annotation_protocol.md`
   (paired CTA+DSA DICOM + ground-truth structured reports + per-case rubrics).

❌ **Real LLM judge** — currently `judge.backend=mock` in configs. Switch to `llm` and set
   `ANTHROPIC_API_KEY` once ready for real rubric grading.

---

## Pipeline Implementation Complete (2026-07-22)

✅ **All pipeline components implemented and verified:**
   - Orchestrator, agent runners (mock/local/docker), judge backends (mock/llm/cli)
   - Scoring system with automatic metrics + LLM judge grading
   - 16 registered objective metrics in `evaluation/metrics/`
   - Complete rubric framework (6 dimensions, example with 24 criteria)
   - Judge validation pipeline (`pipeline/judge_validation.py`)
   - Extension API documentation (`docs/PIPELINE_API.md`)
   - **19/19 tests passing** — full offline gray-box coverage

✅ **Docker + GPU path verified:**
   ```bash
   /opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml \
       --agent-image sweb.base.py.x86_64:latest
   # Verified: GPU injection, mounts, prediction.json scoring
   cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt  # NVIDIA H20 detected
   ```

✅ **Unified TOML registry working:**
   ```bash
   /opt/anaconda3/bin/python -m pipeline.cli agents --toml benchmark.toml
   # Lists: cardiomni, local_script, mock, vlm_baseline
   ```

**See `docs/PIPELINE_COMPLETION.md` for full implementation report.**

**Ready for:** Real agent code + clinical data annotation. The pipeline runs end-to-end with
mock backends (no Docker, no API keys, no data required). All four swap axes (agent, base
model, judge, tasks) are implemented and documented.

---

## Quick Commands

```bash
# Sync from GitHub (when git works again — currently timing out on fetch)
git fetch origin main && git reset --hard origin/main

# List registered agents
/opt/anaconda3/bin/python -m pipeline.cli agents --toml benchmark.toml
# Output: cardiomni, local_script, mock, vlm_baseline

# Run tests
/opt/anaconda3/bin/python -m pytest tests/ -q

# Docker build status (running in background via nohup since ~14:15)
docker images cardiomni:latest
tail -20 /tmp/cardiomni_build.log   # CUDA base layer pulling slowly from DockerHub

# Docker+GPU gray-box check (uses existing sweb.base.py image as substitute)
/opt/anaconda3/bin/python -m pipeline.cli run --config configs/smoke_docker.yaml \
    --agent-image sweb.base.py.x86_64:latest
cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt

# Real run (once agent + data + keys are ready)
export ANTHROPIC_API_KEY=sk-ant-...
/opt/anaconda3/bin/python -m pipeline.cli run --toml benchmark.toml --agent cardiomni
```

---

## Next Steps (优先级顺序)

1. **Implement the Cardiomni agent** (`docker/agent/src/` or similar):
   - DICOM loader (pydicom + windowing for CTA/DSA)
   - Multimodal VLM call (Claude Opus 4.8 or equivalent, with image attachments)
   - Structured report generator (outputs `prediction.json` matching the schema in
     `tests/fixtures/tasks/case_smoke/task_spec.json`)
   - Command-line interface that the pipeline can call via `agent.command` in `benchmark.toml`

2. **Build/verify `cardiomni:latest` image:**
   - Wait for the current build to finish (or rebuild with Aliyun registry mirror if urgent)
   - Test with `configs/smoke_docker.yaml` using the real image

3. **Annotate clinical cases** (coordinate with domain experts):
   - Follow `docs/annotation_protocol.md`
   - Add to `data/cases/case_*/` (DICOM + `task_spec.json` + `rubric.yaml`)
   - Update `data/splits.yaml` with train/val/test splits

4. **Switch to real LLM judge:**
   - Set `ANTHROPIC_API_KEY` in the environment
   - Run with `--judge-backend llm` or edit `benchmark.toml` to `judge.backend = "llm"`

5. **Validate the judge** (optional but recommended, per BiomniBench-DA):
   - Implement `pipeline/judge_validation.py` (multi-judge Cohen's κ vs expert labels)
   - Prove the LLM judge is reliable before using it to grade agents

---

## Notes for Claude Code

- **This repo is a config-driven evaluation harness**, not the agent itself. The agent is a
  black box to the pipeline — it just needs to write `prediction.json` with the right schema.

- **The four swap axes** (换基座/agent/rubric/任务) are all in `benchmark.toml` or the YAML configs.
  You can test with mock backends offline before real data/keys are available.

- **GPU pinning:** On this shared server, avoid CUDA 0 if others are using it. Pin with
  `gpu_device = "device=7"` (or whichever GPU is free — check `nvidia-smi`).

- **Docker build slow?** DockerHub rate-limits. If urgent, configure Aliyun registry mirror in
  `/etc/docker/daemon.json` and restart docker (but this will kill running containers, so
  coordinate with other users first).

- **Paper not in repo:** `paper/` is managed in its own Overleaf git repo and intentionally
  excluded via `.gitignore`. Don't try to add it here.

---

Handoff complete. 剩下的工作（agent 实现 + 数据标注）继续在这台服务器上用 Claude Code 推进。

Generated: 2025-07-22 by Claude Opus 4.8 (1M context)
