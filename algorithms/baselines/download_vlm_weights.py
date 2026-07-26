#!/usr/bin/env python
"""Sequentially download the VLM baseline weights for CardiomniBench-VD.

Ordered by evaluation priority. Each repo is fetched with snapshot_download, so
re-running the script resumes rather than restarting. Progress markers are
emitted for the Magenta background-shell progress bar.

Env requirements (set by the caller):
    HF_ENDPOINT=https://hf-mirror.com     # huggingface.co is unreachable here
    HF_HOME=/mnt/aliyunsb/Cardiomni/hf_cache   # root fs has only ~14G free
    HF_HUB_DISABLE_XET=1                  # mirror's Xet CAS returns 401

Usage:
    python download_vlm_weights.py [--only NAME ...] [--skip NAME ...]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# (repo_id, short_name, role in the comparison)
REPOS: list[tuple[str, str, str]] = [
    # EchoAgent's medical-specialist MLLM (2023) and its modern successors.
    ("microsoft/llava-med-v1.5-mistral-7b", "llava-med", "medical VLM (EchoAgent original)"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "llava-next-1.6", "LLaVA-NeXT, same Mistral base"),
    ("llava-hf/llama3-llava-next-8b-hf", "llava-next-llama3", "LLaVA-NeXT, Llama3 base"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava-onevision", "LLaVA-OneVision (latest)"),
    # EchoAgent's other general-purpose baselines.
    # deepseek-vl2 intentionally excluded per project decision.
    ("deepseek-ai/Janus-Pro-7B", "janus-pro", "general VLM (EchoAgent original)"),
    (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "r1-distill",
        "reasoning-distilled LLM (EchoAgent original)",
    ),
    # 2025 medical VLMs, newer than LLaVA-Med.
    ("lingshu-medical-mllm/Lingshu-7B", "lingshu", "2025 medical VLM"),
    (
        "FreedomIntelligence/HuatuoGPT-Vision-7B",
        "huatuo-vision",
        "Chinese medical VLM (matches the 中山 report template)",
    ),
]


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="Short names to fetch.")
    parser.add_argument("--skip", nargs="*", default=[], help="Short names to skip.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("WARN: HF_ENDPOINT unset; huggingface.co is unreachable here", file=sys.stderr)

    from huggingface_hub import snapshot_download

    targets = [
        (repo, name, role)
        for repo, name, role in REPOS
        if (args.only is None or name in args.only) and name not in args.skip
    ]
    if not targets:
        print("nothing to do")
        return 0

    print(f"Fetching {len(targets)} repo(s) into {os.environ.get('HF_HOME', '~/.cache')}\n")
    results: list[tuple[str, str, str]] = []
    started = time.time()

    for index, (repo, name, role) in enumerate(targets, 1):
        print(f"[{index}/{len(targets)}] {name}  ({repo})\n    role: {role}", flush=True)
        attempt_started = time.time()
        try:
            path = snapshot_download(repo, max_workers=args.workers)
            size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(path)
                for f in files
            )
            elapsed = time.time() - attempt_started
            print(f"    OK  {human(size)} in {elapsed:.0f}s\n", flush=True)
            results.append((name, "ok", human(size)))
        except Exception as exc:  # noqa: BLE001 - report and continue to the next repo
            print(f"    FAILED  {type(exc).__name__}: {str(exc)[:200]}\n", file=sys.stderr, flush=True)
            results.append((name, "failed", f"{type(exc).__name__}"))
        print(f"@@progress {index / len(targets):.4f}", flush=True)

    print("=" * 60)
    print(f"SUMMARY  ({time.time() - started:.0f}s total)")
    for name, status, detail in results:
        flag = "ok " if status == "ok" else "ERR"
        print(f"  [{flag}] {name:20} {detail}")
    failures = [r for r in results if r[1] != "ok"]
    print(f"\n{len(results) - len(failures)}/{len(results)} succeeded")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
