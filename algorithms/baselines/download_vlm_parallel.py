#!/usr/bin/env python
"""Parallel HuggingFace model downloader with aggressive concurrency.

Downloads N models simultaneously (default 3), each with M workers (default 8).
Total parallelism = N × M = 24 concurrent file fetches by default, vs the
sequential script's 4. Speeds up multi-model bulk downloads by ~6×.

Env requirements:
    HF_ENDPOINT=https://hf-mirror.com
    HF_HOME=/mnt/aliyunsb/Cardiomni/hf_cache
    HF_HUB_DISABLE_XET=1

Usage:
    python download_vlm_parallel.py [--concurrency N] [--workers M]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# (repo_id, short_name)
REPOS = [
    ("microsoft/llava-med-v1.5-mistral-7b", "llava-med"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "llava-next-1.6"),
    ("llava-hf/llama3-llava-next-8b-hf", "llava-next-llama3"),
    ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava-onevision"),
    ("deepseek-ai/Janus-Pro-7B", "janus-pro"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "r1-distill"),
    ("lingshu-medical-mllm/Lingshu-7B", "lingshu"),
    ("FreedomIntelligence/HuatuoGPT-Vision-7B", "huatuo-vision"),
]


def download_one(repo: str, name: str, workers: int) -> tuple[str, str, str]:
    """Download a single repo. Returns (name, status, detail)."""
    from huggingface_hub import snapshot_download

    try:
        started = time.time()
        path = snapshot_download(repo, max_workers=workers)
        elapsed = time.time() - started
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(path)
            for f in files
        )
        gb = size / 1e9
        return (name, "ok", f"{gb:.2f}GB in {elapsed:.0f}s")
    except Exception as exc:
        return (name, "failed", f"{type(exc).__name__}: {str(exc)[:150]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Models to download in parallel"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Workers per model"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Only download these short names"
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Skip these short names"
    )
    args = parser.parse_args()

    if "HF_ENDPOINT" not in os.environ:
        print("WARN: HF_ENDPOINT not set", file=sys.stderr)

    targets = [
        (repo, name)
        for repo, name in REPOS
        if (args.only is None or name in args.only) and name not in args.skip
    ]
    if not targets:
        print("nothing to do")
        return 0

    print(
        f"Downloading {len(targets)} models with concurrency={args.concurrency}, "
        f"workers={args.workers} (total {args.concurrency * args.workers} threads)\n",
        flush=True,
    )
    started = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(download_one, repo, name, args.workers): name
            for repo, name in targets
        }
        for index, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            short_name, status, detail = future.result()
            flag = "ok " if status == "ok" else "ERR"
            print(f"[{index}/{len(targets)}] [{flag}] {short_name:20} {detail}", flush=True)
            results.append((short_name, status, detail))

    elapsed = time.time() - started
    failures = [r for r in results if r[1] != "ok"]
    print(f"\n{len(results) - len(failures)}/{len(results)} succeeded in {elapsed:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
