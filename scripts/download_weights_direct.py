#!/usr/bin/env python3
"""
Download specific weight files from HuggingFace repos using Python API
Bypasses git-lfs issues with hf-mirror CDN
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

weights_dir = "algorithms/specialist_models/weights"
os.makedirs(weights_dir, exist_ok=True)

downloads = [
    {
        "name": "CM-UNet",
        "repo_id": "Camsouille/CM-UNet",
        "files": ["CM-UNet_weights.pth"],
    },
    {
        "name": "MesserMMP-SYNTAX",
        "repo_id": "MesserMMP/coronary-syntax-prediction",
        "files": [
            "leftBinSyntax_R3D_full_fold00.pt",
            "leftBinSyntax_R3D_full_fold01.pt",
            "leftBinSyntax_R3D_full_fold02.pt",
            "leftBinSyntax_R3D_full_fold03.pt",
            "leftBinSyntax_R3D_full_fold04.pt",
        ],
    },
    {
        "name": "TC-SemiSAM",
        "repo_id": "ly17/TC-SemiSAM-checkpoints",
        "files": [
            "sam_vit_b_01ec64.pth",
            "vmamba_tiny_e292.pth",
        ],
    },
    {
        "name": "SAM3-vessel",
        "repo_id": "ly17/sam3-vessel-segmentation",
        "files": [
            "checkpoint_baseline.pt",
            "checkpoint_dice_optimized.pt",
            "sam3_original.pt",
        ],
    },
]

print("=== Downloading weights via HuggingFace Python API ===\n")

for dl in downloads:
    print(f"[{dl['name']}] {dl['repo_id']}")
    local_dir = os.path.join(weights_dir, dl['name'].replace(' ', '_'))
    os.makedirs(local_dir, exist_ok=True)

    try:
        for filename in dl['files']:
            print(f"  Downloading {filename}...", flush=True)
            local_path = hf_hub_download(
                repo_id=dl['repo_id'],
                filename=filename,
                cache_dir=local_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"    ✓ {filename} ({size_mb:.1f} MB)")
        print(f"  ✓ {dl['name']} complete\n")
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")

print("=== Download complete ===")
