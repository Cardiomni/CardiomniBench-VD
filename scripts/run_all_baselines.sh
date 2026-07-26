#!/usr/bin/env bash
# Run all available baselines on all four tasks and generate paper-ready tables.
#
# Usage:
#   ./scripts/run_all_baselines.sh [--device cuda:N] [--dry-run]
#
# This is the reproducibility entry point for the CardiomniBench-VD evaluation.
# It orchestrates every baseline method × every task, writes results to runs/,
# and produces consolidated CSV/LaTeX tables ready for the paper.
#
# Requirements:
#   - .venv must be synced (uv sync --extra llm --extra vlm --extra specialist)
#   - HF_HOME must point to the 92GB model cache (see env.sh)
#   - At least one idle GPU (8x H20, device 0 reserved by convention)
#
# Output:
#   runs/all_baselines_YYYYMMDD_HHMMSS/
#     ├── arcade_segmentation/{cases.jsonl,summary.json,table.txt,table.tex}
#     ├── arcade_stenosis/
#     ├── cardiosyntax_scoring/
#     └── cca_segmentation/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# shellcheck source=env.sh
source env.sh

DEVICE="$CARDIOMNI_DEFAULT_DEVICE"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

RUN_DIR="runs/all_baselines_$(date +%Y%m%d_%H%M%S)"

# Methods are grouped by the modality they actually accept, not by family name.
# coronary_unet and sam_med3d are 3D CTA models and cannot serve 2D XCA tasks;
# listing them under ARCADE would produce guaranteed failures, not baselines.
XCA_2D=(
    coronary_cm_unet
    coronary_cm_unet_native
)
SYNTAX_SCORERS=(
    cardiosyntax_r3d
    cardiosyntax_r3d_calibrated
    cardiosyntax_r3d_fold0
    cardiosyntax_r3d_fold1
    cardiosyntax_r3d_fold2
    cardiosyntax_r3d_fold3
    cardiosyntax_r3d_fold4
)
CTA_3D=(
    coronary_unet
    sam_med3d
)

# All 6 VLMs now have complete shards (verified 2026-07-26 03:00 via
# check_available(): 13.9-17.1GB each). Earlier runs listed only 3 because an
# interrupted download had left .incomplete blobs behind, which check_available()
# correctly refuses but cannot distinguish from a download still in flight.
#
# The two Qwen entries additionally need `qwen-vl-utils`, which is NOT in gkp-gsa:
#   /opt/anaconda3/envs/gkp-gsa/bin/pip install qwen-vl-utils
# transformers (5.14.1) and accelerate are already present, and both
# LlavaNextProcessor and Qwen2_5_VLForConditionalGeneration import cleanly.
VLMS_READY=(
    llava_16_mistral_7b
    llama3_llava_next_8b
    llava_onevision_7b
    lingshu_7b
)

# Uncomment once qwen-vl-utils is installed.
# VLMS_READY+=(qwen25_vl_7b qwen3_vl_8b)

TASKS=(arcade_segmentation arcade_stenosis cardiosyntax_scoring cca_segmentation)

echo "==================================================================="
echo "CardiomniBench-VD: Reproducible Baseline Evaluation"
echo "==================================================================="
echo "Output directory: $RUN_DIR"
echo "Device: $DEVICE"
echo "Dry run: $DRY_RUN"
echo "Python: $BENCH_PY ($($BENCH_PY --version 2>&1 | head -1))"
echo

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN — commands that would be executed:"
    echo
fi

for task in "${TASKS[@]}"; do
    # clDice and HD95 each cost minutes per case on the 832x832x576 CCA volumes,
    # so the sweep reports Dice/precision/recall first and leaves the topology
    # metrics to a targeted rerun. Without this a 20-case CCA pass takes hours
    # and starves the other tasks of GPU.
    if [ "$task" = "cca_segmentation" ]; then
        export CARDIOMNI_FAST_METRICS=1
    else
        unset CARDIOMNI_FAST_METRICS
    fi

    case "$task" in
        arcade_segmentation | arcade_stenosis)
            methods=("${XCA_2D[@]}" "${VLMS_READY[@]}")
            ;;
        cardiosyntax_scoring)
            methods=("${SYNTAX_SCORERS[@]}" "${VLMS_READY[@]}")
            ;;
        cca_segmentation)
            methods=("${CTA_3D[@]}")  # text-output models cannot emit a 3D volume
            ;;
        *)
            echo "ERROR: Unknown task $task"
            exit 1
            ;;
    esac

    task_dir="$RUN_DIR/$task"
    echo "--- $task (${#methods[@]} methods) ---"

    for method in "${methods[@]}"; do
        if [ "$DRY_RUN" = true ]; then
            echo "  $method -> $task_dir"
            continue
        fi

        echo "  Running $method on $task ..."
        # One method per invocation, and a failure never aborts the sweep: a
        # missing checkpoint or an OOM on one method must not discard the
        # results already collected for the others.
        if ! "$BENCH_PY" -m benchmark.run_unified \
            --methods "$method" \
            --tasks "$task" \
            --device "$DEVICE" \
            --output-dir "$task_dir"; then
            echo "  FAILED: $method on $task (continuing)"
        fi
    done
    echo
done

if [ "$DRY_RUN" = false ]; then
    echo "All runs complete. Generating consolidated tables ..."
    # tables.py takes run_dir positionally and writes to stdout. It enforces the
    # reporting rules (n travels with every mean, zero-score cases broken out,
    # zero-shot rows labelled) so paper numbers are never typed by hand.
    for task in "${TASKS[@]}"; do
        task_dir="$RUN_DIR/$task"
        [ -f "$task_dir/cases.jsonl" ] || continue
        "$BENCH_PY" -m benchmark.tables "$task_dir" > "$task_dir/table.txt" 2>&1 || true
        "$BENCH_PY" -m benchmark.tables --latex "$task_dir" > "$task_dir/table.tex" 2>&1 || true
    done
    echo
    echo "Per-task tables written to $RUN_DIR/<task>/table.{txt,tex}"
    echo "Raw per-case records: $RUN_DIR/<task>/cases.jsonl"
    echo
    echo "CCA topology metrics (clDice, HD95) were skipped for speed. To add them:"
    echo "  CARDIOMNI_FAST_METRICS= $BENCH_PY -m benchmark.run_unified \\"
    echo "      --methods coronary_unet --tasks cca_segmentation --device $DEVICE \\"
    echo "      --output-dir $RUN_DIR/cca_topology"
fi
