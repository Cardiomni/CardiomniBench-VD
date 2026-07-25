#!/bin/bash
# Download specialist model weights for CardiomniBench-VD
# Run from CardiomniBench-VD root: bash scripts/download_specialist_weights.sh

set -e
WEIGHTS_DIR="algorithms/specialist_models/weights"
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "=== Downloading specialist model weights ==="
echo "Working directory: $(pwd)"
echo

# --- HuggingFace models via hf-mirror.com (China mirror) ---
export HF_ENDPOINT="https://hf-mirror.com"

echo "[1/6] CM-UNet (Contrastive-Masked UNet) from HuggingFace..."
if [ ! -d "CM-UNet" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/Camsouille/CM-UNet
    cd CM-UNet && git lfs pull && cd ..
    echo "✓ CM-UNet downloaded"
else
    echo "✓ CM-UNet already exists"
fi
echo

echo "[2/6] DeepCoro from HuggingFace..."
if [ ! -d "DeepCoro" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/heartwise/DeepCoro
    cd DeepCoro && git lfs pull && cd ..
    echo "✓ DeepCoro downloaded"
else
    echo "✓ DeepCoro already exists"
fi
echo

echo "[3/6] MesserMMP SYNTAX Prediction from HuggingFace..."
if [ ! -d "coronary-syntax-prediction" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/MesserMMP/coronary-syntax-prediction
    cd coronary-syntax-prediction && git lfs pull && cd ..
    echo "✓ MesserMMP SYNTAX downloaded"
else
    echo "✓ MesserMMP SYNTAX already exists"
fi
echo

echo "[4/6] TC-SemiSAM checkpoints from HuggingFace..."
if [ ! -d "TC-SemiSAM-checkpoints" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/ly17/TC-SemiSAM-checkpoints
    cd TC-SemiSAM-checkpoints && git lfs pull && cd ..
    echo "✓ TC-SemiSAM downloaded"
else
    echo "✓ TC-SemiSAM already exists"
fi
echo

echo "[5/6] SAM3 vessel segmentation from HuggingFace..."
if [ ! -d "sam3-vessel-segmentation" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/ly17/sam3-vessel-segmentation
    cd sam3-vessel-segmentation && git lfs pull && cd ..
    echo "✓ SAM3 vessel downloaded"
else
    echo "✓ SAM3 vessel already exists"
fi
echo

# --- SAM-VMNet (also has HuggingFace mirror) ---
echo "[6/6] SAM-VMNet from HuggingFace..."
if [ ! -d "SAM-VMNet" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/ly17/SAM-VMNet
    cd SAM-VMNet && git lfs pull && cd ..
    echo "✓ SAM-VMNet downloaded"
else
    echo "✓ SAM-VMNet already exists"
fi
echo

echo "=== Download summary ==="
du -sh */ 2>/dev/null | sort -h
echo
echo "✓ All available HuggingFace weights downloaded to:"
echo "  $(pwd)"
echo
echo "Note: DeepCORO-CLIP (heartwise/deepcoro_clip, heartwise/VasoVision) requires"
echo "      authentication token. Follow algorithms/specialist_models/deepcoro_clip/DOWNLOAD_STATUS.md"
echo "      to configure HuggingFace token and download those weights."
