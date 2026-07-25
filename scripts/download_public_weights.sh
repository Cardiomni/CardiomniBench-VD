#!/bin/bash
# Download publicly accessible specialist model weights (no auth required)
# Run from CardiomniBench-VD root: bash scripts/download_public_weights.sh

set -e
WEIGHTS_DIR="algorithms/specialist_models/weights"
mkdir -p "$WEIGHTS_DIR"
cd "$WEIGHTS_DIR"

echo "=== Downloading public specialist model weights ==="
echo "Working directory: $(pwd)"
echo

# Use hf-mirror.com for China network access
export HF_ENDPOINT="https://hf-mirror.com"

# --- Public HuggingFace models (no auth, no LFS) ---

echo "[1/5] CM-UNet (Contrastive-Masked UNet)..."
if [ ! -d "CM-UNet" ]; then
    git clone --depth 1 https://hf-mirror.com/Camsouille/CM-UNet
    echo "✓ CM-UNet cloned"
else
    echo "✓ CM-UNet already exists"
fi
echo

echo "[2/5] MesserMMP SYNTAX Prediction..."
if [ ! -d "coronary-syntax-prediction" ]; then
    git clone --depth 1 https://hf-mirror.com/MesserMMP/coronary-syntax-prediction
    echo "✓ MesserMMP SYNTAX cloned"
else
    echo "✓ MesserMMP SYNTAX already exists"
fi
echo

echo "[3/5] TC-SemiSAM checkpoints..."
if [ ! -d "TC-SemiSAM-checkpoints" ]; then
    git clone --depth 1 https://hf-mirror.com/ly17/TC-SemiSAM-checkpoints
    echo "✓ TC-SemiSAM cloned"
else
    echo "✓ TC-SemiSAM already exists"
fi
echo

echo "[4/5] SAM3 vessel segmentation..."
if [ ! -d "sam3-vessel-segmentation" ]; then
    git clone --depth 1 https://hf-mirror.com/ly17/sam3-vessel-segmentation
    echo "✓ SAM3 vessel cloned"
else
    echo "✓ SAM3 vessel already exists"
fi
echo

echo "[5/5] SAM-VMNet..."
if [ ! -d "SAM-VMNet" ]; then
    git clone --depth 1 https://hf-mirror.com/ly17/SAM-VMNet
    echo "✓ SAM-VMNet cloned"
else
    echo "✓ SAM-VMNet already exists"
fi
echo

echo "=== Download summary ==="
du -sh */ 2>/dev/null | sort -h
echo
echo "✓ All public weights downloaded to: $(pwd)"
echo
echo "Skipped (require authentication):"
echo "  - DeepCoro (heartwise/DeepCoro)"
echo "  - DeepCORO-CLIP (heartwise/deepcoro_clip, heartwise/VasoVision)"
echo
echo "Note: Large model files (*.pth, *.bin, *.safetensors) may be Git LFS pointers."
echo "      If needed, install git-lfs and run 'git lfs pull' in each repo directory."
