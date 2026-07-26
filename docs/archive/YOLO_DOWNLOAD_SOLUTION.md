# YOLO Download Solution Guide

**Problem**: GitHub Release Assets use JWT tokens that expire after 1 hour, causing download failures on slow networks.

**Current Status**: All YOLO downloads failed after partial completion (40MB each, all JWT expired)

---

## ✅ Recommended Solution: Direct Download URLs

Use these stable, direct URLs from official sources:

### YOLOv8 Models (Ultralytics Official)

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/yolo_models

# YOLOv8x - Object Detection (131 MB)
wget --continue --timeout=60 --tries=0 \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt

# YOLOv8x-seg - Instance Segmentation (131 MB)  
wget --continue --timeout=60 --tries=0 \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
```

### YOLOv9 Models (WongKinYiu Official)

```bash
# YOLOv9c - Object Detection (102 MB)
wget --continue --timeout=60 --tries=0 \
  https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
```

**Notes**:
- `--continue`: Resume interrupted downloads
- `--timeout=60`: 60s timeout per chunk
- `--tries=0`: Infinite retries

---

## 🔄 Alternative: Use Mirror Sites

### Option A: Hugging Face Mirror (Recommended for China)

```bash
# Install huggingface-hub if not available
pip3 install huggingface-hub --index-url https://mirrors.aliyun.com/pypi/simple/

# Download via Python
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

os.makedirs("yolo_models", exist_ok=True)

models = [
    ("Ultralytics/YOLOv8", "yolov8x.pt"),
    ("Ultralytics/YOLOv8", "yolov8x-seg.pt"),
]

for repo, filename in models:
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            cache_dir="./yolo_models",
            resume_download=True
        )
        print(f"✅ Downloaded: {path}")
    except Exception as e:
        print(f"❌ Failed {filename}: {e}")
EOF
```

**Note**: Need to verify if models exist on HuggingFace. If not, upload them first.

### Option B: Use aria2c (Multi-threaded Downloader)

```bash
# Install aria2
yum install -y aria2

# Download with 8 connections and auto-retry
aria2c --max-connection-per-server=8 --continue=true --max-tries=0 \
  --retry-wait=5 --timeout=60 \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt

aria2c --max-connection-per-server=8 --continue=true --max-tries=0 \
  --retry-wait=5 --timeout=60 \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt

aria2c --max-connection-per-server=8 --continue=true --max-tries=0 \
  --retry-wait=5 --timeout=60 \
  https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
```

---

## 🌐 Option C: Download Externally and Transfer

If server network is unstable, download on a machine with stable internet:

```bash
# On local machine with good internet
cd /tmp
wget -c https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt
wget -c https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt
wget -c https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt

# Verify integrity
ls -lh *.pt
file *.pt  # Should show "Zip archive data"

# Transfer to server
scp *.pt root@your-server:/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/yolo_models/
```

---

## 🔍 Verify Downloaded Models

After download, verify integrity:

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/yolo_models

# Check file sizes (should match expected)
ls -lh *.pt

# Expected sizes:
# yolov8x.pt:     ~131 MB
# yolov8x-seg.pt: ~131 MB  
# yolov9-c.pt:    ~102 MB

# Check file types (should be "Zip archive data")
file *.pt

# Test loading with PyTorch
python3 << 'EOF'
import torch
for model in ['yolov8x.pt', 'yolov8x-seg.pt', 'yolov9-c.pt']:
    try:
        checkpoint = torch.load(model, map_location='cpu')
        print(f"✅ {model}: Valid PyTorch model")
    except Exception as e:
        print(f"❌ {model}: {e}")
EOF
```

---

## 📋 Expected File Checksums

Once successfully downloaded, record MD5 hashes for future verification:

```bash
md5sum *.pt > CHECKSUMS.md5

# Example output format:
# a1b2c3d4... yolov8x.pt
# e5f6g7h8... yolov8x-seg.pt
# i9j0k1l2... yolov9-c.pt
```

---

## 🎯 Quick Start Script

Create an automated download script:

```bash
cat > download_yolo.sh << 'SCRIPT'
#!/bin/bash
set -e

MODELS_DIR="/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/yolo_models"
cd "$MODELS_DIR"

echo "Downloading YOLO models..."

# Function to download with retries
download_model() {
    local url=$1
    local output=$2
    echo "Downloading $output..."
    wget --continue --timeout=60 --tries=0 --progress=bar:force "$url" -O "$output"
}

# Download models
download_model \
  "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt" \
  "yolov8x.pt"

download_model \
  "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-seg.pt" \
  "yolov8x-seg.pt"

download_model \
  "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt" \
  "yolov9c.pt"

echo "Verifying downloads..."
python3 test_integrity.py

echo "✅ All models downloaded successfully!"
SCRIPT

chmod +x download_yolo.sh
```

Run with:
```bash
./download_yolo.sh
```

---

## ⚠️ Known Issues

1. **JWT Expiration**: GitHub release assets have 1-hour token expiration
   - **Solution**: Use `--continue` and retry immediately
   
2. **Slow Network**: Downloads may take 1+ hour per file
   - **Solution**: Use aria2c with multiple connections
   
3. **Connection Reset**: Alibaba Cloud may have network instability
   - **Solution**: Use infinite retries (`--tries=0`)

---

## 🚀 When Models are Ready

After successful download, update toolkit and test:

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms

# Test YOLO integration
python3 << 'EOF'
from toolkit import CardiomniToolkit
import numpy as np

toolkit = CardiomniToolkit(device="cpu")

# Create test image
test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# Test detection
detections = toolkit.detect_stenosis(test_img, conf=0.5)
print(f"✅ Detection working: {len(detections)} objects")

# Test segmentation  
segments = toolkit.segment_vessels(test_img)
print(f"✅ Segmentation working: {len(segments)} masks")

print("\n🎉 YOLO integration complete!")
EOF
```

---

## 📊 Impact on Toolkit

**Before YOLO**: 4/7 tools (57%)
**After YOLO**: 7/7 tools (100%)

This unblocks:
- Stenosis detection (YOLOv8x)
- Vessel segmentation (YOLOv8x-seg)
- Advanced quantification (YOLOv9c)
