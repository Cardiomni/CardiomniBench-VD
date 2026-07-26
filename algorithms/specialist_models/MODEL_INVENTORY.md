# CardiomniBench-VD Model Inventory

**Generated**: 2026-07-24  
**Base Directory**: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models`

---

## ✅ Ready to Use (Available)

### Rule-based Methods (No Weights Needed)
| Method | Task | Status | Location |
|--------|------|--------|----------|
| **SYNTAX Calculator** | SYNTAX Scoring | ✅ Working | `syntax_scoring/rule_based_syntax.py` |
| **DICOM Parser** | Projection Classification | ✅ Working | `projection_classification/dicom_parser.py` |

**Usage**:
```python
# SYNTAX scoring
from algorithms.specialist_models.syntax_scoring.rule_based_syntax import calculate_syntax_score
result = calculate_syntax_score(segments_report, dominance="right")
print(result['syntax_total'])  # 0-67

# Projection classification
from algorithms.specialist_models.projection_classification.dicom_parser import classify_projection
view = classify_projection("path/to/dicom.dcm")
print(view)  # "RAO_30_CAUDAL_20"
```

---

## ⏳ Code Available (Need Weight Discovery)

### GitHub Repositories
| Repo | Task | Status | Location | Next Step |
|------|------|--------|----------|-----------|
| **ARCADE-stenosis** | Stenosis Detection | ✅ Cloned | `github_repos/ARCADE-stenosis/` | Check for `weights/` or `*.pth` |
| **StenUNet** | Stenosis Detection | ✅ Cloned | `github_repos/StenUNet/` | Check for pretrained models |
| **FRNet** | Vessel Segmentation | ✅ Cloned | `github_repos/FRNet/` | Check for weights |
| **Faster-RCNN** | Stenosis Detection | ✅ Cloned | `github_repos/Faster-RCNN/` | Check for checkpoints |

**Weight Discovery Command**:
```bash
cd github_repos
find . -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" -o -name "*.h5" -o -name "*.weights"
```

---

## ❌ Previously Downloaded (Weights Blocked)

| Model | Task | Issue | Location |
|-------|------|-------|----------|
| **SAM-VMNet** | Vessel Segmentation | HuggingFace blocked | `sam_vmnet/` (code only) |
| **CM-UNet** | Vessel Segmentation | HuggingFace blocked | `cm_unet/` (code only) |
| **CardioSYNTAX** | SYNTAX Scoring | HuggingFace blocked | `cardiosyntax/` (code only) |
| **DeepCORO-CLIP** | Multi-task | Auth required | `deepcoro_clip/` (code only) |

**Performance (if weights were available)**:
- SAM-VMNet: IoU 0.63 (ARCADE SOTA)
- CM-UNet: Dice +48.7%
- CardioSYNTAX: End-to-end SYNTAX prediction
- DeepCORO-CLIP: MAE 13.6% stenosis quantification

---

## 🔧 YOLO Models (Requires Ultralytics)

**Note**: Ultralytics installation failed on this system. Alternative approaches:

### Option A: Install with Conda/Mamba
```bash
conda install -c conda-forge ultralytics
```

### Option B: Manual Download
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11x.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov9c.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
```

### Option C: Use HuggingFace Mirror
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline
detector = pipeline("object-detection", model="rachitgoyell/stenosis-detection")
```

**Target Models**:
| Model | Task | Performance | Size |
|-------|------|-------------|------|
| YOLOv11-X | Stenosis Detection | F1 0.7826 | ~200MB |
| YOLOv9c | Stenosis Quantification | F1 0.99 | ~100MB |
| YOLOv8x-seg | Vessel Segmentation | Baseline | ~150MB |
| YOLOv8n-cls | Classification | Baseline | ~6MB |

---

## 📊 Current Coverage by Task

| Task | Available Methods | Status | Recommended |
|------|------------------|--------|-------------|
| **Projection Classification** | DICOM parser | ✅ Ready | Use DICOM metadata |
| **SYNTAX Scoring** | Rule-based calculator | ✅ Ready | Use rule-based |
| **Stenosis Detection** | 4 GitHub repos | ⏳ Check weights | ARCADE-stenosis, StenUNet |
| **Vessel Segmentation** | FRNet + SAM-VMNet/CM-UNet | ⏳ Mixed | FRNet (check weights) |
| **Stenosis Quantification** | DeepCORO-CLIP | ❌ Blocked | Classical QCA algorithm |
| **Dominance Classification** | - | ❌ Need implementation | Train ResNet-50 |

---

## 🚀 Quick Start Guide

### Step 1: Test Rule-based Methods
```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models

# Test SYNTAX calculator
python3 syntax_scoring/rule_based_syntax.py

# Test DICOM parser (requires DICOM file)
# python3 projection_classification/dicom_parser.py
```

### Step 2: Discover Weights in GitHub Repos
```bash
cd github_repos

# ARCADE-stenosis
find ARCADE-stenosis -name "*.pth" -o -name "*.pt" -o -name "*.ckpt"

# StenUNet
find StenUNet -name "*.pth" -o -name "*.pt" -o -name "*.ckpt"

# FRNet
find FRNet -name "*.pth" -o -name "*.pt" -o -name "*.ckpt"

# Faster-RCNN
find Faster-RCNN -name "*.pth" -o -name "*.pt" -o -name "*.ckpt" -o -name "*.h5"
```

### Step 3: Test Toolkit Integration
```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD

# Test toolkit health check
python3 -c "from algorithms.toolkit import get_toolkit; tk = get_toolkit('cpu'); print(tk.health_check())"
```

---

## 📝 Next Actions (Priority Order)

### P0 - Immediate (This Week)
1. ✅ **SYNTAX scoring**: Rule-based calculator working
2. ✅ **Projection classification**: DICOM parser working
3. ⏳ **Check GitHub repo weights**: Run find commands above
4. ⏳ **Test toolkit**: Run health check

### P1 - Short-term (Next Week)
5. ⏳ **Alternative YOLO installation**: Try conda or manual download
6. ⏳ **Implement classical QCA**: For stenosis quantification
7. ⏳ **Train dominance classifier**: ResNet-50 on ARCADE (if labels available)

### P2 - Medium-term (Before Paper)
8. ⏳ **Contact authors**: SAM-VMNet, CardioSYNTAX for weights
9. ⏳ **Alternative segmentation**: MedSAM fine-tuning
10. ⏳ **Integrate into agent**: Connect toolkit to 4-stage SOP

---

## 🎯 Minimum Viable Stack (MVP)

**Goal**: Get Cardiomni agent running with available tools

| Component | Tool | Status |
|-----------|------|--------|
| Stage 1: Dominance | Manual annotation → ResNet-50 | ⏳ Need training |
| Stage 2: Vessel Scan | FRNet or manual annotation | ⏳ Check weights |
| Stage 3: View Selection | DICOM metadata | ✅ Ready |
| Stage 4: Lesion Assessment | GitHub repos + Classical QCA | ⏳ Check weights |
| SYNTAX Integration | Rule-based calculator | ✅ Ready |

**Critical Path**: Check GitHub repo weights (5 min) → Determine if we need to train alternatives

---

## 📞 Contact Information

**For Weight Access**:
- SAM-VMNet: [GitHub Issues](https://github.com/qimingfan10/SAM-VMNet/issues)
- ARCADE-stenosis: bhattarailab@gmail.com
- StenUNet: huilin0220@gmail.com
- CardioSYNTAX: Check paper authors
- DeepCORO-CLIP: Check paper authors

---

## 📖 Documentation References

- **ARCADE Dataset**: [Research Square](https://doi.org/10.21203/rs.3.rs-3610879/v1)
- **SYNTAX Score**: Serruys et al. EuroIntervention 2005
- **AHA 16-segment Model**: Cerqueira et al. Circulation 2002
- **EchoAgent Reference**: Wang et al. 2026 (our design paradigm)

---

**Last Updated**: 2026-07-24  
**Maintainer**: Cardiomni Team
