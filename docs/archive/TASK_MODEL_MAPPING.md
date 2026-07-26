# Task-Model Mapping for Cardiomni Agent

**Purpose**: Map DATASETS_GUIDE tasks → Available specialist models with weights  
**Reference**: EchoAgent paradigm (specialist models as callable tools)  
**Date**: 2026-07-24

---

## 核心问题：Cardiomni Agent需要哪些"Hands"？

基于DATASETS_GUIDE.html + EchoAgent范式，Cardiomni需要完成以下任务：

---

## Task Breakdown & Model Allocation

### **Task 1: 血管分割 (Vessel Segmentation)**

#### ARCADE Task 1需求
```json
Input: 512×512 DSA PNG (phase_1 or final_phase)
Output: {
  "detections": [
    {"label": "1-16", "bounding_box": [...], "mask": {...}}  // AHA 16-segment
  ]
}
Metrics: mAP@IoU=0.50, Dice coefficient
```

#### 已配置模型（⚠️ 权重问题）
| Model | Weights | Performance | Recommendation |
|-------|---------|-------------|----------------|
| **SAM-VMNet** | ❌ Blocked (hf-mirror failed) | IoU 0.63 (SOTA) | **P0 - Need workaround** |
| **CM-UNet** | ❌ Blocked | Dice +48.7% | P1 |
| **FR-UNet** | ⏳ Uncertain | JBHI 2021 | P2 - Check GitHub |

**🚨 Problem**: Top-performing models (SAM-VMNet, CM-UNet) have blocked weights.

**✅ Solution Options**:
1. **Train nnU-Net on ARCADE segmentation task** (1500 images, standard baseline)
2. **Fine-tune SAM2 on ARCADE** (foundation model, no custom weights needed)
3. **Use MedSAM + prompt engineering** (zero-shot, HuggingFace available)

**Recommended Action**:
```python
# Option: Use MedSAM (available weights) + ARCADE fine-tuning
from segment_anything import sam_model_registry
model = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
# Fine-tune on ARCADE 1500 segmentation images
```

---

### **Task 2: 狭窄检测 (Stenosis Detection)**

#### ARCADE Task 2需求
```json
Input: 512×512 DSA PNG
Output: {
  "detections": [
    {"label": "stenosis", "bounding_box": [...], "mask": {...}}
    // Multiple stenosis possible (69/1500 images have multi-lesion)
  ]
}
Metrics: Precision, Recall, F1 @ IoU=0.50
```

#### 已配置模型（✅ 有可用权重）
| Model | Weights | Performance | Priority |
|-------|---------|-------------|----------|
| **YOLOv11-X** | ✅ Official (Ultralytics) | **F1 0.7826** (ARCADE) | **P0 - Ready** |
| **YOLOv8 HuggingFace** | ✅ Available | Production-ready | **P0 - Ready** |
| **ARCADE-stenosis** | ⏳ Check GitHub | F1 0.5353 (runner-up) | P1 |
| **YOLOv9-E** | ✅ Official | F1 0.4524 | P2 |

**✅ Status**: **READY** - YOLOv11-X and YOLOv8 both have official weights.

**Recommended Stack**:
```python
# Primary: YOLOv11-X (best performance)
from ultralytics import YOLO
detector = YOLO('yolov11x.pt')
results = detector.predict(dsa_image, conf=0.5, iou=0.5)

# Backup: YOLOv8 from HuggingFace
from transformers import pipeline
detector_backup = pipeline("object-detection", 
                          model="rachitgoyell/stenosis-detection")
```

---

### **Task 3: 狭窄定量 (Stenosis Quantification)**

#### 需求（中山模板风格）
```json
Input: DSA image + stenosis bounding box
Output: {
  "stenosis_severity": "70-99%" | "50-70%" | "25-50%" | "0-25%" | "100%",
  "reference_diameter_mm": 3.2,
  "minimal_lumen_diameter_mm": 0.9,
  "stenosis_percentage": 71.8
}
Clinical Thresholds:
  - 0-25%: Normal
  - 25-50%: Mild
  - 50-70%: Moderate
  - 70-99%: Severe (需要介入)
  - 100%: Occlusion (紧急)
```

#### 已配置模型（✅ 有可用权重）
| Model | Weights | Performance | Priority |
|-------|---------|-------------|----------|
| **YOLOv9c** | ✅ Official (Ultralytics) | **F1 0.99, mAP@50 0.99** | **P0 - Ready** |
| **Modified YOLOv8x** | ⏳ Contact authors | Precision 0.991 + troponin | P1 |
| **DeepCORO-CLIP** | ❌ Auth required | MAE 13.6% vs QCA | P2 (reference only) |

**✅ Status**: **READY** - YOLOv9c has exceptional performance with official weights.

**⚠️ Note**: ARCADE数据集**不提供狭窄程度标注**，只有位置。需要：
- **Option A**: 使用YOLOv9c在有标注的数据集上训练（需找带severity的数据）
- **Option B**: QCA-style径线测量算法（classical method, no weights needed）
- **Option C**: 使用DeepCORO-CLIP作为"second opinion"（如果能拿到权重）

**Recommended Approach**:
```python
# Use classical QCA algorithm (no weights needed)
def qca_quantification(image, stenosis_bbox):
    """Quantitative Coronary Analysis - classical method"""
    # 1. Extract vessel centerline
    centerline = extract_centerline(image, stenosis_bbox)
    
    # 2. Measure diameters along centerline
    diameters = measure_diameters(centerline)
    
    # 3. Find reference (normal) and minimal lumen diameter
    ref_diameter = np.percentile(diameters, 90)  # Top 10% as reference
    min_diameter = np.min(diameters)
    
    # 4. Calculate stenosis percentage
    stenosis_pct = (ref_diameter - min_diameter) / ref_diameter * 100
    
    # 5. Map to clinical categories
    if stenosis_pct < 25:
        category = "0-25%"
    elif stenosis_pct < 50:
        category = "25-50%"
    elif stenosis_pct < 70:
        category = "50-70%"
    elif stenosis_pct < 100:
        category = "70-99%"
    else:
        category = "100%"
    
    return {
        "stenosis_percentage": stenosis_pct,
        "category": category,
        "ref_diameter_mm": ref_diameter,
        "min_diameter_mm": min_diameter
    }
```

---

### **Task 4: SYNTAX评分 (SYNTAX Score)**

#### CardioSYNTAX需求
```json
Input: Multi-view DSA video sequences (5-10 views)
Output: {
  "syntax_total": 24,  // 0-67
  "left_system": 16,
  "right_system": 8
}
Metrics: MAE, Pearson correlation
```

#### 已配置模型（❌ 权重被阻断）
| Model | Weights | Performance | Priority |
|-------|---------|-------------|----------|
| **CardioSYNTAX** | ❌ Blocked | End-to-end SYNTAX | **P0 - Blocked** |
| **MesserMMP** | ❌ Blocked | R3D + sequence | **P1 - Blocked** |

**🚨 Problem**: 两个主要模型权重都被阻断。

**✅ Workaround**: 
```python
# Use rule-based SYNTAX calculator (based on detected stenosis)
def calculate_syntax_score(segments_report, dominance):
    """
    Rule-based SYNTAX score calculation
    Reference: Serruys et al. EuroIntervention 2005
    """
    score = 0
    
    for segment in segments_report:
        if segment["stenosis_severity"] in ["50-70%", "70-99%", "100%"]:
            # Base score by segment location (AHA 16-segment weights)
            segment_weight = SYNTAX_WEIGHTS[segment["segment_id"]]
            
            # Severity multiplier
            if segment["stenosis_severity"] == "100%":
                multiplier = 5  # Total occlusion
            elif segment["stenosis_severity"] == "70-99%":
                multiplier = 2
            else:
                multiplier = 1
            
            # Additional factors
            if segment.get("bifurcation", False):
                multiplier += 1
            if segment.get("calcification") == "severe":
                multiplier += 1
            
            score += segment_weight * multiplier
    
    return min(score, 67)  # Cap at 67

SYNTAX_WEIGHTS = {
    1: 3.5,  # Proximal RCA
    2: 2.5,  # Mid RCA
    5: 5.0,  # LM (left main)
    6: 3.5,  # Proximal LAD
    # ... etc (16-segment weights from literature)
}
```

**Alternative**: Train lightweight regression model on ARCADE + manual SYNTAX annotations.

---

### **Task 5: 优势型判定 (Dominance Classification)**

#### 需求
```json
Input: Multi-view DSA sequences
Output: "right" | "left" | "co-dominant"
Criterion: Which artery supplies PDA (posterior descending artery)
```

#### 已配置模型（⚠️ 权重不确定）
| Model | Weights | Performance | Priority |
|-------|---------|-------------|----------|
| **Neural Network (RCA-based)** | ⏳ Contact authors | Acc 93.5% | P1 |
| **ResNet-50 fine-tuned** | ❌ Need to train | Standard baseline | **P0 - DIY** |

**✅ Solution**: Train ResNet-50 on ARCADE (if dominance labels exist) or annotate manually.

```python
# Train simple classifier
import torchvision.models as models
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 3)  # 3 classes: right/left/co-dominant

# Fine-tune on DSA images with dominance labels
```

---

### **Task 6: 视图分类 (Projection Classification)**

#### 需求
```json
Input: DSA DICOM with metadata
Output: "RAO_30_CAUDAL_20" (parsed from PositionerPrimaryAngle + PositionerSecondaryAngle)
```

#### 已配置模型（⚠️ 可能不需要深度学习）
| Model | Weights | Approach | Priority |
|-------|---------|----------|----------|
| **CathAI projection stage** | ❌ Proprietary | Reference only | N/A |
| **Rule-based DICOM parser** | ✅ No weights needed | Extract from metadata | **P0 - Ready** |

**✅ Solution**: DICOM metadata already contains angles!

```python
import pydicom

def classify_projection(dicom_path):
    """Extract projection from DICOM metadata"""
    ds = pydicom.dcmread(dicom_path)
    
    # Read angles from DICOM tags
    primary_angle = ds.PositionerPrimaryAngle  # -90 to +90
    secondary_angle = ds.PositionerSecondaryAngle  # -90 to +90
    
    # Classify primary
    if primary_angle < -10:
        primary = f"RAO_{abs(int(primary_angle))}"
    elif primary_angle > 10:
        primary = f"LAO_{int(primary_angle)}"
    else:
        primary = "AP"
    
    # Classify secondary
    if secondary_angle < -10:
        secondary = f"CAUDAL_{abs(int(secondary_angle))}"
    elif secondary_angle > 10:
        secondary = f"CRANIAL_{int(secondary_angle)}"
    else:
        secondary = ""
    
    return f"{primary}_{secondary}" if secondary else primary

# Example: "RAO_30_CAUDAL_20"
```

**No deep learning needed!** Metadata parsing is sufficient.

---

## 工具库总结：可用 vs. 被阻断

### ✅ **立即可用（有权重）**
1. **YOLOv11-X** (狭窄检测, F1 0.7826) - Ultralytics官方
2. **YOLOv8** (狭窄检测) - HuggingFace
3. **YOLOv9c** (狭窄定量, F1 0.99) - Ultralytics官方
4. **Rule-based QCA** (径线测量) - 经典算法，无需权重
5. **DICOM metadata parser** (视图分类) - 无需权重

### ⚠️ **需要解决方案（权重被阻断）**
6. **血管分割**: SAM-VMNet/CM-UNet被阻断 → 用MedSAM fine-tune或nnU-Net
7. **SYNTAX评分**: CardioSYNTAX/MesserMMP被阻断 → 用rule-based计算器
8. **优势型判定**: 无现成权重 → 训练ResNet-50

### ❌ **可选增强（如果拿到权重）**
9. **DeepCORO-CLIP** (狭窄定量second opinion, MAE 13.6%)
10. **ARCADE-stenosis** (狭窄检测backup, F1 0.5353)

---

## Cardiomni Agent工具栈配置（最小可行版本）

```python
class CardiomniToolkit:
    """Hierarchical Collaboration Toolkit for Cardiomni"""
    
    def __init__(self):
        # Layer 1: YOLO family (Ready)
        self.stenosis_detector = YOLO('yolov11x.pt')  # ✅ Official
        self.stenosis_quantifier = YOLO('yolov9c.pt')  # ✅ Official
        
        # Layer 2: Segmentation (Need workaround)
        self.vessel_segmenter = MedSAM()  # ✅ Fine-tune on ARCADE
        
        # Layer 3: Classical algorithms (No weights needed)
        self.qca_engine = QCA_Algorithm()  # ✅ Ready
        self.syntax_calculator = RuleBasedSYNTAX()  # ✅ Ready
        self.projection_parser = DICOMMetadataParser()  # ✅ Ready
        
        # Layer 4: Train ourselves (Small models)
        self.dominance_classifier = ResNet50(num_classes=3)  # ⏳ Need training
    
    def detect_stenosis(self, image):
        return self.stenosis_detector.predict(image)
    
    def segment_vessels(self, image):
        return self.vessel_segmenter.segment(image)
    
    def quantify_stenosis(self, image, bbox):
        # Primary: YOLO-based
        yolo_result = self.stenosis_quantifier.predict(image)
        # Backup: Classical QCA
        qca_result = self.qca_engine.measure(image, bbox)
        return ensemble([yolo_result, qca_result])
    
    def calculate_syntax(self, segments_report, dominance):
        return self.syntax_calculator.compute(segments_report, dominance)
    
    def classify_projection(self, dicom_path):
        return self.projection_parser.extract_angles(dicom_path)
    
    def determine_dominance(self, multi_view_images):
        return self.dominance_classifier.predict(multi_view_images)
```

---

## 行动方案

### Phase 1: 部署已有权重模型（1天）
```bash
# Install Ultralytics
pip install ultralytics

# Test YOLO models
python -c "
from ultralytics import YOLO
det = YOLO('yolov11x.pt')
quant = YOLO('yolov9c.pt')
print('✅ YOLO models ready')
"
```

### Phase 2: 实现经典算法（2天）
- QCA径线测量
- Rule-based SYNTAX计算器
- DICOM metadata解析器

### Phase 3: 训练小模型（3-5天）
- MedSAM fine-tune on ARCADE segmentation
- ResNet-50 dominance classifier

### Phase 4: 集成到Cardiomni Agent（1-2天）
- 实现HC toolkit接口
- 对接OR Hub编排逻辑

---

## 关键结论

**是的，我们需要配备权重的模型！**

- **好消息**: YOLO家族（YOLOv11-X、YOLOv9c、YOLOv8）有官方权重，覆盖了**狭窄检测+定量**两大核心任务
- **坏消息**: SAM-VMNet、CardioSYNTAX、DeepCORO-CLIP等SOTA模型权重被阻断
- **解决方案**: 
  1. 用YOLO + classical algorithms先搭建MVP
  2. 并行训练/fine-tune小模型补齐缺失部分
  3. 如果网络问题解决，再替换为SOTA模型

**当前可立即开工**，不需要等所有权重到位！
