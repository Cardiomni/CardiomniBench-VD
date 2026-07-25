# CardiomniBench-VD 任务规范 (Task Specification)

**Version:** 1.0  
**Date:** 2026-07-23  
**Status:** Draft

---

## 1. 任务定义 (Task Definition)

### 1.1 核心目标

CardiomniBench-VD 评估 **自主诊断 Agent** 在冠状动脉 DSA 多视角推理任务中的能力：
- **输入**: 一个患者的多视角 DSA 视频集合
- **输出**: 结构化诊断报告（SYNTAX score、节段狭窄评估）
- **关键挑战**: Agent 需自主选择视角、整合多视角信息、进行临床推理

### 1.2 与陈秀川 Case 的关系

陈秀川 DSA case 是 **prototype**（原型任务），具有：
- 完整 DICOM 数据
- 金标准节段级狭窄标注
- 临床 SOP 参考

CardioSYNTAX 50 cases 是 **scale evaluation**（规模评估），具有：
- 标准化投影角度
- SYNTAX score 标注
- 足够统计显著性

---

## 2. 数据格式 (Data Format)

### 2.1 输入格式

#### Option A: 统一目录结构（推荐）
```
<case_id>/
├── metadata.json          # 案例元数据
└── videos/
    ├── video_001.npy      # 视频数据（或 DICOM）
    ├── video_001.json     # 视频元数据
    ├── video_002.npy
    ├── video_002.json
    └── ...
```

**metadata.json 格式:**
```json
{
  "case_id": "chen_xiuchuan_dsa",
  "case_type": "dicom",  // 或 "cardiosyntax"
  "patient_info": {
    "anonymized_id": "...",
    "acquisition_date": "2026-01-04"
  },
  "total_videos": 7
}
```

**video_XXX.json 格式:**
```json
{
  "video_id": "video_001",
  "file_path": "video_001.npy",  // 或 "IM000000.dcm"
  "modality": "XA",
  "artery": "LCA",  // LCA | RCA | Unknown
  "projection": {
    "primary_angle": -31.49,   // RAO- / LAO+
    "secondary_angle": -21.26  // CAU- / CRA+
  },
  "shape": [26, 512, 512],  // [frames, height, width]
  "frame_rate": 7.5
}
```

#### Option B: 原始格式兼容

Agent 需处理两种原始格式：
1. **陈秀川**: DICOM 目录，自行解析 DICOM headers
2. **CardioSYNTAX**: .npy 文件 + all.json metadata

---

### 2.2 输出格式

#### 统一 JSON Schema

```json
{
  "case_id": "chen_xiuchuan_dsa",
  "agent_name": "cardiomni_v1",
  "timestamp": "2026-07-23T10:30:00Z",
  
  "syntax_score": {
    "total": 27.0,
    "left": 22.0,
    "right": 5.0
  },
  
  "dominance": "right",  // "left" | "right" | "balanced"
  
  "segments": [
    {
      "segment_id": "LAD_proximal",
      "segment_name": "Left Anterior Descending - Proximal",
      "stenosis_percent": 80,
      "stenosis_grade": "severe",  // "normal" | "mild" | "moderate" | "severe" | "occluded"
      "confidence": 0.85,
      "evidence": {
        "primary_views": ["video_003", "video_005"],
        "supporting_views": ["video_001"],
        "key_frames": [
          {"video_id": "video_003", "frame_idx": 12},
          {"video_id": "video_005", "frame_idx": 8}
        ]
      }
    },
    {
      "segment_id": "D1",
      "segment_name": "First Diagonal Branch",
      "stenosis_percent": 30,
      "stenosis_grade": "mild",
      "confidence": 0.72,
      "evidence": {
        "primary_views": ["video_003"],
        "supporting_views": [],
        "key_frames": [{"video_id": "video_003", "frame_idx": 15}]
      }
    }
  ],
  
  "view_selection_log": [
    {
      "step": 1,
      "action": "dominance_check",
      "selected_views": ["video_006", "video_007"],
      "rationale": "LAO+CRA views optimal for determining dominance"
    },
    {
      "step": 2,
      "action": "lad_assessment",
      "selected_views": ["video_003", "video_005"],
      "rationale": "RAO+CAU and AP cranial for LAD proximal segment"
    }
  ],
  
  "reasoning_trace": {
    "stage1_dominance": "...",
    "stage2_systematic_scan": "...",
    "stage3_view_selection": "...",
    "stage4_lesion_assessment": "..."
  }
}
```

---

## 3. 评估指标 (Evaluation Metrics)

### 3.1 三轴评估框架

根据用户需求："概念创新、结果创新、工作完整度"，映射到：

#### Axis 1: **Correctness** (正确性)
评估诊断结果的准确性。

**For CardioSYNTAX (SYNTAX score only):**
- **Metric 1.1**: SYNTAX Score MAE
  ```
  MAE = mean(|predicted_syntax - ground_truth_syntax|)
  ```
  
- **Metric 1.2**: SYNTAX Score Tolerance Accuracy
  ```
  Acc@5 = % cases where |error| ≤ 5 points
  Acc@10 = % cases where |error| ≤ 10 points
  ```

**For 陈秀川 (segment-level annotations):**
- **Metric 1.3**: Segment Stenosis MAE
  ```
  MAE_seg = mean(|predicted_stenosis% - ground_truth%|) for each segment
  ```
  
- **Metric 1.4**: Grade Classification Accuracy
  ```
  Accuracy of stenosis_grade (normal/mild/moderate/severe/occluded)
  ```

#### Axis 2: **Completeness** (完整性)
评估 Agent 是否系统地覆盖所有必要信息。

**Metric 2.1**: Segment Coverage
```
Coverage = (predicted_segments ∩ ground_truth_segments) / ground_truth_segments
```

**Metric 2.2**: Negative Segment Detection (anti-hallucination)
```
对于 ground truth 标注为 "无狭窄" 的节段，Agent 是否正确识别
Precision = true_negative / (true_negative + false_positive)
```

**Metric 2.3**: View Selection Completeness
```
评估 Agent 是否使用了关键视角（基于临床 SOP）
- 优势判断：是否使用 LAO+CRA 视角
- LAD 评估：是否使用 RAO+CAU 视角
- RCA 评估：是否使用 LAO 视角
```

#### Axis 3: **Groundedness** (证据基础)
评估推理过程的可信度和可解释性。

**Metric 3.1**: Evidence Validity
```
Agent 引用的 key_frames 是否真实包含该病变
需要人工或模型辅助验证
```

**Metric 3.2**: View Consistency
```
对同一节段，不同视角的评估是否一致
Consistency = 1 - std(stenosis% across views for same segment)
```

**Metric 3.3**: Hallucination Detection
```
Agent 引用的 video_id 是否存在于输入中
Invalid_ref_rate = invalid_references / total_references
```

---

### 3.2 评估分层

#### Tier 1: 快速评估（CardioSYNTAX 50 cases）
- **输入**: .npy videos + JSON metadata
- **输出**: SYNTAX score 预测
- **评估**: MAE, Acc@5, Acc@10
- **用途**: 快速对比不同 Agent 的基准性能

#### Tier 2: 深度评估（陈秀川 + 精选 5-10 cases）
- **输入**: DICOM 或统一目录格式
- **输出**: 完整结构化报告
- **评估**: 完整三轴评估 + 人工审查
- **用途**: 详细分析 Agent 推理过程

---

## 4. Baseline 设定

根据用户反馈：**不比普通模型，只比 Coding Agent 和 PureLLM**

### 4.1 Baseline 配置

#### Baseline 1: **PureLLM** (None-Harness)
- **描述**: 直接将所有视频帧 + metadata 输入大模型，无 harness
- **实现**: 
  ```python
  prompt = f"Based on these {n} DSA videos, predict SYNTAX score: [frames_base64]"
  response = llm.generate(prompt)
  ```
- **目的**: 证明 Agent harness 的必要性

#### Baseline 2: **Cardiomni (Ours)**
- **描述**: 完整 4-stage SOP agent
- **配置**: 不同 base model (GPT-4o, Claude 3.5, Gemini 1.5)

#### Baseline 3: **Generic Coding Agent**
- **选项**:
  - **OpenHands** (原 OpenDevin): General-purpose coding agent
  - **SWE-Agent**: Specialized for software tasks, adapted to medical domain
  - **AutoGen**: Multi-agent conversation framework
- **适配**: 提供相同的工具包（视频读取、DICOM 解析、OpenCV 工具）

---

### 4.2 消融实验 (Ablations)

在 Cardiomni 框架内的消融：

1. **-Stage1**: 跳过优势判断，直接进入系统扫描
2. **-Stage2**: 跳过系统扫描，直接选择视角
3. **-View Selection**: 使用所有视角，不做选择
4. **-Tool Use**: 禁用专科模型工具，仅靠基础 VLM

---

## 5. 专科模型的定位

根据用户反馈：**专科模型作为工具，不作为对比 baseline**

### 5.1 作为工具 (Tools for Agent)

在 `tools/` 中提供：
- `segment_vessel()`: 血管分割工具
- `detect_stenosis()`: 狭窄检测工具  
- `estimate_qca()`: QCA 定量分析工具

Agent 可选择性调用这些工具。

### 5.2 作为 Upper-Bound Reference

单独报告专科模型的性能，作为 **理论上限**：
- 如果有专门训练的 SYNTAX score 预测模型，报告其在 CardioSYNTAX 上的性能
- 说明：这是 **单任务优化** 的结果，不具备通用推理能力

---

## 6. 数据准备计划

### 6.1 当前状态

- ✅ **陈秀川 DSA**: 已有完整 DICOM + 金标准标注
- ✅ **CardioSYNTAX 元数据**: all.json 已下载
- 🔄 **CardioSYNTAX 视频**: 正在下载 Part 6 + Part 9 (18.4 GB, 50 cases)

### 6.2 数据预处理流程

```python
# 伪代码
def prepare_cardiosyntax_task(study_uid):
    # 1. 从 all.json 提取元数据
    metadata = extract_metadata(study_uid)
    
    # 2. 从 .npy 文件加载视频
    videos = load_videos_from_parts(metadata['videos'])
    
    # 3. 转换为统一格式
    task = {
        'case_id': study_uid,
        'metadata': standardize_metadata(metadata),
        'videos': [convert_video(v) for v in videos],
        'ground_truth': {
            'syntax_score': metadata['syntax'],
            'syntax_left': metadata['syntax_left'],
            'syntax_right': metadata['syntax_right']
        }
    }
    
    return task
```

---

## 7. 与现有 Pipeline 的集成

### 7.1 当前 Pipeline 回顾

```python
# From CLAUDE.md
python -m pipeline.cli run --toml benchmark.toml --agent cardiomni
```

**四个交换轴**:
1. Base model (GPT-4o, Claude, Gemini)
2. Agent type (cardiomni, openhands, purellm)
3. Judge model (for auto-evaluation)
4. Task set (cardiosyntax50, chenxiuchuan, mixed)

### 7.2 任务集定义

在 `benchmark.toml` 中：

```toml
[tasks.cardiosyntax50]
type = "cardiosyntax"
data_dir = "CardiomniBench-VD/.raw_data/CardioSyntax/processed_50/"
manifest = "selected_50_studies_optimized.json"
evaluation = ["syntax_mae", "syntax_acc5", "syntax_acc10"]

[tasks.chenxiuchuan]
type = "dicom_case"
data_dir = "CardiomniBench-VD/.tmp/陈秀川-DSA/"
ground_truth = "DSA-流程.docx"  # 需转为 JSON
evaluation = ["segment_mae", "grade_accuracy", "completeness", "groundedness"]

[tasks.mixed]
include = ["cardiosyntax50", "chenxiuchuan"]
```

---

## 8. 下一步行动

1. **完成数据下载** (Task #9)
   - Part 6 + Part 9 下载完成
   - 解压并验证数据完整性

2. **数据预处理脚本** 
   - 实现 `prepare_cardiosyntax.py`
   - 转换 50 cases 为统一格式

3. **Ground Truth 准备**
   - 陈秀川: 将 `DSA-流程.docx` 转为 JSON
   - CardioSYNTAX: 已有 SYNTAX score

4. **Baseline 实现**
   - PureLLM baseline
   - Generic Coding Agent 适配

5. **评估脚本**
   - 实现三轴评估指标
   - 自动化评估 harness

---

## 附录 A: 临床术语对照

| 术语 | 全称 | 说明 |
|-----|------|------|
| LAD | Left Anterior Descending | 左前降支 |
| LCX | Left Circumflex | 左回旋支 |
| RCA | Right Coronary Artery | 右冠状动脉 |
| D1, D2 | Diagonal Branch 1/2 | 对角支 |
| OM1, OM2 | Obtuse Marginal 1/2 | 钝缘支 |
| RAO | Right Anterior Oblique | 右前斜位 |
| LAO | Left Anterior Oblique | 左前斜位 |
| CAU | Caudal | 足侧 |
| CRA | Cranial | 头侧 |

---

**End of Task Specification v1.0**
