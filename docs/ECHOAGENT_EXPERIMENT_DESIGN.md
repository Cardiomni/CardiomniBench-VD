# EchoAgent 实验设计详解与 DSA 对比

**基于 EchoAgent 论文的实验设计分析**

---

## 1. 数据集选择

### 1.1 EchoAgent 使用的数据集

#### Dataset 1: CAMUS
- **规模**: 1000 subjects, 1000 videos, 9268 frames
- **任务**: EF grading (Ejection Fraction 分级)
- **标注**: 左心室分割 + EF 值
- **划分**: 7:1:2 (train:val:test)
- **特点**: 
  - A2C (apical-2-chamber) 和 A4C (apical-4-chamber) 两种视图
  - 3 个等级：Normal (EF≥50%), Mildly reduced (40%-50%), Considerably reduced (EF<40%)

#### Dataset 2: MIMIC-EchoQA
- **规模**: 622 subjects, 622 videos, 51,194 frames
- **任务**: Multi-structure QA (多结构问答)
- **覆盖**: 48 distinct views, 14 cardiac structures
- **问题类型**: 
  - Pericardium (心包): 7 anatomical groups
  - Aortic valve (主动脉瓣): Mitral valve, Ventricles (心室)
  - Atria (心房): Vessels, Others
- **特点**: Multiple-choice clinical questions（多选题格式）

### 1.2 对应到 DSA 领域

**DSA 是否有类似数据集？**

✅ **有类似的，但需要组合**：

| EchoAgent 数据集 | DSA 对应数据集 | 相似度 | 差异 |
|------------------|---------------|--------|------|
| CAMUS (1000 cases, single task) | **CardioSYNTAX** (1844 studies, SYNTAX score) | ⭐⭐⭐⭐ | Echo 是单帧分割，DSA 是视频序列 |
| MIMIC-EchoQA (622 cases, multi-structure QA) | **ARCADE** (1500 images, 26 segments) + 陈秀川 (1 case, 完整标注) | ⭐⭐⭐ | Echo 有 QA 格式，DSA 只有节段标注 |

**DSA 数据集现状**:
1. **CardioSYNTAX** (我们已选用):
   - 1,844 studies, 14,219 videos
   - ✅ SYNTAX score（整体评分）
   - ❌ 无节段级狭窄标注
   - ✅ 投影角度标注
   - 类似 CAMUS（单一任务）

2. **ARCADE**:
   - 1,500 XCA frames
   - ✅ 26 节段标注（SYNTAX segments）
   - ❌ 数据集未找到（论文有但 Zenodo 链接失效）
   - 类似 MIMIC-EchoQA（多结构）

3. **陈秀川 DSA**（我们独有）:
   - 1 case, 7 DICOM files
   - ✅ 完整节段级狭窄标注（LAD 80%, D1 30%/60%）
   - ✅ 临床 SOP 文档
   - ✅ 金标准诊断
   - **价值**: 类似 MIMIC-EchoQA 的"深度 case study"

**策略**:
```
我们的双数据源 = EchoAgent 的双数据集策略
├── CardioSYNTAX 50 cases → 对应 CAMUS (广度评估)
│   └── 任务: SYNTAX score prediction
└── 陈秀川 1 case → 对应 MIMIC-EchoQA (深度评估)
    └── 任务: Segment-level stenosis + reasoning trace
```

---

## 2. Pipeline 设计

### 2.1 EchoAgent 的实验 Pipeline

#### Task 1: Single-structure (EF Grading)

**输入**: Echo 视频 → 自动识别 A2C/A4C 视图

**Pipeline**:
```
1. Perceptual Layer:
   - EchoPrime 解析视频 → 识别标准视图
   
2. Operational Layer:
   - Anatomy Segmentor → LV segmentation mask
   - Quantification → 计算 EF (ejection fraction)
   
3. Functional Layer:
   - Partition → 分类为 Normal/Mildly/Considerably reduced
   
4. Orchestrated Reasoning:
   - 构建推理图: [Video] → [LV mask] → [EF=X%] → [Grade=Y]
   - 置信度评估
```

**对应到 DSA - Cardiomni Pipeline**:
```
任务: SYNTAX Score Prediction

1. Perceptual Layer:
   - ProjectionClassifier → 识别 RAO/LAO, CRA/CAU
   - ArteryClassifier → LCA/RCA
   
2. Operational Layer:
   - SAM-VMNet → 冠脉分割
   - StenosisDetector → 狭窄检测
   
3. Functional Layer:
   - SyntaxCalculator → 计算 SYNTAX score
   
4. Orchestrated Reasoning:
   - [Multiple views] → [Stenosis per segment] → [SYNTAX=X]
```

#### Task 2: Multi-structure (EchoQA)

**输入**: Echo 视频 + 诊断查询（"Is there any abnormality of the echocardiography?"）

**Pipeline**:
```
1. Knowledge Retrieval:
   - 检索相关知识原语（"abnormality" → 7 major anatomical groups）
   
2. Dynamic Action Sequence:
   Step 1: Identify all visible structures → [Pericardium, LV, RV, ...]
   Step 2: For each structure → run Anatomy Segmentor
   Step 3: Compare with normal range (from knowledge base)
   Step 4: Aggregate evidence → Final diagnosis
   
3. Adaptive Reasoning:
   - If confidence < threshold → request additional views
   - If contradiction detected → trigger alternative pathway
```

**对应到 DSA - Cardiomni 4-Stage SOP**:
```
输入: DSA 多视角视频 + 查询（"Calculate SYNTAX score"）

Pipeline:
1. Knowledge Retrieval:
   - 检索 "SYNTAX score" → 17 segments, dominance, lesion criteria
   
2. Dynamic Action Sequence:
   Stage 1: Dominance Check
     → Select LAO+CRA views → DominanceDetector
   
   Stage 2: Systematic Scan
     → For each segment → check coverage
   
   Stage 3: View Selection
     → For LAD proximal → select RAO+CAU, AP cranial
     → For RCA → select LAO, RAO
   
   Stage 4: Lesion Assessment
     → Run StenosisDetector on selected views
     → Aggregate multi-view evidence
     → Calculate SYNTAX score
   
3. Adaptive Reasoning:
   - If segment not covered → report "insufficient views"
   - If multi-view results conflict → weighted voting
```

### 2.2 关键设计对比

| 维度 | EchoAgent | Cardiomni (我们) |
|------|-----------|------------------|
| **输入** | Echo 视频（单模态） | DSA 视频（单模态，但多视角依赖强） |
| **知识库** | 48 structures, 14 categories | 17 segments, SYNTAX rules |
| **工具层次** | 3 层（感知/操作/功能） | 3 层（视角识别/测量/评分） |
| **推理模式** | 动态推理图 + 自适应 | 4-stage SOP + 证据聚合 |
| **输出** | QA 答案 + 推理图 | SYNTAX score + 节段狭窄 + 证据链 |

---

## 3. 对比实验设计

### 3.1 EchoAgent 的 Baseline 矩阵

#### 类别 1: Task-specific Models（专科模型）

**目的**: 证明 end-to-end agentic system 优于单任务模型

**方法**:
- **H2former**: Echo 分割专用网络
- **MemSAM**: Memory-based SAM for Echo
- **EchoONE**: Echo 分割统一模型
- **OmnimaNet**: 视图识别

**结果**:
- 专科模型在 EF grading 上: 74% acc
- EchoAgent: 88% acc
- **Claim**: "专科模型缺乏推理能力，只能做单一任务"

#### 类别 2: General-purpose MLLMs（通用大模型）

**目的**: 证明领域知识（"Mind"）的必要性

**方法**:
- **LLaVA-Med**: 医学视觉语言模型
- **Qwen2.5-7B-VL**: 通用 VLM
- **Deepseek-VL2**: 最新 VLM
- **GPT-5**: 最强闭源模型

**结果**:
- 通用 MLLM 在 EchoQA 上: 最高 74.39% (Deepseek-VL2)
- EchoAgent: 84.15% acc
- **Claim**: "通用模型缺乏领域知识，无法可靠诊断"

#### 类别 3: "E-H-M" Workflows（不同协同模式）

**目的**: 证明完整 "Eyes-Hands-Minds" 协同的必要性

**消融实验**:
| 配置 | Eyes (感知) | Hands (操作) | Minds (推理) | EF Grading Acc | EchoQA Acc |
|------|-------------|--------------|--------------|----------------|------------|
| Baseline | ✓ | ✗ | ✗ | 55.00% | 43.57% |
| Baseline+EDC | ✓ | ✗ | ✓ | 50.00% | 51.43% |
| Baseline+HC | ✓ | ✓ | ✗ | 73.00% | 59.97% |
| **EchoAgent (E+H+M+OR)** | ✓ | ✓ | ✓ | **88.00%** | **79.42%** |

**结论**:
- 仅 Eyes: 不够（55%）
- Eyes+Minds: 仍不足（50%）→ 缺乏精确测量
- Eyes+Hands: 较好（73%）→ 但缺乏推理
- **完整协同**: 最佳（88%）

### 3.2 对应到 Cardiomni 的 Baseline 设计

#### 类别 1: Specialist Models（专科模型作为工具）

**目的**: 专科模型可以作为工具，但无法完成端到端任务

**方法**:
| 模型 | 任务 | 用途 | 状态 |
|------|------|------|------|
| SAM-VMNet | 分割+狭窄检测 | 工具（Hands） | ✅ 已集成 |
| CardioSyntax | SYNTAX 预测 | 对比 baseline | 🔄 权重待下载 |
| CM-UNet | 冠脉分割 | 工具（Hands） | ✅ 已集成 |
| MesserMMP | SYNTAX 预测 | 对比 baseline | 🔄 下载中 |

**实验设计**:
```python
# 单独运行专科模型
result_cardiosyntax = CardioSyntax.predict(all_videos)
# 缺点：无视角选择，无多步推理

# Cardiomni 调用专科模型作为工具
result_cardiomni = Cardiomni.predict(all_videos)
# → Stage 3 选择关键视角
# → Stage 4 调用 SAM-VMNet 测量
# → 综合多视角证据
```

**Claim**: "专科模型提供精确测量（Hands），但 Cardiomni 提供系统性推理（Minds）"

#### 类别 2: PureLLM（通用大模型直接推理）

**目的**: 证明结构化推理优于 end-to-end MLLM

**方法**:
- **GPT-4o**: 直接输入所有视频帧
- **Claude 3.5 Sonnet**: 直接输入
- **Gemini 1.5 Pro**: 直接输入

**实验设计**:
```python
# PureLLM: 直接输入所有帧
prompt = f"Given {n} DSA videos, predict SYNTAX score: [frames_base64]"
result = llm.generate(prompt)

# 缺点：
# 1. 无视角选择（可能看了100个视频帧，但没找到关键的 RAO+CAU）
# 2. 无精确测量（只能"目测"狭窄）
# 3. 推理过程不可追溯
```

**Claim**: "PureLLM 缺乏领域知识和操作能力"

#### 类别 3: Agent Ablation（智能体消融）

**目的**: 证明 4-stage SOP 的必要性

**消融实验**（参考 EchoAgent 的 E-H-M 消融）:

| 配置 | Stage 1 (Dominance) | Stage 2 (Scan) | Stage 3 (Select) | Stage 4 (Assess) | SYNTAX MAE |
|------|---------------------|----------------|------------------|------------------|------------|
| Baseline (PureLLM) | ✗ | ✗ | ✗ | ✗ | ~15.0 |
| + Stage 4 only | ✗ | ✗ | ✗ | ✓ | ~10.0 |
| + Stage 3+4 | ✗ | ✗ | ✓ | ✓ | ~7.0 |
| + Stage 2+3+4 | ✗ | ✓ | ✓ | ✓ | ~5.5 |
| **Full Cardiomni** | ✓ | ✓ | ✓ | ✓ | **~4.5** |

**结论**:
- 每个 stage 都有贡献
- Stage 3 (View Selection) 最关键（从 10.0 降到 7.0）
- 完整 4-stage 达到接近 CardioSyntax (MAE 4.2) 的性能

---

## 4. 评估指标设计

### 4.1 EchoAgent 的指标

#### Task 1: EF Grading

**主指标**: Accuracy (3-class classification)

**辅助指标**: 
- G-mean (geometric mean)：平衡不同类别
- AUROC：分类能力

**为什么选这些**:
- Acc: 临床标准（准确率）
- G-mean: 处理类别不平衡（Normal vs Reduced 比例不均）
- AUROC: 评估模型的辨别能力

#### Task 2: EchoQA

**主指标**: Accuracy (multi-choice QA)

**辅助指标**:
- Per-group Acc：每个解剖结构的准确率
- Per-view Acc：每种视图的准确率

**可解释性指标**:
- Reasoning graph depth：推理图深度
- Evidence consistency：证据一致性

### 4.2 对应到 Cardiomni 的三轴评估

根据 EchoAgent，我们设计：

#### Axis 1: Correctness（正确性）

**对应 EchoAgent 的 Accuracy**

**指标**:
```
1. SYNTAX Score MAE
   - 对应 EchoAgent 的 EF grading accuracy
   - CardioSYNTAX 50 cases 上评估
   
2. Segment Stenosis MAE
   - 对应 EchoAgent 的 per-group accuracy
   - 陈秀川 case 上评估（有节段级标注）
   
3. Grade Classification Acc
   - Normal/Mild/Moderate/Severe/Occluded
   - 对应 EchoAgent 的 3-class EF grading
```

#### Axis 2: Completeness（完整性）

**对应 EchoAgent 的 Multi-structure 能力**

**指标**:
```
1. Segment Coverage
   - 类似 EchoAgent 评估 "是否检查了所有心脏结构"
   - Cardiomni 是否系统评估了所有 17 节段
   
2. View Selection Completeness
   - 类似 EchoAgent 的 "是否使用了所有必要视图"
   - Cardiomni 是否选择了临床 SOP 要求的关键视角
```

#### Axis 3: Groundedness（证据基础）

**对应 EchoAgent 的 Reasoning Graph**

**指标**:
```
1. Evidence Validity
   - 类似 EchoAgent 的推理图节点验证
   - Agent 引用的 key_frames 是否真实包含病变
   
2. Reasoning Depth
   - 对应 EchoAgent 的 reasoning graph depth
   - Cardiomni 的推理步骤数
   
3. Hallucination Rate
   - 类似 EchoAgent 的 evidence consistency
   - Agent 是否引用了不存在的视频或帧
```

---

## 5. DSA 领域的数据集现状总结

### 5.1 存在的 DSA 数据集

| 数据集 | 规模 | 标注 | 公开性 | 类似 EchoAgent 的哪个 |
|--------|------|------|--------|------------------------|
| **CardioSYNTAX** | 1,844 studies | SYNTAX score | ✅ Zenodo | CAMUS (单任务) |
| **ARCADE** | 1,500 images | 26 segments | ❌ 链接失效 | MIMIC-EchoQA (多结构) |
| **DCA1** | 130 images | Binary stenosis | ❌ 未找到 | - |
| **ImageCAS** | ~1000 CTA volumes | 3D segmentation | ⚠️ 需申请 | - |
| **DeepCORO-mini** | 1,000 cases (~7k videos) | - | 🔄 即将发布 (Physionet) | MIMIC-EchoQA |

### 5.2 我们的策略

**组合现有资源 + 独特优势**:

1. **CardioSYNTAX 50 cases** → 对应 CAMUS
   - 广度评估
   - SYNTAX score prediction
   
2. **陈秀川 1 case** → 对应 MIMIC-EchoQA 的深度
   - 完整节段标注
   - 临床 SOP 对齐
   - Case study 展示
   
3. **工具库（6 个开源方法）** → 对应 EchoAgent 的 HC toolkit
   - SAM-VMNet, CM-UNet 等
   - 作为 Agent 可调用工具

**优势**:
- 陈秀川 case 是**独有的完整标注**（其他数据集都缺乏这种深度）
- CardioSYNTAX 提供**统计显著性**
- 工具库提供**可复现性**

---

## 6. 关键启示

### 6.1 从 EchoAgent 学到什么

1. **三层架构是通用模式**:
   - Eyes (感知) = 视角识别
   - Hands (操作) = 测量工具
   - Minds (推理) = 知识库 + 动态推理

2. **对比实验要全面**:
   - 专科模型（证明需要推理）
   - 通用 MLLM（证明需要领域知识）
   - 消融实验（证明每个组件的贡献）

3. **双数据集策略**:
   - 单任务数据集（快速评估，统计显著）
   - 多任务数据集（深度评估，可解释性）

4. **Claim 要清晰**:
   - Gap: 现有方法的缺陷
   - Solution: 我们的独特设计
   - Evidence: 全面的实验验证

### 6.2 Cardiomni 的实施计划

基于 EchoAgent 的经验，我们的实验设计：

```
实验 1: Single-task (CardioSYNTAX 50 cases)
├── Baseline 1: CardioSyntax (专科模型)
├── Baseline 2: PureLLM (GPT-4o, Claude)
├── Baseline 3: Generic Agents (OpenHands)
└── Cardiomni (完整 4-stage)
    → 评估: SYNTAX MAE, Acc@5, Acc@10

实验 2: Multi-structure (陈秀川 case)
├── Task: 节段级狭窄评估
├── 评估: 三轴 (Correctness, Completeness, Groundedness)
└── Case study: 展示完整推理图

实验 3: Ablation (消融)
├── Baseline: PureLLM
├── + Stage 4 only
├── + Stage 3+4
├── + Stage 2+3+4
└── Full Cardiomni (4-stage)

实验 4: Tool Analysis (工具分析)
├── 不同工具组合的效果
└── 证明工具库的价值
```

---

**结论**: EchoAgent 提供了一个完整的 Agent 设计和评估范式，我们可以直接借鉴其 "Eyes-Hands-Minds" 架构和全面的对比实验设计，适配到 DSA 领域。

**关键适配点**:
1. Echo 的 48 structures → DSA 的 17 segments
2. Echo 的 A2C/A4C views → DSA 的 RAO/LAO/CRA/CAU projections
3. Echo 的 EF grading → DSA 的 SYNTAX score
4. Echo 的 QA format → DSA 的 segment-level stenosis

**数据策略**: CardioSYNTAX (广度) + 陈秀川 (深度) = CAMUS + MIMIC-EchoQA
