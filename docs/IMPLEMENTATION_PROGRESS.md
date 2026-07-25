# CardiomniBench-VD 实现进度

**更新时间**: 2026-07-23 12:30  
**核心定位**: 类似 BioML-Bench 的智能体评估框架

---

## ✅ 今日完成

### 1. 任务规范设计
- **文档**: `docs/TASK_SPECIFICATION.md`
- **内容**:
  - 统一输入输出格式（兼容 DICOM 和 .npy）
  - 三轴评估框架（Correctness, Completeness, Groundedness）
  - Baseline 配置（PureLLM, Cardiomni, Generic Agents）
  - 专科模型定位（工具 + upper-bound reference）

### 2. 数据集精选
- **CardioSYNTAX**: 从 1,844 studies 中筛选 50 个高质量 cases
  - 条件: SYNTAX > 0, LCA ≥ 2 views, RCA ≥ 2 views
  - 优化策略: 集中在 Part 6 + 9（18.4 GB vs 145 GB）
  - SYNTAX 范围: 2.0 - 53.0
  - 保存: `selected_50_studies_optimized.json`

### 3. 算法库框架 ⭐
参照 BioML-Bench 设计，已实现：

**文件结构**:
```
algorithms/
├── README.md              ✅ 算法库说明
├── base.py                ✅ 标准接口（BaseAlgorithm）
├── __init__.py            ✅ 注册和发现系统
├── baselines/
│   └── pure_llm/          ✅ PureLLM baseline 实现
├── specialist_models/     📁 待添加（DeepCORO-CLIP 等）
└── tools/                 📁 待实现（DICOM parser 等）
```

**核心特性**:
- **统一接口**: 所有算法实现 `BaseAlgorithm`
- **标准化输出**: `AlgorithmOutput` 数据类
- **自动注册**: 通过 `register.py` 自动发现算法
- **智能体调用**: 
  ```python
  from algorithms import load_algorithm, list_available_algorithms
  
  # 查看可用工具
  algos = list_available_algorithms()
  
  # 加载并调用
  model = load_algorithm('pure_llm')
  result = model.predict(case_data)
  ```

**评估维度**:
- 准确性: SYNTAX MAE, 节段 MAE
- 效率: 执行时间, GPU 显存, Tokens 消耗
- 完整性: 节段覆盖率
- 可解释性: 证据链、推理过程

### 4. PureLLM Baseline
- **实现**: `algorithms/baselines/pure_llm/pure_llm.py`
- **功能**: 直接将视频帧 + metadata 输入 LLM，无 harness
- **目的**: 证明 Agent harness 的必要性

---

## 🔄 进行中

### 数据下载
- **Part 6**: 1.8 MB / 15 GB (0.01%)
- **Part 9**: 1.9 MB / 3 GB (0.06%)
- **状态**: 下载速度极慢，预计需要 10+ 小时

---

## 📋 待办事项

### 立即可做（不依赖下载）

1. **实现通用工具模块** ⏳
   ```python
   algorithms/tools/
   ├── dicom_parser.py      # DICOM 读取和解析
   ├── video_loader.py      # .npy/.avi 视频加载
   ├── projection_utils.py  # 投影角度计算
   └── syntax_calculator.py # SYNTAX score 计算规则
   ```

2. **整理陈秀川 Ground Truth** ⏳
   - 从 `DSA-流程.docx` 提取金标准诊断
   - 转为 JSON 格式
   - 位置: `.tmp/陈秀川-DSA/ground_truth.json`

3. **DeepCORO-CLIP 集成准备** ⏳
   - 创建 `algorithms/specialist_models/deepcoro_clip/`
   - 准备模型下载脚本（HuggingFace）
   - 实现标准接口适配器

4. **文档完善** ⏳
   - BioML-Bench 对比文档
   - 智能体调用示例
   - 评估协议详细说明

### 等待数据下载完成后

5. **数据预处理**
   - 解压 Part 6, 9
   - 验证 50 个 studies 的 .npy 文件
   - 转为统一目录结构

6. **陈秀川 DICOM 整理**
   - 转为统一格式
   - 添加到评估集

7. **运行第一轮测试**
   - PureLLM on 陈秀川 case
   - 验证评估流程

---

## 🎯 核心设计思想：类似 BioML-Bench

### BioML-Bench 模式
- 提供标准化的生物信息学工具（BLAST, BWA, etc.）
- 智能体自主选择工具、设计 pipeline
- 评估维度：准确性 + 效率 + 可解释性

### CardiomniBench-VD 模式
- 提供标准化的 DSA 分析工具（DeepCORO-CLIP, QCA, 视角分类器等）
- 智能体自主选择工具、设计多步推理流程
- 评估维度：Correctness + Completeness + Groundedness

### 关键差异
| 维度 | BioML-Bench | CardiomniBench-VD |
|------|-------------|-------------------|
| 数据类型 | 序列/表格 | 医学影像 + DICOM |
| 工具类型 | 命令行工具 | Python 模型 API |
| 输出 | 文本/文件 | 结构化 JSON |
| 评估重点 | 流程正确性 | 临床推理过程 |

---

## 📊 算法库规划

### Tier 1: Baselines（已实现）
- ✅ **PureLLM**: 直接 LLM，无 harness

### Tier 2: 专科模型（待实现）
- ⏳ **DeepCORO-CLIP**: SYNTAX score 预测（VLM）
- ⏳ **CardioSyntax**: Video Transformer baseline
- ⏳ **传统 QCA**: 边缘检测 + 几何测量

### Tier 3: 工具函数（待实现）
- ⏳ **View Classifier**: 投影角度分类
- ⏳ **Stenosis Detector**: 节段狭窄检测
- ⏳ **DICOM Parser**: 元数据提取

### Tier 4: Agent Harness（待实现）
- ⏳ **Cardiomni**: 4-stage SOP agent（我们的方法）
- ⏳ **OpenHands**: Generic coding agent 适配
- ⏳ **SWE-Agent**: Software engineering agent 适配

---

## 🔍 DeepCORO-CLIP 资源状态

### 已知信息
- **论文**: "DeepCORO-CLIP: A Foundation Model for Coronary Angiography Analysis"
- **数据**: DeepCORO-mini (1,000 cases, ~7,000 videos), 将发布到 Physionet
- **权重**: https://huggingface.co/collections/heartwise/deepcoro-clip
- **代码**: https://github.com/HeartWiseAI/DeepCORO_CLIP

### 当前状态
- ❌ GitHub 仓库不存在（repository not found）
- ❌ HuggingFace 网络访问受限（无法抓取）
- ⏳ 可能尚未公开，等待论文发表

### 下一步
1. 等待仓库公开
2. 或直接联系作者请求早期访问
3. 或使用 CardioSyntax 作为主要专科模型

---

## 📞 关键决策点

### Q1: 数据下载策略
- **当前**: wget 下载极慢（~10 小时）
- **选项**:
  - A. 继续等待下载完成
  - B. 暂时用陈秀川 1 case + 元数据开发
  - C. 寻找其他数据源（ARCADE, DCA1）

### Q2: DeepCORO-CLIP 替代方案
- **当前**: 代码和权重暂不可访问
- **选项**:
  - A. 等待公开后集成
  - B. 先用 CardioSyntax 替代
  - C. 实现简单的 CNN baseline

### Q3: 开发优先级
- **选项**:
  - A. 先完善算法库框架（工具、接口）
  - B. 先实现 Cardiomni Agent
  - C. 并行：工具库 + Agent 核心推理

---

## 💡 建议的下一步

1. **不等下载，先开发**:
   - 实现 DICOM parser 和 video loader
   - 整理陈秀川 ground truth
   - 用 1 case 验证完整流程

2. **暂缓 DeepCORO-CLIP**:
   - 等待公开或使用 CardioSyntax
   - 先实现简单的传统 baseline

3. **优先 Cardiomni Agent**:
   - 实现 4-stage SOP
   - 证明 structured reasoning 的价值

---

**下次同步**: 数据下载完成或明天继续开发工具库
