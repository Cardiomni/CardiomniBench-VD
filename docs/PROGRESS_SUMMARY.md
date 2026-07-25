# CardiomniBench-VD 进度总结

**日期**: 2026-07-23  
**状态**: 任务设计完成，数据准备中

---

## ✅ 已完成

### 1. 项目定位明确
- **核心贡献**: Cardiomni agent（DSA 多视角推理）
- **Benchmark 角色**: 评估工具，非独立贡献
- **数据策略**: DSA-only（CTA 融合移至 future work）

### 2. 数据集选择
- **陈秀川 DSA**: 1 个高质量 DICOM case，完整金标准标注
- **CardioSYNTAX**: 50 个精选 cases
  - 从 1,844 studies 中筛选
  - 条件: SYNTAX score > 0, LCA ≥ 2 views, RCA ≥ 2 views
  - 策略优化: 集中在 Part 6 + Part 9（18.4 GB，而非全部 145 GB）
  - SYNTAX score 范围: 2.0 - 53.0
  - 平均每个 case 9 个视角

### 3. 任务规范设计 ✅
文档: `docs/TASK_SPECIFICATION.md`

**核心设计**:
- **输入格式**: 统一目录结构（兼容 DICOM 和 .npy）
- **输出格式**: JSON schema（SYNTAX score + 节段狭窄 + 证据链）
- **三轴评估**:
  - Correctness: SYNTAX MAE, 节段狭窄 MAE
  - Completeness: 节段覆盖率, 视角选择完整性
  - Groundedness: 证据有效性, 一致性, 反幻觉
- **Baseline 配置**:
  - PureLLM (None-Harness)
  - Cardiomni (Ours)
  - Generic Coding Agents (OpenHands/SWE-Agent)
- **专科模型定位**: 作为工具 + upper-bound reference，不作为对比 baseline

### 4. 论文背景完善 ✅
- 已将临床路径（三门诊断流程）写入 Paper Introduction
- 解释了为什么 DSA 多视角推理是关键临床问题

---

## 🔄 进行中

### 数据下载 (Task #9)
- **状态**: 下载中
- **目标**: Part 6 (15.2 GB) + Part 9 (3.2 GB)
- **当前**: ~3-5 MB 已下载
- **预计**: 需要数小时（取决于 Zenodo 速度）

---

## 📋 待办事项

### 短期（本周）

1. **完成数据下载和验证**
   - 等待 Part 6, 9 下载完成
   - 解压并验证数据完整性
   - 验证 50 个 studies 的 .npy 文件可访问

2. **数据预处理**
   ```python
   # 实现脚本
   scripts/prepare_cardiosyntax.py
   scripts/prepare_chenxiuchuan.py
   ```
   - 将 CardioSYNTAX 转为统一格式
   - 将陈秀川 DICOM + Word 文档转为 JSON ground truth

3. **Ground Truth 标注整理**
   - 陈秀川: 提取 `DSA-流程.docx` 中的金标准诊断为 JSON
   - CardioSYNTAX: 已有 SYNTAX score，无需额外处理

4. **PureLLM Baseline 实现**
   - 实现最简单的直接输入 baseline
   - 验证评估流程可运行

### 中期（下周）

5. **Cardiomni Agent 核心实现**
   - Stage 1: Dominance 判断
   - Stage 2: Systematic scan
   - Stage 3: View selection
   - Stage 4: Lesion assessment

6. **评估 Harness 实现**
   - 三轴指标自动计算
   - 结果可视化
   - 对比报告生成

7. **Generic Coding Agent 适配**
   - OpenHands 适配到医学影像任务
   - 提供统一工具接口

### 长期（两周后）

8. **完整实验运行**
   - CardioSYNTAX 50 cases × 所有 baselines
   - 陈秀川深度评估
   - 消融实验

9. **论文实验章节撰写**
   - Results tables
   - Qualitative analysis
   - Case studies

---

## 📊 当前文件结构

```
CardiomniBench-VD/
├── docs/
│   ├── TASK_SPECIFICATION.md       ✅ 任务规范
│   └── PROGRESS_SUMMARY.md         ✅ 本文档
│
├── .raw_data/
│   └── CardioSyntax/
│       ├── all.json                ✅ 完整元数据
│       ├── selected_50_studies_optimized.json  ✅ 精选 50 cases
│       └── downloaded_parts/       🔄 下载中
│           ├── 6.zip               🔄 1.9 MB / 15.2 GB
│           └── 9.zip               🔄 2.8 MB / 3.2 GB
│
├── .tmp/
│   └── 陈秀川-DSA/                 ✅ DICOM + 金标准
│       ├── IM000000 ... IM000006   ✅ 7 个 DICOM 文件
│       ├── view_summary.json       ✅ 投影角度
│       └── DSA-流程.docx           ⏳ 待转为 JSON
│
├── pipeline/                       ✅ 已实现
│   ├── cli.py
│   ├── evaluator.py
│   └── ...
│
└── scripts/                        ⏳ 待实现
    ├── prepare_cardiosyntax.py
    └── prepare_chenxiuchuan.py
```

---

## 🎯 核心创新点总结

### 概念创新
1. **Process-level benchmark**: 评估多步推理过程，非单一预测结果
2. **三轴评估框架**: Correctness + Completeness + Groundedness
3. **Agent-as-protagonist**: Benchmark 服务于 Agent 评估，非对等贡献

### 结果创新
1. **首个 DSA 多视角推理 benchmark**: 现有工作聚焦单模态单视角
2. **与临床 SOP 对齐**: 评估维度直接对应临床诊断流程
3. **Agent harness 对比**: 证明 structured reasoning 优于 end-to-end

### 工作完整度
1. **双数据源策略**: 深度（陈秀川）+ 广度（CardioSYNTAX 50）
2. **完整 baseline 矩阵**: PureLLM + Generic Agents + Specialist Models
3. **可复现**: 公开数据 + 开源代码 + 详细规范

---

## 📝 备注

- **数据下载时间**: Zenodo 速度不稳定，建议过夜下载
- **磁盘空间**: 确保至少 25 GB 可用空间（18.4 GB 压缩包 + 解压后）
- **BibTeX 待整理**: Paper 中的 4 个引用需从注释转移到 .bib 文件

---

**下次同步节点**: 数据下载完成后，开始数据预处理和 ground truth 整理
