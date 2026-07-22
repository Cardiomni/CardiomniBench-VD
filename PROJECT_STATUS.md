# CardiomniBench-VD 项目现状总结

**更新时间:** 2026-07-22  
**服务器:** H20 8×NVIDIA H20 (阿里云)  
**状态:** Pipeline 完成 ✅ | 等待第一个标注 Case

---

## 📊 项目进度总览

```
[████████████████████████░░░░░░░░] 60% Complete

✅ Pipeline 实现        100%  (19/19 tests passing)
✅ 评估指标系统        100%  (16 metrics registered)
✅ Docker+GPU 验证     100%  (H20 detected, mounts working)
✅ 文档完善           100%  (API + 对齐分析 + 行动计划)
⏳ 数据标注对齐         30%  (分析完成，等待专家标注)
❌ Cardiomni Agent      0%  (核心工作，待实现)
❌ 临床数据标注          0%  (0/20 cases，等待专家)
❌ Judge 验证           0%  (等待 validation cases)
```

---

## 🎯 今日完成工作 (2026-07-22)

### 1. Pipeline 实现完整验证 ✅

**核心组件 (全部通过测试):**
- Orchestrator: 端到端流程 (discover → run → score → aggregate)
- Agent Runners: mock / local / docker (GPU 支持)
- Judge Backends: mock / llm / cli
- Scoring System: automatic metrics + LLM judge grading
- Metric Registry: 16 objective metrics
- Config System: YAML + TOML 双格式支持

**测试覆盖:**
```bash
$ /opt/anaconda3/bin/python -m pytest tests/ -v
19 passed in 0.44s
```

**Docker + GPU 灰盒路径验证:**
```bash
$ cat runs/smoke_docker/rerun_0/case_smoke/gpu.txt
GPU 0: NVIDIA H20 (UUID: GPU-419a4afc...)  ✅
```

**Unified TOML Registry:**
```bash
$ /opt/anaconda3/bin/python -m pipeline.cli agents --toml benchmark.toml
cardiomni
local_script
mock
vlm_baseline  ✅
```

### 2. 数据标注规范对齐分析 ✅

**基于:** 冠脉造影数据标注规范与模型训练对齐会 (张冠兆 + Jiaming Ma)

**核心发现:**
- ✅ **已对齐:** 狭窄量化标准、DSA 优先原则、数据筛选标准
- ⚠️ **需调整:** core_views 支持、阴性段标注要求、报告模板映射

**关键决策:**
1. **四大核心体位 + SYNTAX 17 段** 两者结合 (满足临床标准 + 国际标准)
2. **DSA 优先但必须保留 CTA** (fusion_reasoning 占 20% 权重，是核心创新)
3. **阴性段必须标注** (新增 segment_coverage_recall metric + rubric criterion)

**Sample Case 分析:**
- 收到 `.tmp/陈秀川-DSA` (7 DICOM files)
- 创建解析脚本 `scripts/parse_dsa_metadata.py`
- 等待张冠兆提供诊断结论 + 报告模板

### 3. 文档体系完善 ✅

**新增文档 (4 份):**

1. **`docs/PIPELINE_COMPLETION.md`** (5.5k 字)
   - Pipeline 实现报告
   - 测试覆盖清单
   - 快速开始指南

2. **`docs/PIPELINE_API.md`** (6.8k 字)
   - 四大扩展点详细说明
   - 代码示例 (metric / agent / judge / config)
   - 最佳实践

3. **`docs/DATA_ANNOTATION_ALIGNMENT.md`** (7.2k 字)
   - 会议要点 vs Benchmark 设计对齐分析
   - Schema 调整 Roadmap (Phase 1-3)
   - 当前 Case 处理建议

4. **`docs/ANNOTATION_ACTION_PLAN.md`** (5.9k 字)
   - 按优先级排序的行动计划
   - 角色分工 (临床专家 vs AI/Pipeline)
   - 关键决策点梳理

**更新文档:**
- `HANDOFF.md`: 添加 Pipeline 完成状态
- `README.md`: 无需修改 (已完整)

### 4. 工具脚本开发 ✅

**创建:**
- `scripts/parse_dsa_metadata.py`: DICOM 元数据解析 + core_views 映射
- `pipeline/judge_validation.py`: Cohen's κ / Fleiss' κ judge 验证

---

## 🔧 技术栈验证

| 组件 | 状态 | 版本/配置 |
|------|------|-----------|
| Python | ✅ | 3.13.9 |
| Docker | ✅ | 26.1.3 |
| GPU | ✅ | 8× NVIDIA H20 (97GB each) |
| pytest | ✅ | 19 tests passing |
| Pipeline | ✅ | Mock/Local/Docker backends |
| TOML Registry | ✅ | 4 agents registered |
| Metrics | ✅ | 16 registered |
| Judge | ✅ | Mock/LLM/CLI backends |

---

## 📋 待办事项（按优先级）

### P0 - 本周必须完成

#### 👤 张冠兆 (临床专家)

1. **提供陈秀川 Case 诊断结论** (书面形式)
   - 狭窄位置 (如"右冠近端")
   - 狭窄程度 (百分比，如 75%)
   - 血管名称 (SYNTAX 命名)
   - **关键:** 包括阴性段 (如"LAD 未见明显狭窄")

2. **提供中山医院报告模板** (Word/PDF)
   - 用于验证 agent 输出格式
   - 将转换为 prediction.json schema 映射

3. **确认 DICOM 文件清洁性**
   - `.tmp/陈秀川-DSA` 的 7 个文件是否都是 pre-intervention?
   - 是否包含介入治疗图像需要剔除?

#### 🤖 Jiaming Ma (AI/Pipeline)

1. **安装 pydicom 并解析 Case**
   ```bash
   pip install pydicom
   python scripts/parse_dsa_metadata.py \
       --input .tmp/陈秀川-DSA/Exposure\ 7.5\ fps \
       --output .tmp/陈秀川-DSA/metadata_report.json \
       --template .tmp/陈秀川-DSA/gold_standard_template.yaml
   ```

2. **扩展 schema 支持 core_views** (Phase 1)
   - 修改 `tasks/task_template.yaml`
   - 更新 `pipeline/orchestrator.py` 验证逻辑
   - 新增 `segment_coverage_recall` metric
   - 添加 rubric criterion C015

3. **第一个 Case 的完整 pipeline 试跑**
   ```bash
   # 移动到 data/cases/
   mkdir -p data/cases/case_chxc_001
   cp -r .tmp/陈秀川-DSA/* data/cases/case_chxc_001/
   
   # 试跑
   /opt/anaconda3/bin/python -m pipeline.cli run \
       --toml benchmark.toml --agent mock --limit 1
   ```

### P1 - 下周完成

#### 👤 张冠兆

- 后续 10-20 个 CTA+DSA 配对病例数据
- 专家标注结果 (使用生成的模板工具)

#### 🤖 Jiaming Ma

- 报告模板映射工具 (等中山模板)
- 标注质控检查脚本 (`scripts/annotation_qc.py`)
- 标注 UI 原型 (可选，加速批量标注)

### P2 - 后续迭代

- Cardiomni Agent 实现 (`docker/agent/src/`)
- 构建 `cardiomni:latest` Docker 镜像
- Judge 验证 (需要 validation cases with expert labels)
- 完整 20 cases 标注 + 评测
- 论文实验 (fusion-lift, ablation studies)

---

## 🚀 如何继续推进

### Scenario 1: 张冠兆本周提供诊断结论

**时间线:**
- Day 1: 收到诊断 → 填充 `gold_standard.yaml`
- Day 2: Schema 扩展 (core_views) + metric 实现
- Day 3: Pipeline 试跑 + 验证 scoring 逻辑
- Day 4-5: 报告模板映射 + 文档更新

**里程碑:** 第一个 Case 完整跑通，可作为后续 19 cases 的模板

### Scenario 2: 等待时间较长 (>1 周)

**可并行推进的工作:**
1. **Schema Phase 1 扩展** (不依赖真实标注)
   - 添加 core_views 字段定义
   - 实现 segment_coverage_recall metric
   - 更新 rubric with C015 criterion

2. **Cardiomni Agent 骨架搭建** (使用 mock 数据测试)
   - DICOM loader (pydicom + windowing)
   - VLM call interface (Claude Opus 4.8)
   - Structured report generator (prediction.json writer)

3. **Judge 验证准备**
   - 准备 judge_validation.py 的使用文档
   - 设计 expert_grades.yaml schema
   - 招募 3 位专家准备 validation set

---

## 📚 文档索引

### 核心文档
- `README.md` — 项目概述 + Quickstart
- `HANDOFF.md` — 服务器交接文档 + 当前状态

### Pipeline 技术文档
- `docs/PIPELINE_COMPLETION.md` — 实现报告 + 测试覆盖
- `docs/PIPELINE_API.md` — 扩展指南 (4 swap axes)
- `docs/annotation_protocol.md` — 原始标注协议 (4-stage workflow)

### 数据标注对齐文档 (新增)
- `docs/DATA_ANNOTATION_ALIGNMENT.md` — 会议要点 vs Benchmark 对齐分析
- `docs/ANNOTATION_ACTION_PLAN.md` — 行动计划 + 角色分工

### 参考文档
- `docs/PROPOSAL.md` — Benchmark 设计提案 (原始需求)
- `rubrics/rubric_dimensions.yaml` — 6 维度评分标准
- `rubrics/examples/case_001_rubric.yaml` — 完整 rubric 示例 (24 criteria)

---

## 🎓 关键设计决策回顾

### 1. 为什么是 DSA 优先但必须保留 CTA?

**临床路径:**
- CTA: 初筛 (无症状、低风险人群)
- DSA: 金标准 (症状典型、CTA 重度狭窄 → 确诊 + 治疗)

**Benchmark 定位:**
- **核心创新:** CTA-DSA fusion reasoning (钙化 blooming 校正、CTO 综合判断)
- **差异化:** 单模态 benchmark 已有 (CTA-only, DSA-only)，fusion 是空白
- **权重分配:** fusion_reasoning 占 20%，仅次于 perception_accuracy (25%)

**结论:** DSA 是评测基准，但 CTA 是论文卖点，两者缺一不可。

### 2. 为什么 SYNTAX 17 段 + 四大核心体位?

**SYNTAX 17 段:**
- 国际标准，可复现
- 直接对应 SYNTAX Score 计算
- Benchmark 需要与国际接轨

**四大核心体位:**
- 中山医院临床标准
- 数据采集的实际约束
- 体位信息是 DICOM 元数据的关键

**整合方案:**
- `input.dsa.core_views`: 记录体位 (满足临床标准)
- `stage1b_dsa.segments`: 按 SYNTAX 段组织 (满足国际标准)
- `best_view` 字段: 链接两者 (segment → 最佳观察体位)

### 3. 为什么需要 segment_coverage_recall metric?

**会议要求:** "即使血管未见狭窄也需明确标注"

**临床理由:** 
- 阴性发现同样重要 (排除诊断)
- 遗漏阴性段 = incomplete workup = 临床风险

**Metric 定义:**
```python
Recall = len(gold_segments ∩ pred_segments) / len(gold_segments)
# Gold 中有 10 段 (含 3 段阴性), agent 只标了 7 段 → Recall = 0.7 → Grade C
```

**Rubric 惩罚:** Recall < 95% → 0 points (criterion C015)

---

## 💡 后续优化建议

### 短期 (1-2 周)
1. **半自动标注工具** — 从 DICOM 自动提取体位 + 预填模板 → 专家只需填狭窄程度
2. **标注质控 dashboard** — 实时显示标注完整性 (core_views 齐全性、阴性段覆盖率)
3. **Mock agent 智能化** — 生成符合分布的 mock prediction (非全 0/100)，测试 rubric 敏感度

### 中期 (1 个月)
1. **CTA 数据 pipeline** — 当前只有 DSA 解析脚本，需要 CTA (3D volume) 处理
2. **Fusion metrics 完善** — `compute_fusion_lift` 需要 single-modality baseline 数据
3. **Judge model selection** — 运行 judge_validation，选择最可靠的 LLM (κ > 0.8)

### 长期 (论文投稿前)
1. **Inter-annotator agreement** — 2-3 位专家独立标注 overlap set，报告 κ
2. **Solvability analysis** — 哪些 case 连人类专家都有分歧? (排除或标记为 controversial)
3. **Benchmark leaderboard** — 公开 test set 结果，定期更新 SOTA

---

## 📞 联系与协作

**技术问题 (Pipeline/Metrics/Schema):**
- GitHub Issues: `https://github.com/Cardiomni/CardiomniBench-VD/issues`
- 本地路径: `/mnt/aliyunsb/CardiomniBench-VD`

**临床标注 (诊断结论/报告模板):**
- 张冠兆 (临床专家)
- 数据格式参考: `docs/ANNOTATION_ACTION_PLAN.md`

**会议记录:**
- 冠脉造影数据标注规范与模型训练对齐会 (2026-07-22)
- 对齐分析: `docs/DATA_ANNOTATION_ALIGNMENT.md`

---

## ✅ Checklist (本周)

### 张冠兆
- [ ] 提供陈秀川 Case 诊断结论 (含阴性段)
- [ ] 提供中山医院报告模板
- [ ] 确认 DICOM 文件是否 pre-intervention

### Jiaming Ma
- [ ] 安装 pydicom
- [ ] 运行 parse_dsa_metadata.py
- [ ] Schema Phase 1 扩展 (core_views + segment_coverage_recall)
- [ ] 第一个 Case pipeline 试跑

---

**Pipeline Ready. Waiting for First Annotated Case.** 🚀
