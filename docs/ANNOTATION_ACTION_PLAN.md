# 数据标注对齐 - 行动计划

**Date:** 2026-07-22  
**Status:** Pipeline 完成，等待第一个标注 Case  
**Sample Case:** `.tmp/陈秀川-DSA` (7 DICOM files)

---

## 一、关键对齐结论

### ✅ 已对齐（无需修改）

1. **狭窄程度量化标准** — 0-100 scale，MAE 评估
2. **DSA 优先原则** — fusion_reasoning 维度已覆盖钙化校正逻辑
3. **数据筛选标准** — pre-intervention only，schema 支持标记
4. **模型验证框架** — 报告模板对比机制已在 pipeline 中

### ⚠️ 需调整（优先级排序）

| 优先级 | 调整项 | 影响范围 | 工作量 |
|-------|--------|---------|-------|
| **P0** | 扩展 schema 支持 `core_views` | gold_standard.yaml | 1-2 天 |
| **P0** | 新增 metric: segment_coverage_recall | metric_registry.py | 半天 |
| **P0** | DICOM 元数据解析脚本 | scripts/ | 1 天 |
| **P1** | Rubric 增加"遗漏阴性段"惩罚 | rubric_dimensions.yaml | 半天 |
| **P1** | 报告模板映射工具 | scripts/ | 1-2 天 |
| **P2** | 标注质控检查脚本 | scripts/ | 1 天 |

---

## 二、当前 Case 处理流程

### Step 1: DICOM 元数据解析 (待执行)

**脚本已创建：** `scripts/parse_dsa_metadata.py`

```bash
# 需要先安装 pydicom
pip install pydicom

# 解析 DICOM 元数据
python scripts/parse_dsa_metadata.py \
    --input .tmp/陈秀川-DSA/Exposure\ 7.5\ fps \
    --output .tmp/陈秀川-DSA/metadata_report.json \
    --template .tmp/陈秀川-DSA/gold_standard_template.yaml
```

**输出：**
1. `metadata_report.json` — 所有 DICOM 文件的元数据清单 + 质控报告
2. `gold_standard_template.yaml` — 待填充的标注模板（已预填 core_views）

### Step 2: 专家标注 (等待张冠兆)

**需要提供的信息（按会议 Action Items）：**

1. **临床背景**：
   - 患者年龄、性别、症状（如"活动后胸闷"）
   - 危险因素（高血压、糖尿病等）

2. **诊断结论（书面形式）**：
   - 狭窄位置（如"右冠近端"）
   - 狭窄程度（百分比，如 75%）
   - 血管名称（SYNTAX 命名）
   - **关键：即使未见狭窄的段也需明确标注**

3. **中山医院报告模板**（Word/PDF）：
   - 用于验证 agent 输出格式的合理性
   - 将转换为 `prediction.json` schema 映射表

**标注工具：**
- 使用生成的 `gold_standard_template.yaml` 作为起点
- 填充 `stage1b_dsa.segments[]` 数组：
  ```yaml
  segments:
    - segment_id: "RCA_1"
      best_view: "rao30_cau0"  # 从 core_views 选择
      stenosis_percent: 75
      stenosis_grade: "severe"
      plaque_type: "calcified"
      # ...
    
    - segment_id: "LAD_6"
      stenosis_percent: 0  # 阴性发现也要标注
      stenosis_grade: "none"
  ```

### Step 3: Schema 验证 (自动化)

```bash
# 验证标注文件符合 schema
python -m pipeline.cli validate \
    --case-dir .tmp/陈秀川-DSA \
    --schema tasks/task_template.yaml

# 质控检查
python scripts/annotation_qc.py \
    --case .tmp/陈秀川-DSA/gold_standard.yaml \
    --check-coverage  # 验证四大核心体位齐全
    --check-completeness  # 验证阴性段是否标注
```

### Step 4: Pipeline 试跑

```bash
# 将 Case 移动到 data/cases/
mkdir -p data/cases/case_chxc_001
cp -r .tmp/陈秀川-DSA/* data/cases/case_chxc_001/

# 更新 splits.yaml
echo "test: [case_chxc_001]" >> data/splits.yaml

# 用 mock agent 试跑（验证 pipeline 能读取）
/opt/anaconda3/bin/python -m pipeline.cli run \
    --toml benchmark.toml \
    --agent mock \
    --split test \
    --limit 1
```

---

## 三、Schema 扩展实现计划

### Phase 1: core_views 支持 (本周)

**文件修改清单：**

1. **`tasks/task_template.yaml`** — 扩展 input.dsa schema
   ```yaml
   input:
     dsa:
       core_views:  # 新增字段
         - view_id: str
           description: str
           dicom_files: [str]
           positioner_primary_angle: float
           positioner_secondary_angle: float
           target_vessels: [str]
   ```

2. **`pipeline/orchestrator.py`** — 验证 core_views 存在性
   ```python
   def _validate_case(self, case: dict) -> bool:
       dsa = case.get("input", {}).get("dsa", {})
       core_views = dsa.get("core_views", [])
       if len(core_views) < 3:  # 至少 3 个体位
           logger.warning("case %s: insufficient core views", case["case_id"])
       return True
   ```

3. **`pipeline/metric_registry.py`** — 新增 segment_coverage_recall
   ```python
   def _segment_coverage_recall(gold: dict, pred: dict) -> float:
       """Recall: agent 是否标注了所有 gold 中出现的段（包括阴性）"""
       gold_segs = set(s["segment_id"] for s in _gold_dsa_segments(gold))
       pred_segs = set(s["segment_id"] for s in _pred_dsa_segments(pred))
       if not gold_segs:
           return 1.0
       return len(gold_segs & pred_segs) / len(gold_segs)
   
   REGISTRY["segment_coverage_recall"] = _segment_coverage_recall
   ```

4. **`rubrics/rubric_dimensions.yaml`** — 新增 criterion C015
   ```yaml
   - criterion_id: "C015"
     description: "Segment coverage: labeled all visible segments including negatives"
     evaluation_method: "automatic"
     metric: "segment_coverage_recall"
     grading_scale:
       type: "continuous"
       grades:
         - grade: "A"
           points: 5
           threshold: {min: 0.95, max: 1.0}
         - grade: "C"
           points: 0
           threshold: {min: 0.0, max: 0.95}
   ```

### Phase 2: 报告模板映射 (下周)

**等待输入：** 中山医院报告模板（张冠兆提供）

**实现：**
1. 解析报告模板 → 提取结构化字段映射
2. 创建 `scripts/map_report_to_prediction.py`
3. 生成 `docs/report_template_mapping.md` 文档

### Phase 3: 批量标注工具 (10-20 cases 后)

**目标：** 加速后续 10-20 个 case 的标注

**功能：**
- Web UI 标注界面（Flask + Vue）
- 自动从 DICOM 提取体位信息
- 实时 schema 验证
- 批量导出 gold_standard.yaml

---

## 四、待办事项（按角色）

### 👤 张冠兆 (临床专家)

**Priority 0 (本周):**
- [ ] 提供陈秀川 Case 的诊断结论（书面形式）
  - 狭窄位置、程度、血管名称
  - **包括阴性段**（如"LAD 未见明显狭窄"）
- [ ] 提供中山医院冠脉造影检查报告模板（Word/PDF）
- [ ] 确认 `.tmp/陈秀川-DSA` 的 7 个 DICOM 文件是否都是 pre-intervention

**Priority 1 (下周):**
- [ ] 后续 10-20 个 CTA+DSA 配对病例数据
- [ ] 专家标注结果（可使用生成的模板工具）

### 🤖 Jiaming Ma (AI/Pipeline)

**Priority 0 (已完成):**
- [x] Pipeline 实现完成（19 tests passing）
- [x] 数据标注对齐分析文档
- [x] DICOM 元数据解析脚本

**Priority 0 (本周):**
- [ ] 安装 pydicom: `pip install pydicom`
- [ ] 运行 `scripts/parse_dsa_metadata.py` 解析陈秀川 Case
- [ ] 扩展 schema 支持 core_views（Phase 1）
- [ ] 实现 segment_coverage_recall metric

**Priority 1 (下周):**
- [ ] 报告模板映射工具（等中山模板）
- [ ] 标注质控检查脚本
- [ ] 第一个 Case 的完整 pipeline 试跑

---

## 五、关键决策点

### 决策 1: 是否只用 DSA（不要 CTA）？

**会议倾向：** DSA 优先，但 CTA 是 benchmark 的核心差异化点

**建议：**
- **短期（10 个 case）：** 可以只标注 DSA，先验证 pipeline
- **长期（完整 benchmark）：** 必须有 CTA+DSA 配对，才能评估 fusion_reasoning
- **折中方案：** 前 5 个 case 只标 DSA，后 15 个标 CTA+DSA

**理由：** 
- `fusion_reasoning` 维度占 20% 权重，是论文核心创新点
- 只有 DSA 会导致钙化 blooming 校正、CTO 综合判断等 criteria 无法评分

### 决策 2: 四大核心体位 vs SYNTAX 17 段？

**会议标准：** 四大核心体位 + 四大血管

**Benchmark 设计：** SYNTAX 17 段 (国际标准)

**建议：** **两者结合**
- `input.dsa.core_views` 记录体位信息（满足会议标准）
- `stage1b_dsa.segments` 按 SYNTAX 17 段组织（国际对比）
- 每个 segment 链接到 `best_view`（建立映射关系）

**好处：**
- 满足中山医院标注习惯（四大体位）
- 保持国际对比能力（SYNTAX 分段是全球标准）
- Agent 可以学习"哪个体位最适合看哪个段"

### 决策 3: 负分惩罚的力度？

**会议提到：** 治疗建议需谨慎（术中情况会变）

**Rubric 当前设计：** source_reliability 可以给负分（hallucination penalty）

**建议：** **保持当前设计**
- 编造 FFR/IVUS 数据 → -20 points（严重）
- 不当治疗建议 → 0 points（不给分，但不扣分）
- Agent 可以输出狭窄程度，但治疗方案仅作参考建议

**理由：** 诊断错误可量化（对比 gold），治疗建议更依赖术中判断

---

## 六、关键里程碑

| 里程碑 | 交付物 | 预计时间 |
|-------|--------|---------|
| **M1: 第一个 Case 标注完成** | `case_chxc_001/gold_standard.yaml` | 本周五 |
| **M2: Pipeline 完整试跑** | `runs/case_chxc_001/evaluation.json` | 下周一 |
| **M3: Schema v2.0 发布** | core_views + segment_coverage | 下周三 |
| **M4: 10 个 Case 标注完成** | 10 cases in `data/cases/` | 2 周后 |
| **M5: 报告模板映射** | `docs/report_template_mapping.md` | 3 周后 |
| **M6: 20 个 Case + Judge 验证** | Judge κ > 0.8 | 4 周后 |

---

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| pydicom 解析失败（DICOM 格式异常） | 高 | 提供原始 DICOM 文件样本，手动验证 |
| 中山报告模板格式不标准 | 中 | 先用通用 schema，后期再映射 |
| 标注工作量超预期（20 case × 17 段） | 高 | 开发半自动标注工具，AI 预标注 + 专家修正 |
| CTA 数据暂时缺失 | 中 | 先跑 DSA-only pipeline，CTA 后补 |
| 体位数量不统一（不同医院） | 低 | 明确只用中山数据（会议已确认） |

---

**下一步 Action：**
1. 安装 pydicom 并运行 `scripts/parse_dsa_metadata.py`
2. 等待张冠兆提供陈秀川 Case 的诊断结论
3. 完成 Schema Phase 1 扩展（core_views）

**文档路径：**
- 详细分析: `docs/DATA_ANNOTATION_ALIGNMENT.md`
- Pipeline 完成报告: `docs/PIPELINE_COMPLETION.md`
- 解析脚本: `scripts/parse_dsa_metadata.py`
