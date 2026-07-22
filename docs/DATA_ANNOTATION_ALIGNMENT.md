# CardiomniBench-VD 数据标注规范对齐分析

**Date:** 2026-07-22  
**Based on:** 冠脉造影数据标注规范与模型训练对齐会 (张冠兆、Jiaming Ma)  
**Sample Case:** 陈秀川-DSA (7 DICOM files in `.tmp/`)

---

## 一、会议要点与 Benchmark 设计的对齐

### 1.1 数据筛选标准 ✅ 已对齐

**会议决策：**
- **仅关注造影前期图像**（通常前 10 张以内）
- **剔除介入治疗图像**（导丝、支架、球囊）
- **仅保留用于评估血管基础狭窄程度的原始图像**

**Benchmark 设计对齐：**
- ✅ `gold_standard.yaml` 中 `input.dsa.exclude_interventional = true` 标记
- ✅ 标注协议 (`docs/annotation_protocol.md`) 明确要求"pre-intervention frames only"
- ✅ Agent 不会看到治疗后图像，评估的是 baseline stenosis

**当前 Case (`陈秀川-DSA`)：**
- 7 个 DICOM 文件 (IM000000-IM000006)
- 文件名 "Exposure 7.5 fps" 表示这是造影序列
- **需确认：** 这 7 个文件是否都是 pre-intervention？是否包含多个体位？

**Action Items:**
1. 解析 DICOM 元数据确认每个文件对应的体位 (RAO/LAO, CAU/CRA 角度)
2. 如果 7 个文件包含治疗图像，按会议标准剔除
3. 按体位分组（如 IM000000-001 是右冠 RAO 30°，IM000002-003 是左冠 LAO 60° 等）

---

### 1.2 体位标注策略 ⚠️ 需调整

**会议决策：**
- **四大核心体位对应四大血管**的标准化标注
- 中山医院标准：左 6 右 3 体位（其他中心可能不同）
- 优先建立在中山医院标准化数据基础上

**Benchmark 当前设计：**
- ❓ `gold_standard.yaml` schema 未明确约定"四大核心体位"
- ❓ 当前 `stage1b_dsa.segments` 是按血管段 (segment_id) 组织，未按体位组织
- ❓ 未要求"即使血管未见狭窄也需明确标注"

**差距分析：**

| 会议标准 | Benchmark 当前设计 | 需要调整 |
|---------|------------------|---------|
| 四大核心体位 + 四大血管 | 按 SYNTAX 17 段组织 | ✅ 需在 `input.dsa` 增加 `core_views` 字段 |
| 未见狭窄也需标注 | 可选字段 | ✅ 需在 rubric 增加"遗漏阴性段"惩罚 |
| 体位元数据 (RAO/CAU) 是核心 | 未要求提取 | ✅ 需在数据处理时验证元数据可用性 |

**提议的 Schema 调整：**

```yaml
# gold_standard.yaml (新增字段)
input:
  dsa:
    core_views:  # 四大核心体位 (中山标准)
      - view_id: "rca_rao30_cau0"
        description: "右冠 RAO 30°"
        dicom_files: ["IM000000"]  # 对应文件
        positioner_primary_angle: 30.0  # RAO (+) / LAO (-)
        positioner_secondary_angle: 0.0  # CRA (+) / CAU (-)
        target_vessels: ["RCA"]
        
      - view_id: "lca_rao30_cra25"
        description: "左冠 RAO 30° CRA 25°"
        dicom_files: ["IM000002"]
        positioner_primary_angle: 30.0
        positioner_secondary_angle: 25.0
        target_vessels: ["LM", "LAD", "LCX"]
      
      # ... 其他核心体位

stage1b_dsa:
  segments:  # 保持按血管段组织（SYNTAX 标准）
    - segment_id: "RCA_1"
      best_view: "rca_rao30_cau0"  # 链接到 core_views
      stenosis_percent: 50
      # ...
    
    - segment_id: "LAD_6"
      best_view: "lca_rao30_cra25"
      stenosis_percent: 0  # 明确标注"未见狭窄"
      # ...
```

**Rubric 调整：**

```yaml
# rubrics/rubric_dimensions.yaml (新增 criterion)
- criterion_id: "C015"
  description: "Completeness: labeled all visible segments including negative findings"
  evaluation_method: "automatic"
  metric: "segment_coverage_recall"  # 新增 metric
  grading_scale:
    grades:
      - grade: "A"
        points: 5
        threshold: {min: 0.95, max: 1.0}  # 覆盖 ≥95% 血管段
      - grade: "C"
        points: 0
        threshold: {min: 0.0, max: 0.95}
```

---

### 1.3 狭窄程度量化标准 ✅ 已对齐

**会议标准：**
- 数值越大 → 狭窄越严重
- 0 = 无狭窄，100 = 完全闭塞
- 计算逻辑：狭窄后直径 / 原始直径

**Benchmark 设计：**
- ✅ `stenosis_percent` 字段定义为 0-100
- ✅ Metric `stenosis_mae` 计算 MAE
- ✅ Rubric 对 MAE > 20% 或分档错 ≥2 档进行惩罚

**对齐验证：**
```python
# evaluation/metrics/perception_metrics.py::compute_stenosis_mae
# 已实现：计算 gold vs pred 的 stenosis_percent 差值
# ✅ 符合会议标准
```

---

### 1.4 DSA 优先原则 ✅ 核心设计已对齐

**会议决策：**
- **DSA 是金标准**，直接指导治疗
- CTA 主要用于初筛和验证
- 模型训练应**优先基于 DSA 数据**

**Benchmark 设计对齐：**
- ✅ `fusion_reasoning` 维度 (20% 权重) 评估 CTA-DSA 整合能力
- ✅ 钙化 blooming 校正：重度钙化时应采用 DSA 定量（不是 CTA）
- ✅ Rubric `C040` 明确惩罚"重度钙化时未采用 DSA，错用 CTA 高估值"

**会议补充：CTA 假阳性识别能力**
- ✅ Benchmark 已覆盖：`fusion_reasoning` 评估 agent 是否能识别 CTA 过估（钙化影）
- ✅ Gold standard 包含 `stage2_fusion.blooming_correction` 标注哪些段需校正

**优先级验证：**
- ✅ 即使只有 DSA 数据（无 CTA），pipeline 也能运行并评分
- ✅ `input.cta` 可为空，`stage1a_cta` 相关 criteria 会降级到最低分但不 crash

---

## 二、当前 Case (`陈秀川-DSA`) 数据处理建议

### 2.1 数据结构分析

**已知信息：**
- 7 个 DICOM 文件 (IM000000-IM000006)
- 目录名 "Exposure 7.5 fps" 表示采集帧率
- 文件大小 4-6 MB，可能是单帧或多帧序列

**需要解析的 DICOM 元数据：**

```python
# 关键字段 (pydicom)
ds.Modality                     # 应为 "XA" (X-Ray Angiography)
ds.PositionerPrimaryAngle       # RAO (+) / LAO (-) 角度
ds.PositionerSecondaryAngle     # CRA (+) / CAU (-) 角度
ds.SeriesDescription            # 可能包含体位描述
ds.ProtocolName                 # 可能包含左冠/右冠标识
ds.NumberOfFrames               # 如果 >1，是多帧序列
ds.AcquisitionNumber            # 采集序号，区分不同体位
```

**处理流程：**

1. **解析元数据** → 生成 `core_views` 映射表
2. **按体位分组** → 确认四大核心体位是否齐全
3. **剔除介入治疗帧** → 如果 `SeriesDescription` 包含 "PCI" / "Stent" / "Balloon"
4. **提取关键帧** → 如果是多帧序列，选择造影剂充盈最佳的帧（通常是中间帧）

### 2.2 标注任务分解（待张冠兆提供）

**当前缺失：**
- ❌ 诊断结论（书面形式）
- ❌ 狭窄位置 + 程度 + 血管名称
- ❌ 中山医院报告模板

**标注输出格式（基于会议标准）：**

```yaml
# .tmp/陈秀川-DSA/gold_standard.yaml (示例)
case_id: "case_chxc_dsa_001"
patient_id: "陈秀川"
clinical_context:
  age: 65
  gender: "M"
  symptoms: "活动后胸闷"  # 典型症状 → 直接 DSA 路径

input:
  dsa:
    dicom_files:
      - path: "Exposure 7.5 fps/IM000000"
        view_id: "rca_rao30_cau0"
        modality: "XA"
      # ... 其他文件
    
    core_views:  # 由 DICOM 元数据自动生成
      - view_id: "rca_rao30_cau0"
        dicom_files: ["IM000000"]
        positioner_primary_angle: 30.0
        positioner_secondary_angle: 0.0
        target_vessels: ["RCA"]
      # ... 其他体位

stage1b_dsa:
  segments:
    - segment_id: "RCA_1"  # 右冠近端
      best_view: "rca_rao30_cau0"
      stenosis_percent: 75  # 示例：重度狭窄
      stenosis_grade: "severe"
      plaque_type: "calcified"
      timi_flow: 3
      
    - segment_id: "LAD_6"  # 前降支中段
      best_view: "lca_rao30_cra25"
      stenosis_percent: 0  # 明确标注：未见狭窄
      stenosis_grade: "none"
      
    # ... 其他段（包括阴性发现）

stage3_scoring:
  syntax_score:
    total: 23  # 基于 DSA 的 SYNTAX Score
    risk_tier: "intermediate"

clinical_decision:
  recommendation: "建议右冠近端支架置入"
  rationale: "右冠近端重度狭窄（75%），症状典型，SYNTAX 中危"
```

### 2.3 模型训练对齐 Check List

- [ ] **DICOM 解析脚本**：提取体位元数据 → `core_views` 映射表
- [ ] **数据清洗规则**：剔除介入治疗图像的自动化脚本
- [ ] **标注模板工具**：基于 `gold_standard.yaml` schema 的标注界面
- [ ] **报告模板对齐**：将中山医院报告模板转为 `prediction.json` schema 映射
- [ ] **质控检查**：验证"四大核心体位"齐全性、阴性段是否标注

---

## 三、Schema 调整 Roadmap

### Phase 1: 紧急对齐 (本周完成)

**目标：** 让当前 Case 能跑通 pipeline

1. **扩展 `tasks/task_template.yaml`**:
   - 增加 `input.dsa.core_views[]` 字段
   - 增加 `stage1b_dsa.segments[].best_view` 链接

2. **新增 metric**:
   ```python
   # pipeline/metric_registry.py
   "segment_coverage_recall": _segment_coverage  # 检查阴性段是否标注
   "view_metadata_completeness": _view_metadata  # 检查体位元数据
   ```

3. **更新 rubric example**:
   - 增加 C015: 完整性检查（阴性段标注）
   - 增加 C004: 体位元数据提取正确性

4. **写 DICOM 解析工具**:
   ```bash
   # 新脚本
   python scripts/parse_dsa_metadata.py \
       --input .tmp/陈秀川-DSA/Exposure\ 7.5\ fps/ \
       --output .tmp/陈秀川-DSA/core_views.yaml
   ```

### Phase 2: 批量标注准备 (下周)

**目标：** 支持 10-20 个 CTA+DSA Case 的标注流水线

1. **标注工具开发**:
   - Web UI 或 CLI 工具，基于 DICOM viewer + YAML 编辑器
   - 自动填充体位元数据、自动检测阴性段遗漏

2. **报告模板转换器**:
   ```python
   # scripts/convert_hospital_report.py
   # 输入：中山医院 Word/PDF 报告
   # 输出：gold_standard.yaml
   ```

3. **质控 Pipeline**:
   - Inter-annotator agreement (Cohen's κ) 计算
   - 自动检测标注缺失字段、不合理狭窄值 (>100%)

### Phase 3: 模型训练对齐 (标注完成后)

1. **训练数据格式转换**:
   - `gold_standard.yaml` → 模型训练的 JSON/CSV
   - 保留体位元数据用于模型的多视角学习

2. **评估对齐验证**:
   - 模型输出 `prediction.json` → Benchmark 自动评分
   - 对比模型报告 vs 中山医院报告模板的格式一致性

---

## 四、待办事项 (Action Items)

### 🔴 高优先级 (本周)

1. **安装 pydicom** (`pip install pydicom`) 用于 DICOM 解析
2. **解析当前 Case 的 DICOM 元数据**:
   ```bash
   python scripts/parse_dsa_metadata.py --input .tmp/陈秀川-DSA/
   ```
3. **联系张冠兆获取**:
   - 当前 Case 的诊断结论（文字版）
   - 中山医院冠脉造影报告模板
   - 确认 7 个 DICOM 文件是否都是 pre-intervention

4. **Schema 扩展**:
   - 更新 `tasks/task_template.yaml` 增加 `core_views` 字段
   - 新增 2 个 metrics (`segment_coverage_recall`, `view_metadata_completeness`)

### 🟡 中优先级 (下周)

5. **标注工具开发** (简易版):
   - CLI 工具：读取 DICOM + YAML 模板 → 交互式标注 → 输出 `gold_standard.yaml`

6. **报告模板转换脚本**:
   - 将中山医院报告 → `gold_standard.yaml` 的映射规则

7. **质控检查脚本**:
   - 验证"四大核心体位"齐全性
   - 检测阴性段是否标注
   - 检查 stenosis_percent 范围 [0, 100]

### 🟢 低优先级 (批量标注时)

8. **批量处理 10-20 个 Case**:
   - 统一目录结构 `data/cases/case_XXX/{dsa/cta/}`
   - 批量解析 DICOM 元数据
   - 并行标注 + 质控

9. **Inter-annotator agreement**:
   - 请 2-3 位专家标注同一批 Case
   - 计算 Cohen's κ (stenosis grading)

10. **模型训练数据转换**:
    - `gold_standard.yaml` → 训练集 JSON
    - 生成体位-血管对应表用于多视角学习

---

## 五、关键对齐验证点

### ✅ 已对齐
- DSA 金标准地位 (pipeline 优先支持 DSA-only 运行)
- 狭窄程度量化标准 (0-100, MAE 计算)
- 钙化 blooming 校正评估 (fusion_reasoning 维度)

### ⚠️ 需调整
- **四大核心体位 + 四大血管** 的标注格式（当前按 17 段组织，未按体位）
- **阴性段标注要求**（当前可选，会议要求必填）
- **体位元数据验证**（当前未要求，会议强调是核心）

### ❌ 待实现
- DICOM 元数据自动解析工具
- 标注工具（支持体位映射 + 阴性段提醒）
- 报告模板转换器（中山医院格式 → benchmark schema）

---

## 六、附录：四大核心体位 vs SYNTAX 17 段映射

**中山医院标准体位（示例）：**

| 体位 ID | 描述 | 角度 | 目标血管 | SYNTAX 段覆盖 |
|--------|------|------|---------|--------------|
| RCA_1 | 右冠 RAO 30° | RAO 30° / CAU 0° | RCA | RCA 1-3 |
| LCA_1 | 左冠 RAO 30° CRA 25° | RAO 30° / CRA 25° | LM, LAD | LM 5, LAD 6-9 |
| LCA_2 | 左冠 LAO 60° CRA 25° | LAO 60° / CRA 25° | LCX | LCX 11-15 |
| LCA_3 | 左冠 LAO 45° CAU 25° | LAO 45° / CAU 25° | LAD | LAD 6-10 |

**映射逻辑：**
- Agent 需理解：不同体位看到的血管段不同
- Gold standard 需标注：每个段的 `best_view`（最清晰体位）
- Rubric 需评估：Agent 是否正确选择体位进行狭窄评估

---

**生成时间：** 2026-07-22  
**待更新：** 收到张冠兆提供的诊断结论和报告模板后，补充 Phase 1 实现细节
