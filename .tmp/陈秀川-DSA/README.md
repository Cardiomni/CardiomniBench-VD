# 陈秀川-DSA Case 处理指南

**Case ID:** case_chxc_001 (临时编号)  
**状态:** 等待诊断结论标注  
**DICOM 文件:** 7 个文件 (IM000000-IM000006)

---

## 📁 目录结构

```
.tmp/陈秀川-DSA/
├── Exposure 7.5 fps/          # DICOM 文件目录
│   ├── IM000000               # 6.6 MB
│   ├── IM000001               # 4.1 MB
│   ├── IM000002               # 5.6 MB
│   ├── IM000003               # 4.3 MB
│   ├── IM000004               # 4.3 MB
│   ├── IM000005               # 4.8 MB
│   └── IM000006               # 5.1 MB
├── DSA-流程.docx              # 原始流程文档 (未解析)
└── README.md                  # 本文件
```

---

## 🔄 处理流程

### Step 1: 解析 DICOM 元数据

**前置条件:** 安装 pydicom
```bash
pip install pydicom
```

**运行解析脚本:**
```bash
cd /mnt/aliyunsb/CardiomniBench-VD

python scripts/parse_dsa_metadata.py \
    --input ".tmp/陈秀川-DSA/Exposure 7.5 fps" \
    --output .tmp/陈秀川-DSA/metadata_report.json \
    --template .tmp/陈秀川-DSA/gold_standard_template.yaml
```

**输出文件:**
- `metadata_report.json`: DICOM 元数据 + 质控报告
- `gold_standard_template.yaml`: 待填充的标注模板

**检查内容:**
1. 每个 DICOM 文件的体位角度 (RAO/LAO, CAU/CRA)
2. 是否包含介入治疗图像 (需要剔除)
3. core_views 自动分组 (按相似角度归类)

### Step 2: 临床专家标注

**需要张冠兆医生提供:**

1. **临床背景**
   ```yaml
   clinical_context:
     age: 65
     gender: "M"
     symptoms: "活动后胸闷"
     risk_factors: ["高血压", "糖尿病"]
   ```

2. **DSA 诊断结论** (每个血管段)
   ```yaml
   stage1b_dsa:
     segments:
       - segment_id: "RCA_1"  # 右冠近段
         stenosis_percent: 75
         stenosis_grade: "severe"
         plaque_type: "calcified"
         best_view: "rao30_cau0"  # 从 core_views 选择
       
       - segment_id: "LAD_6"  # 前降支中段
         stenosis_percent: 0
         stenosis_grade: "none"  # 阴性发现也要标注
       # ... 其他段
   ```

3. **SYNTAX Score 计算**
   ```yaml
   stage3_scoring:
     syntax_score:
       total: 23
       risk_tier: "intermediate"
   ```

4. **临床决策**
   ```yaml
   clinical_decision:
     recommendation: "建议右冠近段支架置入"
     rationale: "右冠近端重度狭窄 75%，症状典型"
   ```

**参考文档:**
- `/mnt/aliyunsb/CardiomniBench-VD/CHECKLIST_FOR_CLINICIAN.md` (双语工作清单)
- `/mnt/aliyunsb/CardiomniBench-VD/tasks/task_template.yaml` (完整 schema)
- `/mnt/aliyunsb/CardiomniBench-VD/rubrics/examples/case_001_rubric.yaml` (示例)

### Step 3: Schema 验证

**验证标注文件格式:**
```bash
python -m pipeline.cli validate \
    --case-dir .tmp/陈秀川-DSA \
    --schema tasks/task_template.yaml
```

**质控检查:**
```bash
python scripts/annotation_qc.py \
    --case .tmp/陈秀川-DSA/gold_standard.yaml \
    --check-coverage      # 验证四大核心体位齐全
    --check-completeness  # 验证阴性段是否标注
```

### Step 4: 移动到正式目录

**标注完成后:**
```bash
# 创建正式 case 目录
mkdir -p data/cases/case_chxc_001

# 复制文件
cp -r .tmp/陈秀川-DSA/* data/cases/case_chxc_001/

# 重命名
cd data/cases/case_chxc_001
mv "Exposure 7.5 fps" dsa_dicoms
mv gold_standard_template.yaml gold_standard.yaml  # 标注完成后
```

**更新 splits.yaml:**
```bash
echo "test:" >> data/splits.yaml
echo "  - case_chxc_001" >> data/splits.yaml
```

### Step 5: Pipeline 试跑

**用 mock agent 验证:**
```bash
/opt/anaconda3/bin/python -m pipeline.cli run \
    --toml benchmark.toml \
    --agent mock \
    --split test \
    --limit 1
```

**检查输出:**
```bash
cat runs/cardiomni_bench/rerun_0/case_chxc_001/evaluation.json | jq .
cat runs/cardiomni_bench/summary.json | jq .
```

**验证内容:**
- ✅ 所有 dimensions 都有分数
- ✅ Criteria 能正确提取 gold/pred 数据
- ✅ Metrics 计算没有 crash
- ✅ Overall score 在合理范围 (0-100)

---

## 📋 关键注意事项

### 1. 阴性段标注要求 ⚠️

**错误示例 (不完整):**
```yaml
segments:
  - segment_id: "RCA_1"
    stenosis_percent: 75  # 只标注了有病变的段
```

**正确示例 (完整):**
```yaml
segments:
  - segment_id: "RCA_1"
    stenosis_percent: 75
  - segment_id: "RCA_2"
    stenosis_percent: 0   # 阴性段也要标注
  - segment_id: "LAD_5"
    stenosis_percent: 0
  - segment_id: "LAD_6"
    stenosis_percent: 60
  # ... 所有可见血管段
```

### 2. 体位信息 (core_views) 的重要性

**会议要求:** 四大核心体位对应四大血管

**pipeline 会自动生成 core_views:**
```yaml
input:
  dsa:
    core_views:
      - view_id: "rao30_cau0"
        description: "RAO 30° CAU 0°"
        dicom_files: ["IM000000"]
        target_vessels: ["RCA"]
      # ... 其他体位
```

**标注时只需引用 view_id:**
```yaml
stage1b_dsa:
  segments:
    - segment_id: "RCA_1"
      best_view: "rao30_cau0"  # 引用上面的 view_id
```

### 3. 介入治疗图像的剔除

**检查 DICOM SeriesDescription:**
- 如果包含 "stent" / "balloon" / "PCI" → 需要剔除
- 如果包含 "wire" / "guide" → 可能是介入，需要确认

**pipeline 会在 metadata_report.json 中标记:**
```json
{
  "file_name": "IM000003",
  "likely_intervention": true,  // ← 标记可疑文件
  "series_description": "RCA PCI with stent"
}
```

### 4. SYNTAX Score 计算

**如果不确定:**
- 可以使用在线计算器: http://www.syntaxscore.org
- 或者参考: `rubrics/clinical_standards.yaml` 中的定义

**Benchmark 允许误差:**
- MAE < 5 分 → Grade A
- MAE 5-10 分 → Grade B
- MAE > 10 分 → Grade C

---

## 🔧 故障排查

### 问题 1: pydicom 安装失败

**解决:**
```bash
# 使用 pip3 或指定 Python 版本
/opt/anaconda3/bin/pip install pydicom

# 或者使用 conda
conda install -c conda-forge pydicom
```

### 问题 2: parse_dsa_metadata.py 报错

**常见原因:**
- DICOM 文件路径包含空格 → 用引号括起来
- DICOM 文件损坏 → 检查 `file` 命令输出

**调试:**
```bash
# 检查 DICOM 文件是否可读
file ".tmp/陈秀川-DSA/Exposure 7.5 fps/IM000000"
# 应输出: DICOM medical imaging data

# 手动测试 pydicom
python -c "import pydicom; ds = pydicom.dcmread('.tmp/陈秀川-DSA/Exposure 7.5 fps/IM000000'); print(ds.Modality)"
# 应输出: XA
```

### 问题 3: gold_standard.yaml 格式错误

**验证工具:**
```bash
# 检查 YAML 语法
python -c "import yaml; yaml.safe_load(open('.tmp/陈秀川-DSA/gold_standard.yaml'))"

# 使用 pipeline 验证
python -m pipeline.cli validate --case-dir .tmp/陈秀川-DSA
```

---

## 📞 联系与协作

### 文档参考

**优先级排序:**
1. `CHECKLIST_FOR_CLINICIAN.md` — 给张冠兆医生的工作清单 (双语)
2. `docs/ANNOTATION_ACTION_PLAN.md` — 行动计划 + 优先级
3. `docs/DATA_ANNOTATION_ALIGNMENT.md` — 会议对齐分析 (技术细节)
4. `tasks/task_template.yaml` — 完整 schema 定义

### 下一步

**等待输入 (Priority 0):**
- [ ] 临床背景信息 (年龄、性别、症状)
- [ ] DSA 诊断结论 (包括阴性段)
- [ ] 中山医院报告模板
- [ ] 确认 DICOM 文件清洁性

**可并行工作 (不依赖标注):**
- [ ] 安装 pydicom
- [ ] 运行 DICOM 解析脚本
- [ ] 扩展 schema 支持 core_views
- [ ] 实现 segment_coverage_recall metric

**目标时间线:**
- 收到诊断结论后 2-3 天完成第一个 Case
- 作为后续 19 cases 的模板

---

**最后更新:** 2026-07-22  
**维护者:** Jiaming Ma  
**项目路径:** `/mnt/aliyunsb/CardiomniBench-VD`
