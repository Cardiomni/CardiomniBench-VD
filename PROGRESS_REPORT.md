# CardiomniBench-VD 任务进度报告

**日期**: 2026-07-23  
**状态**: 进行中

---

## 📋 任务总览

### ✅ 任务 1: DeepCORO-CLIP 资源下载

**GitHub 仓库**: ✅ **已完成**
- 路径: `/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/algorithms/specialist_models/deepcoro_clip/`
- 文件数: 270 个
- 内容: 完整的代码库、配置文件、文档

**模型权重**: ⏳ **等待 HuggingFace Token**
- 需要下载的模型:
  - `heartwise/deepcoro_clip` (主模型，狭窄检测)
  - `heartwise/VasoVision` (预处理模型)
- 状态: 需要认证 token，已创建下载脚本 `download_weights.sh`
- 文档: `DOWNLOAD_STATUS.md` 详细说明

**DeepCORO-CLIP 信息**:
- 论文: arXiv:2603.17675 (2026年3月提交)
- 训练数据: 203,808 个造影视频，28,117 名患者
- 性能: 狭窄检测 AUROC 0.888 (内部) / 0.89 (外部验证)
- 架构: mVIT (视频) + BioMedBERT (文本) + CLIP 对比学习

### 🔄 任务 2: CardioSyntax 完整数据集下载

**状态**: 🔄 **下载中（速度慢）**
- 总大小: ~145 GB (9 个部分)
- 当前进度: Part 0 下载中
- 下载速度: ~18-20 KB/s ⚠️
- 预计完成: 9-10 天/部分 (串行)
- 文档: `DOWNLOAD_STATUS.md` 

**已有数据**:
- 元数据: ✅ 完整 (all.json, metadata.json, 各 part.json)
- Part 6: ⚠️ 部分 (32 MB / ~16 GB)
- Part 9: ⚠️ 部分 (34 MB / ~16 GB)

**改进建议**:
- 使用 aria2c 多线程下载
- 考虑其他数据源或镜像
- 当前下载可继续后台运行

---

## 🛠️ 已集成的算法工具

### 当前状态

| 算法 | GitHub | 权重 | Wrapper | 状态 |
|------|--------|------|---------|------|
| SAM-VMNet | ✅ | ⏳ | ⏳ | 代码已下载 |
| CM-UNet | ✅ | ⏳ | ⏳ | 代码已下载 |
| CardioSyntax | ✅ | ⏳ | ⏳ | 代码已下载 |
| DeepCORO | ✅ | ⏳ | ⏳ | 代码已下载 |
| DeepCORO-CLIP | ✅ | ⏳ | ⏳ | 代码已下载 |

### 权重下载进度

**HuggingFace 模型** (需要 token):
1. ⏳ heartwise/deepcoro_clip
2. ⏳ heartwise/VasoVision
3. ⏳ (其他模型待确认)

**Google Drive 模型** (从之前的调研):
- ⏳ SAM-VMNet
- ⏳ CM-UNet
- ⏳ MesserMMP
- ⏳ TC-SemiSAM

---

## 📊 项目结构

```
/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/
├── algorithms/
│   ├── base.py                          # ✅ 基础接口
│   ├── __init__.py                      # ✅ 注册系统
│   ├── README.md                        # ✅ 算法库概览
│   ├── INTEGRATION_STATUS.md            # ✅ 集成状态
│   ├── baselines/
│   │   └── pure_llm/                    # ✅ PureLLM 基线
│   └── specialist_models/
│       ├── sam_vmnet/                   # ✅ 代码
│       ├── cm_unet/                     # ✅ 代码
│       ├── cardiosyntax/                # ✅ 代码
│       ├── deepcoro/                    # ✅ 代码
│       ├── deepcoro_clip/               # ✅ 代码
│       └── weights/                     # ⏳ 权重下载中
│
├── .raw_data/
│   ├── CardioSyntax/
│   │   ├── metadata.json                # ✅ 元数据
│   │   ├── selected_50_studies_optimized.json  # ✅ 50案例选择
│   │   ├── download_all_parts.sh        # ✅ 下载脚本
│   │   ├── DOWNLOAD_STATUS.md           # ✅ 状态文档
│   │   └── downloaded_parts/            # 🔄 下载中
│   └── opensource_methods_survey.json   # ✅ 方法调研
│
└── docs/
    ├── TASK_SPECIFICATION.md            # ✅ 任务定义
    ├── ECHOAGENT_ANALYSIS.md            # ✅ EchoAgent分析
    ├── ECHOAGENT_EXPERIMENT_DESIGN.md   # ✅ 实验设计
    └── ECHOAGENT_LESSONS.md             # ✅ 迁移指南
```

---

## 🎯 下一步行动

### 紧急 (需要用户操作)

1. **提供 HuggingFace Token**
   - 获取: https://huggingface.co/settings/tokens
   - 配置到: `deepcoro_clip/api_key.json`
   - 运行: `bash download_weights.sh`

### 高优先级

2. **优化 CardioSyntax 下载**
   - 尝试 aria2c 多线程下载
   - 或让当前下载后台运行

3. **下载其他模型权重**
   - SAM-VMNet (Google Drive)
   - CM-UNet (Google Drive)
   - 其他已调研模型

### 中优先级

4. **编写 Wrapper 类**
   - 为每个算法实现 BaseAlgorithm 接口
   - 统一输入输出格式

5. **实现通用工具**
   - DICOM 解析器
   - 视频加载器
   - 投影角度计算器

6. **组织陈秀川案例数据**
   - 从 DSA-流程.docx 提取标注
   - 转换为 JSON 格式

### 低优先级

7. **实现 Cardiomni Agent**
   - 4-stage SOP 逻辑
   - 工具调度系统

8. **构建评估框架**
   - 三轴评估指标
   - 自动化测试流程

9. **首轮实验**
   - 在 50 案例上运行
   - 收集性能数据

---

## 📈 进度统计

**完成度**:
- 文档设计: 95%
- 代码下载: 100%
- 权重下载: 5%
- Wrapper实现: 0%
- Agent实现: 0%
- 评估框架: 0%
- 实验运行: 0%

**总体进度**: ~30%

---

## ⚠️ 阻塞因素

1. **HuggingFace Token** - 阻塞 DeepCORO-CLIP 权重下载
2. **下载速度** - CardioSyntax 下载极慢
3. **权重获取** - 其他模型权重下载方式待确认

---

## 📝 备注

- 所有已下载代码库位于 `algorithms/specialist_models/`
- 下载脚本均已创建并测试
- 详细状态文档已同步更新
- 后台下载进程正在运行

**联系事项**: 请提供 HuggingFace token 以继续权重下载。

---

**报告生成时间**: 2026-07-23 15:15  
**下次更新**: 权重下载完成后
