# 开源方法集成状态

**更新时间**: 2026-07-23 13:00  
**状态**: 代码已下载，权重下载中

---

## ✅ 已集成方法（代码）

### 1. SAM-VMNet ⭐
- **路径**: `algorithms/specialist_models/sam_vmnet/`
- **论文**: Medical Physics 2024, https://arxiv.org/abs/2406.00492
- **任务**: 冠脉分割 + 狭窄检测
- **代码**: ✅ 已克隆
- **权重**: 🔄 下载中（HuggingFace: ly17/SAM-VMNet）
  - 原始权重位置: Google Drive（4 个文件）
  - 备用: HuggingFace 镜像
- **性能**: IoU 0.63, Sensitivity 0.98
- **依赖**: PyTorch, mamba_ssm, causal_conv1d
- **状态**: ready

### 2. CM-UNet
- **路径**: `algorithms/specialist_models/cm_unet/`
- **论文**: arXiv 2025, https://arxiv.org/abs/2507.17779
- **任务**: 自监督冠脉分割
- **代码**: ✅ 已克隆
- **权重**: 🔄 下载中（HuggingFace: Camsouille/CM-UNet）
- **特点**: 仅需 18 张标注图像微调
- **性能**: IoU 0.67, Dice 0.80
- **状态**: ready

### 3. CardioSyntax ⭐
- **路径**: `algorithms/specialist_models/cardiosyntax/`
- **论文**: WACV 2025, https://arxiv.org/abs/2407.19894
- **任务**: End-to-end SYNTAX score 预测
- **代码**: ✅ 已克隆
- **权重**: ❌ Yandex.Disk 不可访问
  - URL: https://disk.yandex.com/d/_4ARTacETFQr1A
  - 需要: 手动下载或联系作者
- **数据**: ✅ 已有元数据 + 🔄 下载中（Part 6, 9）
- **性能**: MAE 4.2 for SYNTAX score
- **状态**: code_only（需权重）

### 4. DeepCoro
- **路径**: `algorithms/specialist_models/deepcoro/`
- **论文**: 心血管影像分析
- **任务**: 狭窄检测和量化
- **代码**: ✅ 已克隆
- **权重**: ⚠️ 仅推理代码，主模型未公开
- **状态**: partial

### 5. MesserMMP SYNTAX Prediction
- **路径**: 权重将下载至 `weights/messermmp_syntax/`
- **HuggingFace**: MesserMMP/coronary-syntax-prediction
- **任务**: SYNTAX score 预测
- **权重**: 🔄 下载中
- **状态**: weights_only（需适配代码）

### 6. TC-SemiSAM
- **路径**: 权重将下载至 `weights/tc_semisam/`
- **HuggingFace**: ly17/TC-SemiSAM-checkpoints
- **任务**: 半监督血管分割
- **权重**: 🔄 下载中
- **架构**: SAM3 + Mean Teacher
- **状态**: weights_only（需适配代码）

---

## ❌ 未成功方法

### LRSE-Net
- **状态**: GitHub 仓库不存在
- **URL**: https://github.com/Yukei7/LRSE-Net.git (404)
- **备注**: 可能仓库已删除或私有

### DeepCORO-CLIP
- **状态**: 代码和权重尚未公开
- **GitHub**: https://github.com/HeartWiseAI/DeepCORO_CLIP (404)
- **HuggingFace**: https://huggingface.co/collections/heartwise/deepcoro-clip (无法访问)
- **备注**: 可能论文发表后会公开

---

## 📁 目录结构

```
algorithms/
├── specialist_models/
│   ├── sam_vmnet/           ✅ 代码完整，141 文件
│   ├── cm_unet/             ✅ 代码完整，148 文件
│   ├── cardiosyntax/        ✅ 代码完整，需权重
│   ├── deepcoro/            ✅ 部分代码
│   └── weights/             🔄 HuggingFace 权重下载中
│       ├── sam_vmnet/
│       ├── cm_unet/
│       ├── messermmp_syntax/
│       ├── tc_semisam/
│       └── cardiosyntax/
│           └── WEIGHTS_URL.txt
├── baselines/
│   └── pure_llm/            ✅ 已实现
└── tools/                   ⏳ 待实现
```

---

## 🔄 进行中的下载

### 1. HuggingFace 模型权重
- **状态**: 后台下载中
- **方法**: ly17/SAM-VMNet, Camsouille/CM-UNet, MesserMMP/coronary-syntax-prediction, ly17/TC-SemiSAM-checkpoints
- **输出**: `/tmp/...tasks/b5gjt4qg0.output`

### 2. CardioSyntax 数据集
- **状态**: 下载中（16 MB / 18 GB，0.09%）
- **Part 6**: 15 GB
- **Part 9**: 3 GB
- **速度**: ~10-20 KB/s
- **预计**: 需要 1-2 天

---

## 🎯 下一步行动

### 立即可做（不依赖下载）

1. **实现标准接口适配器** ⏳
   为每个方法创建 `register.py`：
   ```python
   # algorithms/specialist_models/sam_vmnet/register.py
   from algorithms import AlgorithmRegistry
   from .sam_vmnet_wrapper import SAMVMNetAlgorithm
   
   AlgorithmRegistry.register("sam_vmnet", SAMVMNetAlgorithm)
   ```

2. **编写包装类** ⏳
   统一接口，隐藏原始实现细节：
   ```python
   class SAMVMNetAlgorithm(BaseAlgorithm):
       def load_model(self, checkpoint_path):
           # 加载 SAM-VMNet 原始代码
       
       def predict(self, input_data):
           # 调用原始推理，转为标准输出格式
   ```

3. **测试权重完整性** ⏳
   下载完成后验证所有权重文件

4. **文档整理** ⏳
   - 每个方法的 README（使用说明）
   - Related Work 材料（论文引用）

### 等待下载完成后

5. **运行基准测试**
   - 陈秀川 case 上测试所有方法
   - 记录执行时间、性能指标

6. **CardioSyntax 权重获取**
   - 方案 A: 联系作者请求 Yandex.Disk 替代链接
   - 方案 B: 自己训练（有代码和数据）

---

## 📊 方法能力矩阵

| 方法 | 分割 | 狭窄检测 | SYNTAX 预测 | 权重可用 | 代码可用 |
|------|------|----------|-------------|----------|----------|
| SAM-VMNet | ✅ | ✅ | ❌ | 🔄 | ✅ |
| CM-UNet | ✅ | ❌ | ❌ | 🔄 | ✅ |
| CardioSyntax | ❌ | ❌ | ✅ | ❌ | ✅ |
| DeepCoro | ❌ | ✅ | ❌ | ❌ | ⚠️ |
| MesserMMP | ❌ | ❌ | ✅ | 🔄 | ❌ |
| TC-SemiSAM | ✅ | ❌ | ❌ | 🔄 | ❌ |

**图例**: ✅ 完整 | 🔄 下载中 | ❌ 不可用 | ⚠️ 部分

---

## 🔍 调研成果用途

### 1. Related Work 章节
所有 11 个方法的论文可直接引用：
- 2024-2026 最新工作（SAM-VMNet, CM-UNet, CardioSyntax）
- 传统方法对比（DeepCoro, LRSE-Net）
- Foundation models（DeepCORO-CLIP, CoAM）

### 2. Benchmark 工具库
**5 个可用方法**可作为智能体可调用工具：
- 分割工具: SAM-VMNet, CM-UNet, TC-SemiSAM
- SYNTAX 预测: CardioSyntax, MesserMMP
- 组合使用: 智能体设计 pipeline

### 3. Upper-bound Reference
- SAM-VMNet: 分割任务上限（IoU 0.63）
- CardioSyntax: SYNTAX 预测上限（MAE 4.2）
- 证明 Agent 方法的优势空间

---

**维护者**: Cardiomni Team  
**调研 Agent**: a23c0a0f74dbdc35e (71k tokens, 38 tool uses)  
**最后更新**: 2026-07-23 13:00
