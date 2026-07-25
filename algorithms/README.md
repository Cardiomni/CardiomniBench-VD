# CardiomniBench-VD 算法库

**目标**: 类似 BioML-Bench，提供标准化的 DSA 分析算法接口，供智能体调用和组合。

---

## 目录结构

```
algorithms/
├── README.md                    # 本文档
├── specialist_models/           # 专科模型（作为工具 + upper-bound）
│   ├── deepcoro_clip/          # DeepCORO-CLIP (SYNTAX score)
│   ├── cardiosyntax/           # CardioSyntax baseline
│   ├── stenosis_detectors/     # 狭窄检测模型
│   └── view_classifiers/       # 投影角度分类
├── baselines/                   # Baseline agents
│   ├── pure_llm/               # 直接输入 LLM
│   ├── openhands/              # OpenHands 适配
│   └── swe_agent/              # SWE-Agent 适配
└── tools/                       # 通用工具函数
    ├── dicom_parser.py
    ├── video_loader.py
    ├── projection_calculator.py
    └── syntax_calculator.py
```

---

## 已调研的 DSA 相关方法

### 1. **DeepCORO-CLIP** ⭐
- **论文**: "DeepCORO-CLIP: A Foundation Model for Coronary Angiography Analysis"
- **任务**: SYNTAX score 预测、视角分类、病变检测
- **架构**: Vision-Language Model (CLIP-based)
- **数据**: 130,000+ angiography videos from Montreal Heart Institute
- **性能**: R² = 0.85 for SYNTAX score
- **代码**: https://github.com/HeartWiseAI/DeepCORO_CLIP (尚未公开)
- **权重**: https://huggingface.co/collections/heartwise/deepcoro-clip
- **数据集**: DeepCORO-mini (1,000 cases, ~7,000 videos) on Physionet (即将发布)
- **状态**: ⏳ 待下载模型权重

### 2. **CardioSyntax**
- **论文**: WACV 2025, https://arxiv.org/abs/2407.19894
- **任务**: End-to-end SYNTAX score 预测
- **数据集**: CardioSyntax (1,844 studies, 14,219 videos)
- **架构**: Video Transformer
- **性能**: MAE = 4.2 for SYNTAX score
- **代码**: 随 Zenodo 数据集发布
- **状态**: ✅ 元数据已有，图像数据下载中

### 3. **ARCADE Dataset Methods**
- **论文**: Nature Scientific Data 2023
- **数据**: 1,500 XCA frames, 26 SYNTAX segments 标注
- **方法**: 传统 CV + ML (SVM, Random Forest)
- **任务**: 节段级狭窄检测
- **状态**: ❌ 数据集链接未找到

### 4. **DCA1 Dataset Methods**
- **论文**: Applied Sciences 2019
- **数据**: 130 XCA images (Kaggle)
- **方法**: CNN-based stenosis classification
- **任务**: 二分类（有/无狭窄）
- **状态**: ❌ Kaggle 链接未找到

### 5. **传统 QCA (Quantitative Coronary Angiography)**
- **方法**: 基于边缘检测和几何测量
- **工具**: CAAS, QAngio, CMS
- **优点**: 可解释、临床标准
- **缺点**: 需要人工选择参考点
- **状态**: 可作为传统 baseline

---

## 标准接口设计

所有算法需实现统一接口，便于智能体调用：

### Python API

```python
from algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, config):
        """初始化模型"""
        pass
    
    def load_model(self, checkpoint_path):
        """加载预训练权重"""
        pass
    
    def predict(self, input_data):
        """
        Args:
            input_data: dict with keys:
                - case_id: str
                - videos: List[VideoData]
                - metadata: dict
        
        Returns:
            dict with keys:
                - syntax_score: float
                - segments: List[SegmentPrediction]
                - confidence: float
                - execution_time: float
                - tokens_used: int (if applicable)
        """
        pass
    
    def get_metadata(self):
        """返回模型元信息"""
        return {
            "name": "MyAlgorithm",
            "version": "1.0",
            "paper": "https://arxiv.org/...",
            "task": ["syntax_score", "stenosis_detection"],
            "input_format": ["dicom", "npy"],
            "output_format": "json"
        }
```

### CLI Interface

所有算法提供统一 CLI：

```bash
# 单 case 推理
python -m algorithms.deepcoro_clip.infer \
    --input /path/to/case/ \
    --output result.json \
    --config config.yaml

# Batch 推理
python -m algorithms.deepcoro_clip.batch_infer \
    --input_dir /path/to/cases/ \
    --output_dir results/ \
    --num_workers 4
```

---

## 智能体调用方式

类似 BioML-Bench，智能体可以：

1. **查看可用工具**
   ```python
   from algorithms import list_available_algorithms
   algos = list_available_algorithms()
   # ['deepcoro_clip', 'cardiosyntax', 'qca_tool', ...]
   ```

2. **读取工具文档**
   ```python
   from algorithms import get_algorithm_doc
   doc = get_algorithm_doc('deepcoro_clip')
   # Returns: API documentation, usage examples
   ```

3. **调用工具**
   ```python
   from algorithms import load_algorithm
   model = load_algorithm('deepcoro_clip', config='default')
   result = model.predict(case_data)
   ```

4. **组合多个工具**
   ```python
   # Agent 可以设计 pipeline
   step1 = view_classifier.predict(videos)
   step2 = stenosis_detector.predict(selected_views)
   step3 = syntax_calculator.compute(stenosis_results)
   ```

---

## 评估维度

对每个算法（包括 Agent 组合的 pipeline）评估：

1. **准确性**:
   - SYNTAX score MAE
   - 节段狭窄 MAE
   - 分类准确率

2. **效率**:
   - 执行时间
   - GPU 显存占用
   - Tokens 消耗（LLM-based）

3. **完整性**:
   - 节段覆盖率
   - 反幻觉能力

4. **可解释性**:
   - 是否提供证据（关键帧、视角引用）
   - 推理过程可追溯性

---

## 下一步

### 短期
1. ✅ 下载 CardioSyntax Part 6, 9
2. ⏳ 下载 DeepCORO-CLIP 权重
3. ⏳ 克隆 DeepCORO-CLIP 代码（等待公开）
4. ⏳ 实现标准接口基类 `BaseAlgorithm`
5. ⏳ 适配 DeepCORO-CLIP 到标准接口

### 中期
1. 搜索其他开源 DSA 分析方法
2. 实现传统 QCA baseline
3. 实现 PureLLM baseline
4. 实现 Generic Coding Agent 适配

### 长期
1. 扩展到更多专科模型
2. 提供模型 API 服务
3. 建立 leaderboard

---

**维护者**: Cardiomni Team  
**最后更新**: 2026-07-23
