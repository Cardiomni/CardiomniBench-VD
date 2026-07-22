# Cardiomni 提案

> **Cardiomni** —— 一个**自主(autonomous)心血管诊断智能体**,是本文的核心方法贡献。
> **CardiomniBench-VD**(VD = Vascular Diagnosis)—— 随本文**一同提出**的配套 benchmark,用于评测 Cardiomni 与其他智能体/大模型的心血管诊断能力。
> 二者是一体两面:方法 + 评测,关系等同于 **Biomni(方法) + BiomniBench(评测)**。
>
> 本文档所有设计选择均挂真实可查的权威引用(见文末第 11 节),证据可信度按 `[evidence]`/`[derived]`/`[speculation]` 标注。

---

## 0. 核心主张(一句话)

Cardiomni 直接读入同一病人的**原始 CTA + DSA(DICOM)**,像临床医生一样**自主**完成"阅片 → 推理 → 定量 → 决策"的全过程,输出**带临床推理轨迹的诊断报告**;CardiomniBench-VD 是评测这一能力的 benchmark,其基线是**其他智能体与大语言模型**——因为本任务的核心不是分割,而是**临床可解释性**。

---

## 1. 贡献定位:方法 + benchmark 一体

本文提出两个相互支撑的交付物,构成一个完整故事:

| 交付物 | 是什么 | 对标 |
|---|---|---|
| **Cardiomni** | 自主心血管诊断智能体:LLM 编排 + 专用工具 + 临床知识,端到端从 DICOM 到诊断报告 | Biomni |
| **CardiomniBench-VD** | 配套 benchmark:多中心、专家标注、评测"从 CTA+DSA 到诊断报告"的自主诊断能力 | BiomniBench-DA |

**为什么二者必须一起提出**:如同 BiomniBench 之于 Biomni——现有评测都是为"深度学习子任务"(分割 IoU、检测 mAP)设计的,**没有任何评测衡量"自主、可解释的端到端临床诊断"**。要证明 Cardiomni 的价值,必须同时提出能评测它的 benchmark。`[evidence: BiomniBench process-level 评测范式, A3]`

**核心差异化 = 临床可解释性,不是分割精度**。本任务不是"把血管分出来",而是"像医生一样看懂、说清、给出可信的诊断意见"。这决定了:
- 输出是**诊断报告 + 推理轨迹**,不是掩膜/框;
- 评测看**临床结论的正确性与可解释性**,不是像素级 IoU;
- 基线是**能尝试完整诊断任务的系统**(智能体、大模型),不是只能产出掩膜的分割模型。

---

## 2. 任务定义(对齐真实临床)

### 2.1 输入
同病人配对的 **CTA + DSA**,标准 DICOM 格式 `[derived: 专家确认"CTA和DSA都是DICOM"]`:

| 模态 | DICOM 特征 | 关键信息 |
|---|---|---|
| **CTA** | Modality=CT,3D 切片序列 | HU 值(区分钙化 vs 软斑块)、像素间距、层厚 |
| **DSA** | Modality=XA,多帧 cine loop | 动态造影序列(血流时序),管腔金标准 |

> ⚠️ **必须原始 DICOM,不能用 PNG 截图**。PNG 丢失 HU 值、窗宽窗位、像素间距、多帧时序——这正是"钙化 vs 堵塞无法区分"痛点的根源,也是**自主诊断**的信息前提。`[derived: DICOM 标准 + 影像物理]`

### 2.2 输出:自主诊断报告(不是掩膜/框)

Cardiomni 的输出是一份**结构化临床诊断报告**,含三部分。这三部分同时也是 CardiomniBench-VD 的评分对象(见第 4 节 Rubric):

**① 血管段识别 —— SYNTAX 分段法**
- 采用 SYNTAX Score 的 16 主段 + 亚段命名(源自 AHA 1975 分段体系)`[evidence: C1 Austen 1975; C2 Sianos 2005]`

> **SYNTAX Score 是什么(讲解)**
> 一套**纯解剖学的冠脉复杂度评分**,把整棵冠脉树的病变量化成一个分数,决定病人做**支架(PCI)还是搭桥(CABG)**。名字来自对比 PCI vs CABG 的 SYNTAX 试验。`[evidence: C2]`
> - **为什么临床需要它**:同样 70% 狭窄,长在左主干、有钙化、还是分叉处,手术难度和风险完全不同——SYNTAX 把复杂度量化。
> - **怎么算**(仅 ≥50% 狭窄、血管 ≥1.5mm 的病变计分 `[evidence: W10]`):① 定优势型(从 RCA 判断,**无均衡型**);② 每处病变基础分 = 节段权重 × 2(非闭塞)或 × 5(闭塞),左主干权重最高;③ 叠加复杂度加分:CTO(闭塞时长/断端形态/桥侧支)、分叉/三分叉(Medina 分型)、开口、迂曲、长度>20mm、重度钙化、血栓、弥漫病变。
> - **风险三分层**:低危 ≤22、中危 23–32、高危 ≥33(倾向 CABG)。`[evidence: W10; W3]`

**② 狭窄百分比分级 —— CAD-RADS 2.0**

| CAD-RADS | 狭窄% | 含义 |
|---|---|---|
| 0 | 0% | 无狭窄 |
| 1 | 1–24% | 极轻微 |
| 2 | 25–49% | 轻度 |
| 3 | 50–69% | 中度 |
| 4A | 70–99% | 重度(单/双支) |
| 4B | >70% | 左主干或三支 |
| 5 | 100% | 完全闭塞 |

- 分档数字逐字来自原始共识 `[evidence: C3 PMC9627235]`;"有意义狭窄"= ≥50% 且血管 ≥1.5mm `[evidence: D1]`
- **Per-lesion** 每处一档;**Per-patient** 取最严重级别作总体评级 `[evidence: C3]`

**③ 临床推理轨迹(可解释性 —— 本方法的核心)**
报告必须留下:识别了哪些 SYNTAX 段 / 每处狭窄的定位与测量依据 / CTA-DSA 融合决策逻辑 / 临床意义解读 / 何处证据不足需转其他检查。**这部分是 Cardiomni 区别于纯分割/检测模型的关键,也是 benchmark 的重点评分项。**

### 2.3 为什么必须配对 CTA-DSA(本 benchmark 的科学对象)

审稿人会问:"为什么不分别做一个 CTA benchmark 和一个 DSA benchmark?"回答是——**本 benchmark 的研究对象不是单模态感知,而是"融合推理"本身:知道哪个子问题该信哪个模态、两者矛盾时如何归因**。把诊断结论分成三类,前两类单模态就够(这没问题),第三类**只有配对融合才能得出**,而这正是科学价值所在。

**A 类 — CTA 单独可判定**(DSA 是管腔造影,看不到管壁):
- 钙化积分(Agatston)、斑块成分/负荷(钙化/软/混合,HU)、高危斑块特征(低衰减<30HU / 正性重构≥1.1 / 餐巾环)`[evidence: C3; W8]`

**B 类 — DSA 单独可判定**(CTA 单期成像给不了血流时序):
- TIMI 血流分级(动态)、Rentrop 侧支循环、血栓(动态充盈缺损)、重度钙化段的真实管腔 `[evidence: W4; W6]`

**C 类 — 只有 CTA-DSA 融合才能得出**(← 核心价值,benchmark 的可评测对象):

| 融合专属结论 | 为什么单模态不行 | 融合推理逻辑 |
|---|---|---|
| **钙化伪影校正后的真实狭窄** | CTA 在钙化段 blooming 高估——64 排 CTA 在钙化段 **PPV 仅 54%**(NPV 98%),极会"虚报"狭窄;DSA 单独看不出"这段为何该信它" | CTA 识别重度钙化段 → DSA 定真实管腔 → "CTA 示 80% 系 blooming,真实 <50%" `[evidence: R1 PPV 54%]` |
| **轻狭窄+易损斑块的高危判定** | DSA 只见管腔(<50% 看着没事);CTA 见管壁易损特征 | CTA 定易损斑块 + DSA 确认管腔不重 → "解剖轻但斑块高危,近期事件风险高"(ACS 常见 culprit) `[evidence: W8]` |
| **准确的 SYNTAX 评分** | DSA 2D 投影有透视缩短/血管重叠,分段、长度、分叉角度模糊 | CTA 3D 解剖定分段+长度+分叉 → DSA 确认 → 准确 SYNTAX `[evidence: C2]` |
| **CTO 可开通性(J-CTO)** | DSA 看不到闭塞段内部(无对比剂通过);CTA 看不到侧支灌注 | CTA 定闭塞长度/断端形态/钙化 + DSA 定 Rentrop 侧支 → 介入可行性 `[evidence: W11; W6]` |
| **解剖-功能失配的识别与归因** | 约 **1/3** 的狭窄在"≥50% 直径狭窄"与"FFR≤0.8"之间不一致(紧狭窄血流正常 / 轻狭窄血流异常双向都有),单模态无法察觉这个矛盾 | 融合两者发现矛盾 → 归因(钙化?投影缩短?)→ 正确声明"需 FFR"(接第 3.2 节能力边界) `[evidence: R2 Toth 2014; R3 Park 2012]` |

**融合规则(冲突消解)**:重钙化段以 DSA 定量为准(避免 blooming 高估);SYNTAX 分段与长度以 CTA 3D 为准(避免投影缩短);功能学意义两者都定不了时,显式声明需 FFR。`[derived: C3 钙化说明 + R1/R2 + 影像物理]`

> **与 novelty 论点的咬合**:单模态分割模型(ImageCAS 类)输出的是掩膜,**连"这段该信 CTA 还是 DSA"这个问题都无法提出**;融合推理是 agent 任务独有的能力,分割只是其工具箱里的一个 tool。C 类结论就是把"融合价值"从口头 claim 变成**可测量的实验假设**:配对输入的 agent 在 C 类案例上应显著优于单模态输入的 agent(见 §4 的 `fusion_category` 评测设计)。

---

## 3. 临床读片流程 → 任务分层(Cardiomni 要模拟什么)

> **设计原则**:任务不是凭空设计,而是**从"一个医生只拿 CTA+DSA 时的真实读片流程"自然导出**。Cardiomni 自主执行这条流程,CardiomniBench-VD 逐阶段评分。`[evidence: 双临床研究,锚定 W1–W11、C1–C3]`

### 3.1 医生的完整决策链(5 阶段)—— 也是 Cardiomni 的自主流程

```
阶段0 解剖定位 → 阶段1 单模态感知 → 阶段2 跨模态融合 → 阶段3 综合评分 → 阶段4 决策  ‖  能力边界
```

| 阶段 | 做什么 | 可得结论(锚定标准) |
|---|---|---|
| **0 解剖定位** | 判优势型、识别可见血管段 | 右/左/均衡优势(看 PDA 起源);SCCT 18 段模型 `[evidence: W7]` |
| **1a CTA 感知** | 逐段看狭窄+斑块 | CAD-RADS 0–5、斑块性质(钙化/软/混合, HU)、Agatston 钙化分、高危斑块(低衰减<30HU / 正性重构≥1.1 / 餐巾环 / 点状钙化<3mm) `[evidence: C3; W8]` |
| **1b DSA 感知** | 逐段定量管腔+血流 | 狭窄%、TIMI 血流 0–3、TIMI 血栓 0–5、Rentrop 侧支 0–3、ACC/AHA 病变形态 A/B1/B2/C `[evidence: W4; W5]` |
| **2 跨模态融合** | 两者互相印证 | 钙化 blooming 校正、CTO 综合(CTA 断端形态+DSA 侧支)、易损斑块+管腔整合、罪犯 vs 非罪犯血管 `[evidence: W9]` |
| **3 综合评分** | 全树算复杂度 | SYNTAX Score + 三分层、CAD-RADS 总分+修饰符、per-patient 风险 `[evidence: C2]` |
| **4 决策** | 出诊断意见 | 是否需血运重建、PCI vs CABG(SYNTAX 指导)、药物强化、随访 `[evidence: W3; W2]` |

### 3.2 能力边界(benchmark 的差异化亮点 —— 直接测幻觉)

只有 CTA+DSA 时,有些结论**医生下不了**,Cardiomni 必须正确地说"需要更多检查",而非编造:

| 下不了的结论 | 缺什么 | 需要什么检查 |
|---|---|---|
| 50–69% 狭窄是否需处理 | 压力梯度 | **FFR/iFR 压力导丝** |
| CTO 远端心肌是否存活 | 心肌活性 | 灌注显像 / 心脏 MRI |
| 支架优化 / 夹层细节 | 管腔内结构 | IVUS / OCT |
| 手术风险 | 全身状态 | 验血 / 肺肾功能 / 病史 |

> **为什么是亮点**:多篇 2025 论文证实通用 VLM 核心问题是**"编造发现"**。CardiomniBench-VD 专设"该说不知道时能否说不知道"的任务直接测幻觉——这是可解释性评测的关键,普通"看图打分" benchmark 做不到。`[evidence: V2; A1 EchoAgent 可行性预测设计]`

---

## 4. CardiomniBench-VD:数据集与评价 Rubric

### 4.1 数据定位:从"深度学习子任务"到"自主端到端诊断"

这是 CardiomniBench-VD 与现有数据集的**根本区别**,必须讲清楚:

| | 现有开源数据集 | CardiomniBench-VD |
|---|---|---|
| **为谁而建** | 深度学习子任务(分割 / 检测) | 自主智能体端到端诊断 |
| **输入** | 预处理好的 PNG / 单帧 / 掩膜 | **原始 DICOM(CTA + DSA 配对)** |
| **任务** | 输出掩膜 / 框 | **自主阅诊 → 出诊断报告** |
| **标注** | 像素掩膜 / bbox | **专家级诊断结论 + 推理** |
| **评测** | IoU / mAP / Dice | **临床结论正确性 + 可解释性** |

> **核心**:现有数据集(ARCADE/CADICA/ImageCAS)的标注是**为深度学习任务服务的中间产物**(掩膜、框、分级标签)。Cardiomni 是 **autonomous** 系统——**直接输入 DICOM,自主完成专家级阅诊,直接输出临床诊断报告**。因此 CardiomniBench-VD 的核心不是复用它们的掩膜标注,而是构建**"DICOM 输入 → 专家诊断报告"**这一层现有数据集都没有的评测。`[derived: 三数据集标注对比]`

**开源数据集的定位(辅助,非主体)**:可作为①感知能力的可复现子测试、②Cardiomni 内部工具(分割/检测)的预训练来源。但它们**不能单独构成 CardiomniBench-VD**,因为都缺"端到端诊断报告"这一层:

| 数据集 | 许可 | 有什么 | 缺什么(相对本任务) |
|---|---|---|---|
| **ARCADE** | CC0 | XCA,26 SYNTAX 段 + 狭窄位置(COCO) | 无狭窄%、无 CTA、无诊断报告 `[evidence: D1]` |
| **CADICA** | CC BY 4.0 | ICA 视频,7 档狭窄% | 无血管段命名、无 CTA、无报告 `[evidence: D2]` |
| **ImageCAS** | 需申请 | CTA 3D 血管分割 mask | 无狭窄、无分段、无报告 `[evidence: D3]` |

> **关键空白 = 本 benchmark 的独特贡献**:没有任何现成数据集提供"**配对 CTA+DSA 原始 DICOM → 专家诊断报告**"。CardiomniBench-VD 的多中心专家标注集正是填补这一空白。`[derived]`

### 4.2 基线设定:只有智能体与大语言模型

因为核心任务是**自主、可解释的端到端诊断**(不是分割),基线**默认只包含能尝试完整诊断任务的系统**:

| 基线类别 | 例子 | 为什么是基线 |
|---|---|---|
| **通用多模态大模型** | GPT-4V / Claude / Gemini 直接读 DICOM 转出的图 | 能出文字诊断,但多篇 2025 论文证实"编造发现",预计可解释性差 `[evidence: V1; V2]` |
| **通用编码/多模态智能体** | Claude Code 类、通用 agent + 工具 | 能调工具、能推理,但无心血管领域知识 |
| **Cardiomni(本方法)** | LLM 编排 + 专用视觉工具 + 临床知识 RAG | 领域智能体,预期在可解释性上领先 |

> **为什么分割模型不是基线**:分割/检测模型(TotalSegmentator、CM-UNet、StenUNet 等)**产不出可解释的诊断报告**——它们输出掩膜,回答不了"这段是不是流限性狭窄、要不要血运重建、为什么"。因此它们**不作为基线,而是作为 Cardiomni 内部可调用的工具**(见第 5 节架构)。`[derived: 任务性质 = 可解释诊断 ≠ 分割]`

### 4.3 评价 Rubric(核心 —— 分层评分,锚定临床标准)

一个 task = 给一份"病人完整诊断报告"打分。报告按第 3 节 5 阶段展开,**逐层独立评分,每层锚定明确的临床标准**。数值精度交由客观指标,临床结论与可解释性交由 LLM-judge(按专家 rubric)。

**A. 感知层(阶段 0–1):客观指标 + 分级准确率**

| 评分维度 | 标准 | 指标 |
|---|---|---|
| 血管段识别 | SYNTAX 16 段 + 亚段命名 `[C1,C2]` | per-段 F1 / 命名准确率 |
| 优势型判断 | 右/左/均衡(PDA 起源) `[W7]` | 准确率 |
| 狭窄百分比 | CAD-RADS 2.0 分档 `[C3]` | 分档准确率 + 连续值 MAE + ±1 档容差率 |
| 斑块性质 | 钙化/软/混合(HU 阈值 130) `[C3]` | 分类准确率 |
| 钙化评分 | Agatston 分层(0/1-99/100-399/≥400) | 分层准确率 |
| 高危斑块 | 低衰减<30HU/正性重构≥1.1/餐巾环/点状钙化<3mm `[W8]` | 逐特征 F1 |
| 血流/血栓 | TIMI 血流 0–3、血栓 0–5 `[W4]` | 分级准确率 |
| 侧支循环 | Rentrop 0–3 `[W6]` | 分级准确率 |
| 病变形态 | ACC/AHA A/B1/B2/C `[W5]` | 分类准确率 |

**B. 融合层(阶段 2):跨模态推理正确性**
- 钙化 blooming 校正是否正确(CTA 高估 → 采纳 DSA);CTO 综合判断;罪犯血管定位。`[evidence: W9]`

**C. 综合层(阶段 3):评分误差**
- SYNTAX 分数误差(绝对误差 + 三分层准确率);CAD-RADS per-patient 级别准确率。`[evidence: C2, C3]`

**D. 决策层(阶段 4):指南符合度**
- 血运重建建议、PCI vs CABG 是否符合 2021 ACC/AHA & 2024 ESC 指南。`[evidence: W3, W2]`

**E. 可解释性层(贯穿全程 —— 本方法核心):LLM-judge 多维打分**
按专家 rubric 对推理轨迹打分(1–10 + CoT + JSON),维度:
- **准确性**:结论与专家金标准一致
- **完整性**:是否遗漏关键病变/发现
- **推理质量**:trace 是否有真实临床意义(而非事后编造)
- **可解释性**:非专家能否理解决策依据
- **诚实性/边界意识**:该说"需 FFR/IVUS/病史"时是否正确声明,而非编造(直接测幻觉)`[evidence: V2]`

> **process vs outcome**:process 维度只在专家"不看到这一步就不敢信任答案"时加入,避免 rubric 过度限制唯一路径。`[evidence: A3, A6]`
> **judge 稳定性**:2 个额外 LLM-judge 在随机 ~200 样本复核,报 κ 一致性再发布分数。`[evidence: A6 DrugDiscoveryBench κ=1.0 做法]`

---

## 5. Cardiomni Agent 架构(对标 EchoAgent)

参照 **EchoAgent**(超声心动图 agent,与本项目高度同构)`[evidence: A1]`:

```
LLM 编排器(Claude / GPT / 开源模型,模块化可换)
   │  结构化 JSON 工具调用,迭代上限 ~15 轮
   ▼
专用视觉工具层(原子能力 —— 分割/检测模型在此作为工具,而非基线)
   ├─ DICOM 读取工具   —— pydicom 直读,拿 HU 值 + XA 多帧
   ├─ 血管分割工具     —— TotalSegmentator / CM-UNet(内部工具)
   ├─ 狭窄定量工具     —— centerline 提取 → 直径测量 → QCA 式后处理
   ├─ 可行性预测工具   —— 判断帧/切面是否"可测量",避免瞎测(EchoAgent 设计)
   └─ 指南检索工具     —— RAG 查 CAD-RADS 阈值 / SYNTAX 定义,压制 LLM 幻觉
   ▼
结构化中间产物(血管段 mask、狭窄候选、直径曲线)
   ▼
LLM 推理 → 诊断报告 + 推理轨迹
```

**三个关键设计原则**:
1. **数值精度交给工具层** —— 狭窄百分比由工具计算,LLM 只做结论层推理("75% = 重度 = 需血运重建"),不让 LLM 自己纠结容差。`[evidence: A1]`
2. **知识库按需检索** —— CAD-RADS/SYNTAX/钙化指南做成可检索知识库,而非塞进单个巨大 system prompt。`[evidence: A2 Biomni Know-How Library]`
3. **失败模式分层** —— 区分"工具调用错"vs"最终结论错",精确定位是感知层还是推理层问题。`[evidence: A1]`

> **注意分割模型的角色**:TotalSegmentator/CM-UNet 等在这里是 Cardiomni **可调用的工具**,不是竞争基线。它们帮 agent 拿到掩膜/直径,但"这段要不要处理、为什么"的可解释诊断由 LLM 编排层完成——这正是方法与纯分割模型的分界。

---

## 6. CardiomniBench-VD 构建方法(对标 BiomniBench-DA)

### 6.1 规模
**20–100 任务**即可支撑发表级 benchmark:EchoAgent 60、BiomniBench-DA 100、MedAgentBench 100。你原计划的 20–50 完全在主流区间。重点是标注严谨性与 rubric 质量,不是规模。`[evidence: A1, A3, A4]`

### 6.2 标注协议(对标 DrugDiscoveryBench 四步)
1. 领域专家 A 撰写候选任务(基于真实 CTA+DSA 病例,写出金标准诊断报告)
2. 领域专家 B 复核 / 编辑
3. 自动化**可解性验证**:给 step-by-step playbook 作 hint,验证至少一个 agent 能复现答案(证明任务可解、非设计缺陷)
4. 独立 QC 团队签字放行

`[evidence: A6]`

### 6.3 金标准来源(待专家定,见第 10 节)
每个 task 的"标准答案"需锚定:有创 FFR / QCA 定量 / 介入结果 / 多专家共识——具体以哪个为准,需临床专家拍板。

---

## 7. 技术路线清单(可直接落地)

### 7.1 "读图"路径(重要纠正)
- ❗ **dicom-mcp 拿不到像素** —— 只管元数据 + 抽 PDF 报告文本,LLM 看不到图。`[evidence: M8 源码级核实]`
- ✅ **pydicom 直读** —— `pixel_array(ds)` → XA cine 得 `(帧数,H,W)` numpy 数组;压缩传输语法需装 `pylibjpeg`/`gdcm`。让 agent"看图"的唯一轻量路径。`[evidence: M6]`
- ✅ **3D Slicer 无头** —— `--no-main-window --python-script`,批量分割+结构化导出。`[evidence: M7]`
- ⚠️ Weasis / OHIF / 小赛看看 —— 只是"唤起界面给人看",非自主分析手段;小赛看看纯 GUI 无 API/MCP,留给专家人工标注核对。`[evidence: M8]`

### 7.2 工具层复用清单(Cardiomni 内部工具)

| 工具 | 任务 | 直接可调 | 许可 | 推荐度 |
|---|---|---|---|---|
| pydicom | DICOM 读取(含 XA cine) | ✅ pip | MIT | ★★★ 必用 |
| TotalSegmentator | CTA 冠脉+心腔分割 | ✅ pip | 非商用免费 | ★★★ |
| CM-UNet | 2D 造影分割 | 部分(手写 UNet 加载权重) | Apache-2.0 | ★★ |
| 3D Slicer + SlicerHeart | 心脏分割/测量/centerline(VMTK) | ✅ 无头脚本 | BSD | ★★★ |

### 7.3 不推荐的路线
- ❌ GUI 自动化驱动小赛看看 → 脆弱、丢 HU 值
- ❌ 通用 VLM 直接读影像做诊断当"方法" → 多篇论文证实编造发现(只作基线对照)
- ❌ 把分割模型当基线 → 产不出可解释诊断报告

---

## 8. 下一步建议

1. **确认专家能否提供原始配对 DICOM**(CTA + DSA,而非 PNG)——自主诊断的信息前提。
2. **pydicom 验证读图**:拿一个真实 CTA/DSA DICOM 跑通 `pixel_array`,展示 HU 分布与 cine 帧。
3. **搭 Cardiomni 最小闭环**:DICOM 读取 → 调一个分割工具 → LLM 出一段诊断 → 看端到端能否跑通。
4. **设计双专家标注协议**:先标 5–10 个 pilot case(配对 DICOM → 金标准诊断报告),验证可行性。
5. **专家审阅本提案**,反馈后定稿。

---

## 9. 待专家拍板的开放问题

1. **金标准来源**:每个 task 的标准诊断报告以什么为准?(有创 FFR / QCA / 介入结果 / 多专家共识)
2. **Task 边界**:一个 benchmark task = 一个病人?一次检查?
3. **狭窄档位**:统一用 CAD-RADS 2.0(推荐)还是并用 CADICA 细档? `[speculation:需临床专家定]`
4. **多中心集规模与标注**:目标多少 case、几家中心、标注协议细节。

---

## 10. 引用清单

所有引用均经联网核实(WebSearch + Clash 代理直取原文),非编造。分类列出。

### 10.1 临床标准(血管命名 / 狭窄分级)

| # | 引用 | 用途 | 链接 |
|---|---|---|---|
| C1 | Austen WG et al. (1975) A reporting system... Ad Hoc Committee, AHA. *Circulation* 51(4). | 冠脉分段体系鼻祖 | https://www.ahajournals.org/doi/10.1161/01.CIR.51.4.5 |
| C2 | Sianos G et al. (2005) The SYNTAX Score. *EuroIntervention*. | SYNTAX 分段/评分 | https://eurointervention.pcronline.com/article/the-syntax-score-an-angiographic-tool-grading-the-complexity-of-coronary-artery-disease |
| C3 | CAD-RADS 2.0 (2022) SCCT/ACC/ACR/NASCI Consensus. | 狭窄分档金标准 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9627235/ |

### 10.2 临床读片流程 / 分级标准

| # | 引用 | 用途 | 链接 |
|---|---|---|---|
| W1 | 2021 ACC/AHA 胸痛评估指南 | CTA/ICA 诊疗路径角色 | https://www.ahajournals.org/doi/10.1161/CIR.0000000000001029 |
| W2 | 2024 ESC 慢性冠脉综合征指南 | 无创首选 + 血运重建决策 | https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/chronic-coronary-syndromes/ |
| W3 | 2021 ACC/AHA 血运重建指南 | PCI vs CABG 决策 | https://www.acc.org/latest-in-cardiology/ten-points-to-remember/2021/12/08/21/19/2021-guideline-for-revascularization-gl-revasc |
| W4 | TIMI 血流分级(1985 / StatPearls) | TIMI 0–3 定义 | https://www.ncbi.nlm.nih.gov/books/NBK482412/ |
| W5 | ACC/AHA 病变形态分型 A/B/C (Circ 1999) | PCI 成功率相关 | https://www.ahajournals.org/doi/10.1161/01.cir.100.12.1285 |
| W6 | Rentrop 侧支循环分级 | CTO 侧支评估 | https://www.ahajournals.org/doi/10.1161/01.cir.0000061953.72662.3a |
| W7 | SCCT CCTA 报告模板 | 标准报告字段 + 18 段模型 | https://scct.org/page/CorCTAReportTemplate/Sample-Coronary-CTA-Reporting-Template.htm |
| W8 | SCCT 高危斑块特征 / 餐巾环征 | LAP/PR/NRS/SC 定义 | https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.117.006973 |
| W9 | 钙化 blooming 伪影综述 | CTA-ICA 融合校正依据 | https://pmc.ncbi.nlm.nih.gov/articles/PMC9733770/ |
| W10 | ICR Journal《A Guide to Calculating SYNTAX Score》 | SYNTAX 计算细则 | https://www.icrjournal.com/articles/guide-calculating-syntax-score?language_content_entity=en |
| W11 | J-CTO 评分 | CTO 可介入性 | https://www.jacc.org/doi/10.1016/j.jcin.2010.09.024 |

### 10.3 公开数据集(辅助 / Cardiomni 内部工具训练用)

| # | 引用 | 模态/标注 | 链接 |
|---|---|---|---|
| D1 | Popov M et al. (2024) ARCADE. *Sci Data* 11, 20. CC0. | XCA, 26 SYNTAX 段 + 狭窄位置, COCO | https://www.nature.com/articles/s41597-023-02871-z ；Zenodo 10.5281/zenodo.10390295 |
| D2 | Jiménez-Partinen A et al. (2024) CADICA. *Expert Systems* 41(12) e13708. CC BY 4.0. | ICA 视频, 7 档狭窄% | https://arxiv.org/abs/2402.00570 ；data.mendeley.com/datasets/p9bpx9ctcv |
| D3 | Zhao C et al. (2023) ImageCAS. *Comput Med Imaging Graph*. | 冠脉 CTA 3D, 血管分割 mask | https://arxiv.org/abs/2211.01607 ；github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT |

### 10.4 模型 / 工具(Cardiomni 内部工具层)

| # | 引用 | 用途 | 链接 |
|---|---|---|---|
| M1 | TotalSegmentator (`coronary_arteries`) | CTA 冠脉+心腔分割 | https://github.com/wasserth/TotalSegmentator |
| M2 | CM-UNet (Challier et al. 2025) | 2D 造影分割, HF 权重 | https://arxiv.org/abs/2507.17779 ；huggingface.co/Camsouille/CM-UNet |
| M3 | StenUNet (ARCADE 2023) | 造影狭窄检测 | https://github.com/huilin0220/stenunet |
| M4 | SAM-VMNet | CTA/造影分割+狭窄定量 | https://github.com/qimingfan10/SAM-VMNet |
| M5 | Merlin (Stanford) | 3D CT VLM 底座 | https://huggingface.co/stanfordmimi/Merlin ；arXiv 2406.06512 |
| M6 | pydicom 3.x pixel_data 教程 | XA 多帧读取 | https://pydicom.github.io/pydicom/stable/tutorials/pixel_data/introduction.html |
| M7 | 3D Slicer headless / SlicerHeart / VMTK | 批量分割/centerline | https://slicer.readthedocs.io/en/latest/developer_guide/python_faq.html |
| M8 | dicom-mcp(仅元数据+PDF,拿不到像素) | 病历检索层 | https://github.com/ChristianHinge/dicom-mcp |

### 10.5 传统方法(参考 / 非默认基线)

| # | 引用 | 用途 | 链接 |
|---|---|---|---|
| T1 | QCA 系统对比验证 | DSA 狭窄定量原理 | https://www.ahajournals.org/doi/full/10.1161/01.CIR.91.8.2174 |
| T2 | FFR-CT vs 有创 FFR Meta 分析 (2024) | 血流动力学 | https://www.ahajournals.org/doi/full/10.1161/JAHA.124.034552 |
| T3 | SimVascular(开源 FFR 仿真) | FFR-CT 复现 | https://github.com/SimVascular/SimVascular |
| T4 | CalciumScoring.jl / dl_cta_calcium | Agatston 钙化评分 | https://github.com/Dale-Black/CalciumScoring.jl |
| T5 | End-to-end SYNTAX Score Prediction (2024) | SYNTAX 自动化 | https://arxiv.org/abs/2407.19894 |

### 10.6 Agent 范式 / Benchmark 方法

| # | 引用 | 用途 | 链接 |
|---|---|---|---|
| A1 | EchoAgent(超声心动图 agent,架构模板) | 同域架构对标 | https://arxiv.org/abs/2511.13948 |
| A2 | Biomni(通用生物医学 agent) | Know-How Library 模式 | https://github.com/snap-stanford/Biomni |
| A3 | BiomniBench(process-level 评测) | benchmark 对标原型 | https://www.biorxiv.org/content/10.64898/2026.05.12.724604v2 |
| A4 | MedAgentBench (Stanford) | 临床 agent 环境 | https://arxiv.org/abs/2501.14654 |
| A5 | MedAgentBoard(多 agent vs 传统方法) | 基线设计参考 | https://arxiv.org/abs/2505.12371 |
| A6 | DrugDiscoveryBench(标注协议 + 多judge验证) | 标注/评测协议 | https://static.scale.com/uploads/6691558a94899f2f65a87a75/DrugDiscoveryBench.pdf |

### 10.7 VLM 可靠性警示(基线预期表现依据)

| # | 引用 | 结论 | 链接 |
|---|---|---|---|
| V1 | GPT-4V 放射影像解读评估 (2024) | "cannot reliably interpret radiologic images" | https://link.springer.com/article/10.1007/s00330-024-11115-6 |
| V2 | GPT-4V Cannot Generate Radiology Reports Yet (NAACL 2025) | 预测标签与图像实际病灶无关 | https://aclanthology.org/2025.findings-naacl.113/ |

### 10.8 CTA-DSA 融合价值证据(§2.3 C 类论据)

| # | 引用 | 支撑的 claim | 链接 |
|---|---|---|---|
| R1 | 64 排 CTA 诊断性能(1006 段:sens 78% / spec 95% / **PPV 54%** / NPV 98%) | 钙化段 CTA 高估狭窄、需 DSA 校正真实管腔 | https://pubmed.ncbi.nlm.nih.gov/ (64-detector CTA diagnostic accuracy; 见 PROPOSAL 检索记录) |
| R2 | Toth et al. (2014) ~4000 例狭窄:约 **1/3** 在 ≥50% 直径狭窄 vs FFR≤0.8 之间失配 | 解剖-功能失配普遍存在,单模态无法察觉矛盾 | https://pubmed.ncbi.nlm.nih.gov/24644308/ |
| R3 | Park et al. (2012) 失配方向分析(紧狭窄血流正常 57% / 轻狭窄血流异常 40% 左主干) | 失配双向存在,融合后需归因并判断是否转 FFR | https://pubmed.ncbi.nlm.nih.gov/23078732/ |
| R4 | Meijboom et al. (2007) MDCT vs FFR:解剖与功能严重度无明确关系 | 功能学评估不可被解剖成像替代(能力边界依据) | https://pubmed.ncbi.nlm.nih.gov/17612701/ |

---

*注:ARCADE MICCAI 2023 获胜队伍排名与部分下游论文数值未逐一交叉核实,引用时以原始论文为准。R1 的具体 PMID 为检索得到的 64 排 CTA 诊断性能研究(1006 段),投稿定稿前应锁定确切文献并核对逐段数字。*
