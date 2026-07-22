# Paper Reference — Biomni & BiomniBench 全量 Claim 清单

> 用途:为 CardiomniBench-VD(心血管 CTA/DSA 血管堵塞诊断 Agent + Benchmark)提供可对标、可引用的原文 claim 库。每条尽量保留原文数字与措辞。
>
> 两篇关系:配套双件套。**Biomni**(Huang 等,Stanford)提出通用生物医学 Agent;**BiomniBench**(Qu 等)对这类 Agent 做过程级(process-level)评测。BiomniBench 是 bioRxiv 预印本(2026-05-18),其模型命名含未来/占位版本号(如 Claude Opus 4.7、GPT-5.5、Gemini 3.1 Pro),此处照录原文。

---

# 文档 A:Biomni — A General-Purpose Biomedical AI Agent

(Huang, Gao, et al., Stanford。以下 claim 编号 C-A*)

## A0. 核心身份 / 定位 claim

- **C-A0.1** Biomni 是首个"通用型(general-purpose)"生物医学 AI Agent,可跨多个子领域自主执行大量、多样的研究任务;区别于以往"一个任务训一个专用模型"的碎片化范式。
- **C-A0.2** 系统由两个可命名、可独立引用的组件构成:**Biomni-E1**(环境 / 行动空间 environment)与 **Biomni-A1**(Agent 架构)。这是全文的骨架式命名法。
- **C-A0.3** 不训练新基座模型,走"编排 + 环境"路线,底座使用通用 LLM(Claude 系)。核心增益来自环境与编排,而非炼模型。
- **C-A0.4** 主张能显著提升生物医学研究的自动化程度与速度,充当研究者的"协作加速器"。

## A1. 环境 Biomni-E1 的规模 claim(精确数字)

- **C-A1.1** E1 通过"action discovery"从近期高影响力文献/综述蒸馏而来,覆盖 **25 个**生物医学子领域,每领域约 100 篇、合计约 **2,500 篇**论文。
- **C-A1.2** E1 包含 **150 个**专业工具(specialized tools)。
- **C-A1.3** E1 集成 **105 个**软件包(software packages)。
- **C-A1.4** E1 集成 **59 个**数据库(databases)。
- **C-A1.5** 方法论 claim:"工具从哪来"被形式化为一套半自动 + 专家参与的 action-discovery 流程(从文献中系统性挖掘研究者真正会用的动作),而非穷举 API。这是本文最核心的方法贡献之一。

## A2. Agent 架构 Biomni-A1 的机制 claim

- **C-A2.1** 采用**代码即行动(code-as-action / CodeAct 风格)**:Agent 自己生成并执行代码来编排 E1 中的工具,而非固定 API 调用。这让它能处理复杂数据流(读文件、跑统计、画图)。
- **C-A2.2** 具备**自适应规划**:先生成计划,再在执行中根据中间结果动态修正(generate + adaptively refine)。
- **C-A2.3** 开放可扩展生态:社区可持续贡献新工具与数据,形成飞轮(open, extensible)。

## A3. 实验设计 claim(三层递进结构)

- **C-A3.1** 第一层——标准化 benchmark:在 **LAB-Bench**(含 DbQA、SeqQA、ProtocolQA 等子任务)与 **Humanity's Last Exam (HLE)** 生物医学子集上评测。基线:base LLM、LLM+coding(coding agent)、Biomni-ReAct(同环境但用 ReAct 而非 code-act)。
- **C-A3.2** 第二层——**8 个真实研究任务**(real-world research scenarios),由领域专家按质量打分,从"选择题正确率"迈向"开放式研究质量"。
- **C-A3.3** 第三层——若干**端到端案例研究**,证明真实科研价值与效率跃迁。
- **C-A3.4** 递进逻辑清晰:能做对(benchmark)→ 做得好(专家评审)→ 做出价值(案例)。

## A4. Benchmark 结果 claim(精确数字)

- **C-A4.1** 在 8 个真实任务上,Biomni 相对 base LLM 平均提升 **402.3%**。
- **C-A4.2** 相对 coding agent 提升 **43.0%**。
- **C-A4.3** 相对 Biomni-ReAct(同环境、ReAct 编排)提升 **20.4%**。
- **C-A4.4** 关键消融含义:在**相同环境**下 code-act 显著优于 ReAct(由 C-A4.3 支撑),证明"编排方式本身"带来增益,而非仅靠工具堆叠。
- **C-A4.5** 在 LAB-Bench(DbQA/SeqQA/ProtocolQA)与 HLE 生物医学子集上均优于基线(SeqQA、DbQA 领先尤为明显;精确数值见原文 Fig)。

## A5. 案例研究 claim(证明新能力)

- **C-A5.1** 案例覆盖真实端到端场景:遗传病诊断 / 变异解读、CRISPR 筛选设计、单细胞与组学分析、可穿戴传感数据分析等。
- **C-A5.2** 效率主张:原本需专家数天至数周的分析,Biomni 可在数分钟到数小时内完成端到端流程。
- **C-A5.3** Biomni 能自主选工具、串联多步分析、产出可解释的中间结果与图表。

## A6. Discussion / 安全与局限 claim

- **C-A6.1** 定位为研究者的协作者 / 加速器,而非替代专家;强调 human-in-the-loop。
- **C-A6.2** 明确局限:可能出错、结果需人工验证;将安全与负责任使用单列讨论。
- **C-A6.3** 开放生态主张:环境可被社区持续扩充,推动通用生物医学 Agent 的迭代飞轮。

## A7. 文章组织结构(供仿写)

- **C-A7.1** Nature 风格:Abstract → Introduction(碎片化痛点 + 通用 Agent 主张)→ Results(环境与架构总览 → benchmark → 专家评审 → 案例)→ Discussion → Methods(E1 构建、A1 架构、评测协议置于最后)。
- **C-A7.2** 图策略:一张总览图(环境 + 架构双组件)+ 每层实验一张主图。
- **C-A7.3** 叙事技巧:开篇先讲"研究者的痛"(跨越无数工具的迷宫),让"通用 Agent"显得是自然解药,再引出技术。

---

# 文档 B:BiomniBench — Process-level Evaluation of LLM Agents for Real-world Biomedical Research

(Qu 等。本文实际主体是子基准 **BiomniBench-DA**(DA = Data Analysis)。以下 claim 编号 C-B*)

## B0. 核心身份 / 定位 claim

- **C-B0.1** BiomniBench 是面向真实生物医学研究的 Agent **过程级(process-level)** 评测基准——不只评最终答案,更评"分析过程 / 推理轨迹"。
- **C-B0.2** 核心论点:真实研究中"方法错但蒙对答案 ≠ 好研究";结果级评测会高估 Agent 能力,必须过程级评测。
- **C-B0.3** 本文放出的首个子基准为 **BiomniBench-DA**,聚焦"数据分析"这一最主导的真实研究意图。
- **C-B0.4** 框架可评测多个 Agent / 多个底座模型(非绑定 Biomni 一家)。

## B1. Benchmark 构成 claim(精确数字)

- **C-B1.1** BiomniBench-DA 共 **100 个**数据分析任务(tasks)。
- **C-B1.2** 覆盖 **17 种**任务类型(task types)。
- **C-B1.3** 覆盖 **5 个**疾病领域(disease areas)+ 通用生物学(general biology)。
- **C-B1.4** 任务源自 **21 篇**已发表的高影响力论文(真实、可溯源;含 Cell、Nature、Science 系列等)。
- **C-B1.5** 一篇论文派生多个任务(1 → 多),每个任务对应论文中一个可复现的分析问题。
- **C-B1.6** 疾病领域分布:肿瘤/癌症占比最高(近半,含结直肠癌、黑色素瘤等),其余为代谢与内分泌(如 gender-affirming 激素治疗)、免疫、神经、心血管等 + 通用生物学。
- **C-B1.7** 任务类型分布:association testing、mutation analysis、differential expression、survival analysis、enrichment/pathway、cell-type annotation 等六类为主,长尾覆盖其余类型。
- **C-B1.8** 选题代表性论证:分析了 **32,014 条**真实 Biomni 用户查询,发现"数据分析(data analysis)"是最主导意图,以此论证首个子基准聚焦 DA 并非任意选择。

## B2. 评测机制 claim

- **C-B2.1** Agent 每个任务须产出两类产物:完整分析**轨迹(trace,如 trace.md / 执行记录)**与**最终答案(answer)**;评测同时审两者。
- **C-B2.2** 每个任务配一份由领域专家编写的 **rubric(评分量表)**,总分归一到 100。
- **C-B2.3** 每条 rubric 标准分 **A / B / C 三档**,各档预设分值(A=满分档、B=部分、C=最低/零),由专家按重要性配权。
- **C-B2.4** 全基准合计约 **数百条**评分标准(每任务约 5–10 条),覆盖端到端分析流程。
- **C-B2.5** 每条标准打上**维度标签**,聚合成 **6 个维度**级得分(用于诊断,见 B3)。
- **C-B2.6** 设**反幻觉 / 负分标准**:如"来源可靠性 / source reliability"的最低档可为**负分**,专门惩罚编造引用、假 p 值、凭空标签、幻觉结论。
- **C-B2.7** 受控执行环境:Agent 拿到数据文件 + 问题描述,环境预装 Python/R 与常用包、允许联网装包,但**明令禁止检索或阅读原始论文**——必须靠数据本身 + 领域知识独立完成。

## B3. 六个评测维度 claim(供 CardiomniBench 直接对标)

- **C-B3.1** 六维度(措辞近似原文 Figure 1):
  1. **Data Handling** 数据加载 / 预处理 / 质控;
  2. **Method Selection** 方法与统计检验的选择是否恰当;
  3. **Statistical Rigor / Execution** 统计执行与结果报告是否正确;
  4. **Biological Interpretation** 生物 / 临床解释是否到位;
  5. **Scientific Reasoning** 科学推理链是否严谨、可辩护;
  6. **Source Reliability** 来源可靠性 / 反幻觉(唯一含负分档)。
- **C-B3.2** 维度分解的价值主张:把 benchmark 从"排名工具"升级为"诊断工具"——精确定位 Agent 在**哪一步垮掉**(数据?方法?解释?)。

## B4. LLM-Judge 验证 claim(本文最核心的科学性,务必仿照)

- **C-B4.1** 用 LLM 作评委(judge)自动按 rubric 打分,但**不默认其可靠**;先做评委可信度验证再用其评测。
- **C-B4.2** 验证方式:LLM-judge 打分 vs **人类专家金标准**打分做逐条标准对齐。
- **C-B4.3** 报告一致性指标:**Cohen's κ**(含加权版本)与 **exact accuracy(逐档精确匹配率)**。
- **C-B4.4** 比较多个候选评委模型,选一致性最高者作正式评委;原文给出 **Gemini 3.1 Pro** 最优(exact accuracy 约 **0.82**),据此选为正式评委。
- **C-B4.5** 分析各评委的**系统性失效模式**:例如某些模型把 B 档塌缩进 A 档(过于宽松)、或对负分档判定不稳。
- **C-B4.6** 科学表达法总结:**先证明"尺子准"(judge validation),再用尺子量(Agent 评测)**——这是整套 benchmark 可信度的地基。

## B5. Agent 评测结果 claim(精确数字,Table 1/2)

- **C-B5.1** 评测覆盖前沿闭源模型(如 Claude Opus 系、GPT-5.5、Gemini 系)与开源权重模型,配 **4 种 agent harness / scaffold**。
- **C-B5.2** 发现一:即便最强模型,总分仍**远低于满分**(整体偏低),说明真实研究型数据分析对当前 Agent 仍很难。
- **C-B5.3** 发现二:**过程分显著低于结果 / 答案分**——Agent 常"答案看似对、过程站不住",印证过程级评测的必要(呼应 C-B0.2)。
- **C-B5.4** 发现三:失分集中在**方法选择**与**统计严谨性 / 生物解释**维度,而非数据加载;即"跑得动"不等于"做得对"。
- **C-B5.5** agent scaffold / 编排方式对得分有明显影响(同底座不同 harness 差异大)。

## B6. 失效模式案例研究 claim(定性,Appendix)

- **C-B6.1** 用真实案例展示"失败长什么样",而非只给数字。
- **C-B6.2** 典型失效:**方法误用**(如用 Kruskal-Wallis 冒充趋势检验 / 选错统计模型)。
- **C-B6.3** 典型失效:**生物 / 临床解释脱靶**(统计结果对但解读错误方向)。
- **C-B6.4** 典型失效:**科学推理链断裂**(中间步骤跳步、结论无法由过程支撑)。
- **C-B6.5** 典型失效:**幻觉**(编造引用 / 数值 / 标签),对应负分维度。

## B7. 相关工作 / 局限 / 结构 claim(供仿写)

- **C-B7.1** 与既有 benchmark 区分:多数只做**结果级、封闭式(选择题/QA)**评测;BiomniBench 做**开放式、过程级、rubric 驱动**评测。
- **C-B7.2** 声明局限:首版仅覆盖数据分析(DA)子域;任务量与疾病覆盖仍有限;评委仍是 LLM(尽管已验证)。
- **C-B7.3** 结构:Abstract → Intro(为何需要过程级)→ Benchmark 构建(任务来源 / rubric / 维度)→ Judge 验证 → Agent 结果 → 失效模式 → 相关工作 → 局限 → 附录(完整 rubric 示例、任务清单 Table、判分 prompt)。
- **C-B7.4** 附录含**一个完整 rubric 全文示例**(逐条 A/B/C 分值)与**评委 prompt**,保证可复现——强可复现性是其被引用价值之一。

---

# 文档 C:对 CardiomniBench-VD 的可对标映射(桥接层)

> 这一层不是原文 claim,而是把 A/B 两篇的每个可复用点映射到你的心血管项目,供后续设计直接取用。

## C1. 结构与命名对标

- **M-1** Biomni 双组件命名法 → 你的 **Cardiomni-Env**(DICOM 解析 / 血管分割 / 中心线提取 / 狭窄量化 / 钙化评分 / 报告结构化等原子工具)+ **Cardiomni-A1**(多模态感知 + 代码/工具编排 + 报告生成)。对标 C-A0.2。
- **M-2** 三层实验递进(标准指标 → 专家评审 → 真实案例)可整体照搬。对标 C-A3.1~3.3。
- **M-3** Benchmark 用 Biomni 的"配套第二篇"定位,但初期建议合并入方法文章的实验章(你只有一个 Agent,独立成篇易被质疑"自出自考")。对标两篇关系。

## C2. 评测科学性对标(护城河,来自 BiomniBench)

- **M-4** 过程级 > 结果级:血管诊断"报告结论对但狭窄定位/量化错"极危险,必须评推理轨迹。对标 C-B0.2、C-B5.3。
- **M-5** Rubric + A/B/C 分档 + 维度聚合。对标 C-B2.2~2.5。
- **M-6** 六维度改写建议:① 影像感知/DICOM 处理;② 血管识别与解剖定位;③ 狭窄量化准确性(对齐 FFR/直径狭窄率);④ 临床分级与风险判断;⑤ 诊断推理严谨性;⑥ 来源可靠性/反幻觉(负分)。对标 C-B3.1。
- **M-7** LLM-Judge 验证(最关键,勿省):judge 打分 vs 放射科/心内科专家金标准,报告 Cohen's κ 与 exact accuracy,选最优评委。对标 C-B4.1~4.6。
- **M-8** 反幻觉负分项:"编造病灶/量化值/分级即重扣分"。对标 C-B2.6。
- **M-9** 失效模式定性案例(错误血管诊断长什么样)。对标 C-B6.*。
- **M-10** 选题代表性论证:用真实临床需求/查询分布证明聚焦血管堵塞诊断非任意。对标 C-B1.8。
- **M-11** 受控环境:给 CTA/DSA + 问题,禁止检索原始报告/诊断结论,靠影像与知识自解。对标 C-B2.7。

## C3. 两篇未覆盖、心血管影像域必须自行补足的点

- **G-1** 多模态融合:CTA(3D 体数据)与 DSA(2D 时序造影)如何在 Agent 内对齐/互补——你的独特技术点,需单列。
- **G-2** 金标准来源:FFR、专家共识、随访结局——决定 Benchmark 可信度,需在 Methods 明确。
- **G-3** 临床安全/监管:错误诊断后果、human-in-the-loop、辅助 vs 自动定位。
- **G-4** 数据合规:DICOM 去标识化、伦理审批,需在 Methods 交代。

## C4. 引用注意

- BiomniBench 为预印本,模型命名含占位/未来版本号(Claude Opus 4.7、GPT-5.5、Gemini 3.1 Pro),引用其具体分数时注明版本与日期。
- Biomni 的 402.3% / 43.0% / 20.4% 三个提升是相对不同基线(base LLM / coding agent / ReAct),引用时务必指明分母,避免误用。
