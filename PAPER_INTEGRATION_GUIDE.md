# Methods Library Integration Guide for Paper Writing

**Target**: AnonymousSubmission2027.tex  
**Date**: 2026-07-24

---

## Quick Stats for Introduction

Use these numbers to establish the field's maturity and the specialist-model baseline:

```latex
Coronary angiography analysis has seen rapid progress, with over 30 deep learning 
methods published since 2020 addressing vessel segmentation~\citep{samvmnet2024,cmunet2025,vmcagseg2025}, 
stenosis detection~\citep{arcade2023,ltyo lo2025}, and SYNTAX scoring~\citep{cardiosyntax2024,messermmp2024}.
These specialist models—trained on datasets ranging from hundreds to 200K+ videos—now 
report high single-task performance: IoU 0.63 for vessel segmentation~\citep{samvmnet2024}, 
AUROC 0.888 for stenosis detection~\citep{deepcoro2026}, and MAE 13.6\% for stenosis 
quantification against core-lab QCA~\citep{deepcoro2026}.
```

---

## Related Work Section Structure

### Section 2.1: Automated Coronary Angiography Analysis

**Current content** (lines 202-203):
- ✅ Already cites: CathAI, DeepCORO-CLIP, CardioSYNTAX, ARCADE/SAM-VMNet
- ✅ Already positions them as **specialist models** (not competitors)

**Additions needed**:
1. **Add classical baseline mention** (1 sentence):
   ```latex
   Early approaches applied classical computer vision techniques—Frangi vesselness 
   filtering~\citep{frangi1998multiscale} and active contours—but required extensive 
   hand-tuning and struggled with overlapping structures and noise.
   ```

2. **Add trajectory narrative** (expand deep learning paragraph):
   ```latex
   Deep learning revolutionized vessel segmentation: U-Net~\citep{ronneberger2015unet} 
   established the encoder-decoder paradigm; nnU-Net~\citep{isensee2021nnunet} brought 
   self-configuring architectures; and recent Transformer~\citep{transcc2023} and 
   graph neural network~\citep{gnn_coronary_labeling2022} methods capture long-range 
   dependencies and topological structure. Foundation models now generalize across 
   domains: SAM~\citep{kirillov2023sam} and its medical adaptations~\citep{medsam2023,medsam2_2024} 
   enable zero-shot segmentation, while video-text models like DeepCORO-CLIP aggregate 
   multi-view projections and generate preliminary reports.
   ```

3. **Current positioning is perfect** — keep:
   > "These are all \emph{specialist models}: each requires large-scale domain training, 
   > exposes limited interpretability... \methodname{} treats these models not as competitors 
   > but as \emph{callable tools} and reference upper bounds."

### Section 2.2: Agent Benchmarks
**Status**: ✅ Already well-written (SWE-bench, MLE-bench, MLAgentBench, BixBench)

### Section 2.3: Medical Agents and the Cardiovascular Gap
**Status**: ✅ Already positions the gap

---

## Experiments Section Integration

### Table 1: Main Results (Harness Comparison)

**Format** (placeholder table in line 264):
```latex
\begin{table}[t]
\centering
\caption{Harness comparison on CardiomniBench-VD. All harnesses use the same base model 
(Claude Opus 4.8), tool library, and task set. Specialist models (bottom rows) are 
reported as reference upper bounds trained on 195K--203K videos, not competing harnesses.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Method & Stenosis MAE & Coverage & Naming & Trace \\
\midrule
\multicolumn{5}{l}{\textit{Agent Harnesses (zero-training):}} \\
Naive Tool-Caller & [TBD] & [TBD] & [TBD] & [TBD] \\
Claude Code & [TBD] & [TBD] & [TBD] & [TBD] \\
Codex & [TBD] & [TBD] & [TBD] & [TBD] \\
\textbf{Cardiomni (ours)} & [TBD] & [TBD] & [TBD] & [TBD] \\
\midrule
\multicolumn{5}{l}{\textit{Reference upper bounds (specialist models):}} \\
DeepCORO-CLIP$^{\dagger}$ & 13.6\% & — & — & — \\
CathAI$^{\dagger}$ & — & AUC 0.862$^{\ddagger}$ & — & — \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item[$\dagger$] Trained on 195K--203K videos. Not comparable to zero-training harnesses.
\item[$\ddagger$] AUROC for $\geq$70\% stenosis on different test set.
\end{tablenotes}
\end{table}
```

### Section 3.1: Callable Tools

Add a short paragraph listing available tools:
```latex
\paragraph{Tool Library.} Following the BioML-Bench paradigm, we expose specialist 
models as callable tools via a unified API. Available tools include vessel segmentation 
(SAM-VMNet~\citep{samvmnet2024}, CM-UNet~\citep{cmunet2025}), stenosis detection 
(ARCADE baselines~\citep{arcade2023}), SYNTAX scoring (CardioSYNTAX~\citep{cardiosyntax2024}, 
MesserMMP~\citep{messermmp2024}), and the DeepCORO-CLIP foundation model~\citep{deepcoro2026}. 
All harnesses have equal access to this library; performance differences reflect 
orchestration, not available tools.
```

---

## Citations to Add to aaai2027.bib

**Critical additions** (copy from `methods_library.bib`):
1. `@article{samvmnet2024, ...}` — SAM-VMNet
2. `@article{cmunet2025, ...}` — CM-UNet
3. `@article{messermmp2024, ...}` — MesserMMP SYNTAX
4. `@article{frangi1998multiscale, ...}` — Frangi filter
5. `@article{ronneberger2015unet, ...}` — U-Net
6. `@article{isensee2021nnunet, ...}` — nnU-Net
7. `@article{kirillov2023sam, ...}` — SAM
8. `@article{medsam2023, ...}` — MedSAM
9. `@article{transcc2023, ...}` — TransCC (Transformer baseline)
10. `@article{gnn_coronary_labeling2022, ...}` — GNN labeling

**Optional (if used in ablations/discussion)**:
- YOLO variants, G2ViT, LT-YOLO, DiGDA, etc.

---

## Figure Suggestions

### Figure 1: Method Evolution Timeline (Introduction/Related Work)
```
Classical CV     Early DL        Modern DL         Foundation Models    Agent Orchestration
(pre-2015)      (2015-2020)     (2020-2024)       (2024-2026)          (This Work)
───────────────────────────────────────────────────────────────────────────────────────
Frangi Filter → U-Net/AngioNet → SAM-VMNet        → DeepCORO-CLIP      → Cardiomni
                                  CM-UNet             (203K videos)        (zero-training,
                                  Transformer                              tool orchestration,
                                  GNN                                      reasoning trace)
                                  nnU-Net
```

### Figure 2: Task Coverage Matrix (Methods Section)
```
                    Seg   Sten   SYNTAX   Proj   Dom   Multi-view
SAM-VMNet            ✓     ✓      —       —      —      —
CM-UNet              ✓     —      —       —      —      —
CardioSYNTAX         —     —      ✓       —      —      ✓
DeepCORO-CLIP        —     ✓      ✓       —      —      ✓
CathAI               —     ✓      —       ✓      ✓      —
Cardiomni (ours)     ✓     ✓      ✓       ✓      ✓      ✓  (via tool orchestration)
```

---

## Discussion Points

### Specialist Model Limitations (to emphasize agent value)

```latex
While specialist models achieve impressive single-metric performance (DeepCORO-CLIP's 
13.6\% MAE approaches inter-reader variability~\citep{deepcoro2026}), they face three 
fundamental limitations: (1) \textbf{opacity}—DeepCORO-CLIP's authors explicitly list 
``interpretability'' and ``quality self-assessment'' as open problems; (2) \textbf{data 
hunger}—CathAI trained on 195K videos, DeepCORO-CLIP on 203K, yet both report performance 
degradation on out-of-distribution cases; and (3) \textbf{rigidity}—each new task 
(projection classification, dominance, lesion characterization) requires retraining. 
In contrast, \methodname's zero-training, tool-orchestrating approach adapts to novel 
tasks by composing existing tools under explicit clinical reasoning, offering a path 
toward transparent, data-efficient diagnostic systems.
```

### Why This Matters (Conclusion)

```latex
The abundance of specialist models (30+ methods addressing coronary angiography since 
2020) paradoxically highlights the need for agent orchestration: no single model covers 
the complete clinical workflow, and retraining end-to-end models for each new task is 
prohibitively expensive. By treating specialist models as composable tools and encoding 
the clinical SOP as an explicit reasoning pipeline, \methodname{} demonstrates that 
agent harnesses—not larger specialist models—may be the path toward generalizable, 
interpretable cardiovascular diagnosis.
```

---

## Checklist for Paper Integration

- [ ] Add 10 critical citations to `aaai2027.bib` from `methods_library.bib`
- [ ] Expand Related Work Section 2.1 with classical → DL → foundation trajectory (3 sentences)
- [ ] Add "Tool Library" paragraph in Section 3 (Experiments)
- [ ] Update Table 1 caption to emphasize specialist models as upper bounds, not competitors
- [ ] Add Figure 1 (evolution timeline) if space permits
- [ ] Revise Discussion to use specialist model limitations as motivation
- [ ] Update Conclusion to reference the "30+ methods" landscape

---

## Key Messaging

**For reviewers to understand**:
1. The field is **mature** (30+ methods, strong baselines)
2. These methods are **tools** in our benchmark, not competitors
3. Our **contribution** is orchestration (harness engineering), not end-to-end training
4. Our **positioning** follows MLE-bench: hold model + tools fixed, swap harness

**Numbers to emphasize**:
- 30+ methods published since 2020
- 195K--203K videos for state-of-art specialist models
- IoU 0.63, AUROC 0.888, MAE 13.6% (reference upper bounds)
- Zero-training agent vs. data-hungry specialists (the value proposition)
