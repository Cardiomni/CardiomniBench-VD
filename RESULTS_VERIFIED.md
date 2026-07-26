# Verified baseline results

Every number here came from a completed run in `runs/`. Nothing is estimated,
extrapolated, or copied from an upstream paper. Where a run is partial the case
count says so, and where a metric is zero the reason is stated, because several of
these zeros are artefacts of the measurement rather than of the method.

Regenerate everything with:

```bash
cd /mnt/aliyunsb/Cardiomni/CardiomniBench-VD
source env.sh
./scripts/run_all_baselines.sh --device cuda:5
```

## At a glance

| task | best method | headline | n |
|---|---|---|---|
| cardiosyntax_scoring | `cardiosyntax_r3d_fold1` | MAE 6.157 ± 7.661 | 420/420 (7 variants) |
| arcade_segmentation | `coronary_cm_unet_native` | pixel_dice 0.5924 ± 0.1531 | 42/42 |
| arcade_stenosis | `coronary_cm_unet_native` | pixel_dice 0.4054 ± 0.1885 | 69/69 |
| cca_segmentation | `coronary_unet` | Dice 0.3726 ± 0.1018 | 20/20 |

684 case-level predictions across the four completed runs above (420 SYNTAX,
222 ARCADE, 20 sam_med3d, 20 CCA `coronary_unet`), plus a 2-case VLM smoke test
and a 9-case topology subset. Three things a reader should carry away before
looking at any single number:

1. **Every label-aware F1 in this table is 0.0000, for two unrelated reasons.**
   CM-UNet has no segment vocabulary at all; a VLM emitting perfect gold bounding
   boxes also scores 0 because vessels fill only 22.9% of their own box. The
   `label_set_precision` / `label_set_recall` columns are what separate these.
2. **Single-checkpoint numbers are not stable.** The seven CardioSYNTAX folds span
   MAE 6.16 to 8.14, so quoting one fold invites a 30% swing.
3. **`sam_med3d` is not a zero-shot row.** It is prompted from the gold mask and
   still scores Dice 0.0003. Report it as a gold-guided reference or not at all.

## cardiosyntax_scoring (60 cases)

Run: `runs/syntax_all_folds/`, 420/420 predictions, all seven checkpoint variants.

| variant | n | MAE | pearson r | tier acc |
|---|---|---|---|---|
| `cardiosyntax_r3d_fold1` | 60/60 | **6.157 ± 7.661** | 0.805 | 0.767 |
| `cardiosyntax_r3d_fold3` | 60/60 | 6.850 ± 7.786 | 0.804 | 0.717 |
| `cardiosyntax_r3d` (default) | 60/60 | 6.900 ± 7.806 | 0.788 | 0.750 |
| `cardiosyntax_r3d_fold4` | 60/60 | 6.999 ± 7.877 | 0.769 | 0.767 |
| `cardiosyntax_r3d_fold0` | 60/60 | 7.109 ± 8.549 | 0.769 | 0.733 |
| `cardiosyntax_r3d_calibrated` | 60/60 | 7.514 ± 7.971 | 0.787 | 0.767 |
| `cardiosyntax_r3d_fold2` | 60/60 | 8.138 ± 8.197 | 0.715 | 0.683 |

Two things to carry into the paper. The spread across folds is 6.16 to 8.14, about
2 MAE points, so a single fold is not a stable estimate of this model's ability and
quoting one number invites a 30% swing depending on which checkpoint was picked.
The default `cardiosyntax_r3d` at 6.900 happens to sit near the fold median, which
is why it is a defensible representative, but the fold range belongs alongside it.

The `calibrated` variant is worse than the uncalibrated default (7.514 vs 6.900)
while its correlation is essentially unchanged (0.787 vs 0.788). Whatever the
calibration was fitted on, it does not transfer to these 60 cases: it shifts
predictions without improving their ordering.

Gold spread over the 60 studies is mean 13.9, range 0–58, so an MAE near 7 is
roughly half the mean score: a weak-but-real signal, not a solved task.
Three-expert disagreement on the same studies spans about 8.6 points, so the model
sits inside the human consistency band on most cases.

An earlier run of the default checkpoint scored MAE 7.97. The difference was
normalisation (Kinetics-400 statistics instead of ImageNet) plus a missing
`antialias=True` on the resize. Paired t-test over the 60 cases: p = 0.0014,
Cohen dz = 0.43, tier accuracy 0.667 → 0.750. Preprocessing, not the weights.

## arcade_segmentation (42 cases) and arcade_stenosis (69 cases)

Run: `runs/arcade_cmunet_v2/`, 222/222 cases, both variants, both tasks. Same
checkpoint in every row; the only change is `pad_to`.

| task | variant | pixel_dice | f1 | f1_label_agnostic | IoU (label-agn.) | label_set_prec / rec |
|---|---|---|---|---|---|---|
| segmentation | `pad_to=1536` (upstream) | 0.0002 ± 0.0011 | 0.0000 | 0.0000 | 0.0004 | 0.0000 / 0.0000 |
| segmentation | `pad_to=0` (adapted) | **0.5924 ± 0.1531** | 0.0000 | 0.0273 ± 0.0686 | 0.3083 | 0.0000 / 0.0000 |
| stenosis | `pad_to=1536` (upstream) | 0.0000 | 0.0000 | — | 0.0000 | — |
| stenosis | `pad_to=0` (adapted) | **0.4054 ± 0.1885** | 0.0078 ± 0.0454 | — | f1@0.25: 0.1786 | — |

Intensity normalisation ablation, 5 cases, same checkpoint (gold vessel fraction
0.0254): raw uint8 0.000, ÷255 0.321, z-score 0.709, unsharp + z-score **0.726**.
The last row is the upstream pipeline: the training `.npy` files had already been
through `Unsharper(radius=60, amount=3)` and a per-image z-score in an offline
notebook, so the model's own `dataset.py` legitimately contains no normalisation.
Transcribing preprocessing from the layer nearest the model produced the 0.000.

### Why label-aware F1 is 0.0000 for CM-UNet

Structural, not a bug, and the `label_set_*` columns make the reason explicit:
both precision and recall over the predicted *set of segment ids* are also
0.0000. CM-UNet does not merely mislabel segments, it emits no segment vocabulary
at all. Three limits stack:

1. One binary vessel class, so every label-aware metric is 0 by construction.
2. 2.93 predicted connected components against 6.52 gold anatomical branches.
3. Label-agnostic IoU 0.31, below the 0.50 matching threshold.

`f1_label_agnostic` is 0.0273 rather than exactly 0 for the adapted variant: it
occasionally matches one component well enough to clear the threshold, which is
what separates "wrong granularity" from "no overlap at all".

`pixel_dice` 0.5924 is its real capability. On stenosis, F1 is 0.0078: using a
vessel segmenter as a stenosis detector does not work, since it outlines the whole
vessel while a stenosis is a local narrowing. Both are legitimate negative results
about tool scope, and both are why the segment-naming slot is still empty.

### Why bbox-only VLM predictions also score F1 = 0.0000

Independent finding, and it matters for reading the table. Feeding gold boxes back
in as a perfect prediction still scores 0:

```
case_arcade_seg_0001_101-5, prediction = gold bbox reproduced exactly
  f1 (IoU>=0.5, headline)  0.0000
  f1_at_iou025             0.2500
  pixel_dice               0.3353
  label_set_precision      1.0000
  label_set_recall         1.0000
```

Gold stores a vessel mask inside each bbox; a box-only prediction is scored as a
filled rectangle. Vessels are thin and diagonal, so they fill only part of their
own box: mean fill 0.229 (median 0.189, p90 0.468) across 274 instances. A perfect
box clears IoU ≥ 0.5 on 24/274 instances, 8.8%. On stenosis, fill is 0.371 and
17.4% clear the threshold.

So an F1 of 0 on this task means two entirely different things depending on the
method, and `label_set_precision` / `label_set_recall` are the columns that
separate them. Both are now in `TASK_COLUMNS`.

## cca_segmentation (20 cases)

Two runs, because the topology metrics are expensive. `runs/cca_unet_full/` has
the full metric set on the first 9 cases; `runs/cca_unet_fast20/` covers all 20
with Dice/precision/recall only, and `runs/cca_sam_med3d/` adds the second method.

| method | n | Dice | clDice | HD95 (mm) | precision | recall |
|---|---|---|---|---|---|---|
| `coronary_unet` (full, fast metrics) | 20/20 | 0.3726 ± 0.1018 | — | — | 0.6334 | 0.2712 |
| `coronary_unet` (topology subset) | 9/20 | 0.3547 ± 0.1177 | 0.3650 ± 0.0959 | 73.54 ± 23.15 | 0.6177 | 0.2576 |
| `sam_med3d` (gold-prompted) | 20/20 | 0.0003 ± 0.0005 | — | — | 0.0009 | 0.0002 |

The two rows agree on Dice (0.3726 over 20 vs 0.3547 over 9), which is the check
that `CARDIOMNI_FAST_METRICS=1` only skips clDice and HD95 rather than perturbing
the overlap metrics.

`coronary_unet` precision 0.6334 against recall 0.2712 is under-segmentation: about
two thirds of what it labels vessel really is vessel, but it finds only a quarter of
the tree. That is a different failure from `sam_med3d`, whose predicted volume is
plausible in size yet lands in the wrong place. Both show up as low Dice, so the
precision/recall split is what separates "missed most of it" from "found the wrong
thing".

### sam_med3d: an oracle-prompted upper bound that still fails

Run: `runs/cca_sam_med3d/`, 20/20. This row is **not zero-shot** and must not be
read as one. SAM-Med3D is promptable and its own README states that ground-truth
labels are required to generate prompt points, so `sam_med3d_runner.py` samples 5
positive point prompts *from the gold mask* inside each 128³ patch. It is given an
oracle advantage no other method here receives.

It still scores Dice 0.0003, and 9 of the 20 cases are exactly 0.0 — not "nearly
zero" but not one overlapping voxel. The runner's docstring explains the mechanism:
SAM-Med3D operates on 128³ crops at 1.5 mm spacing, a 192 mm cube, while a coronary
tree spans roughly 416x416x288 mm, so a single crop covers under a quarter of the
vessel tree.

The predicted volume is not absurd: 122k voxels against 442k gold, a volume ratio
of 0.30. So it outputs a plausible *amount* of foreground in the wrong *places*.
That is the same signature as the CCA preprocessing bug (right volume, wrong
location), which is why the volume ratio alone would have hidden it and why
precision 0.0009 is the number that settles it.

The honest reading: an organ-scale 3D foundation model does not transfer to
coronary-scale structures even with gold-guided prompts. Report it as a
gold-guided foundation-model reference, never as a zero-shot baseline.

Two provenance errors were corrected in `methods/sam_med3d.toml` while recording
this. It declared `domain_relation = "zero_shot"`, which is not a member of
`DomainRelation` (the enum has `not_trained`) and asserted exactly the property
this row does not have. Its limitations text also described the prompt as a
bounding box, while the runner uses point prompts and its docstring notes that box
prompts raise `IndexError` because `_embed_boxes()` indexes point embeddings 2 and
3 that are never created. Both now match the implementation.

Recall 0.258 against precision 0.618 is consistent systematic under-segmentation,
stable across all 9 cases (Dice sd 0.118). Cross-dataset: the checkpoint was
trained on ImageCAS, which labels the full lumen, while CCA labels the inner
diameter (`domain_relation="cross_dataset"`, upstream reported Dice 0.788).

clDice 0.365 sits slightly above Dice 0.355, which says the predicted centreline
overlaps gold about as well as the volume does: the model is not breaking the
vessel tree into disconnected fragments, it is drawing it too thin. That is the
same conclusion recall 0.258 gives, from an independent measurement.

HD95 of 73.5 mm is large for a coronary tree and varies widely (sd 23.2), meaning
some predicted surfaces land far from any gold vessel. Combined with high
precision this points to isolated false-positive components rather than a
uniformly displaced mask.

clDice and HD95 each cost minutes per case on these 832x832x576 volumes, so
`CARDIOMNI_FAST_METRICS=1` drops them for full sweeps and `scripts/run_all_baselines.sh`
sets it automatically for this task. The command to add them back is printed at
the end of that script.

`algorithms/baselines/cca_unet_agent.py` used to score 0.0482 on the same
checkpoint and the same case, an 11x gap, because it reimplemented preprocessing
and skipped resampling, orientation, body-crop and the correct HU window. Its
`foreground_ratio` of 0.00097 against a gold 0.00102 is why that failure was
silent: the predicted volume was right and only its location was wrong.

It is now an 88-line shell (from 259) that calls `monai_unet_runner.predict()`
through `load_case_input()` + `load_method_config()`. Re-verified on the same
case: Dice **0.5365**, `pred_fg` 215700 — bit-identical to the runner, not merely
close, because it is now literally the same code path. Its `diagnostics` show the
preprocessing that the old version skipped:

```
pixdim [1.0, 1.0, 1.0]        hu_window [-260.0, 760.0]     normalize minmax
prepared_shape [432, 432, 304]   reference_shape [832, 832, 576]   threshold 0.5
```

`prepared_shape != reference_shape` is the direct evidence that resampling from
0.5 mm to 1.0 mm and the body crop actually ran; in the old version those two
shapes were identical.

## Not yet measured

- 6 VLMs on all three text-output tasks. Blocked on the unified `.venv`; the run
  attempted with a conda interpreter failed 180/180 cases with
  `ModuleNotFoundError: No module named 'transformers'`.
- `lingshu_7b`, `qwen25_vl_7b`, `qwen3_vl_8b`: weights are now complete on disk
  (14.8 / 13.9 / 17.1 GB of shards), but `check_available()` reported them
  unavailable because 9.4 GB of `.incomplete` blobs were left behind by an
  interrupted download. The check is right to refuse a half-downloaded model, but
  it cannot tell a stale artefact from an in-flight one, so leftovers produce a
  false negative. Cleared; re-verify with `check_available()` before running.
  The first download attempt also failed on `lingshu_7b` with a 401 because the
  script hardcoded `FlagOpen/Lingshu-7B`; `benchmark/vlms.py` had the correct
  `lingshu-medical-mllm/Lingshu-7B` all along, which is the argument for reading
  repo ids from the registry rather than retyping them.
- `sam_med3d` on CCA: done, see above.
- SAM-VMNet and DeepCORO-CLIP: no usable weights on disk. SAM-VMNet's `.pth`
  files are 132-byte git-lfs pointers; DeepCORO's weight directories are empty and
  the upstream repos return 401/403 pending author approval.

## Difficulty stratification is not currently reportable

The paper plans a harness x difficulty table. All 191 cases do carry a
`difficulty_level`, but the labels come from two incompatible scales and several
cells are too small to support a mean:

| task | labels present |
|---|---|
| arcade_segmentation | easy 9, medium 8, hard 25 |
| arcade_stenosis | medium 67, hard 2 |
| cardiosyntax_scoring | low 44, intermediate 4, high 12 |
| cca_segmentation | medium 20 |

The ARCADE tasks use easy/medium/hard while CardioSYNTAX uses low/intermediate/
high, so there is no single tier axis to stratify on. Even within one scale the
counts do not support the intended three-tier reading: `arcade_stenosis` has 2
hard cases and `cardiosyntax_scoring` has 4 intermediate ones, and a mean over 2
cases is not a tier result.

Two things are needed before that table can be filled: a mapping between the two
vocabularies (or a decision to report them separately), and either accepting the
skew explicitly with n shown per cell, or re-deriving tiers from case properties.
This is a data-definition question, not something to resolve by relabelling to fit
the table.





## 修正 prompt 后的 VLM 基线（2026-07-26 19:52，进行中 28/513）

**修正**：移除字面示例坐标后重跑 3 个 llava 系列（`runs/vlm_fixed_prompt`）。

**ARCADE segmentation（28 case 初步数据）**：
- **单框重复现象占主导**：24/28 (86%) 的 case 里，所有节段拿到同一个框。
- **少数 case 会分出多个框**：4/28 给出 2–5 个不同的框（case_0008: 2 框, 0014: 3 框, 0015: 5 框, 0019: 2 框）。
- **但 IoU 仍为 0**：即便分出多个框，f1@0.5 均值 0.0000，max 0.0000 —— 这些框不与金标准的节段边界对齐。
- **过度预测**：均值 10.8 个实例 vs 金标准 6.2，说明模型把每个可能的节段 ID 都列出来了，而不是只报告可见的。

**结论（可写入论文）**：当前开放 VLM (llava 系列) 不具备冠脉节段级定位能力。它们通常输出一个覆盖整棵血管树的粗 ROI 并贴上所有可能的节段标签（86% case），偶尔产生 2–5 个空间不同的框（14%），但在所有测试 case 中均未达到 IoU>0.5，即从未正确定位任何单个节段。这一行为在明确要求"不同节段必须用不同坐标"的 prompt 下仍然出现，说明是模型能力的真实边界，而非 prompt 设计问题。

相比之前被 prompt 污染的结果（逐图输出完全相同），这一数据反映的是模型在干净 prompt 下的真实表现。
## Prompt 泄漏消融：VLM 的 f1=0.0 之前是被 prompt 污染的（2026-07-26 19:35）

**结论：之前两次全量 VLM run 的数字作废，prompt 已修，需重跑。**

起因：`llava_16_mistral_7b` 和 `llama3_llava_next_8b`（两种不同架构）对**不同的**造影图
输出**逐字节相同**的结果，坐标恰好是 prompt 里格式示例的那两组
`[0.21, 0.17, 0.40, 0.34]` / `[0.24, 0.42, 0.47, 0.63]`。

消融实验（`/tmp/probe_prompt.py`，同模型同解码同 3 个 case，只改示例块）：

| 变体 | case_0001 | case_0002 | case_0003 |
|---|---|---|---|
| ORIGINAL（带字面示例） | `[0.24,0.42,0.47,0.63]` | `[0.24,0.42,0.47,0.63]` | 两组示例坐标交替 |
| NO_EXAMPLE（只留 schema） | `[0.156,0.152,0.456,0.619]` | `[0.186,0.152,0.550,0.841]` | `[0.083,0.042,0.456,0.986]` |

**两个独立结论，必须分开写**：

1. **示例坐标会被照抄**。去掉字面示例后，框**随图片变化**了 —— 说明模型确实在看图,
   之前的"逐字节相同"是 prompt 自己泄漏答案，不是模型完全无视图像。
   → 之前 f1=0.0 是"prompt 缺陷 + 能力不足"的混合物，**不能当作纯能力测量写进论文**。

2. **但模型仍然不做 per-segment 定位**。NO_EXAMPLE 下,同一张图里 6 个 segment 拿到的是
   **同一个框**（整棵冠脉树的粗略范围），只是逐图不同。这是"给出一个血管树 ROI"而非
   "定位每个节段"。这一条在修 prompt 前后都成立，是可以写的真实结论。

**已修**（`benchmark/runners/arcade_vlm_runner.py`，SEGMENTATION_PROMPT 与 STENOSIS_PROMPT）：
删掉字面示例坐标，改为 schema 占位符 `[x_min, y_min, x_max, y_max]` + 明确要求
"不同节段不得共用同一组坐标"。

**代价**：已跑的 `runs/vlm_llava16_all3`（40/171）和 `runs/vlm_llavanext_all3`（12/171）
在修复前的 prompt 下产生，**已中止，不可用**。需在新 prompt 下重跑。
教训：baseline prompt 里任何字面示例值都可能被当成答案抄走，格式示例必须用占位符。

## 6 个 VLM 里只有 3 个权重完整（2026-07-26 18:55）

lingshu_7b 全量 171 例**全部失败**：`SafetensorError: Error while deserializing header: incomplete metadata`。
不是环境问题，是**权重下载被截断**——4 个 shard 文件都在，但字节数比 index 声明的少 1.66GB。

按 `model.safetensors.index.json` 的 `metadata.total_size` 逐个核对磁盘实际字节：

| method | 声明 | 磁盘 | 差 | 可用 |
|---|---|---|---|---|
| lingshu_7b | 15816M | 14156M | **-1660M** | ✗ 截断 |
| qwen25_vl_7b | 15816M | 13217M | **-2599M** | ✗ 截断 |
| qwen3_vl_8b | 16722M | 16297M | **-425M** | ✗ 截断 |
| llava_16_mistral_7b | 14432M | 14433M | +0M | ✓ |
| llama3_llava_next_8b | 15936M | 15937M | +0M | ✓ |
| llava_onevision_7b | 15318M | 15318M | +0M | ✓ |

**代价**：`check_available()` 原来只检查 shard 文件**是否存在**，不比对字节数，所以三个截断模型
一路通过校验、加载失败、171 例全废。已修（`benchmark/vlms.py`）：现在用
`os.path.getsize(os.path.realpath(...))`（HF cache 是 symlink，必须 realpath）求和后与
`metadata.total_size` 比对，短缺 >50MB 即判不可用并给出差额。

**影响 VLM baseline 的覆盖面**：医学微调的 lingshu_7b 和两个较新的 Qwen-VL 都跑不了，
当前只能报 3 个 llava 系模型。这意味着**"医学微调是否改变抄 prompt 行为"这个问题暂时无法回答**——
需要补下载才能验证。论文里 VLM 行只能先写 3 个，或说明缓存受限。

补下载命令（未执行，需确认，约 4.7GB）：
```
HF_HOME=/mnt/aliyunsb/Cardiomni/hf_cache huggingface-cli download <repo_id> --resume
```

## VLM baseline 首次真实推理成功（2026-07-26 18:45，`runs/vlm_direct_smoke/`）

环境修好后 llava_16_mistral_7b 在 arcade_segmentation 上跑通 2 例。**关键发现不是分数，是模型在抄 prompt。**

两例的 `raw_output` **逐字节相同**：
```
SEGMENT: 6 [0.21, 0.17,0.40,0.34]
SEGMENT: 7 [0.24,0.42,0.47,0.63]
SEGMENT: 8 [0.24,0.42,0.47,0.63]
... （9..15 全部重复同一个框）
```
prompt 里的格式示例正是 `SEGMENT: 1 [0.21, 0.17, 0.40, 0.34]` 和 `SEGMENT: 2 [0.24, 0.42, 0.47, 0.63]`。
模型把这两个坐标搬过来，套在 6-15 号段上，两张不同的造影图给出完全一样的答案。

指标后果：
| 指标 | case_0001 | case_0002 |
|---|---|---|
| f1 (IoU-matched) | 0.0000 | 0.0000 |
| pixel_dice | 0.2774 | 0.1574 |
| n_pred / n_gold | 10 / 4 | 10 / 7 |

`f1=0` 是真实的；`pixel_dice` 非零只是因为抄来的框碰巧压在血管territory上——**这个 dice 不代表任何视觉能力**，
论文里若报 VLM 的 pixel_dice 必须同时说明它可由固定框获得。`rejected_unknown_label=0`、
`rejected_degenerate_box=0`，说明格式解析完全正常，问题不在 parser。

**这本身是可写进论文的结果**：通用多模态模型零样本面对侵入性冠脉造影时，倾向于复述提示里的
格式示例而非读图。它给 Cardiomni 的 gap 提供了干净的下界，也说明"给 VLM 更好的 prompt"不是
提升路径——模型缺的是把 SYNTAX 编号对应到血管影像的能力，不是输出格式的知识。

**待验证**：这是 llava 特有还是普遍现象。已在 GPU4/GPU5 并行启动 lingshu_7b（医学微调）和
llava_16_mistral_7b 的全量三任务（各 171 例），结果出来后即可判断医学微调是否改变这个行为。
单例推理约 13s（首例 254s 含加载），全量单模型约 37 分钟。

## 难度分层结果（2026-07-26 18:35 新增）

难度**不进 prompt**。每个 case 一视同刻地跑，出结果后按 `task.yaml` 里 `case_metadata.difficulty_level`
分桶。实现见 `benchmark/results.py::stratify_by_difficulty` + `case_difficulty`。

ARCADE 用自己的 easy/medium/hard，CardioSYNTAX 用自己的 low/intermediate/high，**不统一量表**——
两套词汇表是否可比未经证实，强行合并会暗示它们等价。每格 n 照实报，n=2 就写 n=2。

### arcade_segmentation / coronary_cm_unet_native  [pixel_dice]
| difficulty | n | pixel_dice |
|---|---|---|
| easy | 9 | 0.7304 ± 0.1132 |
| medium | 8 | 0.5155 ± 0.1842 |
| hard | 25 | 0.5673 ± 0.1268 |

**反直觉**：hard(0.5673) 高于 medium(0.5155)。难度标签与该模型的失败模式不单调相关。
ARCADE 的 difficulty 来自标注者对**病变判读**难度的判断，而 pixel_dice 量的是**血管树覆盖**——
两者不是同一件事。写进论文时不能暗示"难度↑ 性能↓"。

### arcade_stenosis / coronary_cm_unet_native  [pixel_dice]
| difficulty | n | pixel_dice |
|---|---|---|
| medium | 67 | 0.4038 ± 0.1909 |
| hard | 2 | 0.4590 ± 0.0745 |

hard 仅 2 例，均值不可解释。

### cardiosyntax_scoring / cardiosyntax_r3d  [MAE，越低越好]
| difficulty | n | MAE |
|---|---|---|
| low | 44 | 4.1930 ± 4.9024 |
| intermediate | 4 | 5.4435 ± 2.7312 |
| high | 12 | 17.3115 ± 9.0855 |

**这条是单调的且幅度很大**：low→high 的 MAE 从 4.19 涨到 17.31（4.1 倍）。
模型在高复杂度病例上系统性低估——`signed_error` 可进一步验证方向。这是有意义的分层结果。
intermediate 仅 4 例。

### cca_segmentation  [Dice]
| method | difficulty | n | Dice |
|---|---|---|---|
| coronary_unet | medium | 20 | 0.3726 ± 0.1018 |
| sam_med3d | medium | 20 | 0.0003 ± 0.0005 |

CCA 全部 20 例都是 medium，无分层信息。

**注意**：`runs/arcade_cmunet_full/` 同时含 `coronary_cm_unet`（修复前，dice≈0.0002）和
`coronary_cm_unet_native`（修复后，0.5924）两个变体。取数必须指定 `_native`，否则会拿到废弃变体。

