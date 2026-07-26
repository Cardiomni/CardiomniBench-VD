# Method configuration convention

Every deep-learning method in this benchmark is described by a TOML file here,
named exactly after its registered method name (`methods/<name>.toml`). The
runner reads it through `benchmark/method_config.py`; nothing about a model's
preprocessing, decision rule, or inference geometry lives in Python.

The point is not tidiness. Two rules follow from it, and both exist because of
what this benchmark is measuring.

## Rule 1 — preprocessing is explicit, never implicit

A checkpoint is only half a method. The same weights produce very different
numbers depending on spacing, intensity windowing, normalisation, input
resolution, and how a probability map becomes a mask. If those choices sit
hardcoded in a runner, the reported score is unattributable: nobody can tell
whether a model underperformed or was simply fed the wrong input.

So every value that affects the input tensor or the output mask is declared in
TOML with its provenance. Where a value comes from an upstream release, the
comment says which file and which line it mirrors. Where a value is our own
choice, the comment says so and gives the reasoning. `coronary_cm_unet.toml`
cites `dataset.py`'s `PadIfNeeded(1536)` and `resize((256,256), BICUBIC)`;
`coronary_nnunet.toml` marks its preprocessing block as *references* because the
authoritative values are read from `plans.json` at runtime.

## Rule 2 — variants are all reported, never selected on results

When a preprocessing decision is genuinely uncertain, it becomes **two
registered methods**, and both are evaluated and reported. Picking the better
number after the fact would make the benchmark a story about our tuning rather
than about the methods.

Current variant pairs:

| Faithful-to-upstream | Adapted | What differs |
|---|---|---|
| `coronary_unet` | `coronary_unet_argmax` | `threshold` 0.5 vs `argmax` decision rule |
| `coronary_cm_unet` | `coronary_cm_unet_native` | pad to 1536 before the 256 resize, vs resize 512→256 directly |

Naming: the faithful reproduction keeps the plain method name. The variant gets
a suffix describing *what it changes* (`_argmax`, `_native`), not a judgement
(`_better`, `_v2`).

One empirical note worth keeping, because it stops a pointless experiment from
being repeated: for a 2-class head, `argmax` and `threshold` at 0.5 are the same
rule. `coronary_unet` and `coronary_unet_argmax` produced bit-identical masks on
`case_cca_0001_0` — every metric equal to four decimals, `foreground_fraction`
0.000541 for both — with the TOML confirmed to have taken effect (`decision_rule`
recorded as `threshold` and `argmax` respectively in each prediction's
diagnostics). The pair is kept because the *equality* is the finding; a threshold
other than 0.5 would be a real variant.

## Rule 3 — the variant gap is itself a measurement

These pairs are not only bookkeeping. The gap between a faithful and an adapted
configuration quantifies how sensitive a model is to preprocessing mismatch, and
that is a question the Cardiomni agent has to answer when it calls a tool.

An agent handed CM-UNet and a 512×512 frame has to decide whether to follow the
model card literally (pad to 1536, losing 3× resolution on vessels a few pixels
wide) or to recognise that the card assumes ~1536px native angiograms. Knowing
the real cost of each choice — because we measured both — turns this into a
scoreable agent capability rather than a matter of opinion.

## Rule 4 — transcribe from the layer that built the weights, not the layer nearest the model

CM-UNet's `Finetuning/dataset.py` applies no intensity normalisation: it loads a
`.npy`, resizes to 256, and hands it to the network. Transcribing that gave
`normalize = false`, which is a true statement about that file and still the wrong
configuration, because the `.npy` files were written by
`data_processing/data_processing.ipynb` after `Unsharper(radius=60, amount=3)` and
`Intensity_normalizer` (per-image z-score).

Measured on the same weights and the same 5 ARCADE cases:

| `[preprocess]` | Dice | foreground |
|---|---|---|
| raw uint8 (`normalize = "none"`) | 0.000 | 0.0000 |
| `"divide255"` | 0.321 | 0.0066 |
| `"zscore"` | 0.709 | 0.0234 |
| `"zscore"` + unsharp 60/3 | **0.726** | 0.0229 |

Gold vessel coverage is 0.0254, so the working configurations recover a plausible
amount of vessel and the raw-input one predicts essentially nothing.

The failure mode to watch for: raw input did not error, warn, or produce obvious
garbage. It produced an empty mask and a clean 0.000, which reads exactly like a
model that cannot transfer to this dataset. Any "this checkpoint does not
generalise" conclusion is worth re-checking against the offline preprocessing that
produced its training arrays.

## Required sections

Not every section applies to every method; declare the ones that do.

- `[preprocess]` — 3D volumetric input: `pixdim`, `orientation`, intensity
  window, normalisation mode, any enhancement (denoise / CLAHE / vesselness).
- `[video_preprocess]` — 2D+t cine input: frame count and sampling, resize mode,
  rescaling, channel replication, normalisation statistics.
- `[preprocess]` for 2D image input: padding, model input resolution,
  normalisation.
- `[inference]` — sliding-window geometry, AMP, architecture parameters,
  ensembling, output transforms.
- `[decision]` — how logits or probabilities become a discrete answer.
- `[instances]` — for instance-list tasks: label assignment and component
  filtering.

## Provenance lives with the method, not the config

Dataset, training domain, `DomainRelation`, reported upstream metrics, and known
limitations are declared on the method object in `benchmark/specialists.py`, so
the results table can label a cross-domain transfer as such. The TOML covers
*how to run it*; the `Provenance` covers *what it is and what it may fairly be
compared against*.
