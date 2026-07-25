# Task: ARCADE — Coronary Stenosis Detection

> **One line**: Given one coronary angiography frame, localize every stenosis
> (narrowed) region.

## Task type
`arcade_stenosis` — instance segmentation, single class `"stenosis"` (location only).

## Source
ARCADE (MICCAI 2023), `stenosis` task, official **test** split. Our locked subset
= **69 images** — the multi-lesion cases only (**≥2 stenoses/image**), deliberately
excluding the 77% single-lesion majority to keep the set hard. Distribution: 54
images with 2 lesions, 13 with 3, 2 with 4. Manifest:
`Datasets/ARCADE_FO/_subset_stenosis_multilesion.json`.

## Input (what the agent reads)
One 512×512 grayscale XCA frame.

- `input.modality`: `"XCA"`
- `input.image.file_path`: `"image.png"`
- `input.image.width` / `height`: 512 / 512

## Output (what the agent writes → `prediction.json`)
A list of stenosis instances:

```json
{
  "instances": [
    {"label": "stenosis", "bbox_xywh_norm": [x, y, w, h], "mask": "<...>"},
    {"label": "stenosis", "bbox_xywh_norm": [x, y, w, h], "mask": "<...>"}
  ]
}
```

- `label` — always `"stenosis"`.
- `bbox_xywh_norm` — `[x, y, w, h]` normalized to [0,1].
- `mask` — per-lesion binary mask.

**Not in this task**: stenosis percent, clinical tier, or which segment the lesion
sits on. ARCADE stenosis labels are location-only. (Percent/segment grading is the
gap Cardiomni's full report task must fill from other sources.)

## Gold standard (stripped from `task_spec.json`)
`task.yaml.gold_standard.instances` (label + normalized bbox) + `masks_file`
→ `../../gold/<case>/masks.npz`.

## Metric (deferred)
**Mean F1 per image, then averaged** (ARCADE official).

## Cardiomni relevance
This is the *detection* half of Cardiomni **Stage 4 (lesion assessment)**: find
where the disease is. Grading the severity (%/tier) is a separate step this public
label cannot supervise — flagged as the anti-hallucination boundary (don't invent
a percent the data can't confirm).
