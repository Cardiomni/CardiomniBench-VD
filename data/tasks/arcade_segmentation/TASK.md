# Task: ARCADE — Coronary Artery Segment Segmentation

> **One line**: Given one coronary angiography frame, identify and delineate every
> coronary artery segment and label it with its SYNTAX segment id.

## Task type
`arcade_segmentation` — instance segmentation over 25 SYNTAX segment classes.

## Source
ARCADE (MICCAI 2023 grand challenge; Nature Sci Data s41597-023-02871-z),
`syntax` task, official **test** split. Our locked subset = **42 images** chosen
for coverage (24/25 segments, rare segments 12/10/15, fine vessels, RCA+LCA,
easy→9-segment cases). Selection manifest: `Datasets/ARCADE_FO/_subset_segmentation.json`.

## Input (what the agent reads)
One 512×512 grayscale X-ray coronary angiography (XCA) frame.

- `input.modality`: `"XCA"`
- `input.image.file_path`: `"image.png"` (relative to the case folder)
- `input.image.width` / `height`: 512 / 512

No clinical context, no video, no percent labels. Single static frame only.

## Output (what the agent writes → `prediction.json`)
A list of segment instances. Each instance:

```json
{
  "instances": [
    {"label": "6", "bbox_xywh_norm": [x, y, w, h], "mask": "<...>"},
    {"label": "11", "bbox_xywh_norm": [x, y, w, h], "mask": "<...>"}
  ]
}
```

- `label` — SYNTAX segment id (string). Label space (25 classes):
  `1 2 3 4 5 6 7 8 9 9a 10 10a 11 12 12a 12b 13 14 14a 14b 15 16 16a 16b 16c`.
- `bbox_xywh_norm` — `[x, y, w, h]` normalized to [0,1] (multiply by 512 for px).
- `mask` — per-instance binary mask (pixel-level). Encoding TBD when scoring is
  wired; the gold stores bbox-local `uint8` arrays.

## Gold standard (stripped from `task_spec.json`)
In `task.yaml.gold_standard`: `instances` (label + normalized bbox) +
`masks_file` → `../../gold/<case>/masks.npz` (one array per `instance_id`,
bbox-local binary). Kept out of the agent-visible case folder.

## Metric (deferred — scoring not wired yet)
**Mean F1 per image, then averaged** (ARCADE official ranking metric; NOT Dice,
NOT mAP). Dice in the ARCADE paper is only for annotator agreement.

## SYNTAX segment id reference (abbreviated)
RCA: 1 prox, 2 mid, 3 dist, 4 PDA, 16/16a/16b/16c posterolateral.
LM: 5. LAD: 6 prox, 7 mid, 8 dist, 9/9a diagonal, 10/10a.
LCX: 11 prox, 12/12a/12b intermediate/OM, 13 dist, 14/14a/14b posterolateral, 15 PDA(left-dom).

## Cardiomni relevance
This is Cardiomni **Stage 2 (systematic segment scan)** as a standalone,
gold-checkable subtask: can the agent name and localize every segment without
omission? Maps to the "segment coverage" and "naming accuracy" evaluation axes.
