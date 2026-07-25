# Task: CardioSYNTAX — Study-level SYNTAX Score + Dominance

> **One line**: Given all multi-view angiography cine videos of one patient,
> predict the coronary dominance and the anatomical SYNTAX score.

## Task type
`cardiosyntax_scoring` — study-level regression (score) + classification (dominance).

## Source
CardioSYNTAX (Ponomarchuk et al., WACV 2025, arXiv:2407.19894). The on-disk **Part 9**
subset = **44 studies** with real cine videos. Source: `Datasets/CardioSYNTAX/9/`,
metadata `part9.json`.

## Input (what the agent reads)
All projection cine videos for one patient study.

- `input.modality`: `"XCA_cine"`
- `input.views[]`: one entry per video —
  - `file_path`: `"videos/<name>.npy"` (relative to case folder)
  - `artery`: `"LCA"` or `"RCA"` (which side this run injects)
  - `positioner_primary_angle` / `positioner_secondary_angle`: C-arm projection (RAO/LAO, CRA/CAU)
  - `shape`: `[frames, 512, 512]` (uint8)
- Typically **6–7 views** per study (LCA + RCA runs at different angles).

This is the **only** dataset whose input matches Cardiomni's real input format:
multiple projections + angle metadata, to be integrated across views.

## Output (what the agent writes → `prediction.json`)
```json
{
  "syntax_score": 12.0,
  "syntax_left": 7.0,
  "syntax_right": 5.0,
  "dominance": "right"
}
```

- `syntax_score` — total anatomical SYNTAX score (float, ~0–67).
- `syntax_left` / `syntax_right` — per-side subtotals.
- `dominance` — `"right"` or `"left"` (SYNTAX defines **no** co-dominant option).

## Gold standard (stripped from `task_spec.json`)
`task.yaml.gold_standard`: `syntax_score`, `syntax_left`, `syntax_right`,
`dominance`, `bypass`, `risk_band` (low ≤22 / intermediate 23–32 / high ≥33).

Data notes: 66% of these 44 studies are normal (score 0); dominance is labeled on
only some studies; 60-study 3-expert annotations exist upstream (mean inter-expert
spread 8.6 pts, 17% full agreement) — useful later as a gold-reliability ceiling.

## Metric (deferred)
Score: **MAE / RMSE / R²** vs gold. Dominance: **accuracy**. Optional:
fraction of predictions within the 3-expert range.

## Status note
SYNTAX score is **future-work** per the pivot (not a current paper claim). But the
**raw multi-view videos are the highest-value asset here** — they are the closest
public stand-in for Cardiomni's true input. Use the videos to exercise multi-view
integration even if the SYNTAX label itself stays future-work.

## Cardiomni relevance
Exercises the full multi-view pipeline: **Stage 1 dominance** (from PDA origin
across views) + **Stage 3 view selection** (which projection shows a segment best)
+ evidence integration across projections. Dominance is a directly gold-checkable
Stage-1 output.
