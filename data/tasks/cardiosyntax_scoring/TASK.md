# Task: CardioSYNTAX — Study-level SYNTAX Score + Dominance

> **One line**: Given all multi-view angiography cine videos of one patient,
> predict the coronary dominance and the anatomical SYNTAX score.

## Task type
`cardiosyntax_scoring` — study-level regression (score) + classification (dominance).

## Source
CardioSYNTAX (Ponomarchuk et al., WACV 2025, arXiv:2407.19894). Our subset = the
**60 three-expert-annotated studies** — the studies for which three interventional
cardiologists independently scored SYNTAX. This is the highest-value slice: it
carries a gold *reliability band*, not just a point label. Videos on disk at
`Datasets/CardioSYNTAX/datasets/<uid>/*.npy` (478 videos total, 6–14 per study,
avg 8). Score metadata from `all.json`; expert triples from `three_experts.json`.

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
`task.yaml.gold_standard`:
- **Primary label** (official, paper-aligned): `syntax_score`, `syntax_left`,
  `syntax_right`, `dominance`, `bypass`, `risk_band` (low ≤22 / intermediate
  23–32 / high ≥33).
- **Reliability band** (3 independent expert reads): `expert_scores` (the triple),
  `expert_median`, `expert_mean`, `expert_min`, `expert_max`, `expert_spread`.

Subset stats (60 studies): score range 0–58; **12 normal (0) / 48 with disease**;
risk bands 44 low / 4 intermediate / 12 high; **mean inter-expert spread 8.6 pts**
(max 30) — the built-in ceiling for any predictor. Dominance labeled on **11/60**
(`case_metadata.has_dominance_label` flags them).

## Metric (deferred)
Score: **MAE / RMSE / R²** vs the official `syntax_score`. Dominance:
**accuracy** (on the 11 labeled studies). Reliability-aware aux: **fraction of
predictions inside the 3-expert `[min,max]` band** — credits a prediction that
lands where the experts themselves disagree, matching Cardiomni's tolerance model.

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
