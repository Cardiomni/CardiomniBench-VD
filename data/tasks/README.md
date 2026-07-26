# CardiomniBench-VD — Public-Data Task Suite

This directory holds the benchmark's public-data tasks. Each task is a folder;
each case is a folder inside it. The harness is unchanged — it discovers cases by
globbing `case_*` under a task's `cases/` dir (see per-task config in
`configs/tasks/`).

## The suite: 4 datasets → 4 tasks (input → output)

| Task folder | Dataset | Input | Output (the label in the data) | Metric | Cases |
|---|---|---|---|---|---|
| `arcade_segmentation` | ARCADE | 1 XCA frame (512²) | segment instances: SYNTAX id + bbox + mask | mean F1/img | 42 |
| `arcade_stenosis` | ARCADE | 1 XCA frame (512²) | stenosis instances: bbox + mask (location only) | mean F1/img | 69 |
| `cca_segmentation` | CCA | 1 CTA volume (3D) | binary 3D vessel mask | Dice | 20 |
| `cardiosyntax_scoring` | CardioSYNTAX | N cine videos + angles | SYNTAX score (+3-expert band) + dominance | MAE/R²/acc | 60 |

Total **191 cases**. Rubric/scoring is **deferred** — right now every task is a
clean **input → output** contract (input = the images/videos, output = exactly the
labels present in the source data). See each task's `TASK.md`.

## Folder layout (uniform)

```
data/tasks/<task>/
  TASK.md                 # task definition: input contract, output contract, metric
  _cases.json             # generated manifest of case ids
  cases/
    case_<...>/
      task.yaml           # case_metadata + input + gold_standard (gold auto-stripped)
      image.png|.nii.gz   # input, SYMLINK to Datasets/ (portable, no copy)
      videos/*.npy        # (CardioSYNTAX) input videos, symlinks
  gold/                   # (ARCADE only) per-case binary masks, OUTSIDE the case dir
    case_<...>/masks.npz
```

### Case naming
- `case_arcade_seg_<NNNN>_<srcfile>` (e.g. `case_arcade_seg_0001_101-5`)
- `case_arcade_sten_<NNNN>_<srcfile>`
- `case_cca_<NNNN>_<n>`
- `case_csyn_<NNNN>_<uid8>` (last 8 chars of study UID)

### Agent visibility (critical)
The harness builds `task_spec.json` by stripping `gold_standard` from `task.yaml`.
The agent sees: `case_id`, `task_type`, `input`, `expected_output`. It must NOT
read `task.yaml` or the `gold/` tree. Scalar gold lives in `task.yaml.gold_standard`
(stripped); binary gold (ARCADE masks) lives in the sibling `gold/` tree; CCA/
CardioSYNTAX gold are scalars/paths in `task.yaml.gold_standard`.

## Run a task (offline, mock)

```bash
python -m pipeline.cli list --config configs/tasks/arcade_segmentation.yaml   # 42
python -m pipeline.cli run  --config configs/tasks/cardiosyntax_scoring.yaml   # mock
```

## Regenerate the cases

```bash
python scripts/gen_arcade_cases.py        # 42 + 69
python scripts/gen_cca_cases.py           # 20
python scripts/gen_cardiosyntax_cases.py  # 60 (three-expert subset)
```

## Relation to the existing DSA-report task
`data/cases/case_chxc_001/` (中山 real case, prose report) is the separate
DSA-report task and is untouched. This suite is the public-data complement.
See `AGENT_SPEC.md` for how Cardiomni reads/writes across all tasks.
