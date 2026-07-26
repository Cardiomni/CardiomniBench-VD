# Data Conversion Scripts for CardiomniBench-VD

This directory contains scripts for converting public datasets (ARCADE, CardioSYNTAX) to the CardiomniBench-VD format and managing train/val/test splits.

## Overview

CardiomniBench-VD currently includes **171 public cases**:
- **ARCADE**: 111 cases (42 segmentation + 69 stenosis detection)
- **CardioSYNTAX**: 60 cases (SYNTAX score prediction with 3-expert annotations)

All cases have been pre-generated and are ready to use. The scripts below provide verification, statistics, and regeneration utilities.

## Quick Start

```bash
# Verify all datasets
python scripts/convert_arcade.py --stats
python scripts/convert_syntax.py --stats

# Regenerate splits.yaml (train/val/test)
python scripts/update_splits.py

# Regenerate all cases (if needed)
python scripts/gen_arcade_cases.py
python scripts/gen_cardiosyntax_cases.py
```

## Scripts

### 1. `update_splits.py` - Train/Val/Test Split Generator

**Purpose**: Generate stratified train/val/test splits for all cases.

**Features**:
- Stratified splitting by task type and difficulty level
- 60/20/20 train/val/test ratio
- Reproducible with random seed
- Balanced representation across splits

**Usage**:
```bash
python scripts/update_splits.py
python scripts/update_splits.py --data-root /path/to/data --seed 42
```

**Output**: `data/splits.yaml`

**Current Split Summary** (171 cases):
- Train: 100 cases (58.5%)
- Val: 30 cases (17.5%)
- Test: 41 cases (24.0%)

---

### 2. `convert_arcade.py` - ARCADE Dataset Verifier

**Purpose**: Verify existing ARCADE cases and show statistics.

**Dataset**: ARCADE (Angiographic RCA Dataset with Expert Annotations)
- Source: https://github.com/ARCADE-Coronary/ARCADE
- License: CC0 (Public Domain)
- Tasks: Vessel segmentation + stenosis detection

**Usage**:
```bash
# Show statistics and verify cases
python scripts/convert_arcade.py --stats

# Verify only (no stats)
python scripts/convert_arcade.py --verify

# Regenerate all ARCADE cases
python scripts/gen_arcade_cases.py
```

**Current Dataset** (111 cases):

| Task | Total | Easy | Medium | Hard |
|------|-------|------|--------|------|
| Segmentation | 42 | 9 | 8 | 25 |
| Stenosis | 69 | 0 | 67 | 2 |

**Coronary System Distribution**:
- Segmentation: 32 LCA, 10 RCA
- Stenosis: 69 LCA, 0 RCA

**Case Structure**:
```
data/tasks/arcade_segmentation/cases/case_arcade_seg_XXXX_YYY/
├── image.png                    # Symlink to source image
├── task.yaml                    # Case metadata + gold standard
└── ../../gold/case_*/masks.npz  # Instance masks (bbox-local)

data/tasks/arcade_stenosis/cases/case_arcade_sten_XXXX_YYY/
├── image.png
├── task.yaml
└── ../../gold/case_*/masks.npz
```

---

### 3. `convert_syntax.py` - CardioSYNTAX Dataset Verifier

**Purpose**: Verify existing CardioSYNTAX cases and show statistics.

**Dataset**: CardioSYNTAX
- Source: Three-expert annotated subset (60 studies)
- Task: SYNTAX score prediction from multi-view cine videos
- Includes expert agreement bands for reliability assessment

**Usage**:
```bash
# Show statistics and verify cases
python scripts/convert_syntax.py --stats

# Verify only (no stats)
python scripts/convert_syntax.py --verify

# Regenerate all CardioSYNTAX cases
python scripts/gen_cardiosyntax_cases.py
```

**Current Dataset** (60 cases):

**SYNTAX Score Distribution**:
- Mean: 13.94 ± 16.45
- Median: 5.50
- Range: 0.00 - 58.00

**Risk Bands**:
- Low (≤22): 44 cases (73%)
- Intermediate (23-32): 4 cases (7%)
- High (≥33): 12 cases (20%)

**Expert Agreement**:
- Mean spread: 8.57 points
- Median spread: 7.00 points
- Max spread: 30.00 points

**Case Structure**:
```
data/tasks/cardiosyntax_scoring/cases/case_csyn_XXXX_UID/
├── videos/
│   ├── video1.npy           # Cine frames (T x 512 x 512, uint8)
│   ├── video2.npy
│   └── ...
└── task.yaml                # Gold: syntax_score, expert_scores, dominance
```

---

### 4. `gen_arcade_cases.py` - ARCADE Case Generator

**Purpose**: Generate ARCADE case folders from FiftyOne samples.json.

**Source Data**:
- Location: `/mnt/aliyunsb/Cardiomni/Datasets/ARCADE_FO/`
- Input: `samples.json`, `_subset_segmentation.json`, `_subset_stenosis_multilesion.json`

**Usage**:
```bash
python scripts/gen_arcade_cases.py
```

**Output**:
- `data/tasks/arcade_segmentation/cases/` (42 cases)
- `data/tasks/arcade_stenosis/cases/` (69 cases)
- Gold masks: `data/tasks/*/gold/case_*/masks.npz`

---

### 5. `gen_cardiosyntax_cases.py` - CardioSYNTAX Case Generator

**Purpose**: Generate CardioSYNTAX case folders from three-expert annotations.

**Source Data**:
- Location: `/mnt/aliyunsb/Cardiomni/Datasets/CardioSYNTAX/`
- Input: `.raw_data/CardioSyntax/three_experts.json`, `all.json`
- Videos: `Datasets/CardioSYNTAX/datasets/<uid>/*.npy`

**Usage**:
```bash
python scripts/gen_cardiosyntax_cases.py
```

**Output**:
- `data/tasks/cardiosyntax_scoring/cases/` (60 cases)
- Symlinks to video files in original dataset

---

## Data Format

All cases follow the same structure defined in `task.yaml`:

```yaml
task_version: "1.0.0"
case_id: "case_<dataset>_<id>"

case_metadata:
  task_type: "<task_name>"
  source_dataset: "ARCADE" | "CardioSYNTAX"
  difficulty_level: "easy" | "medium" | "hard"
  coronary_system: "LCA" | "RCA" | "MIXED"

input:
  modality: "XCA" | "XCA_cine"
  image: {...}           # For single-frame tasks
  views: [...]           # For multi-view tasks

expected_output:
  format: "instance_list" | "structured_json"
  target: "..."
  metric: "..."

gold_standard:
  instances: [...]       # For detection/segmentation
  syntax_score: X.X      # For SYNTAX scoring
  dominance: "left" | "right"
  # ... task-specific fields
```

## Pipeline Integration

Cases are discovered by the pipeline through:

1. **Task-based discovery**: `data/tasks/<task_name>/cases/`
2. **Split-based filtering**: `data/splits.yaml` (train/val/test)
3. **Config-based selection**: `configs/*.yaml` can filter by task, split, difficulty

Example pipeline commands:

```bash
# List all cases in a task
python -m pipeline.cli list --config configs/smoke.yaml

# Run evaluation on ARCADE segmentation test set
python -m pipeline.cli run --config configs/arcade_seg.yaml

# Run on specific split
python -m pipeline.cli run --toml benchmark.toml --agent mock \
    --task arcade_segmentation --split test
```

## Dataset Licenses

- **ARCADE**: CC0 (Public Domain) - https://github.com/ARCADE-Coronary/ARCADE
- **CardioSYNTAX**: Research use only - contact dataset maintainers for commercial use

## Maintenance

### Adding New Cases

1. Place raw data in `/mnt/aliyunsb/Cardiomni/Datasets/<dataset>/`
2. Create a generator script following the pattern in `gen_*_cases.py`
3. Run the generator to create `data/tasks/<task>/cases/`
4. Run `update_splits.py` to regenerate splits with new cases
5. Verify with `convert_*.py --stats`

### Regenerating Everything

```bash
# Clean existing cases (CAUTION: destructive)
rm -rf data/tasks/*/cases/case_*
rm -rf data/tasks/*/gold/case_*

# Regenerate
python scripts/gen_arcade_cases.py
python scripts/gen_cardiosyntax_cases.py

# Regenerate splits
python scripts/update_splits.py

# Verify
python scripts/convert_arcade.py --stats
python scripts/convert_syntax.py --stats
```

## Troubleshooting

**Issue**: Cases not discovered by pipeline
- Check `data/splits.yaml` contains case IDs
- Verify `task.yaml` exists in each case directory
- Run pipeline with `--verbose` to see discovery logs

**Issue**: Missing images or videos
- ARCADE: Check symlinks point to `/mnt/aliyunsb/Cardiomni/Datasets/ARCADE_FO/data/`
- CardioSYNTAX: Check symlinks point to `/mnt/aliyunsb/Cardiomni/Datasets/CardioSYNTAX/datasets/`
- Run `convert_*.py --verify` to diagnose

**Issue**: Splits unbalanced
- Adjust ratios in `update_splits.py` (default 60/20/20)
- Change random seed for different shuffle
- Re-run `python scripts/update_splits.py --seed <new_seed>`

## Contact

For questions about:
- **Pipeline integration**: See `docs/PIPELINE_API.md`
- **Dataset formats**: See case `task.yaml` files and this README
- **Rubric/metrics**: See `rubrics/` directory
