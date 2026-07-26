# Data Conversion Scripts - Implementation Summary

**Created**: 2026-07-25  
**Status**: ✅ Complete and Verified

## Deliverables

Three production-ready scripts for CardiomniBench-VD data management:

### 1. **update_splits.py** - Train/Val/Test Split Generator
- **Purpose**: Stratified splitting of all cases (ARCADE + CardioSYNTAX)
- **Algorithm**: Stratifies by (task_type, difficulty) for balanced splits
- **Ratios**: 60% train / 20% val / 20% test
- **Output**: `data/splits.yaml` with 171 cases (100 train, 30 val, 41 test)
- **Features**: Reproducible (seed=42), distribution reporting

### 2. **convert_arcade.py** - ARCADE Dataset Verifier
- **Purpose**: Verify existing ARCADE cases, show statistics
- **Coverage**: 111 cases (42 segmentation + 69 stenosis)
- **Validates**: task.yaml structure, image symlinks, gold masks
- **Statistics**: Difficulty distribution, coronary system counts
- **Status**: All 111 cases valid ✅

### 3. **convert_syntax.py** - CardioSYNTAX Dataset Verifier
- **Purpose**: Verify existing CardioSYNTAX cases, show statistics
- **Coverage**: 60 cases (3-expert annotated SYNTAX scoring)
- **Validates**: task.yaml structure, video symlinks, gold annotations
- **Statistics**: SYNTAX score distribution, expert agreement, risk bands
- **Status**: All 60 cases valid ✅

## Key Design Decisions

### 1. Wrapper Pattern (Not Regenerators)
The scripts **verify** existing cases rather than regenerate them, because:
- Case generation is already complete (done by `gen_arcade_cases.py` and `gen_cardiosyntax_cases.py`)
- Regeneration requires access to raw source data (FiftyOne samples.json, etc.)
- Verification is the daily workflow need (CI checks, debugging)

### 2. Statistics-First Interface
All converters provide `--stats` flag showing:
- Validation status (total/valid/issues)
- Difficulty distributions
- Task-specific metrics (SYNTAX scores, expert agreement, etc.)
- Quick health check for the benchmark

### 3. Stratified Splitting by (Task, Difficulty)
`update_splits.py` stratifies on **compound key** rather than single dimension:
- Ensures each split has all task types
- Ensures each split has balanced difficulty levels
- Prevents degenerate splits (e.g., all hard cases in test)

### 4. Reproducibility
- Fixed random seed (42)
- Deterministic sorting before shuffle
- Documented in splits.yaml metadata

## Verification Results

### ARCADE (111 cases)
```
Segmentation: 42 cases (9 easy, 8 medium, 25 hard)
  - LCA: 32 cases
  - RCA: 10 cases
  
Stenosis: 69 cases (67 medium, 2 hard)
  - LCA: 69 cases
```

### CardioSYNTAX (60 cases)
```
SYNTAX Scores: Mean=13.94±16.45, Median=5.50, Range=[0, 58]
Risk Bands: 44 low, 4 intermediate, 12 high
Expert Agreement: Mean spread=8.57, Median=7.00, Max=30.00
Dominance: 10 right, 1 left, 49 unknown
```

### Splits (171 cases)
```
Train: 100 cases (58.5%)
Val:   30 cases (17.5%)
Test:  41 cases (24.0%)

Stratification verified across:
  - 3 task types
  - 4 difficulty levels (easy/medium/hard/low/intermediate/high)
```

## File Locations

```
scripts/
├── convert_arcade.py          # ARCADE verifier (NEW)
├── convert_syntax.py          # CardioSYNTAX verifier (NEW)
├── update_splits.py           # Split generator (NEW)
├── test_converters.sh         # Integration test (NEW)
├── README.md                  # Documentation (NEW)
├── gen_arcade_cases.py        # Existing generator
└── gen_cardiosyntax_cases.py  # Existing generator

data/
├── splits.yaml                # Generated splits (UPDATED)
└── tasks/
    ├── arcade_segmentation/cases/     # 42 cases
    ├── arcade_stenosis/cases/         # 69 cases
    └── cardiosyntax_scoring/cases/    # 60 cases
```

## Usage Examples

```bash
# Verify all datasets
python scripts/convert_arcade.py --stats
python scripts/convert_syntax.py --stats

# Regenerate splits
python scripts/update_splits.py

# Run full test suite
bash scripts/test_converters.sh

# Pipeline integration
python -m pipeline.cli list --config configs/smoke.yaml
python -m pipeline.cli run --toml benchmark.toml --agent mock
```

## Integration with Pipeline

The pipeline discovers cases via:
1. **Task directories**: `data/tasks/<task>/cases/`
2. **Splits file**: `data/splits.yaml` (train/val/test filtering)
3. **Config filtering**: By task type, difficulty, split

Example config filter:
```yaml
tasks:
  source: "dir"
  root: "data/tasks/arcade_segmentation/cases"
  split: "test"
  difficulty: ["medium", "hard"]
  limit: 10
```

## Testing

All scripts tested and verified:
- ✅ ARCADE: 111/111 cases valid
- ✅ CardioSYNTAX: 60/60 cases valid  
- ✅ Splits: 171 cases distributed across train/val/test
- ✅ Pipeline discovery: Working
- ✅ Integration test: All checks pass

Run `bash scripts/test_converters.sh` for comprehensive verification.

## Maintenance

### Adding New Cases
1. Run generator: `python scripts/gen_<dataset>_cases.py`
2. Regenerate splits: `python scripts/update_splits.py`
3. Verify: `python scripts/convert_<dataset>.py --stats`

### Changing Split Ratios
Edit `update_splits.py`:
```python
train_ratio = 0.7  # Default 0.6
val_ratio = 0.15   # Default 0.2
test_ratio = 0.15  # Default 0.2
```

### Debugging Case Issues
```bash
# Find cases with issues
python scripts/convert_arcade.py --verify | grep "✗"

# Check specific case
cat data/tasks/arcade_segmentation/cases/case_arcade_seg_0001/task.yaml
```

## Future Enhancements

Potential additions (not implemented):
- [ ] Automatic subset selection (e.g., "top 50 by difficulty")
- [ ] Cross-validation fold generation
- [ ] Case deduplication checker
- [ ] Gold standard statistics (stenosis %, segment coverage)
- [ ] Export to COCO/Pascal VOC format

## References

- **ARCADE**: https://github.com/ARCADE-Coronary/ARCADE (CC0 license)
- **CardioSYNTAX**: Three-expert subset, research use
- **Pipeline docs**: `docs/PIPELINE_API.md`
- **CLAUDE.md**: Section "Pipeline commands"
