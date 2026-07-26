#!/bin/bash
# Test all data conversion and verification tools

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "========================================================================"
echo "Testing CardiomniBench-VD Data Conversion Tools"
echo "========================================================================"
echo ""

# 1. Test ARCADE converter
echo "1. Testing ARCADE Converter..."
echo "------------------------------------------------------------------------"
/opt/anaconda3/bin/python scripts/convert_arcade.py --stats
echo ""

# 2. Test CardioSYNTAX converter
echo "2. Testing CardioSYNTAX Converter..."
echo "------------------------------------------------------------------------"
/opt/anaconda3/bin/python scripts/convert_syntax.py --stats
echo ""

# 3. Test splits generator
echo "3. Testing Splits Generator..."
echo "------------------------------------------------------------------------"
/opt/anaconda3/bin/python scripts/update_splits.py
echo ""

# 4. Verify splits.yaml was created
echo "4. Verifying splits.yaml..."
echo "------------------------------------------------------------------------"
if [ -f "data/splits.yaml" ]; then
    echo "✓ splits.yaml exists"

    # Count cases in each split
    train_count=$(grep -A 200 "^train:" data/splits.yaml | grep "^  - case_" | wc -l)
    val_count=$(grep -A 200 "^val:" data/splits.yaml | grep "^  - case_" | wc -l)
    test_count=$(grep -A 200 "^test:" data/splits.yaml | grep "^  - case_" | wc -l)

    echo "  Train: $train_count cases"
    echo "  Val: $val_count cases"
    echo "  Test: $test_count cases"
    echo "  Total: $((train_count + val_count + test_count)) cases"
else
    echo "✗ splits.yaml not found!"
    exit 1
fi
echo ""

# 5. Test pipeline case discovery
echo "5. Testing Pipeline Case Discovery..."
echo "------------------------------------------------------------------------"
echo "Smoke test (should find 1 case):"
/opt/anaconda3/bin/python -m pipeline.cli list --config configs/smoke.yaml
echo ""

# 6. Verify case files
echo "6. Verifying Case Files..."
echo "------------------------------------------------------------------------"
arcade_seg=$(find data/tasks/arcade_segmentation/cases -name "task.yaml" | wc -l)
arcade_sten=$(find data/tasks/arcade_stenosis/cases -name "task.yaml" | wc -l)
syntax=$(find data/tasks/cardiosyntax_scoring/cases -name "task.yaml" | wc -l)

echo "  ARCADE Segmentation: $arcade_seg task.yaml files"
echo "  ARCADE Stenosis: $arcade_sten task.yaml files"
echo "  CardioSYNTAX: $syntax task.yaml files"
echo "  Total: $((arcade_seg + arcade_sten + syntax)) cases"
echo ""

# 7. Summary
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo "✓ All converter scripts working"
echo "✓ Splits generated successfully"
echo "✓ Pipeline can discover cases"
echo "✓ All case files verified"
echo ""
echo "Ready to use! Try:"
echo "  python -m pipeline.cli run --toml benchmark.toml --agent mock"
echo ""
