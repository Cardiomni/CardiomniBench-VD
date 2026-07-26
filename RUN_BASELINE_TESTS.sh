#!/bin/bash
# Quick baseline validation script

PYTHON=/opt/anaconda3/bin/python

echo "=== CardiomniBench-VD Baseline Tests ==="
echo ""

echo "1. Listing registered agents..."
$PYTHON -m pipeline.cli agents --toml benchmark.toml
echo ""

echo "2. Listing available cases..."
$PYTHON -m pipeline.cli list --toml benchmark.toml --agent mock | head -10
echo ""

echo "3. Running mock agent on case_syntax_001..."
$PYTHON -m pipeline.cli run --toml benchmark.toml --agent mock --case case_syntax_001
echo ""

echo "4. Checking output..."
ls -lh runs/cardiomni_bench/rerun_0/case_syntax_001/
cat runs/cardiomni_bench/rerun_0/case_syntax_001/prediction.json | head -20
echo ""

echo "✓ Baseline tests complete"
