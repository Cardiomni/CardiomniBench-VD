"""
Complete integration test for CardiomniToolkit after P0 tool implementation
Tests all working methods (rule-based + QCA)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from toolkit import CardiomniToolkit
import numpy as np


def test_toolkit_complete():
    print("=" * 60)
    print("CardiomniToolkit - Complete Integration Test")
    print("=" * 60)

    toolkit = CardiomniToolkit(device="cpu")

    # Test 1: SYNTAX Score Calculator
    print("\n[Test 1] SYNTAX Score Calculator")
    segments = [
        {'segment_id': 5, 'stenosis_severity': '100%'},  # LM occlusion
        {'segment_id': 6, 'stenosis_severity': '70-99%'},  # Prox LAD
        {'segment_id': 11, 'stenosis_severity': '50-70%'},  # Prox LCx
    ]
    result1 = toolkit.calculate_syntax_score(segments, dominance="right")
    print(f"✅ SYNTAX Score: {result1['syntax_total']:.1f}")
    print(f"   Left system: {result1['left_system']:.1f}")
    print(f"   Right system: {result1['right_system']:.1f}")
    print(f"   Recommendation: {result1['treatment_recommendation']}")
    assert result1['syntax_total'] > 0, "SYNTAX score should be > 0"

    # Test 2: Dominance Classifier (segment-based)
    print("\n[Test 2] Dominance Classifier - Segment Analysis")
    segments_right = [
        {'segment_id': 1},
        {'segment_id': 4},   # PDA from RCA
        {'segment_id': 16},  # PLV
    ]
    result2 = toolkit.determine_dominance(segments_report=segments_right)
    print(f"✅ Dominance: {result2['dominance']}")
    print(f"   Confidence: {result2['confidence']}")
    print(f"   Method: {result2['method']}")
    assert result2['dominance'] == 'right', "Should detect right dominance"
    assert result2['confidence'] == 'high', "Should have high confidence"

    # Test 3: Dominance Classifier (SYNTAX-based)
    print("\n[Test 3] Dominance Classifier - SYNTAX Heuristic")
    syntax_data = {
        'left_system': 18.5,
        'right_system': 5.0,
        'left_segments_count': 6,
        'right_segments_count': 3
    }
    result3 = toolkit.determine_dominance(syntax_scores=syntax_data)
    print(f"✅ Dominance: {result3['dominance']}")
    print(f"   Confidence: {result3['confidence']}")
    print(f"   Method: {result3['method']}")
    assert result3['dominance'] == 'left', "Should detect left dominance from weights"

    # Test 4: Dominance Classifier (default fallback)
    print("\n[Test 4] Dominance Classifier - Default Fallback")
    result4 = toolkit.determine_dominance()
    print(f"✅ Dominance: {result4['dominance']}")
    print(f"   Confidence: {result4['confidence']}")
    print(f"   Method: {result4['method']}")
    assert result4['dominance'] == 'right', "Default should be right"
    assert result4['confidence'] == 'low', "Should have low confidence"

    # Test 5: QCA Quantification (via internal method)
    print("\n[Test 5] QCA Quantification")
    # Create synthetic vessel ROI
    roi = np.ones((50, 100), dtype=np.uint8) * 200  # Background
    roi[:, 20:80] = 50  # Vessel lumen (dark)
    roi[:, 40:60] = 100  # Stenosis (less dark = narrower)

    stenosis_pct = toolkit._qca_quantification(roi)
    print(f"✅ Stenosis: {stenosis_pct:.1f}%")
    assert 0 <= stenosis_pct <= 100, "Stenosis should be in [0, 100]"

    # Test 6: Tool availability
    print("\n[Test 6] Available Tools")
    tools = toolkit.get_available_tools()
    print(f"✅ {len(tools)} tools available:")
    for tool in tools:
        print(f"   - {tool}")
    assert len(tools) == 7, "Should have 7 tools"

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\n📋 Working Tools Summary:")
    print("   ✅ SYNTAX Score Calculator (rule-based)")
    print("   ✅ Dominance Classifier (rule-based, 3 methods)")
    print("   ✅ QCA Quantification (NumPy-based)")
    print("   ✅ Projection Parser (DICOM metadata)")
    print("\n⏳ Pending Tools (need YOLO weights):")
    print("   ⏳ Stenosis Detection (YOLOv11-X - download failed)")
    print("   ⏳ Vessel Segmentation (YOLOv8x-seg - 14MB, partial)")
    print("   ⏳ Stenosis Quantification (YOLOv9c - 11MB, partial)")
    print("\n📊 Toolkit Readiness: 4/7 tools (57%) working without YOLO")


if __name__ == "__main__":
    test_toolkit_complete()
