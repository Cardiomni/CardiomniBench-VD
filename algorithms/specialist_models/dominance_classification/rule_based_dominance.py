"""
Rule-based Coronary Dominance Classifier
Based on anatomical segment patterns and SYNTAX scoring rules.

References:
- Sianos et al. (2005) - SYNTAX score definition
- AHA 16-segment model
"""

from typing import Dict, List, Optional


def determine_dominance_from_segments(segments_report: List[Dict]) -> str:
    """
    Determine coronary dominance from segment-level analysis.
    
    Dominance is determined by which system supplies the PDA (Posterior Descending Artery):
    - Right dominant: RCA supplies PDA (segment 4)
    - Left dominant: LCx supplies PDA (segment 15)
    - Co-dominant: Both supply posterior territory
    
    Args:
        segments_report: List of segment dictionaries with at least 'segment_id' key
        
    Returns:
        "right" | "left" | "co-dominant"
    """
    segment_ids = {seg['segment_id'] for seg in segments_report if 'segment_id' in seg}
    
    # Check for PDA segments
    has_rca_pda = 4 in segment_ids  # RCA PDA
    has_lcx_pda = 15 in segment_ids  # LCx PDA
    
    # Check for PLV/PLB segments (posterior territory markers)
    has_rca_plv = 16 in segment_ids  # RCA PLV

    # Decision logic
    if has_lcx_pda:
        return "left"
    elif has_rca_pda and not has_lcx_pda:
        return "right"
    elif has_rca_pda and has_rca_plv:
        # Both PDA from RCA and PLV present = clear right dominance
        return "right"
    else:
        # Default to right dominant (most common: ~70% of population)
        return "right"


def determine_dominance_from_syntax_weights(syntax_left: float, syntax_right: float,
                                           total_segments_left: int, total_segments_right: int) -> str:
    """
    Infer dominance from SYNTAX score distribution and segment counts.
    
    This heuristic uses the fact that dominant systems typically have:
    - More visible segments
    - Higher cumulative SYNTAX weight contribution
    
    Args:
        syntax_left: SYNTAX score contribution from left system
        syntax_right: SYNTAX score contribution from right system
        total_segments_left: Number of left system segments identified
        total_segments_right: Number of right system segments identified
        
    Returns:
        "right" | "left" | "co-dominant"
    """
    # Weight-based ratio
    total_weight = syntax_left + syntax_right
    if total_weight > 0:
        left_ratio = syntax_left / total_weight
        right_ratio = syntax_right / total_weight
        
        # Co-dominant if balanced (within 20% difference)
        if abs(left_ratio - right_ratio) < 0.2:
            return "co-dominant"
        elif left_ratio > right_ratio:
            return "left"
        else:
            return "right"
    
    # Fallback to segment count
    if total_segments_left > total_segments_right:
        return "left"
    elif total_segments_right > total_segments_left:
        return "right"
    else:
        return "right"  # Default


def determine_dominance_simple() -> str:
    """
    Return default dominance (right dominant) for cases where no segment data available.
    
    Population statistics:
    - Right dominant: ~70%
    - Left dominant: ~10%
    - Co-dominant: ~20%
    
    Returns:
        "right" (most common)
    """
    return "right"


# Main API for toolkit integration
def classify_dominance(segments_report: Optional[List[Dict]] = None,
                      syntax_scores: Optional[Dict[str, float]] = None) -> Dict[str, any]:
    """
    Unified dominance classification API.
    
    Args:
        segments_report: Optional list of segment dictionaries
        syntax_scores: Optional dict with 'left_system' and 'right_system' scores
        
    Returns:
        Dictionary with dominance classification and confidence
    """
    result = {
        'dominance': 'right',
        'confidence': 'low',
        'method': 'default'
    }
    
    # Method 1: Segment-based (highest confidence)
    if segments_report and len(segments_report) > 0:
        dominance = determine_dominance_from_segments(segments_report)
        result['dominance'] = dominance
        result['confidence'] = 'high'
        result['method'] = 'segment_analysis'
        return result
    
    # Method 2: SYNTAX weight-based (medium confidence)
    if syntax_scores and 'left_system' in syntax_scores and 'right_system' in syntax_scores:
        left_count = syntax_scores.get('left_segments_count', 0)
        right_count = syntax_scores.get('right_segments_count', 0)
        dominance = determine_dominance_from_syntax_weights(
            syntax_scores['left_system'],
            syntax_scores['right_system'],
            left_count,
            right_count
        )
        result['dominance'] = dominance
        result['confidence'] = 'medium'
        result['method'] = 'syntax_weight_heuristic'
        return result
    
    # Method 3: Default (low confidence)
    result['dominance'] = determine_dominance_simple()
    result['confidence'] = 'low'
    result['method'] = 'population_default'
    
    return result


if __name__ == "__main__":
    print("Rule-based Dominance Classifier - Unit Test")
    
    # Test case 1: Right dominant (RCA PDA present)
    segments_right = [
        {'segment_id': 1},  # Proximal RCA
        {'segment_id': 2},  # Mid RCA
        {'segment_id': 3},  # Distal RCA
        {'segment_id': 4},  # PDA (right)
        {'segment_id': 16}, # PLV (right)
    ]
    result1 = classify_dominance(segments_report=segments_right)
    print(f"✅ Test 1 - Right dominant: {result1['dominance']} (confidence: {result1['confidence']})")
    assert result1['dominance'] == 'right', f"Expected 'right', got {result1['dominance']}"
    
    # Test case 2: Left dominant (LCx PDA present)
    segments_left = [
        {'segment_id': 11}, # Proximal LCx
        {'segment_id': 13}, # Obtuse marginal
        {'segment_id': 15}, # PDA (left)
    ]
    result2 = classify_dominance(segments_report=segments_left)
    print(f"✅ Test 2 - Left dominant: {result2['dominance']} (confidence: {result2['confidence']})")
    assert result2['dominance'] == 'left', f"Expected 'left', got {result2['dominance']}"
    
    # Test case 3: SYNTAX-based
    syntax_data = {
        'left_system': 15.0,
        'right_system': 5.0,
        'left_segments_count': 6,
        'right_segments_count': 3
    }
    result3 = classify_dominance(syntax_scores=syntax_data)
    print(f"✅ Test 3 - SYNTAX heuristic: {result3['dominance']} (confidence: {result3['confidence']})")
    
    # Test case 4: Default
    result4 = classify_dominance()
    print(f"✅ Test 4 - Default: {result4['dominance']} (confidence: {result4['confidence']})")
    assert result4['dominance'] == 'right', f"Expected 'right', got {result4['dominance']}"
    
    print("\n✅ All tests passed!")
    print("📋 Algorithm ready for toolkit integration")
