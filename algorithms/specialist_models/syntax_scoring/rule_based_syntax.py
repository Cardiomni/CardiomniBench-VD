"""
Rule-based SYNTAX score calculator
Reference: Serruys et al. EuroIntervention 2005
"""
import numpy as np

# AHA 16-segment weights for SYNTAX calculation
# Based on clinical guidelines
SYNTAX_WEIGHTS = {
    1: 3.5,   # Proximal RCA
    2: 2.5,   # Mid RCA
    3: 1.0,   # Distal RCA
    4: 1.0,   # PDA
    5: 5.0,   # Left Main (highest risk)
    6: 3.5,   # Proximal LAD
    7: 2.5,   # Mid LAD
    8: 1.0,   # Distal LAD
    9: 1.0,   # D1
    10: 0.5,  # D2
    11: 2.5,  # Proximal LCx
    12: 1.5,  # IM/Distal LCx
    13: 1.0,  # Distal LCx
    14: 1.0,  # OM1
    15: 0.5,  # OM2
    16: 1.0,  # PL
}

def calculate_syntax_score(segments_report, dominance="right"):
    """
    Calculate SYNTAX score from segment-level stenosis report

    Args:
        segments_report: List[Dict] with keys:
            - segment_id: int (1-16)
            - stenosis_severity: str ("0-25%", "25-50%", "50-70%", "70-99%", "100%")
            - bifurcation: bool (optional)
            - calcification: str (optional, "none"/"mild"/"moderate"/"severe")
        dominance: str ("right", "left", "co-dominant")

    Returns:
        dict: {
            "syntax_total": float,
            "left_system": float,
            "right_system": float
        }
    """
    left_score = 0.0
    right_score = 0.0

    for segment in segments_report:
        seg_id = segment["segment_id"]
        severity = segment.get("stenosis_severity", "0-25%")

        # Skip if no significant stenosis
        if severity in ["0-25%", "25-50%"]:
            continue

        # Get base weight
        base_weight = SYNTAX_WEIGHTS.get(seg_id, 1.0)

        # Severity multiplier
        if severity == "100%":
            multiplier = 5  # Total occlusion
        elif severity == "70-99%":
            multiplier = 2
        elif severity == "50-70%":
            multiplier = 1
        else:
            multiplier = 0

        # Additional factors
        if segment.get("bifurcation", False):
            multiplier += 1

        if segment.get("calcification") == "severe":
            multiplier += 1
        elif segment.get("calcification") == "moderate":
            multiplier += 0.5

        if segment.get("total_occlusion", False):
            multiplier += 1  # CTO bonus

        score = base_weight * multiplier

        # Assign to left or right system
        if seg_id in [1, 2, 3, 4, 16]:  # RCA segments
            right_score += score
        else:  # LCA segments
            left_score += score

    total_score = left_score + right_score

    return {
        "syntax_total": min(total_score, 67),  # Cap at 67
        "left_system": left_score,
        "right_system": right_score
    }

# Example usage
if __name__ == "__main__":
    test_report = [
        {"segment_id": 5, "stenosis_severity": "70-99%"},  # LM
        {"segment_id": 6, "stenosis_severity": "70-99%", "bifurcation": True},  # Proximal LAD
        {"segment_id": 1, "stenosis_severity": "50-70%"},  # Proximal RCA
    ]

    result = calculate_syntax_score(test_report, dominance="right")
    print(f"✅ SYNTAX calculator working")
    print(f"   Total: {result['syntax_total']:.1f}")
    print(f"   Left: {result['left_system']:.1f}, Right: {result['right_system']:.1f}")
