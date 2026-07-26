"""
Extract projection angles from DICOM metadata
No deep learning needed - use DICOM tags directly
"""
import pydicom
import numpy as np

def classify_projection(dicom_path):
    """
    Extract projection view from DICOM metadata

    Args:
        dicom_path: str, path to DICOM file

    Returns:
        str: View label (e.g., "RAO_30_CAUDAL_20")
    """
    ds = pydicom.dcmread(dicom_path)

    # Read angles from DICOM tags (0018,1510) and (0018,1511)
    try:
        primary_angle = float(ds.PositionerPrimaryAngle)
        secondary_angle = float(ds.PositionerSecondaryAngle)
    except AttributeError:
        return "UNKNOWN"

    # Classify primary angle (RAO/LAO)
    if primary_angle < -10:
        primary = f"RAO_{abs(int(primary_angle))}"
    elif primary_angle > 10:
        primary = f"LAO_{int(primary_angle)}"
    else:
        primary = "AP"

    # Classify secondary angle (Cranial/Caudal)
    if secondary_angle < -10:
        secondary = f"CAUDAL_{abs(int(secondary_angle))}"
    elif secondary_angle > 10:
        secondary = f"CRANIAL_{int(secondary_angle)}"
    else:
        secondary = ""

    # Combine
    view_label = f"{primary}_{secondary}" if secondary else primary

    return view_label

# Example usage
if __name__ == "__main__":
    print("✅ DICOM projection parser ready")
    print("   No weights needed - uses DICOM metadata directly")
    print("   Example output: RAO_30_CAUDAL_20")
