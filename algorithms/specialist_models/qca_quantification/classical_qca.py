"""
Classical QCA (Quantitative Coronary Angiography) Algorithm
Based on edge detection and vessel diameter measurement principles.

References:
- Reiber et al. (1984) - First QCA system
- Janssen et al. (1991) - Automated edge detection
"""

import numpy as np
from typing import Dict, Tuple, Optional
import cv2


def detect_vessel_edges(image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vessel edges using gradient-based method.
    
    Args:
        image: Grayscale angiography image
        roi_coords: (x1, y1, x2, y2) bounding box of vessel segment
        
    Returns:
        left_edge, right_edge: Arrays of edge coordinates
    """
    x1, y1, x2, y2 = roi_coords
    roi = image[y1:y2, x1:x2]
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
    
    # Compute gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    
    # Find edges by scanning perpendicular to vessel
    height, width = roi.shape
    left_edge = []
    right_edge = []
    
    for y in range(height):
        profile = sobelx[y, :]
        
        # Find left edge (negative gradient)
        left_peaks = np.where(profile < -np.std(profile))[0]
        if len(left_peaks) > 0:
            left_edge.append(left_peaks[0])
        else:
            left_edge.append(0)
            
        # Find right edge (positive gradient)
        right_peaks = np.where(profile > np.std(profile))[0]
        if len(right_peaks) > 0:
            right_edge.append(right_peaks[-1])
        else:
            right_edge.append(width - 1)
    
    return np.array(left_edge), np.array(right_edge)


def measure_diameter(left_edge: np.ndarray, right_edge: np.ndarray, 
                     pixel_spacing: float = 0.2) -> Dict[str, float]:
    """
    Measure vessel diameter from detected edges.
    
    Args:
        left_edge: Left edge coordinates
        right_edge: Right edge coordinates
        pixel_spacing: mm per pixel (typical: 0.15-0.25mm)
        
    Returns:
        Dictionary with diameter measurements
    """
    # Calculate diameter at each position
    diameters = (right_edge - left_edge) * pixel_spacing
    
    # Reference diameter (proximal normal segment, typically max diameter)
    reference_diameter = np.percentile(diameters, 90)  # Use 90th percentile to avoid outliers
    
    # Minimum lumen diameter (MLD) at stenosis
    mld = np.min(diameters)
    mld_position = np.argmin(diameters)
    
    # Percent diameter stenosis
    percent_stenosis = ((reference_diameter - mld) / reference_diameter) * 100
    
    return {
        'reference_diameter_mm': float(reference_diameter),
        'mld_mm': float(mld),
        'mld_position': int(mld_position),
        'percent_stenosis': float(percent_stenosis),
        'mean_diameter_mm': float(np.mean(diameters))
    }


def classify_stenosis_severity(percent_stenosis: float) -> str:
    """
    Classify stenosis severity according to clinical guidelines.
    
    Args:
        percent_stenosis: Percent diameter stenosis
        
    Returns:
        Severity classification string
    """
    if percent_stenosis < 25:
        return "0-25%"
    elif percent_stenosis < 50:
        return "25-50%"
    elif percent_stenosis < 70:
        return "50-70%"
    elif percent_stenosis < 100:
        return "70-99%"
    else:
        return "100%"


def quantify_stenosis(image_path: str, roi_coords: Tuple[int, int, int, int],
                      pixel_spacing: float = 0.2) -> Dict[str, any]:
    """
    Complete QCA pipeline: load image → detect edges → measure diameter.
    
    Args:
        image_path: Path to angiography image
        roi_coords: (x1, y1, x2, y2) bounding box of vessel segment
        pixel_spacing: mm per pixel calibration
        
    Returns:
        Dictionary with QCA measurements
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Detect edges
    left_edge, right_edge = detect_vessel_edges(image, roi_coords)
    
    # Measure diameter
    measurements = measure_diameter(left_edge, right_edge, pixel_spacing)
    
    # Classify severity
    severity = classify_stenosis_severity(measurements['percent_stenosis'])
    measurements['severity_class'] = severity
    
    # Add clinical interpretation
    if measurements['percent_stenosis'] >= 70:
        measurements['clinical_significance'] = "Hemodynamically significant - intervention indicated"
    elif measurements['percent_stenosis'] >= 50:
        measurements['clinical_significance'] = "Moderate - consider FFR/iFR"
    else:
        measurements['clinical_significance'] = "Non-obstructive"
    
    return measurements


# Mock function for toolkit integration (when ROI detection is available)
def quantify_from_detection(image_path: str, bbox: Dict[str, int],
                            pixel_spacing: float = 0.2) -> Dict[str, any]:
    """
    Quantify stenosis from YOLO detection result.
    
    Args:
        image_path: Path to angiography image
        bbox: Detection bounding box {'x1': int, 'y1': int, 'x2': int, 'y2': int}
        pixel_spacing: mm per pixel calibration
        
    Returns:
        QCA measurements
    """
    roi_coords = (bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'])
    return quantify_stenosis(image_path, roi_coords, pixel_spacing)


if __name__ == "__main__":
    # Test with synthetic data
    print("Classical QCA Algorithm - Unit Test")
    
    # Create synthetic vessel profile
    test_image = np.ones((100, 200), dtype=np.uint8) * 128
    # Simulate vessel with stenosis
    for y in range(100):
        if y < 40 or y > 60:
            # Normal segment
            test_image[y, 50:150] = 50
        else:
            # Stenotic segment (narrower)
            test_image[y, 80:120] = 50
    
    # Save test image
    cv2.imwrite('/tmp/test_vessel.png', test_image)
    
    # Run QCA
    result = quantify_stenosis('/tmp/test_vessel.png', 
                               roi_coords=(30, 0, 170, 100),
                               pixel_spacing=0.2)
    
    print(f"✅ Reference diameter: {result['reference_diameter_mm']:.2f} mm")
    print(f"✅ MLD: {result['mld_mm']:.2f} mm")
    print(f"✅ Stenosis: {result['percent_stenosis']:.1f}%")
    print(f"✅ Severity: {result['severity_class']}")
    print(f"✅ Clinical: {result['clinical_significance']}")
