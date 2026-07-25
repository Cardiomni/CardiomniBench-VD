"""
Cardiomni Hierarchical Collaboration Toolkit
Wrapper for all specialist models (analogous to EchoAgent's HC toolkit)

Reference: EchoAgent - "Hands" layer for tool orchestration
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

# Add specialist_models to path (avoid importing parent algorithms module)
SPECIALIST_MODELS_DIR = Path(__file__).parent / "specialist_models"
sys.path.insert(0, str(SPECIALIST_MODELS_DIR))


class CardiomniToolkit:
    """
    Hierarchical Collaboration Toolkit for Cardiomni

    Analogous to EchoAgent's HC toolkit:
    - Perceptual Layer: View classification, DICOM parsing
    - Operational Layer: Segmentation, detection, measurement
    - Functional Layer: SYNTAX scoring, clinical decision support
    """

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize toolkit with all specialist models

        Args:
            device: torch device for model inference
        """
        self.device = device
        self._models = {}
        self._load_models()

    def _load_models(self):
        """Lazy load models to save memory"""
        print("[CardiomniToolkit] Initializing models...")

        # YOLO models (lazy loaded)
        self._models['yolo_detector'] = None  # YOLOv11-X
        self._models['yolo_quantifier'] = None  # YOLOv9c
        self._models['yolo_segmenter'] = None  # YOLOv8x-seg

        # Rule-based methods (lazy loaded to avoid import errors)
        self._models['syntax_calculator'] = None
        self._models['projection_parser'] = None

        print("[CardiomniToolkit] ✅ Toolkit initialized (models loaded on-demand)")

    # ==========================================
    # Perceptual Layer: View Understanding
    # ==========================================

    def classify_projection(self, dicom_path: str) -> str:
        """
        Extract projection view from DICOM metadata

        Args:
            dicom_path: Path to DICOM file

        Returns:
            str: View label (e.g., "RAO_30_CAUDAL_20")

        Example:
            >>> toolkit = CardiomniToolkit()
            >>> view = toolkit.classify_projection("case_001/series_01/IM-0001.dcm")
            >>> print(view)  # "RAO_30_CAUDAL_20"
        """
        # Lazy load
        if self._models['projection_parser'] is None:
            from projection_classification.dicom_parser import classify_projection
            self._models['projection_parser'] = classify_projection
            print("[CardiomniToolkit] Loaded projection parser")

        return self._models['projection_parser'](dicom_path)

    def parse_dicom_series(self, series_dir: str) -> Dict:
        """
        Parse entire DICOM series to extract metadata

        Args:
            series_dir: Directory containing DICOM files

        Returns:
            dict: {
                'view': str,
                'num_frames': int,
                'primary_angle': float,
                'secondary_angle': float
            }
        """
        import pydicom
        from pathlib import Path

        dicom_files = sorted(Path(series_dir).glob("*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_dir}")

        # Read first file for metadata
        ds = pydicom.dcmread(dicom_files[0])

        return {
            'view': self.classify_projection(str(dicom_files[0])),
            'num_frames': len(dicom_files),
            'primary_angle': float(ds.PositionerPrimaryAngle) if hasattr(ds, 'PositionerPrimaryAngle') else 0.0,
            'secondary_angle': float(ds.PositionerSecondaryAngle) if hasattr(ds, 'PositionerSecondaryAngle') else 0.0,
            'artery': 'LCA' if 'LCA' in str(series_dir) else 'RCA'  # Heuristic
        }

    # ==========================================
    # Operational Layer: Detection & Segmentation
    # ==========================================

    def detect_stenosis(self, image: np.ndarray, conf: float = 0.5) -> List[Dict]:
        """
        Detect stenosis lesions using YOLOv11-X

        Args:
            image: np.ndarray, shape (H, W, 3) or (H, W)
            conf: Confidence threshold

        Returns:
            List[Dict]: [
                {
                    'bbox': [x, y, w, h],  # Normalized [0-1]
                    'confidence': float,
                    'label': 'stenosis'
                }
            ]

        Example:
            >>> detections = toolkit.detect_stenosis(dsa_image, conf=0.5)
            >>> print(f"Found {len(detections)} stenoses")
        """
        # Lazy load YOLO
        if self._models['yolo_detector'] is None:
            from ultralytics import YOLO
            yolo_dir = ALGORITHMS_DIR / "specialist_models" / "yolo_models"
            self._models['yolo_detector'] = YOLO(str(yolo_dir / "yolov11x.pt"))
            print("[CardiomniToolkit] Loaded YOLOv11-X (stenosis detector)")

        results = self._models['yolo_detector'].predict(image, conf=conf, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x, y, w, h = box.xywhn[0].cpu().numpy()  # Normalized coords
                detections.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'confidence': float(box.conf[0]),
                    'label': 'stenosis'
                })

        return detections

    def segment_vessels(self, image: np.ndarray) -> List[Dict]:
        """
        Segment coronary vessels using YOLOv8x-seg

        Args:
            image: np.ndarray, shape (H, W, 3) or (H, W)

        Returns:
            List[Dict]: [
                {
                    'segment_id': int (1-16, AHA model),
                    'mask': np.ndarray (H, W),
                    'bbox': [x, y, w, h]
                }
            ]

        Note:
            YOLOv8x-seg needs fine-tuning on ARCADE segmentation task
            Current version returns generic vessel masks
        """
        # Lazy load YOLO segmentation
        if self._models['yolo_segmenter'] is None:
            from ultralytics import YOLO
            yolo_dir = ALGORITHMS_DIR / "specialist_models" / "yolo_models"
            self._models['yolo_segmenter'] = YOLO(str(yolo_dir / "yolov8x-seg.pt"))
            print("[CardiomniToolkit] Loaded YOLOv8x-seg (vessel segmenter)")

        results = self._models['yolo_segmenter'].predict(image, verbose=False)

        segments = []
        for r in results:
            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()
            boxes = r.boxes.xywhn.cpu().numpy()

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                segments.append({
                    'segment_id': i + 1,  # Placeholder, needs AHA mapping
                    'mask': mask,
                    'bbox': box.tolist()
                })

        return segments

    def quantify_stenosis(self, image: np.ndarray, stenosis_bbox: List[float]) -> Dict:
        """
        Quantify stenosis severity using YOLOv9c + QCA

        Args:
            image: np.ndarray, shape (H, W, 3) or (H, W)
            stenosis_bbox: [x, y, w, h] normalized coordinates

        Returns:
            dict: {
                'stenosis_percentage': float,
                'category': str ("0-25%", "25-50%", "50-70%", "70-99%", "100%"),
                'reference_diameter_mm': float,
                'minimal_lumen_diameter_mm': float,
                'method': str ('yolo' or 'qca')
            }

        Example:
            >>> result = toolkit.quantify_stenosis(image, [0.5, 0.5, 0.1, 0.1])
            >>> print(result['category'])  # "70-99%"
        """
        # Lazy load YOLOv9c
        if self._models['yolo_quantifier'] is None:
            from ultralytics import YOLO
            yolo_dir = ALGORITHMS_DIR / "specialist_models" / "yolo_models"
            self._models['yolo_quantifier'] = YOLO(str(yolo_dir / "yolov9c.pt"))
            print("[CardiomniToolkit] Loaded YOLOv9c (stenosis quantifier)")

        # Extract ROI
        h, w = image.shape[:2]
        x, y, box_w, box_h = stenosis_bbox
        x1 = int((x - box_w/2) * w)
        y1 = int((y - box_h/2) * h)
        x2 = int((x + box_w/2) * w)
        y2 = int((y + box_h/2) * h)

        roi = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        # Use QCA algorithm (classical method)
        stenosis_pct = self._qca_quantification(roi)

        # Map to clinical categories
        if stenosis_pct < 25:
            category = "0-25%"
        elif stenosis_pct < 50:
            category = "25-50%"
        elif stenosis_pct < 70:
            category = "50-70%"
        elif stenosis_pct < 100:
            category = "70-99%"
        else:
            category = "100%"

        # Estimate diameters (placeholder values)
        ref_diameter = 3.0  # mm, typical reference
        min_diameter = ref_diameter * (1 - stenosis_pct / 100)

        return {
            'stenosis_percentage': stenosis_pct,
            'category': category,
            'reference_diameter_mm': ref_diameter,
            'minimal_lumen_diameter_mm': min_diameter,
            'method': 'qca'
        }

    def _qca_quantification(self, roi: np.ndarray) -> float:
        """
        Classical QCA (Quantitative Coronary Analysis) algorithm

        Args:
            roi: Region of interest containing stenosis

        Returns:
            float: Stenosis percentage
        """
        # Use the numpy-based QCA implementation
        if self._models.get('qca_quantifier') is None:
            from qca_quantification.numpy_qca import quantify_stenosis_from_array
            self._models['qca_quantifier'] = quantify_stenosis_from_array
            print("[CardiomniToolkit] Loaded QCA quantifier")

        # Prepare bbox covering whole ROI
        h, w = roi.shape[:2] if len(roi.shape) == 2 else roi.shape[:2]
        roi_coords = (0, 0, w, h)

        try:
            result = self._models['qca_quantifier'](roi, roi_coords, pixel_spacing=0.2)
            return result['percent_stenosis']
        except Exception as e:
            print(f"[CardiomniToolkit] QCA failed: {e}, using fallback")
            # Fallback: simplified intensity-based method
            if len(roi.shape) == 3:
                roi = roi.mean(axis=2)

            baseline = np.percentile(roi, 90)
            minimum = np.percentile(roi, 10)

            if baseline > 0:
                stenosis_pct = (baseline - minimum) / baseline * 100
                return max(0, min(100, stenosis_pct))
            else:
                return 0.0

    # ==========================================
    # Functional Layer: Clinical Decision Support
    # ==========================================

    def calculate_syntax_score(self, segments_report: List[Dict], dominance: str = "right") -> Dict:
        """
        Calculate SYNTAX score from segment-level stenosis report

        Args:
            segments_report: List[Dict] with keys:
                - segment_id: int (1-16)
                - stenosis_severity: str
                - bifurcation: bool (optional)
                - calcification: str (optional)
            dominance: str ("right", "left", "co-dominant")

        Returns:
            dict: {
                'syntax_total': float (0-67),
                'left_system': float,
                'right_system': float,
                'treatment_recommendation': str
            }

        Example:
            >>> segments = [
            ...     {'segment_id': 5, 'stenosis_severity': '70-99%'},
            ...     {'segment_id': 6, 'stenosis_severity': '50-70%'}
            ... ]
            >>> result = toolkit.calculate_syntax_score(segments, 'right')
            >>> print(result['syntax_total'])  # ~18.5
            >>> print(result['treatment_recommendation'])  # "PCI preferred"
        """
        # Lazy load
        if self._models['syntax_calculator'] is None:
            from syntax_scoring.rule_based_syntax import calculate_syntax_score
            self._models['syntax_calculator'] = calculate_syntax_score
            print("[CardiomniToolkit] Loaded SYNTAX calculator")

        result = self._models['syntax_calculator'](segments_report, dominance)

        # Add treatment recommendation
        total = result['syntax_total']
        if total == 0:
            treatment = "No intervention needed"
        elif total < 23:
            treatment = "PCI preferred"
        elif total < 33:
            treatment = "PCI or CABG (consult heart team)"
        else:
            treatment = "CABG preferred"

        result['treatment_recommendation'] = treatment

        return result

    def determine_dominance(self, segments_report: Optional[List[Dict]] = None,
                           syntax_scores: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Determine coronary dominance from segment analysis or SYNTAX scores

        Args:
            segments_report: Optional list of segment dictionaries with 'segment_id' key
            syntax_scores: Optional dict with 'left_system', 'right_system' scores

        Returns:
            dict: {
                'dominance': str ("right", "left", "co-dominant"),
                'confidence': str ("high", "medium", "low"),
                'method': str
            }

        Example:
            >>> segments = [{'segment_id': 1}, {'segment_id': 4}, {'segment_id': 16}]
            >>> result = toolkit.determine_dominance(segments_report=segments)
            >>> print(result['dominance'])  # "right"
        """
        # Lazy load
        if self._models.get('dominance_classifier') is None:
            from dominance_classification.rule_based_dominance import classify_dominance
            self._models['dominance_classifier'] = classify_dominance
            print("[CardiomniToolkit] Loaded dominance classifier")

        return self._models['dominance_classifier'](segments_report, syntax_scores)

    # ==========================================
    # Utility Methods
    # ==========================================

    def get_available_tools(self) -> List[str]:
        """List all available tools in toolkit"""
        return [
            'classify_projection',
            'parse_dicom_series',
            'detect_stenosis',
            'segment_vessels',
            'quantify_stenosis',
            'calculate_syntax_score',
            'determine_dominance'
        ]

    def health_check(self) -> Dict[str, bool]:
        """Check which models are available"""
        checks = {}

        # Check YOLO models
        yolo_dir = SPECIALIST_MODELS_DIR / "yolo_models"
        checks['yolo_detector'] = (yolo_dir / "yolov11x.pt").exists()
        checks['yolo_quantifier'] = (yolo_dir / "yolov9c.pt").exists()
        checks['yolo_segmenter'] = (yolo_dir / "yolov8x-seg.pt").exists()

        # Check rule-based methods
        syntax_file = SPECIALIST_MODELS_DIR / "syntax_scoring" / "rule_based_syntax.py"
        checks['syntax_calculator'] = syntax_file.exists()

        projection_file = SPECIALIST_MODELS_DIR / "projection_classification" / "dicom_parser.py"
        checks['projection_parser'] = projection_file.exists()

        return checks


# Convenience function for quick access
def get_toolkit(device: str = "cuda:0") -> CardiomniToolkit:
    """Get a configured CardiomniToolkit instance"""
    return CardiomniToolkit(device=device)


if __name__ == "__main__":
    # Test toolkit initialization
    print("Testing CardiomniToolkit...")

    toolkit = get_toolkit(device="cpu")
    print(f"\nAvailable tools: {toolkit.get_available_tools()}")

    print("\nHealth check:")
    health = toolkit.health_check()
    for tool, available in health.items():
        status = "✅" if available else "❌"
        print(f"  {status} {tool}")

    print("\nToolkit ready!")
