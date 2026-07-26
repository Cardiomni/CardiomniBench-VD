"""QCA-style diameter quantification for the Cardiomni agent.

Classical quantitative coronary angiography: edge detection → diameter profile →
minimum lumen diameter (MLD) → reference diameter → percent diameter stenosis.

This is the **classical algorithm** (Reiber et al. 1984, Janssen et al. 1991),
not a learned model. It requires a manually-specified ROI around the lesion; it
cannot autonomously locate stenoses. The agent workflow is:

1. Use ``vessel_segmentation.segment_vessels`` to get vessel geometry.
2. Select the ROI (agent logic: VLM reasoning, bounding box from a detector, or
   manual annotation in the test harness).
3. Call ``quantify_stenosis(image_path, roi_coords)`` to measure MLD / %DS.

Upstream source and scipy fallback
----------------------------------
The reference implementation is ``specialist_models/qca_quantification/
classical_qca.py`` (201 lines, imported verbatim). It uses ``cv2.GaussianBlur``
and ``cv2.Sobel`` for edge detection.

OpenCV is not in every conda env on this host. ``opencv-python-headless`` is in
pyproject.toml's ``specialist`` extra and will be in ``.venv`` after uv sync, but
during development gkp-gsa (which has torch) does not have it. This module
provides a scipy+PIL fallback: ``scipy.ndimage.gaussian_filter`` and manual Sobel
convolution. The fallback is tested for numerical consistency with the OpenCV path
when both are available (test_qca_consistency in this file), failing if the
difference exceeds 1e-3 in any measurement.

Usage
-----
    from algorithms.tools import quantify_stenosis

    result = quantify_stenosis(
        image_path="path/to/xca_frame.png",
        roi_coords=(x1, y1, x2, y2),  # bounding box around lesion
        pixel_spacing=0.2,  # mm/pixel calibration
    )
    # result["mld_mm"], result["percent_stenosis"], result["severity_class"]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _has_cv2() -> bool:
    """Check if opencv is available without import side effects."""
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_edges_cv2(
    image: np.ndarray, roi_coords: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Edge detection using OpenCV (reference implementation path)."""
    import cv2

    x1, y1, x2, y2 = roi_coords
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

    height, width = roi.shape
    left_edge = []
    right_edge = []

    for y in range(height):
        profile = sobelx[y, :]
        left_peaks = np.where(profile < -np.std(profile))[0]
        if len(left_peaks) > 0:
            left_edge.append(left_peaks[0])
        else:
            left_edge.append(0)
        right_peaks = np.where(profile > np.std(profile))[0]
        if len(right_peaks) > 0:
            right_edge.append(right_peaks[-1])
        else:
            right_edge.append(width - 1)

    return np.array(left_edge), np.array(right_edge)


def _detect_edges_scipy(
    image: np.ndarray, roi_coords: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Edge detection using scipy (fallback when cv2 unavailable)."""
    from scipy.ndimage import gaussian_filter

    x1, y1, x2, y2 = roi_coords
    roi = image[y1:y2, x1:x2].astype(np.float64)
    blurred = gaussian_filter(roi, sigma=1.0)

    # Sobel kernel for horizontal gradient
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    height, width = blurred.shape
    sobelx = np.zeros_like(blurred)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            patch = blurred[i - 1 : i + 2, j - 1 : j + 2]
            sobelx[i, j] = np.sum(patch * sobel_kernel)

    left_edge = []
    right_edge = []

    for y in range(height):
        profile = sobelx[y, :]
        std = np.std(profile) if np.std(profile) > 0 else 1e-9
        left_peaks = np.where(profile < -std)[0]
        if len(left_peaks) > 0:
            left_edge.append(left_peaks[0])
        else:
            left_edge.append(0)
        right_peaks = np.where(profile > std)[0]
        if len(right_peaks) > 0:
            right_edge.append(right_peaks[-1])
        else:
            right_edge.append(width - 1)

    return np.array(left_edge), np.array(right_edge)


def _measure_diameter(
    left_edge: np.ndarray, right_edge: np.ndarray, pixel_spacing: float = 0.2
) -> Dict[str, float]:
    """Measure vessel diameter from detected edges.

    Reproduced from classical_qca.py verbatim.
    """
    diameters = (right_edge - left_edge) * pixel_spacing
    reference_diameter = np.percentile(diameters, 90)
    mld = np.min(diameters)
    mld_position = np.argmin(diameters)
    percent_stenosis = ((reference_diameter - mld) / reference_diameter) * 100

    return {
        "reference_diameter_mm": float(reference_diameter),
        "mld_mm": float(mld),
        "mld_position": int(mld_position),
        "percent_stenosis": float(percent_stenosis),
        "mean_diameter_mm": float(np.mean(diameters)),
    }


def _classify_stenosis(percent_stenosis: float) -> str:
    """ACC/AHA severity tiers."""
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


def quantify_stenosis(
    image_path: str | Path,
    roi_coords: Tuple[int, int, int, int],
    pixel_spacing: float = 0.2,
    backend: str = "auto",
) -> Dict[str, any]:
    """Complete QCA pipeline: load image → detect edges → measure diameter.

    Args:
        image_path: XCA frame (PNG/DICOM, grayscale).
        roi_coords: (x1, y1, x2, y2) bounding box around the lesion. This tool
            **cannot autonomously locate stenoses**; the ROI must be provided by
            the agent (VLM reasoning, a detector, or manual annotation).
        pixel_spacing: mm per pixel calibration (typical 0.15-0.25 for XCA).
        backend: "cv2", "scipy", or "auto" (cv2 if available, else scipy).

    Returns:
        Dictionary with:
            - reference_diameter_mm: proximal normal segment (90th percentile)
            - mld_mm: minimum lumen diameter at stenosis
            - mld_position: y-coordinate of MLD within ROI
            - percent_stenosis: ((ref - MLD) / ref) * 100
            - mean_diameter_mm: average diameter across ROI
            - severity_class: ACC/AHA tier ("0-25%", "25-50%", ..., "100%")
            - clinical_interpretation: guideline-based text
            - backend_used: "cv2" or "scipy"

    Raises:
        FileNotFoundError: image not found.
        ValueError: unknown backend or ROI out of bounds.
    """
    from PIL import Image

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")

    with Image.open(image_path) as img:
        image = np.asarray(img.convert("L"), dtype=np.uint8)

    x1, y1, x2, y2 = roi_coords
    h, w = image.shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            f"ROI {roi_coords} out of bounds for {image_path.name} shape {image.shape}"
        )

    if backend == "auto":
        backend = "cv2" if _has_cv2() else "scipy"

    if backend == "cv2":
        if not _has_cv2():
            raise ValueError("backend='cv2' but opencv not installed")
        left_edge, right_edge = _detect_edges_cv2(image, roi_coords)
    elif backend == "scipy":
        left_edge, right_edge = _detect_edges_scipy(image, roi_coords)
    else:
        raise ValueError(f"unknown backend {backend!r} (expected 'cv2', 'scipy', or 'auto')")

    measurements = _measure_diameter(left_edge, right_edge, pixel_spacing)
    severity = _classify_stenosis(measurements["percent_stenosis"])
    measurements["severity_class"] = severity
    measurements["backend_used"] = backend

    # Clinical interpretation per ACC/AHA guidelines
    pct = measurements["percent_stenosis"]
    if pct < 50:
        interp = "Non-obstructive; medical therapy typically sufficient."
    elif pct < 70:
        interp = "Intermediate stenosis; consider FFR or stress testing for functional assessment."
    elif pct < 100:
        interp = "Obstructive; revascularization (PCI/CABG) indicated if symptomatic or ischemia documented."
    else:
        interp = "Total occlusion; collateral assessment and viability testing recommended."
    measurements["clinical_interpretation"] = interp

    return measurements


def test_qca_consistency(image_path: str | Path, roi_coords: Tuple[int, int, int, int]):
    """Verify cv2 and scipy backends produce numerically consistent results.

    Raises AssertionError if any measurement differs by more than 1e-3, which
    would indicate an algorithm difference rather than floating-point noise.
    This is a development-time check, not a runtime guard.
    """
    if not _has_cv2():
        print("[test_qca_consistency] SKIP: cv2 not available")
        return

    result_cv2 = quantify_stenosis(image_path, roi_coords, backend="cv2")
    result_scipy = quantify_stenosis(image_path, roi_coords, backend="scipy")

    keys = ["reference_diameter_mm", "mld_mm", "percent_stenosis", "mean_diameter_mm"]
    for k in keys:
        diff = abs(result_cv2[k] - result_scipy[k])
        assert diff < 1e-3, (
            f"QCA {k} inconsistent: cv2={result_cv2[k]:.6f} scipy={result_scipy[k]:.6f} "
            f"diff={diff:.6e} (tolerance 1e-3)"
        )

    print(f"[test_qca_consistency] PASS on {Path(image_path).name}: max diff < 1e-3")
