#!/usr/bin/env python
"""
Parse DSA DICOM files and extract metadata for CardiomniBench-VD annotation.

This script analyzes DSA DICOM files and generates:
1. core_views mapping (view_id → DICOM files + angles)
2. Annotation template (gold_standard.yaml skeleton)
3. Data quality report (missing metadata, intervention frames, etc.)

Usage:
    python scripts/parse_dsa_metadata.py \
        --input .tmp/陈秀川-DSA/Exposure\ 7.5\ fps \
        --output .tmp/陈秀川-DSA/metadata_report.json \
        --template .tmp/陈秀川-DSA/gold_standard_template.yaml

Based on: 冠脉造影数据标注规范与模型训练对齐会 (2026-07-22)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("WARNING: pydicom not available. Install with: pip install pydicom")

import yaml


def parse_dicom_metadata(dicom_path: Path) -> Dict[str, Any]:
    """Extract key metadata from a DSA DICOM file."""
    if not PYDICOM_AVAILABLE:
        return {"error": "pydicom not installed"}

    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)

    metadata = {
        "file_name": dicom_path.name,
        "modality": str(ds.get("Modality", "N/A")),
        "series_description": str(ds.get("SeriesDescription", "N/A")),
        "protocol_name": str(ds.get("ProtocolName", "N/A")),
        "acquisition_number": int(ds.get("AcquisitionNumber", 0)),
        "instance_number": int(ds.get("InstanceNumber", 0)),
        "number_of_frames": int(ds.get("NumberOfFrames", 1)),

        # Key angles for view identification
        "positioner_primary_angle": float(ds.get("PositionerPrimaryAngle", 0.0)),
        "positioner_secondary_angle": float(ds.get("PositionerSecondaryAngle", 0.0)),

        # Image properties
        "rows": int(ds.get("Rows", 0)),
        "columns": int(ds.get("Columns", 0)),

        # Additional context
        "view_position": str(ds.get("ViewPosition", "N/A")),
        "patient_position": str(ds.get("PatientPosition", "N/A")),
    }

    # Detect intervention frames (flag for exclusion)
    desc_lower = metadata["series_description"].lower()
    protocol_lower = metadata["protocol_name"].lower()

    intervention_keywords = ["stent", "balloon", "pci", "wire", "guide", "intervention"]
    metadata["likely_intervention"] = any(
        kw in desc_lower or kw in protocol_lower for kw in intervention_keywords
    )

    return metadata


def classify_view(primary_angle: float, secondary_angle: float) -> Dict[str, Any]:
    """
    Classify DSA view based on angulation (C-arm position).

    Conventions:
    - Primary angle: RAO (right anterior oblique) > 0, LAO (left anterior oblique) < 0
    - Secondary angle: CRA (cranial) > 0, CAU (caudal) < 0

    Returns:
        view_id, description, likely_target_vessels
    """
    # Round to nearest 5 degrees for classification
    pri = round(primary_angle / 5) * 5
    sec = round(secondary_angle / 5) * 5

    # RAO/LAO classification
    if pri > 10:
        rao_lao = f"RAO{int(pri)}"
    elif pri < -10:
        rao_lao = f"LAO{int(abs(pri))}"
    else:
        rao_lao = "AP"  # Anterior-Posterior (straight)

    # CRA/CAU classification
    if sec > 10:
        cra_cau = f"CRA{int(sec)}"
    elif sec < -10:
        cra_cau = f"CAU{int(abs(sec))}"
    else:
        cra_cau = ""

    view_name = f"{rao_lao}_{cra_cau}".strip("_")

    # Infer target vessels based on common clinical practice
    # (This is heuristic; actual vessel depends on catheter position)
    if "RAO" in view_name and "CAU" not in view_name:
        target_vessels = ["RCA"]  # Right coronary views often use RAO
    elif "LAO" in view_name or "CRA" in view_name:
        target_vessels = ["LM", "LAD", "LCX"]  # Left coronary views
    else:
        target_vessels = ["RCA", "LM", "LAD", "LCX"]  # Ambiguous

    return {
        "view_id": view_name.lower(),
        "description": view_name.replace("_", " "),
        "target_vessels": target_vessels,
        "positioner_primary_angle": primary_angle,
        "positioner_secondary_angle": secondary_angle,
    }


def group_by_view(metadata_list: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group DICOM files by view (similar angles = same view)."""
    views = {}

    for meta in metadata_list:
        if meta.get("likely_intervention"):
            continue  # Skip intervention frames

        view_info = classify_view(
            meta["positioner_primary_angle"],
            meta["positioner_secondary_angle"]
        )
        view_id = view_info["view_id"]

        if view_id not in views:
            views[view_id] = {
                "view_info": view_info,
                "dicom_files": []
            }
        views[view_id]["dicom_files"].append(meta["file_name"])

    return views


def generate_gold_standard_template(
    case_id: str,
    views: Dict[str, Any],
    metadata_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate a gold_standard.yaml template for manual annotation."""

    core_views = []
    for view_id, view_data in views.items():
        core_views.append({
            "view_id": view_id,
            "description": view_data["view_info"]["description"],
            "dicom_files": view_data["dicom_files"],
            "positioner_primary_angle": view_data["view_info"]["positioner_primary_angle"],
            "positioner_secondary_angle": view_data["view_info"]["positioner_secondary_angle"],
            "target_vessels": view_data["view_info"]["target_vessels"],
        })

    template = {
        "case_id": case_id,
        "clinical_context": {
            "age": "TODO: fill from medical record",
            "gender": "TODO: M/F",
            "symptoms": "TODO: e.g., 活动后胸闷",
            "risk_factors": "TODO: e.g., hypertension, diabetes",
        },
        "input": {
            "dsa": {
                "dicom_directory": "TODO: relative path",
                "core_views": core_views,
                "exclude_interventional": True,
            }
        },
        "stage0_anatomy": {
            "dominance": "TODO: right/left/co-dominant",
            "segments": "TODO: list all 17 SYNTAX segments with present=true/false",
        },
        "stage1b_dsa": {
            "segments": [
                {
                    "segment_id": "RCA_1",
                    "segment_name": "TODO: e.g., 右冠近端",
                    "best_view": "TODO: view_id from core_views",
                    "stenosis_percent": "TODO: 0-100",
                    "stenosis_grade": "TODO: none/mild/moderate/severe/occluded",
                    "plaque_type": "TODO: calcified/soft/mixed",
                    "timi_flow": "TODO: 0-3",
                    "lesion_morphology": "TODO: type A/B1/B2/C",
                },
                # Add more segments...
                # IMPORTANT: Include negative findings (stenosis_percent=0)
            ]
        },
        "stage3_scoring": {
            "syntax_score": {
                "total": "TODO: calculated from segment scores",
                "risk_tier": "TODO: low/intermediate/high",
            }
        },
        "clinical_decision": {
            "recommendation": "TODO: e.g., 建议右冠近段支架置入",
            "rationale": "TODO: 基于 DSA 的诊断依据",
        },
        "capability_boundary": {
            "notes": "TODO: 需要但缺失的检查（如 FFR、IVUS）",
        }
    }

    return template


def generate_report(
    input_dir: Path,
    metadata_list: List[Dict[str, Any]],
    views: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate a data quality report."""

    report = {
        "input_directory": str(input_dir),
        "total_files": len(metadata_list),
        "intervention_frames_detected": sum(
            1 for m in metadata_list if m.get("likely_intervention")
        ),
        "clean_frames": sum(
            1 for m in metadata_list if not m.get("likely_intervention")
        ),
        "unique_views": len(views),
        "views_summary": {
            view_id: {
                "num_files": len(data["dicom_files"]),
                "description": data["view_info"]["description"],
                "target_vessels": data["view_info"]["target_vessels"],
            }
            for view_id, data in views.items()
        },
        "missing_metadata_warnings": [],
    }

    # Check for missing critical metadata
    for meta in metadata_list:
        if meta["modality"] == "N/A":
            report["missing_metadata_warnings"].append(
                f"{meta['file_name']}: Modality tag missing"
            )
        if meta["positioner_primary_angle"] == 0.0 and meta["positioner_secondary_angle"] == 0.0:
            report["missing_metadata_warnings"].append(
                f"{meta['file_name']}: Angulation data missing or all zeros"
            )

    return report


def main():
    parser = argparse.ArgumentParser(description="Parse DSA DICOM metadata for annotation")
    parser.add_argument("--input", required=True, help="Input directory with DICOM files")
    parser.add_argument("--output", default="metadata_report.json", help="Output JSON report")
    parser.add_argument("--template", default="gold_standard_template.yaml", help="Output YAML template")
    parser.add_argument("--case-id", default="case_unnamed", help="Case ID for the template")

    args = parser.parse_args()

    if not PYDICOM_AVAILABLE:
        print("ERROR: pydicom is required. Install with: pip install pydicom")
        return 1

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 1

    # Find all DICOM files
    dicom_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ])

    if not dicom_files:
        print(f"ERROR: No DICOM files found in {input_dir}")
        return 1

    print(f"Found {len(dicom_files)} files in {input_dir}")

    # Parse metadata
    metadata_list = []
    for dcm_file in dicom_files:
        print(f"Parsing {dcm_file.name}...")
        try:
            meta = parse_dicom_metadata(dcm_file)
            metadata_list.append(meta)
        except Exception as e:
            print(f"  WARNING: Failed to parse {dcm_file.name}: {e}")

    # Group by view
    views = group_by_view(metadata_list)

    # Generate outputs
    report = generate_report(input_dir, metadata_list, views)
    template = generate_gold_standard_template(args.case_id, views, metadata_list)

    # Write report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata report written to: {args.output}")

    # Write template
    with open(args.template, "w") as f:
        yaml.dump(template, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"Annotation template written to: {args.template}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total files: {report['total_files']}")
    print(f"Intervention frames (to exclude): {report['intervention_frames_detected']}")
    print(f"Clean frames: {report['clean_frames']}")
    print(f"Unique views detected: {report['unique_views']}")
    print(f"\nViews:")
    for view_id, summary in report["views_summary"].items():
        print(f"  - {view_id}: {summary['num_files']} files, "
              f"targets {', '.join(summary['target_vessels'])}")

    if report["missing_metadata_warnings"]:
        print(f"\nWARNINGS ({len(report['missing_metadata_warnings'])}):")
        for warn in report["missing_metadata_warnings"][:5]:
            print(f"  - {warn}")

    print(f"\nNext steps:")
    print(f"1. Review {args.template} and fill in TODO fields")
    print(f"2. Verify view classification matches clinical interpretation")
    print(f"3. Add all 17 SYNTAX segments (including negative findings)")
    print(f"4. Calculate SYNTAX Score based on segment annotations")

    return 0


if __name__ == "__main__":
    exit(main())
