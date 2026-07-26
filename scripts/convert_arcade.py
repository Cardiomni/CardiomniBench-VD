#!/usr/bin/env python3
"""
ARCADE Dataset Converter for CardiomniBench-VD

Converts ARCADE (Angiographic RCA Dataset with Expert Annotations) to benchmark format.
ARCADE provides:
- PNG images (512x512) of coronary angiograms
- Segmentation masks for vessel structures (test_case_seg subset)
- Stenosis region annotations (test_case_sten subset)

This converter wraps the existing gen_arcade_cases.py functionality with a cleaner
CLI interface. The actual case generation is already complete.

Reference: https://github.com/ARCADE-Coronary/ARCADE
License: CC0 (Public Domain)

Usage:
  python scripts/convert_arcade.py --verify         # Verify existing cases
  python scripts/convert_arcade.py --regenerate     # Regenerate all cases
  python scripts/convert_arcade.py --stats          # Show statistics
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import argparse
from collections import defaultdict


class ARCADEConverter:
    """Convert ARCADE dataset to CardiomniBench-VD format (wrapper for gen_arcade_cases.py)"""

    # ARCADE segment labels (26 classes) - SYNTAX nomenclature
    SEGMENT_MAPPING = {
        "1": "RCA_1",    # RCA proximal
        "2": "RCA_2",    # RCA mid
        "3": "RCA_3",    # RCA distal
        "16": "RCA_16",  # PDA
        "16a": "RCA_16a",
        "16b": "RCA_16b",
        "16c": "RCA_16c",
        "5": "LM_5",     # Left main
        "6": "LAD_6",    # LAD proximal
        "7": "LAD_7",    # LAD mid
        "8": "LAD_8",    # LAD distal
        "9": "LAD_9",    # D1 (first diagonal)
        "9a": "LAD_9a",
        "10": "LAD_10",  # D2 (second diagonal)
        "10a": "LAD_10a",
        "11": "LCX_11",  # LCX proximal
        "12": "LCX_12",  # OM1 (first obtuse marginal)
        "12a": "LCX_12a",
        "12b": "LCX_12b",
        "13": "LCX_13",  # LCX mid
        "14": "LCX_14",  # OM2
        "14a": "LCX_14a",
        "14b": "LCX_14b",
        "15": "LCX_15",  # LCX distal
    }

    RCA_SEGMENTS = {"1", "2", "3", "4", "16", "16a", "16b", "16c"}
    LCA_SEGMENTS = {"5", "6", "7", "8", "9", "9a", "10", "10a", "11", "12", "12a", "12b", "13", "14", "14a", "14b", "15"}

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.tasks_root = self.data_root / "tasks"

        self.task_dirs = {
            "segmentation": self.tasks_root / "arcade_segmentation" / "cases",
            "stenosis": self.tasks_root / "arcade_stenosis" / "cases",
        }


    def verify_cases(self) -> Dict[str, Any]:
        """Verify all existing ARCADE cases"""
        stats = {
            "segmentation": {"total": 0, "valid": 0, "issues": []},
            "stenosis": {"total": 0, "valid": 0, "issues": []},
        }

        for task_name, task_dir in self.task_dirs.items():
            if not task_dir.exists():
                stats[task_name]["issues"].append(f"Directory not found: {task_dir}")
                continue

            for case_dir in sorted(task_dir.glob("case_*")):
                stats[task_name]["total"] += 1
                case_id = case_dir.name

                # Check task.yaml
                task_file = case_dir / "task.yaml"
                if not task_file.exists():
                    stats[task_name]["issues"].append(f"{case_id}: Missing task.yaml")
                    continue

                try:
                    with open(task_file) as f:
                        task_data = yaml.safe_load(f)

                    # Validate structure
                    required_keys = ["case_id", "case_metadata", "input", "gold_standard"]
                    missing = [k for k in required_keys if k not in task_data]
                    if missing:
                        stats[task_name]["issues"].append(
                            f"{case_id}: Missing keys in task.yaml: {missing}"
                        )
                        continue

                    # Check image symlink
                    image_path = case_dir / task_data["input"]["image"]["file_path"]
                    if not image_path.exists():
                        stats[task_name]["issues"].append(f"{case_id}: Image not found")
                        continue

                    # Check gold standard masks
                    masks_file = task_data["gold_standard"]["masks_file"]
                    masks_abs = case_dir / masks_file
                    if not masks_abs.exists():
                        stats[task_name]["issues"].append(f"{case_id}: Masks file not found")
                        continue

                    stats[task_name]["valid"] += 1

                except Exception as e:
                    stats[task_name]["issues"].append(f"{case_id}: {str(e)}")

        return stats

    def show_statistics(self):
        """Show statistics for existing ARCADE cases"""
        stats = self.verify_cases()

        print("\n" + "="*70)
        print("ARCADE Dataset Statistics")
        print("="*70)

        for task_name, data in stats.items():
            print(f"\n{task_name.upper()}:")
            print(f"  Total cases: {data['total']}")
            print(f"  Valid cases: {data['valid']}")
            print(f"  Issues: {len(data['issues'])}")

            if data["issues"]:
                print(f"\n  Issues found:")
                for issue in data["issues"][:10]:  # Show first 10
                    print(f"    - {issue}")
                if len(data["issues"]) > 10:
                    print(f"    ... and {len(data['issues']) - 10} more")

        # Aggregate difficulty distribution
        print(f"\n" + "="*70)
        print("Difficulty Distribution:")
        print("="*70)

        for task_name, task_dir in self.task_dirs.items():
            if not task_dir.exists():
                continue

            difficulty_counts = defaultdict(int)
            system_counts = defaultdict(int)

            for case_dir in task_dir.glob("case_*"):
                task_file = case_dir / "task.yaml"
                if task_file.exists():
                    with open(task_file) as f:
                        data = yaml.safe_load(f)
                    difficulty_counts[data["case_metadata"].get("difficulty_level", "unknown")] += 1
                    system_counts[data["case_metadata"].get("coronary_system", "unknown")] += 1

            print(f"\n{task_name.upper()}:")
            print(f"  By difficulty: {dict(difficulty_counts)}")
            print(f"  By system: {dict(system_counts)}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="ARCADE Dataset Converter/Verifier for CardiomniBench-VD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_arcade.py --verify          # Verify existing cases
  python scripts/convert_arcade.py --stats           # Show statistics
  python scripts/convert_arcade.py --regenerate      # Regenerate using gen_arcade_cases.py

Note: The actual case generation is handled by gen_arcade_cases.py.
      This script provides verification and statistics utilities.
        """
    )
    parser.add_argument("--data-root", type=Path,
                       default=Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/data"),
                       help="Path to data directory")
    parser.add_argument("--verify", action="store_true",
                       help="Verify existing ARCADE cases")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics for existing cases")
    parser.add_argument("--regenerate", action="store_true",
                       help="Regenerate cases using gen_arcade_cases.py")

    args = parser.parse_args()

    converter = ARCADEConverter(args.data_root)

    if args.regenerate:
        print("To regenerate ARCADE cases, run:")
        print("  python scripts/gen_arcade_cases.py")
        return

    if args.verify or args.stats:
        converter.show_statistics()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

