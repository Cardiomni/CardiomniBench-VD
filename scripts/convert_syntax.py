#!/usr/bin/env python3
"""
CardioSYNTAX Dataset Converter for CardiomniBench-VD

Converts CardioSYNTAX metadata to benchmark format.
CardioSYNTAX provides:
- SYNTAX score annotations (total, left, right)
- Three expert annotations for reliability assessment
- Study-level metadata with video references

This converter wraps the existing gen_cardiosyntax_cases.py functionality
with verification and statistics utilities.

Dataset location: /mnt/aliyunsb/Cardiomni/Datasets/CardioSYNTAX/
Reference JSON: three_experts.json (60 studies)

Usage:
  python scripts/convert_syntax.py --verify    # Verify existing cases
  python scripts/convert_syntax.py --stats     # Show statistics
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import argparse
from collections import defaultdict
import statistics


class CardioSYNTAXConverter:
    """Convert CardioSYNTAX dataset to CardiomniBench-VD format"""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.tasks_root = self.data_root / "tasks"
        self.task_dir = self.tasks_root / "cardiosyntax_scoring" / "cases"

    def verify_cases(self) -> Dict[str, Any]:
        """Verify all existing CardioSYNTAX cases"""
        stats = {
            "total": 0,
            "valid": 0,
            "issues": [],
            "missing_videos": 0,
            "missing_dominance": 0,
        }

        if not self.task_dir.exists():
            stats["issues"].append(f"Directory not found: {self.task_dir}")
            return stats

        for case_dir in sorted(self.task_dir.glob("case_csyn_*")):
            stats["total"] += 1
            case_id = case_dir.name

            # Check task.yaml
            task_file = case_dir / "task.yaml"
            if not task_file.exists():
                stats["issues"].append(f"{case_id}: Missing task.yaml")
                continue

            try:
                with open(task_file) as f:
                    task_data = yaml.safe_load(f)

                # Validate structure
                required_keys = ["case_id", "case_metadata", "input", "gold_standard"]
                missing = [k for k in required_keys if k not in task_data]
                if missing:
                    stats["issues"].append(
                        f"{case_id}: Missing keys in task.yaml: {missing}"
                    )
                    continue

                # Check videos directory
                videos_dir = case_dir / "videos"
                if not videos_dir.exists():
                    stats["missing_videos"] += 1
                    stats["issues"].append(f"{case_id}: Videos directory not found")
                    continue

                num_videos = len(list(videos_dir.glob("*.npy")))
                expected_videos = len(task_data["input"]["views"])
                if num_videos != expected_videos:
                    stats["issues"].append(
                        f"{case_id}: Video count mismatch (found {num_videos}, expected {expected_videos})"
                    )

                # Check dominance label
                if task_data["gold_standard"].get("dominance") is None:
                    stats["missing_dominance"] += 1

                stats["valid"] += 1

            except Exception as e:
                stats["issues"].append(f"{case_id}: {str(e)}")

        return stats

    def show_statistics(self):
        """Show statistics for existing CardioSYNTAX cases"""
        stats = self.verify_cases()

        print("\n" + "="*70)
        print("CardioSYNTAX Dataset Statistics")
        print("="*70)

        print(f"\nTotal cases: {stats['total']}")
        print(f"Valid cases: {stats['valid']}")
        print(f"Cases missing videos: {stats['missing_videos']}")
        print(f"Cases missing dominance: {stats['missing_dominance']}")
        print(f"Issues: {len(stats['issues'])}")

        if stats["issues"]:
            print(f"\nIssues found:")
            for issue in stats["issues"][:10]:  # Show first 10
                print(f"  - {issue}")
            if len(stats["issues"]) > 10:
                print(f"  ... and {len(stats['issues']) - 10} more")

        # Aggregate statistics
        print(f"\n" + "="*70)
        print("SYNTAX Score Distribution:")
        print("="*70)

        if not self.task_dir.exists():
            return stats

        syntax_scores = []
        expert_spreads = []
        risk_bands = defaultdict(int)
        dominance_counts = defaultdict(int)

        for case_dir in self.task_dir.glob("case_csyn_*"):
            task_file = case_dir / "task.yaml"
            if task_file.exists():
                with open(task_file) as f:
                    data = yaml.safe_load(f)

                gold = data.get("gold_standard", {})
                score = gold.get("syntax_score")
                if score is not None:
                    syntax_scores.append(score)

                spread = gold.get("expert_spread")
                if spread is not None:
                    expert_spreads.append(spread)

                risk_band = gold.get("risk_band", "unknown")
                risk_bands[risk_band] += 1

                dominance = gold.get("dominance")
                if dominance:
                    dominance_counts[dominance] += 1

        if syntax_scores:
            print(f"\nSYNTAX Scores (n={len(syntax_scores)}):")
            print(f"  Mean: {statistics.mean(syntax_scores):.2f}")
            print(f"  Median: {statistics.median(syntax_scores):.2f}")
            print(f"  Min: {min(syntax_scores):.2f}")
            print(f"  Max: {max(syntax_scores):.2f}")
            if len(syntax_scores) > 1:
                print(f"  Std Dev: {statistics.stdev(syntax_scores):.2f}")

        if expert_spreads:
            print(f"\nExpert Agreement (spread, n={len(expert_spreads)}):")
            print(f"  Mean spread: {statistics.mean(expert_spreads):.2f}")
            print(f"  Median spread: {statistics.median(expert_spreads):.2f}")
            print(f"  Max spread: {max(expert_spreads):.2f}")

        print(f"\nRisk Bands:")
        for band, count in sorted(risk_bands.items()):
            print(f"  {band}: {count}")

        print(f"\nDominance:")
        for dom, count in sorted(dominance_counts.items()):
            print(f"  {dom}: {count}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="CardioSYNTAX Dataset Converter/Verifier for CardiomniBench-VD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_syntax.py --verify     # Verify existing cases
  python scripts/convert_syntax.py --stats      # Show statistics
  python scripts/convert_syntax.py --regenerate # Regenerate using gen_cardiosyntax_cases.py

Note: The actual case generation is handled by gen_cardiosyntax_cases.py.
      This script provides verification and statistics utilities.
        """
    )
    parser.add_argument("--data-root", type=Path,
                       default=Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/data"),
                       help="Path to data directory")
    parser.add_argument("--verify", action="store_true",
                       help="Verify existing CardioSYNTAX cases")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics for existing cases")
    parser.add_argument("--regenerate", action="store_true",
                       help="Regenerate cases using gen_cardiosyntax_cases.py")

    args = parser.parse_args()

    converter = CardioSYNTAXConverter(args.data_root)

    if args.regenerate:
        print("To regenerate CardioSYNTAX cases, run:")
        print("  python scripts/gen_cardiosyntax_cases.py")
        return

    if args.verify or args.stats:
        converter.show_statistics()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
