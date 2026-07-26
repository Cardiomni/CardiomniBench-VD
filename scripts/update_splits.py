#!/usr/bin/env python3
"""
Update splits.yaml with all generated cases from ARCADE and CardioSYNTAX tasks.

Stratifies cases by:
- Difficulty level (easy/medium/hard)
- Task type (segmentation/stenosis/syntax_scoring)
- Dataset source (ARCADE/CardioSYNTAX)

Split ratios: 60% train, 20% val, 20% test
"""

import yaml
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import random


class SplitsGenerator:
    """Generate train/val/test splits for CardiomniBench-VD"""

    def __init__(self, data_root: Path, random_seed: int = 42):
        self.data_root = Path(data_root)
        self.random_seed = random_seed
        random.seed(random_seed)

        self.task_dirs = [
            "arcade_segmentation",
            "arcade_stenosis",
            "cardiosyntax_scoring",
        ]

    def load_all_cases(self) -> Dict[str, Dict]:
        """Load all case metadata from task directories"""
        all_cases = {}

        for task_name in self.task_dirs:
            task_dir = self.data_root / "tasks" / task_name / "cases"
            if not task_dir.exists():
                print(f"Warning: {task_dir} not found, skipping")
                continue

            for case_dir in sorted(task_dir.glob("case_*")):
                task_file = case_dir / "task.yaml"
                if not task_file.exists():
                    continue

                with open(task_file) as f:
                    case_data = yaml.safe_load(f)

                case_id = case_data["case_id"]
                all_cases[case_id] = {
                    "case_id": case_id,
                    "task_type": case_data["case_metadata"]["task_type"],
                    "difficulty": case_data["case_metadata"].get("difficulty_level", "medium"),
                    "source": case_data["case_metadata"].get("source_dataset", "unknown"),
                    "task_dir": task_name,
                }

        print(f"Loaded {len(all_cases)} total cases")
        return all_cases

    def stratified_split(self, cases: Dict[str, Dict],
                        train_ratio: float = 0.6,
                        val_ratio: float = 0.2,
                        test_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
        """
        Stratified split by task_type and difficulty.
        Ensures balanced representation in each split.
        """
        # Group by (task_type, difficulty)
        strata = defaultdict(list)
        for case_id, meta in cases.items():
            key = (meta["task_type"], meta["difficulty"])
            strata[key].append(case_id)

        train, val, test = [], [], []

        for stratum_key, case_ids in strata.items():
            # Shuffle within stratum
            shuffled = case_ids.copy()
            random.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
            test.extend(shuffled[n_train + n_val:])

            print(f"  {stratum_key}: {len(shuffled)} cases -> "
                  f"train={n_train}, val={n_val}, test={n - n_train - n_val}")

        # Final shuffle
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return sorted(train), sorted(val), sorted(test)

    def compute_distributions(self, case_ids: List[str],
                             all_cases: Dict[str, Dict]) -> Dict[str, int]:
        """Compute difficulty and task type distributions"""
        difficulty_dist = defaultdict(int)
        task_dist = defaultdict(int)

        for case_id in case_ids:
            meta = all_cases[case_id]
            difficulty_dist[meta["difficulty"]] += 1
            task_dist[meta["task_type"]] += 1

        return {
            "difficulty": dict(difficulty_dist),
            "task_type": dict(task_dist),
            "total": len(case_ids),
        }

    def generate_splits_yaml(self, output_path: Path = None):
        """Generate complete splits.yaml file"""
        if output_path is None:
            output_path = self.data_root / "splits.yaml"

        # Load all cases
        all_cases = self.load_all_cases()

        if not all_cases:
            print("No cases found!")
            return

        # Stratified split
        print("\nStratified splitting by (task_type, difficulty):")
        train_ids, val_ids, test_ids = self.stratified_split(all_cases)

        # Compute distributions
        train_dist = self.compute_distributions(train_ids, all_cases)
        val_dist = self.compute_distributions(val_ids, all_cases)
        test_dist = self.compute_distributions(test_ids, all_cases)

        # Build splits.yaml structure
        splits_data = {
            "split_version": "1.0.0",
            "creation_date": "2026-07-25",
            "split_strategy": "stratified",
            "random_seed": self.random_seed,
            "description": "Train/val/test splits for CardiomniBench-VD (ARCADE + CardioSYNTAX public data)",

            "ratios": {
                "train": 0.60,
                "val": 0.20,
                "test": 0.20,
            },

            "stratification": {
                "by_difficulty": {
                    "easy": "Single-vessel, clear stenosis or simple segmentation",
                    "medium": "Multi-vessel or moderate complexity",
                    "hard": "High SYNTAX score, many segments, or complex anatomy",
                },
                "by_task_type": {
                    "arcade_segmentation": "Vessel structure segmentation (ARCADE)",
                    "arcade_stenosis": "Stenosis region detection (ARCADE)",
                    "cardiosyntax_scoring": "SYNTAX score prediction (CardioSYNTAX)",
                },
            },

            "train": {
                "case_ids": train_ids,
                "num_cases": len(train_ids),
                "distribution": train_dist,
                "purpose": "Agent development, few-shot learning, model training",
            },

            "val": {
                "case_ids": val_ids,
                "num_cases": len(val_ids),
                "distribution": val_dist,
                "purpose": "Hyperparameter tuning, judge calibration, checkpoint selection",
            },

            "test": {
                "case_ids": test_ids,
                "num_cases": len(test_ids),
                "distribution": test_dist,
                "purpose": "Final held-out evaluation reported in paper",
                "embargo": False,  # Public data, not embargoed
            },

            "constraints": [
                "Each split has balanced representation of task types",
                "Each split has balanced representation of difficulty levels",
                "No overlap between splits (disjoint sets)",
                "Reproducible with random_seed=42",
            ],

            "dataset_summary": {
                "total_cases": len(all_cases),
                "arcade_segmentation": len([c for c in all_cases.values()
                                           if c["task_type"] == "arcade_segmentation"]),
                "arcade_stenosis": len([c for c in all_cases.values()
                                       if c["task_type"] == "arcade_stenosis"]),
                "cardiosyntax_scoring": len([c for c in all_cases.values()
                                            if c["task_type"] == "cardiosyntax_scoring"]),
            },
        }

        # Write splits.yaml
        with open(output_path, 'w') as f:
            yaml.dump(splits_data, f, default_flow_style=False,
                     allow_unicode=True, sort_keys=False)

        print(f"\n✓ Written {output_path}")
        print(f"\nSplit Summary:")
        print(f"  Train: {len(train_ids)} cases ({len(train_ids)/len(all_cases)*100:.1f}%)")
        print(f"  Val:   {len(val_ids)} cases ({len(val_ids)/len(all_cases)*100:.1f}%)")
        print(f"  Test:  {len(test_ids)} cases ({len(test_ids)/len(all_cases)*100:.1f}%)")
        print(f"  Total: {len(all_cases)} cases")

        return splits_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate splits.yaml for CardiomniBench-VD")
    parser.add_argument("--data-root", type=Path,
                       default=Path("/mnt/aliyunsb/Cardiomni/CardiomniBench-VD/data"),
                       help="Path to data directory")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output path for splits.yaml (default: data/splits.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    generator = SplitsGenerator(args.data_root, random_seed=args.seed)
    generator.generate_splits_yaml(output_path=args.output)


if __name__ == "__main__":
    main()
