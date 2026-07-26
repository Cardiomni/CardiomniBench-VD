#!/usr/bin/env python3
"""
SYNTAX Score Calculator Baseline Agent

FUSION-ERA LEGACY — SUPERSEDED, RETURNS MOCK PREDICTIONS
────────────────────────────────────────────────────────────────────────────────
Kept for reference only. Two reasons it is not usable:

1. It returns placeholder output. ``calculate_syntax_score()`` returns a
   hard-coded dict with zeros; the rule-based scoring logic is a TODO.

2. Its schema is fusion-era. It reads ``input.dsa`` / ``input.lesions`` and
   writes ``comprehensive_scoring`` / ``clinical_decision``, none of which
   exist in the current four public tasks (see data/tasks/AGENT_SPEC.md).

Per PROPOSAL.md §2.6, SYNTAX score calculation is integrated into the Cardiomni
agent's Stage 4 lesion assessment, not a separate baseline. The rule-based
implementation, if needed, should be a utility function rather than a CLI agent.
────────────────────────────────────────────────────────────────────────────────

Rule-based SYNTAX score calculator as baseline agent.
Uses the validated syntax scoring logic from specialist_models.
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "algorithms/specialist_models"))


class SYNTAXAgent:
    """SYNTAX score calculator baseline agent"""

    def __init__(self):
        self.syntax_calculator = None

    def load_calculator(self):
        """Load SYNTAX calculator"""
        if self.syntax_calculator is not None:
            return

        print("[SYNTAX] Loading calculator...")
        # TODO: Import actual SYNTAX scoring logic
        # from syntax_scoring import SYNTAXCalculator
        # self.syntax_calculator = SYNTAXCalculator()
        print("[SYNTAX] ✓ Calculator loaded (placeholder)")

    def calculate_syntax_score(self, lesions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate SYNTAX score from lesion data

        Args:
            lesions: List of lesions with location and stenosis

        Returns:
            dict: SYNTAX score breakdown
        """
        self.load_calculator()

        # TODO: Implement actual SYNTAX scoring
        # 1. Map lesions to SYNTAX segments
        # 2. Apply SYNTAX scoring algorithm
        # 3. Calculate left/right/total scores

        # Placeholder
        return {
            "total": 0.0,
            "left": 0.0,
            "right": 0.0,
            "risk_tier": "low",
            "lesion_scores": [],
        }

    def run_task(self, task_spec_path: Path, output_path: Path):
        """Execute SYNTAX scoring task"""
        with open(task_spec_path) as f:
            task_spec = json.load(f)

        case_id = task_spec.get("case_id", "unknown")
        task_type = task_spec.get("task_type", "unknown")
        print(f"[SYNTAX] Scoring {case_id} (task: {task_type})...")

        # Extract lesion data from DSA findings
        input_data = task_spec.get("input", {})
        dsa_data = input_data.get("dsa", {})

        # Parse lesions from views or explicit lesions field
        lesions = input_data.get("lesions", [])
        if not lesions and dsa_data:
            # Construct lesions from DSA views if needed
            views = dsa_data.get("views", [])
            print(f"[SYNTAX] Found {len(views)} DSA views (no explicit lesions)")

        # Calculate SYNTAX score
        syntax_result = self.calculate_syntax_score(lesions)

        # Build prediction matching benchmark schema
        prediction = {
            "case_id": case_id,
            "anatomical_localization": {
                "dominance": "",
                "segments_identified": []
            },
            "dsa_findings": {
                "segments": [],
                "timi_flow": [],
                "collaterals": {}
            },
            "comprehensive_scoring": {
                "syntax_score": syntax_result,
                "cadrads_per_patient": ""
            },
            "clinical_decision": {
                "recommendation": f"SYNTAX {syntax_result['risk_tier']} risk: {self._get_recommendation(syntax_result)}",
                "rationale": f"SYNTAX score {syntax_result['total']:.1f} indicates {syntax_result['risk_tier']} anatomical complexity"
            },
            "capability_boundary_statement": "SYNTAX calculator specialist: rule-based scoring per ACC/AHA guidelines",
            "reasoning_trace": f"Rule-based SYNTAX 2.0 calculation on {len(lesions)} lesions",
            "report": ""
        }

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)

        print(f"[SYNTAX] ✓ Score={syntax_result['total']:.1f}, {syntax_result['risk_tier']} → {output_path}")

    def _get_recommendation(self, syntax_result: Dict[str, Any]) -> str:
        """Get clinical recommendation based on SYNTAX score"""
        score = syntax_result['total']
        tier = syntax_result['risk_tier']

        if tier == "low" or score <= 22:
            return "PCI or CABG may be considered based on clinical factors"
        elif tier == "intermediate" or score <= 32:
            return "Team-based decision recommended; consider SYNTAX II score"
        else:  # high risk
            return "CABG preferred for multivessel disease; PCI for select cases only"


def main():
    parser = argparse.ArgumentParser(description="SYNTAX Score Baseline Agent")
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    agent = SYNTAXAgent()
    agent.run_task(args.task_spec, args.output)


if __name__ == "__main__":
    main()
