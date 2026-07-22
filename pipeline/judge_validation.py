"""Judge validation — prove the LLM judge is reliable before using it.

Following BiomniBench-DA methodology: before trusting an LLM judge to grade
agent predictions, validate it against expert human labels. Run multiple judge
models on a validation set with expert A/B/C grades, then report:

    - Inter-judge agreement (Cohen's κ, Fleiss' κ for 3+ judges)
    - Exact-match accuracy vs expert consensus
    - Per-dimension breakdown
    - Recommended judge model (highest κ and accuracy)

This ensures the "ruler is accurate before measuring with it" — the core
principle behind BiomniBench-DA's process-level evaluation.

Usage:
    python -m pipeline.judge_validation \\
        --validation-cases data/validation_cases/ \\
        --judge-models claude-opus-4-8,claude-sonnet-4,gpt-4o \\
        --output results/judge_validation.json

Each validation case needs:
    - case_dir/gold_standard.yaml (with the prediction)
    - case_dir/expert_grades.yaml (expert A/B/C labels per criterion)
    - case_dir/rubric.yaml (or use default)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import JudgeConfig, PipelineConfig
from .judge_backends import make_judge
from .scoring import score_criterion

logger = logging.getLogger(__name__)


def compute_cohens_kappa(ratings1: List[str], ratings2: List[str]) -> float:
    """Compute Cohen's κ for two raters on the same items.

    Args:
        ratings1: List of ratings from rater 1 (e.g., ["A", "B", "C", ...])
        ratings2: List of ratings from rater 2 (same length)

    Returns:
        Cohen's κ in [-1, 1]; κ > 0.8 = excellent, > 0.6 = good agreement
    """
    if len(ratings1) != len(ratings2):
        raise ValueError("rating lists must have the same length")

    n = len(ratings1)
    if n == 0:
        return 0.0

    # Observed agreement
    agreements = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2)
    p_o = agreements / n

    # Expected agreement by chance
    categories = set(ratings1) | set(ratings2)
    p_e = 0.0
    for cat in categories:
        p1 = sum(1 for r in ratings1 if r == cat) / n
        p2 = sum(1 for r in ratings2 if r == cat) / n
        p_e += p1 * p2

    # Cohen's κ
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def compute_fleiss_kappa(ratings: List[List[str]]) -> float:
    """Compute Fleiss' κ for 3+ raters on the same items.

    Args:
        ratings: List of [rater1_grade, rater2_grade, ...] per item

    Returns:
        Fleiss' κ in [-1, 1]
    """
    if not ratings:
        return 0.0

    n = len(ratings)  # number of items
    k = len(ratings[0])  # number of raters

    if k < 2:
        return 0.0

    # Build category set
    categories = sorted(set(grade for item in ratings for grade in item))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    num_cats = len(categories)

    # Count matrix: n_ij = number of raters who assigned category j to item i
    counts = [[0] * num_cats for _ in range(n)]
    for i, item_ratings in enumerate(ratings):
        for grade in item_ratings:
            j = cat_to_idx[grade]
            counts[i][j] += 1

    # P_i: proportion of agreement for item i
    p_i_sum = 0.0
    for i in range(n):
        sum_sq = sum(c * c for c in counts[i])
        p_i_sum += (sum_sq - k) / (k * (k - 1))
    p_bar = p_i_sum / n

    # P_j: overall proportion of category j across all assignments
    p_j = [0.0] * num_cats
    for j in range(num_cats):
        total = sum(counts[i][j] for i in range(n))
        p_j[j] = total / (n * k)

    p_e_bar = sum(p ** 2 for p in p_j)

    if p_e_bar == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0
    return (p_bar - p_e_bar) / (1.0 - p_e_bar)


def validate_judge(
    validation_cases_dir: Path,
    judge_models: List[str],
    rubric_dimensions_file: Path,
    default_rubric_file: Path,
    api_key_env: str = "ANTHROPIC_API_KEY",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Run judge validation across multiple judge models.

    Args:
        validation_cases_dir: Directory with case_*/expert_grades.yaml
        judge_models: List of model names to validate (e.g., ["claude-opus-4-8"])
        rubric_dimensions_file: Path to rubric_dimensions.yaml
        default_rubric_file: Path to default case rubric
        api_key_env: Environment variable name for API key
        temperature: Judge temperature (0.0 for deterministic)

    Returns:
        Validation report with κ, accuracy, recommended model
    """
    case_dirs = sorted(d for d in validation_cases_dir.glob("case_*") if d.is_dir())
    if not case_dirs:
        logger.warning("no validation cases found in %s", validation_cases_dir)
        return {"error": "no validation cases"}

    logger.info("validating %d judge model(s) on %d case(s)", len(judge_models), len(case_dirs))

    results_per_model: Dict[str, Dict[str, Any]] = {}

    for model_name in judge_models:
        logger.info("validating judge model: %s", model_name)
        judge_cfg = JudgeConfig(
            backend="llm",
            model=model_name,
            temperature=temperature,
            api_key_env=api_key_env,
        )
        judge = make_judge(judge_cfg)

        expert_grades_all: List[str] = []
        judge_grades_all: List[str] = []
        per_dimension: Dict[str, Dict[str, List[str]]] = defaultdict(
            lambda: {"expert": [], "judge": []}
        )

        for case_dir in case_dirs:
            case = _load_case(case_dir)
            gold = case.get("gold_standard", {}) or {}
            prediction = case.get("prediction", {}) or {}
            expert_grades = _load_expert_grades(case_dir)
            rubric = _load_rubric(case_dir, default_rubric_file)

            if not expert_grades or not rubric:
                logger.warning("skipping %s: missing expert_grades or rubric", case_dir.name)
                continue

            for dim in rubric.get("dimensions", []):
                dim_name = dim["dimension_name"]
                for criterion in dim.get("criteria", []):
                    crit_id = criterion.get("criterion_id", "")
                    if criterion.get("evaluation_method") != "llm_judge":
                        continue

                    expert_grade = expert_grades.get(crit_id)
                    if not expert_grade:
                        continue

                    # Run the judge
                    result = score_criterion(criterion, gold, prediction, judge)
                    judge_grade = result.get("grade")

                    if expert_grade and judge_grade:
                        expert_grades_all.append(expert_grade)
                        judge_grades_all.append(judge_grade)
                        per_dimension[dim_name]["expert"].append(expert_grade)
                        per_dimension[dim_name]["judge"].append(judge_grade)

        if not expert_grades_all:
            logger.warning("no llm_judge criteria found for model %s", model_name)
            results_per_model[model_name] = {"error": "no llm_judge criteria"}
            continue

        # Compute overall metrics
        kappa = compute_cohens_kappa(expert_grades_all, judge_grades_all)
        exact_match = sum(1 for e, j in zip(expert_grades_all, judge_grades_all) if e == j)
        accuracy = exact_match / len(expert_grades_all)

        # Per-dimension metrics
        dim_metrics = {}
        for dim_name, grades in per_dimension.items():
            if grades["expert"] and grades["judge"]:
                dim_kappa = compute_cohens_kappa(grades["expert"], grades["judge"])
                dim_exact = sum(1 for e, j in zip(grades["expert"], grades["judge"]) if e == j)
                dim_acc = dim_exact / len(grades["expert"])
                dim_metrics[dim_name] = {
                    "kappa": dim_kappa,
                    "accuracy": dim_acc,
                    "n": len(grades["expert"]),
                }

        results_per_model[model_name] = {
            "kappa": kappa,
            "accuracy": accuracy,
            "n_criteria": len(expert_grades_all),
            "per_dimension": dim_metrics,
        }
        logger.info("%s: κ=%.3f, accuracy=%.3f", model_name, kappa, accuracy)

    # Recommend the model with highest κ (or accuracy as tiebreaker)
    if results_per_model:
        best_model = max(
            results_per_model.keys(),
            key=lambda m: (
                results_per_model[m].get("kappa", -1),
                results_per_model[m].get("accuracy", -1),
            ),
        )
    else:
        best_model = None

    return {
        "validation_cases": len(case_dirs),
        "models_tested": judge_models,
        "results": results_per_model,
        "recommended_model": best_model,
        "interpretation": _interpretation(results_per_model.get(best_model, {}) if best_model else {}),
    }


def _interpretation(result: Dict[str, Any]) -> str:
    """Return human-readable interpretation of validation results."""
    kappa = result.get("kappa", 0.0)
    accuracy = result.get("accuracy", 0.0)

    if kappa > 0.8:
        kappa_interp = "excellent agreement"
    elif kappa > 0.6:
        kappa_interp = "good agreement"
    elif kappa > 0.4:
        kappa_interp = "moderate agreement"
    else:
        kappa_interp = "poor agreement — judge may not be reliable"

    return (
        f"κ={kappa:.3f} ({kappa_interp}), accuracy={accuracy:.1%}. "
        f"{'Judge is reliable for scoring.' if kappa > 0.6 else 'Consider using a different judge model or refining prompts.'}"
    )


def _load_case(case_dir: Path) -> Dict[str, Any]:
    """Load case YAML (gold_standard + prediction)."""
    for candidate in ("case.yaml", "gold_standard.yaml", "task.yaml"):
        p = case_dir / candidate
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _load_expert_grades(case_dir: Path) -> Dict[str, str]:
    """Load expert_grades.yaml: {criterion_id: "A"|"B"|"C"}."""
    p = case_dir / "expert_grades.yaml"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    return data.get("grades", {}) or {}


def _load_rubric(case_dir: Path, default_rubric_file: Path) -> Optional[Dict[str, Any]]:
    """Load per-case or default rubric."""
    local = case_dir / "rubric.yaml"
    if local.exists():
        with open(local, "r") as f:
            return yaml.safe_load(f)
    if default_rubric_file and default_rubric_file.exists():
        with open(default_rubric_file, "r") as f:
            return yaml.safe_load(f)
    return None


def main():
    """CLI entry point for judge validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate LLM judge reliability")
    parser.add_argument("--validation-cases", required=True, help="Directory with case_*/expert_grades.yaml")
    parser.add_argument("--judge-models", required=True, help="Comma-separated list of models to test")
    parser.add_argument("--rubric-dimensions", default="rubrics/rubric_dimensions.yaml")
    parser.add_argument("--default-rubric", default="rubrics/examples/case_001_rubric.yaml")
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--output", default="results/judge_validation.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.judge_models.split(",")]

    result = validate_judge(
        Path(args.validation_cases),
        models,
        Path(args.rubric_dimensions),
        Path(args.default_rubric),
        api_key_env=args.api_key_env,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Judge validation complete: {output_path}")
    if result.get("recommended_model"):
        print(f"Recommended judge: {result['recommended_model']}")
        print(result.get("interpretation", ""))


if __name__ == "__main__":
    main()
