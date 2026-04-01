"""Evaluation metrics for nutrition estimation.

Primary metric: Mean Absolute Error (MAE) for mass, calories, fat, carbs, protein.
"""

from typing import Dict, List


METRIC_KEYS = ["mass_g", "calories", "fat_g", "carb_g", "protein_g"]

METRIC_DISPLAY_NAMES = {
    "mass_g": "Mass (g)",
    "calories": "Energy (kcal)",
    "fat_g": "Fat (g)",
    "carb_g": "Carbohydrate (g)",
    "protein_g": "Protein (g)",
}


def compute_per_sample_errors(
    predicted: Dict[str, float],
    ground_truth: Dict[str, float],
) -> Dict[str, float]:
    """Compute absolute error for each metric for a single sample.

    Args:
        predicted: Dict with keys from METRIC_KEYS.
        ground_truth: Dict with keys from METRIC_KEYS.

    Returns:
        Dict mapping metric name to absolute error.
    """
    errors = {}
    for key in METRIC_KEYS:
        pred = predicted.get(key, 0.0)
        gt = ground_truth.get(key, 0.0)
        errors[key] = abs(pred - gt)
    return errors


def compute_mae(
    all_errors: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute Mean Absolute Error across all samples.

    Args:
        all_errors: List of per-sample error dicts from compute_per_sample_errors.

    Returns:
        Dict mapping metric name to MAE.
    """
    if not all_errors:
        return {key: 0.0 for key in METRIC_KEYS}

    n = len(all_errors)
    mae = {}
    for key in METRIC_KEYS:
        total = sum(e[key] for e in all_errors)
        mae[key] = total / n
    return mae
