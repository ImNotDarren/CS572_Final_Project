"""Main evaluation script for comparing DietAI24 baseline vs MIA24.

Runs both agents on the Nutrition5k test subset and saves detailed
per-sample and summary results to CSV.
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.agents import DietAI24Agent, MIA24Agent
from src.config import Settings
from src.data_processing import Nutrition5kDataset
from src.nutrition import NutritionCalculator
from src.vector_store import ChromaRetriever

from .metrics import (
    METRIC_DISPLAY_NAMES,
    METRIC_KEYS,
    compute_mae,
    compute_per_sample_errors,
)

logger = logging.getLogger(__name__)


CSV_FIELDNAMES = [
    "dish_id",
    "method",
    "gt_mass_g",
    "gt_calories",
    "gt_fat_g",
    "gt_carb_g",
    "gt_protein_g",
    "pred_mass_g",
    "pred_calories",
    "pred_fat_g",
    "pred_carb_g",
    "pred_protein_g",
    "ae_mass_g",
    "ae_calories",
    "ae_fat_g",
    "ae_carb_g",
    "ae_protein_g",
    "num_food_codes",
    "food_codes",
    "description",
    "error",
]


def _init_per_sample_csv(output_path: Path) -> None:
    """Create CSV file with header row."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()


def _append_row_to_csv(row: Dict, output_path: Path) -> None:
    """Append a single result row to the CSV file."""
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(row)


def _save_summary_csv(
    summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Save MAE summary across methods to a CSV file."""
    fieldnames = ["method"] + [METRIC_DISPLAY_NAMES[k] for k in METRIC_KEYS]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, mae_vals in summary.items():
            row = {"method": method_name}
            for key in METRIC_KEYS:
                row[METRIC_DISPLAY_NAMES[key]] = f"{mae_vals[key]:.2f}"
            writer.writerow(row)

    logger.info("Saved summary to %s", output_path)


def _run_single_method(
    agent,
    dataset: Nutrition5kDataset,
    method_name: str,
    output_path: Path,
    is_mia24: bool = False,
) -> Dict[str, float]:
    """Run one method on the full dataset, writing each result to CSV immediately.

    Args:
        agent: The estimation agent to use.
        dataset: The Nutrition5k dataset.
        method_name: Name of the method being evaluated.
        output_path: Path to the CSV file (created with header before calling).
        is_mia24: Whether this is the MIA24 agent.

    Returns:
        MAE dict across all samples.
    """
    all_errors: List[Dict[str, float]] = []

    for i, sample in enumerate(tqdm(dataset.samples, desc=method_name)):
        logger.info(
            "[%s] Processing %d/%d: %s",
            method_name, i + 1, len(dataset), sample.dish_id,
        )

        try:
            if is_mia24:
                result = agent.estimate(sample.image_path, dish_sample=sample)
            else:
                result = agent.estimate(sample.image_path)

            predicted = result["predicted"]
            gt = sample.ground_truth
            errors = compute_per_sample_errors(predicted, gt)
            all_errors.append(errors)

            row = {
                "dish_id": sample.dish_id,
                "method": method_name,
                "gt_mass_g": f"{gt['mass_g']:.2f}",
                "gt_calories": f"{gt['calories']:.2f}",
                "gt_fat_g": f"{gt['fat_g']:.2f}",
                "gt_carb_g": f"{gt['carb_g']:.2f}",
                "gt_protein_g": f"{gt['protein_g']:.2f}",
                "pred_mass_g": f"{predicted['mass_g']:.2f}",
                "pred_calories": f"{predicted['calories']:.2f}",
                "pred_fat_g": f"{predicted['fat_g']:.2f}",
                "pred_carb_g": f"{predicted['carb_g']:.2f}",
                "pred_protein_g": f"{predicted['protein_g']:.2f}",
                "ae_mass_g": f"{errors['mass_g']:.2f}",
                "ae_calories": f"{errors['calories']:.2f}",
                "ae_fat_g": f"{errors['fat_g']:.2f}",
                "ae_carb_g": f"{errors['carb_g']:.2f}",
                "ae_protein_g": f"{errors['protein_g']:.2f}",
                "num_food_codes": len(result.get("food_items", [])),
                "food_codes": ";".join(
                    item["food_code"] for item in result.get("food_items", [])
                ),
                "description": result.get("description", "")[:200],
                "error": result.get("error", ""),
            }

        except Exception as e:
            logger.error(
                "[%s] Error processing %s: %s", method_name, sample.dish_id, e
            )
            gt = sample.ground_truth
            errors = {key: gt[key] for key in METRIC_KEYS}
            all_errors.append(errors)

            row = {
                "dish_id": sample.dish_id,
                "method": method_name,
                "gt_mass_g": f"{gt['mass_g']:.2f}",
                "gt_calories": f"{gt['calories']:.2f}",
                "gt_fat_g": f"{gt['fat_g']:.2f}",
                "gt_carb_g": f"{gt['carb_g']:.2f}",
                "gt_protein_g": f"{gt['protein_g']:.2f}",
                "pred_mass_g": "0.00",
                "pred_calories": "0.00",
                "pred_fat_g": "0.00",
                "pred_carb_g": "0.00",
                "pred_protein_g": "0.00",
                "ae_mass_g": f"{errors['mass_g']:.2f}",
                "ae_calories": f"{errors['calories']:.2f}",
                "ae_fat_g": f"{errors['fat_g']:.2f}",
                "ae_carb_g": f"{errors['carb_g']:.2f}",
                "ae_protein_g": f"{errors['protein_g']:.2f}",
                "num_food_codes": 0,
                "food_codes": "",
                "description": "",
                "error": str(e),
            }

        # Write row immediately so progress is visible and data is not lost
        _append_row_to_csv(row, output_path)
        logger.info(
            "[%s] Saved result %d/%d to %s",
            method_name, i + 1, len(dataset), output_path.name,
        )

    mae = compute_mae(all_errors)
    return mae


def run_evaluation(
    methods: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    provider: str = "claude",
) -> None:
    """Run full evaluation pipeline.

    Args:
        methods: List of method names to run. Default: ["DietAI24", "MIA24"]
        max_samples: Limit number of samples (for testing). None = all.
        provider: LLM provider — "openai" or "claude".
    """
    if methods is None:
        methods = ["DietAI24", "MIA24"]

    # Initialize shared components
    settings = Settings.for_provider(provider)
    output_dir = settings.provider_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Provider: %s  |  Vision: %s  |  Chat: %s",
        provider, settings.vision_model, settings.chat_model,
    )

    logger.info("Initializing ChromaDB retriever...")
    retriever = ChromaRetriever(settings)

    logger.info("Initializing nutrition calculator...")
    calculator = NutritionCalculator(settings)

    logger.info("Loading Nutrition5k dataset...")
    dataset = Nutrition5kDataset(settings)

    if max_samples and max_samples < len(dataset):
        logger.info("Limiting to %d samples", max_samples)
        dataset._samples = dataset._samples[:max_samples]

    logger.info("Evaluation: %d samples, methods: %s", len(dataset), methods)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_mae: Dict[str, Dict[str, float]] = {}

    for method_name in methods:
        logger.info("=" * 60)
        logger.info("Running method: %s", method_name)
        logger.info("=" * 60)

        # Create CSV with header before processing starts
        per_sample_path = (
            output_dir / f"{method_name}_results_{timestamp}.csv"
        )
        _init_per_sample_csv(per_sample_path)
        logger.info("Results will be written incrementally to %s", per_sample_path)

        start_time = time.time()

        if method_name == "DietAI24":
            agent = DietAI24Agent(settings, retriever, calculator)
            mae = _run_single_method(
                agent, dataset, method_name, per_sample_path, is_mia24=False
            )
        elif method_name == "MIA24":
            agent = MIA24Agent(settings, retriever, calculator)
            mae = _run_single_method(
                agent, dataset, method_name, per_sample_path, is_mia24=True
            )
        else:
            logger.error("Unknown method: %s", method_name)
            continue

        elapsed = time.time() - start_time
        summary_mae[method_name] = mae

        logger.info("Saved %d rows to %s", len(dataset), per_sample_path)

        # Log MAE
        logger.info("-" * 40)
        logger.info("MAE for %s (%.1f min):", method_name, elapsed / 60)
        for key in METRIC_KEYS:
            logger.info(
                "  %s: %.2f", METRIC_DISPLAY_NAMES[key], mae[key]
            )

    # Save summary
    summary_path = output_dir / f"summary_mae_{timestamp}.csv"
    _save_summary_csv(summary_mae, summary_path)

    # Print final comparison
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    header = f"{'Metric':<20}"
    for method in summary_mae:
        header += f"{method:>15}"
    print(header)
    print("-" * (20 + 15 * len(summary_mae)))
    for key in METRIC_KEYS:
        row = f"{METRIC_DISPLAY_NAMES[key]:<20}"
        for method in summary_mae:
            row += f"{summary_mae[method][key]:>15.2f}"
        print(row)
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
