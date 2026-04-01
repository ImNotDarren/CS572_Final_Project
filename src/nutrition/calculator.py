"""Nutrition calculator using FNDDS reference data.

Scales per-100g nutrient values to actual consumption amounts,
matching the DietAI24 nutritionCalculator.js logic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import openpyxl

from src.config import Settings

logger = logging.getLogger(__name__)


class NutritionCalculator:
    """Calculates nutritional content from food codes and weights."""

    # The five metrics we evaluate against Nutrition5k ground truth
    EVAL_NUTRIENTS = ["Energy (kcal)", "Total Fat (g)", "Carbohydrate (g)", "Protein (g)"]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fndds_lookup: Dict[str, str] = {}
        self._food_portions: Dict = {}
        self._nutrient_ref: Dict[str, Dict[str, float]] = {}

        self._load_fndds_lookup()
        self._load_food_portions()
        self._load_nutrient_reference()

    def _load_fndds_lookup(self) -> None:
        """Load food code -> description mapping from fndds.json."""
        path = self._settings.fndds_json_path
        if path.exists():
            with open(path, "r") as f:
                self._fndds_lookup = json.load(f)
            logger.info("Loaded %d FNDDS food codes", len(self._fndds_lookup))
        else:
            logger.warning("FNDDS lookup file not found: %s", path)

    def _load_food_portions(self) -> None:
        """Load portion reference data from foodPortions.json."""
        path = self._settings.food_portions_path
        if path.exists():
            with open(path, "r") as f:
                self._food_portions = json.load(f)
            logger.info("Loaded %d portion references", len(self._food_portions))
        else:
            logger.warning("Food portions file not found: %s", path)

    def _load_nutrient_reference(self) -> None:
        """Load per-100g nutrient values from FNDDS Excel file."""
        path = self._settings.fndds_excel_path
        if not path.exists():
            logger.warning("FNDDS Excel file not found: %s", path)
            return

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active

        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return

        # Find the header row (contains "Food code")
        header_idx = None
        for idx, row in enumerate(rows):
            row_strs = [str(cell).lower() if cell else "" for cell in row]
            if any("food code" in s for s in row_strs):
                header_idx = idx
                break

        if header_idx is None:
            logger.error("Could not find header row with 'Food code' in FNDDS Excel")
            return

        header = [str(h).replace("\n", " ") if h else "" for h in rows[header_idx]]
        food_code_col = None
        for i, h in enumerate(header):
            if "food code" in h.lower():
                food_code_col = i
                break

        # Identify numeric nutrient columns
        nutrient_cols: Dict[str, int] = {}
        for i, h in enumerate(header):
            if i == food_code_col:
                continue
            if any(kw in h.lower() for kw in [
                "energy", "protein", "fat", "carbohydrate", "fiber",
                "sugar", "calcium", "iron", "sodium", "cholesterol",
                "vitamin", "water", "alcohol", "magnesium", "phosphorus",
                "potassium", "zinc", "copper", "selenium", "folate",
                "choline", "retinol", "carotene", "lycopene",
            ]):
                nutrient_cols[h] = i

        for row in rows[header_idx + 1:]:
            try:
                code = str(int(row[food_code_col])).zfill(8) if row[food_code_col] else ""
            except (ValueError, TypeError):
                continue
            if not code:
                continue
            nutrients = {}
            for name, col_idx in nutrient_cols.items():
                val = row[col_idx]
                try:
                    nutrients[name] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    nutrients[name] = 0.0
            self._nutrient_ref[code] = nutrients

        wb.close()
        logger.info(
            "Loaded nutrient reference for %d food codes (%d nutrient columns)",
            len(self._nutrient_ref),
            len(nutrient_cols),
        )

    def is_valid_code(self, food_code: str) -> bool:
        """Check if a food code exists in both FNDDS lookup and nutrient reference."""
        padded = food_code.zfill(8)
        return food_code in self._fndds_lookup and padded in self._nutrient_ref

    def get_food_name(self, food_code: str) -> str:
        """Get the description for a food code."""
        return self._fndds_lookup.get(food_code, "Unknown food")

    def get_portion_reference(self, food_code: str) -> Optional[str]:
        """Get formatted portion reference text for weight estimation."""
        portions = self._food_portions.get(food_code)
        if not portions or "portions" not in portions:
            return None

        lines = []
        for p in portions["portions"]:
            lines.append(f"  {p['description']} = {p['weight_g']}g")
        return "\n".join(lines)

    def get_default_weight(self, food_code: str) -> float:
        """Get a default weight from portion reference, or 100g fallback."""
        portions = self._food_portions.get(food_code)
        if portions and "portions" in portions and portions["portions"]:
            return portions["portions"][0]["weight_g"]
        return 100.0

    def calculate(
        self, food_items: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate total nutrition from a list of food items.

        Args:
            food_items: List of dicts with 'food_code' and 'weight_grams'.

        Returns:
            Dict mapping nutrient names to total values.
        """
        totals: Dict[str, float] = {}

        for item in food_items:
            code = str(item["food_code"]).zfill(8)
            weight = item["weight_grams"]

            if code not in self._nutrient_ref:
                logger.warning("Food code %s not in nutrient reference", code)
                continue

            ref = self._nutrient_ref[code]
            for nutrient_name, per_100g in ref.items():
                scaled = (per_100g * weight) / 100.0
                totals[nutrient_name] = totals.get(nutrient_name, 0.0) + scaled

        return totals

    def calculate_eval_metrics(
        self, food_items: List[Dict[str, float]], total_weight: float
    ) -> Dict[str, float]:
        """Calculate the five evaluation metrics: mass, calories, fat, carbs, protein.

        Args:
            food_items: List of dicts with 'food_code' and 'weight_grams'.
            total_weight: Sum of all weight_grams.

        Returns:
            Dict with keys: mass_g, calories, fat_g, carb_g, protein_g
        """
        nutrition = self.calculate(food_items)

        return {
            "mass_g": total_weight,
            "calories": nutrition.get("Energy (kcal)", 0.0),
            "fat_g": nutrition.get("Total Fat (g)", 0.0),
            "carb_g": nutrition.get("Carbohydrate (g)", 0.0),
            "protein_g": nutrition.get("Protein (g)", 0.0),
        }
