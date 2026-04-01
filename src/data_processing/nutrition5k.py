"""Nutrition5k dataset loader and processor.

Parses the dish metadata CSVs and pairs them with overhead RGB images
for evaluation.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ingredient:
    """A single ingredient in a dish."""

    ingr_id: str
    name: str
    grams: float
    calories: float
    fat: float
    carb: float
    protein: float


@dataclass(frozen=True)
class DishSample:
    """A single evaluation sample from Nutrition5k."""

    dish_id: str
    image_path: Path
    total_calories: float
    total_mass: float
    total_fat: float
    total_carb: float
    total_protein: float
    ingredients: List[Ingredient] = field(default_factory=list)

    @property
    def ground_truth(self) -> Dict[str, float]:
        """Return ground truth as a dict matching evaluation metric keys."""
        return {
            "mass_g": self.total_mass,
            "calories": self.total_calories,
            "fat_g": self.total_fat,
            "carb_g": self.total_carb,
            "protein_g": self.total_protein,
        }

    @property
    def ingredient_names(self) -> List[str]:
        """Return list of ingredient names for this dish."""
        return [ingr.name for ingr in self.ingredients]

    @property
    def ingredient_summary(self) -> str:
        """Human-readable ingredient summary with weights."""
        parts = []
        for ingr in self.ingredients:
            parts.append(f"{ingr.name} ({ingr.grams:.0f}g)")
        return ", ".join(parts)


class Nutrition5kDataset:
    """Loads and provides access to the Nutrition5k evaluation dataset."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._data_dir = settings.data_dir
        self._samples: List[DishSample] = []
        self._load()

    def _load(self) -> None:
        """Load selected dishes with their ground truth and images."""
        # Read selected dish IDs
        selected_path = self._data_dir / "selected_dish_ids.txt"
        if not selected_path.exists():
            raise FileNotFoundError(
                f"Selected dish IDs file not found: {selected_path}"
            )

        with open(selected_path, "r") as f:
            selected_ids = {line.strip() for line in f if line.strip()}

        logger.info("Loading %d selected dishes", len(selected_ids))

        # Parse dish metadata
        dish_data: Dict[str, Dict] = {}
        for csv_name in ["dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"]:
            csv_path = self._data_dir / csv_name
            if not csv_path.exists():
                continue
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    dish_id = row[0]
                    if dish_id not in selected_ids:
                        continue

                    # Parse ingredients (every 7 fields after the first 6)
                    ingredients = []
                    i = 6
                    while i + 6 < len(row):
                        ingredients.append(Ingredient(
                            ingr_id=row[i],
                            name=row[i + 1],
                            grams=float(row[i + 2]),
                            calories=float(row[i + 3]),
                            fat=float(row[i + 4]),
                            carb=float(row[i + 5]),
                            protein=float(row[i + 6]),
                        ))
                        i += 7

                    dish_data[dish_id] = {
                        "total_calories": float(row[1]),
                        "total_mass": float(row[2]),
                        "total_fat": float(row[3]),
                        "total_carb": float(row[4]),
                        "total_protein": float(row[5]),
                        "ingredients": ingredients,
                    }

        # Match with images
        images_dir = self._data_dir / "images"
        for dish_id in sorted(selected_ids):
            if dish_id not in dish_data:
                logger.warning("No metadata for %s, skipping", dish_id)
                continue

            image_path = images_dir / f"{dish_id}.png"
            if not image_path.exists():
                logger.warning("No image for %s, skipping", dish_id)
                continue

            data = dish_data[dish_id]
            self._samples.append(DishSample(
                dish_id=dish_id,
                image_path=image_path,
                total_calories=data["total_calories"],
                total_mass=data["total_mass"],
                total_fat=data["total_fat"],
                total_carb=data["total_carb"],
                total_protein=data["total_protein"],
                ingredients=data["ingredients"],
            ))

        logger.info("Loaded %d complete samples", len(self._samples))

    @property
    def samples(self) -> List[DishSample]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DishSample:
        return self._samples[idx]
