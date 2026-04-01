"""DietAI24 baseline agent — single-pass RAG pipeline.

Replicates the original DietAI24 paper pipeline in Python:
  Image -> Describe -> Multi-query RAG search
  -> Select food codes (vision) -> Estimate weights -> Nutrition

Key: food code selection is dish-level (not ingredient decomposition).
The vision model sees the image + candidate codes and picks the best match.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DietAI24Agent(BaseAgent):
    """Baseline: single-pass food recognition without clarification."""

    def estimate(self, image_path: Path) -> Dict[str, Any]:
        """Run full DietAI24 pipeline on a food image.

        Pipeline steps:
            1. Describe food from image (vision)
            2. Generate multiple RAG queries from description
            3. Search ChromaDB for candidate food entries
            4. Select best FNDDS food codes via vision + candidates
            5. Estimate weight per food code (vision)
            6. Calculate nutrition totals
        """
        # Step 1: Describe food
        description = self._describe_food(image_path)
        if "can't help" in description.lower():
            logger.warning("Image analysis failed: %s", description[:100])
            return {
                "food_items": [],
                "predicted": {
                    "mass_g": 0.0, "calories": 0.0,
                    "fat_g": 0.0, "carb_g": 0.0, "protein_g": 0.0,
                },
                "description": description,
                "error": "Image analysis failed",
            }

        # Step 2: Generate search queries
        queries = self._generate_queries(description)

        # Step 3: Search ChromaDB with queries
        # Include the original description as an additional query
        all_queries = queries + [description]
        candidates = self._retriever.multi_search(
            all_queries, top_k=self._settings.rag_top_k
        )

        if not candidates:
            logger.warning("No RAG candidates found")
            return {
                "food_items": [],
                "predicted": {
                    "mass_g": 0.0, "calories": 0.0,
                    "fat_g": 0.0, "carb_g": 0.0, "protein_g": 0.0,
                },
                "description": description,
                "error": "No candidates from RAG search",
            }

        # Step 4: Select food codes (vision model sees image + candidates)
        food_codes = self._select_food_codes(candidates, image_path)

        if not food_codes:
            logger.warning("No valid food codes selected")
            return {
                "food_items": [],
                "predicted": {
                    "mass_g": 0.0, "calories": 0.0,
                    "fat_g": 0.0, "carb_g": 0.0, "protein_g": 0.0,
                },
                "description": description,
                "error": "No valid food codes found",
            }

        # Step 5: Estimate weight for each food code
        food_items = []
        total_weight = 0.0
        for code in food_codes:
            weight = self._estimate_weight(code, image_path)
            food_name = self._calculator.get_food_name(code)
            food_items.append({
                "food_code": code,
                "food_name": food_name,
                "weight_grams": weight,
            })
            total_weight += weight

        # Step 6: Calculate nutrition
        predicted = self._calculator.calculate_eval_metrics(
            food_items, total_weight
        )

        return {
            "food_items": food_items,
            "predicted": predicted,
            "description": description,
        }
