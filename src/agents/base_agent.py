"""Base agent class with shared OpenAI and RAG utilities."""

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from src.config import Settings
from src.nutrition import NutritionCalculator
from src.prompts import DietAI24Prompts
from src.vector_store import ChromaRetriever

logger = logging.getLogger(__name__)


class BaseAgent:
    """Shared functionality for food recognition agents."""

    def __init__(
        self,
        settings: Settings,
        retriever: ChromaRetriever,
        calculator: NutritionCalculator,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._calculator = calculator
        self._client = OpenAI(api_key=settings.openai_api_key)

    def _retry_api_call(self, fn, max_retries: int = 3):
        """Retry an OpenAI API call with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                logger.warning("API error (attempt %d/%d), retrying in %ds: %s",
                               attempt + 1, max_retries, wait, e)
                time.sleep(wait)

    def _encode_image(self, image_path: Path) -> str:
        """Encode a local image to base64 data URI."""
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"

    def _call_vision(
        self,
        prompt: str,
        image_path: Path,
        system: str = DietAI24Prompts.SYSTEM,
        max_tokens: int = 1024,
    ) -> str:
        """Call OpenAI vision model with an image."""
        image_uri = self._encode_image(image_path)

        def _api_call():
            return self._client.chat.completions.create(
                model=self._settings.vision_model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_uri, "detail": "high"},
                            },
                        ],
                    },
                ],
            )

        response = self._retry_api_call(_api_call)
        content = response.choices[0].message.content
        return (content or "").strip()

    def _call_chat(
        self,
        prompt: str,
        system: str = DietAI24Prompts.SYSTEM,
        max_tokens: int = 1024,
    ) -> str:
        """Call OpenAI chat model (text only)."""

        def _api_call():
            return self._client.chat.completions.create(
                model=self._settings.chat_model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )

        response = self._retry_api_call(_api_call)
        content = response.choices[0].message.content
        return (content or "").strip()

    def _describe_food(self, image_path: Path) -> str:
        """Step 1: Describe the food in the image."""
        description = self._call_vision(
            DietAI24Prompts.FOOD_DESCRIPTION, image_path
        )
        logger.debug("Food description: %s", description[:200])
        return description

    def _generate_queries(self, description: str) -> List[str]:
        """Step 2: Generate multiple search queries from the description."""
        n = self._settings.num_query_variations
        prompt = DietAI24Prompts.RETRIEVE_QUERIES.format(
            n=n, description=description
        )
        response = self._call_chat(prompt)

        # Parse numbered lines
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned:
                queries.append(cleaned)

        logger.debug("Generated %d queries", len(queries))
        return queries[:n]

    def _extract_ingredients(self, description: str) -> List[str]:
        """Step 3: Extract individual ingredients from description."""
        prompt = DietAI24Prompts.INGREDIENT_EXTRACT.format(
            max_items=self._settings.max_ingredients,
            description=description,
        )
        response = self._call_chat(prompt)
        ingredients = [
            item.strip() for item in response.split(",") if item.strip()
        ]
        logger.debug("Extracted %d ingredients: %s", len(ingredients), ingredients)
        return ingredients[: self._settings.max_ingredients]

    def _select_food_codes(
        self, candidates: List[Dict], image_path: Path
    ) -> List[str]:
        """Step 4: Select FNDDS food codes using vision model + candidates.

        The model sees the food image and the candidate code list, then
        picks the codes that best match the visible food(s).
        """
        if not candidates:
            logger.warning("No candidates for food code selection")
            return []

        # Format candidates
        candidate_text = "\n".join(
            f"Food code: {c['food_code']}  Description: {c['description']}"
            for c in candidates
        )

        prompt = DietAI24Prompts.CODE_SELECTION.format(
            candidates=candidate_text,
        )
        response = self._call_vision(prompt, image_path)

        # Parse comma-separated food codes
        raw_codes = re.findall(r"\d{8}", response)
        valid_codes = [
            code for code in raw_codes if self._calculator.is_valid_code(code)
        ]
        logger.debug("Selected %d valid food codes from %d raw", len(valid_codes), len(raw_codes))
        return valid_codes

    def _estimate_weight(
        self, food_code: str, image_path: Path
    ) -> float:
        """Step 5: Estimate weight of a food item from the image."""
        food_name = self._calculator.get_food_name(food_code)
        portion_ref = self._calculator.get_portion_reference(food_code)

        portion_text = ""
        if portion_ref:
            portion_text = DietAI24Prompts.PORTION_REFERENCE_TEMPLATE.format(
                portions=portion_ref
            )

        prompt = DietAI24Prompts.WEIGHT_ESTIMATION.format(
            food_name=food_name,
            portion_reference=portion_text,
        )

        response = self._call_vision(prompt, image_path)
        weight = self._parse_weight_response(response)

        if weight is None:
            weight = self._calculator.get_default_weight(food_code)
            logger.warning(
                "Weight estimation failed for %s, using default: %.1fg",
                food_code, weight,
            )

        return weight

    def _parse_weight_response(self, response: str) -> Optional[float]:
        """Parse weight from JSON response string."""
        try:
            # Try direct JSON parse
            cleaned = response.strip()
            # Remove markdown code fences if present
            cleaned = re.sub(r"```json\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            weight = data.get("weight_grams")
            if weight is not None:
                return float(weight)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: extract first number after "weight" keyword
        match = re.search(r"weight[_\s]*(?:grams)?[\":\s]*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Last resort: find any reasonable number
        numbers = re.findall(r"\b(\d{2,4}(?:\.\d+)?)\b", response)
        if numbers:
            return float(numbers[0])

        return None

    def estimate(self, image_path: Path) -> Dict[str, Any]:
        """Run the full estimation pipeline on a single image.

        Returns dict with:
            - food_items: list of {food_code, food_name, weight_grams}
            - predicted: {mass_g, calories, fat_g, carb_g, protein_g}
            - description: food description text
            - ingredients: extracted ingredients list
        """
        raise NotImplementedError("Subclasses must implement estimate()")
