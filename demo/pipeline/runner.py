"""Wraps existing MIA24 agents to expose each pipeline step individually.

Reuses all code from src/ — no duplication of LLM calls, prompts, or logic.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.mia24 import MIA24Agent
from src.config import Settings
from src.nutrition import NutritionCalculator
from src.prompts import DietAI24Prompts
from src.vector_store import ChromaRetriever

logger = logging.getLogger(__name__)


class DemoRunner:
    """Exposes each MIA24 pipeline step as an independent method."""

    def __init__(self) -> None:
        self.settings = Settings.for_provider("openai")
        logger.info("Connecting to ChromaDB...")
        self.retriever = ChromaRetriever(self.settings)
        logger.info("Loading FNDDS nutrition data...")
        self.calculator = NutritionCalculator(self.settings)
        self.agent = MIA24Agent(
            self.settings, self.retriever, self.calculator
        )
        logger.info("DemoRunner ready")

    # ── Step 1: Food Description (Vision) ──────────────────────

    def describe_food(self, image_path: Path) -> str:
        """Vision model describes the food in the image."""
        return self.agent._describe_food(image_path)

    # ── Step 2: Clarification Questions (Chat) ─────────────────

    def generate_clarification(self, description: str) -> List[str]:
        """Generate 3 targeted clarification questions."""
        return self.agent._generate_clarification_questions(description)

    # ── Step 2b: Suggest Answers (Vision) ─────────────────────

    def suggest_answers(
        self,
        description: str,
        questions: List[str],
        image_path: Path,
    ) -> List[List[str]]:
        """Vision model suggests answers to clarification questions."""
        q_text = "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(questions)
        )
        prompt = (
            "You are looking at a food image. Based on what you can "
            "observe, suggest 2-3 short answer options for each "
            "clarification question.\n\n"
            f"Food description: {description}\n\n"
            f"Questions:\n{q_text}\n\n"
            "Format your response exactly as:\n"
            "Q1: option1 | option2 | option3\n"
            "Q2: option1 | option2 | option3\n"
            "Q3: option1 | option2 | option3\n\n"
            "Keep each option short (under 8 words). Be definitive and "
            "certain. Do NOT use hedging words like 'likely', 'possibly', "
            "'probably', 'appears to be', or 'seems like'. State answers "
            "as facts based on what you see."
        )
        raw = self.agent._call_vision(prompt, image_path)

        suggestions: List[List[str]] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("Q") and ":" in line:
                _, options = line.split(":", 1)
                opts = [
                    o.strip() for o in options.split("|") if o.strip()
                ]
                suggestions.append(opts)

        while len(suggestions) < len(questions):
            suggestions.append([])

        return suggestions[: len(questions)]

    # ── Step 3: Query Expansion + Ingredient Decomposition ─────

    def expand_query(
        self,
        description: str,
        questions: List[str],
        answers: List[str],
    ) -> Tuple[str, List[str]]:
        """Expand description and decompose into ingredient-level queries."""
        expanded_desc, _ = self.agent._expand_query(
            description, questions, answers
        )

        prompt = (
            "Given this food description, decompose it into individual "
            "ingredients. Each ingredient should map to its own USDA "
            "FNDDS food code.\n\n"
            f"Food description: {expanded_desc}\n\n"
            "Examples:\n"
            '- "pineapple fried rice" -> pineapple chunks, green peas, '
            "carrots diced, fried rice\n"
            '- "chicken caesar salad" -> grilled chicken breast, romaine '
            "lettuce, parmesan cheese, caesar dressing, croutons\n\n"
            "List each ingredient as a precise USDA FNDDS search query "
            "(one per line, numbered):\n"
        )
        raw = self.agent._call_chat(prompt, max_tokens=512)

        queries: List[str] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            match = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if match:
                ingredient = match.group(1).strip().rstrip(".")
                if ingredient:
                    queries.append(ingredient)

        if not queries:
            queries = [expanded_desc]

        return expanded_desc, queries

    # ── Step 4: RAG Retrieval (ChromaDB) ───────────────────────

    def retrieve(
        self, queries: List[str], expanded_desc: str
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB per ingredient, return grouped results."""
        groups: List[Dict[str, Any]] = []
        for query in queries:
            candidates = self._search_single_query(query, top_k=5)
            for c in candidates:
                c["food_name"] = self.calculator.get_food_name(
                    c["food_code"]
                )
            groups.append({
                "ingredient": query,
                "candidates": candidates,
            })
        return groups

    def _search_single_query(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for a single query."""
        results: List[Dict[str, Any]] = []
        try:
            embeddings = self.retriever._embed([query])
            raw = self.retriever._client.query(
                collection_id=self.retriever._collection_id,
                query_embeddings=embeddings,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            if raw["ids"] and raw["ids"][0]:
                for i, doc_id in enumerate(raw["ids"][0]):
                    meta = (
                        raw["metadatas"][0][i]
                        if raw.get("metadatas")
                        else {}
                    )
                    doc = (
                        raw["documents"][0][i]
                        if raw.get("documents")
                        else ""
                    )
                    code = str(meta.get("Food code", doc_id))
                    dist = (
                        raw["distances"][0][i]
                        if raw.get("distances")
                        else None
                    )
                    results.append({
                        "food_code": code,
                        "description": doc,
                        "distance": (
                            round(dist, 4) if dist else None
                        ),
                    })
        except Exception as exc:
            logger.warning(
                "Search failed for query '%s': %s", query[:50], exc
            )
        results.sort(key=lambda r: r.get("distance") or 999)
        return results

    # ── Step 5: Food Code Selection (Vision) ───────────────────

    def select_food_codes(
        self, candidates: List[Dict], image_path: Path
    ) -> List[Dict[str, str]]:
        """Vision model selects best FNDDS codes from candidates."""
        codes = self.agent._select_food_codes(candidates, image_path)
        return [
            {
                "food_code": code,
                "food_name": self.calculator.get_food_name(code),
            }
            for code in codes
        ]

    # ── Step 6: Weight Estimation (Vision) ─────────────────────

    def estimate_weights(
        self, selected_codes: List[Dict], image_path: Path
    ) -> List[Dict[str, Any]]:
        """Estimate weight with natural display units."""
        items: List[Dict[str, Any]] = []

        for item in selected_codes:
            code = item["food_code"]
            food_name = self.calculator.get_food_name(code)
            portion_ref = self.calculator.get_portion_reference(code)

            portion_text = ""
            if portion_ref:
                portion_text = (
                    DietAI24Prompts.PORTION_REFERENCE_TEMPLATE.format(
                        portions=portion_ref
                    )
                )

            prompt = (
                f"The image contains {food_name}. Estimate the weight "
                f"of the {food_name} shown in the image.\n\n"
                f"{portion_text}"
                "Provide your answer as valid JSON with nothing else. "
                "Use the most natural unit for this food type:\n"
                "- Meat, fish, vegetables, fruit: grams (g)\n"
                "- Rice, grains, pasta: cups\n"
                "- Sauces, oils, dressings: tablespoons (tbsp)\n"
                "- Seasonings, spices: teaspoons (tsp)\n"
                "- Beverages: fluid ounces (fl oz) or cups\n\n"
                "JSON format:\n"
                "{\n"
                '  "weight_grams": number or null,\n'
                '  "display_weight": "amount with natural unit '
                '(e.g. 1.5 cups, 2 tbsp, 150 g)",\n'
                '  "reasoning": "brief reasoning in one sentence"\n'
                "}\n\n"
                "Do not include units in weight_grams — just the "
                "number in grams or null."
            )

            raw = self.agent._call_vision(prompt, image_path)
            weight = self.agent._parse_weight_response(raw)
            reasoning = self._extract_reasoning(raw)
            display_weight = self._extract_display_weight(raw, weight)

            if weight is None:
                weight = self.calculator.get_default_weight(code)

            items.append({
                "food_code": code,
                "food_name": food_name,
                "weight_grams": round(weight, 1),
                "display_weight": display_weight,
                "reasoning": reasoning,
            })

        return items

    @staticmethod
    def _extract_reasoning(raw_response: str) -> str:
        """Pull the reasoning field out of a weight-estimation response."""
        try:
            cleaned = raw_response.strip()
            cleaned = re.sub(r"```json\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            data = json.loads(cleaned.strip())
            return data.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            return raw_response[:200]

    @staticmethod
    def _extract_display_weight(
        raw_response: str, fallback_grams: Optional[float]
    ) -> str:
        """Extract display_weight from weight-estimation response."""
        try:
            cleaned = raw_response.strip()
            cleaned = re.sub(r"```json\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)
            data = json.loads(cleaned.strip())
            dw = data.get("display_weight", "")
            if dw:
                return dw
        except (json.JSONDecodeError, ValueError):
            pass
        if fallback_grams is not None:
            return f"{round(fallback_grams, 1)} g"
        return "N/A"

    # ── Step 7: Nutrition Calculation ──────────────────────────

    def calculate_nutrition(
        self, food_items: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate final nutrition values from food items."""
        total_weight = sum(it["weight_grams"] for it in food_items)
        result = self.calculator.calculate_eval_metrics(
            food_items, total_weight
        )
        return {k: round(v, 1) for k, v in result.items()}
