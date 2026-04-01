"""MIA24 proposed agent — adds clarification stage to DietAI24.

Pipeline:
  Image -> Describe -> Generate clarification questions
  -> Simulated user answers -> Query expansion
  -> RAG search -> Select food codes -> Estimate weights -> Nutrition

The key innovation is the clarification stage that sits between
initial image understanding and nutrition database retrieval.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.data_processing.nutrition5k import DishSample
from src.prompts import MIA24Prompts

logger = logging.getLogger(__name__)


class MIA24Agent(BaseAgent):
    """Proposed method: interactive clarification before retrieval."""

    def _generate_clarification_questions(self, description: str) -> List[str]:
        """Generate targeted clarification questions about uncertain aspects."""
        prompt = MIA24Prompts.CLARIFICATION_QUESTIONS.format(
            description=description
        )
        response = self._call_chat(prompt)

        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and "?" in cleaned:
                questions.append(cleaned)

        logger.debug("Generated %d clarification questions", len(questions))
        return questions[:3]

    def _simulate_user_response(
        self,
        description: str,
        questions: List[str],
        image_path: Path,
        dish_sample: Optional[DishSample] = None,
    ) -> List[str]:
        """Simulate user answers to clarification questions.

        Uses the ground truth ingredient info as a hint so the simulated
        user can give realistic answers (as described in the proposal).
        """
        ingredients_hint = "Not available"
        if dish_sample:
            ingredients_hint = dish_sample.ingredient_summary

        questions_text = "\n".join(
            f"{i+1}. {q}" for i, q in enumerate(questions)
        )

        prompt = MIA24Prompts.SIMULATED_USER_RESPONSE.format(
            description=description,
            ingredients_hint=ingredients_hint,
            questions=questions_text,
        )

        # Use vision so the simulated user can also reference the image
        response = self._call_vision(prompt, image_path, max_tokens=512)

        answers = []
        for line in response.strip().split("\n"):
            line = line.strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned:
                answers.append(cleaned)

        logger.debug("Simulated %d user answers", len(answers))
        return answers[:len(questions)]

    def _expand_query(
        self,
        description: str,
        questions: List[str],
        answers: List[str],
    ) -> tuple[str, List[str]]:
        """Expand the food description and generate refined queries.

        Returns:
            Tuple of (expanded_description, refined_queries)
        """
        # Format Q&A pairs
        qa_text = "\n".join(
            f"Q{i+1}: {q}\nA{i+1}: {a}"
            for i, (q, a) in enumerate(zip(questions, answers))
        )

        n = self._settings.num_query_variations
        prompt = MIA24Prompts.QUERY_EXPANSION.format(
            description=description,
            qa_pairs=qa_text,
            n=n,
        )
        response = self._call_chat(prompt, max_tokens=1024)

        # Parse expanded description
        expanded_desc = description  # fallback
        desc_match = re.search(
            r"EXPANDED_DESCRIPTION:\s*(.+?)(?=QUERIES:|$)",
            response,
            re.DOTALL,
        )
        if desc_match:
            expanded_desc = desc_match.group(1).strip()

        # Parse refined queries
        queries = []
        queries_match = re.search(r"QUERIES:\s*(.+)", response, re.DOTALL)
        if queries_match:
            for line in queries_match.group(1).strip().split("\n"):
                line = line.strip()
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned:
                    queries.append(cleaned)

        if not queries:
            # Fallback: use expanded description as a single query
            queries = [expanded_desc]

        logger.debug(
            "Query expansion: %d queries from expanded description (%d chars)",
            len(queries), len(expanded_desc),
        )
        return expanded_desc, queries[:n]

    def estimate(
        self,
        image_path: Path,
        dish_sample: Optional[DishSample] = None,
    ) -> Dict[str, Any]:
        """Run full MIA24 pipeline with clarification stage.

        Pipeline steps:
            1. Describe food (vision)
            2. Generate clarification questions (NEW)
            3. Simulate user answers (NEW)
            4. Query expansion with clarification (NEW)
            5. Search ChromaDB with expanded queries
            6. Select food codes (vision + candidates)
            7. Estimate weight per food code (vision)
            8. Calculate nutrition totals

        Args:
            image_path: Path to the food image.
            dish_sample: Optional ground truth sample for simulating user.
        """
        # Step 1: Describe food (same as baseline)
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
                "expanded_description": description,
                "clarification_questions": [],
                "clarification_answers": [],
                "error": "Image analysis failed",
            }

        # Step 2: Generate clarification questions (NEW in MIA24)
        questions = self._generate_clarification_questions(description)

        # Step 3: Simulate user answers (NEW in MIA24)
        answers = self._simulate_user_response(
            description, questions, image_path, dish_sample
        )

        # Step 4: Query expansion with clarification (NEW in MIA24)
        expanded_desc, refined_queries = self._expand_query(
            description, questions, answers
        )

        # Step 5: Search ChromaDB with expanded queries + original description
        all_queries = refined_queries + [expanded_desc]
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
                "expanded_description": expanded_desc,
                "clarification_questions": questions,
                "clarification_answers": answers,
                "error": "No candidates from RAG search",
            }

        # Step 6: Select food codes (vision model sees image + candidates)
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
                "expanded_description": expanded_desc,
                "clarification_questions": questions,
                "clarification_answers": answers,
                "error": "No valid food codes found",
            }

        # Step 7: Estimate weight for each food code
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

        # Step 8: Calculate nutrition
        predicted = self._calculator.calculate_eval_metrics(
            food_items, total_weight
        )

        return {
            "food_items": food_items,
            "predicted": predicted,
            "description": description,
            "expanded_description": expanded_desc,
            "clarification_questions": questions,
            "clarification_answers": answers,
        }
