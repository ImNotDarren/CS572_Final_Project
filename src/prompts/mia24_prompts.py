"""Prompts for the MIA24 proposed method with clarification stage.

Extends DietAI24 with interactive clarification question generation,
simulated user response, and query expansion.
"""

from .dietai24_prompts import DietAI24Prompts


class MIA24Prompts(DietAI24Prompts):
    """Prompt templates for MIA24 — adds clarification and query expansion."""

    CLARIFICATION_QUESTIONS = (
        "You are analyzing a food image for nutritional assessment. Based on "
        "the initial description below, generate targeted clarification questions "
        "about aspects that are uncertain or ambiguous from the image alone.\n\n"
        "Focus on questions about:\n"
        "- Hidden or unclear ingredients (sauces, seasonings, fillings)\n"
        "- Cooking method (fried vs. baked, oil type used)\n"
        "- Portion size details (thickness, number of servings)\n"
        "- Specific food variants (whole wheat vs. white bread, skim vs. whole milk)\n"
        "- Toppings, dressings, or condiments not clearly visible\n\n"
        "Initial food description:\n{description}\n\n"
        "Generate exactly 3 targeted questions that would most improve "
        "nutritional estimation accuracy. Return one question per line, numbered."
    )

    SIMULATED_USER_RESPONSE = (
        "You are simulating a user who just ate the food shown in the image. "
        "A dietary assessment system has asked you clarification questions about "
        "your meal.\n\n"
        "The food in the image has been described as:\n{description}\n\n"
        "The ground truth ingredients for this dish are:\n{ingredients_hint}\n\n"
        "Answer each question based on what you can see in the image and the "
        "ingredient information provided. Give realistic, concise answers as a "
        "real user would — not overly detailed, but helpful.\n\n"
        "Questions:\n{questions}\n\n"
        "Provide one answer per line, numbered to match the questions."
    )

    QUERY_EXPANSION = (
        "You are refining a food database search based on new information "
        "from the user.\n\n"
        "Original food description:\n{description}\n\n"
        "Clarification Q&A:\n{qa_pairs}\n\n"
        "Based on both the original description and the clarification answers, "
        "generate an expanded, more precise food description that incorporates "
        "all the new details. Then generate exactly {n} refined search queries "
        "for a USDA nutrition database.\n\n"
        "Format:\n"
        "EXPANDED_DESCRIPTION: <one paragraph with refined details>\n"
        "QUERIES:\n"
        "1. <query 1>\n"
        "2. <query 2>\n"
        "...\n"
        "{n}. <query {n}>"
    )

    EXPANDED_INGREDIENT_EXTRACT = (
        "Break down the following food into its individual ingredient components "
        "suitable for searching a USDA nutrition database.\n\n"
        "Use the expanded description which includes user-provided clarifications "
        "about hidden ingredients, cooking methods, and portion details.\n\n"
        "Rules:\n"
        "- Maximum {max_items} ingredients\n"
        "- Use common USDA-style food names\n"
        "- Include cooking method when relevant\n"
        "- Separate condiments and sauces as individual items\n"
        "- Prioritize user-clarified details over visual guesses\n\n"
        "Expanded food description:\n{description}\n\n"
        "Return ingredients as a comma-separated list."
    )
