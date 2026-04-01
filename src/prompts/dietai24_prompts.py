"""Prompts for the DietAI24 baseline pipeline.

Ported from the original DietAI24 paper implementation to match behavior.
Key design: dish-level food code recognition (not ingredient decomposition).
"""


class DietAI24Prompts:
    """All prompt templates used by the DietAI24 baseline."""

    SYSTEM = (
        "You are an American food assistant. You can recognize American foods "
        "from images and estimate the weight of food portions in grams. "
        "If you are unsure, respond with null for food name or weight "
        "as appropriate."
    )

    FOOD_DESCRIPTION = (
        "Please describe the food shown in the image using the following "
        "structure:\n"
        '1. Start with: "The image shows the food category of [food name, '
        'including key visible ingredients or characteristics]."\n'
        "2. If additional information is apparent (such as preparation, "
        "fat/sodium level, main ingredients, serving style, brand/commercial "
        "product, dietary attributes, or cuisine type), add a second sentence: "
        '"Additional details include [list of details, separated by semicolons '
        'if more than one]."\n'
        '3. If something important cannot be determined from the image, state '
        '"not specified subcategory as to [attribute]" as appropriate.\n'
        "4. Do not guess. Only include attributes you can observe in the image.\n"
        "5. If you cannot recognize the food, reply with: "
        "\"I can't help to analyze this image.\" On the next line, briefly "
        "state the reason."
    )

    RETRIEVE_QUERIES = (
        "You are an AI assistant. For the given food description, generate "
        "{n} differently worded phrasings that follow the style below, as "
        "seen in the food vector database. Each should be formatted as:\n"
        '- "The image shows the food category of [food name/description]."\n'
        '- Optionally, add: "Additional details include [attributes]."\n'
        "Vary the focus of each variant (e.g., main ingredient, preparation "
        "method, dietary attribute, cuisine, serving style, commercial brand, "
        "level of processing, or combinations). Use semicolons to separate "
        "multiple details. Only include information that could reasonably be "
        "observed from an image or is common for that food type.\n\n"
        "Provide all {n} descriptions, each on a new line, numbered. Then, "
        "on the next line, include the original food description exactly as "
        "given (do not rephrase it).\n\n"
        "Food to describe:\n{description}"
    )

    CODE_SELECTION = (
        "Given the following context about the food (e.g., candidate codes "
        "and descriptions):\n{candidates}\n\n"
        "If more than one food is present in the image and no single code "
        "can fully describe all foods shown, provide a list of all eight-digit "
        "food codes needed to cover all distinct foods present in the image, "
        "using only the provided context. Assign only one food code per food "
        "type (i.e., do not repeat codes for the same food). Separate multiple "
        "codes with commas (e.g., 12345678, 23456789). If there is no relevant "
        "code, reply with exactly: No appropriate food codes found from the "
        "context information.\n"
        "Only provide the code(s) on the first line. Do not include any "
        "explanation or extra text."
    )

    WEIGHT_ESTIMATION = (
        "The image contains {food_name}. Estimate the weight of the "
        "{food_name} shown in the image as a single number in grams only. "
        "If you cannot estimate, return null. If the container size is not "
        "provided, make a reasonable assumption based on what you see in the "
        "image or use your general knowledge.\n\n"
        "{portion_reference}"
        "Provide your answer as valid JSON, with nothing else before or after. "
        "The JSON format must be:\n"
        '{{\n  "weight_grams": number or null,\n'
        '  "reasoning": "brief reasoning in one sentence"\n}}\n\n'
        "Do not include units such as 'g' in your answer — just the number "
        "or null. Only provide a single value for weight_grams. Do not provide "
        "a range or multiple numbers. If you are unsure or cannot estimate, "
        'set "weight_grams" to null and briefly explain why in "reasoning".'
    )

    PORTION_REFERENCE_TEMPLATE = (
        "Recommend using the reference information below for portion sizes "
        "and their weights if it matches what you see:\n{portions}\n\n"
    )
