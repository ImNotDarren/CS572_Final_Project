# MIA24: A Retrieval-Based Agent for Accurate Automated 24-Hour Recall

CS572 Information Retrieval — Emory University, Spring 2026

**Authors:** Darren Liu, Yuzhu Mao

## Overview

MIA24 (Multimodal Intelligent Agent-based 24-hour dietary assessment tool) introduces a **clarification stage** between initial image understanding and nutrition database retrieval to improve automated dietary assessment. It builds upon the DietAI24 baseline by generating targeted clarification questions about uncertain food aspects (hidden ingredients, cooking methods, sauces, portion sizes), then using the answers to expand and refine the retrieval query before searching the FNDDS nutrition database.

## Results

### OpenAI Results (GPT-4.1-mini, 70 Nutrition5k test samples)

| Metric | DietAI24 (Baseline) | MIA24 (Proposed) | Improvement |
|--------|---------------------|-------------------|-------------|
| Mass (g) | 103.71 | **90.40** | 12.8% |
| Energy (kcal) | 112.29 | **103.25** | 8.1% |
| Fat (g) | 6.87 | **6.30** | 8.3% |
| Carbohydrate (g) | 12.61 | **9.91** | 21.4% |
| Protein (g) | 10.37 | **7.94** | 23.4% |

### Claude Results (Claude Sonnet 4.6, 70 Nutrition5k test samples)

| Metric | DietAI24 (Baseline) | MIA24 (Proposed) | Improvement |
|--------|---------------------|-------------------|-------------|
| Mass (g) | 142.13 | **112.37** | 20.9% |
| Energy (kcal) | 219.91 | **155.30** | 29.4% |
| Fat (g) | 13.07 | **9.58** | 26.7% |
| Carbohydrate (g) | 24.11 | **13.61** | 43.6% |
| Protein (g) | 12.08 | **9.58** | 20.7% |

All values are Mean Absolute Error (MAE) — lower is better. MIA24 outperforms DietAI24 on all 5 metrics across both providers. The clarification stage yields larger relative improvements with Claude (20–44%) compared to OpenAI (8–23%).

### Cross-Provider Comparison

| Metric | OpenAI DietAI24 | OpenAI MIA24 | Claude DietAI24 | Claude MIA24 |
|--------|-----------------|--------------|-----------------|--------------|
| Mass (g) | 103.71 | **90.40** | 142.13 | 112.37 |
| Energy (kcal) | 112.29 | **103.25** | 219.91 | 155.30 |
| Fat (g) | 6.87 | **6.30** | 13.07 | 9.58 |
| Carbohydrate (g) | 12.61 | **9.91** | 24.11 | 13.61 |
| Protein (g) | 10.37 | **7.94** | 12.08 | 9.58 |

GPT-4.1-mini achieves lower absolute MAE, but Claude Sonnet 4.6 shows larger relative improvements from the clarification stage, suggesting MIA24's clarification mechanism provides greater value when the base model has higher initial error.

### Comparison with DietAI24 Paper

The DietAI24 paper ([Yan et al., 2025](https://doi.org/10.1038/s43856-025-01159-0)) reports MAE on a separate 1000-sample Nutrition5k subset using GPT-4 Turbo:

| Metric | Paper (1000 samples, GPT-4 Turbo) | Our DietAI24 (70 samples, GPT-4.1-mini) |
|--------|-----------------------------------|----------------------------------------|
| Mass (g) | 45.1 | 103.71 |
| Energy (kcal) | 68.2 | 112.29 |
| Fat (g) | 4.01 | 6.87 |
| Carbohydrate (g) | 5.98 | 12.61 |
| Protein (g) | 3.88 | 10.37 |

Differences are expected due to: (1) different test subsets (70 vs 1000 samples), (2) different model (GPT-4.1-mini vs GPT-4 Turbo), and (3) different sample selection methodology.

### Detailed Statistics

| Statistic | OpenAI DietAI24 | OpenAI MIA24 | Claude DietAI24 | Claude MIA24 |
|-----------|-----------------|--------------|-----------------|--------------|
| Avg food codes per dish | 1.31 | 2.11 | 1.80 | 2.39 |
| Samples with errors | 3 | 0 | 1 | 2 |
| Total samples | 70 | 70 | 70 | 70 |

### Output Files

Results are saved in provider-specific subdirectories under `results/`:

```
results/
├── open_ai_results/           # GPT-4.1-mini results
│   ├── DietAI24_results_TIMESTAMP.csv
│   ├── MIA24_results_TIMESTAMP.csv
│   └── summary_mae_TIMESTAMP.csv
└── claude_results/            # Claude Sonnet 4.6 results
    ├── DietAI24_results_TIMESTAMP.csv
    ├── MIA24_results_TIMESTAMP.csv
    └── summary_mae_TIMESTAMP.csv
```

CSV columns: `dish_id`, `method`, ground truth (`gt_*`), predictions (`pred_*`), absolute errors (`ae_*`), `num_food_codes`, `food_codes`, `description`, `error`.

## Project Structure

```
MIA24/
├── run_evaluation.py              # Main entry point (--provider openai|claude)
├── .env                           # API keys (OPENAI_API_KEY, CLAUDE_API_KEY, CHROMA_URL)
├── src/
│   ├── config/
│   │   └── settings.py            # Environment config, provider models, paths
│   ├── vector_store/
│   │   └── chroma_client.py       # ChromaDB v2 HTTP client for FNDDS retrieval
│   ├── agents/
│   │   ├── base_agent.py          # Multi-provider vision/chat + RAG utilities
│   │   ├── dietai24.py            # Baseline: single-pass RAG pipeline
│   │   └── mia24.py               # Proposed: clarification + query expansion
│   ├── prompts/
│   │   ├── dietai24_prompts.py    # Baseline prompt templates
│   │   └── mia24_prompts.py       # MIA24-specific prompts
│   ├── nutrition/
│   │   └── calculator.py          # FNDDS nutrient calculation (per-100g scaling)
│   └── data_processing/
│       └── nutrition5k.py         # Nutrition5k dataset loader
├── evaluation/
│   ├── evaluate.py                # Evaluation runner (both methods, multi-provider)
│   └── metrics.py                 # MAE computation
├── data/
│   ├── fndds/                     # FNDDS reference data
│   │   ├── fndds.json             # 5,624 food code -> description mappings
│   │   ├── foodPortions.json      # Portion size references
│   │   └── *.xlsx                 # Per-100g nutrient values
│   └── nutrition5k/               # Nutrition5k dataset (downloaded separately)
│       ├── dish_metadata_cafe1.csv
│       ├── dish_metadata_cafe2.csv
│       ├── ingredients_metadata.csv
│       ├── selected_dish_ids.txt
│       ├── splits/
│       └── images/                # Overhead RGB images
├── results/
│   ├── open_ai_results/           # GPT-4.1-mini evaluation outputs
│   └── claude_results/            # Claude Sonnet 4.6 evaluation outputs
└── docs/
    ├── CS572_Proposal.pdf
    └── s43856-025-01159-0.pdf     # DietAI24 paper
```

## Setup

### 1. Create Conda Environment

```bash
conda create -n mia24 python=3.11 -y
conda activate mia24
pip install openai anthropic chromadb python-dotenv pandas numpy Pillow openpyxl tqdm httpx
```

### 2. Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key
CLAUDE_API_KEY=your-anthropic-api-key
CHROMA_URL=https://your-chromadb-server/chroma
```

- `OPENAI_API_KEY` — Used for text embeddings (`text-embedding-3-large`) and GPT-4.1-mini vision/chat
- `CLAUDE_API_KEY` — Used for Claude Sonnet 4.6 vision/chat
- `CHROMA_URL` — Remote ChromaDB v2 server with pre-populated `fndds` collection

### 3. FNDDS Reference Data

Already included in the project under `data/fndds/`:
- `fndds.json` — 5,624 food code to description mappings
- `foodPortions.json` — Portion size references per food code
- `2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx` — Per-100g nutrient values for all 65 nutrients

### 4. Download Nutrition5k Data

```bash
cd data/nutrition5k

# Download metadata
gsutil -o "GSUtil:parallel_process_count=1" cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv" .
gsutil -o "GSUtil:parallel_process_count=1" cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv" .
gsutil -o "GSUtil:parallel_process_count=1" cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/ingredients_metadata.csv" .

# Download splits
mkdir -p splits
gsutil -o "GSUtil:parallel_process_count=1" cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_test_ids.txt" splits/
gsutil -o "GSUtil:parallel_process_count=1" cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/splits/rgb_train_ids.txt" splits/

# Download overhead RGB images for selected dishes
mkdir -p images
while IFS= read -r dish_id; do
  gsutil -o "GSUtil:parallel_process_count=1" cp \
    "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/${dish_id}/rgb.png" \
    "images/${dish_id}.png" 2>/dev/null
done < selected_dish_ids.txt
```

## Running the Evaluation

```bash
conda activate mia24

# Run both methods with Claude Sonnet 4.6 (default)
python run_evaluation.py

# Run both methods with OpenAI GPT-4.1-mini
python run_evaluation.py --provider openai

# Run only the baseline with Claude
python run_evaluation.py --methods DietAI24 --provider claude

# Run only the proposed method with OpenAI
python run_evaluation.py --methods MIA24 --provider openai

# Quick test with 3 samples
python run_evaluation.py --max-samples 3

# Debug mode
python run_evaluation.py --max-samples 1 --log-level DEBUG
```

## Running the Demo

An interactive web UI that lets you upload a food image and step through the MIA24 pipeline visually.

```bash
conda activate mia24
pip install fastapi uvicorn python-multipart

# From the project root:
python demo/app.py
```

Then open http://localhost:8000 in your browser.

**How it works:**

1. Upload a food image
2. Click through each pipeline stage: Describe → Clarify → Expand → Retrieve → Select → Weight → Nutrition
3. Each step shows the intermediate LLM output so you can see how the clarification stage refines the query

**Requirements:** The same `.env` variables as the evaluation (`OPENAI_API_KEY`, `CHROMA_URL`). The demo uses the OpenAI provider by default.

## Methods

### DietAI24 (Baseline)

Single-pass dish-level RAG pipeline following the [DietAI24 paper](https://doi.org/10.1038/s43856-025-01159-0):

1. **Describe food** — Vision model generates a structured description of the food image
2. **Generate queries** — Create 5 search query variations emphasizing different aspects
3. **RAG search** — Multi-query search against ChromaDB FNDDS collection (5,624 foods)
4. **Select food codes** — Vision model sees the **image + candidate codes** and picks the best-matching FNDDS 8-digit food codes
5. **Estimate weight** — Vision model estimates weight in grams using FNDDS portion size references
6. **Calculate nutrition** — Scale per-100g FNDDS nutrient values to estimated weights

Key design: food code selection is **dish-level** (the model matches visible food to whole FNDDS entries), not ingredient-level decomposition.

### MIA24 (Proposed)

Adds a clarification stage between image understanding and retrieval:

1. **Describe food** — Same as baseline
2. **Generate clarification questions** — Ask about hidden ingredients, cooking method, portion size, toppings (**NEW**)
3. **Simulate user answers** — LLM role-plays as the user, using ground truth ingredient hints (**NEW**)
4. **Query expansion** — Generate expanded description + refined search queries from Q&A (**NEW**)
5. **RAG search** — With clarification-refined queries
6. **Select food codes** — Same vision-based selection as baseline
7-8. Estimate weight, calculate nutrition — Same as baseline

### Multi-Provider Support

The pipeline supports two LLM providers for vision and chat inference:

| Provider | Vision/Chat Model | Embedding Model |
|----------|------------------|-----------------|
| `openai` | GPT-4.1-mini | text-embedding-3-large |
| `claude` | Claude Sonnet 4.6 | text-embedding-3-large |

Embeddings always use OpenAI's `text-embedding-3-large` regardless of provider, since the ChromaDB FNDDS collection was indexed with these embeddings.

### Simulated User

During evaluation, a prompted LLM simulates user responses to clarification questions. It receives:
- The original food description from the image
- Ground truth ingredient information (to give realistic answers)
- The food image itself

This approximates a real user interaction as described in the proposal.

## Evaluation

### Dataset

- **Nutrition5k** ([Thames et al., CVPR 2021](https://github.com/google-research-datasets/Nutrition5k)): Overhead RGB images of plated food with ground truth nutrition
- **Subset**: 70 test-set dishes with available overhead images
- **Ground truth**: Per-dish mass (g), calories (kcal), fat (g), carbohydrate (g), protein (g)

### Metric

**Mean Absolute Error (MAE)** — average absolute difference between predicted and ground truth nutrient values across all test samples:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| openai | 2.30+ | Text embeddings + GPT-4.1-mini vision/chat |
| anthropic | 0.49+ | Claude Sonnet 4.6 vision/chat |
| httpx | 0.28+ | ChromaDB HTTP client |
| python-dotenv | 1.2+ | Load .env configuration |
| pandas | 3.0+ | Data processing |
| numpy | 2.4+ | Numerical operations |
| Pillow | 12+ | Image handling |
| openpyxl | 3.1+ | FNDDS Excel file reading |
| tqdm | 4.67+ | Progress bars |

## Key Configuration

Edit `src/config/settings.py` to change:
- `PROVIDER_MODELS` — Model mappings per provider
- `embedding_model` — Embedding model (default: `text-embedding-3-large`)
- `rag_top_k` — Number of RAG results per query (default: 8)
- `num_query_variations` — Number of search queries generated (default: 5)

Or pass `--provider openai|claude` to `run_evaluation.py` to select the LLM provider at runtime.

## References

- Yan, R. et al. "DietAI24 as a framework for comprehensive nutrition estimation using multimodal large language models." *Communications Medicine*, 5(1):458, 2025. https://doi.org/10.1038/s43856-025-01159-0
- Thames, Q. et al. "Nutrition5k: Towards automatic nutritional understanding of generic food." *CVPR*, 2021.
