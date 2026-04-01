"""Project-wide configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


PROVIDER_MODELS = {
    "openai": {"vision": "gpt-4.1-mini", "chat": "gpt-4.1-mini"},
    "claude": {"vision": "claude-sonnet-4-6", "chat": "claude-sonnet-4-6"},
}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    openai_api_key: str = field(
        default_factory=lambda: os.environ["OPENAI_API_KEY"])
    chroma_url: str = field(default_factory=lambda: os.environ["CHROMA_URL"])
    claude_api_key: str = field(
        default_factory=lambda: os.environ["CLAUDE_API_KEY"])

    # Provider: "openai" or "claude"
    provider: str = "claude"

    # Model configuration (defaults to Claude; override via provider)
    vision_model: str = "claude-sonnet-4-6"
    chat_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-large"  # Always OpenAI

    # ChromaDB
    chroma_collection: str = "fndds"
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    # RAG parameters
    rag_top_k: int = 8
    num_query_variations: int = 5
    max_ingredients: int = 10

    # Paths
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data" / "nutrition5k"
    results_dir: Path = _PROJECT_ROOT / "results"

    # FNDDS reference data (local to project)
    fndds_json_path: Path = _PROJECT_ROOT / "data" / "fndds" / "fndds.json"
    food_portions_path: Path = _PROJECT_ROOT / \
        "data" / "fndds" / "foodPortions.json"
    fndds_excel_path: Path = (
        _PROJECT_ROOT / "data" / "fndds"
        / "2019-2020 FNDDS At A Glance - FNDDS Nutrient Values.xlsx"
    )

    @property
    def provider_results_dir(self) -> Path:
        """Return provider-specific results subdirectory."""
        subdir = "claude_results" if self.provider == "claude" else "open_ai_results"
        return self.results_dir / subdir

    @classmethod
    def for_provider(cls, provider: str) -> "Settings":
        """Create Settings configured for a specific provider."""
        models = PROVIDER_MODELS[provider]
        return cls(
            provider=provider,
            vision_model=models["vision"],
            chat_model=models["chat"],
        )
