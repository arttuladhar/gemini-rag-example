from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the sample RAG pipeline."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", Path.cwd() / "data"))
    persist_directory: Path = Path(
        os.getenv("RAG_VECTOR_CACHE_DIR", Path.cwd() / ".rag_cache")
    )
    sample_question: str = os.getenv(
        "RAG_SAMPLE_QUESTION", "How should I describe the architecture in a quick demo?"
    )

    def ensure_api_key(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in the environment or .env file."
            )


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_api_key()
    settings.persist_directory.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
