"""Central configuration for the advanced hybrid RAG system.

All tunables live here so ingestion, indexing, retrieval, and generation
modules share a single source of truth. Values are overridable via
environment variables (see .env.example) through pydantic-settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Paths -----------------------------------------------------------
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    sqlite_path: Path = Field(default=Path("data/processed/document_store.db"))
    chroma_persist_dir: Path = Field(default=Path("data/processed/chroma"))
    bm25_index_path: Path = Field(default=Path("data/processed/bm25_index.pkl"))

    # --- Ingestion / chunking ---------------------------------------------
    narrative_chunk_target_tokens: int = 400
    narrative_chunk_min_tokens: int = 300
    narrative_chunk_max_tokens: int = 500
    narrative_chunk_overlap_pct: float = 0.175
    header_footer_repetition_threshold: float = 0.60  # >60% of pages -> noise

    # --- Embedding / retrieval ---------------------------------------------
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL_NAME")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_NAME")
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 8
    final_context_chunks: int = 5

    hf_hub_token: str = Field(default="", alias="HUGGINGFACE_HUB_TOKEN")

    generation_provider: Literal["groq", "gemini", "hf"] = Field(
        default="groq", alias="GENERATION_PROVIDER"
    )
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama3-70b-8192", alias="GROQ_MODEL")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.6-flash", alias="GEMINI_MODEL")
    hf_inference_token: str = Field(default="", alias="HF_INFERENCE_TOKEN")
    hf_fallback_model: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct", alias="HF_FALLBACK_MODEL"
    )
    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # --- OCR fallback --------------------------------------------------------
    tesseract_cmd: str | None = None  # override if tesseract isn't on PATH

    # --- Security / robustness ------------------------------------------------
    enable_prompt_injection_scanning: bool = True
    injection_risk_block_threshold: float = 0.75  # chunks scoring >= this are excluded from context
    enable_numeric_groundedness_check: bool = True
    enable_table_checksum_validation: bool = True

    # --- Observability -----------------------------------------------------------
    enable_tracing: bool = True

    @field_validator("generation_provider", mode="before")
    @classmethod
    def _normalize_generation_provider(cls, value: str) -> str:
        # Accept uppercase env values (e.g. GENERATION_PROVIDER=GEMINI).
        return value.lower() if isinstance(value, str) else value

    def ensure_directories(self) -> None:
        """Create data directories if they do not already exist."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    def configure_hf_auth(self) -> None:
        """Propagate hf_hub_token into the env vars huggingface_hub reads
        (`HF_TOKEN` in current releases, `HUGGINGFACE_HUB_TOKEN` in older
        ones) so authenticated downloads work regardless of the installed
        huggingface_hub version. No-op if no token was provided -- both
        models used locally (bge-small, ms-marco-MiniLM) are public and
        will still download anonymously, just subject to stricter rate
        limits.
        """
        if not self.hf_hub_token:
            return
        os.environ.setdefault("HF_TOKEN", self.hf_hub_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_hub_token)


settings = Settings()
settings.configure_hf_auth()
