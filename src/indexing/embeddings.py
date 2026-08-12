"""Local dense embedding model wrapper.

Uses BAAI/bge-small-en-v1.5 via sentence-transformers rather than an API
call: stronger MTEB retrieval performance than all-MiniLM-L6-v2 at a
similar CPU-friendly size, and running locally removes one API key from
the reproduction steps.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from src.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# bge models are trained with an instruction prefix on the query side only;
# document/passage text is embedded without a prefix.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self._model = self._load_model(self.model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        logger.info("Loading embedding model %s", model_name)
        return SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passage/document text (no instruction prefix)."""
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()))
        try:
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed {len(texts)} documents: {exc}") from exc

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with the bge instruction prefix."""
        try:
            return self._model.encode(
                _BGE_QUERY_PREFIX + query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc
