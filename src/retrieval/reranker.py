"""Cross-encoder reranking of the RRF-fused candidate list.

Runs before parent expansion, not after: reranking small, precise child
chunks against the original (non-rewritten) query is far more reliable
than reranking large expanded parent tables/sections.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import settings
from src.exceptions import RerankError
from src.schemas import RetrievalHit

logger = logging.getLogger(__name__)


class Reranker:
    """Wraps a local cross-encoder model for precise pairwise scoring."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model_name
        self._model = self._load_model(self.model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> CrossEncoder:
        logger.info("Loading reranker model %s", model_name)
        return CrossEncoder(model_name)

    def rerank(
        self, query: str, hits: list[RetrievalHit], chunk_texts: dict[str, str], top_k: int
    ) -> list[RetrievalHit]:
        """Re-score `hits` against the original query using the
        cross-encoder, and return the top_k.

        Args:
            query: the original (not rewritten/HyDE) user query.
            hits: fused candidate hits.
            chunk_texts: chunk_id -> chunk text, for pairwise scoring.
            top_k: number of hits to keep after reranking.
        """
        candidates = [(h, chunk_texts.get(h.chunk_id, "")) for h in hits if h.chunk_id in chunk_texts]
        if not candidates:
            return []
        try:
            pairs = [(query, text) for _, text in candidates]
            scores = self._model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise RerankError(f"Cross-encoder reranking failed: {exc}") from exc

        reranked = [
            RetrievalHit(chunk_id=hit.chunk_id, score=float(score), source="reranked")
            for (hit, _), score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k]
