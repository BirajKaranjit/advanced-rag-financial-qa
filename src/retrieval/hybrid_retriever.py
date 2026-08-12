"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion (RRF).

Runs dense search (against the HyDE passage or rewritten query, per the
router's decision) and BM25 sparse search against the same rewritten
query, both over child chunks, then fuses the two ranked lists.
"""

from __future__ import annotations

from src.indexing.sparse_index import SparseIndex
from src.indexing.vector_store import VectorStore
from src.schemas import RetrievalHit


class HybridRetriever:
    """Combines dense and sparse search via Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_index: SparseIndex,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k

    def dense_search(
        self, dense_query_text: str, metadata_filter: dict | None = None
    ) -> list[RetrievalHit]:
        return self.vector_store.query(
            dense_query_text, top_k=self.dense_top_k, metadata_filter=metadata_filter
        )

    def sparse_search(self, sparse_query_text: str) -> list[RetrievalHit]:
        return self.sparse_index.query(sparse_query_text, top_k=self.sparse_top_k)

    def fuse(
        self, dense_hits: list[RetrievalHit], sparse_hits: list[RetrievalHit]
    ) -> list[RetrievalHit]:
        """Reciprocal Rank Fusion: score(d) = sum over rankers of
        1 / (rrf_k + rank_in_that_ranker), rank starting at 1.

        RRF is used (rather than a weighted score blend) because dense
        cosine-similarity scores and BM25 scores are not on comparable
        scales; fusing by rank sidesteps that entirely.
        """
        scores: dict[str, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        fused = [
            RetrievalHit(chunk_id=chunk_id, score=score, source="fused")
            for chunk_id, score in scores.items()
        ]
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused
