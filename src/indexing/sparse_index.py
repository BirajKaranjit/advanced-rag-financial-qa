"""BM25 sparse index over child chunks, persisted to disk and rebuilt
from the document store on demand.

Uses `bm25s` for speed. Sparse search is what makes exact number/token
matches (e.g. a specific dollar figure) reliable in a way dense search
alone is not.
"""

from __future__ import annotations

import logging
import pickle
import re

import bm25s

from config import settings
from src.exceptions import SparseIndexError
from src.schemas import Chunk, RetrievalHit

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$.]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseIndex:
    """BM25 index keyed by chunk_id, persisted alongside the vector store."""

    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build (or rebuild) the index from a full list of child chunks."""
        if not chunks:
            logger.warning("SparseIndex.build called with no chunks")
            return
        try:
            corpus_tokens = [_tokenize(c.text) for c in chunks]
            self._chunk_ids = [c.chunk_id for c in chunks]
            self._retriever = bm25s.BM25()
            self._retriever.index(corpus_tokens)
            self._persist()
        except Exception as exc:  # noqa: BLE001
            raise SparseIndexError(f"Failed to build BM25 index: {exc}") from exc

    def query(self, query_text: str, top_k: int) -> list[RetrievalHit]:
        if self._retriever is None:
            self._load()
        if self._retriever is None:
            raise SparseIndexError("Sparse index not built or persisted yet")

        try:
            tokens = _tokenize(query_text)
            results, scores = self._retriever.retrieve(
                bm25s.tokenize([" ".join(tokens)], show_progress=False),
                k=min(top_k, len(self._chunk_ids)),
            )
        except Exception as exc:  # noqa: BLE001
            raise SparseIndexError(f"BM25 query failed: {exc}") from exc

        hits = []
        for idx, score in zip(results[0], scores[0]):
            hits.append(
                RetrievalHit(
                    chunk_id=self._chunk_ids[idx], score=float(score), source="sparse"
                )
            )
        return hits

    def _persist(self) -> None:
        settings.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.bm25_index_path, "wb") as f:
            pickle.dump({"retriever": self._retriever, "chunk_ids": self._chunk_ids}, f)

    def _load(self) -> None:
        if not settings.bm25_index_path.exists():
            return
        with open(settings.bm25_index_path, "rb") as f:
            payload = pickle.load(f)
        self._retriever = payload["retriever"]
        self._chunk_ids = payload["chunk_ids"]
