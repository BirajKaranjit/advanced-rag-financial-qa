"""Tests for dense vector-store ingestion."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.indexing.vector_store import VectorStore
from src.schemas import Chunk, ChunkType


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, chunk_type=ChunkType.NARRATIVE, text=text)


def test_add_chunks_deduplicates_duplicate_ids_before_upsert():
    store = VectorStore.__new__(VectorStore)
    store.embedding_model = MagicMock()
    store.embedding_model.embed_documents.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]]
    )
    store._collection = MagicMock()

    chunks = [
        _chunk("dup", "first version"),
        _chunk("keep", "middle version"),
        _chunk("dup", "second version"),
    ]

    store.add_chunks(chunks)

    store.embedding_model.embed_documents.assert_called_once_with(
        ["second version", "middle version"]
    )
    store._collection.upsert.assert_called_once()
    kwargs = store._collection.upsert.call_args.kwargs
    assert kwargs["ids"] == ["dup", "keep"]
    assert kwargs["documents"] == ["second version", "middle version"]
    assert len(kwargs["embeddings"]) == 2

