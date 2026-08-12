"""Tests for Reciprocal Rank Fusion in HybridRetriever.fuse()."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.retrieval.hybrid_retriever import HybridRetriever
from src.schemas import RetrievalHit


def _retriever(rrf_k: int = 60) -> HybridRetriever:
    return HybridRetriever(
        vector_store=MagicMock(), sparse_index=MagicMock(), rrf_k=rrf_k
    )


def test_fuse_scores_match_rrf_formula():
    retriever = _retriever(rrf_k=60)
    dense = [RetrievalHit(chunk_id="a", score=0.9, source="dense"),
             RetrievalHit(chunk_id="b", score=0.5, source="dense")]
    sparse = [RetrievalHit(chunk_id="b", score=10.0, source="sparse"),
              RetrievalHit(chunk_id="a", score=8.0, source="sparse")]

    fused = retriever.fuse(dense, sparse)
    fused_scores = {h.chunk_id: h.score for h in fused}

    # a: rank 1 in dense (1/(60+1)), rank 2 in sparse (1/(60+2))
    expected_a = 1 / 61 + 1 / 62
    # b: rank 2 in dense (1/(60+2)), rank 1 in sparse (1/(60+1))
    expected_b = 1 / 62 + 1 / 61

    assert fused_scores["a"] == expected_a
    assert fused_scores["b"] == expected_b
    assert expected_a == expected_b  # symmetric ranks -> equal fused score


def test_fuse_ranks_document_found_by_both_over_single_source_document():
    retriever = _retriever(rrf_k=60)
    dense = [RetrievalHit(chunk_id="a", score=0.9, source="dense"),
              RetrievalHit(chunk_id="c", score=0.8, source="dense")]
    sparse = [RetrievalHit(chunk_id="a", score=9.0, source="sparse")]

    fused = retriever.fuse(dense, sparse)

    assert fused[0].chunk_id == "a"  # found by both rankers -> highest fused score
    assert len(fused) == 2


def test_fuse_handles_disjoint_result_sets():
    retriever = _retriever(rrf_k=60)
    dense = [RetrievalHit(chunk_id="x", score=0.7, source="dense")]
    sparse = [RetrievalHit(chunk_id="y", score=5.0, source="sparse")]

    fused = retriever.fuse(dense, sparse)
    fused_ids = {h.chunk_id for h in fused}

    assert fused_ids == {"x", "y"}
    # Both are rank 1 in their respective ranker -> equal fused score.
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["x"] == scores["y"] == 1 / 61


def test_fuse_empty_inputs_returns_empty_list():
    retriever = _retriever()
    assert retriever.fuse([], []) == []
