"""Custom exception hierarchy for the RAG pipeline.

Using specific exception types instead of bare Exception lets callers
(especially the Streamlit UI) distinguish recoverable ingestion issues
from retrieval/generation failures and surface useful messages.
"""

from __future__ import annotations


class RagPipelineError(Exception):
    """Base class for all pipeline-specific errors."""


# --- Ingestion -------------------------------------------------------------


class IngestionError(RagPipelineError):
    """Raised when a document cannot be parsed at all."""


class PdfExtractionError(IngestionError):
    """Raised when pdfplumber (and OCR fallback) both fail on a page."""


class TableParsingError(IngestionError):
    """Raised when a detected table cannot be resolved into a structured
    cell grid.
    """


class ChunkingError(IngestionError):
    """Raised when chunk construction fails (e.g. empty parent element)."""


# --- Indexing ----------------------------------------------------------------


class IndexingError(RagPipelineError):
    """Base class for embedding / vector-store / sparse-index failures."""


class EmbeddingError(IndexingError):
    """Raised when the local embedding model fails to encode text."""


class VectorStoreError(IndexingError):
    """Raised on ChromaDB read/write failures."""


class SparseIndexError(IndexingError):
    """Raised on BM25 index build/query failures."""


class DocumentStoreError(IndexingError):
    """Raised on SQLite document-store read/write failures."""


# --- Retrieval -----------------------------------------------------------------


class RetrievalError(RagPipelineError):
    """Base class for retrieval-pipeline failures."""


class QueryRoutingError(RetrievalError):
    """Raised when the query router cannot classify a query."""


class RerankError(RetrievalError):
    """Raised when the cross-encoder reranker fails."""


class NumericStoreError(RetrievalError):
    """Raised when a structured numeric-store computation fails
    (e.g. referenced row/column not found, division by zero).
    """


# --- Generation ------------------------------------------------------------------


class GenerationError(RagPipelineError):
    """Raised when the LLM client fails on both primary and fallback
    providers.
    """
