"""ChromaDB-backed dense vector store.

Local, persistent, SQLite-backed under the hood -- matches the project's
"minimal setup, runs locally" requirement without a hosted vector DB.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.exceptions import VectorStoreError
from src.indexing.embeddings import EmbeddingModel
from src.schemas import Chunk, RetrievalHit

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "child_chunks"


class VectorStore:
    """Wraps a persistent Chroma collection for child-chunk dense search."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert a batch of child chunks."""
        if not chunks:
            return
        try:
            unique_chunks = self._dedupe_chunks(chunks)
            texts = [c.text for c in unique_chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            self._collection.upsert(
                ids=[c.chunk_id for c in unique_chunks],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[self._to_metadata(c) for c in unique_chunks],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add {len(chunks)} chunks: {exc}") from exc

    def query(
        self, query_text: str, top_k: int, metadata_filter: dict | None = None
    ) -> list[RetrievalHit]:
        """Dense-search the collection and return ranked hits.

        Args:
            query_text: the (possibly rewritten/HyDE) query to embed.
            top_k: number of results to return.
            metadata_filter: optional Chroma `where` clause, e.g.
                {"fiscal_period": "Q3-2022"}, applied when the query
                rewrite step extracted explicit entities.
        """
        try:
            query_embedding = self.embedding_model.embed_query(query_text)
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=metadata_filter or None,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Dense query failed: {exc}") from exc

        hits: list[RetrievalHit] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for chunk_id, distance in zip(ids, distances):
            # Chroma returns cosine distance; convert to a similarity score.
            score = 1.0 - distance
            hits.append(RetrievalHit(chunk_id=chunk_id, score=score, source="dense"))
        return hits

    @staticmethod
    def _to_metadata(chunk: Chunk) -> dict:
        return {
            "chunk_type": chunk.chunk_type.value,
            "section_path": chunk.section_path or "",
            "table_title": chunk.table_title or "",
            "fiscal_period": ",".join(chunk.fiscal_periods),
            "parent_id": chunk.parent_id or "",
            "injection_risk_score": chunk.injection_risk_score,
        }

    @staticmethod
    def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
        """Preserve the last occurrence of each chunk_id within a batch.

        Chroma rejects duplicate IDs inside a single upsert request. If a
        parser or chunker bug introduces duplicates, we keep the last copy so
        the batch remains writable and the most recent payload wins.
        """
        deduped: dict[str, Chunk] = {}
        duplicate_ids: list[str] = []
        for chunk in chunks:
            if chunk.chunk_id in deduped:
                duplicate_ids.append(chunk.chunk_id)
            deduped[chunk.chunk_id] = chunk
        if duplicate_ids:
            logger.warning(
                "Dropping %s duplicate chunk IDs before vector upsert: %s",
                len(duplicate_ids),
                ", ".join(sorted(set(duplicate_ids))),
            )
        return list(deduped.values())

