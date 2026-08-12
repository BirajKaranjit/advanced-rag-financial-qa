"""End-to-end pipeline orchestration: ingestion (build) and query-time
retrieval + generation (ask), in the exact retrieval order specified in
the brief (router -> transform -> hybrid search -> RRF -> rerank ->
parent expansion -> compression -> generation).

Three cross-cutting concerns are layered on top of that pipeline, all
documented in ARCHITECTURE.md:
- Observability: every stage is wrapped in `trace_span`, so `trace.spans`
  carries per-stage latency for the retrieval trace panel.
- Prompt-injection defense: every piece of text entering the generation
  context is re-scanned immediately before assembly, wrapped in explicit
  untrusted-data delimiters, and excluded outright if it scores above
  `settings.injection_risk_block_threshold`.
- Numeric groundedness: the final answer is checked against the context
  it was generated from, and a caveat is appended if a significant number
  in the answer is not supported by anything retrieved.
"""

from __future__ import annotations

import logging

from config import settings
from src.generation.groundedness import find_unsupported_numbers, verify_numeric_groundedness
from src.generation.llm_client import LlmClient
from src.generation.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_generation_user_prompt,
    wrap_untrusted_context,
)
from src.generation.tools import NUMERIC_STORE_TOOL_SCHEMA, NumericStoreTool
from src.indexing.document_store import DocumentStore
from src.indexing.sparse_index import SparseIndex
from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunker
from src.ingestion.metadata_extractor import link_figure_captions
from src.ingestion.pdf_parser import PdfParser
from src.retrieval.compressor import ContextualCompressor
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.reranker import Reranker
from src.retrieval.tracer import trace_span
from src.schemas import GenerationResult, QueryType, RetrievalTrace
from src.security.prompt_injection import scan_for_injection

logger = logging.getLogger(__name__)


class RagPipeline:
    """Wires ingestion and retrieval/generation components together.

    `mode="advanced"` runs the full hybrid pipeline described in the
    brief. `mode="basic"` runs dense-only retrieval with no rerank,
    fusion, or numeric tool, so the Streamlit UI can show a side-by-side
    comparison during a demo.
    """

    def __init__(self, mode: str = "advanced") -> None:
        settings.ensure_directories()
        self.mode = mode
        self.document_store = DocumentStore()
        self.vector_store = VectorStore()
        self.sparse_index = SparseIndex()
        self.llm_client = LlmClient()
        self.query_router = QueryRouter()
        self.query_transformer = QueryTransformer(self.llm_client)
        self.hybrid_retriever = HybridRetriever(
            self.vector_store,
            self.sparse_index,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.sparse_top_k,
            rrf_k=settings.rrf_k,
        )
        self.reranker = Reranker()
        self.compressor = ContextualCompressor()
        self.numeric_tool = NumericStoreTool(self.document_store)

    # -- ingestion ----------------------------------------------------------------

    def ingest(self, pdf_path: str) -> None:
        """Parse, chunk, and index a PDF."""
        logger.info("Ingesting %s", pdf_path)
        parser = PdfParser(pdf_path)
        elements = parser.parse()
        link_figure_captions(elements)

        chunker = Chunker(
            target_tokens=settings.narrative_chunk_target_tokens,
            min_tokens=settings.narrative_chunk_min_tokens,
            max_tokens=settings.narrative_chunk_max_tokens,
            overlap_pct=settings.narrative_chunk_overlap_pct,
        )
        parents, children = chunker.chunk_elements(elements)

        self.document_store.save_parent_chunks(parents)
        self.document_store.save_child_chunks(children)
        self.vector_store.add_chunks(children)
        self.sparse_index.build(children)
        logger.info(
            "Ingested %s parents, %s children from %s", len(parents), len(children), pdf_path
        )

    # -- query --------------------------------------------------------------------

    def ask(self, query: str) -> GenerationResult:
        if self.mode == "basic":
            return self._ask_basic(query)
        return self._ask_advanced(query)

    def _assemble_context(
        self, trace: RetrievalTrace, pieces: list[tuple[str, str]]
    ) -> str:
        """Turns a list of (chunk_or_parent_id, text) pairs into the final
        context string: excludes anything over the injection-risk block
        threshold, wraps everything else in untrusted-data delimiters.

        Live-scans the actual text being assembled (post-compression, for
        expanded parents) rather than trusting the ingestion-time score
        alone, since compression can change what content survives into
        context.
        """
        included: list[str] = []
        for source_id, text in pieces:
            result = scan_for_injection(text)
            if settings.enable_prompt_injection_scanning and (
                result.risk_score >= settings.injection_risk_block_threshold
            ):
                trace.excluded_high_risk_chunk_ids.append(source_id)
                logger.warning(
                    "Excluded %s from generation context: injection risk score %.2f "
                    "(patterns=%s) at or above block threshold %.2f",
                    source_id, result.risk_score, result.matched_patterns,
                    settings.injection_risk_block_threshold,
                )
                continue
            included.append(wrap_untrusted_context(source_id, text))
        return "\n\n".join(included)

    def _check_groundedness(self, trace: RetrievalTrace, answer: str, context: str) -> str:
        if not settings.enable_numeric_groundedness_check:
            return answer
        tool_result = trace.numeric_tool_result or ""
        grounded = verify_numeric_groundedness(answer, context + "\n" + tool_result)
        trace.groundedness_passed = grounded
        if not grounded:
            unsupported = find_unsupported_numbers(answer, context + "\n" + tool_result)
            logger.warning("Answer contains unsupported figures: %s", unsupported)
            answer += (
                "\n\n(Note: one or more figures above could not be automatically "
                "verified against the retrieved context and may warrant a manual check.)"
            )
        return answer

    def _ask_basic(self, query: str) -> GenerationResult:
        """Dense-only retrieval, no fusion/rerank/numeric tool -- for
        side-by-side comparison against the advanced pipeline.
        """
        trace = RetrievalTrace(query=query, rewritten_query=query)

        with trace_span("dense_search", trace.spans) as span:
            dense_hits = self.hybrid_retriever.dense_search(query)
            span.metadata["hit_count"] = len(dense_hits)
        trace.dense_hits = dense_hits

        with trace_span("context_assembly", trace.spans) as span:
            pieces = []
            for hit in dense_hits[: settings.final_context_chunks]:
                chunk = self.document_store.get_child_chunk(hit.chunk_id)
                if chunk:
                    hit.chunk = chunk
                    pieces.append((chunk.chunk_id, chunk.text))
            context = self._assemble_context(trace, pieces)
            span.metadata["included_pieces"] = len(pieces)
        trace.compressed_context = context

        with trace_span("generation", trace.spans):
            answer = self.llm_client.complete(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=build_generation_user_prompt(query, context),
            )

        answer = self._check_groundedness(trace, answer, context)
        return GenerationResult(answer=answer, trace=trace)

    def _ask_advanced(self, query: str) -> GenerationResult:
        trace = RetrievalTrace(query=query)

        # 1. Query router
        with trace_span("query_router", trace.spans) as span:
            query_type = self.query_router.classify(query)
            span.metadata["query_type"] = query_type.value
        trace.query_type = query_type

        # 2. Query transformation (rewrite always; HyDE for narrative only)
        with trace_span("query_transform", trace.spans) as span:
            rewritten = self.query_transformer.rewrite(query)
            hyde_doc = self.query_transformer.maybe_hyde(rewritten, query_type)
            span.metadata["hyde_used"] = hyde_doc is not None
        trace.rewritten_query = rewritten
        trace.hyde_document = hyde_doc
        dense_query_text = hyde_doc or rewritten
        metadata_filter = self.query_transformer.extract_metadata_filter(rewritten)

        # 3. Hybrid search
        with trace_span("hybrid_search", trace.spans) as span:
            dense_hits = self.hybrid_retriever.dense_search(dense_query_text, metadata_filter)
            sparse_hits = self.hybrid_retriever.sparse_search(rewritten)
            span.metadata["dense_count"] = len(dense_hits)
            span.metadata["sparse_count"] = len(sparse_hits)
        trace.dense_hits = dense_hits
        trace.sparse_hits = sparse_hits

        # 4. Reciprocal Rank Fusion
        with trace_span("rrf_fusion", trace.spans) as span:
            fused_hits = self.hybrid_retriever.fuse(dense_hits, sparse_hits)
            span.metadata["fused_count"] = len(fused_hits)
        trace.fused_hits = fused_hits

        # 5. Cross-encoder rerank (against the ORIGINAL query, before expansion)
        with trace_span("rerank", trace.spans) as span:
            candidate_ids = [
                h.chunk_id for h in fused_hits[: settings.dense_top_k + settings.sparse_top_k]
            ]
            chunk_texts = {}
            for chunk_id in candidate_ids:
                chunk = self.document_store.get_child_chunk(chunk_id)
                if chunk:
                    chunk_texts[chunk_id] = chunk.text
            reranked_hits = self.reranker.rerank(
                query, fused_hits, chunk_texts, top_k=settings.rerank_top_k
            )
            span.metadata["reranked_count"] = len(reranked_hits)
        trace.reranked_hits = reranked_hits

        # 6-7. Parent expansion + contextual compression (large parents only)
        with trace_span("parent_expansion_and_compression", trace.spans) as span:
            pieces: list[tuple[str, str]] = []
            expanded_parent_ids: list[str] = []
            for hit in reranked_hits[: settings.final_context_chunks]:
                chunk = self.document_store.get_child_chunk(hit.chunk_id)
                if chunk is None:
                    continue
                if chunk.parent_id:
                    parent = self.document_store.get_parent_chunk(chunk.parent_id)
                    if parent:
                        pieces.append((parent.parent_id, self.compressor.compress(query, parent)))
                        expanded_parent_ids.append(parent.parent_id)
                        continue
                pieces.append((chunk.chunk_id, chunk.text))
            span.metadata["expanded_parents"] = len(expanded_parent_ids)
        trace.expanded_parent_ids = expanded_parent_ids

        with trace_span("context_assembly", trace.spans) as span:
            context = self._assemble_context(trace, pieces)
            span.metadata["excluded_count"] = len(trace.excluded_high_risk_chunk_ids)
        trace.compressed_context = context

        # 8. Generation, with numeric-store tool available for
        # numeric_lookup / comparative queries.
        with trace_span("generation", trace.spans) as span:
            if query_type in (QueryType.NUMERIC_LOOKUP, QueryType.COMPARATIVE):
                answer, tool_called, tool_result = self.llm_client.complete_with_tools(
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    user_prompt=build_generation_user_prompt(query, context),
                    tools=[NUMERIC_STORE_TOOL_SCHEMA],
                    tool_executor=self.numeric_tool.execute,
                )
                trace.numeric_tool_called = tool_called
                trace.numeric_tool_result = tool_result
            else:
                answer = self.llm_client.complete(
                    system_prompt=GENERATION_SYSTEM_PROMPT,
                    user_prompt=build_generation_user_prompt(query, context),
                )
            span.metadata["numeric_tool_called"] = trace.numeric_tool_called

        # 9. Numeric groundedness check (secondary safety net)
        with trace_span("groundedness_check", trace.spans) as span:
            answer = self._check_groundedness(trace, answer, context)
            span.metadata["groundedness_passed"] = trace.groundedness_passed

        return GenerationResult(answer=answer, trace=trace)
