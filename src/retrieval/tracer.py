"""Native in-memory observability: a context manager that captures
per-stage latency (and optional metadata like token counts or hit
counts) as structured JSON, logged to stdout and appended to a trace
list rendered in the Streamlit retrieval-trace panel.

Zero-cost replacement for a hosted tracer (LangSmith/Arize): no external
service, no additional daemon, works identically in Docker and local runs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("pipeline_tracer")


class PipelineSpan:
    """One timed unit of pipeline work."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0
        self.metadata: dict[str, Any] = {}


@contextmanager
def trace_span(
    span_name: str, trace_store: list[dict[str, Any]] | None = None
) -> Generator[PipelineSpan, None, None]:
    """Times a block of code and records it as structured JSON.

    Usage:
        spans: list[dict] = []
        with trace_span("dense_search", spans) as span:
            hits = vector_store.query(...)
            span.metadata["hit_count"] = len(hits)

    Args:
        span_name: label for this stage, e.g. "query_router", "rerank".
        trace_store: optional list to append the completed span's JSON
            payload to (typically `RetrievalTrace.spans`).
    """
    span = PipelineSpan(span_name)
    span.start_time = time.perf_counter()
    try:
        yield span
    finally:
        span.duration_ms = round((time.perf_counter() - span.start_time) * 1000, 2)
        log_payload = {"span": span.name, "duration_ms": span.duration_ms, **span.metadata}
        logger.info("SPAN_COMPLETE: %s", log_payload)
        if trace_store is not None:
            trace_store.append(log_payload)
