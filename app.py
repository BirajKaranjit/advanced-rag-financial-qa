"""Streamlit UI: PDF upload, chat over the filing, and a collapsible
retrieval trace panel showing every intermediate retrieval stage.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Financial Filing Q&A", layout="wide")


def _get_pipeline(mode: str) -> RagPipeline:
    key = f"pipeline_{mode}"
    if key not in st.session_state:
        st.session_state[key] = RagPipeline(mode=mode)
    return st.session_state[key]


def _render_trace(trace) -> None:
    with st.expander("Retrieval trace", expanded=False):
        st.markdown(f"**Query type:** `{trace.query_type.value}`")
        st.markdown(f"**Rewritten query:** {trace.rewritten_query}")
        if trace.hyde_document:
            st.markdown("**HyDE passage:**")
            st.code(trace.hyde_document)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Dense hits**")
            for hit in trace.dense_hits[:10]:
                st.text(f"{hit.score:.3f}  {hit.chunk_id}")
        with col2:
            st.markdown("**Sparse (BM25) hits**")
            for hit in trace.sparse_hits[:10]:
                st.text(f"{hit.score:.3f}  {hit.chunk_id}")
        with col3:
            st.markdown("**Fused (RRF) hits**")
            for hit in trace.fused_hits[:10]:
                st.text(f"{hit.score:.3f}  {hit.chunk_id}")

        st.markdown("**Reranked (cross-encoder)**")
        for hit in trace.reranked_hits:
            st.text(f"{hit.score:.3f}  {hit.chunk_id}")

        st.markdown(f"**Expanded parent chunks:** {trace.expanded_parent_ids}")

        if trace.numeric_tool_called:
            st.markdown("**Numeric-store tool result:**")
            st.code(trace.numeric_tool_result or "")

        if trace.excluded_high_risk_chunk_ids:
            st.warning(
                f"Excluded {len(trace.excluded_high_risk_chunk_ids)} chunk(s) from context "
                f"for exceeding the prompt-injection risk threshold: "
                f"{trace.excluded_high_risk_chunk_ids}"
            )

        if trace.groundedness_passed is False:
            st.warning(
                "Numeric groundedness check failed: the answer contains a figure not "
                "found in the retrieved context."
            )

        st.markdown("**Compressed context passed to generation:**")
        st.text_area(
            "context",
            trace.compressed_context,
            height=200,
            label_visibility="collapsed",
            key=f"context_{id(trace)}",
        )

        if trace.spans:
            st.markdown("**Stage timings:**")
            for span in trace.spans:
                extra = {k: v for k, v in span.items() if k not in ("span", "duration_ms")}
                st.text(f"{span['span']}: {span['duration_ms']} ms  {extra if extra else ''}")


def main() -> None:
    st.title("Financial Filing Q&A")

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        pipeline_mode = st.radio(
            "Retrieval mode",
            options=["advanced", "basic"],
            format_func=lambda m: "Advanced Hybrid RAG" if m == "advanced" else "Basic RAG (dense-only)",
        )

        if uploaded_file is not None and st.button("Ingest document"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = Path(tmp.name)
            with st.spinner("Parsing, chunking, and indexing..."):
                pipeline = _get_pipeline(pipeline_mode)
                pipeline.ingest(str(tmp_path))
            st.success("Document ingested.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("trace") is not None:
                _render_trace(message["trace"])

    query = st.chat_input("Ask a question about the filing")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        pipeline = _get_pipeline(pipeline_mode)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = pipeline.ask(query)
            st.markdown(result.answer)
            _render_trace(result.trace)
        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer, "trace": result.trace}
        )


if __name__ == "__main__":
    main()
