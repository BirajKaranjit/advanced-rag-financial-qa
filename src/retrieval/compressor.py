"""Contextual compression of expanded parent chunks.

Applied only to large parent chunks (full tables / full sections); small
child text is passed through unchanged. Extracts the sentences/rows that
actually answer the query before generation, to control token cost and
reduce hallucination surface from irrelevant table rows.
"""

from __future__ import annotations

import re

from src.schemas import ParentChunk

_LARGE_PARENT_CHAR_THRESHOLD = 1200


class ContextualCompressor:
    """Lightweight, embedding-free relevance filter for large parents.

    Uses lexical overlap (shared significant tokens) between the query
    and each row/sentence of a parent chunk rather than a second model
    call, keeping the compression step cheap relative to the reranker.
    """

    def __init__(self, max_output_chars: int = 2000) -> None:
        self.max_output_chars = max_output_chars

    def compress(self, query: str, parent: ParentChunk) -> str:
        source_text = parent.markdown if parent.structured_table else parent.full_text
        if len(source_text) <= _LARGE_PARENT_CHAR_THRESHOLD:
            return source_text

        query_tokens = self._significant_tokens(query)
        units = self._split_units(source_text, is_table=bool(parent.structured_table))

        scored = []
        for unit in units:
            unit_tokens = self._significant_tokens(unit)
            overlap = len(query_tokens & unit_tokens)
            scored.append((overlap, unit))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected: list[str] = []
        total_chars = 0
        for score, unit in scored:
            if score == 0 and selected:
                break
            if total_chars + len(unit) > self.max_output_chars:
                continue
            selected.append(unit)
            total_chars += len(unit)
            if total_chars >= self.max_output_chars:
                break

        return "\n".join(selected) if selected else source_text[: self.max_output_chars]

    @staticmethod
    def _significant_tokens(text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "of", "for", "and", "or", "to", "in", "on",
            "was", "were", "is", "are", "what", "how", "did",
        }
        tokens = re.findall(r"[a-z0-9%$.]+", text.lower())
        return {t for t in tokens if t not in stopwords and len(t) > 1}

    @staticmethod
    def _split_units(text: str, is_table: bool) -> list[str]:
        if is_table:
            return [line for line in text.splitlines() if line.strip()]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
