"""Query transformation: always rewrite the raw query for clarity; only
run HyDE for narrative queries.

HyDE is skipped for numeric_lookup because a hallucinated hypothetical
number would actively mislead dense search on table data -- the rewritten
query is embedded directly instead.
"""

from __future__ import annotations

import logging
import re

from src.generation.llm_client import LlmClient
from src.schemas import QueryType

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions about a financial filing into a single, "
    "clear, self-contained search query. Preserve every entity, number, "
    "date, and fiscal period mentioned. Do not answer the question. "
    "Return only the rewritten query, nothing else."
)

_HYDE_SYSTEM_PROMPT = (
    "You write a short hypothetical passage (3-5 sentences) that would "
    "plausibly answer the user's question if it appeared in a company's "
    "financial filing. Do not include specific numbers you are not "
    "certain about -- focus on the narrative language and structure a "
    "real answer would use. Return only the passage."
)


class QueryTransformer:
    """Rewrites queries and, for narrative queries, generates a HyDE
    passage to search against instead of the raw query.
    """

    def __init__(self, llm_client: LlmClient | None = None) -> None:
        self.llm_client = llm_client or LlmClient()

    def rewrite(self, query: str) -> str:
        try:
            rewritten = self.llm_client.complete(
                system_prompt=_REWRITE_SYSTEM_PROMPT, user_prompt=query, max_tokens=120
            )
            return rewritten.strip() or query
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed, falling back to raw query: %s", exc)
            return query

    def maybe_hyde(self, rewritten_query: str, query_type: QueryType) -> str | None:
        """Generate a HyDE document only for narrative queries."""
        if query_type != QueryType.NARRATIVE:
            return None
        try:
            return self.llm_client.complete(
                system_prompt=_HYDE_SYSTEM_PROMPT,
                user_prompt=rewritten_query,
                max_tokens=200,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE generation failed, skipping: %s", exc)
            return None

    def extract_metadata_filter(self, rewritten_query: str) -> dict | None:
        """Best-effort extraction of an explicit fiscal-period entity from
        the rewritten query, for use as a Chroma metadata filter.

        Known limitation: `fiscal_period` is stored as a comma-joined
        string on multi-period chunks, and Chroma's `where` clause only
        supports exact/operator matches on metadata fields (substring
        matching is only available via `where_document`, not metadata).
        This filter therefore only fires precisely for chunks tagged with
        a single fiscal period equal to the extracted tag; multi-period
        chunks fall through to unfiltered ranking instead. See
        ARCHITECTURE.md, Limitations.
        """
        match = re.search(
            r"\b(Q[1-4]|9M|H1|FY)[- ](\d{4})\b", rewritten_query, re.IGNORECASE
        )
        if not match:
            return None
        tag = f"{match.group(1).upper()}-{match.group(2)}"
        return {"fiscal_period": tag}
