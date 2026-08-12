"""Query router: tags each incoming query as numeric_lookup, narrative, or
comparative. A heuristic/regex classifier is sufficient for Core scope; a
learned classifier is a natural upgrade path (see ARCHITECTURE.md).

This decision drives whether HyDE runs (narrative only) and whether the
generation step is nudged toward the structured numeric-store tool
(numeric_lookup / comparative).
"""

from __future__ import annotations

import re

from src.schemas import QueryType

_NUMERIC_KEYWORDS = re.compile(
    r"\b(how much|what (was|were)|total|amount|percentage|percent|"
    r"eps|net sales|revenue|margin|cash flow|income)\b",
    re.IGNORECASE,
)
_COMPARATIVE_KEYWORDS = re.compile(
    r"\b(compare|compared to|versus|vs\.?|grow(th)?|faster|slower|"
    r"increase|decrease|change|difference|higher|lower|year.over.year)\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(r"\$|\d")


class QueryRouter:
    """Regex/heuristic query classifier."""

    def classify(self, query: str) -> QueryType:
        has_comparative = bool(_COMPARATIVE_KEYWORDS.search(query))
        has_numeric = bool(_NUMERIC_KEYWORDS.search(query)) or bool(_HAS_NUMBER.search(query))

        if has_comparative and has_numeric:
            return QueryType.COMPARATIVE
        if has_numeric:
            return QueryType.NUMERIC_LOOKUP
        return QueryType.NARRATIVE
