"""Post-generation groundedness check: confirms that numeric figures in
the generated answer actually appear somewhere in the retrieved context.

This is a cheap, deterministic secondary signal -- not a replacement for
the structured numeric-store tool, which is the primary defense against
LLM arithmetic errors. It catches a different failure mode: the model
citing a number that appears nowhere in what was retrieved at all,
whether from a hallucination or (relevant to the injection threat model
in ARCHITECTURE.md) from following an instruction embedded in a
malicious chunk that told it to state a fabricated figure.
"""

from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _normalize_numbers(text: str) -> set[str]:
    """Extracts numeric tokens and strips thousands separators so
    "82,959" and "82959" are treated as the same figure.
    """
    return {match.replace(",", "") for match in _NUMBER_RE.findall(text)}


def verify_numeric_groundedness(response_text: str, context_text: str) -> bool:
    """Checks whether every "significant" number in the response also
    appears in the retrieved context.

    Single-digit numbers are excluded from the check: they are common in
    ordinary prose ("nine months ended", "Item 2") and would otherwise
    dominate false positives without indicating a grounding problem.

    Args:
        response_text: the final generated answer.
        context_text: the consolidated context passed to generation
            (compressed parent expansions + any numeric-tool result).

    Returns:
        True if all significant numbers in the response are supported by
        the context, False if at least one is not.
    """
    response_numbers = _normalize_numbers(response_text)
    context_numbers = _normalize_numbers(context_text)

    significant = {n for n in response_numbers if len(n.replace(".", "").replace("%", "")) > 1}
    unsupported = significant - context_numbers
    return len(unsupported) == 0


def find_unsupported_numbers(response_text: str, context_text: str) -> list[str]:
    """Same check as verify_numeric_groundedness, but returns the specific
    unsupported figures for logging/trace display instead of a bool.
    """
    response_numbers = _normalize_numbers(response_text)
    context_numbers = _normalize_numbers(context_text)
    significant = {n for n in response_numbers if len(n.replace(".", "").replace("%", "")) > 1}
    return sorted(significant - context_numbers)
