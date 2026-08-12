"""Tests for src.generation.groundedness: numeric groundedness checking
between a generated answer and its retrieved context.
"""

from __future__ import annotations

from src.generation.groundedness import find_unsupported_numbers, verify_numeric_groundedness

CONTEXT = (
    "In the table 'Condensed Consolidated Statements of Operations', "
    "Total net sales were $82,959 for the three months ended June 25, 2022, "
    "compared to $81,434 for the three months ended June 26, 2021."
)


def test_grounded_answer_passes():
    answer = "Total net sales for the three months ended June 25, 2022 were $82,959 million."
    assert verify_numeric_groundedness(answer, CONTEXT) is True


def test_hallucinated_number_fails():
    answer = "Total net sales for the three months ended June 25, 2022 were $99,999 million."
    assert verify_numeric_groundedness(answer, CONTEXT) is False


def test_thousands_separator_normalization():
    # "82959" (no comma) should still match "$82,959" in context.
    answer = "Net sales were 82959 for the quarter."
    assert verify_numeric_groundedness(answer, CONTEXT) is True


def test_single_digit_numbers_are_ignored():
    answer = "This is discussed in Note 3 of the filing, item 2."
    # "3" and "2" are single-digit and excluded from the significance check,
    # so this should pass even though neither appears in CONTEXT verbatim.
    assert verify_numeric_groundedness(answer, CONTEXT) is True


def test_percent_figures_are_checked():
    context = "Products gross margin was 34.5% for the quarter."
    grounded_answer = "Products gross margin was 34.5%."
    ungrounded_answer = "Products gross margin was 50.0%."
    assert verify_numeric_groundedness(grounded_answer, context) is True
    assert verify_numeric_groundedness(ungrounded_answer, context) is False


def test_find_unsupported_numbers_returns_the_offending_figures():
    answer = "Net sales were $82,959 million, up from a hallucinated $12,345 million."
    unsupported = find_unsupported_numbers(answer, CONTEXT)
    assert "12345" in unsupported
    assert "82959" not in unsupported


def test_empty_context_flags_any_significant_number():
    answer = "The figure was $500 million."
    assert verify_numeric_groundedness(answer, "") is False
