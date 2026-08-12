"""Metadata extraction helpers: fiscal-period normalization and figure
caption association.

Kept separate from pdf_parser.py because these rules (period regexes,
caption-proximity heuristics) are the parts most likely to need tuning
per document family, without touching the extraction/classification code.
"""

from __future__ import annotations

import re

from src.schemas import Figure, RawElement

# Matches "Three Months Ended June 25, 2022", "Nine Months Ended June 26, 2021", etc.
PERIOD_RE = re.compile(
    r"(Three|Six|Nine|Twelve)\s+Months\s+Ended\s+"
    r"([A-Z][a-z]+\s+\d{1,2},\s*(\d{4}))",
    re.IGNORECASE,
)

QUARTER_MONTHS = {"three": "Q", "six": "H1", "nine": "9M", "twelve": "FY"}


def extract_fiscal_periods(text: str) -> list[str]:
    """Normalize period phrases into short tags like 'Q3-2022', '9M-2022'.

    Best-effort regex approach: for 'Three Months Ended' we cannot derive
    the quarter number from the phrase alone, so we tag it 'Q-<year>' and
    let the caller refine using surrounding context (e.g. a nearby
    'fiscal 2022 third quarter' phrase) if present.
    """
    tags: set[str] = set()
    for match in PERIOD_RE.finditer(text):
        span_word, _full_date, year = match.groups()
        prefix = QUARTER_MONTHS.get(span_word.lower(), span_word.upper())
        tags.add(f"{prefix}-{year}")

    quarter_match = re.search(r"\b(first|second|third|fourth)\s+quarter\b", text, re.IGNORECASE)
    year_match = re.search(r"\b(20\d{2})\b", text)
    if quarter_match and year_match:
        quarter_num = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}[
            quarter_match.group(1).lower()
        ]
        tags = {t.replace("Q-", f"{quarter_num}-") for t in tags} or {
            f"{quarter_num}-{year_match.group(1)}"
        }

    return sorted(tags)


def link_figure_captions(elements: list[RawElement], caption_window: int = 2) -> None:
    """Mutates figures in-place: attaches the nearest narrative-text line
    on the same page as a caption, searched within `caption_window`
    elements before/after the figure in document order.
    """
    for i, element in enumerate(elements):
        if element.element_type.value != "figure" or element.figure is None:
            continue
        figure: Figure = element.figure
        candidates = elements[max(0, i - caption_window) : i + caption_window + 1]
        for candidate in candidates:
            if (
                candidate.element_type.value == "narrative_text"
                and candidate.page_numbers == element.page_numbers
                and re.search(r"^(figure|fig\.|exhibit)\s*\d*", candidate.text, re.IGNORECASE)
            ):
                figure.caption = candidate.text
                break
