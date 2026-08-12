"""Tests for structure-aware parent/child chunking."""

from __future__ import annotations

from src.ingestion.chunker import Chunker
from src.schemas import (
    ElementType,
    FootnoteLink,
    RawElement,
    StructuredTable,
    TableCell,
)


def _make_narrative_elements(paragraph_count: int, section: str = "Item 2 MD&A") -> list[RawElement]:
    return [
        RawElement(
            element_type=ElementType.NARRATIVE_TEXT,
            page_numbers=[1],
            section_path=section,
            text="Sentence one. Sentence two provides more financial detail. " * 8,
        )
        for _ in range(paragraph_count)
    ]


def _make_table(with_footnote: bool = False) -> StructuredTable:
    cells = [
        TableCell(row=0, col=0, text="", is_header=True),
        TableCell(row=0, col=1, text="Three Months Ended June 25, 2022", is_header=True),
        TableCell(row=0, col=2, text="Three Months Ended June 26, 2021", is_header=True),
        TableCell(row=1, col=0, text="Total net sales" + (" (1)" if with_footnote else "")),
        TableCell(row=1, col=1, text="$82,959"),
        TableCell(row=1, col=2, text="$81,434"),
    ]
    footnotes = (
        [
            FootnoteLink(
                marker="(1)",
                footnote_text="Includes deferred revenue adjustments.",
                target_row=1,
                target_col=0,
                target_description="row 1, col 0",
            )
        ]
        if with_footnote
        else []
    )
    return StructuredTable(
        table_id="tbl_test_1",
        title="Condensed Consolidated Statements of Operations",
        section_path="Item 1 Financial Statements",
        page_numbers=[3],
        header_levels=[["", "Three Months Ended June 25, 2022", "Three Months Ended June 26, 2021"]],
        cells=cells,
        footnotes=footnotes,
        markdown="| dummy |",
    )


def test_narrative_chunk_has_breadcrumb_prefix():
    chunker = Chunker(target_tokens=50, min_tokens=30, max_tokens=80, overlap_pct=0.2)
    elements = _make_narrative_elements(3)
    _, children = chunker.chunk_elements(elements)

    assert children, "expected at least one narrative chunk"
    assert children[0].text.startswith("Item 2 MD&A")


def test_narrative_chunking_produces_multiple_overlapping_chunks():
    chunker = Chunker(target_tokens=50, min_tokens=30, max_tokens=80, overlap_pct=0.2)
    elements = _make_narrative_elements(6)
    _, children = chunker.chunk_elements(elements)

    assert len(children) > 1, "expected multiple chunks given enough narrative volume"
    # The overlap window carries at least one shared paragraph between
    # consecutive chunks, since each paragraph here is identical text --
    # every chunk after the first should contain the repeated sentence.
    for chunk in children[1:]:
        assert "Sentence one." in chunk.text


def test_table_produces_one_parent_and_one_child_per_data_row():
    chunker = Chunker()
    table = _make_table(with_footnote=False)
    element = RawElement(
        element_type=ElementType.TABLE, page_numbers=[3], section_path="Item 1", table=table
    )
    parents, children = chunker.chunk_elements([element])

    assert len(parents) == 1
    assert parents[0].parent_id == table.table_id
    # One data row -> one child chunk.
    assert len(children) == 1
    row_chunk = children[0]
    assert row_chunk.parent_id == table.table_id
    assert "Total net sales" in row_chunk.text
    assert "$82,959" in row_chunk.text
    assert "$81,434" in row_chunk.text


def test_table_row_chunk_denormalizes_linked_footnote_text():
    chunker = Chunker()
    table = _make_table(with_footnote=True)
    element = RawElement(
        element_type=ElementType.TABLE, page_numbers=[3], section_path="Item 1", table=table
    )
    _, children = chunker.chunk_elements([element])

    assert len(children) == 1
    assert "deferred revenue adjustments" in children[0].text


def test_table_never_split_by_fixed_row_count():
    """Regression guard: a table with many rows still produces exactly one
    parent (never multiple parents for the same table_id).
    """
    cells = [
        TableCell(row=0, col=0, text="", is_header=True),
        TableCell(row=0, col=1, text="FY2022", is_header=True),
    ]
    for r in range(1, 21):
        cells.append(TableCell(row=r, col=0, text=f"Line item {r}"))
        cells.append(TableCell(row=r, col=1, text=f"${r * 100}"))

    table = StructuredTable(
        table_id="tbl_big",
        title="Big table",
        section_path="Item 1",
        page_numbers=[5],
        header_levels=[["", "FY2022"]],
        cells=cells,
        markdown="| dummy |",
    )
    element = RawElement(
        element_type=ElementType.TABLE, page_numbers=[5], section_path="Item 1", table=table
    )
    parents, children = Chunker().chunk_elements([element])

    assert len(parents) == 1
    assert len(children) == 20
    assert all(c.parent_id == "tbl_big" for c in children)
