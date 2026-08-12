"""Tests for the structured numeric store: fact extraction on ingest and
rule-based computation (lookup, sum, difference, percent_change) via
NumericStoreTool.
"""

from __future__ import annotations

import pytest

from src.exceptions import NumericStoreError
from src.generation.tools import NumericStoreTool
from src.indexing.document_store import DocumentStore
from src.schemas import ChunkType, ParentChunk, StructuredTable, TableCell


@pytest.fixture()
def store(tmp_path) -> DocumentStore:
    return DocumentStore(db_path=tmp_path / "test.db")


def _net_sales_table() -> StructuredTable:
    cells = [
        TableCell(row=0, col=0, text="", is_header=True),
        TableCell(row=0, col=1, text="Three Months Ended June 25, 2022", is_header=True),
        TableCell(row=0, col=2, text="Three Months Ended June 26, 2021", is_header=True),
        TableCell(row=1, col=0, text="Total net sales"),
        TableCell(row=1, col=1, text="$82,959"),
        TableCell(row=1, col=2, text="$81,434"),
        TableCell(row=2, col=0, text="Total operating expenses"),
        TableCell(row=2, col=1, text="$13,415"),
        TableCell(row=2, col=2, text="$11,652"),
    ]
    return StructuredTable(
        table_id="tbl_ops",
        title="Condensed Consolidated Statements of Operations",
        section_path="Item 1",
        page_numbers=[3],
        header_levels=[["", "Three Months Ended June 25, 2022", "Three Months Ended June 26, 2021"]],
        cells=cells,
        markdown="| dummy |",
    )


def _seed(store: DocumentStore) -> None:
    table = _net_sales_table()
    parent = ParentChunk(
        parent_id=table.table_id,
        chunk_type=ChunkType.TABLE_PARENT,
        markdown=table.markdown,
        structured_table=table,
        page_numbers=table.page_numbers,
        section_path=table.section_path,
    )
    store.save_parent_chunks([parent])


def test_ingest_populates_numeric_facts_with_parsed_values(store):
    _seed(store)
    facts = store.query_numeric_facts(table_id="tbl_ops", row_label="Total net sales")

    assert len(facts) == 2
    values = {f.column_label: f.value for f in facts}
    assert values["Three Months Ended June 25, 2022"] == pytest.approx(82959)
    assert values["Three Months Ended June 26, 2021"] == pytest.approx(81434)
    assert all(f.unit == "USD_millions" for f in facts)


def test_lookup_returns_matching_value(store):
    _seed(store)
    tool = NumericStoreTool(store)
    result = tool.execute(
        "query_numeric_store",
        {
            "operation": "lookup",
            "row_label": "Total net sales",
            "column_label": "Three Months Ended June 25, 2022",
        },
    )
    assert "82959.0" in result or "82959" in result


def test_percent_change_matches_manual_calculation(store):
    _seed(store)
    tool = NumericStoreTool(store)
    result = tool.execute(
        "query_numeric_store",
        {
            "operation": "percent_change",
            "row_label": "Total operating expenses",
            "column_label_a": "Three Months Ended June 25, 2022",
            "column_label_b": "Three Months Ended June 26, 2021",
        },
    )
    expected_pct = (13415 - 11652) / 11652 * 100
    assert f"{expected_pct:.2f}%" in result


def test_difference_matches_manual_subtraction(store):
    _seed(store)
    tool = NumericStoreTool(store)
    result = tool.execute(
        "query_numeric_store",
        {
            "operation": "difference",
            "row_label": "Total net sales",
            "column_label_a": "Three Months Ended June 25, 2022",
            "column_label_b": "Three Months Ended June 26, 2021",
        },
    )
    expected_diff = 82959 - 81434
    assert f"{expected_diff:.2f}" in result


def test_sum_aggregates_all_matching_rows(store):
    _seed(store)
    tool = NumericStoreTool(store)
    result = tool.execute(
        "query_numeric_store",
        {"operation": "sum", "row_label": "Total net sales"},
    )
    expected_sum = 82959 + 81434
    assert f"{expected_sum:.2f}" in result


def test_percent_change_with_zero_baseline_raises_numeric_store_error(store):
    table = _net_sales_table()
    table.cells.append(TableCell(row=1, col=3, text="$0"))
    table.header_levels[0].append("Zero Period")
    parent = ParentChunk(
        parent_id=table.table_id,
        chunk_type=ChunkType.TABLE_PARENT,
        markdown=table.markdown,
        structured_table=table,
        page_numbers=table.page_numbers,
        section_path=table.section_path,
    )
    store.save_parent_chunks([parent])

    tool = NumericStoreTool(store)
    with pytest.raises(NumericStoreError):
        tool.execute(
            "query_numeric_store",
            {
                "operation": "percent_change",
                "row_label": "Total net sales",
                "column_label_a": "Three Months Ended June 25, 2022",
                "column_label_b": "Zero Period",
            },
        )


def test_unknown_tool_name_raises_numeric_store_error(store):
    tool = NumericStoreTool(store)
    with pytest.raises(NumericStoreError):
        tool.execute("not_a_real_tool", {})
