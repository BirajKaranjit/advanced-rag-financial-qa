"""Tests for PDF table ID generation."""

from __future__ import annotations

from src.ingestion.pdf_parser import PdfParser


def test_build_table_id_is_deterministic_for_same_table_content():
    grid = [
        ["", "FY2022"],
        ["Revenue", "100"],
    ]

    table_id_1 = PdfParser._build_table_id("Income Statement", "Item 1", 3, grid)
    table_id_2 = PdfParser._build_table_id("Income Statement", "Item 1", 3, grid)

    assert table_id_1 == table_id_2
    assert table_id_1.startswith("tbl_3_")
    assert len(table_id_1) == len("tbl_3_") + 12


def test_build_table_id_changes_when_table_content_changes():
    base_grid = [
        ["", "FY2022"],
        ["Revenue", "100"],
    ]
    altered_grid = [
        ["", "FY2022"],
        ["Revenue", "101"],
    ]

    base_id = PdfParser._build_table_id("Income Statement", "Item 1", 3, base_grid)
    altered_id = PdfParser._build_table_id("Income Statement", "Item 1", 3, altered_grid)

    assert base_id != altered_id

