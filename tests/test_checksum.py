"""Tests for src.ingestion.validation: deterministic checksum verification
of extracted table totals against their line items.
"""

from __future__ import annotations

from src.ingestion.validation import verify_structured_table_checksum, verify_table_checksum
from src.schemas import TableCell


def test_checksum_passes_when_total_matches_sum():
    row_data = [
        {"row_label": "Products", "value": 100.0},
        {"row_label": "Services", "value": 50.0},
        {"row_label": "Total net sales", "value": 150.0},
    ]
    assert verify_table_checksum(row_data) is True


def test_checksum_fails_on_discrepancy():
    row_data = [
        {"row_label": "Products", "value": 100.0},
        {"row_label": "Services", "value": 50.0},
        {"row_label": "Total net sales", "value": 200.0},  # wrong
    ]
    assert verify_table_checksum(row_data) is False


def test_checksum_passes_when_no_total_row_present():
    row_data = [
        {"row_label": "Products", "value": 100.0},
        {"row_label": "Services", "value": 50.0},
    ]
    assert verify_table_checksum(row_data) is True


def test_checksum_ignores_rows_with_none_value():
    row_data = [
        {"row_label": "Products", "value": 100.0},
        {"row_label": "Footnote reference", "value": None},
        {"row_label": "Total net sales", "value": 100.0},
    ]
    assert verify_table_checksum(row_data) is True


def test_checksum_within_tolerance_for_rounding_noise():
    row_data = [
        {"row_label": "Products", "value": 100.004},
        {"row_label": "Services", "value": 49.998},
        {"row_label": "Total net sales", "value": 150.0},
    ]
    assert verify_table_checksum(row_data, tolerance=1e-2) is True


def _cell(row, col, text, is_header=False):
    return TableCell(row=row, col=col, text=text, is_header=is_header)


def test_structured_checksum_wrapper_passes_for_consistent_table():
    cells = [
        _cell(0, 0, "", is_header=True),
        _cell(0, 1, "FY2022", is_header=True),
        _cell(1, 0, "Products"),
        _cell(1, 1, "100"),
        _cell(2, 0, "Services"),
        _cell(2, 1, "50"),
        _cell(3, 0, "Total net sales"),
        _cell(3, 1, "150"),
    ]
    assert verify_structured_table_checksum([["", "FY2022"]], cells) is True


def test_structured_checksum_wrapper_fails_for_inconsistent_table():
    cells = [
        _cell(0, 0, "", is_header=True),
        _cell(0, 1, "FY2022", is_header=True),
        _cell(1, 0, "Products"),
        _cell(1, 1, "100"),
        _cell(2, 0, "Services"),
        _cell(2, 1, "50"),
        _cell(3, 0, "Total net sales"),
        _cell(3, 1, "999"),
    ]
    assert verify_structured_table_checksum([["", "FY2022"]], cells) is False
