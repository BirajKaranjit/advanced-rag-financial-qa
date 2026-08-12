"""Deterministic arithmetic verification for extracted tables.

Financial tables routinely include a "Total" or "Subtotal" row beneath a
set of line items. If pdfplumber's cell segmentation is slightly off (a
merged cell split wrong, a row boundary misdetected), the extracted line
items will not sum to the extracted total even though the source PDF is
internally consistent -- a strong, cheap signal that a specific table's
extraction should be treated with lower confidence.
"""

from __future__ import annotations

import re

_NUMERIC_STRIP_RE = re.compile(r"[^0-9.\-]")


def _parse_cell_value(text: str) -> float | None:
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    is_negative = cleaned.startswith("(") and cleaned.endswith(")")
    stripped = _NUMERIC_STRIP_RE.sub("", cleaned.replace(",", ""))
    if not stripped or stripped in {"-", "."}:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return -abs(value) if is_negative else value


def verify_table_checksum(row_data: list[dict], tolerance: float = 1e-2) -> bool:
    """Verifies whether an extracted "Total"/"Subtotal" row equals the sum
    of the preceding non-total line items in the same column.

    Args:
        row_data: list of {"row_label": str, "value": float | None} for a
            single column of a table (one call per column to check).
        tolerance: absolute tolerance for the equality check, to absorb
            floating point/rounding noise in the source filing.

    Returns:
        True if the checksum passes, or if no row labeled "total" /
        "subtotal" is present (nothing to check). False on a numeric
        discrepancy between the extracted total and the calculated sum.
    """
    calculated_sum = 0.0
    extracted_total = None

    for row in row_data:
        label = str(row.get("row_label", "")).lower()
        val = row.get("value")
        if val is None or not isinstance(val, (int, float)):
            continue
        if "total" in label or "subtotal" in label:
            extracted_total = val
        else:
            calculated_sum += val

    if extracted_total is not None:
        return abs(calculated_sum - extracted_total) < tolerance
    return True


def verify_structured_table_checksum(header_levels: list[list[str]], cells: list) -> bool:
    """Convenience wrapper: runs verify_table_checksum per data column of a
    StructuredTable's cell grid, returning True only if every column with
    a total row checks out.

    `cells` accepts either TableCell objects or dicts with
    row/col/text/is_header keys, to keep this module decoupled from
    src.schemas (avoids a circular import between ingestion and schemas).
    """
    header_row_count = len(header_levels) or 1

    def _get(cell, attr):
        return getattr(cell, attr) if hasattr(cell, attr) else cell[attr]

    columns = sorted({_get(c, "col") for c in cells})
    for col in columns:
        # Only data rows (row >= header_row_count) are considered: header
        # cells routinely contain digits (fiscal years, "2022") that the
        # loose numeric parser below would otherwise misread as a value
        # and fold into the sum, corrupting the checksum.
        column_cells = sorted(
            (c for c in cells if _get(c, "col") == col and _get(c, "row") >= header_row_count),
            key=lambda c: _get(c, "row"),
        )
        # First data-row cell in this column-major slice acts as the row
        # label lookup: we need the row's *first-column* label paired with
        # this column's value, so re-derive per row.
        row_data = []
        rows_seen = sorted({_get(c, "row") for c in column_cells})
        for row_idx in rows_seen:
            label_cell = next(
                (c for c in cells if _get(c, "row") == row_idx and _get(c, "col") == 0), None
            )
            value_cell = next(
                (c for c in cells if _get(c, "row") == row_idx and _get(c, "col") == col), None
            )
            if label_cell is None or value_cell is None or col == 0:
                continue
            row_data.append(
                {"row_label": _get(label_cell, "text"), "value": _parse_cell_value(_get(value_cell, "text"))}
            )
        if row_data and not verify_table_checksum(row_data):
            return False
    return True
