"""PDF extraction and element classification.

Primary extractor is pdfplumber (digitally-generated PDFs). Pages that
yield no extractable text fall back to pytesseract OCR so the pipeline
degrades gracefully on scanned invoices rather than failing outright.

Tables are extracted with cell-level position intact and their header
rows resolved into a multi-level hierarchy rather than a single flattened
string. Footnote markers found in table titles/cells are linked to the
footnote text that follows the table. Every narrative line and table
cell also carries its PDF-page bounding box, so a generated answer can
be traced back to an exact page coordinate (see document_store.py and
ARCHITECTURE.md, "Bounding-box lineage metadata").
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

from config import settings
from src.exceptions import PdfExtractionError, TableParsingError
from src.ingestion.validation import verify_structured_table_checksum
from src.schemas import ElementType, Figure, RawElement, StructuredTable, TableCell

logger = logging.getLogger(__name__)

FOOTNOTE_MARKER_RE = re.compile(r"\((\d{1,2})\)")
CURRENCY_SPACE_RE = re.compile(r"\$\s+(?=\d)")
PERCENT_SPACE_RE = re.compile(r"(?<=\d)\s+%")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class PdfParser:
    """Parses a PDF into classified, structure-preserving elements."""

    def __init__(self, pdf_path: str | Path) -> None:
        self.pdf_path = Path(pdf_path)
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    def parse(self) -> list[RawElement]:
        """Extract and classify every element in the document.

        Returns:
            Ordered list of RawElement, one per detected title, narrative
            block, table, or figure.

        Raises:
            PdfExtractionError: if the file cannot be opened at all.
        """
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                pages_lines = [self._extract_page_lines(page) for page in pdf.pages]
                repeated_lines = self._detect_repeated_headers_footers(pages_lines)

                elements: list[RawElement] = []
                current_section = ""
                for page_index, page in enumerate(pdf.pages):
                    page_number = page_index + 1
                    tables_on_page = self._extract_tables(page, page_number)

                    lines = pages_lines[page_index]
                    if not lines:
                        ocr_text = self._ocr_fallback(page)
                        lines = [(ln, None) for ln in ocr_text.splitlines() if ln.strip()]

                    cleaned_lines = self._clean_lines(lines, repeated_lines)

                    for line_text, bbox in cleaned_lines:
                        if self._looks_like_title(line_text):
                            current_section = line_text
                            elements.append(
                                RawElement(
                                    element_type=ElementType.TITLE,
                                    page_numbers=[page_number],
                                    section_path=current_section,
                                    text=line_text,
                                    bbox=bbox,
                                )
                            )
                        elif line_text.strip():
                            elements.append(
                                RawElement(
                                    element_type=ElementType.NARRATIVE_TEXT,
                                    page_numbers=[page_number],
                                    section_path=current_section,
                                    text=line_text,
                                    bbox=bbox,
                                )
                            )

                    for raw_table in tables_on_page:
                        structured = self._structure_table(
                            raw_table, page_number, current_section
                        )
                        elements.append(
                            RawElement(
                                element_type=ElementType.TABLE,
                                page_numbers=[page_number],
                                section_path=current_section,
                                table=structured,
                                bbox=list(raw_table["bbox"]) if raw_table.get("bbox") else None,
                            )
                        )

                    for figure in self._extract_figures(page_number):
                        elements.append(
                            RawElement(
                                element_type=ElementType.FIGURE,
                                page_numbers=[page_number],
                                section_path=current_section,
                                figure=figure,
                            )
                        )

                return elements
        except FileNotFoundError as exc:
            raise PdfExtractionError(f"PDF not found: {self.pdf_path}") from exc
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise PdfExtractionError(f"Failed to parse {self.pdf_path}: {exc}") from exc

    # -- text extraction ----------------------------------------------------

    def _extract_page_lines(self, page: "pdfplumber.page.Page") -> list[tuple[str, list[float] | None]]:
        """Extracts (line_text, bbox) pairs for a page using pdfplumber's
        line-grouping extractor, which aggregates word-level bounding
        boxes into a per-line box. Falls back to plain `.extract_text()`
        with no bbox if the line-level API is unavailable or raises
        (older pdfplumber versions, unusual page layouts).
        """
        try:
            raw_lines = page.extract_text_lines(strip=True) or []
            return [
                (
                    ln.get("text", ""),
                    [ln["x0"], ln["top"], ln["x1"], ln["bottom"]]
                    if all(k in ln for k in ("x0", "top", "x1", "bottom"))
                    else None,
                )
                for ln in raw_lines
                if ln.get("text", "").strip()
            ]
        except Exception as exc:  # noqa: BLE001
            logger.debug("extract_text_lines unavailable, falling back to plain text: %s", exc)
            text = page.extract_text() or ""
            return [(ln, None) for ln in text.splitlines() if ln.strip()]

    def _ocr_fallback(self, page: "pdfplumber.page.Page") -> str:
        """Route a text-less page through Tesseract OCR."""
        logger.info("Page %s yielded no text; falling back to OCR", page.page_number)
        try:
            image = page.to_image(resolution=300).original
            if not isinstance(image, Image.Image):
                image = Image.open(image)
            return pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR fallback failed on page %s: %s", page.page_number, exc)
            return ""

    def _detect_repeated_headers_footers(
        self, pages_lines: list[list[tuple[str, list[float] | None]]]
    ) -> set[str]:
        """Frequency-based heuristic: any line repeating near-identically
        across more than `header_footer_repetition_threshold` of pages is
        treated as noise (running headers/footers).
        """
        if not pages_lines:
            return set()
        line_counts: Counter[str] = Counter()
        for lines in pages_lines:
            seen_this_page = {text.strip() for text, _ in lines if text.strip()}
            line_counts.update(seen_this_page)

        threshold = settings.header_footer_repetition_threshold * len(pages_lines)
        return {line for line, count in line_counts.items() if count > threshold}

    def _clean_lines(
        self, lines: list[tuple[str, list[float] | None]], repeated_lines: set[str]
    ) -> list[tuple[str, list[float] | None]]:
        cleaned = []
        for text, bbox in lines:
            stripped = text.strip()
            if not stripped or stripped in repeated_lines:
                continue
            stripped = CURRENCY_SPACE_RE.sub("$", stripped)
            stripped = PERCENT_SPACE_RE.sub("%", stripped)
            stripped = MULTI_SPACE_RE.sub(" ", stripped)
            cleaned.append((stripped, bbox))
        return cleaned

    def _looks_like_title(self, line: str) -> bool:
        """Heuristic title detector: short, capitalized/numbered lines
        (e.g. 'Item 2. Management's Discussion...', 'PART I').
        """
        if len(line) > 90:
            return False
        if re.match(r"^(PART|Item|Note)\s+[IVX\d]", line, re.IGNORECASE):
            return True
        words = line.split()
        if words and sum(w[:1].isupper() for w in words if w[:1].isalpha()) / max(
            len(words), 1
        ) > 0.7:
            return True
        return False

    # -- table extraction -----------------------------------------------------

    def _extract_tables(self, page: "pdfplumber.page.Page", page_number: int) -> list[dict]:
        tables = []
        for table_obj in page.find_tables():
            grid = table_obj.extract()
            if not grid or not any(any(cell for cell in row) for row in grid):
                continue
            # `table_obj.cells` is a flat, row-major list of (x0, top, x1,
            # bottom) bboxes, one per grid cell, in the same order as
            # iterating `grid` row by row. It is captured here and zipped
            # against the flattened cell list in _structure_table; if the
            # counts ever mismatch (merged-cell edge cases), bbox is left
            # None for that table rather than raising.
            tables.append(
                {
                    "grid": grid,
                    "bbox": table_obj.bbox,
                    "page": page_number,
                    "cell_bboxes": getattr(table_obj, "cells", None),
                }
            )
        return tables

    def _structure_table(
        self, raw_table: dict, page_number: int, section_path: str
    ) -> StructuredTable:
        """Resolve a raw grid into cells + multi-level header hierarchy +
        linked footnotes, then run checksum validation.
        """
        grid = raw_table["grid"]
        if not grid:
            raise TableParsingError("Empty table grid")

        header_row_count = self._detect_header_row_count(grid)
        header_levels = [
            [str(cell).strip() if cell else "" for cell in grid[i]]
            for i in range(header_row_count)
        ]

        n_grid_cells = sum(len(row) for row in grid)
        cell_bboxes = raw_table.get("cell_bboxes")
        bboxes_usable = isinstance(cell_bboxes, list) and len(cell_bboxes) == n_grid_cells

        cells: list[TableCell] = []
        flat_index = 0
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                text = str(value).strip() if value else ""
                text = CURRENCY_SPACE_RE.sub("$", text)
                text = PERCENT_SPACE_RE.sub("%", text)
                bbox = list(cell_bboxes[flat_index]) if bboxes_usable else None
                cells.append(
                    TableCell(
                        row=r, col=c, text=text, is_header=r < header_row_count, bbox=bbox
                    )
                )
                flat_index += 1

        title = self._infer_table_title(grid, section_path)
        table_id = f"tbl_{page_number}_{abs(hash((title, page_number))) % 100000}"

        footnotes = self._extract_footnotes(grid, cells)
        markdown = self._to_markdown(grid, header_row_count)

        checksum_passed = None
        if settings.enable_table_checksum_validation:
            checksum_passed = verify_structured_table_checksum(header_levels, cells)
            if not checksum_passed:
                logger.warning(
                    "Table checksum mismatch for '%s' (table_id=%s, page=%s): a Total/"
                    "Subtotal row does not equal the sum of its line items -- extraction "
                    "for this table should be treated with lower confidence.",
                    title, table_id, page_number,
                )

        return StructuredTable(
            table_id=table_id,
            title=title,
            section_path=section_path,
            page_numbers=[page_number],
            header_levels=header_levels,
            cells=cells,
            footnotes=footnotes,
            markdown=markdown,
            checksum_passed=checksum_passed,
        )

    def _detect_header_row_count(self, grid: list[list]) -> int:
        """Multi-level headers are common in financial tables (e.g. a
        period-name row above a fiscal-year row). Treat consecutive leading
        rows as headers while they contain no numeric-looking cells.
        """
        count = 0
        for row in grid:
            texts = [str(c).strip() if c else "" for c in row]
            if any(re.search(r"\d", t) and not re.search(r"[A-Za-z]{4,}", t) for t in texts):
                break
            count += 1
            if count >= 3:  # cap: financial statements rarely exceed 3 header levels
                break
        return max(count, 1)

    def _infer_table_title(self, grid: list[list], section_path: str) -> str:
        first_row_text = " ".join(str(c) for c in grid[0] if c).strip()
        return first_row_text[:120] if first_row_text else section_path or "Untitled table"

    def _extract_footnotes(self, grid: list[list], cells: list[TableCell]):
        """Find footnote markers inside cells, then look for a definition
        list in the trailing rows of the same table (common pattern: a
        table's last rows are '(1) Explanation text...').
        """
        from src.schemas import FootnoteLink

        marker_cells = [
            c for c in cells if FOOTNOTE_MARKER_RE.search(c.text) and not c.is_header
        ]
        # Trailing rows that start with a marker are treated as definitions,
        # not data.
        definitions: dict[str, str] = {}
        for row in grid:
            row_text = " ".join(str(c) for c in row if c).strip()
            match = re.match(r"^\((\d{1,2})\)\s*(.+)", row_text)
            if match:
                definitions[match.group(1)] = match.group(2).strip()

        links = []
        for cell in marker_cells:
            for marker_match in FOOTNOTE_MARKER_RE.finditer(cell.text):
                marker = marker_match.group(1)
                if marker in definitions:
                    links.append(
                        FootnoteLink(
                            marker=f"({marker})",
                            footnote_text=definitions[marker],
                            target_row=cell.row,
                            target_col=cell.col,
                            target_description=f"row {cell.row}, col {cell.col}: {cell.text}",
                        )
                    )
        return links

    def _to_markdown(self, grid: list[list], header_row_count: int) -> str:
        lines = []
        for i, row in enumerate(grid):
            cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == header_row_count - 1:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(lines)

    # -- figure extraction ------------------------------------------------------

    def _extract_figures(self, page_number: int) -> list[Figure]:
        """Extract embedded images via PyMuPDF. Returns an empty list
        cleanly if the page has no images -- no fabricated figures section.
        """
        figures: list[Figure] = []
        try:
            doc = fitz.open(self.pdf_path)
            page = doc[page_number - 1]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                out_dir = settings.data_processed_dir / "figures"
                out_dir.mkdir(parents=True, exist_ok=True)
                image_path = out_dir / f"p{page_number}_img{img_index}.{ext}"
                image_path.write_bytes(image_bytes)
                figures.append(
                    Figure(
                        figure_id=f"fig_{page_number}_{img_index}",
                        page_number=page_number,
                        image_path=str(image_path),
                        caption="",  # captioning path is optional; see metadata_extractor
                    )
                )
            doc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Figure extraction skipped for page %s: %s", page_number, exc)
        return figures
