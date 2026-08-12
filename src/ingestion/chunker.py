"""Structure-aware, parent/child chunking.

Deliberately does not do fixed-size character splitting. Narrative text is
grouped by line into token-bounded windows with a breadcrumb prefix.
Tables are never split by row count blindly: the parent is the full table
(markdown + structured JSON + footnotes), and each child is one flattened
natural-language sentence per data row, with title/section/footnote text
denormalized into the chunk so embedding and BM25 both work well on it.

Every child chunk is also scored for prompt-injection risk here, at
ingestion time, so the score is available in the retrieval trace for
every retrieved hit regardless of whether it goes on to be expanded into
generation context (the enforcement decision -- exclude vs. include -- is
made later, in pipeline.py, against a live re-scan of the actual text
being assembled into context; see src/security/prompt_injection.py).
"""

from __future__ import annotations

import logging
import uuid

from src.exceptions import ChunkingError
from src.ingestion.metadata_extractor import extract_fiscal_periods
from src.schemas import Chunk, ChunkType, ElementType, ParentChunk, RawElement, StructuredTable
from src.security.prompt_injection import scan_for_injection

logger = logging.getLogger(__name__)

# Rough token estimate: ~4 characters per token for English financial prose.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class Chunker:
    """Builds parent chunks (document store) and child chunks (indexed)."""

    def __init__(
        self,
        target_tokens: int = 400,
        min_tokens: int = 300,
        max_tokens: int = 500,
        overlap_pct: float = 0.175,
    ) -> None:
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_pct = overlap_pct

    def chunk_elements(
        self, elements: list[RawElement]
    ) -> tuple[list[ParentChunk], list[Chunk]]:
        """Convert classified elements into parent/child chunk sets.

        Returns:
            (parent_chunks, child_chunks)
        """
        parents: list[ParentChunk] = []
        children: list[Chunk] = []

        narrative_buffer: list[RawElement] = []

        def flush_narrative() -> None:
            if narrative_buffer:
                children.extend(self._chunk_narrative(narrative_buffer))
                narrative_buffer.clear()

        for element in elements:
            if element.element_type == ElementType.NARRATIVE_TEXT:
                narrative_buffer.append(element)
            elif element.element_type == ElementType.TITLE:
                # Titles terminate the current narrative window so the
                # breadcrumb reflects the new section for subsequent text.
                flush_narrative()
            elif element.element_type == ElementType.TABLE:
                flush_narrative()
                if element.table is None:
                    continue
                parent, table_children = self._chunk_table(element.table)
                parents.append(parent)
                children.extend(table_children)
            elif element.element_type == ElementType.FIGURE:
                flush_narrative()
                if element.figure is None:
                    continue
                children.append(self._chunk_figure(element))

        flush_narrative()
        self._tag_injection_risk(children)
        return parents, children

    def _tag_injection_risk(self, children: list[Chunk]) -> None:
        """Scores every child chunk for prompt-injection indicators and
        sets `injection_risk_score` in place.
        """
        for chunk in children:
            result = scan_for_injection(chunk.text)
            chunk.injection_risk_score = result.risk_score
            if result.is_suspicious:
                logger.warning(
                    "Chunk %s flagged for possible prompt injection (score=%.2f, patterns=%s)",
                    chunk.chunk_id, result.risk_score, result.matched_patterns,
                )

    # -- narrative --------------------------------------------------------------

    def _chunk_narrative(self, elements: list[RawElement]) -> list[Chunk]:
        section_path = elements[0].section_path
        units = [(e.text, e.page_numbers, e.bbox) for e in elements if e.text.strip()]
        if not units:
            return []

        chunks: list[Chunk] = []
        window: list[tuple[str, list[int], list[float] | None]] = []
        window_tokens = 0

        def emit(window_units: list[tuple[str, list[int], list[float] | None]]) -> None:
            if not window_units:
                return
            body = "\n".join(u[0] for u in window_units)
            breadcrumb = f"{section_path}\n\n" if section_path else ""
            text = breadcrumb + body
            page_numbers = sorted({p for u in window_units for p in u[1]})
            bbox = self._union_bbox([u[2] for u in window_units if u[2]])
            chunks.append(
                Chunk(
                    chunk_id=f"narr_{uuid.uuid4().hex[:10]}",
                    chunk_type=ChunkType.NARRATIVE,
                    text=text,
                    page_numbers=page_numbers,
                    section_path=section_path,
                    fiscal_periods=extract_fiscal_periods(text),
                    metadata={"bbox": bbox} if bbox else {},
                )
            )

        for unit in units:
            unit_tokens = _estimate_tokens(unit[0])
            if window_tokens + unit_tokens > self.max_tokens and window:
                emit(window)
                overlap_count = max(1, int(len(window) * self.overlap_pct))
                window = window[-overlap_count:]
                window_tokens = sum(_estimate_tokens(u[0]) for u in window)

            window.append(unit)
            window_tokens += unit_tokens

            if window_tokens >= self.target_tokens:
                emit(window)
                overlap_count = max(1, int(len(window) * self.overlap_pct))
                window = window[-overlap_count:]
                window_tokens = sum(_estimate_tokens(u[0]) for u in window)

        emit(window)
        return chunks

    @staticmethod
    def _union_bbox(boxes: list[list[float]]) -> list[float] | None:
        """Combines several [x0, top, x1, bottom] boxes into their
        bounding envelope, for a chunk spanning multiple source lines.
        """
        if not boxes:
            return None
        x0 = min(b[0] for b in boxes)
        top = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        bottom = max(b[3] for b in boxes)
        return [x0, top, x1, bottom]

    # -- tables -----------------------------------------------------------------

    def _chunk_table(self, table: StructuredTable) -> tuple[ParentChunk, list[Chunk]]:
        parent = ParentChunk(
            parent_id=table.table_id,
            chunk_type=ChunkType.TABLE_PARENT,
            markdown=table.markdown,
            structured_table=table,
            page_numbers=table.page_numbers,
            section_path=table.section_path,
        )

        children: list[Chunk] = []
        header_row_count = len(table.header_levels)
        data_rows = sorted({c.row for c in table.cells if c.row >= header_row_count})
        col_headers = self._flatten_headers(table.header_levels)

        for row_idx in data_rows:
            row_cells = sorted(
                (c for c in table.cells if c.row == row_idx), key=lambda c: c.col
            )
            if not row_cells or not row_cells[0].text.strip():
                continue
            row_label = row_cells[0].text
            fragments = []
            for cell in row_cells[1:]:
                if not cell.text.strip():
                    continue
                col_label = col_headers.get(cell.col, f"column {cell.col}")
                fragments.append(f"{col_label} was {cell.text}")

            if not fragments:
                continue

            footnote_text = " ".join(
                f"Note {fn.marker}: {fn.footnote_text}."
                for fn in table.footnotes
                if fn.target_row == row_idx
            )

            sentence = (
                f"In the table '{table.title}' ({table.section_path}), "
                f"for {row_label}: " + "; ".join(fragments) + "."
            )
            if footnote_text:
                sentence += f" {footnote_text}"

            row_bbox = self._union_bbox([c.bbox for c in row_cells if c.bbox])
            metadata = {"row_label": row_label}
            if row_bbox:
                metadata["bbox"] = row_bbox
            if table.checksum_passed is False:
                metadata["checksum_passed"] = False

            children.append(
                Chunk(
                    chunk_id=f"row_{table.table_id}_{row_idx}",
                    parent_id=table.table_id,
                    chunk_type=ChunkType.TABLE_ROW,
                    text=sentence,
                    table_title=table.title,
                    page_numbers=table.page_numbers,
                    section_path=table.section_path,
                    fiscal_periods=extract_fiscal_periods(sentence),
                    metadata=metadata,
                )
            )

        return parent, children

    def _flatten_headers(self, header_levels: list[list[str]]) -> dict[int, str]:
        """Combine multi-level header rows into one label per column,
        e.g. level0='Three Months Ended' + level1='June 25, 2022' ->
        'Three Months Ended June 25, 2022'.
        """
        if not header_levels:
            return {}
        n_cols = max(len(row) for row in header_levels)
        combined: dict[int, str] = {}
        for col in range(n_cols):
            parts = []
            for level in header_levels:
                if col < len(level) and level[col].strip():
                    parts.append(level[col].strip())
            combined[col] = " ".join(parts) if parts else f"column {col}"
        return combined

    # -- figures ------------------------------------------------------------------

    def _chunk_figure(self, element: RawElement) -> Chunk:
        figure = element.figure
        if figure is None:
            raise ChunkingError("Figure element missing figure payload")
        text = figure.caption or f"Figure on page {figure.page_number} (no caption detected)."
        return Chunk(
            chunk_id=figure.figure_id,
            chunk_type=ChunkType.FIGURE,
            text=text,
            page_numbers=[figure.page_number],
            section_path=element.section_path,
            metadata={"image_path": figure.image_path},
        )
