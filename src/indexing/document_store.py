"""SQLite document store.

Holds parent_chunks, child_chunks, and structured_tables (foreign-keyed to
parent_chunks), plus a normalized long-format numeric_facts table that
backs the structured numeric store described in section 3.3 of the brief:
computation questions are answered by querying values directly instead of
asking the LLM to do arithmetic over retrieved text.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import settings
from src.exceptions import DocumentStoreError
from src.schemas import Chunk, ChunkType, NumericFact, ParentChunk, StructuredTable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id TEXT PRIMARY KEY,
    chunk_type TEXT NOT NULL,
    markdown TEXT,
    structured_table_json TEXT,
    full_text TEXT,
    page_numbers TEXT,
    section_path TEXT
);

CREATE TABLE IF NOT EXISTS child_chunks (
    chunk_id TEXT PRIMARY KEY,
    parent_id TEXT,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    table_title TEXT,
    page_numbers TEXT,
    section_path TEXT,
    fiscal_periods TEXT,
    metadata_json TEXT,
    injection_risk_score REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (parent_id) REFERENCES parent_chunks (parent_id)
);

CREATE TABLE IF NOT EXISTS structured_tables (
    table_id TEXT PRIMARY KEY,
    title TEXT,
    section_path TEXT,
    page_numbers TEXT,
    header_levels_json TEXT,
    footnotes_json TEXT,
    FOREIGN KEY (table_id) REFERENCES parent_chunks (parent_id)
);

CREATE TABLE IF NOT EXISTS numeric_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id TEXT NOT NULL,
    row_label TEXT NOT NULL,
    column_label TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    FOREIGN KEY (table_id) REFERENCES structured_tables (table_id)
);

CREATE INDEX IF NOT EXISTS idx_numeric_facts_lookup
    ON numeric_facts (table_id, row_label, column_label);
"""

# Matches values like "82,959", "$82,959", "(1,234)", "34.5%", "4.82"
_NUMERIC_RE = None


def _parse_numeric(text: str) -> tuple[float, str] | None:
    """Best-effort parse of a cell's text into (value, unit)."""
    import re

    cleaned = text.strip()
    if not cleaned:
        return None
    is_percent = "%" in cleaned
    is_negative = cleaned.startswith("(") and cleaned.endswith(")")
    stripped = re.sub(r"[^0-9.\-]", "", cleaned.replace(",", ""))
    if not stripped or stripped in {"-", "."}:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    if is_negative:
        value = -abs(value)
    unit = "percent" if is_percent else "USD_millions"
    return value, unit


class DocumentStore:
    """SQLite-backed store for parent chunks, structured tables, and the
    normalized numeric facts table.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- writes -----------------------------------------------------------------

    def save_parent_chunks(self, parents: list[ParentChunk]) -> None:
        try:
            with self._connect() as conn:
                for parent in parents:
                    conn.execute(
                        """INSERT OR REPLACE INTO parent_chunks
                           (parent_id, chunk_type, markdown, structured_table_json,
                            full_text, page_numbers, section_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            parent.parent_id,
                            parent.chunk_type.value,
                            parent.markdown,
                            parent.structured_table.model_dump_json()
                            if parent.structured_table
                            else None,
                            parent.full_text,
                            json.dumps(parent.page_numbers),
                            parent.section_path,
                        ),
                    )
                    if parent.structured_table:
                        self._save_structured_table(conn, parent.structured_table)
        except Exception as exc:  # noqa: BLE001
            raise DocumentStoreError(f"Failed to save parent chunks: {exc}") from exc

    def _save_structured_table(self, conn: sqlite3.Connection, table: StructuredTable) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO structured_tables
               (table_id, title, section_path, page_numbers, header_levels_json, footnotes_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                table.table_id,
                table.title,
                table.section_path,
                json.dumps(table.page_numbers),
                json.dumps(table.header_levels),
                json.dumps([f.model_dump() for f in table.footnotes]),
            ),
        )
        self._save_numeric_facts(conn, table)

    def _save_numeric_facts(self, conn: sqlite3.Connection, table: StructuredTable) -> None:
        header_row_count = len(table.header_levels)
        col_labels = self._flatten_headers(table.header_levels)
        conn.execute("DELETE FROM numeric_facts WHERE table_id = ?", (table.table_id,))

        rows = sorted({c.row for c in table.cells if c.row >= header_row_count})
        for row_idx in rows:
            row_cells = sorted(
                (c for c in table.cells if c.row == row_idx), key=lambda c: c.col
            )
            if not row_cells:
                continue
            row_label = row_cells[0].text
            if not row_label.strip():
                continue
            for cell in row_cells[1:]:
                parsed = _parse_numeric(cell.text)
                if parsed is None:
                    continue
                value, unit = parsed
                col_label = col_labels.get(cell.col, f"column {cell.col}")
                conn.execute(
                    """INSERT INTO numeric_facts (table_id, row_label, column_label, value, unit)
                       VALUES (?, ?, ?, ?, ?)""",
                    (table.table_id, row_label, col_label, value, unit),
                )

    @staticmethod
    def _flatten_headers(header_levels: list[list[str]]) -> dict[int, str]:
        if not header_levels:
            return {}
        n_cols = max(len(row) for row in header_levels)
        combined: dict[int, str] = {}
        for col in range(n_cols):
            parts = [
                level[col].strip()
                for level in header_levels
                if col < len(level) and level[col].strip()
            ]
            combined[col] = " ".join(parts) if parts else f"column {col}"
        return combined

    def save_child_chunks(self, children: list[Chunk]) -> None:
        try:
            with self._connect() as conn:
                for chunk in children:
                    conn.execute(
                        """INSERT OR REPLACE INTO child_chunks
                           (chunk_id, parent_id, chunk_type, text, table_title,
                            page_numbers, section_path, fiscal_periods, metadata_json,
                            injection_risk_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.chunk_id,
                            chunk.parent_id,
                            chunk.chunk_type.value,
                            chunk.text,
                            chunk.table_title,
                            json.dumps(chunk.page_numbers),
                            chunk.section_path,
                            json.dumps(chunk.fiscal_periods),
                            json.dumps(chunk.metadata),
                            chunk.injection_risk_score,
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            raise DocumentStoreError(f"Failed to save child chunks: {exc}") from exc

    # -- reads --------------------------------------------------------------------

    def get_child_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM child_chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(conn_columns=self._child_columns(), row=row)

    def get_all_child_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM child_chunks").fetchall()
        return [self._row_to_chunk(self._child_columns(), r) for r in rows]

    def get_parent_chunk(self, parent_id: str) -> ParentChunk | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM parent_chunks WHERE parent_id = ?", (parent_id,)
            ).fetchone()
        if row is None:
            return None
        cols = ["parent_id", "chunk_type", "markdown", "structured_table_json",
                "full_text", "page_numbers", "section_path"]
        data = dict(zip(cols, row))
        structured_table = (
            StructuredTable.model_validate_json(data["structured_table_json"])
            if data["structured_table_json"]
            else None
        )
        return ParentChunk(
            parent_id=data["parent_id"],
            chunk_type=ChunkType(data["chunk_type"]),
            markdown=data["markdown"] or "",
            structured_table=structured_table,
            full_text=data["full_text"] or "",
            page_numbers=json.loads(data["page_numbers"] or "[]"),
            section_path=data["section_path"] or "",
        )

    def query_numeric_facts(
        self, table_id: str | None = None, row_label: str | None = None,
        column_label: str | None = None,
    ) -> list[NumericFact]:
        """Fuzzy (LIKE-based) lookup into the normalized numeric facts
        table, used by the structured numeric-store generation tool.
        """
        clauses, params = [], []
        if table_id:
            clauses.append("table_id = ?")
            params.append(table_id)
        if row_label:
            clauses.append("row_label LIKE ?")
            params.append(f"%{row_label}%")
        if column_label:
            clauses.append("column_label LIKE ?")
            params.append(f"%{column_label}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT table_id, row_label, column_label, value, unit FROM numeric_facts {where}",
                params,
            ).fetchall()
        return [
            NumericFact(table_id=r[0], row_label=r[1], column_label=r[2], value=r[3], unit=r[4])
            for r in rows
        ]

    @staticmethod
    def _child_columns() -> list[str]:
        return [
            "chunk_id", "parent_id", "chunk_type", "text", "table_title",
            "page_numbers", "section_path", "fiscal_periods", "metadata_json",
            "injection_risk_score",
        ]

    @staticmethod
    def _row_to_chunk(conn_columns: list[str], row: tuple) -> Chunk:
        data = dict(zip(conn_columns, row))
        return Chunk(
            chunk_id=data["chunk_id"],
            parent_id=data["parent_id"],
            chunk_type=ChunkType(data["chunk_type"]),
            text=data["text"],
            table_title=data["table_title"],
            page_numbers=json.loads(data["page_numbers"] or "[]"),
            section_path=data["section_path"] or "",
            fiscal_periods=json.loads(data["fiscal_periods"] or "[]"),
            metadata=json.loads(data["metadata_json"] or "{}"),
            injection_risk_score=data.get("injection_risk_score", 0.0) or 0.0,
        )
