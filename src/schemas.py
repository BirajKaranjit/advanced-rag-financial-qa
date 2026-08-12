"""Shared pydantic data models passed between pipeline layers.

Keeping these in one module avoids circular imports between ingestion,
indexing, and retrieval, and gives every inter-layer boundary a validated
schema instead of a raw dict.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Classification applied to every extracted PDF element."""

    TITLE = "title"
    NARRATIVE_TEXT = "narrative_text"
    TABLE = "table"
    FIGURE = "figure"


class ChunkType(str, Enum):
    """Classification applied to indexed chunks (distinct from ElementType:
    a table produces both a parent table chunk and many row-level children).
    """

    NARRATIVE = "narrative"
    TABLE_PARENT = "table_parent"
    TABLE_ROW = "table_row"
    FIGURE = "figure"


class QueryType(str, Enum):
    """Router output tagging how a query should be handled downstream."""

    NUMERIC_LOOKUP = "numeric_lookup"
    NARRATIVE = "narrative"
    COMPARATIVE = "comparative"


# --------------------------------------------------------------------------
# Ingestion-layer models
# --------------------------------------------------------------------------


class TableCell(BaseModel):
    """A single cell in a parsed table, positioned by row/column index."""

    row: int
    col: int
    text: str
    is_header: bool = False
    bbox: list[float] | None = Field(
        default=None,
        description="[x0, top, x1, bottom] in PDF page coordinates, for lineage/auditability.",
    )


class FootnoteLink(BaseModel):
    """A footnote marker resolved to its explanatory text and target."""

    marker: str  # e.g. "(1)"
    footnote_text: str
    target_row: int | None = None
    target_col: int | None = None
    target_description: str = ""  # e.g. "row label 'Net sales', column 'Q3 2022'"


class StructuredTable(BaseModel):
    """A fully parsed table: cell grid, header hierarchy, and footnotes."""

    table_id: str
    title: str
    section_path: str
    page_numbers: list[int]
    header_levels: list[list[str]] = Field(
        default_factory=list,
        description="One list of column labels per header level, top to bottom.",
    )
    cells: list[TableCell]
    footnotes: list[FootnoteLink] = Field(default_factory=list)
    markdown: str = ""
    checksum_passed: bool | None = Field(
        default=None,
        description=(
            "Result of verify_table_checksum: True if any 'Total'/'Subtotal' "
            "row matches the sum of preceding line items, False on discrepancy, "
            "None if no such row was present to check."
        ),
    )

    def cell_at(self, row: int, col: int) -> str | None:
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell.text
        return None


class Figure(BaseModel):
    """An extracted embedded image and any nearby caption text."""

    figure_id: str
    page_number: int
    caption: str = ""
    image_path: str
    section_path: str = ""


class RawElement(BaseModel):
    """An element as classified straight out of the PDF parser, before
    chunking. Tables carry their StructuredTable; everything else carries
    plain text.
    """

    element_type: ElementType
    page_numbers: list[int]
    section_path: str = ""
    text: str = ""
    table: StructuredTable | None = None
    figure: Figure | None = None
    bbox: list[float] | None = Field(
        default=None, description="[x0, top, x1, bottom] for narrative text lines."
    )


# --------------------------------------------------------------------------
# Chunking-layer models
# --------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrievable unit: either narrative text, a table-row sentence,
    a serialized parent table, or a figure description.
    """

    chunk_id: str
    parent_id: str | None = None
    chunk_type: ChunkType
    text: str
    table_title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_path: str = ""
    fiscal_periods: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    injection_risk_score: float = Field(
        default=0.0,
        description=(
            "0.0-1.0 heuristic score from the prompt-injection scanner run at "
            "ingestion time. Surfaced in the retrieval trace; chunks above "
            "settings.injection_risk_block_threshold are excluded from "
            "generation context."
        ),
    )


class ParentChunk(BaseModel):
    """Full-fidelity parent stored in the document store: either a full
    table (markdown + structured JSON) or a full narrative section.
    """

    parent_id: str
    chunk_type: ChunkType
    markdown: str = ""
    structured_table: StructuredTable | None = None
    full_text: str = ""
    page_numbers: list[int] = Field(default_factory=list)
    section_path: str = ""


# --------------------------------------------------------------------------
# Structured numeric store
# --------------------------------------------------------------------------


class NumericFact(BaseModel):
    """One (row, column, value) fact from a table, in long format, for the
    structured numeric store that backs computation queries.
    """

    table_id: str
    row_label: str
    column_label: str
    value: float
    unit: str = "USD_millions"


# --------------------------------------------------------------------------
# Retrieval-layer models
# --------------------------------------------------------------------------


class RetrievalHit(BaseModel):
    """A single scored chunk from any retrieval stage."""

    chunk_id: str
    score: float
    chunk: Chunk | None = None
    source: str = ""  # "dense" | "sparse" | "fused" | "reranked"


class RetrievalTrace(BaseModel):
    """Full intermediate state of one retrieval pass, surfaced in the UI's
    retrieval trace panel.
    """

    query: str
    rewritten_query: str = ""
    hyde_document: str | None = None
    query_type: QueryType = QueryType.NARRATIVE
    dense_hits: list[RetrievalHit] = Field(default_factory=list)
    sparse_hits: list[RetrievalHit] = Field(default_factory=list)
    fused_hits: list[RetrievalHit] = Field(default_factory=list)
    reranked_hits: list[RetrievalHit] = Field(default_factory=list)
    expanded_parent_ids: list[str] = Field(default_factory=list)
    compressed_context: str = ""
    numeric_tool_called: bool = False
    numeric_tool_result: str | None = None
    spans: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-stage timing spans from tracer.trace_span."
    )
    excluded_high_risk_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk ids excluded from context for exceeding the injection-risk threshold.",
    )
    groundedness_passed: bool | None = Field(
        default=None,
        description="Result of verify_numeric_groundedness on the final answer vs. context.",
    )


class GenerationResult(BaseModel):
    """Final answer plus the trace that produced it."""

    answer: str
    trace: RetrievalTrace


# --------------------------------------------------------------------------
# Evaluation-layer models
# --------------------------------------------------------------------------


class EvalQuestionType(str, Enum):
    NARRATIVE = "narrative"
    SINGLE_TABLE_LOOKUP = "single_table_lookup"
    FOOTNOTE_DEPENDENT = "footnote_dependent"
    MULTI_HOP_COMPARATIVE = "multi_hop_comparative"


class EvalExample(BaseModel):
    """One labeled query/answer pair with ground-truth retrieval targets."""

    query_id: str
    question: str
    expected_answer: str
    question_type: EvalQuestionType
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Per-example evaluation outcome."""

    query_id: str
    question_type: EvalQuestionType
    predicted_answer: str
    retrieved_chunk_ids: list[str]
    hit_at_5: bool
    mrr: float
    answer_correct: bool | None = None  # None when not judged
