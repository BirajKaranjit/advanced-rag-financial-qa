"""Function-calling tool definition and executor for the structured
numeric store.

This is the answer to "how do you handle calculations": rather than
asking the LLM to do arithmetic over retrieved text (unreliable on
financial tables), the generation step can call this tool to look up or
compute over the normalized (table_id, row_label, column_label, value,
unit) facts persisted during ingestion.

Implements a rule-based subset -- percent-change, difference, and sum
over the normalized table -- which is sufficient for Core scope. A full
NL-to-pandas agent is documented as Stretch scope.
"""

from __future__ import annotations

from src.exceptions import NumericStoreError
from src.indexing.document_store import DocumentStore
from src.security.prompt_injection import sanitize_tool_argument

NUMERIC_STORE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_numeric_store",
        "description": (
            "Look up or compute over financial figures extracted from tables "
            "in the filing. Use this whenever the question requires an exact "
            "number, a percent change, a difference, or a sum across periods "
            "or line items, rather than reading numbers out of retrieved text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["lookup", "percent_change", "difference", "sum"],
                    "description": "The computation to perform.",
                },
                "row_label": {
                    "type": "string",
                    "description": "Row line item to match, e.g. 'Total net sales'.",
                },
                "column_label": {
                    "type": "string",
                    "description": (
                        "Column/period to match for 'lookup' or 'sum', e.g. "
                        "'Three Months Ended June 25, 2022'."
                    ),
                },
                "column_label_a": {
                    "type": "string",
                    "description": "First period for percent_change/difference.",
                },
                "column_label_b": {
                    "type": "string",
                    "description": "Second (baseline) period for percent_change/difference.",
                },
                "table_id": {
                    "type": "string",
                    "description": "Optional table_id to disambiguate, if known.",
                },
            },
            "required": ["operation", "row_label"],
        },
    },
}


class NumericStoreTool:
    """Executes `query_numeric_store` calls against the document store."""

    def __init__(self, document_store: DocumentStore) -> None:
        self.document_store = document_store

    def execute(self, name: str, args: dict) -> str:
        if name != "query_numeric_store":
            raise NumericStoreError(f"Unknown tool: {name}")

        # Defense-in-depth: the document store already uses parameterized
        # SQL ('?' placeholders in document_store.py), so these arguments
        # cannot alter query structure regardless of content. Sanitizing
        # here additionally guards against a malicious chunk smuggling an
        # oversized or control-character-laden string into a tool call the
        # LLM constructs from retrieved context, before it ever reaches
        # the query layer.
        args = {
            k: sanitize_tool_argument(v) if isinstance(v, str) else v for k, v in args.items()
        }

        operation = args.get("operation")
        row_label = args.get("row_label")
        table_id = args.get("table_id")

        if operation == "lookup":
            return self._lookup(row_label, args.get("column_label"), table_id)
        if operation == "sum":
            return self._sum(row_label, args.get("column_label"), table_id)
        if operation in ("percent_change", "difference"):
            return self._compare(
                operation, row_label, args.get("column_label_a"), args.get("column_label_b"), table_id
            )
        raise NumericStoreError(f"Unsupported operation: {operation}")

    def _lookup(self, row_label: str, column_label: str | None, table_id: str | None) -> str:
        facts = self.document_store.query_numeric_facts(
            table_id=table_id, row_label=row_label, column_label=column_label
        )
        if not facts:
            return f"No matching value found for row='{row_label}', column='{column_label}'."
        results = [f"{f.row_label} / {f.column_label}: {f.value} ({f.unit})" for f in facts[:5]]
        return "; ".join(results)

    def _sum(self, row_label: str, column_label: str | None, table_id: str | None) -> str:
        facts = self.document_store.query_numeric_facts(
            table_id=table_id, row_label=row_label, column_label=column_label
        )
        if not facts:
            return f"No matching values found to sum for row='{row_label}'."
        total = sum(f.value for f in facts)
        unit = facts[0].unit
        return f"Sum of {len(facts)} matching values for '{row_label}': {total:.2f} ({unit})"

    def _compare(
        self,
        operation: str,
        row_label: str,
        column_label_a: str | None,
        column_label_b: str | None,
        table_id: str | None,
    ) -> str:
        facts_a = self.document_store.query_numeric_facts(
            table_id=table_id, row_label=row_label, column_label=column_label_a
        )
        facts_b = self.document_store.query_numeric_facts(
            table_id=table_id, row_label=row_label, column_label=column_label_b
        )
        if not facts_a or not facts_b:
            return (
                f"Could not find both values for row='{row_label}' "
                f"(a='{column_label_a}', b='{column_label_b}')."
            )
        value_a, value_b = facts_a[0].value, facts_b[0].value

        if operation == "difference":
            return f"{value_a} - {value_b} = {value_a - value_b:.2f} ({facts_a[0].unit})"

        if value_b == 0:
            raise NumericStoreError("Cannot compute percent change: baseline value is zero.")
        pct = (value_a - value_b) / abs(value_b) * 100
        return (
            f"Percent change from '{column_label_b}' ({value_b}) to "
            f"'{column_label_a}' ({value_a}): {pct:.2f}%"
        )
