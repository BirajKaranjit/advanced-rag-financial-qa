"""Tests for Gemini tool schema conversion."""

from __future__ import annotations

import google.generativeai as genai

from src.generation.llm_client import _openai_tool_to_gemini_declaration
from src.generation.tools import NUMERIC_STORE_TOOL_SCHEMA


def test_openai_tool_schema_converts_to_gemini_function_declaration():
    declaration = _openai_tool_to_gemini_declaration(NUMERIC_STORE_TOOL_SCHEMA)

    assert declaration.name == "query_numeric_store"
    assert declaration.parameters.type == genai.protos.Type.OBJECT
    assert set(declaration.parameters.required) == {"operation", "row_label"}
    assert "operation" in declaration.parameters.properties
    assert declaration.parameters.properties["operation"].type == genai.protos.Type.STRING
    assert "row_label" in declaration.parameters.properties

