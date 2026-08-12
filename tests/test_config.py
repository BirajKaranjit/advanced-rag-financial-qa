"""Tests for settings environment overrides."""

from __future__ import annotations

from config import Settings


def test_generation_provider_and_models_can_be_overridden_via_env(monkeypatch):
    monkeypatch.setenv("GENERATION_PROVIDER", "GEMINI")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    monkeypatch.setenv("GROQ_MODEL", "llama3-custom")
    monkeypatch.setenv("HF_FALLBACK_MODEL", "meta-llama/custom")

    cfg = Settings()

    assert cfg.generation_provider == "gemini"
    assert cfg.gemini_model == "gemini-custom-model"
    assert cfg.groq_model == "llama3-custom"
    assert cfg.hf_fallback_model == "meta-llama/custom"

