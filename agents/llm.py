"""Inicjalizacja modeli Google Gemini dla agentów PydanticAI."""

from __future__ import annotations

from functools import lru_cache

from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from config import (
    MODEL_CALCULATOR,
    MODEL_DOCUMENT,
    MODEL_JUDGE,
    MODEL_LEGAL_RAG,
    MODEL_ROUTER,
    get_settings,
)

# Niższa temperatura — mniej „kreatywnych” dopowiedzeń poza fragmentami RAG.
LEGAL_RAG_MODEL_SETTINGS = GoogleModelSettings(temperature=0.15)
JUDGE_MODEL_SETTINGS = GoogleModelSettings(temperature=0.0)


def _provider() -> GoogleProvider:
    settings = get_settings()
    return GoogleProvider(api_key=settings.google_api_key or "unset")


@lru_cache
def router_model() -> GoogleModel:
    return GoogleModel(_model_name(MODEL_ROUTER), provider=_provider())


@lru_cache
def calculator_model() -> GoogleModel:
    return GoogleModel(_model_name(MODEL_CALCULATOR), provider=_provider())


@lru_cache
def legal_rag_model() -> GoogleModel:
    return GoogleModel(
        _model_name(MODEL_LEGAL_RAG),
        provider=_provider(),
        settings=LEGAL_RAG_MODEL_SETTINGS,
    )


@lru_cache
def document_model() -> GoogleModel:
    return GoogleModel(_model_name(MODEL_DOCUMENT), provider=_provider())


@lru_cache
def judge_model() -> GoogleModel:
    return GoogleModel(
        _model_name(MODEL_JUDGE),
        provider=_provider(),
        settings=JUDGE_MODEL_SETTINGS,
    )


def _model_name(prefixed: str) -> str:
    """Konwertuje 'google:gemini-2.0-flash' → 'gemini-2.0-flash'."""
    return prefixed.split(":", 1)[-1]
