"""Konfiguracja aplikacji KodeksPracy AI — MetalTech Sp. z o.o."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

MODEL_ROUTER = "google:gemini-3.1-flash-lite"
MODEL_CALCULATOR = "google:gemini-3.1-flash-lite"
MODEL_LEGAL_RAG = "google:gemini-3.1-flash-lite"
MODEL_DOCUMENT = "google:gemini-3.1-flash-lite"

QDRANT_COLLECTION = "kodeks_pracy_metaltech"
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_RERANK_MODEL = "rerank-v3.5"
EMBED_DIMENSIONS = 1024
MAX_INGEST_CHUNKS = int(os.getenv("MAX_INGEST_CHUNKS", "0"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
EMBED_BATCH_DELAY_SEC = float(os.getenv("EMBED_BATCH_DELAY_SEC", "6"))
COHERE_RETRY_MAX = int(os.getenv("COHERE_RETRY_MAX", "6"))
COHERE_RETRY_BASE_DELAY_SEC = float(os.getenv("COHERE_RETRY_BASE_DELAY_SEC", "20"))
RAG_CHUNK_MAX_LEN = int(os.getenv("RAG_CHUNK_MAX_LEN", "1500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))
RAG_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "5"))
RAG_SEMANTIC_WEIGHT = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.7"))
RAG_DOMAIN_WEIGHT = float(os.getenv("RAG_DOMAIN_WEIGHT", "0.3"))
RAG_MAX_SOURCES = int(os.getenv("RAG_MAX_SOURCES", "5"))
RAG_MAX_ARTICLES = int(os.getenv("RAG_MAX_ARTICLES", "3"))
RAG_MAX_WORDS = int(os.getenv("RAG_MAX_WORDS", "600"))
RAG_SCORE_RELATIVE_MIN = float(os.getenv("RAG_SCORE_RELATIVE_MIN", "0.6"))
RAG_CHUNK_CONTEXT_CHARS = int(os.getenv("RAG_CHUNK_CONTEXT_CHARS", "700"))
RAG_HYBRID_ENABLED = os.getenv("RAG_HYBRID_ENABLED", "true").lower() in ("1", "true", "yes")
RAG_BM25_TOP_K = int(os.getenv("RAG_BM25_TOP_K", "30"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RAG_JUDGE_ENABLED = os.getenv("RAG_JUDGE_ENABLED", "true").lower() in ("1", "true", "yes")
RAG_JUDGE_MAX_REVISIONS = int(os.getenv("RAG_JUDGE_MAX_REVISIONS", "1"))
RAG_JUDGE_MIN_SCORE = float(os.getenv("RAG_JUDGE_MIN_SCORE", "0.75"))
MODEL_JUDGE = os.getenv("MODEL_JUDGE", MODEL_ROUTER)
DATA_DIR = BASE_DIR / "data"
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant")))
INGEST_MANIFEST_PATH = DATA_DIR / "ingest_manifest.json"
AUTH_SECRET = os.getenv("AUTH_SECRET", "zmien-ten-klucz-w-produkcji-metaltech")
SESSION_MAX_AGE_SEC = int(os.getenv("SESSION_MAX_AGE_SEC", str(60 * 60 * 12)))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kodeks:kodeks@localhost:5432/kodekspracy",
)


class CompanyProfile(BaseModel):
    """Profil firmy MetalTech Sp. z o.o. używany przez generator pism."""

    nazwa: str = "MetalTech Sp. z o.o."
    nip: str = "1234567890"
    regon: str = "123456789"
    adres: str = "ul. Przemysłowa 15, 40-001 Katowice"
    branza: str = "produkcja metalowa"
    liczba_pracownikow: int = 150
    system_czasu_pracy: str = "system równoważny czasu pracy"
    reprezentant: str = "Jan Kowalski — Dyrektor HR"
    email_kadry: str = "kadry@metaltech.pl"
    telefon_kadry: str = "+48 32 123 45 67"


class Settings(BaseModel):
    google_api_key: str = Field(default_factory=lambda: GOOGLE_API_KEY)
    cohere_api_key: str = Field(default_factory=lambda: COHERE_API_KEY)
    database_url: str = Field(default_factory=lambda: DATABASE_URL)
    auth_secret: str = Field(default_factory=lambda: AUTH_SECRET)
    session_max_age_sec: int = Field(default_factory=lambda: SESSION_MAX_AGE_SEC)
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_ingest_chunks: int = Field(default_factory=lambda: MAX_INGEST_CHUNKS)
    embed_batch_size: int = Field(default_factory=lambda: EMBED_BATCH_SIZE)
    embed_batch_delay_sec: float = Field(default_factory=lambda: EMBED_BATCH_DELAY_SEC)
    cohere_retry_max: int = Field(default_factory=lambda: COHERE_RETRY_MAX)
    cohere_retry_base_delay_sec: float = Field(default_factory=lambda: COHERE_RETRY_BASE_DELAY_SEC)
    rag_judge_enabled: bool = Field(default_factory=lambda: RAG_JUDGE_ENABLED)
    rag_judge_max_revisions: int = Field(default_factory=lambda: RAG_JUDGE_MAX_REVISIONS)
    rag_judge_min_score: float = Field(default_factory=lambda: RAG_JUDGE_MIN_SCORE)
    company: CompanyProfile = Field(default_factory=CompanyProfile)


@lru_cache
def get_settings() -> Settings:
    return Settings()
