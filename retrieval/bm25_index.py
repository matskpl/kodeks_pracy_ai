"""Indeks BM25 (sparse) — precyzyjne terminy prawne: zrzeczenie, odprawa, ryczałt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from config import DATA_DIR
from vector_store import _chunk_point_id

BM25_INDEX_PATH = DATA_DIR / "bm25_index.json"

_TOKEN_RE = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9²]+", re.UNICODE)


def tokenize_polish(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]
    expanded = list(tokens)
    for t in tokens:
        if len(t) >= 5:
            expanded.append(t[:5])
        if len(t) >= 4:
            expanded.append(t[:4])
    return expanded


def sparse_document_text(chunk: dict[str, Any]) -> str:
    """Tekst do BM25 — metadane + treść (słowa kluczowe prawne)."""
    return f"{chunk.get('article', '')} {chunk.get('topic', '')} {chunk.get('text', '')}"


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class BM25Index:
    """Indeks w pamięci z opcjonalnym zapisem na dysk."""

    def __init__(
        self,
        *,
        chunk_ids: list[str],
        payloads: list[dict[str, Any]],
        bm25: BM25Okapi,
    ) -> None:
        self._chunk_ids = chunk_ids
        self._payloads = payloads
        self._bm25 = bm25
        self._id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

    @classmethod
    def from_chunks(cls, chunks: list[dict[str, Any]]) -> BM25Index:
        chunk_ids: list[str] = []
        payloads: list[dict[str, Any]] = []
        corpus_tokens: list[list[str]] = []
        for ch in chunks:
            cid = _chunk_point_id(ch)
            chunk_ids.append(cid)
            payloads.append(
                {
                    "text": ch.get("text", ""),
                    "article": ch.get("article", ""),
                    "topic": ch.get("topic", ""),
                    "source": ch.get("source", "UNKNOWN"),
                    "url": ch.get("url", ""),
                }
            )
            corpus_tokens.append(tokenize_polish(sparse_document_text(ch)))
        if not corpus_tokens:
            raise ValueError("Pusty korpus BM25")
        bm25 = BM25Okapi(corpus_tokens)
        return cls(chunk_ids=chunk_ids, payloads=payloads, bm25=bm25)

    def search(self, query: str, *, top_k: int = 30) -> list[BM25Hit]:
        tokens = tokenize_polish(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        hits: list[BM25Hit] = []
        for i in ranked_idx:
            hits.append(
                BM25Hit(
                    chunk_id=self._chunk_ids[i],
                    score=float(scores[i]),
                    payload=dict(self._payloads[i]),
                )
            )
        return hits

    def save(self, path: Path | None = None) -> None:
        path = path or BM25_INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunk_ids": self._chunk_ids,
            "payloads": self._payloads,
            "corpus_tokens": [
                tokenize_polish(sparse_document_text(p)) for p in self._payloads
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.info("Zapisano indeks BM25: {} ({} dokumentów)", path, len(self._chunk_ids))

    @classmethod
    def load(cls, path: Path | None = None) -> BM25Index | None:
        path = path or BM25_INDEX_PATH
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        corpus_tokens = raw.get("corpus_tokens") or []
        if not corpus_tokens:
            return None
        bm25 = BM25Okapi(corpus_tokens)
        return cls(
            chunk_ids=list(raw["chunk_ids"]),
            payloads=list(raw["payloads"]),
            bm25=bm25,
        )


_bm25_index: BM25Index | None = None


def rebuild_bm25_index(chunks: list[dict[str, Any]]) -> BM25Index:
    global _bm25_index
    _bm25_index = BM25Index.from_chunks(chunks)
    _bm25_index.save()
    return _bm25_index


def get_bm25_index(*, chunks_fallback: list[dict[str, Any]] | None = None) -> BM25Index | None:
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index
    loaded = BM25Index.load()
    if loaded is not None:
        _bm25_index = loaded
        return _bm25_index
    if chunks_fallback:
        _bm25_index = BM25Index.from_chunks(chunks_fallback)
        return _bm25_index
    try:
        from scrapers.pipeline import load_cached_chunks

        cached = load_cached_chunks()
        if cached:
            _bm25_index = BM25Index.from_chunks(cached)
            return _bm25_index
    except Exception as exc:
        logger.warning("Nie udało się zbudować BM25 z cache: {}", exc)
    return None


def clear_bm25_cache() -> None:
    global _bm25_index
    _bm25_index = None
