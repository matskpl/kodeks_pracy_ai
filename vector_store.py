"""Warstwa wektorowa: Qdrant (trwały dysk) + Cohere Embed v4 + Rerank v3."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import NAMESPACE_DNS, uuid5

import cohere
from cohere.errors import TooManyRequestsError
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import (
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
    DATA_DIR,
    EMBED_DIMENSIONS,
    INGEST_MANIFEST_PATH,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    RAG_HYBRID_ENABLED,
    RAG_RERANK_TOP_N,
    RAG_TOP_K,
    get_settings,
)
from agents.reranking import rank_candidates, select_final_candidates

QUERY_EXPANSIONS: dict[str, str] = {
    "urlop na żądanie": (
        "Art. 167² KP pracodawca obowiązany udzielić na żądanie pracownika urlop 4 dni; "
        "Art. 167 KP odwołanie pracownika z urlopu okoliczności nieprzewidziane"
    ),
    "odmowa urlopu na żądanie": "Art. 167² KP czy pracodawca może odmówić urlopu na żądanie",
    "system równoważny": "Art. 135 KP system równoważnego czasu pracy okres rozliczeniowy produkcja",
    "równoważny czasu pracy": "Art. 135 KP Art. 129 KP normy czasu pracy",
    "rownowazny czasu pracy": "Art. 135 KP system równoważnego czasu pracy",
    "okres wypowiedzenia": (
        "Art. 36 § 1 KP okres wypowiedzenia krócej niż 6 miesięcy co najmniej 6 miesięcy "
        "wlicza się do stażu zakładowego"
    ),
    "wypowiedzenie umowy": (
        "Art. 36 KP wypowiedzenie umowy o pracę czas nieokreślony bieg okresu § 3"
    ),
    "rozwiąże się umowa": "Art. 36 KP koniec umowy okres wypowiedzenia staż u pracodawcy",
    "zrzeczenie się urlopu": (
        "Art. 152 § 2 KP pracownik nie może zrzec się prawa do urlopu wypoczynkowego"
    ),
    "zrzeczenie urlopu": "Art. 152 KP zrzeczenie prawa do urlopu wypoczynkowego",
    "zrzeka się urlopu": "Art. 152 § 2 KP nie może zrzec się prawa do urlopu",
}


def _chunk_embed_text(chunk: dict[str, Any]) -> str:
    return f"{chunk['article']} | {chunk['topic']}\n{chunk['text']}"


def _expand_query(query: str) -> str:
    lowered = query.lower()
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in lowered:
            return f"{query}\n{expansion}"
    return query


def _chunk_point_id(chunk: dict[str, Any]) -> str:
    raw = "|".join(
        [
            chunk.get("source", ""),
            chunk.get("url", ""),
            chunk.get("article", ""),
            chunk.get("topic", ""),
            chunk.get("text", "")[:240],
        ]
    )
    return str(uuid5(NAMESPACE_DNS, hashlib.sha256(raw.encode("utf-8")).hexdigest()))


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    article: str
    topic: str
    source: str
    url: str
    score: float
    semantic_score: float = 0.0
    domain_score: float = 0.0


_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(QDRANT_PATH))
        logger.info("Qdrant (trwały): {}", QDRANT_PATH)
    return _qdrant_client


def get_cohere_client() -> cohere.Client:
    settings = get_settings()
    if not settings.cohere_api_key:
        raise RuntimeError("Brak COHERE_API_KEY — ustaw zmienną środowiskową.")
    return cohere.Client(api_key=settings.cohere_api_key)


T = TypeVar("T")


def _cohere_call_with_retry(label: str, fn: Callable[[], T]) -> T:
    settings = get_settings()
    last_exc: Exception | None = None
    for attempt in range(settings.cohere_retry_max + 1):
        try:
            return fn()
        except TooManyRequestsError as exc:
            last_exc = exc
            if attempt >= settings.cohere_retry_max:
                break
            delay = max(
                60.0,
                settings.cohere_retry_base_delay_sec * (2**attempt),
            )
            logger.warning(
                "Cohere rate limit ({}) — ponawiam za {:.0f}s (próba {}/{})",
                label,
                delay,
                attempt + 1,
                settings.cohere_retry_max,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def ensure_collection(client: QdrantClient | None = None, vector_size: int | None = None) -> None:
    client = client or get_qdrant_client()
    if client.collection_exists(QDRANT_COLLECTION):
        return
    size = vector_size or EMBED_DIMENSIONS
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )
    logger.info("Utworzono kolekcję Qdrant: {} (dim={})", QDRANT_COLLECTION, size)


def reset_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
        logger.info("Usunięto kolekcję Qdrant: {}", QDRANT_COLLECTION)


def collection_point_count(client: QdrantClient | None = None) -> int:
    client = client or get_qdrant_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return 0
    return client.get_collection(QDRANT_COLLECTION).points_count or 0


def load_ingest_manifest() -> dict[str, Any] | None:
    if not INGEST_MANIFEST_PATH.exists():
        return None
    return json.loads(INGEST_MANIFEST_PATH.read_text(encoding="utf-8"))


def save_ingest_manifest(*, chunk_count: int, scraped_at: str | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_count": chunk_count,
        "scraped_at": scraped_at,
        "embed_model": COHERE_EMBED_MODEL,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "collection": QDRANT_COLLECTION,
    }
    INGEST_MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Zapisano manifest ingestu: {}", INGEST_MANIFEST_PATH)


def knowledge_base_is_ready(
    *,
    client: QdrantClient | None = None,
    expected_chunks: int | None = None,
    scraped_at: str | None = None,
) -> bool:
    client = client or get_qdrant_client()
    count = collection_point_count(client)
    if count <= 0:
        return False
    manifest = load_ingest_manifest()
    if manifest is None:
        logger.info("Qdrant ma {} punktów, brak manifestu — uznaję bazę za gotową", count)
        return True
    if expected_chunks is not None and manifest.get("chunk_count") != expected_chunks:
        return False
    if scraped_at is not None and manifest.get("scraped_at") != scraped_at:
        return False
    if manifest.get("embed_model") != COHERE_EMBED_MODEL:
        return False
    logger.debug("Baza wiedzy gotowa — {} punktów w Qdrant", count)
    return True


def embed_texts(
    texts: list[str],
    *,
    input_type: str = "search_document",
    cohere: cohere.Client | None = None,
) -> list[list[float]]:
    co = cohere or get_cohere_client()

    def _call() -> list[list[float]]:
        response = co.embed(
            texts=texts,
            model=COHERE_EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        return [list(item) for item in response.embeddings.float_]

    return _cohere_call_with_retry(f"embed ({len(texts)} tekstów)", _call)


def upsert_chunks(
    chunks: list[dict[str, Any]],
    client: QdrantClient | None = None,
    *,
    recreate: bool = False,
    scraped_at: str | None = None,
) -> int:
    client = client or get_qdrant_client()

    settings = get_settings()
    max_chunks = settings.max_ingest_chunks
    batch_size = max(1, settings.embed_batch_size)
    if max_chunks > 0 and len(chunks) > max_chunks:
        logger.info("Ograniczam ingest do {} chunków (z {}).", max_chunks, len(chunks))
        chunks = chunks[:max_chunks]

    if recreate:
        reset_collection(client)

    total = 0
    vector_size: int | None = None
    batch_count = (len(chunks) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        texts = [_chunk_embed_text(chunk) for chunk in batch]
        vectors = embed_texts(texts, input_type="search_document")
        if vector_size is None:
            vector_size = len(vectors[0])
            ensure_collection(client, vector_size=vector_size)

        points = [
            PointStruct(
                id=_chunk_point_id(chunk),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "article": chunk["article"],
                    "topic": chunk["topic"],
                    "source": chunk.get("source", "UNKNOWN"),
                    "url": chunk.get("url", ""),
                },
            )
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total += len(points)
        logger.info(
            "Embed batch {}/{} — wgrano {} chunków (łącznie {})",
            batch_idx,
            batch_count,
            len(points),
            total,
        )
        if batch_idx < batch_count and settings.embed_batch_delay_sec > 0:
            time.sleep(settings.embed_batch_delay_sec)

    save_ingest_manifest(chunk_count=total, scraped_at=scraped_at)
    try:
        from retrieval.bm25_index import rebuild_bm25_index

        rebuild_bm25_index(chunks)
    except Exception as exc:
        logger.warning("Nie zbudowano indeksu BM25 po ingest: {}", exc)
    logger.info("Wgrano {} chunków do Qdrant", total)
    return total


def search_kodeks(
    query: str,
    *,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    client: QdrantClient | None = None,
    cohere: cohere.Client | None = None,
) -> list[RetrievedChunk]:
    client = client or get_qdrant_client()
    co = cohere or get_cohere_client()
    ensure_collection(client)

    qdrant_limit = top_k or RAG_TOP_K
    final_limit = rerank_top_n or RAG_RERANK_TOP_N

    expanded_query = _expand_query(query)
    hits: list[Any] = []

    if RAG_HYBRID_ENABLED:
        from retrieval.hybrid import search_hybrid

        hits = search_hybrid(
            expanded_query,
            dense_limit=qdrant_limit,
            client=client,
            cohere=co,
        )

    if not hits:
        query_vector = embed_texts([expanded_query], input_type="search_query", cohere=co)[0]
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=qdrant_limit,
            with_payload=True,
        )
        hits = response.points

    if not hits:
        logger.warning("Brak wyników w Qdrant dla zapytania: {}", query[:120])
        return []

    documents = [str((hit.payload or {}).get("text", "")) for hit in hits]

    def _rerank_call():
        return co.rerank(
            model=COHERE_RERANK_MODEL,
            query=expanded_query,
            documents=documents,
            top_n=len(documents),
        )

    rerank = _cohere_call_with_retry("rerank", _rerank_call)
    ranked = rank_candidates(query, hits, rerank.results)
    selected = select_final_candidates(ranked, query, max_sources=final_limit)

    if selected:
        logger.debug(
            "Rerank: best={:.3f}, próg={:.3f}, wybrano {} / {} kandydatów",
            selected[0].final_score,
            selected[0].final_score * 0.6,
            len(selected),
            len(ranked),
        )

    results: list[RetrievedChunk] = []
    for candidate in selected:
        payload = candidate.payload
        results.append(
            RetrievedChunk(
                id=candidate.hit_id,
                text=str(payload.get("text", "")),
                article=str(payload.get("article", "")),
                topic=str(payload.get("topic", "")),
                source=str(payload.get("source", "")),
                url=str(payload.get("url", "")),
                score=candidate.final_score,
                semantic_score=candidate.semantic_score,
                domain_score=candidate.domain_score,
            )
        )
    logger.debug(
        "Rerank hybrydowy: {} wyników (sem+domain) dla: {}",
        len(results),
        query[:80],
    )
    return results
