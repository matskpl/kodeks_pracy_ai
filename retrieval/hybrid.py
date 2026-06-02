"""Hybrydowe wyszukiwanie: dense (Qdrant) + sparse (BM25) + RRF + Cohere rerank."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config import QDRANT_COLLECTION, RAG_BM25_TOP_K, RAG_HYBRID_ENABLED, RAG_RRF_K, RAG_TOP_K
from retrieval.bm25_index import BM25Hit, get_bm25_index
from vector_store import RetrievedChunk, _chunk_point_id, _expand_query, embed_texts, get_cohere_client, get_qdrant_client


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    k: int | None = None,
) -> list[tuple[str, float]]:
    """
    RRF: łączy listy (chunk_id, rank_score) z wielu retrieverów.
    Każda lista powinna być posortowana malejąco po jakości (rank 1 = najlepszy).
    """
    rrf_k = k if k is not None else RAG_RRF_K
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _dense_search(
    query: str,
    *,
    limit: int,
    client: Any,
    cohere: Any,
) -> list[tuple[str, dict[str, Any], float]]:
    expanded = _expand_query(query)
    query_vector = embed_texts([expanded], input_type="search_query", cohere=cohere)[0]
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    out: list[tuple[str, dict[str, Any], float]] = []
    for rank, hit in enumerate(response.points, start=1):
        payload = dict(hit.payload or {})
        cid = str(hit.id)
        # im wyższy rank, tym lepszy — score malejący z pozycją
        score = 1.0 / rank
        out.append((cid, payload, score))
    return out


def _sparse_search(query: str, *, limit: int) -> list[tuple[str, dict[str, Any], float]]:
    index = get_bm25_index()
    if index is None:
        logger.warning("BM25 niedostępny — tylko dense retrieval")
        return []
    hits: list[BM25Hit] = index.search(query, top_k=limit)
    if not hits:
        return []
    max_s = max(h.score for h in hits) or 1.0
    return [
        (h.chunk_id, h.payload, h.score / max_s)
        for h in hits
    ]


def merge_hybrid_hits(
    dense: list[tuple[str, dict[str, Any], float]],
    sparse: list[tuple[str, dict[str, Any], float]],
    *,
    rrf_k: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Scala wyniki RRF; payload z pierwszego źródła, które ma dany id."""
    dense_ranked = [(cid, sc) for cid, _, sc in dense]
    sparse_ranked = [(cid, sc) for cid, _, sc in sparse]
    lists = [lst for lst in (dense_ranked, sparse_ranked) if lst]
    if not lists:
        return []
    if len(lists) == 1:
        fused = lists[0]
    else:
        fused = reciprocal_rank_fusion(lists, k=rrf_k)

    payload_by_id: dict[str, dict[str, Any]] = {}
    for cid, payload, _ in dense + sparse:
        payload_by_id.setdefault(cid, payload)

    return [(cid, payload_by_id[cid]) for cid, _ in fused if cid in payload_by_id]


def search_hybrid(
    query: str,
    *,
    dense_limit: int | None = None,
    sparse_limit: int | None = None,
    client: Any | None = None,
    cohere: Any | None = None,
) -> list[Any]:
    """
    Zwraca listę obiektów kompatybilnych z rank_candidates (id + payload).
    """
    if not RAG_HYBRID_ENABLED:
        return []

    client = client or get_qdrant_client()
    co = cohere or get_cohere_client()
    d_lim = dense_limit or RAG_TOP_K
    s_lim = sparse_limit or RAG_BM25_TOP_K

    dense = _dense_search(query, limit=d_lim, client=client, cohere=co)
    sparse = _sparse_search(query, limit=s_lim)
    merged = merge_hybrid_hits(dense, sparse)

    if not merged:
        return []

    class _Hit:
        def __init__(self, hit_id: str, payload: dict[str, Any]) -> None:
            self.id = hit_id
            self.payload = payload

    logger.debug(
        "Hybrid RRF: dense={}, sparse={}, fused={} dla: {}",
        len(dense),
        len(sparse),
        len(merged),
        query[:80],
    )
    return [_Hit(cid, payload) for cid, payload in merged]


def payload_to_retrieved_chunk(
    chunk_id: str,
    payload: dict[str, Any],
    *,
    score: float,
    semantic_score: float = 0.0,
    domain_score: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=str(payload.get("text", "")),
        article=str(payload.get("article", "")),
        topic=str(payload.get("topic", "")),
        source=str(payload.get("source", "")),
        url=str(payload.get("url", "")),
        score=score,
        semantic_score=semantic_score,
        domain_score=domain_score,
    )


def chunk_dict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": payload.get("text", ""),
        "article": payload.get("article", ""),
        "topic": payload.get("topic", ""),
        "source": payload.get("source", "UNKNOWN"),
        "url": payload.get("url", ""),
    }
