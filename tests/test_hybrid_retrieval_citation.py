"""Hybryd BM25 + RRF oraz uczciwość cytowań (zrzeczenie się urlopu → Art. 152)."""

from __future__ import annotations

from agents.citation_honesty import (
    CITATION_DISCLAIMER_PHRASE,
    CITATION_HONESTY_RULES,
    chunks_cover_claim,
    honesty_context_note,
    top_chunks_missing_expected_article,
)
from agents.legal_rag import build_rag_user_prompt
from agents.answer_judge import rule_based_issues
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import merge_hybrid_hits, reciprocal_rank_fusion
from vector_store import RetrievedChunk


def _make_chunk(
    *,
    cid: str,
    article: str,
    topic: str,
    text: str,
) -> dict:
    return {
        "source": "ISAP",
        "url": "",
        "article": article,
        "topic": topic,
        "text": text,
    }


def test_bm25_zrzeczenie_ranks_art152_first():
    corpus = [
        _make_chunk(
            cid="167",
            article="Art. 167 KP",
            topic="Odwołanie pracownika z urlopu",
            text="Art. 167. Pracodawca moze odwolac pracownika z urlopu w razie nagłej potrzeby zakladu.",
        ),
        _make_chunk(
            cid="152",
            article="Art. 152 KP",
            topic="Urlop wypoczynkowy — zasady",
            text="Art. 152. Paragraf 2. Pracownik nie moze zrzec sie prawa do urlopu wypoczynkowego.",
        ),
    ]
    index = BM25Index.from_chunks(corpus)
    hits = index.search("zrzeczenie sie urlopu", top_k=5)
    assert hits
    articles = [h.payload["article"] for h in hits]
    assert "Art. 152 KP" in articles
    assert articles[0] == "Art. 152 KP"


def test_rrf_promotes_sparse_hit_when_dense_wrong():
    """Sparse trafia w Art. 152, dense tylko Art. 167 — RRF ma faworyzować 152."""
    dense = [
        ("167-id", {"article": "Art. 167 KP"}, 1.0),
        ("152-id", {"article": "Art. 152 KP"}, 0.2),
    ]
    sparse = [("152-id", {"article": "Art. 152 KP"}, 1.0)]
    merged = merge_hybrid_hits(dense, sparse, rrf_k=60)
    assert merged[0][0] == "152-id"


def test_honesty_note_when_art152_missing_from_top():
    chunks = [
        RetrievedChunk(
            id="1",
            text="odwołanie z urlopu art 167",
            article="Art. 167 KP",
            topic="odwołanie",
            source="ISAP",
            url="",
            score=0.9,
            semantic_score=0.9,
            domain_score=0.9,
        ),
    ]
    q = "Czy pracownik może zrzec się urlopu?"
    assert top_chunks_missing_expected_article(q, chunks, expected_article_pattern=r"Art\.?\s*152")
    note = honesty_context_note(q, chunks)
    assert "Art. 152" in note
    assert "167" in note


def test_prompt_includes_citation_honesty():
    chunks = [
        RetrievedChunk(
            id="1",
            text="odwołanie",
            article="Art. 167 KP",
            topic="urlop",
            source="ISAP",
            url="",
            score=0.8,
            semantic_score=0.8,
            domain_score=0.8,
        ),
    ]
    prompt = build_rag_user_prompt("zrzeczenie się urlopu", chunks)
    assert "UCZCIWOŚĆ CYTOWANIA" in prompt or CITATION_HONESTY_RULES[:40] in prompt
    assert "ŹRÓDŁO:" in prompt
    assert "Treść tekstu:" in prompt


def test_judge_rejects_art167_for_zrzeczenie_without_disclaimer():
    chunks = [
        RetrievedChunk(
            id="1",
            text="Art. 167 odwołanie z urlopu",
            article="Art. 167 KP",
            topic="odwołanie",
            source="ISAP",
            url="",
            score=0.9,
            semantic_score=0.9,
            domain_score=0.9,
        ),
    ]
    answer = (
        "Pracownik nie może zrzec się urlopu. Podstawa: Art. 167 KP."
    )
    issues = rule_based_issues(answer, "zrzeczenie się urlopu", chunks)
    assert any("167" in i or "152" in i or "materiałach" in i.lower() for i in issues)


def test_chunks_cover_claim_art152():
    chunks = [
        RetrievedChunk(
            id="2",
            text="Pracownik nie może zrzec się prawa do urlopu.",
            article="Art. 152 KP",
            topic="urlop",
            source="ISAP",
            url="",
            score=0.95,
            semantic_score=0.95,
            domain_score=0.95,
        ),
    ]
    assert chunks_cover_claim("zrzeczenie się urlopu", chunks)


def test_reciprocal_rank_fusion_single_list():
    fused = reciprocal_rank_fusion([[("a", 1.0), ("b", 0.5)]])
    assert fused[0][0] == "a"
