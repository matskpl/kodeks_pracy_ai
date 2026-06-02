"""Testy globalnego grounding RAG."""

from agents.legal_rag import build_rag_user_prompt
from agents.rag_grounding import (
    apply_grounding_guard,
    find_ungrounded_articles,
    format_allowed_articles_block,
)
from vector_store import RetrievedChunk


def _chunk(article: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        id="1",
        text=text or article,
        article=article,
        topic="test",
        source="ISAP",
        url="",
        score=0.9,
        semantic_score=0.9,
        domain_score=0.9,
    )


def test_build_rag_prompt_always_has_global_grounding():
    q = "Czy pracodawca może odmówić urlopu na żądanie?"
    prompt = build_rag_user_prompt(q, [_chunk("Art. 167² KP")])
    assert "Dozwolone artykuły" in prompt
    assert "stylu eksperta HR" in prompt or "eksperta HR" in prompt
    assert "Exceptions First" in prompt or "WYJĄTKÓW" in prompt
    assert "Bez przypisów" in prompt or "bez przypisów" in prompt.lower()


def test_find_ungrounded_article():
    chunks = [_chunk("Art. 36 KP", "Art. 36 § 1 KP okres wypowiedzenia")]
    answer = "Zgodnie z Art. 30 KP oraz Art. 36 KP wynika…"
    extra = find_ungrounded_articles(answer, chunks)
    assert any("30" in a for a in extra)


def test_apply_grounding_guard_appends_note():
    chunks = [_chunk("Art. 167² KP")]
    out = apply_grounding_guard("Podstawa: Art. 999 KP.", chunks)
    assert "Uwaga:" in out
    assert "999" in out
