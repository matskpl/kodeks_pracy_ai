"""Testy wykrywania zapytań o wypowiedzenie dla RAG."""

from agents.legal_rag import build_rag_user_prompt
from agents.legal_rag_queries import is_termination_query, termination_sub_queries
from vector_store import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="1",
        text="Art. 36",
        article="Art. 36 KP",
        topic="wypowiedzenie",
        source="ISAP",
        url="",
        score=0.9,
        semantic_score=0.9,
        domain_score=0.9,
    )


def test_is_termination_query_with_dates():
    q = (
        "Pracownik zatrudniony 1 maja, wypowiedzenie 31 października. "
        "Jaki okres wypowiedzenia i kiedy kończy się umowa?"
    )
    assert is_termination_query(q)


def test_is_termination_query_negative():
    assert not is_termination_query("Ile wynosi minimalne wynagrodzenie?")


def test_termination_sub_queries_includes_art36():
    subs = termination_sub_queries("wypowiedzenie 31 października")
    assert any("Art. 36" in s for s in subs)
    assert any("wlicza" in s for s in subs)


def test_build_rag_prompt_hr_style_and_termination():
    q = "zatrudniony 1 maja, wypowiedzenie 31 października, jaki okres wypowiedzenia"
    prompt = build_rag_user_prompt(q, [_chunk()])
    assert "stylu eksperta HR" in prompt or "eksperta HR" in prompt
    assert "WYJĄTKÓW" in prompt or "Exceptions First" in prompt
    assert "30 listopada" in prompt or "WYPOWIEDZENIE" in prompt
