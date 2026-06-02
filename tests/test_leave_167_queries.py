"""Regresja: urlop na żądanie musi trafiać w 167²; odwołanie — w Art. 167 KP."""

from agents.legal_rag_queries import is_leave_167_query, leave_167_sub_queries
from agents.reranking import _is_art_167_recall_from_vacation, domain_relevance, detect_query_context


def test_is_leave_167_query_on_demand() -> None:
    assert is_leave_167_query("urlop na zadanie ile dni")
    assert is_leave_167_query("Czym rozni sie Art. 167 od Art. 1672 KP?")


def test_leave_167_sub_queries_include_recall() -> None:
    subs = leave_167_sub_queries("urlop na żądanie")
    joined = " ".join(subs).lower()
    assert "167²" in joined or "1672" in joined
    assert "odwołać" in joined or "odwołanie" in joined


def test_rerank_boosts_art_167_recall() -> None:
    ctx = detect_query_context("urlop na żądanie")
    payload = {
        "article": "Art. 167 KP",
        "text": "Art. 167. § 1. Pracodawca może odwołać pracownika z urlopu",
        "topic": "Kodeks pracy",
        "source": "ISAP",
    }
    score = domain_relevance("urlop na żądanie", payload, ctx)
    assert score >= 88.0


def test_recall_detector_excludes_1672() -> None:
    assert _is_art_167_recall_from_vacation(
        "Art. 167² KP",
        "Pracodawca udzieli urlopu na żądanie",
    ) is False
    assert _is_art_167_recall_from_vacation(
        "Art. 167 KP",
        "Pracodawca może odwołać pracownika z urlopu",
    ) is True
