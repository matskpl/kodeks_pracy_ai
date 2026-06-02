"""Testy modułu sędziego źródeł."""

from agents.answer_judge import rule_based_issues, verdict_from_rules
from vector_store import RetrievedChunk


def _chunk(article: str, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        id="c1",
        text=text or article,
        article=article,
        topic="t",
        source="ISAP",
        url="",
        score=0.9,
        semantic_score=0.9,
        domain_score=0.9,
    )


def test_rule_ungrounded_article():
    chunks = [_chunk("Art. 36 KP")]
    answer = "Wynika to z Art. 30 KP i Art. 36 KP."
    issues = rule_based_issues(answer, "pytanie", chunks)
    assert any("30" in i for i in issues)


def test_rule_inline_chunk_refs_forbidden():
    chunks = [_chunk("Art. 167² KP")]
    answer = "Zgodnie z [1] pracodawca musi udzielić urlopu na żądanie."
    issues = rule_based_issues(answer, "urlop na żądanie", chunks)
    assert any("przypisy" in i.lower() for i in issues)


def test_rule_evasive_date_on_termination():
    chunks = [_chunk("Art. 36 KP", "Art. 36 § 3 bieg okresu wypowiedzenia miesiąc kalendarzowy")]
    answer = "W dostarczonych źródłach brakuje informacji pozwalającej wskazać konkretną datę końca umowy."
    query = "zatrudniony 1 maja, wypowiedzenie 31 października, kiedy koniec umowy"
    issues = rule_based_issues(answer, query, chunks)
    assert any("dat" in i.lower() for i in issues)


def test_verdict_from_rules_not_accepted():
    v = verdict_from_rules(["błąd testowy"])
    assert v.accepted is False
    assert v.check_source == "rules"
