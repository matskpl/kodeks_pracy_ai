"""Wspólne reguły grounding RAG i walidacja cytowań względem chunków."""

from __future__ import annotations

import re

from vector_store import RetrievedChunk

# Art. 36 § 1 KP, Art. 167² KP, Art. 135 itd.
_ARTICLE_IN_TEXT = re.compile(
    r"Art\.?\s*(\d+)\s*(?:§\s*(\d+[²¹³]?))?\s*(?:ust\.?\s*(\d+))?\s*(?:pkt\.?\s*(\d+))?\s*KP",
    re.IGNORECASE,
)

def global_grounding_user_rules() -> str:
    """Reguły użytkownika — import z hr_voice, żeby uniknąć duplikacji."""
    from agents.hr_voice import (
        HR_CALENDAR_RULES,
        HR_EXCEPTIONS_FIRST_RULES,
        HR_GROUNDING_RULES,
    )

    return (
        f"{HR_GROUNDING_RULES}\n\n{HR_EXCEPTIONS_FIRST_RULES}\n\n{HR_CALENDAR_RULES}"
    )


# Zachowane dla kompatybilności importów
GLOBAL_GROUNDING_USER_RULES = global_grounding_user_rules()


def _normalize_article(
    num: str,
    paragraph: str | None = None,
    ust: str | None = None,
) -> str:
    base = f"Art. {int(num)}"
    if paragraph:
        base += f" § {int(paragraph)}"
    if ust:
        base += f" ust. {int(ust)}"
    return f"{base} KP"


def allowed_articles_from_chunks(chunks: list[RetrievedChunk]) -> list[str]:
    """Unikalna lista artykułów obecnych w metadanych lub tekście chunków."""
    seen: set[str] = set()
    ordered: list[str] = []
    for ch in chunks:
        candidates = [ch.article.strip()] if ch.article.strip() else []
        for m in _ARTICLE_IN_TEXT.finditer(ch.text):
            candidates.append(
                _normalize_article(m.group(1), m.group(3), m.group(4))
            )
        for raw in candidates:
            if not raw or raw in seen:
                continue
            seen.add(raw)
            ordered.append(raw)
    return ordered


def format_allowed_articles_block(chunks: list[RetrievedChunk]) -> str:
    allowed = allowed_articles_from_chunks(chunks)
    if not allowed:
        return (
            "Dozwolone artykuły w źródłach: (brak jednoznacznych oznaczeń — "
            "opieraj się wyłącznie na treści numerowanych fragmentów [1], [2]…)"
        )
    items = "; ".join(allowed[:12])
    suffix = " …" if len(allowed) > 12 else ""
    return f"Dozwolone artykuły w źródłach (nie cytuj innych): {items}{suffix}"


def articles_cited_in_answer(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for m in _ARTICLE_IN_TEXT.finditer(text):
        norm = _normalize_article(m.group(1), m.group(3), m.group(4))
        if norm not in seen:
            seen.add(norm)
            found.append(norm)
    return found


def _article_base_key(article: str) -> tuple[int, int | None]:
    m = re.search(r"Art\.?\s*(\d+)(?:\s*§\s*(\d+))?", article, re.I)
    if not m:
        return (-1, None)
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def _is_article_allowed(cited: str, allowed: list[str]) -> bool:
    if cited in allowed:
        return True
    c_num, c_par = _article_base_key(cited)
    for a in allowed:
        a_num, a_par = _article_base_key(a)
        if c_num != a_num:
            continue
        if c_par is None or a_par is None:
            return True
        if c_par == a_par:
            return True
    return False


def find_ungrounded_articles(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    allowed = allowed_articles_from_chunks(chunks)
    if not allowed:
        return []
    return [a for a in articles_cited_in_answer(answer) if not _is_article_allowed(a, allowed)]


def grounding_disclaimer(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Tekst ostrzeżenia lub pusty string, gdy cytowania są w granicach chunków."""
    extra = find_ungrounded_articles(answer, chunks)
    if not extra:
        return ""
    listed = ", ".join(extra[:5])
    return (
        f"\n\nUwaga: w odpowiedzi pojawiły się artykuły ({listed}), "
        "których nie ma w pobranych fragmentach bazy — zweryfikuj je w Kodeksie pracy."
    )


def apply_grounding_guard(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Dopina ostrzeżenie, gdy model zacytował artykuły spoza retrievalu."""
    note = grounding_disclaimer(answer, chunks)
    if not note:
        return answer
    return answer.rstrip() + note
