"""Uczciwość cytowań — dopasowanie tezy do treści i metadanych chunków."""

from __future__ import annotations

import re

from vector_store import RetrievedChunk

CITATION_HONESTY_RULES = """
KRYTYCZNA ZASADA: UCZCIWOŚĆ CYTOWANIA (CITATION HONESTY)

Masz prawo podać poprawny fakt prawny z własnej wiedzy ogólnej, ale masz BEZWZGLĘDNY ZAKAZ
przypisywania go do pobranego dokumentu (chunk), jeśli ten dokument nie zawiera wyraźnie tej zasady.

Nie podrabiaj ani nie naciągaj cytowań. Jeśli pobrany fragment wspomina o „Art. 167” (odwołanie z urlopu),
a Twoja odpowiedź mówi o „zrzeczeniu się urlopu”, pod żadnym pozorem NIE CYTUJ Art. 167 jako podstawy tej zasady.

Jeśli podajesz fakt, który nie ma bezpośredniego potwierdzenia w dostarczonych fragmentach, na końcu tego akapitu
dodaj dokładnie tę frazę:
(Uwaga: W dostarczonych materiałach źródłowych brakuje specyficznego artykułu potwierdzającego tę zasadę)

Podstawę prawną na końcu odpowiedzi buduj WYŁĄCZNIE z artykułów, których treść w fragmentach faktycznie
dotyczy omawianego problemu — nie wybieraj artykułu tylko dlatego, że pojawił się w wynikach wyszukiwania.
""".strip()

CITATION_DISCLAIMER_PHRASE = (
    "W dostarczonych materiałach źródłowych brakuje specyficznego artykułu potwierdzającego tę zasadę"
)

# Zapytanie → oczekiwane słowa kluczowe w treści/temacie chunku (dowód zgodności)
_QUERY_EVIDENCE: list[tuple[re.Pattern[str], tuple[str, ...], str | None]] = [
    (
        re.compile(r"zrzecz|zrzeka|zrzec\s", re.I),
        ("zrzec", "nie może zrzec", "prawa do urlopu", "urlopu wypoczynkowego"),
        r"Art\.?\s*152",
    ),
    (
        re.compile(r"urlop\s+na\s+żądanie|urlop\s+na\s+zadanie", re.I),
        ("na żądanie", "na zadanie", "167²", "1672", "nie więcej niż 4 dni"),
        r"167\s*[²2]|1672",
    ),
    (
        re.compile(r"odwołać\s+z\s+urlopu|odwolac\s+z\s+urlopu|odwołanie\s+z\s+urlopu", re.I),
        ("odwołać", "odwołanie", "odwolac", "nieprzewidziane"),
        r"Art\.?\s*167\s*KP",
    ),
    (
        re.compile(r"okres\s+wypowiedzenia|wypowiedzenie\s+umow", re.I),
        ("okres wypowiedzenia", "wypowiedzenie umowy", "art. 36", "art 36"),
        r"Art\.?\s*36",
    ),
]


def _chunk_blob(ch: RetrievedChunk) -> str:
    return f"{ch.article} {ch.topic} {ch.text}".lower()


def chunk_supports_claim(query: str, chunk: RetrievedChunk) -> bool:
    """Czy pojedynczy fragment merytorycznie pasuje do zapytania (nie tylko semantycznie)."""
    blob = _chunk_blob(chunk)
    for pattern, keywords, article_re in _QUERY_EVIDENCE:
        if not pattern.search(query):
            continue
        if article_re and not re.search(article_re, chunk.article, re.I):
            # nadal może być w treści
            if not re.search(article_re, chunk.text, re.I):
                continue
        if any(kw.lower() in blob for kw in keywords):
            return True
    return False


def chunks_cover_claim(query: str, chunks: list[RetrievedChunk]) -> bool:
    """Czy którykolwiek z topowych fragmentów potwierdza temat zapytania."""
    if not chunks:
        return False
    return any(chunk_supports_claim(query, ch) for ch in chunks)


def top_chunks_missing_expected_article(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    expected_article_pattern: str,
) -> bool:
    """True gdy w top N brak oczekiwanego artykułu (np. Art. 152 przy zrzeczeniu)."""
    if not chunks:
        return True
    pat = re.compile(expected_article_pattern, re.I)
    return not any(pat.search(ch.article) or pat.search(ch.text) for ch in chunks)


def honesty_context_note(query: str, chunks: list[RetrievedChunk]) -> str:
    """Uwaga wstrzykiwana do promptu, gdy retrieval nie trafił we właściwy artykuł."""
    if re.search(r"zrzecz|zrzeka|zrzec\s", query, re.I):
        if top_chunks_missing_expected_article(query, chunks, expected_article_pattern=r"Art\.?\s*152"):
            return (
                "UWAGA RETRIEVAL: W pobranych fragmentach brak Art. 152 KP o zrzeczeniu się urlopu. "
                "Nie przypisuj tej zasady do innych artykułów (np. Art. 167). "
                f"Przy fakcie bez potwierdzenia w chunkach użyj frazy: ({CITATION_DISCLAIMER_PHRASE})."
            )
    if re.search(r"urlop\s+na\s+żądanie|urlop\s+na\s+zadanie", query, re.I):
        if chunks_cover_claim(query, chunks):
            return ""
    if not chunks_cover_claim(query, chunks) and chunks:
        return (
            "UWAGA RETRIEVAL: Żaden fragment nie potwierdza wprost tematu pytania. "
            f"Nie naciągaj cytowań; przy wiedzy ogólnej dodaj: ({CITATION_DISCLAIMER_PHRASE})."
        )
    return ""
