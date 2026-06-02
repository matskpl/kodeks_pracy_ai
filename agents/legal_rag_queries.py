"""Wykrywanie typu zapytania i rozszerzone zapytania RAG."""

from __future__ import annotations

import re

TERMINATION_KEYWORDS = (
    "wypowiedzen",
    "okres wypowiedzenia",
    "rozwiąże się umow",
    "rozwiaze sie umow",
    "koniec umowy",
    "kiedy kończy",
)

DATE_IN_TEXT = re.compile(
    r"\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
    r"września|października|pazdziernika|listopada|grudnia)|"
    r"zatrudnion",
    re.IGNORECASE,
)


def is_termination_query(query: str) -> bool:
    lowered = query.casefold()
    if not any(k in lowered for k in TERMINATION_KEYWORDS):
        return False
    return bool(DATE_IN_TEXT.search(lowered)) or "jaki okres" in lowered or "kiedy" in lowered


def termination_sub_queries(user_query: str) -> list[str]:
    """Dodatkowe zapytania do Qdrant — Art. 36 KP + PIP o wliczaniu okresu wypowiedzenia."""
    return [
        user_query,
        (
            "Art. 36 § 1 KP okres wypowiedzenia krócej niż 6 miesięcy "
            "co najmniej 6 miesięcy trzy miesiące zatrudnienie u pracodawcy"
        ),
        (
            "okres wypowiedzenia wlicza się do stażu zakładowego pracy "
            "wydłużenie z dwutygodniowego do jednego miesiąca PIP"
        ),
        "Art. 36 § 3 KP bieg okresu wypowiedzenia miesiąc kalendarzowy koniec umowy",
    ]


LEAVE_167_DEMAND_HINTS = (
    "urlop na żądanie",
    "urlop na zadanie",
    "167²",
    "1672",
    "art. 167",
    "art 167",
)

LEAVE_167_RECALL_HINTS = (
    "odwołać z urlopu",
    "odwolac z urlopu",
    "odwołanie z urlopu",
    "odwolanie z urlopu",
)


def is_leave_167_query(query: str) -> bool:
    """Urlop na żądanie lub porównanie / odwołanie — potrzebne oba warianty art. 167."""
    lowered = query.casefold()
    if any(h in lowered for h in LEAVE_167_DEMAND_HINTS):
        return True
    if any(h in lowered for h in LEAVE_167_RECALL_HINTS):
        return True
    if "167" in lowered and any(
        x in lowered
        for x in ("różnic", "różni", "roznic", "rozni", "roznica", "czym różni", "a art")
    ):
        return True
    return False


def leave_167_sub_queries(user_query: str) -> list[str]:
    """Art. 167² (żądanie) + Art. 167 (odwołanie z urlopu) — osobne zapytania embeddingowe."""
    return [
        user_query,
        (
            "Art. 167² KP pracodawca obowiązany udzielić urlop na żądanie "
            "nie więcej niż 4 dni rok kalendarzowy zgłoszenie"
        ),
        (
            "Art. 167 § 1 KP pracodawca odwołać pracownika z urlopu "
            "okoliczności nieprzewidziane koszty odwołania"
        ),
    ]
