"""Hybrydowy reranking: Cohere semantic + reguły domenowe (metadata-aware)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import (
    RAG_DOMAIN_WEIGHT,
    RAG_MAX_ARTICLES,
    RAG_MAX_SOURCES,
    RAG_SCORE_RELATIVE_MIN,
    RAG_SEMANTIC_WEIGHT,
)

ARTICLE_NUM_RE = re.compile(r"Art\.\s*(\d+)", re.I)

PRODUCTION_HINTS = (
    "produkcj",
    "zakład produkcyjny",
    "zaklad produkcyjny",
    "pracownik produkcyjny",
    "fabryk",
    "huta",
    "metaltech",
    "metal",
)
EQUAL_TIME_HINTS = (
    "równoważn",
    "rownowazn",
    "system równoważny",
    "system rownowazny",
    "okres rozliczeniowy",
)
SPECIAL_ROLE_HINTS = (
    "dozór urządzeń",
    "dozor urzadzen",
    "ochrona osób",
    "ochrona osob",
    "straż pożarn",
    "straz pozarn",
    "ratownictw",
    "pilnowanie mienia",
    "pogotowie do pracy",
)
LEAVE_DEMAND_HINTS = ("urlop na żądanie", "urlop na zadanie", "odmowa urlopu")
LEAVE_WAIVER_HINTS = (
    "zrzeczen",
    "zrzeka",
    "zrzec si",
    "zrzec się",
    "rezygnacj z urlopu",
    "zrzec się urlopu",
)
TERMINATION_HINTS = (
    "wypowiedzen",
    "okres wypowiedzenia",
    "rozwiąże się umow",
    "rozwiaze sie umow",
    "koniec umowy",
    "kiedy kończy",
)
DETAIL_ANALYSIS_HINTS = (
    "szczegółow",
    "szczegolow",
    "wyczerpując",
    "wyczerpujac",
    "pełna analiza",
    "pelna analiza",
    "artykuł po artykule",
    "artykul po artykule",
    "rozwiń",
    "rozwin",
    "dokładnie",
    "dokladnie",
)

SPECIAL_TOPIC_KEYWORDS = (
    "dozór urządzeń",
    "dozor urzadzen",
    "ochrona osób",
    "ochrona osob",
    "straż pożarn",
    "straz pozarn",
    "ratownictw",
    "pilnowanie mienia",
    "pogotowiu do pracy",
)

# Art. numer -> domain_relevance (0-100) per intent
EQUAL_TIME_ARTICLE_SCORES: dict[int, int] = {
    135: 100,
    129: 80,
    150: 60,
    136: 20,
    137: 10,
}

LEAVE_DEMAND_ARTICLE_SCORES: dict[int, int] = {
    167: 100,  # 167² parsed as base 167 with superscript handled separately
}

LEAVE_WAIVER_ARTICLE_SCORES: dict[int, int] = {
    152: 100,
}

TERMINATION_ARTICLE_SCORES: dict[int, int] = {
    36: 100,
    361: 70,
    362: 50,
}


@dataclass(frozen=True)
class QueryContext:
    tags: frozenset[str]
    detail_mode: bool


@dataclass(frozen=True)
class RankedCandidate:
    hit_id: str
    payload: dict[str, Any]
    semantic_score: float
    domain_score: float
    final_score: float


def wants_detail_analysis(query: str) -> bool:
    lowered = query.lower()
    return any(hint in lowered for hint in DETAIL_ANALYSIS_HINTS)


def detect_query_context(query: str) -> QueryContext:
    lowered = query.lower()
    tags: set[str] = set()
    if any(h in lowered for h in PRODUCTION_HINTS):
        tags.add("production")
    if any(h in lowered for h in EQUAL_TIME_HINTS):
        tags.add("equal_time")
    if any(h in lowered for h in SPECIAL_ROLE_HINTS):
        tags.add("special_role")
    if any(h in lowered for h in LEAVE_DEMAND_HINTS):
        tags.add("leave_on_demand")
    if "czas pracy" in lowered or "nadgodzin" in lowered:
        tags.add("working_time")
    if "urlop" in lowered:
        tags.add("leave")
    if any(h in lowered for h in LEAVE_WAIVER_HINTS):
        tags.add("leave_waiver")
    if any(h in lowered for h in TERMINATION_HINTS):
        tags.add("termination")
    return QueryContext(tags=frozenset(tags), detail_mode=wants_detail_analysis(query))


def extract_article_number(article: str) -> int | None:
    match = ARTICLE_NUM_RE.search(article)
    if not match:
        return None
    return int(match.group(1))


def normalize_article_key(article: str) -> str:
    """Klucz do deduplikacji — bez części (cz. 2/3)."""
    base = re.sub(r"\s*\(cz\.\s*\d+.*?\)", "", article, flags=re.I).strip().lower()
    if "167²" in article or "1672" in article.replace(" ", ""):
        if "na" in base or "żądanie" in base or "zadanie" in base:
            return "art_1672_demand"
    num = extract_article_number(article)
    if num is None:
        return base
    return f"art_{num}"


def _is_leave_on_demand_article(article: str, text: str, topic: str) -> bool:
    blob = f"{article} {text} {topic}".lower()
    return "na żądanie" in blob or "na zadanie" in blob or "167²" in article or "1672" in article


def _is_art_167_recall_from_vacation(article: str, text: str) -> bool:
    """Art. 167 KP (odwołanie z urlopu) — nie mylić z 167²."""
    art = article.replace(" ", "")
    if "167²" in article or "1672" in art.lower() or "167³" in article:
        return False
    if extract_article_number(article) != 167:
        return False
    blob = text.lower()
    return "odwołać" in blob or "odwołanie" in blob or "odwolac" in blob or "odwolanie" in blob


def _text_suggests_special_role(text: str, topic: str) -> bool:
    blob = f"{text} {topic}".lower()
    return any(kw in blob for kw in SPECIAL_TOPIC_KEYWORDS)


def domain_relevance(query: str, payload: dict[str, Any], ctx: QueryContext) -> float:
    """Zwraca domain_relevance 0–100."""
    article = str(payload.get("article", ""))
    text = str(payload.get("text", ""))
    topic = str(payload.get("topic", ""))
    source = str(payload.get("source", "")).upper()
    art_num = extract_article_number(article)

    # --- wypowiedzenie umowy (Art. 36 KP) ---
    if "termination" in ctx.tags:
        if art_num in TERMINATION_ARTICLE_SCORES:
            return float(TERMINATION_ARTICLE_SCORES[art_num])
        blob = f"{article} {text} {topic}".lower()
        if "wypowiedz" in blob or "okres wypowiedzenia" in blob:
            return 95.0
        if "wlicza" in blob and "staż" in blob:
            return 90.0
        if source == "PIP" and "wypowiedz" in blob:
            return 85.0
        if source == "ISAP" and art_num is not None:
            return 40.0
        return 30.0

    # --- zrzeczenie się urlopu (Art. 152 § 2 KP) ---
    if "leave_waiver" in ctx.tags:
        if art_num in LEAVE_WAIVER_ARTICLE_SCORES:
            return float(LEAVE_WAIVER_ARTICLE_SCORES[art_num])
        blob = f"{article} {text} {topic}".lower()
        if "zrzec" in blob or "nie może zrzec" in blob:
            return 95.0
        if art_num == 167:
            return 15.0
        if "urlop" in blob:
            return 35.0
        return 25.0

    # --- urlop na żądanie ---
    if "leave_on_demand" in ctx.tags or (
        "leave" in ctx.tags and any(h in query.lower() for h in ("na żądanie", "na zadanie", "odmow"))
    ):
        if _is_leave_on_demand_article(article, text, topic):
            return 100.0
        if _is_art_167_recall_from_vacation(article, text):
            return 90.0
        if art_num == 231:
            return 30.0
        if "urlop" in f"{text} {topic}".lower():
            return 50.0
        return 25.0

    # --- system równoważny (+ produkcja bez ról szczególnych) ---
    if "equal_time" in ctx.tags:
        suppress_special = (
            "production" in ctx.tags
            and "special_role" not in ctx.tags
            and not _text_suggests_special_role(text, topic)
        )
        if art_num in EQUAL_TIME_ARTICLE_SCORES:
            score = float(EQUAL_TIME_ARTICLE_SCORES[art_num])
            if suppress_special and art_num in (136, 137):
                return min(score, 10.0)
            return score
        if source == "PIP" and any(k in topic.lower() for k in ("czas pracy", "równoważn", "rownowazn")):
            return 50.0
        if suppress_special and _text_suggests_special_role(text, topic):
            return 10.0
        if "równoważn" in f"{text} {topic}".lower() or "rownowazn" in f"{text} {topic}".lower():
            return 40.0
        return 30.0

    # --- ogólny czas pracy ---
    if "working_time" in ctx.tags and art_num in (129, 130, 131, 132, 133, 134, 135, 150):
        base_scores = {129: 90, 135: 85, 150: 70, 136: 25, 137: 15}
        return float(base_scores.get(art_num, 40))

    # --- PIP jako uzupełnienie ---
    if source == "PIP":
        return 45.0
    if source == "ZUS":
        return 35.0

    # --- domyślnie: treść bliższa pytaniu ---
    query_words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 3}
    text_words = set(re.findall(r"\w+", f"{text} {topic}".lower()))
    overlap = len(query_words & text_words)
    return min(100.0, 25.0 + overlap * 8.0)


def combine_scores(semantic_score: float, domain_score: float) -> float:
    sem = max(0.0, min(1.0, semantic_score))
    dom = max(0.0, min(100.0, domain_score)) / 100.0
    return RAG_SEMANTIC_WEIGHT * sem + RAG_DOMAIN_WEIGHT * dom


def rank_candidates(
    query: str,
    hits: list[Any],
    rerank_results: list[Any],
) -> list[RankedCandidate]:
    ctx = detect_query_context(query)
    by_index = {item.index: float(item.relevance_score) for item in rerank_results}
    ranked: list[RankedCandidate] = []
    for idx, hit in enumerate(hits):
        payload = dict(hit.payload or {})
        semantic = by_index.get(idx, 0.0)
        domain = domain_relevance(query, payload, ctx)
        final = combine_scores(semantic, domain)
        ranked.append(
            RankedCandidate(
                hit_id=str(hit.id),
                payload=payload,
                semantic_score=semantic,
                domain_score=domain,
                final_score=final,
            )
        )
    ranked.sort(key=lambda c: c.final_score, reverse=True)
    return ranked


def _article_priority(candidate: RankedCandidate) -> int:
    """Niższa wartość = bardziej ogólny/primarny artykuł."""
    art_num = extract_article_number(str(candidate.payload.get("article", "")))
    if art_num in EQUAL_TIME_ARTICLE_SCORES:
        inv = {135: 1, 129: 2, 150: 3, 136: 8, 137: 9}
        return inv.get(art_num, 5)
    return 5


def filter_by_relative_score(
    ranked: list[RankedCandidate],
    *,
    min_ratio: float | None = None,
) -> list[RankedCandidate]:
    """Odrzuca dokumenty znacząco słabsze od najlepszego wyniku (domyślnie < 60%)."""
    if not ranked:
        return []
    ratio = min_ratio if min_ratio is not None else RAG_SCORE_RELATIVE_MIN
    best = ranked[0].final_score
    if best <= 0:
        return ranked
    threshold = best * ratio
    filtered = [c for c in ranked if c.final_score >= threshold]
    return filtered if filtered else ranked[:1]


def select_final_candidates(
    ranked: list[RankedCandidate],
    query: str,
    *,
    max_sources: int | None = None,
    max_articles: int | None = None,
    min_score_ratio: float | None = None,
) -> list[RankedCandidate]:
    ctx = detect_query_context(query)
    max_src = max_sources or RAG_MAX_SOURCES
    max_art = max_articles or RAG_MAX_ARTICLES

    ranked = filter_by_relative_score(ranked, min_ratio=min_score_ratio)

    suppress_special_equal_time = (
        "equal_time" in ctx.tags
        and "production" in ctx.tags
        and "special_role" not in ctx.tags
    )

    selected: list[RankedCandidate] = []
    seen_article_keys: set[str] = set()
    kp_article_keys: set[str] = set()

    for candidate in ranked:
        article = str(candidate.payload.get("article", ""))
        source = str(candidate.payload.get("source", "")).upper()
        art_num = extract_article_number(article)
        art_key = normalize_article_key(article)

        if suppress_special_equal_time and art_num in (136, 137):
            continue

        if (
            "equal_time" in ctx.tags
            and "special_role" not in ctx.tags
            and art_num in (136, 137)
            and candidate.domain_score <= 20
        ):
            continue

        if art_key in seen_article_keys:
            continue

        is_kp = source == "ISAP" and art_num is not None
        if is_kp and art_key not in kp_article_keys and len(kp_article_keys) >= max_art:
            continue

        selected.append(candidate)
        seen_article_keys.add(art_key)
        if is_kp:
            kp_article_keys.add(art_key)

        if len(selected) >= max_src:
            break

    # Preferuj bardziej ogólne artykuły przy remisie
    selected.sort(key=lambda c: (-c.final_score, _article_priority(c)))
    return selected[:max_src]
