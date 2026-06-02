"""Sędzia źródeł — ocena zgodności odpowiedzi RAG z fragmentami + instrukcja poprawki."""

from __future__ import annotations

import re

from loguru import logger
from pydantic_ai import Agent

from agents.llm import judge_model
from agents.models import SourceJudgeVerdict
from agents.citation_honesty import (
    CITATION_DISCLAIMER_PHRASE,
    chunks_cover_claim,
)
from agents.rag_grounding import find_ungrounded_articles
from config import get_settings
from vector_store import RetrievedChunk

_CHUNK_REF = re.compile(r"\[\s*(\d+)\s*\]")

JUDGE_SYSTEM_PROMPT = """
Jesteś modułem kontroli jakości (sędzią) w systemie KodeksPracy AI.

Oceń, czy odpowiedź asystenta HR jest merytorycznie zgodna z fragmentami [1], [2]… — nie z wiedzą ogólną modelu.

Odrzuć (accepted=false), gdy:
- Cytowane są artykuły KP / tezy sprzeczne z fragmentami.
- Podawane są liczby, okresy lub staż sprzeczne z pytaniem lub fragmentami.
- Brak konkretnej daty końca umowy, choć z dat w pytaniu, fragmentów (Art. 36 § 3) i kalendarza da się ją wyliczyć.
- Nadmierna asekurancta: „w źródłach brakuje daty” przy oczywistym scenariuszu miesięcznym wypowiedzenia.
- W treści głównej wstawione przypisy „wg [1]” (styl urzędniczy) — poproś o przeniesienie podstawy na koniec.
- Fragment zawiera wyjątek („chyba że”, „z wyjątkiem”), a odpowiedź stosuje tylko regułę ogólną bez sprawdzenia statusu z pytania.

Akceptuj (accepted=true), gdy:
- Najpierw konkretna odpowiedź, potem wyjaśnienie; podstawa prawna na końcu i zgodna z TEMATEM fragmentów.
- Zrzeczenie urlopu → Art. 152 (nie Art. 167), chyba że jest wymagana fraza o braku artykułu w materiałach.
- Fakt bez pokrycia w chunkach — tylko z frazą o braku specyficznego artykułu w materiałach źródłowych.
- Daty z kalendarza poprawne, gdy wynikają z fragmentów i dat użytkownika.

Zwróć grounding_score 0.0–1.0, issues i revision_instructions po polsku (ton: co poprawić w stylu HR).
""".strip()


judge_agent = Agent(
    judge_model(),
    output_type=SourceJudgeVerdict,
    system_prompt=JUDGE_SYSTEM_PROMPT,
    model_settings=judge_model().settings,
)


def rule_based_issues(
    answer: str,
    query: str,
    chunks: list[RetrievedChunk],
) -> list[str]:
    """Szybka walidacja deterministyczna przed / obok oceny LLM."""
    issues: list[str] = []
    stripped = answer.strip()

    if not chunks:
        if stripped and "brakuje" not in stripped.lower() and "brak fragment" not in stripped.lower():
            issues.append(
                "Brak fragmentów w bazie, a odpowiedź zawiera treść merytoryczną zamiast odmowy."
            )
        return issues

    extra_arts = find_ungrounded_articles(answer, chunks)
    if extra_arts:
        issues.append(
            f"Cytowane artykuły spoza pobranych źródeł: {', '.join(extra_arts[:6])}."
        )

    if _CHUNK_REF.search(answer):
        issues.append(
            "W treści odpowiedzi są przypisy typu [N] — przenieś podstawę prawną na koniec, bez oznaczeń w środku zdań."
        )

    evasive_date = (
        "brakuje" in stripped.lower()
        and "dat" in stripped.lower()
        and ("wypowiedz" in query.lower() or "koniec" in query.lower() or "rozwiąże" in query.lower())
    )
    if evasive_date and not re.search(r"\d{1,2}\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|pazdziernika|listopada|grudnia)", stripped, re.I):
        if re.search(r"\d{1,2}\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|pazdziernika|listopada|grudnia)", query, re.I):
            issues.append(
                "Unikanie podania konkretnej daty końca umowy mimo dat w pytaniu — należy wyliczyć dzień z kalendarza i fragmentów."
            )

    if re.search(r"zrzecz|zrzeka|zrzec\s", query, re.I):
        cites_167 = re.search(r"Art\.?\s*167\b", answer, re.I)
        has_152 = any(
            re.search(r"Art\.?\s*152", ch.article, re.I)
            or "zrzec" in f"{ch.topic} {ch.text}".lower()
            for ch in chunks
        )
        if cites_167 and not has_152 and CITATION_DISCLAIMER_PHRASE not in answer:
            issues.append(
                "Odpowiedź przypisuje zrzeczenie urlopu do Art. 167, choć fragmenty tego nie potwierdzają — "
                f"wymagana fraza: ({CITATION_DISCLAIMER_PHRASE})."
            )
        if not chunks_cover_claim(query, chunks) and "Art. 152" in answer and CITATION_DISCLAIMER_PHRASE not in answer:
            issues.append(
                "Brak Art. 152 w retrievalu, a odpowiedź cytuje Art. 152 bez zastrzeżenia o braku źródła."
            )

    max_idx = len(chunks)
    for m in _CHUNK_REF.finditer(answer):
        idx = int(m.group(1))
        if idx < 1 or idx > max_idx:
            issues.append(
                f"Odwołanie [{idx}] poza zakresem przekazanych fragmentów (1–{max_idx})."
            )

    return issues


def verdict_from_rules(issues: list[str]) -> SourceJudgeVerdict:
    instructions = (
        "Popraw w stylu HR: konkret na początku, „Dlaczego tak?” po ludzku, podstawa prawna na końcu. "
        "Bez przypisów [N] w środku. Tylko artykuły i tezy ze fragmentów; daty wylicz z kalendarza, gdy to możliwe."
    )
    if issues:
        instructions = f"{issues[0]} {instructions}"
    return SourceJudgeVerdict(
        accepted=False,
        grounding_score=max(0.0, 0.4 - 0.1 * len(issues)),
        issues=issues,
        revision_instructions=instructions,
        check_source="rules",
    )


def build_judge_user_prompt(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    rule_issues: list[str] | None = None,
) -> str:
    from agents.legal_rag import format_chunks_for_prompt

    context = format_chunks_for_prompt(chunks)
    rules_block = ""
    if rule_issues:
        rules_block = (
            "\n\nUwaga — automatyczna kontrola wykryła:\n"
            + "\n".join(f"- {i}" for i in rule_issues)
        )
    return (
        f"Fragmenty z bazy wiedzy:\n\n{context}\n\n"
        f"Pytanie użytkownika: {query}\n\n"
        f"Odpowiedź asystenta do oceny:\n{answer}\n"
        f"{rules_block}\n\n"
        "Oceń zgodność odpowiedzi ze źródłami. Ustaw accepted=true tylko przy wysokiej pewności."
    )


async def evaluate_rag_answer(
    query: str,
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    usage: object | None = None,
) -> SourceJudgeVerdict:
    """Walidacja regułowa + opcjonalnie sędzia LLM."""
    issues = rule_based_issues(answer, query, chunks)
    if issues:
        logger.info("Judge (rules): odrzucono — {} problemów", len(issues))
        return verdict_from_rules(issues)

    settings = get_settings()
    if not settings.google_api_key:
        return SourceJudgeVerdict(
            accepted=True,
            grounding_score=0.85,
            check_source="rules",
            revision_instructions="",
        )

    prompt = build_judge_user_prompt(query, answer, chunks)
    run_kwargs: dict = {}
    if usage is not None:
        run_kwargs["usage"] = usage
    result = await judge_agent.run(prompt, **run_kwargs)
    verdict: SourceJudgeVerdict = result.output
    verdict.check_source = "llm"
    min_score = settings.rag_judge_min_score
    if verdict.grounding_score < min_score:
        verdict.accepted = False
    if verdict.issues and not verdict.revision_instructions:
        verdict.revision_instructions = "; ".join(verdict.issues[:3])
    logger.info(
        "Judge (LLM): accepted={} score={:.2f} issues={}",
        verdict.accepted,
        verdict.grounding_score,
        len(verdict.issues),
    )
    return verdict
