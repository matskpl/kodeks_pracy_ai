"""LegalRagAgent — wyszukiwanie Qdrant + hybrydowy rerank + naturalne odpowiedzi prawne."""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
from pydantic_ai import Agent, RunContext

from agents.llm import legal_rag_model
from agents.models import LegalRagRunMeta, LegalSource, SourceJudgeVerdict
from agents.reranking import wants_detail_analysis
from config import RAG_CHUNK_CONTEXT_CHARS, RAG_MAX_ARTICLES, RAG_MAX_SOURCES, RAG_MAX_WORDS, get_settings
from agents.legal_rag_queries import (
    is_leave_167_query,
    is_termination_query,
    leave_167_sub_queries,
    termination_sub_queries,
)
from agents.hr_voice import (
    HR_CALENDAR_RULES,
    HR_EXCEPTIONS_FIRST_RULES,
    HR_VOICE_SYSTEM_RULES,
    TERMINATION_HR_EXAMPLE,
)
from agents.citation_honesty import CITATION_HONESTY_RULES, honesty_context_note
from agents.rag_grounding import (
    apply_grounding_guard,
    format_allowed_articles_block,
    global_grounding_user_rules,
)
from vector_store import RetrievedChunk, search_kodeks

LEGAL_RAG_SYSTEM_PROMPT = """
Jesteś asystentem KodeksPracy AI dla firmy MetalTech Sp. z o.o.

{hr_voice}

{hr_calendar}

{hr_exceptions}

{citation_honesty}

ZASADY MERYTORYCZNE:
1. Korzystaj z fragmentów oznaczonych [ŹRÓDŁO: … | Temat: …] — to Twoja baza; użytkownik widzi tylko Twój tekst.
2. Nie przypisuj tezy do artykułu, którego treść w fragmencie dotyczy innego tematu (np. Art. 167 ≠ zrzeczenie urlopu).
3. Rozróżniaj: Art. 167 KP (odwołanie Z urlopu) vs Art. 167² KP (urlop na żądanie) vs Art. 152 (zrzeczenie urlopu).
4. Przy systemie równoważnym — Art. 135, 129, 150 KP z fragmentów; nie Art. 136/137 bez potrzeby.
5. Nie powołuj się na polityki firmy ani przepisy spoza fragmentów.

LIMITY: max {max_words} słów, max {max_articles} artykułów w podstawie prawnej na końcu,
co najwyżej {max_sources} fragmentów źródłowych.
""".strip()


@dataclass
class LegalRagDeps:
    """Zależności agenta prawnego."""

    company_name: str
    user_query: str = ""
    detail_mode: bool = False
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "Brak fragmentów w bazie wiedzy."
    lines: list[str] = [
        f"Przekazano {len(chunks)} fragmentów (hybrydowe wyszukiwanie + rerank). "
        "Każdy blok ma metadane ŹRÓDŁO i Temat — cytuj tylko gdy treść potwierdza Twoją tezę."
    ]
    for idx, chunk in enumerate(chunks, start=1):
        excerpt = _truncate(chunk.text, RAG_CHUNK_CONTEXT_CHARS)
        lines.append(
            f"\n[{idx}] [ŹRÓDŁO: {chunk.article} | Temat: {chunk.topic}] "
            f"(baza: {chunk.source}, trafność {chunk.score:.2f})\n"
            f"Treść tekstu: {excerpt}"
        )
    return "\n".join(lines)


def build_legal_rag_system_prompt(*, detail_mode: bool = False) -> str:
    max_words = RAG_MAX_WORDS * 2 if detail_mode else RAG_MAX_WORDS
    max_articles = RAG_MAX_ARTICLES + 2 if detail_mode else RAG_MAX_ARTICLES
    return LEGAL_RAG_SYSTEM_PROMPT.format(
        hr_voice=HR_VOICE_SYSTEM_RULES,
        hr_calendar=HR_CALENDAR_RULES,
        hr_exceptions=HR_EXCEPTIONS_FIRST_RULES,
        citation_honesty=CITATION_HONESTY_RULES,
        max_words=max_words,
        max_sources=RAG_MAX_SOURCES,
        max_articles=max_articles,
    )


legal_rag_agent = Agent(
    legal_rag_model(),
    deps_type=LegalRagDeps,
    output_type=str,
    model_settings=legal_rag_model().settings,
)


@legal_rag_agent.tool
async def search_kodeks_pracy(ctx: RunContext[LegalRagDeps], query: str) -> str:
    """Opcjonalne ponowne wyszukiwanie — domyślnie fragmenty są już w wiadomości użytkownika."""
    if ctx.deps.retrieved_chunks:
        return _format_chunks(ctx.deps.retrieved_chunks)
    logger.info("LegalRagAgent: dodatkowe wyszukiwanie — {}", query[:100])
    chunks = search_kodeks(query)
    ctx.deps.retrieved_chunks = chunks
    return _format_chunks(chunks)


@legal_rag_agent.system_prompt
async def dynamic_context(ctx: RunContext[LegalRagDeps]) -> str:
    parts = [
        build_legal_rag_system_prompt(detail_mode=ctx.deps.detail_mode),
        (
            f"Kontekst: {ctx.deps.company_name}, produkcja metalowa, ok. 150 pracowników, "
            "system równoważny czasu pracy."
        ),
    ]
    if ctx.deps.retrieved_chunks:
        parts.append(format_allowed_articles_block(ctx.deps.retrieved_chunks))
        parts.append(
            "Podstawa prawna na końcu odpowiedzi — tylko artykuły z powyższej listy. "
            "W treści głównej bez przypisów [N] i bez urzędniczego stylu."
        )
    if ctx.deps.detail_mode:
        parts.append(
            "Użytkownik prosi o szczegółową analizę — możesz rozwinąć odpowiedź, "
            "ale nadal bez nagłówków Markdown, bez powtarzania treści i bez dopowiedzeń spoza źródeł."
        )
    else:
        parts.append(
            f"Tryb zwięzły: do {RAG_MAX_WORDS} słów, max {RAG_MAX_ARTICLES} artykuły KP, "
            f"max {RAG_MAX_SOURCES} źródeł."
        )
    return "\n\n".join(parts)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    return _format_chunks(chunks)


def _merge_chunks(*chunk_lists: list[RetrievedChunk], limit: int = RAG_MAX_SOURCES) -> list[RetrievedChunk]:
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for batch in chunk_lists:
        for ch in batch:
            if ch.id in seen:
                continue
            seen.add(ch.id)
            merged.append(ch)
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged[:limit]


def retrieve_legal_chunks(query: str) -> list[RetrievedChunk]:
    """Pobiera i rerankuje fragmenty — wypowiedzenie: kilka zapytań pod Art. 36 + PIP."""
    logger.info("LegalRAG: retrieval — {}", query[:100])
    if is_termination_query(query):
        batches = [search_kodeks(sq, top_k=16, rerank_top_n=10) for sq in termination_sub_queries(query)]
        chunks = _merge_chunks(*batches, limit=RAG_MAX_SOURCES + 2)
        logger.info("LegalRAG: wypowiedzenie — scalono {} fragmentów", len(chunks))
        return chunks
    if is_leave_167_query(query):
        batches = [search_kodeks(sq, top_k=14, rerank_top_n=8) for sq in leave_167_sub_queries(query)]
        chunks = _merge_chunks(*batches, limit=RAG_MAX_SOURCES + 2)
        logger.info("LegalRAG: art. 167 / 167² — scalono {} fragmentów", len(chunks))
        return chunks
    return search_kodeks(query)


LEAVE_167_PAIR_RULES = """
Dodatkowe reguły — Art. 167 KP vs Art. 167² KP:
- Art. 167 KP: pracodawca może odwołać pracownika z urlopu (okoliczności nieprzewidziane w chwili rozpoczęcia urlopu); § 2 — koszty odwołania.
- Art. 167² KP: pracownik — urlop na żądanie (max 4 dni/rok), zgłoszenie najpóźniej w dniu rozpoczęcia; pracodawca udziela w wskazanym terminie.
- Gdy pytanie dotyczy wyłącznie urlopu na żądanie — odpowiedz o 167²; Art. 167 wspomnij tylko jednym zdaniem „to inna instytucja”, jeśli jest w fragmentach, bez disclaimeru o braku materiałów.
- Gdy użytkownik pyta o różnicę lub odwołanie — wyjaśnij oba artykuły z fragmentów; nie pisz, że brakuje Art. 167, jeśli fragment [ŹRÓDŁO: Art. 167 KP] jest w kontekście.
""".strip()


TERMINATION_ANSWER_RULES = f"""
Dodatkowe reguły — WYPOWIEDZENIE i data końca umowy:
- Okres z Art. 36 § 1 KP w fragmentach (2 tygodnie / 1 miesiąc / 3 miesiące) + ewentualnie PIP o wliczaniu
  okresu wypowiedzenia do stażu u tego pracodawcy.
- Zawsze podaj konkretny dzień końca umowy, gdy znasz miesiąc wręczenia i bieg okresu (kalendarz + § 3).
- Staż licz tylko u bieżącego pracodawcy z dat w pytaniu.

{TERMINATION_HR_EXAMPLE}
"""


def build_rag_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_chunks(chunks)
    extra_parts: list[str] = []
    if is_termination_query(query):
        extra_parts.append(TERMINATION_ANSWER_RULES)
    if is_leave_167_query(query):
        extra_parts.append(LEAVE_167_PAIR_RULES)
    extra = "\n\n".join(extra_parts)
    if not chunks:
        return (
            f"Fragmenty z bazy wiedzy:\n\nBrak fragmentów.\n\n"
            f"Pytanie użytkownika: {query}\n\n"
            "Nie możesz udzielić odpowiedzi merytorycznej — poinformuj, że w bazie brakuje "
            "odpowiednich przepisów i zaproponuj ponowne zapytanie lub ingest bazy."
        )
    allowed_block = format_allowed_articles_block(chunks)
    retrieval_note = honesty_context_note(query, chunks)
    retrieval_block = f"{retrieval_note}\n\n" if retrieval_note else ""
    return (
        f"Fragmenty z bazy wiedzy:\n\n{context}\n\n"
        f"{allowed_block}\n\n"
        f"{CITATION_HONESTY_RULES}\n\n"
        f"{global_grounding_user_rules()}\n"
        f"{retrieval_block}"
        f"{extra}\n"
        f"Pytanie użytkownika: {query}\n\n"
        "Odpowiedz po polsku, stylem eksperta HR: najpierw konkret (okres, data końca itd.), potem krótkie "
        "„Dlaczego tak?”, na końcu podstawa prawna. Bez przypisów [1]/[2] w środku zdań. "
        "Merytoryka wyłącznie z fragmentów powyżej — daty wylicz z kalendarza, gdy przepisy na to pozwalają."
    )


def build_correction_user_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    draft_answer: str,
    verdict: SourceJudgeVerdict,
) -> str:
    """Prompt do drugiej próby — ten sam RAG, z feedbackiem sędziego."""
    base = build_rag_user_prompt(query, chunks)
    issues = "\n".join(f"- {i}" for i in verdict.issues) or "- (brak listy)"
    hints = verdict.revision_instructions or (
        "Popraw ton (HR, nie urzędnik), usuń nieuzasadnione artykuły i liczby; "
        "dodaj konkretną datę końca, jeśli wynika z kalendarza i fragmentów."
    )
    return (
        f"{base}\n\n"
        "=== POPRAWKA (odrzucona przez weryfikator źródeł) ===\n"
        f"Wykryte problemy:\n{issues}\n\n"
        f"Instrukcja poprawy: {hints}\n\n"
        f"Poprzednia wersja odpowiedzi (NIE kopiuj błędów, przepisz poprawnie):\n{draft_answer}\n\n"
        "Przepisz odpowiedź: konkret na początku, „Dlaczego tak?” po ludzku, podstawa prawna na końcu. "
        "Bez „wg [1]” w tekście. Zachowaj zgodność z fragmentami."
    )


async def _generate_rag_answer(
    query: str,
    chunks: list[RetrievedChunk],
    rag_deps: LegalRagDeps,
    user_prompt: str,
    *,
    usage: object | None = None,
) -> str:
    run_kwargs: dict = {}
    if usage is not None:
        run_kwargs["usage"] = usage
    result = await legal_rag_agent.run(user_prompt, deps=rag_deps, **run_kwargs)
    return result.output


async def run_legal_rag(
    query: str,
    *,
    deps: LegalRagDeps | None = None,
    usage: object | None = None,
    judge: bool | None = None,
) -> tuple[str, list[RetrievedChunk], LegalRagRunMeta]:
    """
    RAG z opcjonalną pętlą sędziego: generacja → ocena → poprawka (max N prób).
    """
    settings = get_settings()
    use_judge = settings.rag_judge_enabled if judge is None else judge
    chunks = deps.retrieved_chunks if deps and deps.retrieved_chunks else retrieve_legal_chunks(query)
    rag_deps = deps or build_legal_rag_deps(user_query=query)
    rag_deps.retrieved_chunks = chunks
    meta = LegalRagRunMeta(judge_enabled=use_judge)

    max_passes = 1 + (settings.rag_judge_max_revisions if use_judge else 0)
    text = ""
    last_verdict: SourceJudgeVerdict | None = None
    user_prompt = build_rag_user_prompt(query, chunks)

    for attempt in range(max_passes):
        text = await _generate_rag_answer(query, chunks, rag_deps, user_prompt, usage=usage)
        if not use_judge:
            meta.final_accepted = True
            break

        from agents.answer_judge import evaluate_rag_answer

        last_verdict = await evaluate_rag_answer(query, text, chunks, usage=usage)
        meta.final_accepted = last_verdict.accepted
        meta.final_score = last_verdict.grounding_score
        meta.judge_issues = list(last_verdict.issues)

        if last_verdict.accepted:
            logger.info("LegalRAG: sędzia zaakceptował odpowiedź (próba {})", attempt + 1)
            break

        if attempt >= max_passes - 1:
            logger.warning(
                "LegalRAG: wyczerpano poprawki ({}), zwracam ostatnią wersję",
                settings.rag_judge_max_revisions,
            )
            text = (
                f"{text.rstrip()}\n\n"
                "Uwaga: odpowiedź nie przeszła w pełni weryfikacji zgodności ze źródłami. "
                "Zweryfikuj u HR przed użyciem w decyzji kadrowej."
            )
            break

        meta.revision_attempts += 1
        logger.info("LegalRAG: sędzia żąda poprawki (próba {})", attempt + 2)
        user_prompt = build_correction_user_prompt(query, chunks, text, last_verdict)

    text = apply_grounding_guard(text, chunks)
    return text, chunks, meta


def build_legal_rag_deps(
    user_query: str = "",
    *,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> LegalRagDeps:
    settings = get_settings()
    return LegalRagDeps(
        company_name=settings.company.nazwa,
        user_query=user_query,
        detail_mode=wants_detail_analysis(user_query),
        retrieved_chunks=list(retrieved_chunks or []),
    )


def split_text_for_sse(text: str, *, chunk_size: int = 48) -> list[str]:
    """Dzieli gotową odpowiedź na kawałki pod streaming SSE (po weryfikacji sędziego)."""
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunks_to_sources(chunks: list[RetrievedChunk]) -> list[LegalSource]:
    return [
        LegalSource(
            article=chunk.article,
            topic=chunk.topic,
            excerpt=chunk.text[:220],
            source=chunk.source,
            url=chunk.url,
            relevance=chunk.score,
        )
        for chunk in chunks
    ]
