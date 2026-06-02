"""MainSupervisorAgent — router intencji z delegacją do pod-agentów."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger
from pydantic_ai import Agent, RunContext

from agents.calculator import (
    build_calculator_deps,
    build_calculator_prompt,
    calculator_agent,
)
from agents.doc_gen import (
    build_document_deps,
    build_document_prompt,
    document_generator_agent,
)
from agents.legal_rag import chunks_to_sources, run_legal_rag
from agents.llm import router_model
from agents.models import (
    AgentIntent,
    DocumentGenerateOutput,
    DocumentGenerateRequest,
    KalkulatorInput,
    KalkulatorOutput,
    SupervisorRoute,
)
from config import get_settings

SUPERVISOR_SYSTEM_PROMPT = """
Jesteś MainSupervisorAgent — nadrzędnym routerem systemu KodeksPracy AI dla MetalTech Sp. z o.o.

Klasyfikuj intencję użytkownika i deleguj pracę do wyspecjalizowanych pod-agentów:

1. legal_rag — pytania o przepisy KP, interpretację przypadków, SKUTKI prawne w czasie
   (np. „zatrudniony 1 maja, wypowiedzenie 31 października — jaki okres i kiedy koniec umowy?”).
2. calculator — WYŁĄCZNIE gdy użytkownik podaje jawne parametry liczbowe do wyliczenia
   (staż X lat Y mies., wymiar etatu) BEZ pełnego scenariusza datowego; typ: urlop/nadgodziny/wypowiedzenie.
3. document — generowanie pisma HR (wniosek, wypowiedzenie jako dokument).
4. general — WYŁĄCZNIE powitania i pytania „co potrafisz / jak działa system” (bez danych kadrowych).

Zasady:
- Scenariusz z DATAMI (dzień + miesiąc zatrudnienia / wypowiedzenia) → ZAWSZE legal_rag, NIGDY calculator.
- NIE zakładaj stażu 5 lat ani innych liczb — jeśli brak danych liczbowych od użytkownika, to legal_rag.
- „Wypowiedzenie” + pytanie „kiedy rozwiąże się umowa” → legal_rag (analiza Art. 36 KP).
- calculator tylko przy: „ile urlopu przy 8 latach stażu”, „oblicz nadgodziny dla etatu 0.5”.
- „Ile mam / ile ma [imię] pozostałego urlopu?” z kontekstem profilu w wiadomości → NIE general
  (to dane z profilu; wybierz calculator lub legal_rag tylko gdy brak liczb w kontekście — preferuj calculator).
- Wybierz dokładnie jedną intencję; refined_query — pełny kontekst dla pod-agenta.
- Nie odpowiadaj merytorycznie sam — klasyfikuj i deleguj.
""".strip()


@dataclass
class SupervisorDeps:
    company_name: str
    last_route: SupervisorRoute | None = field(default=None)


supervisor_agent = Agent(
    router_model(),
    deps_type=SupervisorDeps,
    output_type=SupervisorRoute,
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
)

GENERAL_AGENT_SYSTEM_PROMPT = """
Jesteś asystentem KodeksPracy AI dla MetalTech — życzliwy ekspert HR, nie infolinia marketingowa.

Zasady:
- Odpowiadaj krótko, po polsku, na „ty” (bez formy „Pan/Pani”, bez listy usług na końcu).
- Jeśli w wiadomości są już dane pracownika (urlop pozostały, staż, nadgodziny) — podaj TYLKO odpowiedź
  na pytanie; NIE doklejaj opisu „system obsługuje 1. 2. 3.”.
- Listę możliwości systemu (RAG, kalkulator, pisma) podawaj WYŁĄCZNIE przy powitaniu lub pytaniu
  „co potrafisz / pomoc / jak działa”.
- Nie powtarzaj danych, które użytkownik już zna, jeśli nie pyta o nie wprost.
""".strip()

general_agent = Agent(
    router_model(),
    output_type=str,
    system_prompt=GENERAL_AGENT_SYSTEM_PROMPT,
)


@supervisor_agent.system_prompt
async def supervisor_context(ctx: RunContext[SupervisorDeps]) -> str:
    return f"Obsługujesz pracowników HR firmy {ctx.deps.company_name} (produkcja, 150 osób, system równoważny)."


@supervisor_agent.tool
async def delegate_legal_rag(ctx: RunContext[SupervisorDeps], query: str) -> str:
    """Deleguje pytanie prawne do LegalRagAgent z bazą Qdrant."""
    logger.info("Supervisor → LegalRagAgent: {}", query[:100])
    text, chunks, meta = await run_legal_rag(query, usage=ctx.usage)
    sources = ", ".join(s.article for s in chunks_to_sources(chunks)) or "brak"
    judge_note = ""
    if meta.judge_enabled and meta.revision_attempts:
        judge_note = f"\n\n[Weryfikator źródeł: poprawiono po {meta.revision_attempts} próbie]"
    elif meta.judge_enabled and not meta.final_accepted:
        judge_note = "\n\n[Weryfikator źródeł: wymaga ręcznej weryfikacji]"
    return f"{text}\n\nŹródła z bazy: {sources}{judge_note}"


@supervisor_agent.tool
async def delegate_calculator(ctx: RunContext[SupervisorDeps], calculation_prompt: str) -> str:
    """Deleguje obliczenia kadrowe do CalculatorAgent."""
    logger.info("Supervisor → CalculatorAgent")
    deps = build_calculator_deps()
    result = await calculator_agent.run(calculation_prompt, deps=deps, usage=ctx.usage)
    output: KalkulatorOutput = result.output
    podstawy = "; ".join(output.podstawa_prawna)
    return (
        f"Urlop: {output.urlop_dni} dni\n"
        f"Wypowiedzenie: {output.wypowiedzenie_miesiace} mies.\n"
        f"Podstawa: {podstawy}\n"
        f"Wyliczenie: {output.wyliczenie_opis}"
    )


@supervisor_agent.tool
async def delegate_document_generator(
    ctx: RunContext[SupervisorDeps],
    document_prompt: str,
) -> str:
    """Deleguje generowanie pisma do DocumentGeneratorAgent."""
    logger.info("Supervisor → DocumentGeneratorAgent")
    deps = build_document_deps()
    result = await document_generator_agent.run(document_prompt, deps=deps, usage=ctx.usage)
    doc: DocumentGenerateOutput = result.output
    return f"{doc.tytul}\n\n{doc.tresc}\n\nPodstawy: {', '.join(doc.podstawy_prawne)}"


def build_supervisor_deps() -> SupervisorDeps:
    return SupervisorDeps(company_name=get_settings().company.nazwa)


async def classify_intent(
    message: str,
    *,
    raw_user_message: str | None = None,
) -> SupervisorRoute:
    """Klasyfikuje intencję użytkownika bez pełnej delegacji."""
    from agents.routing_rules import adjust_supervisor_route

    user_text = raw_user_message or message
    deps = build_supervisor_deps()
    result = await supervisor_agent.run(message, deps=deps)
    route: SupervisorRoute = adjust_supervisor_route(user_text, result.output)
    deps.last_route = route
    logger.info(
        "Supervisor klasyfikacja: intent={} confidence={:.2f}",
        route.intent.value,
        route.confidence,
    )
    return route


AGENT_LABELS: dict[AgentIntent, str] = {
    AgentIntent.LEGAL_RAG: "LegalRagAgent — ekspert prawny RAG",
    AgentIntent.CALCULATOR: "CalculatorAgent — kalkulator kadrowy",
    AgentIntent.DOCUMENT: "DocumentGeneratorAgent — generator pism",
    AgentIntent.GENERAL: "MainSupervisorAgent — asystent ogólny",
}


def build_general_prompt(message: str) -> str:
    return (
        f"Wiadomość użytkownika (może zawierać kontekst profilu pracownika):\n{message}\n\n"
        "Odpowiedz tylko na to pytanie. Bez listy „system obsługuje 1/2/3”, chyba że to wyraźne "
        "powitanie lub pytanie o możliwości aplikacji."
    )


def document_request_from_message(message: str) -> DocumentGenerateRequest:
    """Heurystyczne mapowanie wiadomości czatu na strukturę dokumentu."""
    lowered = message.lower()
    if "wypowiedz" in lowered:
        typ = "wypowiedzenie_umowy"
    elif "nadgodzin" in lowered:
        typ = "potwierdzenie_nadgodzin"
    elif "urlop" in lowered:
        typ = "wniosek_urlop"
    else:
        typ = "informacja_o_urlopie"
    return DocumentGenerateRequest(
        typ_pisma=typ,
        imie_nazwisko="[DO UZUPEŁNIENIA]",
        stanowisko="[DO UZUPEŁNIENIA]",
        data_zdarzenia="[DO UZUPEŁNIENIA]",
        szczegoly=message,
    )


def calculator_input_from_message(message: str) -> KalkulatorInput | None:
    """
    Mapuje wiadomość na KalkulatorInput tylko gdy są jawne parametry liczbowe.
    Zwraca None dla scenariuszy datowych (obsługuje silnik wypowiedzenia / RAG).
    """
    from agents.routing_rules import is_termination_scenario_question
    from services.termination import parse_termination_scenario

    if is_termination_scenario_question(message) or parse_termination_scenario(message):
        return None

    lowered = message.lower()
    if any(
        k in lowered
        for k in (
            "zatrudniony",
            "zatrudniona",
            "wręczył",
            "wreczyl",
            "tego samego roku",
            "kiedy rozwiąże",
            "kiedy konczy",
        )
    ) and re.search(r"\d{1,2}\s+(stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|września|października|pazdziernika|listopada|grudnia)", lowered):
        return None

    if "wypowiedz" in lowered or "okres wypowiedzenia" in lowered:
        typ = "wypowiedzenie_umowy"
    elif "nadgodzin" in lowered:
        typ = "nadgodziny"
    elif "urlop" in lowered:
        typ = "urlop_wypoczynkowy"
    else:
        typ = "urlop_wypoczynkowy"

    staz_lata = 0
    staz_miesiace = 0
    m_lat = re.search(r"(\d+)\s*lat(?:a|y)?", lowered)
    m_mies = re.search(r"(\d+)\s*mies", lowered)
    if m_lat:
        staz_lata = int(m_lat.group(1))
    if m_mies:
        staz_miesiace = int(m_mies.group(1))

    if staz_lata == 0 and staz_miesiace == 0 and typ == "wypowiedzenie_umowy":
        return None

    return KalkulatorInput(
        typ_obliczenia=typ,
        staz_lata=staz_lata,
        staz_miesiace=staz_miesiace,
        dodatkowe_info=message,
    )
