"""DocumentGeneratorAgent — prompt chaining do generowania pism kadrowych MetalTech."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from loguru import logger
from pydantic_ai import Agent, RunContext

from agents.llm import document_model
from agents.models import DocumentGenerateOutput, DocumentGenerateRequest
from config import get_settings

DOCUMENT_SYSTEM_PROMPT = """
Jesteś DocumentGeneratorAgent w systemie KodeksPracy AI.

Generujesz formalne pisma kadrowe dla firmy MetalTech Sp. z o.o. (produkcja metalowa, 150 pracowników, system równoważny czasu pracy).

ZASADY:
1. Pismo musi być gotowe do użycia — poprawny język urzędowy, pełna struktura.
2. Używaj danych firmy z kontekstu (nagłówek, podpis, NIP, adres).
3. Podstawy prawne jako lista artykułów [Art. XX KP].
4. Nie wymyślaj faktów spoza danych wejściowych — oznacz brakujące pola jako [DO UZUPEŁNIENIA].
5. Dostosuj ton do typu pisma (wypowiedzenie — formalny, wniosek — neutralny).

Struktura pisma:
- Nagłówek firmy
- Miejscowość i data
- Adresat
- Tytuł pisma
- Treść (akapity numerowane gdy stosowne)
- Podstawa prawna
- Podpis upoważnionej osoby
""".strip()


@dataclass
class DocumentGenDeps:
    company_nazwa: str
    company_nip: str
    company_adres: str
    company_reprezentant: str
    company_email: str
    company_telefon: str
    work_time_system: str


document_generator_agent = Agent(
    document_model(),
    deps_type=DocumentGenDeps,
    output_type=DocumentGenerateOutput,
    system_prompt=DOCUMENT_SYSTEM_PROMPT,
)


@document_generator_agent.system_prompt
async def company_letterhead(ctx: RunContext[DocumentGenDeps]) -> str:
    deps = ctx.deps
    return (
        f"Dane firmy:\n"
        f"- Nazwa: {deps.company_nazwa}\n"
        f"- NIP: {deps.company_nip}\n"
        f"- Adres: {deps.company_adres}\n"
        f"- Reprezentant: {deps.company_reprezentant}\n"
        f"- E-mail kadry: {deps.company_email}\n"
        f"- Telefon: {deps.company_telefon}\n"
        f"- System czasu pracy: {deps.work_time_system}"
    )


def build_document_deps() -> DocumentGenDeps:
    company = get_settings().company
    return DocumentGenDeps(
        company_nazwa=company.nazwa,
        company_nip=company.nip,
        company_adres=company.adres,
        company_reprezentant=company.reprezentant,
        company_email=company.email_kadry,
        company_telefon=company.telefon_kadry,
        work_time_system=company.system_czasu_pracy,
    )


def build_document_prompt(request: DocumentGenerateRequest) -> str:
    logger.info("DocumentGeneratorAgent: generowanie pisma typu {}", request.typ_pisma)
    today = date.today().isoformat()
    return (
        f"Wygeneruj pismo kadrowe.\n"
        f"Typ pisma: {request.typ_pisma}\n"
        f"Pracownik: {request.imie_nazwisko}\n"
        f"Stanowisko: {request.stanowisko}\n"
        f"Data zdarzenia: {request.data_zdarzenia}\n"
        f"Forma wypowiedzenia: {request.forma_wypowiedzenia or 'nie dotyczy'}\n"
        f"Szczegóły: {request.szczegoly}\n"
        f"Dzisiejsza data pisma: {today}\n\n"
        "Krok 1: Zaplanuj strukturę pisma.\n"
        "Krok 2: Napisz treść z podstawami prawnymi.\n"
        "Krok 3: Zweryfikuj zgodność z profilem MetalTech."
    )
