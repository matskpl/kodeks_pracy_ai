"""CalculatorAgent — obliczenia kadrowe ze strukturą KalkulatorOutput."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from pydantic_ai import Agent, RunContext

from agents.llm import calculator_model
from agents.models import KalkulatorInput, KalkulatorOutput
from config import get_settings

CALCULATOR_SYSTEM_PROMPT = """
Jesteś CalculatorAgent w systemie KodeksPracy AI dla MetalTech Sp. z o.o.

Twoje zadanie:
1. Wykonaj precyzyjne obliczenia kadrowe na podstawie parametrów wejściowych.
2. ZAWSZE zwróć w pełni wypełniony obiekt KalkulatorOutput (urlop_dni, wypowiedzenie_miesiace, podstawa_prawna, wyliczenie_opis).
3. Podstawy prawne podawaj jako listę artykułów w formacie [Art. XX KP].
4. W wyliczenie_opis krok po kroku opisz logikę obliczeń po polsku.

Reguły referencyjne (Kodeks pracy):
- Urlop wypoczynkowy: 20 dni (staż <10 lat) lub 26 dni (staż ≥10 lat), proporcjonalnie do wymiaru etatu [Art. 154 KP].
- Wypowiedzenie umowy o pracę na czas nieokreślony [Art. 36 § 1 KP] licz WYŁĄCZNIE od stażu
  u TEGO pracodawcy podanego w danych (staz_lata, staz_miesiace): <6 mies. → 0 mies. wypowiedzenia
  (w wyliczeniu_opis podaj „2 tygodnie”); 6 mies.–3 lata → 1 miesiąc; >3 lat → 3 miesiące.
  NIGDY nie używaj ogólnego stażu zawodowego ani domyślnych 5 lat. Jeśli brak stażu w danych — odmów wyliczenia.
- Umowa na okres próbny: wypowiedzenie 3 dni (≤2 tyg. próby) lub 1 tydzień (>2 tyg.) [Art. 36 § 3 KP].
- Nadgodziny: limit 150h/rok, wynagrodzenie +50% lub czas wolny [Art. 151 KP].
- Jeśli dane dotyczą wyłącznie nadgodzin, ustaw urlop_dni=0 i wypowiedzenie_miesiace=0 z uzasadnieniem.
- Jeśli dane dotyczą wyłącznie urlopu, ustaw wypowiedzenie_miesiace=0 z uzasadnieniem.
- Urlop na żądanie [Art. 167² KP]: max 4 dni w roku kalendarzowym; każdy dział roboczy objęty urlopem
  na żądanie zużywa DOKŁADNIE 1 dzień z tej puli — NIEZALEŻNIE od długości zmiany (4h, 8h, 12h w systemie równoważnym).
  Z puli godzinowej urlopu wypoczynkowego [Art. 154² KP] odejmuje się faktyczne zaplanowane godziny pracy
  w tym dniu (np. 12 h przy zmianie 12-godzinnej). NIGDY nie dziel scheduled_hours przez 8 przy puli na żądanie.
- Wypowiedzenie [Art. 36 § 1 KP]: krócej niż 6 mies. → 2 tygodnie; co najmniej 6 mies. → 1 miesiąc; ≥3 lata → 3 mies.
  Okres wypowiedzenia WLICZA się do stażu — jeśli w trakcie 2-tygodniowego okresu staż osiągnie 6 miesięcy
  (np. zatrudnienie 1 V, wypowiedzenie 31 X), okres wydłuża się do 1 miesiąca. Nie używaj fikcyjnego stażu 5 lat.
  Przy pytaniach z konkretnymi datami w tekście — odmów wyliczenia i wskaż analizę Art. 36.
""".strip()


@dataclass
class CalculatorDeps:
    company_name: str
    work_time_system: str


calculator_agent = Agent(
    calculator_model(),
    deps_type=CalculatorDeps,
    output_type=KalkulatorOutput,
    system_prompt=CALCULATOR_SYSTEM_PROMPT,
)


@calculator_agent.system_prompt
async def metaltech_context(ctx: RunContext[CalculatorDeps]) -> str:
    return (
        f"Firma: {ctx.deps.company_name}. System czasu pracy: {ctx.deps.work_time_system}. "
        "Uwzględnij specyfikę produkcji metalowej przy interpretacji nadgodzin."
    )


def build_calculator_deps() -> CalculatorDeps:
    company = get_settings().company
    return CalculatorDeps(
        company_name=company.nazwa,
        work_time_system=company.system_czasu_pracy,
    )


def build_calculator_prompt(data: KalkulatorInput) -> str:
    logger.info("CalculatorAgent: obliczenie typu {}", data.typ_obliczenia)
    return (
        f"Wykonaj obliczenie kadrowe typu: {data.typ_obliczenia}\n"
        f"Staż: {data.staz_lata} lat {data.staz_miesiace} miesięcy\n"
        f"Wymiar etatu: {data.wymiar_etatu}\n"
        f"Rodzaj umowy: {data.rodzaj_umowy}\n"
        f"Okres próbny (miesiące): {data.okres_probny_miesiace}\n"
        f"Liczba nadgodzin: {data.liczba_nadgodzin}\n"
        f"Okres rozliczeniowy (godziny): {data.okres_rozliczeniowy_godziny}\n"
        f"Dodatkowe informacje: {data.dodatkowe_info or 'brak'}"
    )
