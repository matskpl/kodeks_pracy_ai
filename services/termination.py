"""
Obliczenia okresu wypowiedzenia i daty rozwiązania umowy (Art. 36 KP).

Scenariusze z datami w tekście (np. zatrudnienie 1 maja, wypowiedzenie 31 października)
nie mogą trafiać do CalculatorAgent z domyślnym stażem — stąd deterministyczna logika.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

# Mapowanie form dopełniacza / mianownika
_MONTH_BY_NAME: dict[str, int] = {
    "stycznia": 1,
    "stycznia": 1,
    "styczen": 1,
    "lutego": 2,
    "lutym": 2,
    "marca": 3,
    "marcu": 3,
    "kwietnia": 4,
    "kwietniu": 4,
    "maja": 5,
    "maju": 5,
    "czerwca": 6,
    "czerwcu": 6,
    "lipca": 7,
    "lipcu": 7,
    "sierpnia": 8,
    "sierpniu": 8,
    "września": 9,
    "wrześniu": 9,
    "października": 10,
    "pazdziernika": 10,
    "październiku": 10,
    "pazdzierniku": 10,
    "listopada": 11,
    "listopadzie": 11,
    "grudnia": 12,
    "grudniu": 12,
}


class NoticePeriodKind(str, Enum):
    WEEKS_2 = "2_tygodnie"
    MONTHS_1 = "1_miesiac"
    MONTHS_3 = "3_miesiace"


@dataclass(frozen=True)
class TerminationScenario:
    hire_date: date
    notice_date: date
    contract_type: str = "czas_nieokreslony"


@dataclass(frozen=True)
class TerminationResult:
    scenario: TerminationScenario
    employment_months_at_notice: int
    notice_kind: NoticePeriodKind
    notice_label: str
    contract_end_date: date
    notice_extended_during_period: bool
    legal_basis: list[str]
    explanation: str

    @property
    def employment_months_completed(self) -> int:
        """Alias zachowany dla kompatybilności."""
        return self.employment_months_at_notice


def _normalize(text: str) -> str:
    t = text.casefold().replace("ą", "a").replace("ć", "c").replace("ę", "e")
    t = t.replace("ł", "l").replace("ń", "n").replace("ó", "o").replace("ś", "s")
    t = t.replace("ź", "z").replace("ż", "z")
    return t


def _parse_day_month(text: str, default_year: int | None = None) -> date | None:
    """Parsuje „1 maja”, „31 października”, opcjonalnie z rokiem."""
    lowered = text.casefold()
    m = re.search(
        r"(\d{1,2})\s+([a-ząćęłńóśźż]+)(?:\s+(\d{4}))?",
        lowered,
        re.IGNORECASE,
    )
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2)
    year = int(m.group(3)) if m.group(3) else default_year
    if year is None:
        year = date.today().year
    month = _MONTH_BY_NAME.get(month_name)
    if not month:
        for key, num in _MONTH_BY_NAME.items():
            if _normalize(key) == _normalize(month_name):
                month = num
                break
    if not month:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _extract_year_hint(text: str) -> int | None:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def parse_termination_scenario(message: str) -> TerminationScenario | None:
    """
    Wykrywa scenariusz wypowiedzenia z datami zatrudnienia i wręczenia wypowiedzenia.
    """
    lowered = message.casefold()
    if not any(
        k in lowered
        for k in (
            "wypowiedz",
            "okres wypowiedzenia",
            "rozwiąże",
            "rozwiaze",
            "kończy się umow",
            "konczy sie umow",
        )
    ):
        return None

    year_hint = _extract_year_hint(message)
    same_year = "tego samego roku" in lowered or "w tym roku" in lowered

    hire_date: date | None = None
    notice_date: date | None = None

    hire_patterns = [
        r"zatrudniony[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"nawiązania stosunku pracy[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"rozpoczęcia pracy[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"od\s+(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
    ]
    for pat in hire_patterns:
        m = re.search(pat, lowered, re.IGNORECASE)
        if m:
            hire_date = _parse_day_month(m.group(1), default_year=year_hint)
            break

    notice_patterns = [
        r"wypowiedzenie[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"wręczył[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"wreczyl[^\d]*(\d{1,2}\s+[a-ząćęłńóśźż]+(?:\s+\d{4})?)",
        r"(\d{1,2}\s+października(?:\s+\d{4})?)",
        r"(\d{1,2}\s+pazdziernika(?:\s+\d{4})?)",
    ]
    for pat in notice_patterns:
        m = re.search(pat, lowered, re.IGNORECASE)
        if m:
            notice_date = _parse_day_month(m.group(1), default_year=year_hint)
            if notice_date:
                break

    if not hire_date or not notice_date:
        return None

    if same_year and hire_date and notice_date and not year_hint:
        y = notice_date.year
        hire_date = hire_date.replace(year=y)
        if hire_date > notice_date:
            hire_date = hire_date.replace(year=y - 1)

    if notice_date < hire_date:
        return None

    return TerminationScenario(hire_date=hire_date, notice_date=notice_date)


def employment_months_completed(hire_date: date, as_of: date) -> int:
    """Pełne miesiące zatrudnienia u tego pracodawcy (dzień w dzień)."""
    months = (as_of.year - hire_date.year) * 12 + (as_of.month - hire_date.month)
    if as_of.day < hire_date.day:
        months -= 1
    return max(0, months)


def notice_period_for_tenure(months: int) -> tuple[NoticePeriodKind, str]:
    """
    Art. 36 § 1 KP (DU 2025) — umowa na czas nieokreślony / określony.
    Krócej niż 6 miesięcy: 2 tygodnie; co najmniej 6 miesięcy: 1 miesiąc; co najmniej 3 lata: 3 miesiące.
    """
    if months < 6:
        return NoticePeriodKind.WEEKS_2, "2 tygodnie"
    if months < 36:
        return NoticePeriodKind.MONTHS_1, "1 miesiąc"
    return NoticePeriodKind.MONTHS_3, "3 miesiące"


def _notice_period_start_monthly(notice_date: date) -> date:
    """Pierwszy dzień miesiąca kalendarzowego następującego po miesiącu wręczenia wypowiedzenia."""
    if notice_date.month == 12:
        return date(notice_date.year + 1, 1, 1)
    return date(notice_date.year, notice_date.month + 1, 1)


def contract_end_date_weeks(notice_date: date) -> date:
    """Art. 36 § 1 pkt 1 — 2 tygodnie (uproszczenie: 14 dni kalendarzowych)."""
    return notice_date + timedelta(days=14)


def contract_end_date_monthly(notice_date: date, notice_months: int) -> date:
    """
    Art. 36 § 3 KP — wypowiedzenie w miesiącach kończy się ostatnim dniem miesiąca kalendarzowego.
    Bieg od 1. dnia miesiąca następującego po wręczeniu wypowiedzenia.
    """
    period_start = _notice_period_start_monthly(notice_date)
    end_anchor = _add_months(period_start, notice_months - 1)
    return _last_day_of_month(end_anchor)


def resolve_notice_with_extension(hire_date: date, notice_date: date) -> tuple[NoticePeriodKind, str, bool]:
    """
    Ustala okres wypowiedzenia z uwzględnieniem wliczenia okresu wypowiedzenia do stażu (PIP / praktyka HR).

    Przykład: zatrudnienie 1 V, wypowiedzenie 31 X — w dniu wręczenia <6 mies., ale koniec stosunku pracy
    nastąpi po 6 miesiącach → okres wydłuża się z 2 tygodni do 1 miesiąca (Art. 36 § 1 pkt 2).
    """
    months_at_notice = employment_months_completed(hire_date, notice_date)
    kind, label = notice_period_for_tenure(months_at_notice)
    extended = False

    if kind == NoticePeriodKind.WEEKS_2:
        tentative_end = contract_end_date_weeks(notice_date)
        months_at_end = employment_months_completed(hire_date, tentative_end)
        if months_at_end >= 6:
            kind = NoticePeriodKind.MONTHS_1
            label = "1 miesiąc"
            extended = True

    return kind, label, extended


def _last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def _add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    day = min(d.day, _last_day_of_month(date(y, m, 1)).day)
    return date(y, m, day)


def contract_end_date(notice_date: date, kind: NoticePeriodKind) -> date:
    if kind == NoticePeriodKind.WEEKS_2:
        return contract_end_date_weeks(notice_date)
    notice_months = 1 if kind == NoticePeriodKind.MONTHS_1 else 3
    return contract_end_date_monthly(notice_date, notice_months)


def compute_termination(scenario: TerminationScenario) -> TerminationResult:
    months_at_notice = employment_months_completed(scenario.hire_date, scenario.notice_date)
    kind, label, extended = resolve_notice_with_extension(
        scenario.hire_date, scenario.notice_date
    )
    end = contract_end_date(scenario.notice_date, kind)

    legal_basis: list[str] = []
    if extended:
        legal_basis.append(
            "Art. 36 § 1 pkt 1 i pkt 2 KP — w dniu wręczenia staż <6 mies., "
            "lecz okres wypowiedzenia wlicza się do stażu; koniec stosunku pracy po 6 miesiącach → 1 miesiąc"
        )
    elif kind == NoticePeriodKind.WEEKS_2:
        legal_basis.append("Art. 36 § 1 pkt 1 KP (zatrudnienie krócej niż 6 miesięcy)")
    elif kind == NoticePeriodKind.MONTHS_1:
        legal_basis.append("Art. 36 § 1 pkt 2 KP (zatrudnienie co najmniej 6 miesięcy)")
    else:
        legal_basis.append("Art. 36 § 1 pkt 3 KP (zatrudnienie co najmniej 3 lata)")
    legal_basis.append("Art. 36 § 3 KP — bieg okresu wypowiedzenia w miesiącach")

    extension_note = ""
    if extended:
        extension_note = (
            f" W dniu wręczenia wypowiedzenia staż wynosił {months_at_notice} pełnych miesięcy "
            f"(poniżej progu 6 miesięcy), jednak zgodnie z praktyką PIP okres wypowiedzenia "
            f"wlicza się do stażu zakładowego — rozwiązanie umowy nastąpi po upływie 6 miesięcy pracy, "
            f"więc obowiązuje wydłużenie do 1 miesiąca zamiast 2 tygodni."
        )

    explanation = (
        f"Zatrudnienie od {scenario.hire_date.strftime('%d.%m.%Y')}, "
        f"wypowiedzenie wręczone {scenario.notice_date.strftime('%d.%m.%Y')}. "
        f"Staż u pracodawcy w dniu wręczenia: {months_at_notice} pełnych miesięcy.{extension_note} "
        f"Okres wypowiedzenia: {label}. "
        f"Umowa rozwiązuje się z dniem {end.strftime('%d.%m.%Y')} "
        f"(miesięczny okres kończy się ostatnim dniem miesiąca kalendarzowego)."
    )

    return TerminationResult(
        scenario=scenario,
        employment_months_at_notice=months_at_notice,
        notice_kind=kind,
        notice_label=label,
        contract_end_date=end,
        notice_extended_during_period=extended,
        legal_basis=legal_basis,
        explanation=explanation,
    )


def format_termination_answer(result: TerminationResult) -> str:
    return (
        f"Okres wypowiedzenia: {result.notice_label}.\n"
        f"Data rozwiązania umowy: {result.contract_end_date.strftime('%d.%m.%Y')}.\n"
        f"Podstawa prawna: {', '.join(result.legal_basis)}.\n\n"
        f"{result.explanation}"
    )
