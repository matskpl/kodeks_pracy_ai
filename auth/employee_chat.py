"""Odpowiedzi czatu z profilu pracownika — bez LLM, gdy pytanie dotyczy danych kadrowych."""

from __future__ import annotations

import re
import unicodedata

from auth.access import is_admin
from auth.models import AuthUser, EmployeeProfile

_OWN_LEAVE_RE = re.compile(
    r"(?:ile\s+(?:mam\s+)?(?:zostało|zostalo|pozostało|pozostalo|zostaje)|"
    r"ile\s+mam\s+pozostal\w*\s+urlopu|"
    r"ile\s+urlopu|"
    r"ile\s+(?:ma|zostało|zostalo|pozostało|pozostalo)\s+.*\s+urlopu|"
    r"pozostal\w*\s+urlopu|"
    r"urlopu\s+.*\s+(?:zostało|zostalo|pozostało|pozostalo)|"
    r"mój\s+urlop|moj\s+urlop|"
    r"pozostał\w*\s+urlop|pozostal\w*\s+urlop|"
    r"ile\s+dni\s+urlopu)",
    re.IGNORECASE,
)

_OWN_ON_DEMAND_RE = re.compile(
    r"(?:urlop\s+na\s+żądanie|urlop\s+na\s+zadanie).*(?:ile|pozosta|zostało|zostalo)|"
    r"(?:ile|pozosta).*(?:urlop\s+na\s+żądanie|urlop\s+na\s+zadanie)",
    re.IGNORECASE,
)

_OWN_OVERTIME_RE = re.compile(
    r"(?:ile\s+(?:mam\s+)?|moje\s+|moj\s+|ile\s+ma\s+).*(?:nadgodzin|wykorzystan\w*\s+nadgodzin)|"
    r"pozostał\w*\s+nadgodzin|pozostal\w*\s+nadgodzin",
    re.IGNORECASE,
)

_OWN_SENIORITY_RE = re.compile(
    r"(?:jaki\s+(?:mam\s+)?staż|jaki\s+(?:mam\s+)?staz|ile\s+mam\s+stażu|ile\s+mam\s+stazu|"
    r"jaki\s+ma\s+.*\s+staż|jaki\s+ma\s+.*\s+staz)",
    re.IGNORECASE,
)

_GENERIC_THEORY_RE = re.compile(
    r"(?:co\s+to\s+jest|czym\s+jest|wyjaśnij|wyjasnij|na\s+czym\s+polega|"
    r"przepis|art\.|artykuł|artykul|kodeks)",
    re.IGNORECASE,
)

_SYSTEM_META_RE = re.compile(
    r"(?:co\s+potrafisz|co\s+umiesz|jak\s+działa\s+system|pomoc|pomóż|pomoz|"
    r"możliwości\s+systemu|cześć|czesc|witaj|hello|hi\b)",
    re.IGNORECASE,
)


def _normalize_name(text: str) -> str:
    lowered = text.casefold()
    return "".join(c for c in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(c))


def is_profile_data_question(message: str) -> bool:
    """Pytanie o konkretne dane kadrowe, nie ogólną wykładnię prawa."""
    if _GENERIC_THEORY_RE.search(message):
        return False
    if _SYSTEM_META_RE.search(message) and not _OWN_LEAVE_RE.search(message):
        return False
    return bool(
        _OWN_LEAVE_RE.search(message)
        or _OWN_ON_DEMAND_RE.search(message)
        or _OWN_OVERTIME_RE.search(message)
        or _OWN_SENIORITY_RE.search(message)
    )


def employee_mentioned_in_message(
    message: str,
    employees: list[EmployeeProfile],
) -> EmployeeProfile | None:
    msg_norm = _normalize_name(message)
    best: EmployeeProfile | None = None
    best_len = 0
    for emp in employees:
        name_norm = _normalize_name(emp.imie_nazwisko)
        if len(name_norm) < 5:
            continue
        if name_norm in msg_norm and len(name_norm) > best_len:
            best = emp
            best_len = len(name_norm)
            continue
        parts = name_norm.split()
        if len(parts) >= 2:
            last, first = parts[-1], parts[0]
            if len(last) >= 4 and last[:4] in msg_norm and len(first) >= 3 and first[:3] in msg_norm:
                if len(last) > best_len:
                    best = emp
                    best_len = len(last)
    return best


def resolve_employee_for_profile_answer(
    message: str,
    user: AuthUser,
    employees: list[EmployeeProfile],
    own: EmployeeProfile | None,
) -> EmployeeProfile | None:
    if not is_profile_data_question(message):
        return None
    if is_admin(user):
        mentioned = employee_mentioned_in_message(message, employees)
        if mentioned:
            return mentioned
        return own
    return own


def profile_snapshot_answer(
    message: str,
    employee: EmployeeProfile,
    *,
    third_person: bool = False,
) -> str | None:
    """Krótka odpowiedź HR z danych profilu."""
    if not is_profile_data_question(message):
        return None

    name = employee.imie_nazwisko
    lines: list[str] = []

    if _OWN_LEAVE_RE.search(message):
        if third_person:
            lines.append(
                f"{name} ma {employee.urlop_pozostaly} dni pozostałego urlopu wypoczynkowego "
                f"(wykorzystano {employee.urlop_wykorzystany} z {employee.urlop_roczny_dni} dni rocznie)."
            )
        else:
            lines.append(
                f"Masz {employee.urlop_pozostaly} dni pozostałego urlopu wypoczynkowego "
                f"(wykorzystałeś/aś {employee.urlop_wykorzystany} z {employee.urlop_roczny_dni} dni rocznie)."
            )
        lines.append(
            f"Staż u pracodawcy: {employee.staz_lata} lat {employee.staz_miesiace} mies., "
            f"etat {employee.wymiar_etatu}."
        )

    elif _OWN_ON_DEMAND_RE.search(message):
        left = max(0, 4 - employee.urlop_na_zadanie_wykorzystany)
        if third_person:
            lines.append(
                f"{name} ma jeszcze {left} dni urlopu na żądanie w tym roku "
                f"(wykorzystano {employee.urlop_na_zadanie_wykorzystany} z 4). Podstawa: Art. 167² KP."
            )
        else:
            lines.append(
                f"Z puli urlopu na żądanie zostało Ci {left} dni w tym roku "
                f"(wykorzystano {employee.urlop_na_zadanie_wykorzystany} z 4). Podstawa: Art. 167² KP."
            )

    elif _OWN_OVERTIME_RE.search(message):
        left = max(0.0, float(employee.nadgodziny_limit_godz) - employee.nadgodziny_wykorzystane)
        if third_person:
            lines.append(
                f"{name} wykorzystał/a {employee.nadgodziny_wykorzystane:g} h nadgodzin "
                f"z limitu {employee.nadgodziny_limit_godz} h (pozostało ok. {left:g} h)."
            )
        else:
            lines.append(
                f"W tym roku masz {employee.nadgodziny_wykorzystane:g} h nadgodzin "
                f"z limitu {employee.nadgodziny_limit_godz} h (pozostało ok. {left:g} h)."
            )

    elif _OWN_SENIORITY_RE.search(message):
        if third_person:
            lines.append(
                f"Staż {name} u pracodawcy: {employee.staz_lata} lat "
                f"{employee.staz_miesiace} mies., etat {employee.wymiar_etatu}."
            )
        else:
            lines.append(
                f"Twój staż u pracodawcy: {employee.staz_lata} lat "
                f"{employee.staz_miesiace} mies., etat {employee.wymiar_etatu}."
            )

    if not lines:
        return None
    return "\n\n".join(lines)


def profile_snapshot_answer_for_chat(
    message: str,
    user: AuthUser,
    employees: list[EmployeeProfile],
    own: EmployeeProfile | None,
) -> str | None:
    target = resolve_employee_for_profile_answer(message, user, employees, own)
    if not target:
        return None
    third_person = is_admin(user) and (
        employee_mentioned_in_message(message, employees) is not None
        or (own is not None and target.id != own.id)
    )
    return profile_snapshot_answer(message, target, third_person=third_person)


# Zachowanie wsteczne w testach
is_own_profile_data_question = is_profile_data_question
