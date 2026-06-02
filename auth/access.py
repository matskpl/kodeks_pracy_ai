"""Reguły dostępu do danych pracowników."""

from __future__ import annotations

import re
import unicodedata

from fastapi import HTTPException

from auth.models import AuthUser, EmployeeProfile, UserRole


def is_admin(user: AuthUser) -> bool:
    return user.role == UserRole.ADMIN


def can_view_employee(user: AuthUser, employee_id: str) -> bool:
    if is_admin(user):
        return True
    return user.employee_id == employee_id


def require_employee_access(user: AuthUser, employee_id: str) -> None:
    if not can_view_employee(user, employee_id):
        raise HTTPException(
            status_code=403,
            detail="Brak uprawnień — widzisz tylko własne dane kadrowe.",
        )


def resolve_target_employee_id(
    user: AuthUser,
    requested_id: str | None,
) -> str:
    if is_admin(user):
        if not requested_id:
            raise HTTPException(status_code=400, detail="Admin musi wskazać pracownika (employee_id).")
        return requested_id
    if not user.employee_id:
        raise HTTPException(status_code=403, detail="Konto bez przypisanego profilu pracownika.")
    if requested_id and requested_id != user.employee_id:
        raise HTTPException(status_code=403, detail="Nie możesz przeglądać danych innych pracowników.")
    return user.employee_id


def _normalize_name(text: str) -> str:
    lowered = text.casefold()
    return "".join(c for c in unicodedata.normalize("NFKD", lowered) if not unicodedata.combining(c))


def assert_chat_scope(user: AuthUser, message: str, employees: list[EmployeeProfile]) -> None:
    """Blokuje pytania pracownika o innych pracowników (po imieniu/nazwisku)."""
    if is_admin(user):
        return
    if not user.employee_id:
        raise HTTPException(status_code=403, detail="Brak profilu pracownika.")
    own = next((e for e in employees if e.id == user.employee_id), None)
    if not own:
        return
    own_norm = _normalize_name(own.imie_nazwisko)
    msg_norm = _normalize_name(message)
    for emp in employees:
        if emp.id == user.employee_id:
            continue
        name_norm = _normalize_name(emp.imie_nazwisko)
        if len(name_norm) < 5:
            continue
        if name_norm in msg_norm:
            raise HTTPException(
                status_code=403,
                detail=f"Możesz pytać tylko o swoje dane kadrowe, nie o {emp.imie_nazwisko}.",
            )
    # Krótkie imiona w treści (np. "ile ma urlopu Jan?")
    parts = re.split(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", message, flags=re.IGNORECASE)
    for token in parts:
        if len(token) < 4:
            continue
        t = _normalize_name(token)
        if t in own_norm.split():
            continue
        for emp in employees:
            if emp.id == user.employee_id:
                continue
            for word in _normalize_name(emp.imie_nazwisko).split():
                if len(word) >= 4 and word == t:
                    raise HTTPException(
                        status_code=403,
                        detail="Możesz pytać tylko o własne informacje kadrowe.",
                    )


def build_user_context(user: AuthUser, employee: EmployeeProfile | None) -> str:
    if is_admin(user):
        return (
            f"Zalogowany użytkownik: {user.display_name} (kadry / administrator). "
            "Masz dostęp do danych wszystkich pracowników MetalTech."
        )
    if not employee:
        return f"Zalogowany pracownik: {user.display_name}."
    return (
        f"Zalogowany pracownik: {employee.imie_nazwisko}, {employee.stanowisko}, {employee.dzial}. "
        f"Staż: {employee.staz_lata} lat {employee.staz_miesiace} mies. Etat: {employee.wymiar_etatu}. "
        f"Urlop roczny: {employee.urlop_roczny_dni} dni, wykorzystano: {employee.urlop_wykorzystany}, "
        f"pozostało: {employee.urlop_pozostaly} dni. Urlop na żądanie w tym roku: "
        f"{employee.urlop_na_zadanie_wykorzystany}/4. "
        f"Nadgodziny: {employee.nadgodziny_wykorzystane}/{employee.nadgodziny_limit_godz} h. "
        "Odpowiadaj wyłącznie w kontekście TEGO pracownika — nie podawaj danych innych osób."
    )


EMPLOYEE_ALLOWED_DOCUMENT_TYPES = frozenset(
    {"wniosek_urlop", "informacja_o_urlopie", "potwierdzenie_nadgodzin"}
)


def assert_document_allowed(user: AuthUser, typ_pisma: str) -> None:
    if is_admin(user):
        return
    if typ_pisma not in EMPLOYEE_ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(
            status_code=403,
            detail="To pismo może wygenerować tylko dział kadr (administrator).",
        )
