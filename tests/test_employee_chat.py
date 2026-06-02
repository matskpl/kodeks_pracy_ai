"""Czat: dane kadrowe z profilu — pracownik i admin (Jan Nowak)."""

from auth.employee_chat import (
    employee_mentioned_in_message,
    is_profile_data_question,
    profile_snapshot_answer,
    profile_snapshot_answer_for_chat,
)
from auth.models import AuthUser, EmployeeProfile, UserRole


def _emp() -> EmployeeProfile:
    return EmployeeProfile(
        id="e1",
        imie_nazwisko="Jan Nowak",
        stanowisko="Operator",
        dzial="Produkcja",
        email="jan@example.com",
        staz_lata=8,
        staz_miesiace=0,
        wymiar_etatu=1.0,
        urlop_roczny_dni=26,
        urlop_wykorzystany=14,
        urlop_pozostaly=12,
        urlop_na_zadanie_wykorzystany=1,
        nadgodziny_wykorzystane=42.0,
    )


def test_own_leave_question_detected() -> None:
    assert is_profile_data_question("Ile mam pozostalego urlopu?")
    assert is_profile_data_question("Ile mam pozostałego urlopu?")
    assert is_profile_data_question("Panu Janowi Nowakowi ile zostalo urlopu?")
    assert not is_profile_data_question("Co to jest urlop na zadanie w KP?")


def test_profile_answer_uses_remaining_days() -> None:
    ans = profile_snapshot_answer("Ile mam pozostalego urlopu?", _emp())
    assert ans is not None
    assert "12" in ans
    assert "Masz" in ans
    assert "Pan" not in ans


def test_admin_third_person_jan_nowak() -> None:
    emp = _emp()
    admin = AuthUser(
        username="hr",
        role=UserRole.ADMIN,
        display_name="HR",
        employee_id=None,
    )
    msg = "Ile urlopu zostalo Panu Janowi Nowakowi?"
    assert employee_mentioned_in_message(msg, [emp]) is not None
    ans = profile_snapshot_answer_for_chat(msg, admin, [emp], None)
    assert ans is not None
    assert "Jan Nowak ma 12" in ans
    assert "system KodeksPracy" not in ans
