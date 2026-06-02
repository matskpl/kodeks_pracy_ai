"""Metryka główna kalkulatora zależna od typu obliczenia."""

from agents.models import KalkulatorOutput
from auth.employee_calc import finalize_calculator_output
from auth.models import EmployeeProfile


def _employee(**kwargs) -> EmployeeProfile:
    base = dict(
        id="e1",
        imie_nazwisko="Jan Nowak",
        stanowisko="Operator",
        dzial="Produkcja",
        email="jan@example.com",
        staz_lata=8,
        staz_miesiace=0,
        wymiar_etatu=1.0,
        rodzaj_umowy="czas_nieokreslony",
        urlop_roczny_dni=26,
        urlop_wykorzystany=14,
        urlop_pozostaly=12,
        urlop_na_zadanie_wykorzystany=1,
        nadgodziny_limit_godz=150,
        nadgodziny_wykorzystane=42.0,
    )
    base.update(kwargs)
    return EmployeeProfile(**base)


def _output(**kwargs) -> KalkulatorOutput:
    base = dict(
        urlop_dni=0,
        wypowiedzenie_miesiace=0,
        podstawa_prawna=["Art. 154 KP"],
        wyliczenie_opis="opis",
    )
    base.update(kwargs)
    return KalkulatorOutput(**base)


def test_finalize_urlop_wypoczynkowy_uses_profile() -> None:
    emp = _employee()
    out = finalize_calculator_output(_output(urlop_dni=99), "urlop_wypoczynkowy", emp)
    assert out.wynik_glowny == 12.0
    assert out.wynik_jednostka == "dni"


def test_finalize_wypowiedzenie_uses_model() -> None:
    emp = _employee()
    out = finalize_calculator_output(_output(wypowiedzenie_miesiace=3), "wypowiedzenie_umowy", emp)
    assert out.wynik_glowny == 3.0
    assert out.wynik_jednostka == "mies."


def test_finalize_nadgodziny_uses_profile_hours() -> None:
    emp = _employee()
    out = finalize_calculator_output(_output(), "nadgodziny", emp)
    assert out.wynik_glowny == 42.0
    assert out.wynik_jednostka == "godz."


def test_finalize_urlop_na_zadanie_remaining() -> None:
    emp = _employee(urlop_na_zadanie_wykorzystany=1)
    out = finalize_calculator_output(_output(), "urlop_na_zadanie", emp)
    assert out.wynik_glowny == 3.0
