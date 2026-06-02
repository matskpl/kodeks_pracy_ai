"""Mapowanie profilu pracownika na parametry kalkulatora."""

from __future__ import annotations

from typing import Literal

from agents.models import KalkulatorInput, KalkulatorOutput
from auth.models import EmployeeProfile

CalcType = Literal[
    "urlop_wypoczynkowy",
    "wypowiedzenie_umowy",
    "nadgodziny",
    "urlop_na_zadanie",
]


def employee_to_calculator_input(
    employee: EmployeeProfile,
    *,
    typ_obliczenia: CalcType,
    dodatkowe_info: str | None = None,
    liczba_nadgodzin: float | None = None,
) -> KalkulatorInput:
    nadg = liczba_nadgodzin if liczba_nadgodzin is not None else employee.nadgodziny_wykorzystane
    info_parts = [
        f"Pracownik: {employee.imie_nazwisko}, {employee.stanowisko}.",
        f"Pozostały urlop: {employee.urlop_pozostaly} dni (wykorzystano {employee.urlop_wykorzystany}/{employee.urlop_roczny_dni}).",
        f"Nadgodziny w roku: {employee.nadgodziny_wykorzystane}/{employee.nadgodziny_limit_godz} h.",
    ]
    if dodatkowe_info:
        info_parts.append(dodatkowe_info)
    return KalkulatorInput(
        employee_id=employee.id,
        typ_obliczenia=typ_obliczenia,
        staz_lata=employee.staz_lata,
        staz_miesiace=employee.staz_miesiace,
        wymiar_etatu=employee.wymiar_etatu,
        rodzaj_umowy=employee.rodzaj_umowy,  # type: ignore[arg-type]
        liczba_nadgodzin=nadg,
        dodatkowe_info=" ".join(info_parts),
    )


def finalize_calculator_output(
    output: KalkulatorOutput,
    typ_obliczenia: CalcType,
    employee: EmployeeProfile,
) -> KalkulatorOutput:
    """Uzupełnia metrykę główną pod wybrany typ — karta wyniku w UI."""
    if typ_obliczenia == "urlop_wypoczynkowy":
        wartosc = float(employee.urlop_pozostaly)
        etykieta = "Pozostały urlop wypoczynkowy"
        jednostka = "dni"
    elif typ_obliczenia == "wypowiedzenie_umowy":
        wartosc = float(output.wypowiedzenie_miesiace)
        etykieta = "Okres wypowiedzenia"
        jednostka = "mies."
    elif typ_obliczenia == "nadgodziny":
        wartosc = employee.nadgodziny_wykorzystane
        etykieta = "Nadgodziny w roku"
        jednostka = "godz."
    else:
        wartosc = float(max(0, 4 - employee.urlop_na_zadanie_wykorzystany))
        etykieta = "Pozostało urlopu na żądanie"
        jednostka = "dni"

    return output.model_copy(
        update={
            "typ_obliczenia": typ_obliczenia,
            "wynik_glowny": wartosc,
            "wynik_etykieta": etykieta,
            "wynik_jednostka": jednostka,
        }
    )
