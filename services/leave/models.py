"""Modele pul urlopowych i wniosków o urlop na żądanie."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class VacationPools(BaseModel):
    """
    Bieżące pule urlopowe pracownika.

    - on_demand_pool_days: pozostała liczba dni urlopu na żądanie (max 4 / rok, Art. 167² KP).
    - total_vacation_pool_hours: pozostałe godziny urlopu wypoczynkowego (Art. 154² KP w systemie równoważnym).
    """

    on_demand_pool_days: float = Field(ge=0, le=4)
    total_vacation_pool_hours: float = Field(ge=0)


class OnDemandLeaveDay(BaseModel):
    """Jeden dzień urlopu na żądanie z zaplanowaną długością zmiany."""

    leave_date: date
    scheduled_hours: float = Field(gt=0, le=24, description="Godziny pracy zaplanowane na ten dzień")

    @field_validator("scheduled_hours")
    @classmethod
    def round_hours_precision(cls, v: float) -> float:
        return round(v, 2)


class OnDemandLeaveRequest(BaseModel):
    """Wniosek o urlop na żądanie (jeden lub wiele dni)."""

    employee_id: str
    days: list[OnDemandLeaveDay] = Field(min_length=1)

    @field_validator("days")
    @classmethod
    def unique_dates(cls, days: list[OnDemandLeaveDay]) -> list[OnDemandLeaveDay]:
        seen: set[date] = set()
        for d in days:
            if d.leave_date in seen:
                raise ValueError("Duplikat daty w wniosku o urlop na żądanie.")
            seen.add(d.leave_date)
        return days


class OnDemandLeaveDeduction(BaseModel):
    """Wynik potrącenia za jeden dzień urlopu na żądanie."""

    leave_date: date
    scheduled_hours: float
    on_demand_days_deducted: float = Field(
        default=1.0,
        description="Zawsze 1.0 dzień z puli urlopu na żądanie (niezależnie od długości zmiany).",
    )
    vacation_hours_deducted: float = Field(
        description="Godziny odjęte z puli urlopu wypoczynkowego (= scheduled_hours).",
    )


class OnDemandLeaveResult(BaseModel):
    """Stan pul po przetworzeniu wniosku."""

    pools_after: VacationPools
    deductions: list[OnDemandLeaveDeduction]
    legal_basis: list[str] = Field(
        default_factory=lambda: [
            "Art. 167² § 1 KP — urlop na żądanie max 4 dni w roku kalendarzowym",
            "Art. 154² KP — wymiar urlopu w godzinach w systemie równoważnym czasu pracy",
        ]
    )


class LeaveProcessingError(Exception):
    """Błąd przetwarzania urlopu."""


class InsufficientOnDemandLeaveError(LeaveProcessingError):
    """Brak wystarczającej puli dni urlopu na żądanie."""


class InsufficientVacationHoursError(LeaveProcessingError):
    """Brak wystarczającej puli godzin urlopu wypoczynkowego."""
