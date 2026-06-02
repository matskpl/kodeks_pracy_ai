"""Modele użytkowników i pracowników."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    EMPLOYEE = "employee"
    ADMIN = "admin"


class EmployeeProfile(BaseModel):
    """Dane kadrowe pracownika MetalTech."""

    id: str
    imie_nazwisko: str
    stanowisko: str
    dzial: str
    email: str
    staz_lata: int = Field(ge=0)
    staz_miesiace: int = Field(default=0, ge=0, le=11)
    wymiar_etatu: float = Field(default=1.0, gt=0.0, le=1.0)
    rodzaj_umowy: str = "czas_nieokreslony"
    urlop_roczny_dni: int = Field(ge=0)
    urlop_wykorzystany: int = Field(ge=0)
    urlop_pozostaly: int = Field(ge=0)
    urlop_na_zadanie_wykorzystany: int = Field(default=0, ge=0, le=4)
    nadgodziny_limit_godz: int = 150
    nadgodziny_wykorzystane: float = Field(default=0.0, ge=0)


class UserAccount(BaseModel):
    username: str
    password_hash: str
    role: UserRole
    display_name: str
    employee_id: str | None = None


class AuthUser(BaseModel):
    """Zalogowany użytkownik (bez hasła)."""

    username: str
    role: UserRole
    display_name: str
    employee_id: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=4, max_length=128)


class MeResponse(BaseModel):
    user: AuthUser
    employee: EmployeeProfile | None = None


class EmployeePublic(BaseModel):
    """Lista pracowników — skrócone dane (admin)."""

    id: str
    imie_nazwisko: str
    stanowisko: str
    dzial: str
    urlop_pozostaly: int
    wymiar_etatu: float


class EmployeeDetailResponse(BaseModel):
    employee: EmployeeProfile
