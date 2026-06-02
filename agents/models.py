"""Modele Pydantic dla agentów KodeksPracy AI."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentIntent(str, Enum):
    """Typ intencji rozpoznawanej przez MainSupervisorAgent."""

    LEGAL_RAG = "legal_rag"
    CALCULATOR = "calculator"
    DOCUMENT = "document"
    GENERAL = "general"


class SupervisorRoute(BaseModel):
    """Wynik klasyfikacji intencji przez supervisora."""

    intent: AgentIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    refined_query: str


class KalkulatorInput(BaseModel):
    """Parametry wejściowe kalkulatora kadrowego."""

    employee_id: str | None = Field(
        default=None,
        description="ID pracownika (wymagane dla admina; pracownik — ignorowane, używany własny profil)",
    )
    typ_obliczenia: Literal[
        "urlop_wypoczynkowy",
        "wypowiedzenie_umowy",
        "nadgodziny",
        "urlop_na_zadanie",
    ]
    staz_lata: int = Field(ge=0, description="Staż pracy w latach u pracodawcy")
    staz_miesiace: int = Field(default=0, ge=0, le=11)
    wymiar_etatu: float = Field(default=1.0, gt=0.0, le=1.0)
    rodzaj_umowy: Literal["czas_nieokreslony", "czas_okreslony", "probna"] = "czas_nieokreslony"
    okres_probny_miesiace: int | None = Field(default=None, ge=0, le=3)
    liczba_nadgodzin: float | None = Field(default=None, ge=0)
    okres_rozliczeniowy_godziny: float | None = Field(default=None, ge=0)
    dodatkowe_info: str | None = None


class CalculateRequest(BaseModel):
    """Żądanie z formularza UI — staż i etat biorą się z profilu pracownika."""

    employee_id: str | None = Field(
        default=None,
        description="Admin: ID pracownika; pracownik: pole ignorowane",
    )
    typ_obliczenia: Literal[
        "urlop_wypoczynkowy",
        "wypowiedzenie_umowy",
        "nadgodziny",
        "urlop_na_zadanie",
    ]
    dodatkowe_info: str | None = None
    liczba_nadgodzin: float | None = Field(
        default=None,
        ge=0,
        description="Opcjonalnie nadpisuje liczbę nadgodzin (typ: nadgodziny)",
    )


class KalkulatorOutput(BaseModel):
    """Zwalidowana struktura wyjściowa kalkulatora kadrowego."""

    typ_obliczenia: Literal[
        "urlop_wypoczynkowy",
        "wypowiedzenie_umowy",
        "nadgodziny",
        "urlop_na_zadanie",
    ] | None = None
    wynik_glowny: float | None = Field(
        default=None,
        description="Główna liczba dla wybranego typu obliczenia (UI)",
    )
    wynik_etykieta: str | None = Field(
        default=None,
        description="Etykieta karty wyniku, np. „Pozostały urlop”",
    )
    wynik_jednostka: str | None = Field(
        default=None,
        description="Jednostka: dni, mies., godz.",
    )
    urlop_dni: int
    wypowiedzenie_miesiace: int
    podstawa_prawna: list[str]
    wyliczenie_opis: str


class LegalSource(BaseModel):
    """Źródło prawne z bazy wiedzy."""

    article: str
    topic: str
    excerpt: str
    source: str
    url: str
    relevance: float


class LegalRagResponse(BaseModel):
    """Strukturyzowana odpowiedź agenta prawnego."""

    odpowiedz: str
    zrodla: list[LegalSource]


class DocumentGenerateRequest(BaseModel):
    """Żądanie wygenerowania pisma kadrowego."""

    employee_id: str | None = Field(
        default=None,
        description="ID pracownika (admin); pracownik generuje wyłącznie dla siebie",
    )
    typ_pisma: Literal[
        "wypowiedzenie_umowy",
        "wniosek_urlop",
        "potwierdzenie_nadgodzin",
        "informacja_o_urlopie",
        "wezwanie_do_pracy",
    ]
    imie_nazwisko: str
    stanowisko: str
    data_zdarzenia: str
    szczegoly: str = Field(description="Dodatkowe okoliczności sprawy")
    forma_wypowiedzenia: Literal["z_zachowaniem_okresu", "natychmiastowa", "za_porozumieniem"] | None = None


class DocumentGenerateOutput(BaseModel):
    """Wygenerowane pismo kadrowe."""

    tytul: str
    adresat: str
    data_pisma: str
    tresc: str
    podstawy_prawne: list[str]
    podpis: str


class DocumentGenerateApiResponse(DocumentGenerateOutput):
    """Odpowiedź API z podglądem PDF (base64)."""

    pdf_base64: str = Field(description="Plik PDF zakodowany base64 do podglądu w przeglądarce")


class ChatStreamMeta(BaseModel):
    """Metadane strumienia czatu SSE."""

    agent: AgentIntent
    agent_label: str


class SourceJudgeVerdict(BaseModel):
    """Wynik weryfikacji odpowiedzi względem fragmentów bazy wiedzy."""

    accepted: bool = Field(description="Czy odpowiedź może trafić do użytkownika bez poprawki")
    grounding_score: float = Field(ge=0.0, le=1.0, description="0–1 zgodność ze źródłami")
    issues: list[str] = Field(default_factory=list, description="Konkretne niezgodności")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Tezy lub liczby bez pokrycia w fragmentach",
    )
    revision_instructions: str = Field(
        default="",
        description="Instrukcja dla agenta poprawiającego odpowiedź",
    )
    check_source: Literal["rules", "llm", "rules+llm"] = "rules"


class LegalRagRunMeta(BaseModel):
    """Metadane przebiegu RAG (dla UI / logów)."""

    judge_enabled: bool
    revision_attempts: int = 0
    final_accepted: bool = True
    final_score: float = 1.0
    judge_issues: list[str] = Field(default_factory=list)
