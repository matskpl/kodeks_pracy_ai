"""Reguły routingu — korekta błędnej klasyfikacji LLM."""

from __future__ import annotations

import re

from agents.models import AgentIntent, SupervisorRoute
from auth.employee_chat import is_profile_data_question
from services.termination import parse_termination_scenario

# Scenariusz z datami → RAG / silnik wypowiedzenia, NIE kalkulator z domyślnym stażem
_SCENARIO_DATE = re.compile(
    r"\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
    r"września|października|pazdziernika|listopada|grudnia)",
    re.IGNORECASE,
)


def is_termination_scenario_question(message: str) -> bool:
    return parse_termination_scenario(message) is not None


_CALC_VERBS = ("oblicz", "policz", "wylicz", "przelicz")
_CALC_TOPICS = ("wypowiedz", "okres wypowiedzenia", "urlop", "nadgodzin")
_STAZ_NUM = re.compile(r"(?:sta[zż]|stazu)\s*(\d+)|(\d+)\s*lat(?:a|y)?", re.IGNORECASE)


def is_explicit_calculator_request(message: str) -> bool:
    """
    Jawne polecenie wyliczenia z liczbami (np. „Oblicz okres wypowiedzenia: staż 8 lat”).
    Nie mylić ze scenariuszem datowym → wtedy legal_rag.
    """
    if is_termination_scenario_question(message) or is_scenario_with_dates(message):
        return False
    lowered = message.casefold()
    if not _STAZ_NUM.search(message) and not re.search(r"\d", message):
        return False
    has_topic = any(t in lowered for t in _CALC_TOPICS)
    has_verb = any(v in lowered for v in _CALC_VERBS)
    if has_verb and has_topic:
        return True
    if has_topic and _STAZ_NUM.search(message):
        return True
    return False


def is_scenario_with_dates(message: str) -> bool:
    lowered = message.casefold()
    if not _SCENARIO_DATE.search(lowered):
        return False
    return any(
        k in lowered
        for k in (
            "wypowiedz",
            "zatrudnion",
            "rozwiąże",
            "rozwiaze",
            "okres wypowiedzenia",
            "kiedy kończy",
        )
    )


def adjust_supervisor_route(message: str, route: SupervisorRoute) -> SupervisorRoute:
    """
    Koryguje trasę supervisora — m.in. wypowiedzenie + daty nie idzie do CalculatorAgent.
    """
    if is_explicit_calculator_request(message):
        return route.model_copy(
            update={
                "intent": AgentIntent.CALCULATOR,
                "confidence": max(route.confidence, 0.95),
                "reasoning": (
                    "Jawne parametry liczbowe i polecenie obliczenia — CalculatorAgent, nie RAG."
                ),
            }
        )

    if route.intent == AgentIntent.GENERAL and is_profile_data_question(message):
        return route.model_copy(
            update={
                "intent": AgentIntent.CALCULATOR,
                "confidence": max(route.confidence, 0.9),
                "reasoning": (
                    "Pytanie o własne dane kadrowe (urlop/staż/nadgodziny) — nie ogólny asystent."
                ),
            }
        )

    if route.intent != AgentIntent.CALCULATOR:
        if is_termination_scenario_question(message) and route.intent == AgentIntent.GENERAL:
            return route.model_copy(
                update={
                    "intent": AgentIntent.LEGAL_RAG,
                    "confidence": max(route.confidence, 0.92),
                    "reasoning": (
                        "Scenariusz wypowiedzenia z datami — odpowiedź prawna/RAG, "
                        "nie ogólny asystent."
                    ),
                }
            )
        return route

    if is_termination_scenario_question(message) or is_scenario_with_dates(message):
        return route.model_copy(
            update={
                "intent": AgentIntent.LEGAL_RAG,
                "confidence": max(route.confidence, 0.95),
                "reasoning": (
                    "Pytanie zawiera konkretne daty zatrudnienia/wypowiedzenia — "
                    "wymaga analizy Art. 36 KP (RAG lub silnik dat), "
                    "nie kalkulatora z domyślnymi parametrami stażu."
                ),
            }
        )

    lowered = message.casefold()
    if any(k in lowered for k in ("jaki okres", "kiedy rozwiąże", "kiedy konczy", "ile wynosi okres")):
        if ("wypowiedz" in lowered or "okres wypowiedzenia" in lowered) and not is_explicit_calculator_request(
            message
        ):
            return route.model_copy(
                update={
                    "intent": AgentIntent.LEGAL_RAG,
                    "confidence": max(route.confidence, 0.9),
                    "reasoning": "Pytanie o skutek prawny wypowiedzenia w czasie — LegalRag.",
                }
            )

    return route
