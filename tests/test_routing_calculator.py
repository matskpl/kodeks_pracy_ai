"""Routing: jawne „Oblicz … staż X lat” → kalkulator."""

from agents.models import AgentIntent, SupervisorRoute
from agents.routing_rules import adjust_supervisor_route, is_explicit_calculator_request
from agents.supervisor import calculator_input_from_message

MSG = "Oblicz okres wypowiedzenia: staż 8 lat, umowa na czas nieokreślony"


def test_calculator_input_parses_staz_8() -> None:
    inp = calculator_input_from_message(MSG)
    assert inp is not None
    assert inp.typ_obliczenia == "wypowiedzenie_umowy"
    assert inp.staz_lata == 8


def test_explicit_calculator_request_detected() -> None:
    assert is_explicit_calculator_request(MSG) is True


def test_rag_route_overridden_to_calculator() -> None:
    route = SupervisorRoute(
        intent=AgentIntent.LEGAL_RAG,
        confidence=0.99,
        reasoning="test",
        refined_query=MSG,
    )
    fixed = adjust_supervisor_route(MSG, route)
    assert fixed.intent == AgentIntent.CALCULATOR
