"""Testy scenariusza wypowiedzenia (Art. 36 KP)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from services.termination import (
    NoticePeriodKind,
    compute_termination,
    employment_months_completed,
    notice_period_for_tenure,
    parse_termination_scenario,
)

USER_QUESTION = (
    "Pracownik został zatrudniony 1 maja. Pracodawca wręczył mu wypowiedzenie umowy o pracę "
    "31 października tego samego roku. Jaki okres wypowiedzenia obowiązuje w tym przypadku "
    "i kiedy rozwiąże się umowa?"
)


class TestParseScenario:
    def test_parses_may_hire_october_notice(self) -> None:
        scenario = parse_termination_scenario(USER_QUESTION)
        assert scenario is not None
        assert scenario.hire_date.month == 5
        assert scenario.hire_date.day == 1
        assert scenario.notice_date.month == 10
        assert scenario.notice_date.day == 31


class TestMayOctoberCase:
    """PIP: zatrudnienie 1 V, wypowiedzenie 31 X — wydłużenie do 1 miesiąca."""

    def test_tenure_under_six_months_at_notice(self) -> None:
        hire = date(2025, 5, 1)
        notice = date(2025, 10, 31)
        assert employment_months_completed(hire, notice) == 5

    def test_one_month_notice_after_extension(self) -> None:
        scenario = parse_termination_scenario(USER_QUESTION)
        assert scenario is not None
        result = compute_termination(scenario)
        assert result.employment_months_at_notice == 5
        assert result.notice_kind == NoticePeriodKind.MONTHS_1
        assert result.notice_label == "1 miesiąc"
        assert result.notice_extended_during_period is True
        assert "3 miesiące" not in result.notice_label

    def test_contract_end_last_day_of_november(self) -> None:
        scenario = parse_termination_scenario(USER_QUESTION)
        assert scenario is not None
        result = compute_termination(scenario)
        assert result.contract_end_date.month == 11
        assert result.contract_end_date.day == 30

    def test_not_two_weeks_nor_january_hallucination(self) -> None:
        scenario = parse_termination_scenario(USER_QUESTION)
        result = compute_termination(scenario)  # type: ignore[arg-type]
        assert result.notice_kind != NoticePeriodKind.WEEKS_2
        assert not (result.contract_end_date.month == 1 and result.contract_end_date.day == 31)


class TestNoticeTiers:
    @pytest.mark.parametrize(
        "months,expected",
        [
            (5, NoticePeriodKind.WEEKS_2),
            (6, NoticePeriodKind.MONTHS_1),
            (37, NoticePeriodKind.MONTHS_3),
        ],
    )
    def test_art_36_tiers(self, months: int, expected: NoticePeriodKind) -> None:
        kind, _ = notice_period_for_tenure(months)
        assert kind == expected


class TestRoutingRegression:
    def test_calculator_input_returns_none_for_scenario(self) -> None:
        from agents.supervisor import calculator_input_from_message

        assert calculator_input_from_message(USER_QUESTION) is None

    def test_route_adjustment_away_from_calculator(self) -> None:
        from agents.models import AgentIntent, SupervisorRoute
        from agents.routing_rules import adjust_supervisor_route

        wrong = SupervisorRoute(
            intent=AgentIntent.CALCULATOR,
            confidence=1.0,
            reasoning="test",
            refined_query=USER_QUESTION,
        )
        fixed = adjust_supervisor_route(USER_QUESTION, wrong)
        assert fixed.intent == AgentIntent.LEGAL_RAG
