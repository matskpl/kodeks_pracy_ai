"""Testy urlopu na żądanie — system równoważny (Art. 167², 154² KP)."""

from __future__ import annotations

from datetime import date

import pytest

from services.leave import (
    ON_DEMAND_ANNUAL_LIMIT_DAYS,
    InsufficientOnDemandLeaveError,
    InsufficientVacationHoursError,
    OnDemandLeaveDay,
    OnDemandLeaveRequest,
    VacationPools,
    process_on_demand_leave_request,
    process_on_demand_leave_request_legacy_wrong,
)

EMP_ID = "emp-001"


def _request(scheduled_hours: float, leave_date: date | None = None) -> OnDemandLeaveRequest:
    return OnDemandLeaveRequest(
        employee_id=EMP_ID,
        days=[
            OnDemandLeaveDay(
                leave_date=leave_date or date(2026, 6, 15),
                scheduled_hours=scheduled_hours,
            )
        ],
    )


def _pools(on_demand_days: float = 4.0, vacation_hours: float = 200.0) -> VacationPools:
    return VacationPools(
        on_demand_pool_days=on_demand_days,
        total_vacation_pool_hours=vacation_hours,
    )


class TestOnDemandDayDeductionAlwaysOne:
    """Potrącenie z puli urlopu na żądanie = zawsze 1.0 dzień, niezależnie od zmiany."""

    @pytest.mark.parametrize(
        "scheduled_hours",
        [8.0, 12.0, 4.0],
        ids=["shift_8h", "shift_12h_equivalent", "shift_4h"],
    )
    def test_on_demand_pool_decreases_by_exactly_one_day(self, scheduled_hours: float) -> None:
        pools = _pools(on_demand_days=4.0, vacation_hours=200.0)
        result = process_on_demand_leave_request(pools, _request(scheduled_hours))

        assert result.deductions[0].on_demand_days_deducted == 1.0
        assert result.pools_after.on_demand_pool_days == 3.0

    @pytest.mark.parametrize(
        "scheduled_hours,expected_hours_left",
        [
            (8.0, 192.0),
            (12.0, 188.0),
            (4.0, 196.0),
        ],
        ids=["hours_8h", "hours_12h", "hours_4h"],
    )
    def test_vacation_pool_decreases_by_scheduled_hours(
        self,
        scheduled_hours: float,
        expected_hours_left: float,
    ) -> None:
        pools = _pools(vacation_hours=200.0)
        result = process_on_demand_leave_request(pools, _request(scheduled_hours))

        assert result.deductions[0].vacation_hours_deducted == scheduled_hours
        assert result.pools_after.total_vacation_pool_hours == expected_hours_left

    def test_12h_shift_does_not_deduct_one_and_half_days(self) -> None:
        """Regresja: błędny algorytm 12/8=1.5 nie może wystąpić w poprawnej implementacji."""
        pools = _pools(on_demand_days=4.0)
        result = process_on_demand_leave_request(pools, _request(12.0))

        assert result.pools_after.on_demand_pool_days == 3.0
        assert result.deductions[0].on_demand_days_deducted != 1.5

        legacy = process_on_demand_leave_request_legacy_wrong(pools, _request(12.0))
        assert legacy.deductions[0].on_demand_days_deducted == 1.5
        assert legacy.pools_after.on_demand_pool_days == pytest.approx(2.5)


class TestMultipleDays:
    def test_three_days_each_costs_one_on_demand_day(self) -> None:
        pools = _pools(on_demand_days=4.0, vacation_hours=300.0)
        request = OnDemandLeaveRequest(
            employee_id=EMP_ID,
            days=[
                OnDemandLeaveDay(leave_date=date(2026, 7, 1), scheduled_hours=12.0),
                OnDemandLeaveDay(leave_date=date(2026, 7, 2), scheduled_hours=8.0),
                OnDemandLeaveDay(leave_date=date(2026, 7, 3), scheduled_hours=4.0),
            ],
        )
        result = process_on_demand_leave_request(pools, request)

        assert result.pools_after.on_demand_pool_days == 1.0
        assert result.pools_after.total_vacation_pool_hours == 300.0 - 12.0 - 8.0 - 4.0
        assert all(d.on_demand_days_deducted == 1.0 for d in result.deductions)
        assert [d.vacation_hours_deducted for d in result.deductions] == [12.0, 8.0, 4.0]


class TestValidation:
    def test_insufficient_on_demand_pool(self) -> None:
        pools = _pools(on_demand_days=0.0)
        with pytest.raises(InsufficientOnDemandLeaveError):
            process_on_demand_leave_request(pools, _request(8.0))

    def test_insufficient_vacation_hours(self) -> None:
        pools = _pools(on_demand_days=4.0, vacation_hours=10.0)
        with pytest.raises(InsufficientVacationHoursError):
            process_on_demand_leave_request(pools, _request(12.0))

    def test_duplicate_dates_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplikat"):
            OnDemandLeaveRequest(
                employee_id=EMP_ID,
                days=[
                    OnDemandLeaveDay(leave_date=date(2026, 8, 1), scheduled_hours=8.0),
                    OnDemandLeaveDay(leave_date=date(2026, 8, 1), scheduled_hours=12.0),
                ],
            )

    def test_annual_limit_constant(self) -> None:
        assert ON_DEMAND_ANNUAL_LIMIT_DAYS == 4


class TestLegalBasisInResult:
    def test_result_includes_legal_articles(self) -> None:
        result = process_on_demand_leave_request(_pools(), _request(8.0))
        joined = " ".join(result.legal_basis)
        assert "167" in joined
        assert "154" in joined
