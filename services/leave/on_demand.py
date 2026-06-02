"""
Urlop na żądanie — zgodność z Art. 167² i Art. 154² Kodeksu pracy.

Zasada biznesowa (system równoważny, zmiany 4/8/12 h):
- Każdy dzień urlopu na żądanie odejmuje DOKŁADNIE 1 dzień z puli 4 dni/rok.
- Z puli godzinowej urlopu wypoczynkowego odejmuje się faktyczne scheduled_hours dnia.

BŁĄD DO NAPRAWY (legacy): scheduled_hours / 8 jako „dni urlopu na żądanie” — niezgodne z KP.
"""

from __future__ import annotations

from copy import deepcopy

from services.leave.models import (
    InsufficientOnDemandLeaveError,
    InsufficientVacationHoursError,
    OnDemandLeaveDay,
    OnDemandLeaveDeduction,
    OnDemandLeaveRequest,
    OnDemandLeaveResult,
    VacationPools,
)

ON_DEMAND_ANNUAL_LIMIT_DAYS = 4
ON_DEMAND_DAY_DEDUCTION = 1.0


def _deduct_single_day(
    pools: VacationPools,
    day: OnDemandLeaveDay,
) -> tuple[VacationPools, OnDemandLeaveDeduction]:
    if pools.on_demand_pool_days < ON_DEMAND_DAY_DEDUCTION:
        raise InsufficientOnDemandLeaveError(
            f"Brak puli urlopu na żądanie: pozostało {pools.on_demand_pool_days} dni, "
            f"wymagane {ON_DEMAND_DAY_DEDUCTION} dzień na {day.leave_date}."
        )
    if pools.total_vacation_pool_hours < day.scheduled_hours:
        raise InsufficientVacationHoursError(
            f"Brak godzin urlopu wypoczynkowego: pozostało {pools.total_vacation_pool_hours} h, "
            f"wymagane {day.scheduled_hours} h na {day.leave_date}."
        )

    hours_deducted = round(day.scheduled_hours, 2)
    new_pools = VacationPools(
        on_demand_pool_days=round(pools.on_demand_pool_days - ON_DEMAND_DAY_DEDUCTION, 4),
        total_vacation_pool_hours=round(pools.total_vacation_pool_hours - hours_deducted, 2),
    )
    deduction = OnDemandLeaveDeduction(
        leave_date=day.leave_date,
        scheduled_hours=hours_deducted,
        on_demand_days_deducted=ON_DEMAND_DAY_DEDUCTION,
        vacation_hours_deducted=hours_deducted,
    )
    return new_pools, deduction


def process_on_demand_leave_request(
    pools: VacationPools,
    request: OnDemandLeaveRequest,
) -> OnDemandLeaveResult:
    """
    Przetwarza wniosek o urlop na żądanie.

    Dla każdego dnia:
    - on_demand_pool_days -= 1.0 (nigdy scheduled_hours / 8)
    - total_vacation_pool_hours -= scheduled_hours
    """
    current = deepcopy(pools)
    deductions: list[OnDemandLeaveDeduction] = []

    for day in request.days:
        current, deduction = _deduct_single_day(current, day)
        deductions.append(deduction)

    return OnDemandLeaveResult(pools_after=current, deductions=deductions)


def process_on_demand_leave_request_legacy_wrong(
    pools: VacationPools,
    request: OnDemandLeaveRequest,
) -> OnDemandLeaveResult:
    """
    Błędna implementacja (do testów regresji): dzieli godziny przez 8 przy potrąceniu puli na żądanie.
    """
    current = deepcopy(pools)
    deductions: list[OnDemandLeaveDeduction] = []

    for day in request.days:
        wrong_days = round(day.scheduled_hours / 8.0, 4)
        if current.on_demand_pool_days < wrong_days:
            raise InsufficientOnDemandLeaveError(
                f"Brak puli (legacy): potrzeba {wrong_days} dni za {day.scheduled_hours} h."
            )
        if current.total_vacation_pool_hours < day.scheduled_hours:
            raise InsufficientVacationHoursError(
                f"Brak godzin: potrzeba {day.scheduled_hours} h."
            )
        hours_deducted = round(day.scheduled_hours, 2)
        current = VacationPools(
            on_demand_pool_days=round(current.on_demand_pool_days - wrong_days, 4),
            total_vacation_pool_hours=round(current.total_vacation_pool_hours - hours_deducted, 2),
        )
        deductions.append(
            OnDemandLeaveDeduction(
                leave_date=day.leave_date,
                scheduled_hours=hours_deducted,
                on_demand_days_deducted=wrong_days,
                vacation_hours_deducted=hours_deducted,
            )
        )

    return OnDemandLeaveResult(pools_after=current, deductions=deductions)


def pools_from_employee_snapshot(
    *,
    on_demand_used_days: int,
    urlop_roczny_dni: int,
    urlop_wykorzystany_dni: int,
    wymiar_etatu: float = 1.0,
    daily_norm_hours: float = 8.0,
) -> VacationPools:
    """
    Buduje pule z danych kadrowych (dni → godziny przy normie 8 h × etat).

    W produkcji system równoważny powinien przechowywać total_vacation_pool_hours bezpośrednio.
    """
    annual_hours = urlop_roczny_dni * daily_norm_hours * wymiar_etatu
    used_hours = urlop_wykorzystany_dni * daily_norm_hours * wymiar_etatu
    remaining_hours = max(0.0, round(annual_hours - used_hours, 2))
    on_demand_remaining = max(
        0.0,
        float(ON_DEMAND_ANNUAL_LIMIT_DAYS - min(on_demand_used_days, ON_DEMAND_ANNUAL_LIMIT_DAYS)),
    )
    return VacationPools(
        on_demand_pool_days=on_demand_remaining,
        total_vacation_pool_hours=remaining_hours,
    )
