"""Moduł urlopów — urlop na żądanie i pule godzinowe (system równoważny)."""

from services.leave.models import (
    InsufficientOnDemandLeaveError,
    InsufficientVacationHoursError,
    OnDemandLeaveDay,
    OnDemandLeaveDeduction,
    OnDemandLeaveRequest,
    OnDemandLeaveResult,
    VacationPools,
)
from services.leave.on_demand import (
    ON_DEMAND_ANNUAL_LIMIT_DAYS,
    process_on_demand_leave_request,
    process_on_demand_leave_request_legacy_wrong,
)

__all__ = [
    "ON_DEMAND_ANNUAL_LIMIT_DAYS",
    "InsufficientOnDemandLeaveError",
    "InsufficientVacationHoursError",
    "OnDemandLeaveDay",
    "OnDemandLeaveDeduction",
    "OnDemandLeaveRequest",
    "OnDemandLeaveResult",
    "VacationPools",
    "process_on_demand_leave_request",
    "process_on_demand_leave_request_legacy_wrong",
]
