"""Podpisane tokeny sesji (cookie)."""

from __future__ import annotations

import json
from typing import Any

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from auth.models import AuthUser, UserRole


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="kodekspracy-session")


def create_session_token(user: AuthUser, *, secret: str) -> str:
    payload = {
        "username": user.username,
        "role": user.role.value,
        "display_name": user.display_name,
        "employee_id": user.employee_id,
    }
    return _serializer(secret).dumps(payload)


def load_session_token(token: str, *, secret: str, max_age_sec: int) -> AuthUser | None:
    try:
        data: dict[str, Any] = _serializer(secret).loads(token, max_age=max_age_sec)
    except (BadSignature, SignatureExpired, KeyError, ValueError):
        return None
    return AuthUser(
        username=data["username"],
        role=UserRole(data["role"]),
        display_name=data["display_name"],
        employee_id=data.get("employee_id"),
    )
