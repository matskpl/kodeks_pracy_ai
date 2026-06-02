"""FastAPI dependencies — sesja z cookie."""

from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException

from auth.models import AuthUser, UserRole
from auth.session import load_session_token
from config import get_settings

SESSION_COOKIE = "kp_session"

# Te same atrybuty przy set/delete — inaczej przeglądarka nie usuwa ciasteczka.
SESSION_COOKIE_PARAMS = {
    "httponly": True,
    "samesite": "lax",
    "path": "/",
}


def apply_session_cookie(response, token: str, *, max_age_sec: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=max_age_sec,
        **SESSION_COOKIE_PARAMS,
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, **SESSION_COOKIE_PARAMS)


async def get_current_user(
    kp_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
) -> AuthUser:
    if not kp_session:
        raise HTTPException(status_code=401, detail="Wymagane logowanie.")
    settings = get_settings()
    user = load_session_token(
        kp_session,
        secret=settings.auth_secret,
        max_age_sec=settings.session_max_age_sec,
    )
    if not user:
        raise HTTPException(status_code=401, detail="Sesja wygasła — zaloguj się ponownie.")
    return user


async def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Tylko administrator kadr ma dostęp.")
    return user
