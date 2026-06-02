"""Integracja auth/store z PostgreSQL (pomijane bez DATABASE_URL / serwera)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kodeks:kodeks@localhost:5432/kodekspracy",
)


def _postgres_available() -> bool:
    try:
        import psycopg

        with psycopg.connect(DATABASE_URL, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_available(), reason="PostgreSQL niedostępny")
def test_ensure_database_and_find_jan() -> None:
    os.environ["DATABASE_URL"] = DATABASE_URL
    from config import get_settings

    get_settings.cache_clear()

    from auth.store import ensure_database, find_user, get_employee

    ensure_database()
    user = find_user("jnowak")
    assert user is not None
    assert user.employee_id == "emp-001"
    emp = get_employee("emp-001")
    assert emp is not None
    assert emp.urlop_pozostaly == 12
