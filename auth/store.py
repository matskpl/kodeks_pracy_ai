"""Magazyn użytkowników i pracowników — PostgreSQL."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from psycopg import sql

from auth.database import count_employees, db_connection, init_schema
from auth.models import EmployeeProfile, UserAccount
from auth.seed_data import default_employee_rows, default_user_rows
from config import DATA_DIR, get_settings

LEGACY_USERS_PATH = DATA_DIR / "users.json"

_EMPLOYEE_COLS = (
    "id",
    "imie_nazwisko",
    "stanowisko",
    "dzial",
    "email",
    "staz_lata",
    "staz_miesiace",
    "wymiar_etatu",
    "rodzaj_umowy",
    "urlop_roczny_dni",
    "urlop_wykorzystany",
    "urlop_pozostaly",
    "urlop_na_zadanie_wykorzystany",
    "nadgodziny_limit_godz",
    "nadgodziny_wykorzystane",
)


def _row_to_employee(row: dict[str, Any]) -> EmployeeProfile:
    return EmployeeProfile.model_validate(dict(row))


def _row_to_user(row: dict[str, Any]) -> UserAccount:
    return UserAccount.model_validate(dict(row))


def _insert_employees(conn: Any, employees: list[dict]) -> None:
    if not employees:
        return
    cols = sql.SQL(", ").join(sql.Identifier(c) for c in _EMPLOYEE_COLS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(_EMPLOYEE_COLS))
    q = sql.SQL("INSERT INTO employees ({}) VALUES ({}) ON CONFLICT (id) DO NOTHING").format(
        cols, placeholders
    )
    with conn.cursor() as cur:
        for emp in employees:
            cur.execute(q, tuple(emp[c] for c in _EMPLOYEE_COLS))


def _insert_users(conn: Any, users: list[dict]) -> None:
    if not users:
        return
    q = """
        INSERT INTO users (username, password_hash, role, display_name, employee_id)
        VALUES (%(username)s, %(password_hash)s, %(role)s, %(display_name)s, %(employee_id)s)
        ON CONFLICT (username) DO NOTHING
    """
    with conn.cursor() as cur:
        for user in users:
            cur.execute(q, user)


def _load_legacy_json() -> dict | None:
    if not LEGACY_USERS_PATH.exists():
        return None
    try:
        return json.loads(LEGACY_USERS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Nie można wczytać {}: {}", LEGACY_USERS_PATH, exc)
        return None


def seed_database(*, prefer_legacy_json: bool = True) -> None:
    """Wypełnia tabele, gdy puste — najpierw opcjonalnie z users.json."""
    settings = get_settings()
    employees: list[dict]
    users: list[dict]

    legacy = _load_legacy_json() if prefer_legacy_json else None
    if legacy and legacy.get("employees"):
        employees = legacy["employees"]
        users = legacy.get("users") or default_user_rows(settings.auth_secret)
        logger.info("Seed PostgreSQL z legacy {}", LEGACY_USERS_PATH)
    else:
        employees = default_employee_rows()
        users = default_user_rows(settings.auth_secret)
        logger.info("Seed PostgreSQL — domyślne dane demo MetalTech")

    with db_connection() as conn:
        _insert_employees(conn, employees)
        _insert_users(conn, users)
        conn.commit()


def ensure_database() -> None:
    """Tworzy schemat i seed przy pierwszym uruchomieniu."""
    empty = False
    with db_connection() as conn:
        init_schema(conn)
        empty = count_employees(conn) == 0
        conn.commit()
    if empty:
        seed_database(prefer_legacy_json=True)
    logger.info("Baza PostgreSQL (pracownicy / użytkownicy) gotowa")


def list_employees() -> list[EmployeeProfile]:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(_EMPLOYEE_COLS)} FROM employees ORDER BY imie_nazwisko"
            )
            rows = cur.fetchall()
    return [_row_to_employee(r) for r in rows]


def get_employee(employee_id: str) -> EmployeeProfile | None:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(_EMPLOYEE_COLS)} FROM employees WHERE id = %s",
                (employee_id,),
            )
            row = cur.fetchone()
    return _row_to_employee(row) if row else None


def list_user_accounts() -> list[UserAccount]:
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password_hash, role, display_name, employee_id FROM users"
            )
            rows = cur.fetchall()
    return [_row_to_user(r) for r in rows]


def find_user(username: str) -> UserAccount | None:
    key = username.strip().casefold()
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT username, password_hash, role, display_name, employee_id
                FROM users WHERE LOWER(username) = LOWER(%s)
                """,
                (key,),
            )
            row = cur.fetchone()
    return _row_to_user(row) if row else None


def update_employee_leave_state(
    employee_id: str,
    *,
    urlop_wykorzystany: int,
    urlop_pozostaly: int,
    urlop_na_zadanie_wykorzystany: int,
) -> EmployeeProfile | None:
    """Aktualizacja pul urlopowych po urlopie na żądanie (persystencja w DB)."""
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE employees SET
                    urlop_wykorzystany = %s,
                    urlop_pozostaly = %s,
                    urlop_na_zadanie_wykorzystany = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING """
                + ", ".join(_EMPLOYEE_COLS),
                (
                    urlop_wykorzystany,
                    urlop_pozostaly,
                    urlop_na_zadanie_wykorzystany,
                    employee_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()
    return _row_to_employee(row) if row else None


# Zachowanie wsteczne dla skryptów
ensure_users_file = ensure_database
