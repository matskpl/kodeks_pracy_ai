"""PostgreSQL — schema i połączenie."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg
from loguru import logger
from psycopg.rows import dict_row

from config import get_settings

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS employees (
    id VARCHAR(32) PRIMARY KEY,
    imie_nazwisko TEXT NOT NULL,
    stanowisko TEXT NOT NULL,
    dzial TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    staz_lata INTEGER NOT NULL CHECK (staz_lata >= 0),
    staz_miesiace INTEGER NOT NULL DEFAULT 0 CHECK (staz_miesiace >= 0 AND staz_miesiace <= 11),
    wymiar_etatu DOUBLE PRECISION NOT NULL CHECK (wymiar_etatu > 0 AND wymiar_etatu <= 1),
    rodzaj_umowy TEXT NOT NULL DEFAULT 'czas_nieokreslony',
    urlop_roczny_dni INTEGER NOT NULL CHECK (urlop_roczny_dni >= 0),
    urlop_wykorzystany INTEGER NOT NULL CHECK (urlop_wykorzystany >= 0),
    urlop_pozostaly INTEGER NOT NULL CHECK (urlop_pozostaly >= 0),
    urlop_na_zadanie_wykorzystany INTEGER NOT NULL DEFAULT 0
        CHECK (urlop_na_zadanie_wykorzystany >= 0 AND urlop_na_zadanie_wykorzystany <= 4),
    nadgodziny_limit_godz INTEGER NOT NULL DEFAULT 150,
    nadgodziny_wykorzystane DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (nadgodziny_wykorzystane >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    username VARCHAR(64) PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role VARCHAR(16) NOT NULL CHECK (role IN ('employee', 'admin')),
    display_name TEXT NOT NULL,
    employee_id VARCHAR(32) REFERENCES employees(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_employee_id ON users(employee_id);
"""


@contextmanager
def db_connection() -> Generator[psycopg.Connection[Any], None, None]:
    url = get_settings().database_url
    with psycopg.connect(url, row_factory=dict_row) as conn:
        yield conn


def init_schema(conn: psycopg.Connection[Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()


def count_employees(conn: psycopg.Connection[Any]) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM employees")
        row = cur.fetchone()
    return int(row["c"]) if row else 0
