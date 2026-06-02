#!/usr/bin/env python3
"""Inicjalizacja schematu PostgreSQL i seed (domyślne dane lub import z data/users.json)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auth.store import ensure_database, seed_database  # noqa: E402
from auth.database import count_employees, db_connection, init_schema  # noqa: E402


def main() -> None:
    force = "--force" in sys.argv
    with db_connection() as conn:
        init_schema(conn)
        conn.commit()
    if force:
        seed_database(prefer_legacy_json=True)
        print("Wymuszono seed (INSERT … ON CONFLICT DO NOTHING).")
    else:
        ensure_database()
    with db_connection() as conn:
        n = count_employees(conn)
    print(f"Pracowników w bazie: {n}")


if __name__ == "__main__":
    main()
