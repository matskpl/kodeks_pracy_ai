"""Domyślne dane kadrowe i konta — seed PostgreSQL."""

from __future__ import annotations

from auth.models import UserRole
from auth.password import hash_password


def default_employee_rows() -> list[dict]:
    rows = [
        ("emp-001", "Jan Nowak", "Operator CNC", "Produkcja", "jan.nowak@metaltech.pl", 8, 3, 1.0, 26, 14, 12, 1, 42),
        ("emp-002", "Anna Kowalska", "Spawacz TIG", "Produkcja", "anna.kowalska@metaltech.pl", 5, 6, 1.0, 20, 8, 12, 0, 28),
        ("emp-003", "Piotr Wiśniewski", "Kierownik zmiany", "Produkcja", "piotr.wisniewski@metaltech.pl", 12, 0, 1.0, 26, 18, 8, 2, 65),
        ("emp-004", "Maria Zielińska", "Kontroler jakości", "Jakość", "maria.zielinska@metaltech.pl", 6, 2, 1.0, 20, 5, 15, 0, 18),
        ("emp-005", "Tomasz Wójcik", "Magazynier", "Logistyka", "tomasz.wojcik@metaltech.pl", 3, 4, 0.75, 15, 6, 9, 1, 12),
        ("emp-006", "Katarzyna Kamińska", "Technolog produkcji", "Technologia", "katarzyna.kaminska@metaltech.pl", 9, 1, 1.0, 26, 11, 15, 1, 35),
        ("emp-007", "Michał Lewandowski", "Operator piły", "Produkcja", "michal.lewandowski@metaltech.pl", 4, 8, 1.0, 20, 12, 8, 2, 55),
        ("emp-008", "Agnieszka Dąbrowska", "Specjalista BHP", "BHP", "agnieszka.dabrowska@metaltech.pl", 7, 0, 1.0, 20, 4, 16, 0, 8),
        ("emp-009", "Paweł Kozłowski", "Monter konstrukcji", "Produkcja", "pawel.kozlowski@metaltech.pl", 10, 5, 1.0, 26, 20, 6, 3, 88),
        ("emp-010", "Ewa Jankowska", "Księgowa", "Finanse", "ewa.jankowska@metaltech.pl", 11, 2, 0.5, 13, 3, 10, 0, 5),
        ("emp-011", "Robert Mazur", "Lakiernik proszkowy", "Produkcja", "robert.mazur@metaltech.pl", 2, 6, 1.0, 20, 9, 11, 1, 22),
        ("emp-012", "Joanna Król", "HR Business Partner", "Kadry", "joanna.krol@metaltech.pl", 8, 0, 1.0, 26, 7, 19, 0, 10),
        ("emp-013", "Łukasz Pawlak", "Elektryk utrzymania", "Utrzymanie ruchu", "lukasz.pawlak@metaltech.pl", 6, 9, 1.0, 20, 10, 10, 1, 48),
        ("emp-014", "Monika Sikora", "Młodszy operator", "Produkcja", "monika.sikora@metaltech.pl", 1, 2, 0.5, 10, 2, 8, 0, 6),
    ]
    out: list[dict] = []
    for row in rows:
        (
            eid,
            name,
            pos,
            dept,
            email,
            y,
            m,
            etat,
            urlop_roczny,
            urlop_wyk,
            urlop_poz,
            urlop_zad,
            nadg_wyk,
        ) = row
        out.append(
            {
                "id": eid,
                "imie_nazwisko": name,
                "stanowisko": pos,
                "dzial": dept,
                "email": email,
                "staz_lata": y,
                "staz_miesiace": m,
                "wymiar_etatu": etat,
                "rodzaj_umowy": "czas_nieokreslony",
                "urlop_roczny_dni": urlop_roczny,
                "urlop_wykorzystany": urlop_wyk,
                "urlop_pozostaly": urlop_poz,
                "urlop_na_zadanie_wykorzystany": urlop_zad,
                "nadgodziny_limit_godz": 150,
                "nadgodziny_wykorzystane": float(nadg_wyk),
            }
        )
    return out


def default_user_rows(auth_secret: str) -> list[dict]:
    accounts = [
        ("kadry", "kadry123", UserRole.ADMIN, "Joanna Król — Kadry (admin)", None),
        ("jnowak", "jnowak123", UserRole.EMPLOYEE, "Jan Nowak", "emp-001"),
        ("akowalska", "akowalska123", UserRole.EMPLOYEE, "Anna Kowalska", "emp-002"),
        ("pwisniewski", "pwisniewski123", UserRole.EMPLOYEE, "Piotr Wiśniewski", "emp-003"),
        ("mzielinska", "mzielinska123", UserRole.EMPLOYEE, "Maria Zielińska", "emp-004"),
        ("twojcik", "twojcik123", UserRole.EMPLOYEE, "Tomasz Wójcik", "emp-005"),
        ("kkaminska", "kkaminska123", UserRole.EMPLOYEE, "Katarzyna Kamińska", "emp-006"),
        ("mlewandowski", "mlewandowski123", UserRole.EMPLOYEE, "Michał Lewandowski", "emp-007"),
        ("adabrowska", "adabrowska123", UserRole.EMPLOYEE, "Agnieszka Dąbrowska", "emp-008"),
        ("pkozlowski", "pkozlowski123", UserRole.EMPLOYEE, "Paweł Kozłowski", "emp-009"),
        ("ejankowska", "ejankowska123", UserRole.EMPLOYEE, "Ewa Jankowska", "emp-010"),
        ("rmazur", "rmazur123", UserRole.EMPLOYEE, "Robert Mazur", "emp-011"),
        ("jkrol", "jkrol123", UserRole.EMPLOYEE, "Joanna Król", "emp-012"),
        ("lpawlak", "lpawlak123", UserRole.EMPLOYEE, "Łukasz Pawlak", "emp-013"),
        ("msikora", "msikora123", UserRole.EMPLOYEE, "Monika Sikora", "emp-014"),
    ]
    return [
        {
            "username": u,
            "password_hash": hash_password(p, secret=auth_secret),
            "role": r.value,
            "display_name": d,
            "employee_id": eid,
        }
        for u, p, r, d, eid in accounts
    ]
