"""Hashowanie haseł (PBKDF2, bez zewnętrznych zależności)."""

from __future__ import annotations

import hashlib
import secrets


def hash_password(password: str, *, secret: str) -> str:
    salt = hashlib.sha256(f"metaltech:{secret}".encode()).digest()[:16]
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return digest.hex()


def verify_password(password: str, password_hash: str, *, secret: str) -> bool:
    expected = hash_password(password, secret=secret)
    return secrets.compare_digest(expected, password_hash)
