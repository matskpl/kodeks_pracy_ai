"""Wspólny klient HTTP do scrapingu."""

from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
}


class ScraperClient:
    def __init__(self, *, timeout: float = 120.0, delay_seconds: float = 0.5) -> None:
        self._delay = delay_seconds
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ScraperClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_text(self, url: str) -> str:
        time.sleep(self._delay)
        logger.debug("GET {}", url)
        response = self._client.get(url)
        response.raise_for_status()
        response.encoding = response.encoding or "utf-8"
        return response.text

    def get_bytes(self, url: str) -> bytes:
        time.sleep(self._delay)
        logger.debug("GET (bytes) {}", url)
        response = self._client.get(url)
        response.raise_for_status()
        return response.content
