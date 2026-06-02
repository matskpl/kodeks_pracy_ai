"""Orkiestracja scrapingu i cache lokalny."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from config import BASE_DIR
from scrapers.chunking import merge_small_chunks
from scrapers.http_client import ScraperClient
from scrapers.isap_eli import scrape_kodeks_pracy
from scrapers.models import LegalChunkDict, validate_chunks
from scrapers.pip_scraper import scrape_pip_guides
from scrapers.zus_scraper import scrape_zus_regulations

DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "scraped_chunks.json"


def scrape_all(*, use_cache: bool = True, max_cache_age_hours: int = 168) -> list[LegalChunkDict]:
    """Pobiera dane ze wszystkich źródeł lub wczytuje cache."""
    if use_cache and CACHE_FILE.exists():
        age_hours = (datetime.now(timezone.utc).timestamp() - CACHE_FILE.stat().st_mtime) / 3600
        if age_hours <= max_cache_age_hours:
            cached = load_cached_chunks()
            if cached:
                logger.info("Wczytano {} chunków z cache ({:.1f} h)", len(cached), age_hours)
                return cached

    logger.info("Rozpoczynam scraping ISAP + PIP + ZUS...")
    with ScraperClient() as client:
        isap = scrape_kodeks_pracy(client)
        pip = scrape_pip_guides(client)
        zus = scrape_zus_regulations(client)

    tuples = [
        (c["article"], c["topic"], c["text"], c["source"], c["url"])
        for c in isap + pip + zus
    ]
    merged = merge_small_chunks(tuples)
    chunks = validate_chunks(
        [
            {
                "article": article,
                "topic": topic,
                "text": text,
                "source": source,
                "url": url,
            }
            for article, topic, text, source, url in merged
        ]
    )
    _save_cache(chunks)
    logger.success("Scraping zakończony — łącznie {} chunków", len(chunks))
    return chunks


def scrape_and_cache() -> list[LegalChunkDict]:
    return scrape_all(use_cache=False)


def load_cached_chunks() -> list[LegalChunkDict]:
    if not CACHE_FILE.exists():
        return []
    payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return validate_chunks(payload.get("chunks", []))


def load_cache_metadata() -> dict[str, Any]:
    if not CACHE_FILE.exists():
        return {}
    payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {
        "scraped_at": payload.get("scraped_at"),
        "count": payload.get("count", len(payload.get("chunks", []))),
    }


def _save_cache(chunks: list[LegalChunkDict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "sources": ["ISAP/ELI", "PIP", "ZUS"],
        "count": len(chunks),
        "chunks": chunks,
    }
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Zapisano cache: {}", CACHE_FILE)
