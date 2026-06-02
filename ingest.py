"""Ingest danych ze scrapingu (ISAP/ELI, PIP, ZUS) do Qdrant."""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from config import get_settings
from scrapers.pipeline import load_cache_metadata, load_cached_chunks, scrape_all, scrape_and_cache
from vector_store import (
    collection_point_count,
    get_qdrant_client,
    knowledge_base_is_ready,
    upsert_chunks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scraping + ingest do Qdrant")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Wymuś ponowny scraping (ignoruj cache JSON)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wymuś ponowny embed do Qdrant (nawet gdy baza już istnieje)",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.refresh:
        chunks = scrape_and_cache()
        cache_meta = load_cache_metadata()
    else:
        chunks = scrape_all(use_cache=True)
        cache_meta = load_cache_metadata()

    if not chunks:
        logger.error("Brak chunków po scrapingu — przerwano ingest.")
        sys.exit(1)

    settings = get_settings()
    if settings.max_ingest_chunks > 0:
        chunks = chunks[: settings.max_ingest_chunks]

    client = get_qdrant_client()
    scraped_at = cache_meta.get("scraped_at")

    if (
        not args.force
        and not args.refresh
        and knowledge_base_is_ready(
            client=client,
            expected_chunks=len(chunks),
            scraped_at=scraped_at,
        )
    ):
        logger.success(
            "Baza Qdrant aktualna — {} punktów, pomijam ponowny embed.",
            collection_point_count(client),
        )
        return

    recreate = args.force or args.refresh
    if recreate:
        logger.info("Przebudowa bazy Qdrant ({} chunków)...", len(chunks))
    else:
        logger.info("Pierwszy ingest do Qdrant ({} chunków)...", len(chunks))

    count = upsert_chunks(
        chunks,
        client=client,
        recreate=recreate,
        scraped_at=scraped_at,
    )
    logger.success("Ingest zakończony — wgrano {} chunków do Qdrant", count)


if __name__ == "__main__":
    main()
