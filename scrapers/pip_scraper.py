"""Scraper wytycznych PIP (pip.gov.pl)."""

from __future__ import annotations

from loguru import logger

from scrapers.chunking import append_chunk_parts, chunk_text_blocks, html_to_text, split_by_headings
from scrapers.http_client import ScraperClient
from scrapers.models import LegalChunkDict, chunk_to_dict

PIP_PAGES: list[tuple[str, str, str]] = [
    (
        "Urlopy pracownicze",
        "https://www.pip.gov.pl/dla-pracodawcow/porady-prawne/urlopy-pracownicze",
    ),
    (
        "Czas pracy",
        "https://www.pip.gov.pl/dla-pracodawcow/porady-prawne/czas-pracy",
    ),
    (
        "Stosunek pracy — wypowiedzenia",
        "https://www.pip.gov.pl/dla-pracodawcow/porady-prawne/stosunek-pracy",
    ),
    (
        "Urlopy wypoczynkowe A–Z",
        "https://www.pip.gov.pl/aktualnosci/odpoczynek-rzecz-swieta-urlopy-wypoczynkowe-od-a-do-z",
    ),
    (
        "Plany urlopowe",
        "https://www.pip.gov.pl/aktualnosci/jak-prawidlowo-ulozyc-plany-urlopowe-na-2026-rok",
    ),
    (
        "Q&A — okres wypowiedzenia a staż",
        "https://www.pip.gov.pl/dla-pracodawcow/pytania-i-odpowiedzi/czy-okres-wypowiedzenia-wlicza-sie-do-stazu-pracy",
    ),
]


def scrape_pip_guides(client: ScraperClient) -> list[LegalChunkDict]:
    chunks: list[LegalChunkDict] = []
    for topic, url in PIP_PAGES:
        logger.info("Scraping PIP: {} — {}", topic, url)
        html = client.get_text(url)
        text = html_to_text(html)
        sections = split_by_headings(text)
        if not sections:
            sections = chunk_text_blocks(text)
        if not sections:
            sections = [(topic, text)]
        for section_title, section_body in sections:
            if len(section_body) < 80:
                continue
            append_chunk_parts(
                chunks,
                section_body=section_body,
                article=f"PIP — {section_title[:80]}",
                topic=topic,
                source="PIP",
                url=url,
                chunk_to_dict_fn=chunk_to_dict,
            )
    logger.success("PIP: {} chunków", len(chunks))
    return chunks
