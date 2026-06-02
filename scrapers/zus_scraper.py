"""Scraper regulacji ZUS (zus.pl) — zasiłki, zwolnienia, składki."""

from __future__ import annotations

import io
import re

from loguru import logger

from scrapers.chunking import (
    append_chunk_parts,
    chunk_text_blocks,
    clean_text,
    html_to_text,
    split_by_headings,
    split_long_text,
)
from scrapers.http_client import ScraperClient
from scrapers.models import LegalChunkDict, chunk_to_dict

ZUS_HTML_PAGES: list[tuple[str, str, str]] = [
    (
        "Zasiłek chorobowy — ogólne",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy",
    ),
    (
        "Zasiłek chorobowy — komu przysługuje",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy/z-ubezpieczenia-chorobowego/komu-przysluguje",
    ),
    (
        "Zasiłek chorobowy — okres przysługiwania",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy/z-ubezpieczenia-chorobowego/prawo-do-zasilku-i-okres-przyslugiwania",
    ),
    (
        "Zasiłek chorobowy — wysokość",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy/z-ubezpieczenia-chorobowego/wysokosc",
    ),
    (
        "Zasiłek chorobowy — dokumenty",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy/z-ubezpieczenia-chorobowego/niezbedne-dokumenty",
    ),
    (
        "e-ZLA — pytania i odpowiedzi",
        "https://www.zus.pl/swiadczenia/zasilki/zasilek-chorobowy/z-ubezpieczenia-chorobowego/elektroniczne-zwolnienia-lekarskie-e-zla/najczesciej-zadawane-pytania-e-zla",
    ),
]

ZUS_PDF_PAGES: list[tuple[str, str, str]] = [
    (
        "Świadczenia w razie choroby (brochure PDF)",
        "https://www.zus.pl/documents/10182/167561/%C5%9Awiadczenia+w+razie+choroby.+Zasi%C5%82ek+chorobowy%2C+%C5%9Bwiadczenie+rehabilitacyjne%2C+zasi%C5%82ek+wyr%C3%B3wnawczy/eda06c3e-5b72-4d16-82b6-fc8cb1b3f155",
    ),
]


def scrape_zus_regulations(client: ScraperClient) -> list[LegalChunkDict]:
    chunks: list[LegalChunkDict] = []
    for topic, url in ZUS_HTML_PAGES:
        logger.info("Scraping ZUS HTML: {} — {}", topic, url)
        try:
            html = client.get_text(url)
        except Exception as exc:
            logger.warning("Pominięto ZUS {}: {}", url, exc)
            continue
        text = html_to_text(html)
        if len(text) < 150:
            logger.warning("ZUS: pusty content {}", url)
            continue
        sections = split_by_headings(text)
        if len(sections) <= 1:
            sections = chunk_text_blocks(text)
        if not sections:
            sections = [(topic, text)]
        for section_title, section_body in sections:
            if len(section_body) < 100:
                continue
            append_chunk_parts(
                chunks,
                section_body=section_body,
                article=f"ZUS — {section_title[:90]}",
                topic=topic,
                source="ZUS",
                url=url,
                chunk_to_dict_fn=chunk_to_dict,
            )

    for topic, url in ZUS_PDF_PAGES:
        logger.info("Scraping ZUS PDF: {} — {}", topic, url)
        try:
            chunks.extend(_scrape_zus_pdf(client, topic, url))
        except Exception as exc:
            logger.warning("Pominięto PDF ZUS {}: {}", url, exc)

    logger.success("ZUS: {} chunków", len(chunks))
    return chunks


def _scrape_zus_pdf(client: ScraperClient, topic: str, url: str) -> list[LegalChunkDict]:
    from pypdf import PdfReader

    pdf_bytes = client.get_bytes(url)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages_text.append(clean_text(extracted))

    full_text = "\n".join(pages_text)
    sections = re.split(r"\n(?=[A-ZĄĆĘŁŃÓŚŹŻ][^\n]{5,80}\n)", full_text)
    chunks: list[LegalChunkDict] = []
    for idx, section in enumerate(sections, start=1):
        section = clean_text(section)
        if len(section) < 120:
            continue
        title_match = re.match(r"^(.{10,100})", section)
        title = title_match.group(1) if title_match else f"Sekcja {idx}"
        for part_idx, part in enumerate(split_long_text(section), start=1):
            title = title if part_idx == 1 else f"{title[:70]} (cz. {part_idx})"
            chunks.append(
                chunk_to_dict(
                    text=part,
                    article=f"ZUS PDF — {title[:80]}",
                    topic=topic,
                    source="ZUS",
                    url=url,
                )
            )
    return chunks
