"""Scraper Kodeksu pracy z API ELI/ISAP (api.sejm.gov.pl)."""

from __future__ import annotations

import io
import re

from loguru import logger

from scrapers.chunking import _article_parts, clean_text, normalize_article_label
from scrapers.http_client import ScraperClient
from scrapers.models import LegalChunkDict, chunk_to_dict

KODEKS_JEDNOLITY_PDF_URL = "https://api.sejm.gov.pl/eli/acts/DU/2025/277/text.pdf"
KODEKS_JEDNOLITY_PAGE_URL = "https://api.sejm.gov.pl/eli/acts/DU/2025/277"
KODEKS_HTML_URL = "https://api.sejm.gov.pl/eli/acts/DU/1974/141/text.html"

ARTICLE_SPLIT_RE = re.compile(r"(?=(?:Art\.|Art\.)\s*\d+)", re.IGNORECASE)


def scrape_kodeks_pracy(client: ScraperClient) -> list[LegalChunkDict]:
    logger.info("Scraping jednolity Kodeks pracy (PDF ELI/ISAP): {}", KODEKS_JEDNOLITY_PDF_URL)
    pdf_chunks = _scrape_kodeks_pdf(client)
    if pdf_chunks:
        logger.success("ISAP/ELI PDF: {} chunków", len(pdf_chunks))
        return pdf_chunks

    logger.warning("PDF niedostępny — fallback do HTML ELI")
    return _scrape_kodeks_html(client)


def _scrape_kodeks_pdf(client: ScraperClient) -> list[LegalChunkDict]:
    from pypdf import PdfReader

    pdf_bytes = client.get_bytes(KODEKS_JEDNOLITY_PDF_URL)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = clean_text("\n".join((page.extract_text() or "") for page in reader.pages))
    articles = _split_kodeks_text(full_text)
    logger.info("PDF KP: rozpoznano {} fragmentów artykułów", len(articles))

    chunks: list[LegalChunkDict] = []
    for article, body in articles:
        chunks.append(
            chunk_to_dict(
                text=body,
                article=article,
                topic=_topic_from_article(article, body),
                source="ISAP",
                url=KODEKS_JEDNOLITY_PDF_URL,
            )
        )
    return chunks


def _scrape_kodeks_html(client: ScraperClient) -> list[LegalChunkDict]:
    from bs4 import BeautifulSoup

    html = client.get_text(KODEKS_HTML_URL)
    soup = BeautifulSoup(html, "lxml")
    units = soup.select("div.unit_arti")
    chunks: list[LegalChunkDict] = []

    if units:
        for unit in units:
            body = clean_text(unit.get_text(" ", strip=True))
            match = re.match(r"(Art\.\s*\d+(?:\s*[\u00b9\u00b2\u00b3\u2070-\u2099]+|\^\d+)?)", body, re.I)
            if not match:
                continue
            article = normalize_article_label(match.group(1))
            for part_article, part_body in _article_parts(article, body):
                chunks.append(
                    chunk_to_dict(
                        text=part_body,
                        article=part_article,
                        topic=_topic_from_article(part_article, part_body),
                        source="ISAP",
                        url=KODEKS_HTML_URL,
                    )
                )
    else:
        for article, body in _split_kodeks_text(clean_text(soup.get_text(" ", strip=True))):
            chunks.append(
                chunk_to_dict(
                    text=body,
                    article=article,
                    topic=_topic_from_article(article, body),
                    source="ISAP",
                    url=KODEKS_HTML_URL,
                )
            )

    logger.success("ISAP/ELI HTML: {} chunków", len(chunks))
    return chunks


def _split_kodeks_text(text: str) -> list[tuple[str, str]]:
    parts = ARTICLE_SPLIT_RE.split(text)
    articles: list[tuple[str, str]] = []
    fragment_idx = 1
    for part in parts:
        part = clean_text(part)
        if len(part) < 50:
            continue
        match = re.match(
            r"(Art\.\s*\d+(?:\s*[\u00b9\u00b2\u00b3\u2070-\u2099]+|\^\d+)?(?:\s*§\s*\d+)?)",
            part,
            re.I,
        )
        if match:
            label = normalize_article_label(match.group(1))
            articles.extend(_article_parts(label, part))
        else:
            label = f"Kodeks pracy — fragment {fragment_idx} KP"
            fragment_idx += 1
            articles.extend(_article_parts(label, part))
    return articles


def _topic_from_article(article: str, body: str) -> str:
    lowered = body.lower()
    if "na żądanie" in lowered and "urlop" in lowered:
        return "urlop na żądanie — urlopy i wypoczynek"
    if "urlop" in lowered:
        return "urlopy i wypoczynek"
    if "wypowiedz" in lowered or "rozwiąza" in lowered:
        return "wypowiedzenia i rozwiązanie umowy"
    if "nadgodzin" in lowered or ("godzin" in lowered and "prac" in lowered):
        return "czas pracy i nadgodziny"
    if "wynagrodz" in lowered:
        return "wynagrodzenie"
    if "chorob" in lowered or "zasił" in lowered:
        return "niezdolność do pracy / zasiłki"
    return f"Kodeks pracy — {article}"
