"""Narzędzia do czyszczenia HTML i dzielenia tekstu na chunki RAG (recursive + overlap)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import RAG_CHUNK_MAX_LEN, RAG_CHUNK_OVERLAP

WHITESPACE_RE = re.compile(r"\s+")
ARTICLE_HEADER_RE = re.compile(
    r"(Art\.(?:\s*\d+(?:\s*[\u00b9\u00b2\u00b3\u2070-\u2099]+|\^\d+)?(?:\s*§\s*\d+)?(?:\s*pkt\s*\d+)?(?:\s*KP)?)?)",
    re.IGNORECASE,
)
SUPERSCRIPT_MAP = {
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}

# Kolejność: najpierw akapity i zdania, dopiero na końcu spacje (nie w środku słowa).
POLISH_RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "§ ",
    "; ",
    "? ",
    "! ",
    ", ",
    " ",
    "",
]

MIN_CHUNK_CHARS = 40


def clean_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()
    main = (
        soup.select_one(".journal-content-article")
        or soup.select_one("#main-content")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"journal-content|content|article"))
        or soup.find("body")
    )
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    text = re.sub(r"ścieżka nawigacji.*?(?=\n[A-ZĄĆĘŁŃÓŚŹŻ])", "", text, flags=re.I | re.DOTALL)
    return clean_text(text)


@lru_cache
def _get_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=POLISH_RECURSIVE_SEPARATORS,
        is_separator_regex=False,
        keep_separator=True,
    )


def split_long_text(
    text: str,
    *,
    max_len: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Dzieli tekst rekurencyjnie (akapity → zdania → spacje), z nakładaniem chunków.
    Gwarantuje overlap między sąsiednimi fragmentami — bez „dziur” jak „podle” / „enia”.
    """
    text = clean_text(text)
    if not text:
        return []
    limit = max_len or RAG_CHUNK_MAX_LEN
    overlap_len = overlap if overlap is not None else RAG_CHUNK_OVERLAP
    overlap_len = max(overlap_len, 200)

    if len(text) <= limit:
        return [text]

    splitter = _get_recursive_splitter(limit, overlap_len)
    parts = [p.strip() for p in splitter.split_text(text) if len(p.strip()) >= MIN_CHUNK_CHARS]

    if not parts:
        return [text]

    _log_split_gaps(text, parts, overlap_len)
    return parts


def _log_split_gaps(source: str, parts: list[str], overlap: int) -> None:
    """Ostrzeżenie w logu, gdy między chunkami brakuje mostka (możliwa utrata treści)."""
    if len(parts) < 2:
        return
    for i in range(len(parts) - 1):
        tail = parts[i][-min(80, len(parts[i])) :]
        head = parts[i + 1][: min(80, len(parts[i + 1]))]
        if tail and head and tail[-20:] not in parts[i + 1] and head[:20] not in parts[i]:
            bridge = source.find(tail[-15:]) if len(tail) >= 15 else -1
            if bridge >= 0:
                between = source[bridge + len(tail[-15:]) : bridge + len(tail[-15:]) + 120]
                if between and between.strip()[:40] not in parts[i + 1][: max(overlap, 200)]:
                    logger.debug(
                        "Chunking: możliwa luka między fragmentem {} a {} (sprawdź overlap)",
                        i + 1,
                        i + 2,
                    )


def split_by_headings(
    text: str,
    *,
    min_len: int = 120,
    max_len: int | None = None,
) -> list[tuple[str, str]]:
    """Dzieli tekst poradnika na sekcje (nagłówek, treść) z pełną treścią."""
    limit = max_len or RAG_CHUNK_MAX_LEN
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    sections: list[tuple[str, str]] = []
    current_title = "Ogólne"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_title
        body = clean_text(" ".join(buffer))
        if len(body) < min_len:
            buffer = []
            return
        for idx, part in enumerate(split_long_text(body, max_len=limit), start=1):
            title = current_title if idx == 1 else f"{current_title} (cz. {idx})"
            sections.append((title, part))
        buffer = []

    for line in lines:
        if len(line) < 90 and (line.isupper() or line.endswith(":") or line.startswith("##")):
            flush()
            current_title = line.rstrip(":").lstrip("#").strip()
            continue
        buffer.append(line)
        if sum(len(x) for x in buffer) >= limit * 2:
            flush()
    flush()
    return sections


def split_kodeks_by_articles(html: str) -> list[tuple[str, str]]:
    """Parsuje tekst jednolity Kodeksu pracy z HTML ELI/ISAP."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    articles: list[tuple[str, str]] = []
    nodes = soup.select("div.art, div[id^='art_'], p.art, td.art")
    if not nodes:
        plain = html_to_text(html)
        return _split_plain_kodeks(plain)

    for node in nodes:
        label_node = node.find(class_=re.compile("artlabel|nr-art|label")) or node
        label = clean_text(label_node.get_text(" ", strip=True))
        if not label.lower().startswith("art"):
            match = re.search(r"Art\.\s*\d+", label, re.I)
            label = match.group(0) if match else label[:40]
        body = clean_text(node.get_text(" ", strip=True))
        if len(body) < 60:
            continue
        article_id = normalize_article_label(label)
        articles.extend(_article_parts(article_id, body))

    if not articles:
        return _split_plain_kodeks(html_to_text(html))
    return articles


def _split_plain_kodeks(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=(?:Art\.|Art\.)\s*\d+)", text, flags=re.IGNORECASE)
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
        else:
            label = f"Kodeks pracy — fragment {fragment_idx}"
            fragment_idx += 1
        articles.extend(_article_parts(label, part))
    return articles


def _article_parts(article: str, body: str) -> list[tuple[str, str]]:
    parts = split_long_text(body)
    if len(parts) == 1:
        return [(article, parts[0])]
    return [(f"{article} (cz. {idx}/{len(parts)})", part) for idx, part in enumerate(parts, start=1)]


def normalize_article_label(label: str) -> str:
    label = clean_text(label)
    match = re.match(r"(Art\.\s*)(\d+)(.*)", label, re.I)
    if match:
        prefix, num, rest = match.groups()
        if len(num) >= 4 and num[-1] in SUPERSCRIPT_MAP and int(num[:-1]) >= 100:
            num = f"{num[:-1]}{SUPERSCRIPT_MAP[num[-1]]}"
        label = f"{prefix}{num}{rest}".strip()
    if "KP" not in label.upper():
        label = f"{label} KP"
    return label


def chunk_text_blocks(
    text: str,
    *,
    max_len: int | None = None,
    overlap: int | None = None,
) -> list[tuple[str, str]]:
    """Dzieli długi tekst na bloki (tytuł, treść)."""
    limit = max_len or RAG_CHUNK_MAX_LEN
    parts = split_long_text(text, max_len=limit, overlap=overlap)
    blocks: list[tuple[str, str]] = []
    for idx, body in enumerate(parts, start=1):
        title = body[:80].rsplit(" ", 1)[0] if len(body) > 80 else body[:80]
        if len(parts) > 1:
            title = f"{title or 'Fragment'} (cz. {idx})"
        blocks.append((title or f"Fragment {idx}", body))
    return blocks


def append_chunk_parts(
    chunks: list,
    *,
    section_body: str,
    article: str,
    topic: str,
    source: str,
    url: str,
    chunk_to_dict_fn,
) -> None:
    """Dodaje jeden lub więcej chunków po split_long_text (wspólna ścieżka ZUS/PIP)."""
    for part_idx, part in enumerate(split_long_text(section_body), start=1):
        if len(part) < MIN_CHUNK_CHARS:
            continue
        art = article if part_idx == 1 else f"{article} (cz. {part_idx})"
        chunks.append(
            chunk_to_dict_fn(
                text=part,
                article=art,
                topic=topic,
                source=source,
                url=url,
            )
        )


def merge_small_chunks(
    items: Iterable[tuple[str, str, str, str, str]],
    *,
    min_len: int = 200,
    max_len: int | None = None,
) -> list[tuple[str, str, str, str, str]]:
    """Scala krótkie fragmenty w ramach tego samego artykułu/URL (bez obcinania treści)."""
    limit = max_len or RAG_CHUNK_MAX_LEN
    merged: list[tuple[str, str, str, str, str]] = []
    buffer_text = ""
    buffer_article = ""
    buffer_topic = ""
    buffer_source = ""
    buffer_url = ""

    def flush() -> None:
        nonlocal buffer_text, buffer_article, buffer_topic, buffer_source, buffer_url
        if not buffer_text.strip():
            return
        for part in split_long_text(buffer_text.strip(), max_len=limit):
            merged.append((buffer_article, buffer_topic, part, buffer_source, buffer_url))
        buffer_text = ""
        buffer_article = ""
        buffer_topic = ""
        buffer_source = ""
        buffer_url = ""

    for article, topic, text, source, url in items:
        same_source = (
            buffer_text
            and buffer_source == source
            and buffer_url == url
            and buffer_article == article
        )
        if buffer_text and not same_source:
            flush()
        if not buffer_text:
            buffer_article, buffer_topic, buffer_source, buffer_url = article, topic, source, url
        buffer_text = f"{buffer_text}\n\n{text}".strip() if buffer_text else text.strip()
        if len(buffer_text) >= min_len and same_source:
            flush()
    flush()
    return merged
