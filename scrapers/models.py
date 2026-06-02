"""Modele danych ze scrapingu."""

from __future__ import annotations

from typing import Any, TypedDict

from config import RAG_CHUNK_MAX_LEN
from scrapers.chunking import split_long_text


class LegalChunkDict(TypedDict):
    text: str
    article: str
    topic: str
    source: str
    url: str


def chunk_to_dict(
    *,
    text: str,
    article: str,
    topic: str,
    source: str,
    url: str,
) -> LegalChunkDict:
    return {
        "text": text.strip(),
        "article": article.strip(),
        "topic": topic.strip(),
        "source": source.strip(),
        "url": url.strip(),
    }


def validate_chunks(chunks: list[dict[str, Any]]) -> list[LegalChunkDict]:
    validated: list[LegalChunkDict] = []
    for item in chunks:
        text = str(item.get("text", "")).strip()
        if len(text) < 80:
            continue
        article = str(item.get("article", "—"))
        topic = str(item.get("topic", "—"))
        source = str(item.get("source", "UNKNOWN"))
        url = str(item.get("url", ""))
        # Nie dziel ponownie krótkich chunków ze scrapera (unika podwójnego cięcia i luk).
        if len(text) <= RAG_CHUNK_MAX_LEN:
            parts = [text]
        else:
            parts = split_long_text(text)
        for idx, part in enumerate(parts, start=1):
            part_article = article
            if len(parts) > 1 and "(cz." not in article:
                part_article = f"{article} (cz. {idx}/{len(parts)})"
            validated.append(
                chunk_to_dict(
                    text=part,
                    article=part_article,
                    topic=topic,
                    source=source,
                    url=url,
                )
            )
    return validated
