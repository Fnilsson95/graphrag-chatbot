"""Extract user-facing source citations from a GraphRAG retrieval context.

The GraphRAG search APIs return a ``context`` object whose shape varies by
method: ``local_search`` exposes ``sources`` / ``text_units`` / ``entities``
frames keyed by id, ``basic_search`` exposes only text units, and so on.

This module flattens whichever frames are present into a deduplicated list
of ``Source`` records the frontend can render as citation chips — each
record carries the human-readable document title and a best-effort URL
reconstructed from the scraper's slugified filename.

URLs are reversed from titles like
``kubernetes_io_docs_concepts_workloads_pods.txt`` →
``https://kubernetes.io/docs/concepts/workloads/pods/``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class Source:
    title: str
    url: str | None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _title_to_url(title: str) -> str | None:
    """Reverse the scraper's slugify() — domain + path + .html."""
    if not title:
        return None
    cleaned = title.removesuffix(".txt")
    if not cleaned.startswith("www_"):
        return None
    parts = cleaned.split("_")
    domain_parts: list[str] = []
    while parts and parts[0] not in {"doc", "documentation", "docs"}:
        domain_parts.append(parts.pop(0))
    if not domain_parts or not parts:
        return None
    domain = ".".join(domain_parts)
    path_parts = parts
    if path_parts and path_parts[-1] == "html":
        path_parts = path_parts[:-1]
        page = "_".join(path_parts) + ".html"
    else:
        page = "_".join(path_parts)
    return f"https://{domain}/{page}".replace("//", "/").replace(
        "https:/", "https://"
    )


_URL_LINE_RE = re.compile(r"^URL:\s*(https?://\S+)", re.MULTILINE)


def _url_from_text(text: str) -> str | None:
    match = _URL_LINE_RE.search(text)
    return match.group(1) if match else None


def _source_url(title: str, text: str | None = None) -> str | None:
    url = _title_to_url(title)
    if url:
        return url
    if text:
        return _url_from_text(text)
    return None


def _title_to_label(title: str, url: str | None = None) -> str:
    """Human-readable link text for a documentation page."""
    if url:
        path = urlparse(url).path.rstrip("/")
        name = path.rsplit("/", 1)[-1]
        if name.endswith(".html"):
            name = name[:-5]
        label = name.replace("_", " ").strip()
        if label:
            return label
    cleaned = title.removesuffix(".txt")
    if cleaned.startswith("www_"):
        cleaned = cleaned[len("www_") :]
    return cleaned.replace("_", " ").strip() or "Documentation"


def format_documentation_links(
    sources: list[Source], *, max_links: int = 3
) -> str:
    """Markdown footer with links to the most relevant documentation pages."""
    seen: set[str] = set()
    links: list[str] = []
    for source in sources:
        if not source.url or source.url in seen:
            continue
        seen.add(source.url)
        label = _title_to_label(source.title, source.url)
        links.append(f"[{label}]({source.url})")
        if len(links) >= max_links:
            break
    if not links:
        return ""
    if len(links) == 1:
        return f"\n\nSee also: {links[0]}."
    bullets = "\n".join(f"- {link}" for link in links)
    return f"\n\n**Documentation:**\n{bullets}"


def _extract_source_rows(context: Any) -> list[tuple[str, str | None]]:
    """Pull (title, optional text) pairs from whichever frames are present."""
    if not isinstance(context, dict):
        return []

    rows: list[tuple[str, str | None]] = []
    for key in ("sources", "documents", "text_units", "reports"):
        frame = context.get(key)
        if frame is None:
            continue
        try:
            cols = list(frame.columns)
        except AttributeError:
            continue
        title_col = next(
            (c for c in ("title", "document_title", "source") if c in cols),
            None,
        )
        if title_col is None:
            continue
        has_text = "text" in cols
        for _, row in frame.iterrows():
            title = str(row.get(title_col, "") or "")
            if not title:
                continue
            text = str(row.get("text", "") or "") if has_text else None
            rows.append((title, text))
    return rows


def extract_sources(context: Any, limit: int = 5) -> list[Source]:
    """Flatten a GraphRAG context into a deduplicated source list."""
    # Deduplicate by title first, then by URL so the same page reached via
    # different frame keys (e.g. text_units vs sources) appears once.
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()
    out: list[Source] = []
    for title, text in _extract_source_rows(context):
        if title in seen_titles:
            continue
        seen_titles.add(title)
        url = _source_url(title, text)
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        out.append(Source(title=title, url=url))
        if len(out) >= limit:
            break
    return out
