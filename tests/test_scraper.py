"""Unit tests for data/scraper.py helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from scraper import GraphRAGScraper, canonical_url  # noqa: E402


@pytest.fixture
def scraper() -> GraphRAGScraper:
    return GraphRAGScraper(
        "https://www.example.com/doc/latest/index.html",
        max_depth=2,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://Example.com/doc/page/#section",
            "https://example.com/doc/page",
        ),
        (
            "https://example.com/doc/page/",
            "https://example.com/doc/page",
        ),
        (
            "https://example.com/",
            "https://example.com/",
        ),
    ],
)
def test_canonical_url(raw: str, expected: str) -> None:
    assert canonical_url(raw) == expected


def test_resolve_base_path_from_html(scraper: GraphRAGScraper) -> None:
    assert scraper.base_path == "/doc/latest"


def test_resolve_base_path_from_directory() -> None:
    s = GraphRAGScraper("https://www.example.com/doc/latest/")
    assert s.base_path == "/doc/latest"


def test_is_valid_same_domain_and_path(scraper: GraphRAGScraper) -> None:
    assert scraper.is_valid("https://www.example.com/doc/latest/guide.html")
    assert not scraper.is_valid("https://other.com/doc/latest/guide.html")
    assert not scraper.is_valid("https://www.example.com/other/guide.html")
    assert not scraper.is_valid("https://www.example.com/doc/latest/file.pdf")
    assert scraper.is_valid("https://www.example.com/doc/latest/diagram.svg")


def test_is_index_page(scraper: GraphRAGScraper) -> None:
    assert scraper.is_index_page(
        "https://www.example.com/doc/latest/index.html"
    )
    assert not scraper.is_index_page(
        "https://www.example.com/doc/latest/my-index.html-guide.html"
    )


def test_is_toc_page_detects_toctree(scraper: GraphRAGScraper) -> None:
    html = '<div class="toctree">link</div>'
    content = BeautifulSoup(html, "html.parser")
    assert scraper.is_toc_page(content)


def test_is_toc_page_link_heavy(scraper: GraphRAGScraper) -> None:
    html = """
    <div>
        <a>One</a><a>Two</a><a>Three</a>
        <p>x</p>
    </div>
    """
    content = BeautifulSoup(html, "html.parser").find("div")
    assert scraper.is_toc_page(content)


def test_table_to_markdown_escapes_pipes(scraper: GraphRAGScraper) -> None:
    html = """
    <table>
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1|2</td><td>ok</td></tr>
    </table>
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    md = scraper.table_to_markdown(table)
    assert "1\\|2" in md
    assert md.count("| --- |") == 1


def test_slugify_uses_canonical_form(scraper: GraphRAGScraper) -> None:
    url = canonical_url("https://www.example.com/doc/Page/#frag")
    slug_a = scraper.slugify(url)
    slug_b = scraper.slugify("https://www.example.com/doc/Page/#other")
    assert slug_a == slug_b
