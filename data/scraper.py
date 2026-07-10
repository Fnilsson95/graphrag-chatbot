"""Crawl Kubernetes documentation into GraphRAG-ready text files.

Recursively follows links under a base URL, extracts the main content area
from each HTML page, converts tables/code/images to LLM-friendly markdown,
and writes one ``.txt`` per page under ``data/processed/``. Filenames are
slugified URLs (reversed by ``app.sources._title_to_url`` for citation links).

Run directly::

    python data/scraper.py
"""

import json
import logging
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import lxml.etree as ET
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

SKIP_PAGE_NAMES = frozenset(
    {"index.html", "search.html", "genindex.html", "py-modindex.html"}
)
BLOCKED_EXTENSIONS = (
    ".zip",
    ".pdf",
    ".py",
    ".png",
    ".jpg",
    ".jpeg",
    ".rst",
    ".txt",
    ".xml",
)
USER_AGENT = "Mozilla/5.0 (compatible; GraphRAGScraper/1.0)"


def canonical_url(url: str) -> str:
    """Normalize URL for deduplication and filesystem naming."""
    parsed = urlparse(url.strip())
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            "",
        )
    )


class GraphRAGScraper:
    """Breadth-first crawler that saves high-value doc pages as plain text."""

    def __init__(self, base_url, max_depth=5, delay=0.1):
        self.base_url = base_url
        # Normalize base_url to ensure it ends with a slash if it's a directory
        self.base_path = os.path.dirname(urlparse(base_url).path)
        self.domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.delay = delay
        self.visited: set[str] = set()
        self.saved_count = 0
        self.skipped_count = 0
        self.failed_urls: list[dict[str, str]] = []

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.normpath(
            os.path.join(script_dir, "processed")
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.visited_path = os.path.join(self.output_dir, ".visited_urls.json")
        self.failed_path = os.path.join(self.output_dir, ".failed_urls.json")
        self.session = self._build_session()

    @staticmethod
    def _resolve_base_path(parsed) -> str:
        path = parsed.path or "/"
        if path.endswith((".html", ".htm")):
            return os.path.dirname(path) or "/"
        return path.rstrip("/") or "/"

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def load_visited(self) -> None:
        if not os.path.exists(self.visited_path):
            return
        with open(self.visited_path, encoding="utf-8") as f:
            self.visited = set(json.load(f))
        logger.info("Loaded %d previously visited URLs", len(self.visited))

    def save_visited(self) -> None:
        with open(self.visited_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.visited), f, indent=2)

    def save_failed(self) -> None:
        if not self.failed_urls:
            return
        with open(self.failed_path, "w", encoding="utf-8") as f:
            json.dump(self.failed_urls, f, indent=2)

    def _record_failure(self, url: str, reason: str) -> None:
        self.failed_urls.append({"url": url, "reason": reason})
        logger.warning("Failed %s: %s", url, reason)

    def is_valid(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        is_same_domain = parsed.netloc == self.domain
        is_in_doc_path = parsed.path.startswith(self.base_path)
        is_not_source = "_sources" not in parsed.path
        lower = url.lower()
        is_blocked = any(lower.endswith(ext) for ext in BLOCKED_EXTENSIONS)
        is_svg = lower.endswith(".svg")
        return (
            is_same_domain
            and is_in_doc_path
            and is_not_source
            and (not is_blocked or is_svg)
        )

    def is_index_page(self, url: str) -> bool:
        return os.path.basename(urlparse(url).path) in SKIP_PAGE_NAMES

    def is_toc_page(self, main_content) -> bool:
        if not main_content:
            return True
        if main_content.find(class_=re.compile(r"toctree")):
            return True
        all_text = main_content.get_text(strip=True)
        link_text = "".join(
            a.get_text(strip=True) for a in main_content.find_all("a")
        )
        if not all_text:
            return True
        return (len(link_text) / len(all_text)) > 0.7

    def slugify(self, url: str) -> str:
        name = (
            canonical_url(url).replace("https://", "").replace("http://", "")
        )
        name = re.sub(r"[^\w\s-]", "_", name).strip().lower()
        return name[:150]

    @staticmethod
    def _escape_cell(text: str) -> str:
        return text.replace("|", "\\|")

    def table_to_markdown(self, table) -> str:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [
                self._escape_cell(cell.get_text(strip=True))
                for cell in tr.find_all(["th", "td"])
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        max_cols = max(len(row) for row in rows)
        lines = []
        for row in rows:
            padded = row + [""] * (max_cols - len(row))
            lines.append(f"| {' | '.join(padded)} |")
        separator = f"| {' | '.join(['---'] * max_cols)} |"
        lines.insert(1, separator)
        return "\n".join(lines)

    def _fetch(self, url: str) -> requests.Response:
        response = self.session.get(url, timeout=10, allow_redirects=True)
        final_url = canonical_url(response.url)
        if not self.is_valid(final_url):
            raise ValueError(f"Redirect left allowed scope: {final_url}")
        response.raise_for_status()
        return response

    def _extract_links(self, page_url: str, soup: BeautifulSoup) -> list[str]:
        seen: set[str] = set()
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith(("mailto:", "javascript:")):
                continue
            joined = canonical_url(urljoin(page_url, href))
            if joined not in seen:
                seen.add(joined)
                links.append(joined)
        return links

    def _transform_content(self, soup: BeautifulSoup, main_content) -> str:
        for table in main_content.find_all("table"):
            md = self.table_to_markdown(table)
            table.replace_with(soup.new_string(f"\n{md}\n"))

        for pre in main_content.find_all("pre"):
            text = pre.get_text().strip()
            pre.replace_with(soup.new_string(f"\n```\n{text}\n```\n"))

        for code in main_content.find_all("code"):
            if code.find_parent("pre"):
                continue
            text = code.get_text().strip()
            code.replace_with(soup.new_string(f"`{text}`"))

        for img in main_content.find_all("img"):
            alt = img.get("alt", "No description")
            img.replace_with(soup.new_string(f"\n[IMAGE: {alt}]\n"))

        for headerlink in soup.find_all(class_="headerlink"):
            headerlink.decompose()

        text = main_content.get_text(separator="\n")
        return "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

    def _save_page(self, url: str, content: str) -> None:
        filename = f"{self.slugify(url)}.txt"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n{'-' * 20}\n{content}")
        self.saved_count += 1
        logger.info("Saved %s", filename)

    def parse_svg(self, url: str) -> list[str]:
        """Save SVG pages as text metadata for the GraphRAG pipeline."""
        response = self._fetch(url)
        raw = response.content
        title = ""
        desc = ""
        try:
            tree = ET.fromstring(raw)
            title_el = tree.find(".//{*}title")
            desc_el = tree.find(".//{*}desc")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
            if desc_el is not None and desc_el.text:
                desc = desc_el.text.strip()
        except ET.XMLSyntaxError as exc:
            self._record_failure(url, f"Invalid SVG XML: {exc}")
            return []

        lines = [f"[SVG diagram: {url}]"]
        if title:
            lines.append(f"Title: {title}")
        if desc:
            lines.append(f"Description: {desc}")
        self._save_page(url, "\n".join(lines))
        return []

    def _process_page(self, url: str, depth: int) -> list[str]:
        logger.info("[%d] Processing: %s", depth, url)

        if url.lower().endswith(".svg"):
            self.parse_svg(url)
            return []

        response = self._fetch(url)
        response.encoding = response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        links = self._extract_links(url, soup)

        main_content = (
            soup.find("div", {"role": "main"})
            or soup.find("main")
            or soup.find("div", class_="td-content")
            or soup.find("div", class_="body")
        )


        if self.is_index_page(url) or self.is_toc_page(main_content):
            self.skipped_count += 1
            logger.debug("Skipping save (TOC or index): %s", url)
            return links

        if not main_content:
            self.skipped_count += 1
            logger.debug("Skipping save (no main content): %s", url)
            return links

        clean_text = self._transform_content(soup, main_content)
        self._save_page(url, clean_text)
        return links

    def scrape(self, start_url: str | None = None) -> None:
        start = canonical_url(start_url or self.base_url)
        self.load_visited()
        started_at = time.monotonic()

        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            url, depth = queue.popleft()
            if depth > self.max_depth:
                continue
            clean = canonical_url(url)
            if clean in self.visited:
                continue

            try:
                links = self._process_page(clean, depth)
                self.visited.add(clean)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                self._record_failure(clean, f"HTTP {status}")
                continue
            except requests.RequestException as exc:
                self._record_failure(clean, f"Network error: {exc}")
                continue
            except ValueError as exc:
                self._record_failure(clean, str(exc))
                continue
            except Exception as exc:
                self._record_failure(clean, str(exc))
                continue

            if self.delay:
                time.sleep(self.delay)

            for link in links:
                if self.is_valid(link) and link not in self.visited:
                    queue.append((link, depth + 1))

        elapsed = time.monotonic() - started_at
        self.save_visited()
        self.save_failed()
        self._print_summary(elapsed)

    def _print_summary(self, elapsed: float) -> None:
        logger.info("Done in %.1fs", elapsed)
        logger.info("Output directory: %s", self.output_dir)
        logger.info("Pages saved: %d", self.saved_count)
        logger.info("Pages skipped: %d", self.skipped_count)
        logger.info("URLs visited: %d", len(self.visited))
        logger.info("Failures: %d", len(self.failed_urls))
        if self.failed_urls:
            logger.info("Failed URL log: %s", self.failed_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Not in .env since it's a public site and generalization of the chatbot
    TARGET_SITE = "https://kubernetes.io/docs/concepts/"
    scraper = GraphRAGScraper(TARGET_SITE, max_depth=5)
    scraper.scrape(TARGET_SITE)
