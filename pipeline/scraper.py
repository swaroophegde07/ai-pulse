"""
RSS scraper for AI Pulse.

Responsibilities:
  - Fetch configured RSS/Atom feeds using feedparser.
  - Extract a limited number of recent entries per source.
  - Normalize URLs and deduplicate within a single fetch run.
  - Return a list of raw article dicts ready for summarization.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse, urlunparse

import feedparser

from pipeline.sources import SOURCES, Source

logger = logging.getLogger(__name__)

# Maximum articles to pull from a single feed per run.
MAX_ENTRIES_PER_SOURCE: int = 10


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Return a canonical form of *url* suitable for stable deduplication.

    Transformations applied:
    - Strip leading/trailing whitespace.
    - Lower-case the scheme and host.
    - Remove the default port for http (80) and https (443).
    - Remove the URL fragment (#…).
    - Remove common tracking query parameters (utm_*, fbclid, etc.).
    - Collapse duplicate slashes in the path.
    - Remove a trailing slash from the path (except for root '/').
    """
    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Remove default ports.
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Clean path.
    path = re.sub(r"//+", "/", parsed.path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Strip tracking parameters.
    _TRACKING_PARAMS = re.compile(
        r"^(utm_|fbclid|gclid|msclkid|ref|source|campaign)",
        re.IGNORECASE,
    )
    if parsed.query:
        cleaned_params = [
            p
            for p in parsed.query.split("&")
            if not _TRACKING_PARAMS.match(p.split("=")[0])
        ]
        query = "&".join(cleaned_params)
    else:
        query = ""

    canonical = urlunparse((scheme, netloc, path, parsed.params, query, ""))
    return canonical


def stable_id(url: str) -> str:
    """Return a short, stable identifier derived from the canonical URL."""
    canonical = normalize_url(url)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_date(entry: Any) -> str | None:
    """Extract and normalise a publication date from a feedparser entry.

    Returns an ISO-8601 string (with UTC timezone) or None when no valid
    date can be determined.
    """
    # Validate against future-dated entries (clamped to max +1 day for timezone drift)
    now_utc = datetime.now(UTC)
    future_threshold = now_utc + timedelta(days=1)

    # feedparser provides published_parsed / updated_parsed as time.struct_time
    for attr in ("published_parsed", "updated_parsed"):
        tup = getattr(entry, attr, None)
        if tup:
            try:
                dt = datetime(*tup[:6], tzinfo=UTC)
                if dt > future_threshold:
                    logger.warning("Discarded future-dated timestamp: %s", dt)
                    return None
                return dt.isoformat()
            except Exception:
                pass

    # Fall back to raw string fields.
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None) or ""
        if raw:
            try:
                dt = parsedate_to_datetime(raw).astimezone(UTC)
                if dt > future_threshold:
                    logger.warning("Discarded future-dated raw timestamp: %s", dt)
                    return None
                return dt.isoformat()
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text(entry: Any) -> str:
    """Return the best available plain-text summary or content from an entry."""
    # Try content first (full article body, may contain HTML).
    content = getattr(entry, "content", None)
    if content and isinstance(content, list) and content[0].get("value"):
        text = content[0]["value"]
        # Strip obvious HTML tags for cleanliness.
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        return " ".join(text.split())[:4000]

    # Fall back to summary or description.
    summary = (
        getattr(entry, "summary", None) or getattr(entry, "description", None) or ""
    )
    if summary:
        summary = re.sub(r"<[^>]+>", " ", summary)
        summary = html.unescape(summary)
        return " ".join(summary.split())[:4000]

    return ""


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------


def fetch_feed(source: Source, max_entries: int = MAX_ENTRIES_PER_SOURCE) -> list[dict]:
    """Fetch *source* RSS feed and return a list of raw article dicts.

    Returns an empty list on failure (network error, parse error, etc.).
    Failures are logged as warnings so callers can continue with other feeds.
    """
    name = source["name"]
    url = source["url"]
    logger.info("Fetching feed: %s (%s)", name, url)

    try:
        # feedparser handles HTTP itself; set a user-agent and timeout via
        # the request_headers parameter.
        feed = feedparser.parse(
            url,
            request_headers={
                "User-Agent": "AI-Pulse/1.0 (+https://github.com/ai-pulse)",
                "Accept": "application/rss+xml, application/atom+xml, */*",
            },
        )
    except Exception as exc:
        logger.warning("Failed to fetch feed '%s': %s", name, exc)
        return []

    if feed.bozo:
        exc = feed.get("bozo_exception")
        logger.warning("Feed '%s' parsed with errors (bozo): %s", name, exc)
        # Still attempt to use entries if any were parsed.

    entries = feed.entries or []
    articles: list[dict] = []

    seen_urls: set[str] = set()

    for entry in entries[:max_entries]:
        raw_url = getattr(entry, "link", None) or ""
        if not raw_url:
            logger.debug("Skipping entry with no URL in feed '%s'", name)
            continue

        canon_url = normalize_url(raw_url)

        if canon_url in seen_urls:
            logger.debug("Duplicate URL within feed '%s': %s", name, canon_url)
            continue
        seen_urls.add(canon_url)

        article_id = stable_id(canon_url)
        title = getattr(entry, "title", "") or ""
        text = extract_text(entry)
        pub_date = parse_date(entry)

        articles.append(
            {
                "id": article_id,
                "title": title.strip(),
                "company": source["publisher"],
                "source": source["name"],
                "url": canon_url,
                "raw_url": raw_url,
                "date": pub_date,
                "category": source["category"],
                "text": text,
            }
        )

    logger.info("Fetched %d articles from '%s'", len(articles), name)
    return articles


def fetch_all_sources(
    sources: list[Source] | None = None,
    delay: float = 0.5,
) -> list[dict]:
    """Fetch all configured sources and return deduplicated raw articles.

    *delay* is the number of seconds to wait between feed requests to be
    polite to remote servers.

    Articles are deduplicated across sources by stable ID.
    """
    if sources is None:
        sources = SOURCES

    all_articles: list[dict] = []
    seen_ids: set[str] = set()

    for i, source in enumerate(sources):
        if i > 0:
            time.sleep(delay)
        articles = fetch_feed(source)
        for article in articles:
            if article["id"] in seen_ids:
                logger.debug("Cross-feed duplicate skipped: %s", article["url"])
                continue
            seen_ids.add(article["id"])
            all_articles.append(article)

    logger.info("Total unique articles fetched: %d", len(all_articles))
    return all_articles
