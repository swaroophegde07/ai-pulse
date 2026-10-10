"""
Main ingestion pipeline for AI Pulse.

Orchestrates:
  1. Fetch RSS feeds → raw articles.
  2. Summarize new articles with Gemini.
  3. Merge with existing feed.json, deduplicate, sort, trim.
  4. Write feed.json atomically.

Usage:
  python -m pipeline.run        (from the repo root)
  python pipeline/run.py

Environment variables:
  GEMINI_API_KEY    Required for summarization.
  GEMINI_MODEL      Optional. Defaults to gemini-3.5-flash-lite.
  MAX_ARTICLES      Optional. Maximum articles to retain. Default: 200.
  FEED_PATH         Optional. Path to feed.json. Default: src/data/feed.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline.scraper import fetch_all_sources, normalize_url
from pipeline.sources import ALLOWED_CATEGORIES, SOURCES
from pipeline.summarizer import summarize_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_FEED_PATH = REPO_ROOT / "src" / "data" / "feed.json"
DEFAULT_MAX_ARTICLES = 200

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = frozenset(
    [
        "id",
        "title",
        "company",
        "source",
        "summary",
        "url",
        "date",
        "category",
        "fetched_at",
    ]
)


def validate_record(record: dict) -> bool:
    """Return True if *record* conforms to the feed schema."""
    if not isinstance(record, dict):
        return False
    missing = _REQUIRED_FIELDS - record.keys()
    if missing:
        logger.warning("Record missing fields %s: %r", missing, record.get("id"))
        return False
    if record.get("category") not in ALLOWED_CATEGORIES:
        logger.warning(
            "Record has invalid category '%s': %s",
            record.get("category"),
            record.get("id"),
        )
        return False
    if not isinstance(record.get("url"), str) or not record["url"].startswith("http"):
        logger.warning("Record has invalid URL: %r", record.get("id"))
        return False
    return True


# ---------------------------------------------------------------------------
# Feed I/O
# ---------------------------------------------------------------------------


def load_existing_feed(feed_path: Path) -> list[dict]:
    """Load and return existing records from feed.json.

    Returns an empty list if the file does not exist or is invalid JSON.
    Invalid individual records are skipped with a warning.
    """
    if not feed_path.exists():
        logger.info("No existing feed found at %s; starting fresh.", feed_path)
        return []

    try:
        with feed_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(
            "Could not read existing feed at %s: %s. Starting fresh.",
            feed_path,
            exc,
        )
        return []

    if not isinstance(data, list):
        logger.error("feed.json root is not a list; starting fresh.")
        return []

    valid = [r for r in data if validate_record(r)]
    if len(valid) < len(data):
        logger.warning(
            "Discarded %d invalid records from existing feed.",
            len(data) - len(valid),
        )
    return valid


def write_feed_atomic(records: list[dict], feed_path: Path) -> None:
    """Write *records* to *feed_path* atomically using a temp-file rename.

    If writing fails the existing file is not touched.
    """
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, indent=2, ensure_ascii=False)

    # Write to a sibling temp file then rename for atomicity.
    fd, tmp_path = tempfile.mkstemp(
        dir=feed_path.parent, prefix=".feed_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, feed_path)
    except Exception:
        # Clean up temp file on error; re-raise so callers see the failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Wrote %d records to %s", len(records), feed_path)


# ---------------------------------------------------------------------------
# Merge & deduplication
# ---------------------------------------------------------------------------


DEFAULT_RETENTION_DAYS = 90


def merge_records(
    existing: list[dict],
    new_articles: list[dict],
    max_articles: int,
    retention_days: int | None = None,
) -> list[dict]:
    """Merge *new_articles* into *existing*, deduplicate, sort, and trim.

    - Purges demonstration records so they do not pollute live data.
    - Deduplicates by stable ID and normalized canonical URL.
    - Applies configurable retention window for dated articles.
    - Sorts dated articles strictly by actual publication date descending.
    - Places undated articles after dated articles, ordered by fetched_at.
    - Trims to *max_articles*.
    """
    if retention_days is None:
        env_ret = os.environ.get("RETENTION_DAYS")
        retention_days = int(env_ret) if env_ret else 0

    # Exclude demo records from live dataset
    clean_existing = [r for r in existing if not r.get("_demo")]
    clean_new = [r for r in new_articles if not r.get("_demo")]

    seen_ids: dict[str, dict] = {r["id"]: r for r in clean_existing}
    seen_urls: dict[str, str] = {
        normalize_url(r["url"]): r["id"] for r in clean_existing
    }

    added = 0
    for article in clean_new:
        art_id = article["id"]
        art_url = normalize_url(article["url"])

        if art_id in seen_ids:
            continue
        if art_url in seen_urls:
            continue

        seen_ids[art_id] = article
        seen_urls[art_url] = art_id
        added += 1

    logger.info("Added %d new articles (before retention filter & trim).", added)

    all_records = list(seen_ids.values())

    # Apply retention window to dated articles if configured
    if retention_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        retained = []
        for r in all_records:
            date_str = r.get("date")
            if date_str:
                try:
                    pub_dt = datetime.fromisoformat(date_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=UTC)
                    if pub_dt < cutoff:
                        continue
                except ValueError:
                    pass
            retained.append(r)
        all_records = retained

    def sort_key(r: dict) -> tuple[int, datetime]:
        date_str = r.get("date") or ""
        if date_str:
            try:
                # Priority 1: dated articles, sorted by actual publication date
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return (1, dt)
            except ValueError:
                pass
        # Priority 0: articles without publication date, sorted by fetched_at
        fetched_str = r.get("fetched_at") or ""
        if fetched_str:
            try:
                f_dt = datetime.fromisoformat(fetched_str)
                if f_dt.tzinfo is None:
                    f_dt = f_dt.replace(tzinfo=UTC)
                return (0, f_dt)
            except ValueError:
                pass
        return (0, datetime.min.replace(tzinfo=UTC))

    all_records.sort(key=sort_key, reverse=True)
    return all_records[:max_articles]


# ---------------------------------------------------------------------------
# Article preparation
# ---------------------------------------------------------------------------


def prepare_article(raw: dict) -> dict:
    """Build a validated feed record from a raw + summarized article dict."""
    fetched_at = datetime.now(UTC).isoformat()
    return {
        "id": raw["id"],
        "title": raw.get("title", ""),
        "company": raw.get("company", ""),
        "source": raw.get("source", ""),
        "summary": raw.get("summary", ""),
        "url": raw.get("url", ""),
        "date": raw.get("date"),
        "category": raw.get("category", "News"),
        "fetched_at": fetched_at,
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    feed_path: Path | None = None,
    max_articles: int | None = None,
    sources=None,
) -> int:
    """Execute the full ingestion pipeline.

    Returns the number of new articles added. Raises on unrecoverable errors.
    """
    if feed_path is None:
        env_path = os.environ.get("FEED_PATH", "")
        feed_path = Path(env_path) if env_path else DEFAULT_FEED_PATH

    if max_articles is None:
        max_articles = int(os.environ.get("MAX_ARTICLES", DEFAULT_MAX_ARTICLES))

    if sources is None:
        sources = SOURCES

    logger.info("=== AI Pulse ingestion pipeline starting ===")
    logger.info("Feed path: %s", feed_path)
    logger.info("Max articles: %d", max_articles)

    # 1. Load existing feed.
    existing = load_existing_feed(feed_path)
    existing_ids = {r["id"] for r in existing}
    logger.info("Loaded %d existing articles.", len(existing))

    # 2. Fetch new articles from RSS feeds.
    raw_articles = fetch_all_sources(sources=sources)
    if not raw_articles:
        logger.warning("No articles fetched from any source. Exiting early.")
        return 0

    # 3. Summarize new (unseen) articles with Gemini.
    new_articles_raw = [a for a in raw_articles if a["id"] not in existing_ids]
    logger.info("%d new articles need summarization.", len(new_articles_raw))

    summarized = summarize_articles(new_articles_raw, existing_ids=existing_ids)

    # 4. Prepare final records.
    prepared: list[dict] = []
    for art in summarized:
        record = prepare_article(art)
        if validate_record(record):
            prepared.append(record)
        else:
            logger.warning("Skipping invalid record: %s", art.get("url"))

    # 5. Merge, sort, trim.
    merged = merge_records(existing, prepared, max_articles)

    # 6. Write atomically.
    write_feed_atomic(merged, feed_path)

    logger.info("=== Pipeline complete. Total records: %d ===", len(merged))
    return len(prepared)


if __name__ == "__main__":
    try:
        added = run_pipeline()
        sys.exit(0)
    except Exception as exc:
        logger.critical(
            "Pipeline failed with unrecoverable error: %s", exc, exc_info=True
        )
        sys.exit(1)
