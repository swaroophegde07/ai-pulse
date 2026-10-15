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
from pipeline.relevance import is_article_ai_relevant, split_mixed_newsletter

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
        r"^(utm_|fbclid|gclid|msclkid|ref|source|campaign|mc_cid|mc_eid|_hsenc|_hsmi|mkt_tok|vero_id)",
        re.IGNORECASE,
    )
    if parsed.query:
        cleaned_params = [
            p
            for p in parsed.query.split("&")
            if p and not _TRACKING_PARAMS.match(p.split("=")[0])
        ]
        # Sort query params for deterministic canonical URL
        cleaned_params.sort()
        query = "&".join(cleaned_params)
    else:
        query = ""

    canonical = urlunparse((scheme, netloc, path, parsed.params, query, ""))
    return canonical


def normalize_title(title: str) -> str:
    """Normalize article title for conservative duplicate detection.

    - Lowercases text.
    - Strips common publication suffixes (e.g. ' - The Verge', ' | TechCrunch').
    - Removes punctuation and extra whitespace.
    """
    if not title:
        return ""
    text = title.strip().lower()
    # Strip common publisher suffixes
    text = re.sub(
        r"\s*([|–—-]|by)\s*(the verge|techcrunch|ars technica|wired|venturebeat|reuters|bloomberg|mit technology review|the gradient|openai|anthropic|google deepmind|google blog|google|microsoft).*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Strip punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    return " ".join(text.split())


_COMMON_STOP_WORDS = frozenset([
    "a", "an", "the", "in", "on", "at", "for", "with", "and", "to", "is", "its", "it",
    "from", "by", "as", "of", "new", "how", "why", "what", "are", "be", "has", "have",
    "will", "can", "into", "over", "after", "about", "that", "this", "these", "those",
    "your", "our",
])

_GENERIC_ORGANIZATION_AND_AI = frozenset([
    "ai", "artificial", "intelligence", "model", "models", "google", "openai", "anthropic",
    "meta", "microsoft", "deepmind", "hugging", "face", "nvidia", "tech", "app", "apps",
])

_ORGANIZATION_NAMES = frozenset([
    "epic", "google", "apple", "amazon", "microsoft", "meta", "nvidia", "openai", "anthropic",
])

_MODEL_VARIANTS = frozenset([
    "sonnet", "haiku", "opus", "mini", "flash", "pro", "r1", "v3", "o1", "o3",
    "canvas", "sora", "large", "medium", "small", "base", "max", "ultra",
])

_SUBDOMAIN_ASPECTS = frozenset([
    "robotics", "robot", "robots", "safety", "safely", "responsibly", "alignment",
    "chip", "chips", "hardware", "preview", "evaluations", "evaluation", "eval", "evals",
    "red-teaming", "red team", "governance", "policy", "pricing", "insiders", "lawsuit", "homicide", "murder",
])

_RELEASE_ACTIONS = frozenset([
    "release", "releases", "released", "launch", "launches", "launched",
    "unveils", "unveil", "unveiled", "announces", "announced", "introducing", "introduced",
    "built", "available", "arrives", "debut", "debuts", "rolls out", "rolling out",
])


def are_articles_closely_related(
    art1: dict,
    art2: dict,
    max_hours: int = 48,
) -> bool:
    """Detect if two articles are closely related coverage of the same underlying development.

    Groups related sources where appropriate without deleting genuinely distinct follow-up stories:
    - Must be within max_hours publication window (or undated).
    - Distinct software versions (e.g. 1.141 vs 1.140) are NEVER grouped.
    - Distinct specific model variants (e.g. Sonnet vs Haiku, Flash vs Pro, GPT-4 vs o1) are NEVER grouped.
    - Distinct subdomains/aspects (e.g. robotics vs general release, safety vs general release) are NEVER grouped.
    - Distinct organization subjects (e.g. Epic Games vs Google) are NEVER grouped.
    - Matches exact normalized titles, high Jaccard token overlap (>= 0.50), Dice overlap (>= 0.60),
      or shared specific model compound release coverage (e.g. Gemini 4 Argon release + The Verge coverage).
    """
    if not dates_compatible(art1.get("date"), art2.get("date"), max_hours=max_hours):
        return False

    t1 = normalize_title(art1.get("title", ""))
    t2 = normalize_title(art2.get("title", ""))
    if not t1 or not t2:
        return False
    if t1 == t2:
        return True

    # Software / release version check (e.g. 1.141 vs 1.140, v0.6.3 vs v0.6.2)
    raw_t1 = (art1.get("title") or "").lower()
    raw_t2 = (art2.get("title") or "").lower()
    v1 = set(re.findall(r"\bv?(\d+(?:\.\d+)+)\b", raw_t1))
    v2 = set(re.findall(r"\bv?(\d+(?:\.\d+)+)\b", raw_t2))
    if v1 and v2 and v1 != v2:
        return False

    words1 = [w for w in t1.split() if w not in _COMMON_STOP_WORDS and len(w) > 1]
    words2 = [w for w in t2.split() if w not in _COMMON_STOP_WORDS and len(w) > 1]
    if len(words1) < 2 or len(words2) < 2:
        return False

    set1, set2 = set(words1), set(words2)

    # Distinct organization subjects (e.g. Epic vs Google)
    o1 = set1 & _ORGANIZATION_NAMES
    o2 = set2 & _ORGANIZATION_NAMES
    if o1 and o2 and o1 != o2:
        return False

    # Distinct model variant check: prevent merging different models from the same family
    m1 = set1 & _MODEL_VARIANTS
    m2 = set2 & _MODEL_VARIANTS
    if m1 and m2 and m1 != m2:
        return False

    # Distinct subdomain/aspect check (e.g. Robotics vs general release, Safety vs general release)
    s1 = set1 & _SUBDOMAIN_ASPECTS
    s2 = set2 & _SUBDOMAIN_ASPECTS
    if s1 != s2:
        return False

    intersection = set1 & set2
    union = set1 | set2
    jaccard = len(intersection) / len(union)
    dice = (2.0 * len(intersection)) / (len(set1) + len(set2))

    if jaccard >= 0.50 or dice >= 0.60:
        return True

    # Shared specific model compound release coverage (excluding generic terms like 'ai', 'google', 'model')
    specific_inter = intersection - _GENERIC_ORGANIZATION_AND_AI
    if len(specific_inter) >= 2 or any(entity in specific_inter for entity in ["argon", "retrochimera", "skala"]):
        has_release = bool((set1 & _RELEASE_ACTIONS) or (set2 & _RELEASE_ACTIONS))
        if has_release and not s1 and not s2:
            return True

    return False


def are_titles_duplicate(title1: str, title2: str) -> bool:
    """Determine if two article titles indicate the exact same reporting.

    Uses conservative matching:
    - Exact match after normalization.
    - Or Jaccard token overlap >= 0.85 when both titles have at least 5 words.
    - Respects distinct model variants.
    """
    n1 = normalize_title(title1)
    n2 = normalize_title(title2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True

    words1 = set(n1.split())
    words2 = set(n2.split())

    # Distinct model variant check
    model_variants = {
        "sonnet", "haiku", "opus", "mini", "flash", "pro", "r1", "v3", "o1", "o3",
        "canvas", "sora", "large", "medium", "small", "base", "max", "ultra",
    }
    m1 = words1 & model_variants
    m2 = words2 & model_variants
    if m1 and m2 and m1 != m2:
        return False

    if len(words1) >= 5 and len(words2) >= 5:
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)
        return jaccard >= 0.85

    return False


def stable_id(url: str) -> str:
    """Return a short, stable identifier derived from the canonical URL."""
    canonical = normalize_url(url)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


def dates_compatible(d1_str: str | None, d2_str: str | None, max_hours: int = 48) -> bool:
    """Check if two publication dates are within max_hours of each other (or undated)."""
    if not d1_str or not d2_str:
        return True
    try:
        dt1 = datetime.fromisoformat(d1_str)
        dt2 = datetime.fromisoformat(d2_str)
        if dt1.tzinfo is None:
            dt1 = dt1.replace(tzinfo=UTC)
        if dt2.tzinfo is None:
            dt2 = dt2.replace(tzinfo=UTC)
        return abs((dt1 - dt2).total_seconds()) <= max_hours * 3600
    except Exception:
        return True


def merge_source_into(target_rec: dict, source_art: dict) -> None:
    """Merge a source article's link and metadata into target_rec without duplicates."""
    sources = target_rec.setdefault("sources", [])
    seen_urls = {
        normalize_url(s.get("url", ""))
        for s in sources
        if isinstance(s, dict) and s.get("url")
    }

    src_url = source_art.get("url") or ""
    norm_src_url = normalize_url(src_url)
    if norm_src_url and norm_src_url not in seen_urls:
        seen_urls.add(norm_src_url)
        sources.append({
            "title": source_art.get("title", target_rec.get("title", "")),
            "url": src_url,
            "publisher": source_art.get("company", source_art.get("source", "")),
        })

    # Merge any sources list already present on source_art
    for s in source_art.get("sources") or []:
        if isinstance(s, dict) and s.get("url"):
            norm_s_url = normalize_url(s["url"])
            if norm_s_url and norm_s_url not in seen_urls:
                seen_urls.add(norm_s_url)
                sources.append(s)

    # Merge tags
    existing_tags = list(target_rec.get("tags") or target_rec.get("topics") or [])
    new_tags = list(source_art.get("tags") or source_art.get("topics") or [])
    for t in new_tags:
        if t and t not in existing_tags and len(existing_tags) < 5:
            existing_tags.append(t)
    if existing_tags:
        target_rec["tags"] = existing_tags

    # Preserve key_points if target lacks them
    if not target_rec.get("key_points") and source_art.get("key_points"):
        target_rec["key_points"] = source_art["key_points"]

    # Preserve highest impact score
    s1 = target_rec.get("impact_score")
    s2 = source_art.get("impact_score")
    if s1 or s2:
        target_rec["impact_score"] = max(s1 or 0, s2 or 0)


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

        candidate = {
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

        # Handle mixed newsletters or single stories through the relevance gate
        sub_articles = split_mixed_newsletter(candidate)
        for art in sub_articles:
            is_rel, reason = is_article_ai_relevant(
                art["title"],
                art.get("text", ""),
                tags=[art["category"]],
            )
            if not is_rel:
                logger.info("Filtered non-AI article: '%s' (%s)", art["title"], reason)
                continue
            articles.append(art)

    logger.info("Fetched %d relevant AI articles from '%s'", len(articles), name)
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
