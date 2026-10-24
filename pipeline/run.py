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
  MAX_ARTICLES      Optional. Maximum articles to retain. Default: None (unlimited).
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

from pipeline.relevance import _UNRELATED_DISTRACTORS, is_article_ai_relevant
from pipeline.scraper import (
    are_articles_closely_related,
    are_titles_duplicate,
    dates_compatible,
    fetch_all_sources,
    merge_source_into,
    normalize_title,
    normalize_url,
)
from pipeline.sources import ALLOWED_CATEGORIES, SOURCES
from pipeline.summarizer import (
    CircuitBreaker,
    RateLimiter,
    _get_client,
    is_generic_why_it_matters,
    is_key_point_generic,
    summarize_article,
    summarize_articles,
    validate_key_points,
)

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
DEFAULT_STATUS_PATH = REPO_ROOT / "src" / "data" / "pipeline-status.json"
DEFAULT_MAX_ARTICLES: int | None = None

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

    # Optional fields validation
    if "tags" in record and not isinstance(record["tags"], list):
        logger.warning("Record has invalid tags type: %r", record.get("id"))
        return False
    if "key_points" in record and not isinstance(record["key_points"], list):
        logger.warning("Record has invalid key_points type: %r", record.get("id"))
        return False
    if "impact_score" in record and record["impact_score"] is not None:
        if not isinstance(record["impact_score"], (int, float)) or not (1 <= record["impact_score"] <= 10):
            logger.warning("Record has invalid impact_score: %r", record.get("id"))
            return False
    if "sources" in record and not isinstance(record["sources"], list):
        logger.warning("Record has invalid sources type: %r", record.get("id"))
        return False

    return True


# ---------------------------------------------------------------------------
# Feed I/O
# ---------------------------------------------------------------------------


def load_existing_feed(feed_path: Path) -> list[dict]:
    """Load and return existing records from feed.json.

    Returns an empty list if the file does not exist.
    If the file exists but contains invalid JSON or has a non-list root,
    raises RuntimeError to fail safely and prevent silent loss of historical data.
    Invalid individual records are skipped with a warning.
    """
    if not feed_path.exists():
        logger.info("No existing feed found at %s; starting fresh.", feed_path)
        return []

    try:
        with feed_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.critical(
            "Could not read existing feed at %s: %s. Aborting to protect historical data.",
            feed_path,
            exc,
        )
        raise RuntimeError(
            f"Existing feed at {feed_path} contains invalid JSON ({exc}). "
            "Aborting run to prevent silent loss of historical articles."
        ) from exc

    if not isinstance(data, list):
        logger.critical(
            "Existing feed at %s root is not a list. Aborting to protect historical data.",
            feed_path,
        )
        raise RuntimeError(
            f"Existing feed at {feed_path} root is not a list. "
            "Aborting run to prevent silent loss of historical articles."
        )

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


_merge_source_into = merge_source_into
_dates_compatible = dates_compatible


def merge_records(
    existing: list[dict],
    new_articles: list[dict],
    max_articles: int | None = None,
    retention_days: int | None = None,
    filter_relevance: bool = False,
) -> list[dict]:
    """Merge *new_articles* into *existing*, deduplicate, sort, and trim.

    - Purges demonstration records so they do not pollute live data.
    - Deduplicates by stable ID and normalized canonical URL.
    - Identifies obvious duplicate reporting via normalized title comparison.
    - Merges secondary coverage into the primary article's sources list without duplicate URLs.
    - Applies configurable retention window for dated articles.
    - Sorts dated articles strictly by actual publication date descending.
    - Places undated articles after dated articles, ordered by fetched_at.
    - Trims to *max_articles* if provided and positive (> 0); otherwise preserves all records.
    """
    if retention_days is None:
        env_ret = os.environ.get("RETENTION_DAYS")
        retention_days = int(env_ret) if env_ret else 0

    def _is_verge(rec: dict) -> bool:
        url = normalize_url(rec.get("url") or "")
        company = (rec.get("company") or "").lower()
        source = (rec.get("source") or "").lower()
        return "theverge.com" in url or "the verge" in company or "the verge" in source

    # Exclude demo records and The Verge from live dataset
    clean_existing = [r for r in existing if not r.get("_demo") and not _is_verge(r)]
    clean_new = [
        r for r in new_articles
        if not r.get("_demo") and not _is_verge(r) and (
            not filter_relevance
            or is_article_ai_relevant(
                r.get("title", ""),
                r.get("summary", "") or r.get("text", ""),
                source_name=r.get("source", ""),
            )[0]
        )
    ]

    # Ensure records have sources initialized and strip any The Verge secondary references
    for r in clean_existing:
        if "sources" not in r or not r["sources"]:
            r["sources"] = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "publisher": r.get("company", r.get("source", "")),
                }
            ]
        else:
            r["sources"] = [
                s for s in r["sources"]
                if isinstance(s, dict)
                and "theverge.com" not in normalize_url(s.get("url") or "")
                and "the verge" not in (s.get("publisher") or "").lower()
            ]

    for r in clean_new:
        if "sources" in r and r["sources"]:
            r["sources"] = [
                s for s in r["sources"]
                if isinstance(s, dict)
                and "theverge.com" not in normalize_url(s.get("url") or "")
                and "the verge" not in (s.get("publisher") or "").lower()
            ]

    seen_ids: dict[str, dict] = {r["id"]: r for r in clean_existing}
    seen_urls: dict[str, str] = {
        normalize_url(r["url"]): r["id"] for r in clean_existing
    }

    added = 0
    updated = 0
    merged_duplicates = 0

    for article in clean_new:
        art_id = article["id"]
        art_url = normalize_url(article["url"])

        # 1. Exact ID match
        if art_id in seen_ids:
            existing_rec = seen_ids[art_id]
            if article.get("summary_status") == "generated" and existing_rec.get("summary_status") != "generated":
                existing_rec.update(article)
                updated += 1
            _merge_source_into(existing_rec, article)
            continue

        # 2. Exact URL match
        if art_url in seen_urls:
            matched_id = seen_urls[art_url]
            existing_rec = seen_ids[matched_id]
            if article.get("summary_status") == "generated" and existing_rec.get("summary_status") != "generated":
                existing_rec.update(article)
                updated += 1
            _merge_source_into(existing_rec, article)
            continue

        # 3. Duplicate or closely related coverage of same underlying development
        found_related = False
        art_title = article.get("title", "")
        art_date = article.get("date")

        for existing_id, existing_rec in list(seen_ids.items()):
            if are_articles_closely_related(article, existing_rec):
                # Closely related coverage of same development — group into sources
                _merge_source_into(existing_rec, article)
                # If incoming article has AI summary and existing doesn't, upgrade
                if article.get("summary_status") == "generated" and existing_rec.get("summary_status") != "generated":
                    existing_rec["summary"] = article["summary"]
                    existing_rec["summary_status"] = "generated"
                    if article.get("key_points"):
                        existing_rec["key_points"] = article["key_points"]
                merged_duplicates += 1
                found_related = True
                break

        if found_related:
            continue

        # 4. New unique article
        if "sources" not in article or not article["sources"]:
            article["sources"] = [
                {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "publisher": article.get("company", article.get("source", "")),
                }
            ]
        seen_ids[art_id] = article
        seen_urls[art_url] = art_id
        added += 1

    logger.info(
        "Added %d new articles, updated %d existing, merged %d cross-source duplicates (before retention filter & trim).",
        added,
        updated,
        merged_duplicates,
    )

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
    if max_articles is not None and max_articles > 0:
        return all_records[:max_articles]
    return all_records


def reconcile_existing_feed(
    feed_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Audit and reconcile existing records in feed.json.

    - Removes confirmed false positives and non-AI distractors (e.g. 'The Download: drone traffic controllers').
    - Cleans URL fragment identifiers without breaking canonical URLs.
    - Groups related coverage of identical events (e.g. Gemini 4 Argon general announcement + The Verge media coverage)
      into secondary sources ('Also reported by').
    - Strictly preserves distinct follow-ups (robotics, safety, alignment) and valid research.
    - Fully idempotent across repeated executions.

    Returns:
        Audit report dict with counts and details.
    """
    if feed_path is None:
        env_path = os.environ.get("FEED_PATH", "")
        feed_path = Path(env_path) if env_path else DEFAULT_FEED_PATH

    records = load_existing_feed(feed_path)
    total_initial = len(records)
    logger.info("Starting feed reconciliation on %d records at %s...", total_initial, feed_path)

    # 1. Clean fragment URLs & filter confirmed non-AI distractor false positives
    cleaned: list[dict] = []
    removed_records: list[dict[str, str]] = []
    fragment_urls_cleaned = 0

    for r in records:
        # Clean fragment from URL if present
        url = r.get("url", "")
        if "#" in url:
            clean_url = url.split("#")[0]
            r["url"] = clean_url
            fragment_urls_cleaned += 1
            if "sources" in r and isinstance(r["sources"], list):
                for s in r["sources"]:
                    if isinstance(s, dict) and "#" in s.get("url", ""):
                        s["url"] = s["url"].split("#")[0]

        title = r.get("title", "")
        summary = r.get("summary", "") or r.get("text", "")
        source = r.get("source", "")
        tags = r.get("tags")

        is_rel, reason = is_article_ai_relevant(
            title,
            text=summary,
            tags=tags,
            source_name=source,
        )

        distractor_match = _UNRELATED_DISTRACTORS.search(title)
        # Confirmed distractor without AI centrality
        if not is_rel and distractor_match:
            removed_records.append({"id": r.get("id", ""), "title": title, "reason": reason})
            logger.info("Reconciliation removed distractor: '%s' (%s)", title, reason)
            continue

        # Known false positive newsletter segment
        if r.get("id") == "8b9896f1d469636c":
            removed_records.append({"id": r["id"], "title": title, "reason": "Explicit non-AI mixed newsletter false positive"})
            logger.info("Reconciliation removed known false positive: '%s'", title)
            continue

        cleaned.append(r)

    # 2. Conservative event-level deduplication / source grouping
    grouped: list[dict] = []
    merged_count = 0
    merged_details: list[dict[str, str]] = []

    for item in cleaned:
        merged = False
        for existing in grouped:
            if are_articles_closely_related(existing, item):
                # Prefer lab announcement as primary representation
                if item.get("source_type") in ["Primary Source", "Research Laboratory"] and existing.get("source_type") not in ["Primary Source", "Research Laboratory"]:
                    _merge_source_into(item, existing)
                    grouped[grouped.index(existing)] = item
                    primary_rec, secondary_rec = item, existing
                else:
                    _merge_source_into(existing, item)
                    primary_rec, secondary_rec = existing, item

                merged_count += 1
                merged = True
                merged_details.append({
                    "primary_id": primary_rec.get("id", ""),
                    "primary_title": primary_rec.get("title", ""),
                    "secondary_id": secondary_rec.get("id", ""),
                    "secondary_title": secondary_rec.get("title", ""),
                })
                logger.info(
                    "Grouped secondary coverage '%s' into primary '%s'",
                    secondary_rec.get("title"),
                    primary_rec.get("title"),
                )
                break

        if not merged:
            grouped.append(item)

    report = {
        "feed_path": str(feed_path),
        "total_evaluated": total_initial,
        "total_retained": len(grouped),
        "total_removed": len(removed_records),
        "removed_records": removed_records,
        "total_merged_coverage": merged_count,
        "merged_coverage": merged_details,
        "fragment_urls_cleaned": fragment_urls_cleaned,
    }

    if not dry_run:
        write_feed_atomic(grouped, feed_path)
        logger.info(
            "Feed reconciliation complete. Retained %d records (%d removed, %d merged).",
            len(grouped),
            len(removed_records),
            merged_count,
        )

    return report


# ---------------------------------------------------------------------------
# Article preparation
# ---------------------------------------------------------------------------


def derive_article_metadata(raw: dict) -> dict:
    """Enrich raw article with topics, source type, priority classification, and why it matters."""
    title = (raw.get("title") or "")
    summary = (raw.get("summary") or "")
    content = f"{title} {summary}".lower()
    cat = raw.get("category", "News")
    company = raw.get("company", "")
    source_name = raw.get("source", "")

    # Topic taxonomy keywords
    topic_rules = {
        "Models and Releases": ["model", "weights", "checkpoint", "released", "release", "gemini", "claude", "gpt", "llama", "deepseek", "nemotron", "gemma", "qwen", "falcon", "frontier"],
        "AI Agents": ["agent", "agentic", "tool use", "reasoning loop", "multi-agent", "orchestration"],
        "Coding and Developer Tools": ["code", "coding", "developer", "copilot", "vs code", "programming", "github", "ide", "debugging", "python", "cluster", "gpu"],
        "Research and Breakthroughs": ["paper", "arxiv", "breakthrough", "algorithm", "attention", "transformer", "fine-tuning", "pre-training", "mathematics", "olympiad", "ioi", "imo"],
        "Benchmarks and Evaluations": ["benchmark", "evaluation", "eval", "leaderboard", "swe-bench", "mmlu", "livebench", "gold-level", "accuracy"],
        "Safety and Security": ["safety", "alignment", "security", "guardrail", "watermark", "synthid", "vulnerability", "red team"],
        "Policy and Regulation": ["policy", "regulation", "copyright", "eu ai act", "ftc", "antitrust", "governance"],
        "Business and Industry": ["startup", "funding", "enterprise", "acquisition", "market", "partnership"],
        "Open Source": ["open source", "open-weights", "open weights", "apache 2.0", "hugging face", "github", "mit license"],
        "Infrastructure and Hardware": ["gpu", "cluster", "nvidia", "blackwell", "hopper", "tpu", "datacenter", "scheduling", "inference", "vllm"],
        "Applications and Products": ["product", "app", "feature", "assistant", "chatgpt", "search", "copilot"]
    }

    matched_topics = [t for t, kws in topic_rules.items() if any(kw in content for kw in kws)]
    if not matched_topics:
        if cat == "Product Release":
            matched_topics = ["Models and Releases"]
        elif cat == "Research":
            matched_topics = ["Research and Breakthroughs"]
        else:
            matched_topics = ["Business and Industry"]

    # Priority scoring
    score = 0
    reasons = []

    model_terms = ["introducing", "released", "release", "gemini", "claude", "gpt", "llama", "deepseek", "nemotron", "gemma", "qwen", "weights", "open model"]
    if any(t in title.lower() for t in model_terms) or cat == "Product Release":
        score += 3
        reasons.append("Announces a model or architecture capability")

    bench_terms = ["benchmark", "reasoning", "olympiad", "ioi", "imo", "swe-bench", "evaluation", "results", "gold-level"]
    if any(t in content for t in bench_terms) or cat == "Benchmark":
        score += 3
        reasons.append("Reports measured benchmark or evaluation results")

    if cat == "Research":
        score += 2
        reasons.append("Original research or technical finding")

    if company in ["Google DeepMind", "OpenAI", "Anthropic", "Meta AI"]:
        score += 1
        reasons.append(f"Primary lab announcement from {company}")

    if score >= 5:
        priority_label = "Deep Dive"
    elif score >= 3:
        priority_label = "Must Know"
    else:
        priority_label = "Worth a Look"

    explanation = reasons[0] if reasons else "Relevant AI development"

    raw_wim = raw.get("why_it_matters")
    if raw_wim and isinstance(raw_wim, str) and not is_generic_why_it_matters(raw_wim):
        why_it_matters = raw_wim.strip()
    else:
        why_it_matters = ""

    # Map source type
    source_type = "Technical Publication"
    if company in ["OpenAI", "Anthropic", "Meta AI"]:
        source_type = "Primary Source"
    elif company in ["Google DeepMind", "Microsoft Research"]:
        source_type = "Research Laboratory"
    elif "arxiv" in source_name.lower():
        source_type = "Research or Preprint"
    elif "verge" in source_name.lower() or "ars technica" in source_name.lower() or "mit" in source_name.lower():
        source_type = "Independent Reporting"

    return {
        "topics": matched_topics,
        "source_type": source_type,
        "priority_score": score,
        "priority_label": priority_label,
        "priority_explanation": explanation,
        "why_it_matters": why_it_matters,
    }


def prepare_article(raw: dict) -> dict:
    """Build a validated feed record from a raw + summarized article dict."""
    fetched_at = datetime.now(UTC).isoformat()
    meta = derive_article_metadata(raw)

    # Validate key_points (strictly exactly 3 non-empty, non-generic strings)
    valid_key_points: list[str] | None = None
    raw_kp = raw.get("key_points")
    if isinstance(raw_kp, list) and validate_key_points(raw_kp):
        valid_key_points = [str(p).strip() for p in raw_kp]

    # Validate tags (list of short, useful strings)
    valid_tags: list[str] | None = None
    raw_tags = raw.get("tags")
    if isinstance(raw_tags, list):
        cleaned_tags = []
        for t in raw_tags:
            if isinstance(t, str):
                s = t.strip()
                if 1 <= len(s) <= 50 and s.lower() not in [x.lower() for x in cleaned_tags]:
                    cleaned_tags.append(s)
        if cleaned_tags:
            valid_tags = cleaned_tags

    # Validate impact_score (integer from 1 to 10 when available)
    valid_impact_score: int | None = None
    raw_impact = raw.get("impact_score")
    if raw_impact is not None:
        try:
            val = int(raw_impact)
            if 1 <= val <= 10:
                valid_impact_score = val
        except (ValueError, TypeError):
            pass

    # Validate sources with primary-source fallback
    valid_sources: list[dict] = []
    raw_sources = raw.get("sources")
    if isinstance(raw_sources, list):
        seen_urls = set()
        for s in raw_sources:
            if isinstance(s, dict) and isinstance(s.get("url"), str) and s["url"].startswith("http"):
                u = s["url"].strip()
                if u not in seen_urls:
                    seen_urls.add(u)
                    src_item: dict[str, str] = {"url": u}
                    if s.get("title") and isinstance(s["title"], str):
                        src_item["title"] = s["title"].strip()
                    if s.get("publisher") and isinstance(s["publisher"], str):
                        src_item["publisher"] = s["publisher"].strip()
                    elif s.get("company") and isinstance(s["company"], str):
                        src_item["publisher"] = s["company"].strip()
                    valid_sources.append(src_item)
    if not valid_sources:
        primary_url = raw.get("url", "")
        if isinstance(primary_url, str) and primary_url.startswith("http"):
            valid_sources = [
                {
                    "title": raw.get("title", ""),
                    "url": primary_url,
                    "publisher": raw.get("company") or raw.get("source") or "Primary Source",
                }
            ]

    raw_wim = raw.get("why_it_matters")
    resolved_wim = raw_wim if (raw_wim and isinstance(raw_wim, str) and not is_generic_why_it_matters(raw_wim)) else meta["why_it_matters"]

    record: dict[str, Any] = {
        "id": raw["id"],
        "title": raw.get("title", ""),
        "company": raw.get("company", ""),
        "source": raw.get("source", ""),
        "summary": raw.get("summary", ""),
        "summary_status": raw.get("summary_status", "generated" if raw.get("summary") and not raw.get("summary", "").startswith("[Automated") else "pending_quota"),
        "content_fingerprint": raw.get("content_fingerprint", ""),
        "url": raw.get("url", ""),
        "date": raw.get("date"),
        "category": raw.get("category", "News"),
        "fetched_at": fetched_at,
        "topics": raw.get("topics") or meta["topics"],
        "source_type": raw.get("source_type") or meta["source_type"],
        "priority_score": raw.get("priority_score", meta["priority_score"]),
        "priority_label": raw.get("priority_label", meta["priority_label"]),
        "priority_explanation": raw.get("priority_explanation", meta["priority_explanation"]),
        "why_it_matters": resolved_wim,
    }

    if valid_key_points is not None:
        record["key_points"] = valid_key_points
    if valid_tags is not None:
        record["tags"] = valid_tags
    if valid_impact_score is not None:
        record["impact_score"] = valid_impact_score
    if valid_sources:
        record["sources"] = valid_sources

    return record


# ---------------------------------------------------------------------------
# Date Window & Recent TL;DR Backfill
# ---------------------------------------------------------------------------


def is_article_eligible_in_window(
    date_str: str | None,
    window_hours: float,
    current_dt: datetime | None = None,
) -> bool:
    """Return True if article publication timestamp is valid, not in the future, and within window_hours."""
    if not date_str or not isinstance(date_str, str):
        return False
    try:
        pub_dt = datetime.fromisoformat(date_str)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return False

    now = current_dt or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # Exclude future-dated articles (not later than current time)
    if pub_dt > now:
        return False

    # Check window boundary
    return (now - pub_dt).total_seconds() <= window_hours * 3600


def backfill_recent_tldrs(
    feed_records: list[dict] | None = None,
    feed_path: Path | None = None,
    client: Any = None,
    max_backfill: int | None = None,
    days: int = 7,
    current_dt: datetime | None = None,
) -> tuple[int, int]:
    """Safely backfill missing or invalid TL;DR key_points for recent existing articles.

    Requirements:
    - Bounded to articles published within the past `days` (default 7 days) and not future-dated.
    - Only updates missing or invalid key_points.
    - Preserves existing summaries, URLs, tags, impact scores, source links, and other metadata.
    - Reuses the current Gemini summarizer and its quota safeguards (RateLimiter, CircuitBreaker).
    - Uses existing generic-content validation (validate_key_points).
    - If Gemini fails or quota is exhausted, preserves the existing article and leaves TL;DR unavailable.
    - Writes the updated feed atomically.
    - Safe to rerun; skips articles that already have valid TL;DR points without duplicate API calls.

    Returns:
        (updated_count, skipped_count)
    """
    target_feed_path = feed_path or DEFAULT_FEED_PATH
    if feed_records is None:
        records = load_existing_feed(target_feed_path)
    else:
        records = feed_records

    if not records:
        return (0, 0)

    now = current_dt or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    window_hours = float(days * 24)

    # 1. Identify candidates published within past `days` window, not future-dated, lacking valid TL;DR
    candidates: list[dict] = []
    for art in records:
        d_str = art.get("date")
        if not is_article_eligible_in_window(d_str, window_hours, current_dt=now):
            continue

        existing_kp = art.get("key_points")
        if existing_kp and validate_key_points(existing_kp):
            # Already has valid, verified TL;DR points -> skip without calling API
            continue

        candidates.append(art)

    if not candidates:
        logger.info("No recent articles within the past %d days require TL;DR backfill.", days)
        return (0, 0)

    # 2. Sort candidates newest first
    def get_pub_dt(r: dict) -> datetime:
        try:
            d = datetime.fromisoformat(r["date"])
            return d if d.tzinfo else d.replace(tzinfo=UTC)
        except Exception:
            return datetime.min.replace(tzinfo=UTC)

    candidates.sort(key=get_pub_dt, reverse=True)

    # 3. Apply bounds
    if max_backfill is None:
        env_max = os.environ.get("GEMINI_MAX_BACKFILL_PER_RUN")
        max_backfill = int(env_max) if env_max else 20

    to_process = candidates[:max_backfill]
    logger.info(
        "Found %d candidate articles lacking TL;DR from past %d days; processing bounded batch of %d.",
        len(candidates),
        days,
        len(to_process),
    )

    # 4. Initialize Gemini client (safely handle missing credential)
    if client is None:
        try:
            client = _get_client()
        except ValueError as exc:
            logger.warning("Gemini client initialization skipped for TL;DR backfill: %s", exc)
            return (0, len(candidates))

    # 5. Rate limiting & circuit breaker
    rpm = int(os.environ.get("GEMINI_MAX_REQUESTS_PER_MINUTE", "10"))
    limiter = RateLimiter(requests_per_minute=rpm)
    circuit = CircuitBreaker()

    updated_count = 0
    for art in to_process:
        if circuit.is_open:
            logger.warning("Circuit breaker is open; halting TL;DR backfill early to preserve quota.")
            break

        limiter.wait()

        article_input = {
            "id": art.get("id", ""),
            "title": art.get("title", ""),
            "text": art.get("text") or art.get("summary", ""),
            "url": art.get("url", ""),
            "category": art.get("category", "News"),
        }

        try:
            res = summarize_article(article_input, client=client, circuit_breaker=circuit)
            new_kp = res.get("key_points")
            if new_kp and validate_key_points(new_kp):
                # PRESERVE all existing article summaries, URLs, tags, impact scores, source links, etc.
                # ONLY update missing or invalid TL;DR fields
                art["key_points"] = new_kp
                updated_count += 1
            else:
                logger.info("No valid key_points generated for %s; preserving existing article.", art.get("id"))
        except Exception as exc:
            logger.warning("Error generating TL;DR backfill for %s: %s", art.get("id"), exc)

    # 6. Atomically write updated feed if any article was updated
    if updated_count > 0 and target_feed_path:
        write_feed_atomic(records, target_feed_path)
        logger.info("Successfully backfilled TL;DR for %d articles in %s.", updated_count, target_feed_path)

    return (updated_count, len(candidates) - updated_count)


# ---------------------------------------------------------------------------
# Pipeline Status Snapshot
# ---------------------------------------------------------------------------


def generate_pipeline_status(
    status_path: Path | None = None,
    feed_records: list[dict] | None = None,
    source_outcomes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate structured pipeline status snapshot recording health across feeds and benchmarks."""
    target_path = status_path or DEFAULT_STATUS_PATH
    now_iso = datetime.now(UTC).isoformat()

    # Load all sources including disabled ones, excluding The Verge
    from pipeline.sources import load_sources
    all_sources = load_sources(include_disabled=True)
    all_sources = [
        s for s in all_sources
        if "theverge.com" not in normalize_url(s.get("url") or "")
        and "the verge" not in (s.get("name") or "").lower()
        and "the verge" not in (s.get("publisher") or "").lower()
    ]

    feed_data = feed_records if feed_records is not None else load_existing_feed(DEFAULT_FEED_PATH)

    # RSS feed health summaries
    rss_list = []
    has_feed_failure = False

    for s in all_sources:
        s_name = s.get("name", "")
        matching_count = sum(1 for r in feed_data if r.get("source") == s_name)
        is_enabled = s.get("enabled", True) is not False

        if not is_enabled:
            status = "disabled"
        elif source_outcomes and s_name in source_outcomes:
            status = source_outcomes[s_name]
            if status in ("failure", "fetch_failed", "parse_failed"):
                has_feed_failure = True
        elif matching_count > 0:
            status = "active"
        else:
            status = "empty_response"

        rss_list.append({
            "name": s_name,
            "publisher": s.get("publisher", ""),
            "url": s.get("url", ""),
            "category": s.get("category", "News"),
            "status": status,
            "retained_articles_count": matching_count,
            "last_checked": now_iso,
        })

    # Read benchmarks.json metadata to reflect truthful upstream status
    benchmarks_path = REPO_ROOT / "src" / "data" / "benchmarks.json"
    bench_status = "active"
    livebench_status = "active"
    livebench_release = "2026-10-13"
    bench_health_warnings: list[str] = []
    bench_last_retrieval: str | None = None
    bench_last_eval_date: str | None = None

    if benchmarks_path.exists():
        try:
            with open(benchmarks_path, "r", encoding="utf-8") as bf:
                b_data = json.load(bf)
            b_meta = b_data.get("metadata", {})
            b_health = b_meta.get("health", {})
            meta_status = b_meta.get("status") or b_health.get("status") or "active"
            livebench_release = b_health.get("last_benchmark_release_date") or "2026-10-13"
            bench_last_retrieval = b_health.get("last_retrieval_time")
            bench_last_eval_date = b_health.get("last_successful_evaluation_date")
            bench_health_warnings = b_health.get("warnings") or []
            if meta_status in ("stale_fallback", "degraded", "fallback"):
                livebench_status = "stale_fallback"
                bench_status = "stale_fallback"
        except Exception as exc:
            logger.warning("Failed to read benchmarks metadata: %s", exc)

    # Benchmark and telemetry sources status
    bench_list = [
        {
            "name": "LiveBench",
            "type": "Benchmark Suite",
            "status": livebench_status,
            "release": livebench_release,
            "provenance": "Independent Evaluation",
            "url": "https://livebench.ai/",
            "license": "Apache-2.0",
            "last_retrieval_time": bench_last_retrieval,
            "last_successful_evaluation_date": bench_last_eval_date,
            "warnings": bench_health_warnings,
            "notes": (
                "Contamination-free evaluations with monthly updated problems across reasoning, coding, and mathematics."
                + (" (Preserved last-known-good fallback snapshot; live refresh degraded.)" if livebench_status == "stale_fallback" else "")
            ),
        },
        {
            "name": "OpenRouter Telemetry",
            "type": "Model Telemetry",
            "status": "active",
            "role": "Model catalogue & token pricing telemetry",
            "url": "https://openrouter.ai/",
            "notes": "Verified context lengths and input/output token pricing per 1M tokens.",
        },
        {
            "name": "BenchLM",
            "type": "Benchmark Suite",
            "status": "disabled",
            "reason": "Awaiting official public API endpoint and authentication.",
        },
        {
            "name": "Artificial Analysis",
            "type": "Benchmark Suite",
            "status": "disabled",
            "reason": "Requires ARTIFICIAL_ANALYSIS_API_KEY environment credential.",
        },
        {
            "name": "LM Council",
            "type": "Human Evaluation",
            "status": "disabled",
            "reason": "Source API deprecation or restructuring.",
        },
    ]

    overall_status = "degraded" if (bench_status == "stale_fallback" or has_feed_failure) else "healthy"
    enabled_sources_count = len([s for s in all_sources if s.get("enabled", True) is not False])

    status_payload: dict[str, Any] = {
        "generated_at": now_iso,
        "overall_status": overall_status,
        "feed_health": {
            "total_configured_sources": len(all_sources),
            "active_sources": enabled_sources_count,
            "total_retained_articles": len(feed_data),
        },
        "rss_sources": rss_list,
        "benchmark_sources": bench_list,
        "summary": {
            "total_rss_sources": len(all_sources),
            "active_benchmark_sources": 1 if livebench_status == "stale_fallback" else 2,
            "fallback_benchmark_sources": 1 if livebench_status == "stale_fallback" else 0,
            "disabled_benchmark_sources": 3,
        },
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(status_payload, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent, prefix=".status_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload_str)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Generated pipeline status snapshot at %s", target_path)
    return status_payload


def recover_historical_articles(feed_path: Path | None = None) -> tuple[int, int]:
    """Recover valid, relevant historical articles previously trimmed by MAX_ARTICLES cap.

    Scans git commit history of feed.json, extracts unique valid articles,
    filters for AI relevance, merges them into the current feed without an arbitrary cap,
    and updates the pipeline status snapshot.
    Returns (total_retained, newly_added).
    """
    import subprocess

    target_feed_path = feed_path or DEFAULT_FEED_PATH
    existing = load_existing_feed(target_feed_path)
    existing_ids = {r["id"] for r in existing if not r.get("_demo")}
    existing_urls = {normalize_url(r["url"]): r["id"] for r in existing if not r.get("_demo")}

    try:
        commits = subprocess.check_output(
            ["git", "log", "--format=%h", "--", str(target_feed_path)],
            cwd=str(REPO_ROOT),
        ).decode().split()
    except Exception as exc:
        logger.error("Failed to read git log for %s: %s", target_feed_path, exc)
        return len(existing), 0

    recovered: list[dict] = []
    seen_recovered_ids: set[str] = set()

    for c in commits:
        try:
            rel_path = target_feed_path.relative_to(REPO_ROOT)
            data_str = subprocess.check_output(
                ["git", "show", f"{c}:{rel_path}"],
                cwd=str(REPO_ROOT),
            ).decode()
            commit_feed = json.loads(data_str)
            if not isinstance(commit_feed, list):
                continue
            for art in commit_feed:
                if not isinstance(art, dict) or art.get("_demo"):
                    continue
                aid = art.get("id")
                norm_u = normalize_url(art.get("url", ""))
                if not aid or aid in existing_ids or norm_u in existing_urls or aid in seen_recovered_ids:
                    continue
                if not validate_record(art):
                    continue
                is_rel, _ = is_article_ai_relevant(
                    art.get("title", ""),
                    art.get("summary", "") or art.get("text", ""),
                    source_name=art.get("source", ""),
                )
                if not is_rel:
                    continue
                seen_recovered_ids.add(aid)
                recovered.append(art)
        except Exception:
            continue

    logger.info("Found %d historical articles eligible for recovery.", len(recovered))
    merged = merge_records(existing, recovered, max_articles=None, filter_relevance=True)
    write_feed_atomic(merged, target_feed_path)
    generate_pipeline_status(feed_records=merged)
    logger.info("Feed recovered. Total articles now: %d (added %d)", len(merged), len(merged) - len(existing))
    return len(merged), len(merged) - len(existing)


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
        env_max = os.environ.get("MAX_ARTICLES")
        if env_max and env_max.isdigit():
            max_articles = int(env_max)
        else:
            max_articles = DEFAULT_MAX_ARTICLES

    if sources is None:
        sources = SOURCES

    logger.info("=== AI Pulse ingestion pipeline starting ===")
    logger.info("Feed path: %s", feed_path)
    logger.info("Max articles: %s", max_articles if max_articles is not None else "Unlimited")

    # 1. Load existing feed.
    existing = load_existing_feed(feed_path)
    existing_ids = {r["id"] for r in existing}
    logger.info("Loaded %d existing articles.", len(existing))

    # 2. Fetch new articles from RSS feeds.
    raw_articles = fetch_all_sources(sources=sources)
    if not raw_articles:
        logger.warning("No articles fetched from any source. Exiting early.")
        return 0

    # 3. Identify candidate articles to summarize:
    # - Brand new articles not in existing_ids
    # - Previously deferred articles (pending_quota / unavailable) that need summarization
    candidate_articles: list[dict] = []
    for art in raw_articles:
        art_id = art["id"]
        existing_rec = next((r for r in existing if r.get("id") == art_id), None)
        if not existing_rec:
            candidate_articles.append(art)
        elif existing_rec.get("summary_status") in ["pending_quota", "unavailable"]:
            # Candidate for retry if quota has recovered
            candidate_articles.append(art)
        elif not existing_rec.get("summary") or existing_rec.get("summary", "").startswith("[Automated"):
            candidate_articles.append(art)

    logger.info("%d candidate articles evaluated for summarization/caching.", len(candidate_articles))

    summarized, metrics = summarize_articles(candidate_articles, existing_records=existing)

    # 4. Prepare final records.
    prepared: list[dict] = []
    for art in summarized:
        record = prepare_article(art)
        if validate_record(record):
            prepared.append(record)
        else:
            logger.warning("Skipping invalid record: %s", art.get("url"))

    # 5. Merge, sort, trim.
    merged = merge_records(existing, prepared, max_articles, filter_relevance=True)

    # 6. Safe, bounded backfill for recent existing articles missing valid TL;DR
    try:
        backfill_recent_tldrs(merged, feed_path)
    except Exception as bf_exc:
        logger.warning("Recent TL;DR backfill error (non-fatal): %s", bf_exc)

    # 7. Write atomically.
    write_feed_atomic(merged, feed_path)

    # 8. Run multi-source benchmark ingestion (fails safely without corrupting news feed).
    try:
        from pipeline.benchmarks.runner import run_benchmark_pipeline
        run_benchmark_pipeline()
    except Exception as b_exc:
        logger.warning("Benchmark pipeline execution error (non-fatal): %s", b_exc)

    # 8b. Run official model updates synchronization (fails safely without corrupting news feed).
    try:
        from pipeline.models.updates import sync_model_updates
        sync_model_updates()
    except Exception as u_exc:
        logger.warning("Model updates synchronization error (non-fatal): %s", u_exc)

    # 9. Generate pipeline status snapshot recording health across feeds and benchmarks.
    try:
        generate_pipeline_status(feed_records=merged)
    except Exception as s_exc:
        logger.warning("Pipeline status generation error (non-fatal): %s", s_exc)

    logger.info("=== Pipeline complete. Total records: %d ===", len(merged))
    return len(prepared)


if __name__ == "__main__":
    if "--backfill-tldr" in sys.argv:
        updated, remaining = backfill_recent_tldrs()
        logger.info("Backfill complete: %d updated, %d remaining/skipped.", updated, remaining)
        sys.exit(0)
    if "--reconcile-feed" in sys.argv:
        stats = reconcile_existing_feed()
        logger.info("Reconciliation complete: %s", stats)
        sys.exit(0)
    if "--recover-history" in sys.argv:
        total, added = recover_historical_articles()
        logger.info("History recovery complete: %d total records (%d added).", total, added)
        sys.exit(0)
    if "--update-model-notes" in sys.argv:
        from pipeline.models.updates import sync_model_updates
        total, added = sync_model_updates()
        logger.info("Model updates sync complete: %d total records (%d added).", total, added)
        sys.exit(0)
    try:
        added = run_pipeline()
        sys.exit(0)
    except Exception as exc:
        logger.critical(
            "Pipeline failed with unrecoverable error: %s", exc, exc_info=True
        )
        sys.exit(1)
