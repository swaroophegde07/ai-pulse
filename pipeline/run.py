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

from pipeline.scraper import are_titles_duplicate, fetch_all_sources, normalize_title, normalize_url
from pipeline.sources import ALLOWED_CATEGORIES, SOURCES
from pipeline.summarizer import (
    is_generic_why_it_matters,
    is_key_point_generic,
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


def _merge_source_into(target_rec: dict, source_art: dict) -> None:
    """Merge a source article's link and metadata into target_rec without duplicates."""
    sources = target_rec.setdefault("sources", [])
    seen_urls = {normalize_url(s.get("url", "")) for s in sources if isinstance(s, dict) and s.get("url")}

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


def _dates_compatible(d1_str: str | None, d2_str: str | None, max_hours: int = 48) -> bool:
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


def merge_records(
    existing: list[dict],
    new_articles: list[dict],
    max_articles: int,
    retention_days: int | None = None,
) -> list[dict]:
    """Merge *new_articles* into *existing*, deduplicate, sort, and trim.

    - Purges demonstration records so they do not pollute live data.
    - Deduplicates by stable ID and normalized canonical URL.
    - Identifies obvious duplicate reporting via normalized title comparison.
    - Merges secondary coverage into the primary article's sources list without duplicate URLs.
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

    # Ensure existing records have sources initialized
    for r in clean_existing:
        if "sources" not in r or not r["sources"]:
            r["sources"] = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "publisher": r.get("company", r.get("source", "")),
                }
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

        # 3. Obvious title duplicate of same event
        found_title_duplicate = False
        art_title = article.get("title", "")
        art_date = article.get("date")

        for existing_id, existing_rec in list(seen_ids.items()):
            if are_titles_duplicate(art_title, existing_rec.get("title", "")):
                if _dates_compatible(art_date, existing_rec.get("date")):
                    # Duplicate coverage of same event
                    _merge_source_into(existing_rec, article)
                    # If incoming article has AI summary and existing doesn't, upgrade
                    if article.get("summary_status") == "generated" and existing_rec.get("summary_status") != "generated":
                        existing_rec["summary"] = article["summary"]
                        existing_rec["summary_status"] = "generated"
                        if article.get("key_points"):
                            existing_rec["key_points"] = article["key_points"]
                    merged_duplicates += 1
                    found_title_duplicate = True
                    break

        if found_title_duplicate:
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
    return all_records[:max_articles]


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
# Pipeline Status Snapshot
# ---------------------------------------------------------------------------


def generate_pipeline_status(
    status_path: Path | None = None,
    feed_records: list[dict] | None = None,
) -> dict[str, Any]:
    """Generate structured pipeline status snapshot recording health across feeds and benchmarks."""
    target_path = status_path or DEFAULT_STATUS_PATH
    now_iso = datetime.now(UTC).isoformat()

    # RSS feed health summaries
    rss_list = []
    feed_data = feed_records or []
    for s in SOURCES:
        s_name = s.get("name", "")
        matching_count = sum(1 for r in feed_data if r.get("source") == s_name)
        rss_list.append({
            "name": s_name,
            "publisher": s.get("publisher", ""),
            "url": s.get("url", ""),
            "category": s.get("category", "News"),
            "status": "active" if s.get("enabled", True) else "disabled",
            "retained_articles_count": matching_count,
            "last_checked": now_iso,
        })

    # Benchmark and telemetry sources status
    bench_list = [
        {
            "name": "LiveBench",
            "type": "Benchmark Suite",
            "status": "active",
            "release": "2026-06-25",
            "provenance": "Independent Evaluation",
            "url": "https://livebench.ai/",
            "license": "Apache-2.0",
            "notes": "Contamination-free evaluations with monthly updated problems across reasoning, coding, and mathematics."
        },
        {
            "name": "OpenRouter Telemetry",
            "type": "Model Telemetry",
            "status": "active",
            "role": "Model catalogue & token pricing telemetry",
            "url": "https://openrouter.ai/",
            "notes": "Verified context lengths and input/output token pricing per 1M tokens."
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

    status_payload: dict[str, Any] = {
        "generated_at": now_iso,
        "overall_status": "healthy",
        "feed_health": {
            "total_configured_sources": len(SOURCES),
            "active_sources": len([s for s in SOURCES if s.get("enabled", True)]),
            "total_retained_articles": len(feed_data),
        },
        "rss_sources": rss_list,
        "benchmark_sources": bench_list,
        "summary": {
            "total_rss_sources": len(SOURCES),
            "active_benchmark_sources": 2,
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
    merged = merge_records(existing, prepared, max_articles)

    # 6. Write atomically.
    write_feed_atomic(merged, feed_path)

    # 7. Run multi-source benchmark ingestion (fails safely without corrupting news feed).
    try:
        from pipeline.benchmarks.runner import run_benchmark_pipeline
        run_benchmark_pipeline()
    except Exception as b_exc:
        logger.warning("Benchmark pipeline execution error (non-fatal): %s", b_exc)

    # 8. Generate pipeline status snapshot recording health across feeds and benchmarks.
    try:
        generate_pipeline_status(feed_records=merged)
    except Exception as s_exc:
        logger.warning("Pipeline status generation error (non-fatal): %s", s_exc)

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
