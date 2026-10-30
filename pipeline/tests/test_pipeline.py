"""
Comprehensive test suite for the AI Pulse ingestion pipeline.

Tests cover:
  - URL normalization
  - Stable ID generation
  - Duplicate detection
  - Date parsing
  - Category validation
  - Gemini response validation
  - Article preparation and schema
  - Merging new records with existing data
  - Atomic JSON output
  - Failure handling for unavailable RSS feeds

No real Gemini API key or network requests are made.
All external dependencies are mocked.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.run import (
    backfill_recent_tldrs,
    is_article_eligible_in_window,
    load_existing_feed,
    merge_records,
    prepare_article,
    reconcile_existing_feed,
    validate_record,
    write_feed_atomic,
)

# ---------------------------------------------------------------------------
# Unit under test imports
# ---------------------------------------------------------------------------
from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter
from pipeline.benchmarks.base import ALLOWED_BENCHMARK_CATEGORIES
from pipeline.relevance import is_article_ai_relevant, split_mixed_newsletter
from pipeline.scraper import (
    are_articles_closely_related,
    are_titles_duplicate,
    fetch_feed,
    normalize_title,
    normalize_url,
    parse_date,
    stable_id,
)
from pipeline.sources import ALLOWED_CATEGORIES, load_sources
from pipeline.summarizer import (
    FALLBACK_SUMMARY_PREFIX,
    CircuitBreaker,
    RateLimiter,
    _validate_gemini_response,
    classify_error,
    compute_content_fingerprint,
    is_summary_valid,
    make_fallback_summary,
    summarize_article,
    summarize_articles,
)

# ===========================================================================
# URL normalization tests
# ===========================================================================


class TestNormalizeUrl(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(
            normalize_url("  https://example.com/page  "),
            "https://example.com/page",
        )

    def test_lowercases_scheme_and_host(self):
        self.assertEqual(
            normalize_url("HTTPS://Example.COM/Page"),
            "https://example.com/Page",
        )

    def test_removes_default_https_port(self):
        self.assertEqual(
            normalize_url("https://example.com:443/path"),
            "https://example.com/path",
        )

    def test_removes_default_http_port(self):
        self.assertEqual(
            normalize_url("http://example.com:80/path"),
            "http://example.com/path",
        )

    def test_preserves_non_default_port(self):
        self.assertIn(":8080", normalize_url("https://example.com:8080/path"))

    def test_strips_fragment(self):
        result = normalize_url("https://example.com/page#section")
        self.assertNotIn("#section", result)

    def test_strips_utm_params(self):
        url = "https://example.com/article?utm_source=twitter&utm_medium=social"
        result = normalize_url(url)
        self.assertNotIn("utm_source", result)
        self.assertNotIn("utm_medium", result)

    def test_preserves_non_tracking_params(self):
        url = "https://example.com/search?q=AI&page=2"
        result = normalize_url(url)
        self.assertIn("q=AI", result)

    def test_removes_trailing_slash_from_path(self):
        result = normalize_url("https://example.com/blog/")
        self.assertEqual(result, "https://example.com/blog")

    def test_preserves_root_slash(self):
        result = normalize_url("https://example.com/")
        self.assertTrue(result.endswith("/") or result == "https://example.com")

    def test_collapses_double_slashes_in_path(self):
        result = normalize_url("https://example.com//path//to//page")
        self.assertNotIn("//path", result)

    def test_same_url_produces_same_result(self):
        url = "https://example.com/article?id=42"
        self.assertEqual(normalize_url(url), normalize_url(url))


# ===========================================================================
# Stable ID tests
# ===========================================================================


class TestStableId(unittest.TestCase):
    def test_returns_16_char_hex(self):
        result = stable_id("https://example.com/article")
        self.assertEqual(len(result), 16)
        # Should be hex characters only.
        int(result, 16)

    def test_same_url_same_id(self):
        url = "https://example.com/article"
        self.assertEqual(stable_id(url), stable_id(url))

    def test_different_urls_different_ids(self):
        self.assertNotEqual(
            stable_id("https://example.com/a"),
            stable_id("https://example.com/b"),
        )

    def test_id_stable_across_calls(self):
        url = "https://openai.com/blog/gpt-4"
        expected = stable_id(url)
        for _ in range(10):
            self.assertEqual(stable_id(url), expected)


# ===========================================================================
# Date parsing tests
# ===========================================================================


class TestParseDate(unittest.TestCase):
    def _make_entry(self, **attrs):
        entry = MagicMock()
        # Default: no date attrs.
        entry.published_parsed = None
        entry.updated_parsed = None
        entry.published = ""
        entry.updated = ""
        for k, v in attrs.items():
            setattr(entry, k, v)
        return entry

    def test_parses_published_parsed(self):
        # time.struct_time: 2024-01-15 12:00:00 UTC
        entry = self._make_entry(published_parsed=(2024, 1, 15, 12, 0, 0, 0, 0, 0))
        result = parse_date(entry)
        self.assertIsNotNone(result)
        self.assertIn("2024-01-15", result)

    def test_parses_updated_parsed_when_no_published(self):
        entry = self._make_entry(
            published_parsed=None,
            updated_parsed=(2024, 6, 20, 8, 30, 0, 0, 0, 0),
        )
        result = parse_date(entry)
        self.assertIsNotNone(result)
        self.assertIn("2024-06-20", result)

    def test_returns_none_when_no_date(self):
        entry = self._make_entry()
        result = parse_date(entry)
        self.assertIsNone(result)

    def test_returns_iso8601_string(self):
        entry = self._make_entry(published_parsed=(2024, 3, 10, 0, 0, 0, 0, 0, 0))
        result = parse_date(entry)
        # Should not raise.
        dt = datetime.fromisoformat(result)
        self.assertEqual(dt.year, 2024)

    def test_returns_none_on_malformed_raw_date(self):
        entry = self._make_entry(published="not-a-date")
        result = parse_date(entry)
        self.assertIsNone(result)


# ===========================================================================
# Category validation tests
# ===========================================================================


class TestAllowedCategories(unittest.TestCase):
    def test_all_categories_are_strings(self):
        for cat in ALLOWED_CATEGORIES:
            self.assertIsInstance(cat, str)

    def test_expected_categories_present(self):
        expected = {"News", "Research", "Benchmark", "Product Release", "Opinion"}
        self.assertEqual(set(ALLOWED_CATEGORIES), expected)


# ===========================================================================
# Gemini response validation tests
# ===========================================================================


class TestValidateGeminiResponse(unittest.TestCase):
    def test_valid_response_accepted(self):
        data = {"summary": "This is a valid summary sentence.", "category": "News"}
        result = _validate_gemini_response(data, "https://example.com")
        self.assertIsNotNone(result)
        self.assertEqual(result["summary"], data["summary"])
        self.assertEqual(result["category"], "News")

    def test_non_dict_rejected(self):
        result = _validate_gemini_response("a string", "https://example.com")
        self.assertIsNone(result)

    def test_empty_summary_rejected(self):
        data = {"summary": "", "category": "News"}
        result = _validate_gemini_response(data, "https://example.com")
        self.assertIsNone(result)

    def test_too_short_summary_rejected(self):
        data = {"summary": "Short.", "category": "Research"}
        result = _validate_gemini_response(data, "https://example.com")
        self.assertIsNone(result)

    def test_invalid_category_reset_to_news(self):
        data = {
            "summary": "This is a sufficiently long summary for the test.",
            "category": "InvalidCategory",
        }
        result = _validate_gemini_response(data, "https://example.com")
        self.assertIsNotNone(result)
        self.assertEqual(result["category"], "News")

    def test_all_valid_categories_accepted(self):
        for cat in ALLOWED_CATEGORIES:
            data = {
                "summary": "This is a sufficiently long summary for the test.",
                "category": cat,
            }
            result = _validate_gemini_response(data, "https://example.com")
            self.assertIsNotNone(result)
            self.assertEqual(result["category"], cat)


# ===========================================================================
# Summarize article (mocked Gemini) tests
# ===========================================================================


class TestSummarizeArticle(unittest.TestCase):
    def _make_article(self, url="https://example.com/article"):
        return {
            "id": stable_id(url),
            "title": "Test Article About AI",
            "company": "Test Corp",
            "source": "Test Feed",
            "url": url,
            "date": "2024-01-01T00:00:00+00:00",
            "category": "News",
            "text": "This is the article body text about AI developments.",
        }

    def _make_mock_client(
        self, summary_text="This is a valid summary.", category="Research"
    ):
        response_json = json.dumps({"summary": summary_text, "category": category})
        mock_response = MagicMock()
        mock_response.text = response_json
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        return mock_client

    def test_successful_summarization(self):
        article = self._make_article()
        mock_client = self._make_mock_client(
            "Gemini correctly summarized this article about AI.", "Research"
        )
        result = summarize_article(
            article, client=mock_client, model="gemini-3.5-flash-lite"
        )
        self.assertEqual(
            result["summary"], "Gemini correctly summarized this article about AI."
        )
        self.assertEqual(result["category"], "Research")
        self.assertEqual(result["url"], article["url"])

    def test_fallback_on_api_error(self):
        article = self._make_article()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError(
            "API unavailable"
        )
        result = summarize_article(
            article, client=mock_client, model="test-model", retries=0
        )
        self.assertTrue(result["summary"].startswith(FALLBACK_SUMMARY_PREFIX))
        self.assertEqual(result["url"], article["url"])

    def test_fallback_on_invalid_json_response(self):
        article = self._make_article()
        mock_response = MagicMock()
        mock_response.text = "not valid json {{{"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        result = summarize_article(
            article, client=mock_client, model="test-model", retries=0
        )
        self.assertTrue(result["summary"].startswith(FALLBACK_SUMMARY_PREFIX))

    def test_article_fields_preserved_on_success(self):
        article = self._make_article()
        mock_client = self._make_mock_client("Valid summary of sufficient length here.")
        result = summarize_article(article, client=mock_client, model="test-model")
        self.assertEqual(result["id"], article["id"])
        self.assertEqual(result["title"], article["title"])
        self.assertEqual(result["company"], article["company"])
        self.assertEqual(result["url"], article["url"])

    def test_article_fields_preserved_on_fallback(self):
        article = self._make_article()
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("error")
        result = summarize_article(
            article, client=mock_client, model="test-model", retries=0
        )
        self.assertEqual(result["id"], article["id"])
        self.assertEqual(result["url"], article["url"])


# ===========================================================================
# Record validation tests
# ===========================================================================


class TestValidateRecord(unittest.TestCase):
    def _valid_record(self):
        return {
            "id": "abc123def456789a",
            "title": "Test Article",
            "company": "Test Corp",
            "source": "Test Feed",
            "summary": "A test summary.",
            "url": "https://example.com/article",
            "date": "2024-01-01T00:00:00+00:00",
            "category": "News",
            "fetched_at": "2024-01-01T12:00:00+00:00",
        }

    def test_valid_record_passes(self):
        self.assertTrue(validate_record(self._valid_record()))

    def test_missing_field_fails(self):
        record = self._valid_record()
        del record["summary"]
        self.assertFalse(validate_record(record))

    def test_invalid_category_fails(self):
        record = self._valid_record()
        record["category"] = "MadeUpCategory"
        self.assertFalse(validate_record(record))

    def test_all_allowed_categories_pass(self):
        for cat in ALLOWED_CATEGORIES:
            record = self._valid_record()
            record["category"] = cat
            self.assertTrue(
                validate_record(record), f"Category '{cat}' should be valid"
            )

    def test_non_dict_fails(self):
        self.assertFalse(validate_record("not a dict"))

    def test_invalid_url_fails(self):
        record = self._valid_record()
        record["url"] = "not-a-url"
        self.assertFalse(validate_record(record))

    def test_null_date_is_valid(self):
        record = self._valid_record()
        record["date"] = None
        self.assertTrue(validate_record(record))


# ===========================================================================
# Feed I/O tests
# ===========================================================================


class TestLoadExistingFeed(unittest.TestCase):
    def _valid_record(self, idx=0):
        return {
            "id": f"abc{idx:013d}",
            "title": f"Test Article {idx}",
            "company": "Test Corp",
            "source": "Test Feed",
            "summary": "A test summary.",
            "url": f"https://example.com/article-{idx}",
            "date": "2024-01-01T00:00:00+00:00",
            "category": "News",
            "fetched_at": "2024-01-01T12:00:00+00:00",
        }

    def test_returns_empty_list_when_file_missing(self):
        result = load_existing_feed(Path("/nonexistent/path/feed.json"))
        self.assertEqual(result, [])

    def test_loads_valid_feed(self):
        records = [self._valid_record(i) for i in range(3)]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(records, f)
            tmp = Path(f.name)
        try:
            result = load_existing_feed(tmp)
            self.assertEqual(len(result), 3)
        finally:
            tmp.unlink(missing_ok=True)

    def test_raises_on_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("not valid json {{{")
            tmp = Path(f.name)
        try:
            with self.assertRaises(RuntimeError):
                load_existing_feed(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_raises_on_non_list_root(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"not": "a list"}, f)
            tmp = Path(f.name)
        try:
            with self.assertRaises(RuntimeError):
                load_existing_feed(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_skips_invalid_records(self):
        records = [self._valid_record(0), {"bad": "record"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(records, f)
            tmp = Path(f.name)
        try:
            result = load_existing_feed(tmp)
            self.assertEqual(len(result), 1)
        finally:
            tmp.unlink(missing_ok=True)


class TestWriteFeedAtomic(unittest.TestCase):
    def _valid_record(self, idx=0):
        return {
            "id": f"abc{idx:013d}",
            "title": f"Test Article {idx}",
            "company": "Test Corp",
            "source": "Test Feed",
            "summary": "A test summary.",
            "url": f"https://example.com/article-{idx}",
            "date": "2024-01-01T00:00:00+00:00",
            "category": "News",
            "fetched_at": "2024-01-01T12:00:00+00:00",
        }

    def test_writes_valid_json(self):
        records = [self._valid_record(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            feed_path = Path(tmp_dir) / "feed.json"
            write_feed_atomic(records, feed_path)
            self.assertTrue(feed_path.exists())
            with feed_path.open() as f:
                loaded = json.load(f)
            self.assertEqual(len(loaded), 5)

    def test_creates_parent_directories(self):
        records = [self._valid_record(0)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            feed_path = Path(tmp_dir) / "a" / "b" / "feed.json"
            write_feed_atomic(records, feed_path)
            self.assertTrue(feed_path.exists())

    def test_existing_file_not_corrupted_on_write_error(self):
        """If writing fails, the original file is preserved."""
        records = [self._valid_record(0)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            feed_path = Path(tmp_dir) / "feed.json"
            # Write initial content.
            write_feed_atomic(records, feed_path)
            original_content = feed_path.read_text()

            # Patch os.replace to simulate a failure.
            with patch("pipeline.run.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_feed_atomic([self._valid_record(99)], feed_path)

            # Original content must be intact.
            self.assertEqual(feed_path.read_text(), original_content)


# ===========================================================================
# Merge records tests
# ===========================================================================


class TestMergeRecords(unittest.TestCase):
    def _record(self, idx, date="2024-01-01T00:00:00+00:00", url=None):
        url = url or f"https://example.com/article-{idx}"
        return {
            "id": stable_id(url),
            "title": f"Article {idx}",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": url,
            "date": date,
            "category": "News",
            "fetched_at": "2024-01-01T00:00:00+00:00",
        }

    def test_new_articles_added(self):
        existing = [self._record(1)]
        new = [self._record(2), self._record(3)]
        merged = merge_records(existing, new, max_articles=100)
        self.assertEqual(len(merged), 3)

    def test_duplicates_by_id_excluded(self):
        existing = [self._record(1)]
        # Same URL → same ID.
        new = [self._record(1)]
        merged = merge_records(existing, new, max_articles=100)
        self.assertEqual(len(merged), 1)

    def test_duplicates_by_url_excluded(self):
        url = "https://example.com/article-same"
        existing = [self._record(1, url=url)]
        new = [self._record(2, url=url)]
        merged = merge_records(existing, new, max_articles=100)
        self.assertEqual(len(merged), 1)

    def test_sorted_by_date_descending(self):
        existing = [
            self._record(1, date="2024-01-01T00:00:00+00:00"),
            self._record(2, date="2024-03-01T00:00:00+00:00"),
        ]
        new = [self._record(3, date="2024-02-01T00:00:00+00:00")]
        merged = merge_records(existing, new, max_articles=100)
        dates = [r["date"] for r in merged]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_trimmed_to_max_articles(self):
        existing = [self._record(i) for i in range(10)]
        new = [self._record(i + 10) for i in range(10)]
        merged = merge_records(existing, new, max_articles=5)
        self.assertEqual(len(merged), 5)

    def test_untrimmed_when_max_articles_is_none(self):
        existing = [self._record(i) for i in range(10)]
        new = [self._record(i + 10) for i in range(10)]
        merged = merge_records(existing, new, max_articles=None)
        self.assertEqual(len(merged), 20)

    def test_untrimmed_when_max_articles_is_zero(self):
        existing = [self._record(i) for i in range(10)]
        new = [self._record(i + 10) for i in range(10)]
        merged = merge_records(existing, new, max_articles=0)
        self.assertEqual(len(merged), 20)

    def test_null_dates_sorted_last(self):
        existing = [
            self._record(1, date="2024-01-01T00:00:00+00:00"),
        ]
        new_record = self._record(2)
        new_record["date"] = None
        merged = merge_records(existing, [new_record], max_articles=100)
        # Dated article should be first.
        self.assertIsNotNone(merged[0]["date"])


# ===========================================================================
# RSS feed fetch failure handling tests
# ===========================================================================


class TestFetchFeedFailure(unittest.TestCase):
    def test_returns_empty_on_network_error(self):
        """A network error on a feed should return [] not raise."""
        source = {
            "name": "Broken Feed",
            "url": "https://nonexistent.invalid/feed.xml",
            "publisher": "Nobody",
            "category": "News",
        }
        with patch(
            "pipeline.scraper.feedparser.parse",
            side_effect=Exception("connection refused"),
        ):
            result = fetch_feed(source)
        self.assertEqual(result, [])

    def test_returns_empty_on_completely_empty_feed(self):
        source = {
            "name": "Empty Feed",
            "url": "https://example.com/feed.xml",
            "publisher": "Corp",
            "category": "News",
        }
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = []
        with patch("pipeline.scraper.feedparser.parse", return_value=mock_feed):
            result = fetch_feed(source)
        self.assertEqual(result, [])

    def test_bozo_feed_still_returns_entries(self):
        """A bozo (malformed) feed should still return any valid entries."""
        source = {
            "name": "Bozo Feed",
            "url": "https://example.com/feed.xml",
            "publisher": "Corp",
            "category": "News",
        }
        mock_entry = MagicMock()
        mock_entry.link = "https://example.com/article-1"
        mock_entry.title = "Test Article"
        mock_entry.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
        mock_entry.updated_parsed = None
        mock_entry.published = ""
        mock_entry.updated = ""
        mock_entry.summary = "Article summary text."
        mock_entry.content = None

        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = Exception("parse error")
        mock_feed.entries = [mock_entry]

        with patch("pipeline.scraper.feedparser.parse", return_value=mock_feed):
            result = fetch_feed(source)

        # Should have at least attempted to return entries.
        self.assertIsInstance(result, list)

    def test_deduplicates_within_feed(self):
        """Duplicate URLs within a single feed are deduplicated."""
        source = {
            "name": "Test Feed",
            "url": "https://example.com/feed.xml",
            "publisher": "Corp",
            "category": "News",
        }

        def make_entry(url):
            e = MagicMock()
            e.link = url
            e.title = "Article"
            e.published_parsed = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
            e.updated_parsed = None
            e.published = ""
            e.updated = ""
            e.summary = "Summary."
            e.content = None
            return e

        same_url = "https://example.com/article"
        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [make_entry(same_url), make_entry(same_url)]

        with patch("pipeline.scraper.feedparser.parse", return_value=mock_feed):
            result = fetch_feed(source)

        urls = [a["url"] for a in result]
        self.assertEqual(len(urls), len(set(urls)))


# ===========================================================================
# Prepare article tests
# ===========================================================================


class TestPrepareArticle(unittest.TestCase):
    def test_all_required_fields_present(self):
        raw = {
            "id": "abc123",
            "title": "Test",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": "https://example.com",
            "date": "2024-01-01T00:00:00+00:00",
            "category": "News",
            "text": "body text",
        }
        result = prepare_article(raw)
        required = {
            "id",
            "title",
            "company",
            "source",
            "summary",
            "url",
            "date",
            "category",
            "fetched_at",
        }
        self.assertTrue(required.issubset(set(result.keys())))

    def test_text_field_stripped(self):
        raw = {
            "id": "abc123",
            "title": "Test",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": "https://example.com",
            "date": None,
            "category": "Research",
            "text": "should be stripped",
        }
        result = prepare_article(raw)
        self.assertNotIn("text", result)

    def test_fetched_at_is_iso8601(self):
        raw = {
            "id": "abc123",
            "title": "Test",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": "https://example.com",
            "date": None,
            "category": "News",
            "text": "",
        }
        result = prepare_article(raw)
        # Should parse without error.
        dt = datetime.fromisoformat(result["fetched_at"])
        self.assertIsNotNone(dt)

    def test_preserves_valid_gemini_metadata(self):
        raw = {
            "id": "meta123",
            "title": "Frontier Model Launch",
            "company": "DeepMind",
            "source": "DeepMind Blog",
            "summary": "DeepMind released a new reasoning model achieving gold-level Olympiad scores.",
            "url": "https://deepmind.google/news/model-launch",
            "date": "2024-05-01T12:00:00+00:00",
            "category": "Product Release",
            "key_points": [
                "DeepMind introduced a new frontier reasoning model.",
                "The architecture achieves gold-level performance in competition math.",
                "Researchers and production teams evaluating reasoning workloads.",
            ],
            "tags": ["DeepMind", "Reasoning", "Frontier Models"],
            "impact_score": 9,
            "sources": [
                {
                    "title": "Frontier Model Launch",
                    "url": "https://deepmind.google/news/model-launch",
                    "publisher": "Google DeepMind",
                }
            ],
        }
        result = prepare_article(raw)
        self.assertEqual(len(result["key_points"]), 3)
        self.assertEqual(result["key_points"][0], "DeepMind introduced a new frontier reasoning model.")
        self.assertEqual(result["tags"], ["DeepMind", "Reasoning", "Frontier Models"])
        self.assertEqual(result["impact_score"], 9)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://deepmind.google/news/model-launch")

    def test_fallback_primary_source_when_sources_missing(self):
        raw = {
            "id": "nosrc123",
            "title": "Open Source Weights",
            "company": "Meta AI",
            "source": "Meta AI Blog",
            "summary": "Meta released new open weights under permissive license.",
            "url": "https://ai.meta.com/blog/open-weights",
            "date": "2024-05-02T12:00:00+00:00",
            "category": "Product Release",
        }
        result = prepare_article(raw)
        self.assertIn("sources", result)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["url"], "https://ai.meta.com/blog/open-weights")
        self.assertEqual(result["sources"][0]["title"], "Open Source Weights")
        self.assertEqual(result["sources"][0]["publisher"], "Meta AI")

    def test_rejects_generic_or_invalid_key_points(self):
        raw = {
            "id": "genkp123",
            "title": "Generic Story",
            "company": "Tech",
            "source": "News",
            "summary": "Some general news story about AI.",
            "url": "https://example.com/story",
            "date": "2024-05-02T12:00:00+00:00",
            "category": "News",
            "key_points": [
                "Generic story title.",
                "Provides verified empirical measurements of frontier reasoning performance.",
                "Engineers building autonomous agent workflows and developer tooling.",
            ],
        }
        result = prepare_article(raw)
        self.assertNotIn("key_points", result)

    def test_rejects_invalid_impact_score(self):
        raw = {
            "id": "badscore123",
            "title": "Score Test",
            "company": "Tech",
            "source": "News",
            "summary": "Some general news story.",
            "url": "https://example.com/score",
            "date": "2024-05-02T12:00:00+00:00",
            "category": "News",
            "impact_score": 99,
        }
        result = prepare_article(raw)
        self.assertNotIn("impact_score", result)

    def test_canned_why_it_matters_is_cleared(self):
        raw = {
            "id": "wim123",
            "title": "Test Title",
            "company": "Tech",
            "source": "News",
            "summary": "A test story.",
            "url": "https://example.com/wim",
            "date": "2024-05-02T12:00:00+00:00",
            "category": "News",
            "why_it_matters": "Provides verified empirical measurements of frontier reasoning performance.",
        }
        result = prepare_article(raw)
        self.assertEqual(result["why_it_matters"], "")


# ===========================================================================
# Demonstration records & statistics calculation tests
# ===========================================================================


class TestDemonstrationAndStatistics(unittest.TestCase):
    def _make_record(self, idx: int, is_demo: bool = False, company: str = "Test Co"):
        return {
            "id": f"rec_{idx}",
            "title": f"Article {idx}",
            "company": company,
            "source": f"Source {idx % 2}",
            "summary": "Concrete summary.",
            "url": f"https://example.com/{idx}",
            "date": "2024-05-01T00:00:00+00:00",
            "category": "Research",
            "fetched_at": "2024-05-01T12:00:00+00:00",
            **({"_demo": True} if is_demo else {}),
        }

    def test_merge_records_purges_demo_records(self):
        existing = [
            self._make_record(1, is_demo=True),
            self._make_record(2, is_demo=False),
        ]
        new_records = [
            self._make_record(3, is_demo=True),
            self._make_record(4, is_demo=False),
        ]
        merged = merge_records(existing, new_records, max_articles=10)
        self.assertEqual(len(merged), 2)
        for r in merged:
            self.assertFalse(r.get("_demo", False))
            self.assertIn(r["id"], ["rec_2", "rec_4"])

    def test_real_article_and_publisher_counts(self):
        dataset = [
            self._make_record(1, is_demo=True, company="Demo Corp"),
            self._make_record(2, is_demo=True, company="Demo Corp"),
            self._make_record(3, is_demo=False, company="DeepMind"),
            self._make_record(4, is_demo=False, company="DeepMind"),
            self._make_record(5, is_demo=False, company="Hugging Face"),
        ]
        real_articles = [r for r in dataset if not r.get("_demo")]
        distinct_publishers = {r["company"] for r in real_articles if r.get("company")}

        # 3 real articles from 2 distinct publishers
        self.assertEqual(len(real_articles), 3)
        self.assertEqual(len(distinct_publishers), 2)
        self.assertEqual(distinct_publishers, {"DeepMind", "Hugging Face"})

    def test_last_updated_excludes_demo_records(self):
        dataset = [
            {
                **self._make_record(1, is_demo=True),
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                **self._make_record(2, is_demo=False),
                "fetched_at": "2024-01-01T00:00:00+00:00",
            },
        ]
        real_articles = [r for r in dataset if not r.get("_demo")]
        dates = [r["fetched_at"] for r in real_articles]
        last_updated = sorted(dates, reverse=True)[0]
        self.assertEqual(last_updated, "2024-01-01T00:00:00+00:00")


# ===========================================================================
# Summary quality & filler removal tests
# ===========================================================================


class TestSummaryQuality(unittest.TestCase):
    def test_strips_filler_introductory_phrases(self):
        data = {
            "summary": (
                "This article discusses breakthrough results in quantum "
                "transformer attention."
            ),
            "category": "Research",
        }
        validated = _validate_gemini_response(data, "https://example.com")
        self.assertIsNotNone(validated)
        self.assertEqual(
            validated["summary"],
            "Breakthrough results in quantum transformer attention.",
        )

    def test_strips_this_post_explains_filler(self):
        data = {
            "summary": (
                "This post explains how new benchmarks evaluate "
                "reasoning models."
            ),
            "category": "Benchmark",
        }
        validated = _validate_gemini_response(data, "https://example.com")
        self.assertIsNotNone(validated)
        self.assertEqual(
            validated["summary"],
            "How new benchmarks evaluate reasoning models.",
        )

    def test_rejects_summary_that_is_only_filler(self):
        data = {
            "summary": "This article discusses.",
            "category": "News",
        }
        validated = _validate_gemini_response(data, "https://example.com")
        self.assertIsNone(validated)

    def test_fallback_summary_format(self):
        title = "OpenAI releases new reasoning model"
        text = "The new model achieves 92% on competition math."
        fallback = make_fallback_summary(title, text)
        self.assertTrue(fallback.startswith(FALLBACK_SUMMARY_PREFIX))
        self.assertIn("The new model achieves 92%", fallback)

    def test_fallback_summary_without_text(self):
        title = "DeepMind announces AlphaFold 3 update"
        fallback = make_fallback_summary(title, "")
        self.assertEqual(fallback, f"{FALLBACK_SUMMARY_PREFIX}{title}")


# ===========================================================================
# Retention policy & source configuration tests
# ===========================================================================


class TestRetentionAndSources(unittest.TestCase):
    def test_retention_days_pruning(self):
        recent_date = datetime.now().isoformat()
        old_record = {
            "id": "old_1",
            "title": "Old Article",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": "https://example.com/old",
            "date": "2020-01-01T00:00:00+00:00",
            "category": "News",
            "fetched_at": "2020-01-01T00:00:00+00:00",
        }
        new_record = {
            "id": "new_1",
            "title": "New Article",
            "company": "Corp",
            "source": "Feed",
            "summary": "Summary.",
            "url": "https://example.com/new",
            "date": recent_date,
            "category": "News",
            "fetched_at": recent_date,
        }
        merged = merge_records(
            [old_record], [new_record], max_articles=10, retention_days=30
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "new_1")

    def test_sources_configuration_loaded(self):
        sources = load_sources()
        self.assertIsInstance(sources, list)
        self.assertGreaterEqual(len(sources), 3)
        for src in sources:
            self.assertIn("name", src)
            self.assertIn("url", src)
            self.assertIn("publisher", src)
            self.assertIn("category", src)
            self.assertTrue(src["url"].startswith("http"))


# ===========================================================================
# Benchmark Intelligence & Adapter Tests
# ===========================================================================

import os
from datetime import UTC, datetime
from pipeline.benchmarks.base import (
    BenchmarkRecord,
    deduplicate_and_filter_conflicts,
    validate_benchmark_record,
)
from pipeline.benchmarks.adapters.openrouter import OpenRouterAdapter
from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter
from pipeline.benchmarks.adapters.benchlm import BenchLMAdapter
from pipeline.benchmarks.adapters.artificial_analysis import ArtificialAnalysisAdapter
from pipeline.benchmarks.adapters.lmcouncil import LMCouncilAdapter


class TestBenchmarkValidation(unittest.TestCase):
    def test_valid_benchmark_record(self):
        rec: BenchmarkRecord = {
            "model_name": "OpenAI o1",
            "canonical_id": "openai/o1",
            "provider": "OpenAI",
            "benchmark_name": "LiveBench (Reasoning)",
            "category": "Reasoning",
            "score": 89.0,
            "score_unit": "Percentage",
            "evaluation_date": "2026-01-08",
            "source": "LiveBench",
            "source_url": "https://livebench.ai/",
            "provenance": "Independent Evaluation",
            "retrieved_at": "2026-10-10T00:00:00Z",
            "notes": "Verified test notes.",
        }
        self.assertTrue(validate_benchmark_record(rec))

    def test_missing_field_rejected(self):
        rec = {
            "model_name": "OpenAI o1",
            "score": 89.0,
        }
        self.assertFalse(validate_benchmark_record(rec))

    def test_invalid_category_rejected(self):
        rec = {
            "model_name": "OpenAI o1",
            "canonical_id": "openai/o1",
            "provider": "OpenAI",
            "benchmark_name": "LiveBench (Reasoning)",
            "category": "InvalidCategory",
            "score": 89.0,
            "score_unit": "Percentage",
            "evaluation_date": "2026-01-08",
            "source": "LiveBench",
            "source_url": "https://livebench.ai/",
            "provenance": "Independent Evaluation",
            "retrieved_at": "2026-10-10T00:00:00Z",
            "notes": None,
        }
        self.assertFalse(validate_benchmark_record(rec))

    def test_invalid_provenance_rejected(self):
        rec = {
            "model_name": "OpenAI o1",
            "canonical_id": "openai/o1",
            "provider": "OpenAI",
            "benchmark_name": "LiveBench (Reasoning)",
            "category": "Reasoning",
            "score": 89.0,
            "score_unit": "Percentage",
            "evaluation_date": "2026-01-08",
            "source": "LiveBench",
            "source_url": "https://livebench.ai/",
            "provenance": "InventedProvenance",
            "retrieved_at": "2026-10-10T00:00:00Z",
            "notes": None,
        }
        self.assertFalse(validate_benchmark_record(rec))

    def test_conflict_detection_and_exclusion(self):
        rec1: BenchmarkRecord = {
            "model_name": "DeepSeek-V3",
            "canonical_id": "deepseek-v3",
            "provider": "DeepSeek",
            "benchmark_name": "LiveBench (Reasoning)",
            "category": "Reasoning",
            "score": 44.25,
            "score_unit": "Percentage",
            "evaluation_date": "2026-01-08",
            "source": "LiveBench",
            "source_url": "https://livebench.ai/",
            "provenance": "Independent Evaluation",
            "retrieved_at": "2026-10-10T00:00:00Z",
            "notes": "Run 1",
        }
        rec2: BenchmarkRecord = dict(rec1)
        rec2["score"] = 45.5 # Conflicting score for exact same measurement identity

        clean, conflicts = deduplicate_and_filter_conflicts([rec1, rec2])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(clean), 0) # Excluded ambiguous measurement

        # Identical duplicate should deduplicate safely without conflict
        rec3 = dict(rec1)
        clean2, conflicts2 = deduplicate_and_filter_conflicts([rec1, rec3])
        self.assertEqual(len(conflicts2), 0)
        self.assertEqual(len(clean2), 1)
        self.assertEqual(clean2[0]["score"], 44.25)


class TestBenchmarkAdapters(unittest.TestCase):
    @patch("requests.get")
    def test_openrouter_adapter_normalization(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "name": "OpenAI: GPT-4o",
                    "context_length": 128000,
                    "pricing": {
                        "prompt": "0.0000025",
                        "completion": "0.00001",
                    },
                }
            ]
        }
        mock_get.return_value = mock_resp

        adapter = OpenRouterAdapter()
        metrics = adapter.fetch_metrics()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["model_name"], "OpenAI: GPT-4o")
        self.assertEqual(metrics[0]["context_length"], 128000)
        self.assertEqual(metrics[0]["input_price_per_1m"], 2.5)
        self.assertEqual(metrics[0]["output_price_per_1m"], 10.0)

    @patch("requests.get")
    def test_livebench_adapter_parsing(self, mock_get):
        mock_table_resp = MagicMock()
        mock_table_resp.raise_for_status.return_value = None
        mock_table_resp.text = "model,theory_of_mind,zebra_puzzle\ndeepseek-r1,80.0,85.0\n"

        mock_cat_resp = MagicMock()
        mock_cat_resp.raise_for_status.return_value = None
        mock_cat_resp.json.return_value = {
            "Reasoning": ["theory_of_mind", "zebra_puzzle"]
        }

        mock_get.side_effect = [mock_table_resp, mock_cat_resp]

        adapter = LiveBenchAdapter(release_date="2026-06-25")
        records = adapter.fetch()
        self.assertGreater(len(records), 0)
        reasoning_rec = next((r for r in records if r["category"] == "Reasoning"), None)
        self.assertIsNotNone(reasoning_rec)
        self.assertEqual(reasoning_rec["score"], 82.5)
        self.assertEqual(reasoning_rec["provenance"], "Independent Evaluation")

    def test_benchlm_adapter_disabled_by_default(self):
        with patch.dict(os.environ, {"ENABLE_BENCHLM": "false"}):
            adapter = BenchLMAdapter()
            self.assertFalse(adapter.enabled)
            records = adapter.fetch()
            self.assertEqual(records, [])

    def test_artificial_analysis_disabled_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = ArtificialAnalysisAdapter()
            self.assertFalse(adapter.enabled)
            records = adapter.fetch()
            self.assertEqual(records, [])

    def test_lmcouncil_adapter_status(self):
        adapter = LMCouncilAdapter()
        self.assertFalse(adapter.enabled)
        records = adapter.fetch()
        self.assertEqual(records, [])
        notice = adapter.get_provenance_notice()
        self.assertIn("robots.txt", notice["attribution"])


class TestModelsCatalog(unittest.TestCase):
    def test_models_json_schema(self):
        models_path = Path(__file__).parent.parent.parent / "src" / "data" / "models.json"
        self.assertTrue(models_path.exists())
        with models_path.open("r", encoding="utf-8") as f:
            models = json.load(f)
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 10)
        for m in models:
            self.assertIn("id", m)
            self.assertIn("name", m)
            self.assertIn("provider", m)
            self.assertIn("context_window", m)
            self.assertGreater(m["context_window"], 0)
            self.assertIn("license_type", m)
            self.assertTrue(m["documentation_url"].startswith("http"))

    def test_sync_models_catalog_with_telemetry(self):
        from pipeline.models.updater import sync_models_catalog

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "models.json"
            seed_data = [
                {
                    "id": "openai/o1",
                    "name": "OpenAI o1",
                    "provider": "OpenAI",
                    "family": "o-series",
                    "release_date": "2024-12-17",
                    "lifecycle_status": "Released (Active)",
                    "context_window": 128000,
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "license_type": "Proprietary API",
                    "api_available": True,
                    "consumer_available": True,
                    "pricing": {
                        "input_per_1m": 15.00,
                        "output_per_1m": 60.00,
                        "unit": "USD",
                    },
                    "documentation_url": "https://platform.openai.com/docs/models#o1",
                    "source_url": "https://openai.com/index/introducing-openai-o1-preview/",
                    "last_verified": "2024-12-17",
                }
            ]
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(seed_data, f, indent=2)

            telemetry = [
                {
                    "canonical_id": "openai/o1",
                    "context_length": 200000,
                    "input_price_per_1m": 15.00,
                    "output_price_per_1m": 60.00,
                }
            ]

            updated = sync_models_catalog(models_path=tmp_path, openrouter_metrics=telemetry)
            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0]["context_window"], 200000)
            self.assertNotEqual(updated[0]["last_verified"], "2024-12-17")

    def test_sync_models_catalog_preserves_on_empty_metrics(self):
        from pipeline.models.updater import sync_models_catalog

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "models.json"
            seed_data = [
                {
                    "id": "deepseek/deepseek-r1",
                    "name": "DeepSeek-R1",
                    "provider": "DeepSeek",
                    "family": "DeepSeek-R1",
                    "release_date": "2025-01-20",
                    "lifecycle_status": "Released (Active)",
                    "context_window": 128000,
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "license_type": "Open Weights (MIT)",
                    "api_available": True,
                    "consumer_available": True,
                    "pricing": {
                        "input_per_1m": 0.55,
                        "output_per_1m": 2.19,
                        "unit": "USD",
                    },
                    "documentation_url": "https://api-docs.deepseek.com/",
                    "source_url": "https://github.com/deepseek-ai/DeepSeek-R1",
                    "last_verified": "2025-01-20",
                }
            ]
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(seed_data, f, indent=2)

            updated = sync_models_catalog(models_path=tmp_path, openrouter_metrics=[])
            self.assertEqual(len(updated), 1)
            self.assertEqual(updated[0]["context_window"], 128000)
            self.assertEqual(updated[0]["last_verified"], "2025-01-20")


class TestFutureDateHandling(unittest.TestCase):
    def test_rejects_future_dated_entries(self):
        from pipeline.scraper import parse_date
        entry = MagicMock()
        # Mock published_parsed 1 year in the future
        future_year = datetime.now().year + 2
        entry.published_parsed = (future_year, 1, 1, 0, 0, 0, 0, 0, 0)
        entry.updated_parsed = None
        entry.published = None
        entry.updated = None
        parsed = parse_date(entry)
        self.assertIsNone(parsed)


# ===========================================================================
# Quota-Aware Summarizer & Circuit Breaker Tests
# ===========================================================================


class TestQuotaAwareSummarizer(unittest.TestCase):
    def setUp(self):
        self.article = {
            "id": "art_123",
            "title": "Quantum Attention Mechanisms",
            "company": "DeepMind",
            "source": "DeepMind Blog",
            "url": "https://example.com/quantum",
            "text": "Full article body describing algorithmic advancements.",
            "category": "Research",
        }

    def test_content_fingerprint_deterministic(self):
        fp1 = compute_content_fingerprint("Title", "Body")
        fp2 = compute_content_fingerprint("Title", "Body")
        fp3 = compute_content_fingerprint("Different", "Body")
        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp3)

    def test_is_summary_valid(self):
        self.assertTrue(is_summary_valid({"summary": "A comprehensive 2-sentence summary of genuine content.", "summary_status": "generated"}))
        self.assertFalse(is_summary_valid({"summary": "[Automated summary unavailable] Something", "summary_status": "publisher_excerpt"}))
        self.assertFalse(is_summary_valid({"summary": "Short", "summary_status": "generated"}))
        self.assertFalse(is_summary_valid({}))

    def test_classify_error_quota_exhausted(self):
        err = Exception("429 RESOURCE_EXHAUSTED: quota exceeded. Please retry in 58.5s")
        err_type, delay = classify_error(err)
        self.assertEqual(err_type, "quota_exhausted")
        self.assertEqual(delay, 58.5)

    def test_classify_error_transient(self):
        err = Exception("503 Service Unavailable")
        err_type, _ = classify_error(err)
        self.assertEqual(err_type, "transient")

    def test_classify_error_permanent(self):
        err = Exception("400 Bad Request: Invalid model argument")
        err_type, _ = classify_error(err)
        self.assertEqual(err_type, "permanent")

    def test_circuit_breaker_trips_on_429(self):
        cb = CircuitBreaker()
        self.assertFalse(cb.is_open)
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: free tier limit 15 requests/min")

        res = summarize_article(self.article, client=mock_client, circuit_breaker=cb)
        self.assertTrue(cb.is_open)
        self.assertEqual(res["summary_status"], "pending_quota")
        # Should only have called once (no repeated calls on 429)
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_circuit_breaker_halts_subsequent_articles(self):
        cb = CircuitBreaker()
        cb.trip("Quota exhausted")
        mock_client = MagicMock()

        res = summarize_article(self.article, client=mock_client, circuit_breaker=cb)
        self.assertEqual(res["summary_status"], "pending_quota")
        # Client should not have been called at all while circuit is open
        self.assertEqual(mock_client.models.generate_content.call_count, 0)

    def test_summarize_articles_cache_reuse(self):
        existing = [{
            "id": "art_123",
            "title": "Quantum Attention Mechanisms",
            "summary": "Existing verified summary.",
            "summary_status": "generated",
            "content_fingerprint": compute_content_fingerprint("Quantum Attention Mechanisms", "Full article body describing algorithmic advancements."),
            "category": "Research",
        }]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            results, metrics = summarize_articles([self.article], existing_records=existing)

            self.assertEqual(metrics["skipped_cached"], 1)
            self.assertEqual(metrics["requests_sent"], 0)
            self.assertEqual(results[0]["summary"], "Existing verified summary.")
            self.assertEqual(results[0]["summary_status"], "generated")
            self.assertEqual(mock_client.models.generate_content.call_count, 0)

    def test_changed_content_triggers_regeneration(self):
        existing = [{
            "id": "art_123",
            "title": "Quantum Attention Mechanisms",
            "summary": "Old summary.",
            "summary_status": "generated",
            "content_fingerprint": "old_fingerprint",
            "category": "Research",
        }]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({"summary": "Brand new updated summary for edited article.", "category": "Research"})
            mock_client.models.generate_content.return_value = mock_resp
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles([self.article], existing_records=existing)
            self.assertEqual(metrics["requests_sent"], 1)
            self.assertEqual(results[0]["summary"], "Brand new updated summary for edited article.")

    def test_rate_limiter_interval(self):
        limiter = RateLimiter(requests_per_minute=60) # 1 per second
        self.assertEqual(limiter.interval, 1.0)
        limiter.last_call_time = time.time()
        # Sleep needed should be near 1.0
        sleep_time = limiter.wait()
        self.assertGreaterEqual(sleep_time, 0.5)


# ===========================================================================
# End-to-End Regression Tests for Model Catalog & Benchmark Data Flow
# ===========================================================================


class TestModelCatalogueRefreshRegression(unittest.TestCase):
    def test_catalogue_refresh_appends_new_frontier_models(self):
        from pipeline.models.updater import sync_models_catalog

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "models.json"
            seed_data = [
                {
                    "id": "openai/o1",
                    "name": "OpenAI o1",
                    "provider": "OpenAI",
                    "family": "o-series",
                    "release_date": "2024-12-17",
                    "lifecycle_status": "Released (Active)",
                    "context_window": 128000,
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "license_type": "Proprietary API",
                    "api_available": True,
                    "consumer_available": True,
                    "pricing": {"input_per_1m": 15.0, "output_per_1m": 60.0, "unit": "USD"},
                    "documentation_url": "https://platform.openai.com/docs/models#o1",
                    "source_url": "https://openai.com/index/introducing-openai-o1-preview/",
                    "last_verified": "2024-12-17",
                }
            ]
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(seed_data, f, indent=2)

            telemetry = [
                # Existing model update
                {
                    "canonical_id": "openai/o1",
                    "context_length": 200000,
                    "input_price_per_1m": 15.0,
                    "output_price_per_1m": 60.0,
                    "provider": "OpenAI",
                    "model_name": "OpenAI: o1",
                },
                # Batch variant of existing model (should be ignored)
                {
                    "canonical_id": "openai/o1:batch",
                    "context_length": 200000,
                    "input_price_per_1m": 7.5,
                    "output_price_per_1m": 30.0,
                    "provider": "OpenAI",
                    "model_name": "OpenAI: o1 (Batch)",
                },
                # New frontier model
                {
                    "canonical_id": "openai/gpt-6.1-sol",
                    "context_length": 1050000,
                    "input_price_per_1m": 2.0,
                    "output_price_per_1m": 10.0,
                    "provider": "OpenAI",
                    "model_name": "OpenAI: GPT-6.1 Sol",
                    "source_url": "https://openrouter.ai/openai/gpt-6.1-sol",
                },
                # Batch variant of new model (should NOT be added as separate model)
                {
                    "canonical_id": "openai/gpt-6.1-sol:batch",
                    "context_length": 1050000,
                    "input_price_per_1m": 1.0,
                    "output_price_per_1m": 5.0,
                    "provider": "OpenAI",
                    "model_name": "OpenAI: GPT-6.1 Sol (Batch)",
                },
                # New coding model
                {
                    "canonical_id": "mistralai/devstral-2512",
                    "context_length": 256000,
                    "input_price_per_1m": 0.4,
                    "output_price_per_1m": 1.2,
                    "provider": "Mistral AI",
                    "model_name": "Mistral: Devstral 2 2512",
                    "source_url": "https://openrouter.ai/mistralai/devstral-2512",
                },
                # Uncatalogued frontier candidate without a registered date
                {
                    "canonical_id": "openai/gpt-6.5-polaris",
                    "context_length": 1050000,
                    "input_price_per_1m": 3.0,
                    "output_price_per_1m": 12.0,
                    "provider": "OpenAI",
                    "model_name": "OpenAI: GPT-6.5 Polaris",
                    "source_url": "https://openrouter.ai/openai/gpt-6.5-polaris",
                },
            ]

            updated = sync_models_catalog(models_path=tmp_path, openrouter_metrics=telemetry)
            ids = [m["id"] for m in updated]

            # Verify existing model updated
            self.assertIn("openai/o1", ids)
            o1 = next(m for m in updated if m["id"] == "openai/o1")
            self.assertEqual(o1["context_window"], 200000)

            # Verify new models included
            self.assertIn("openai/gpt-6.1-sol", ids)
            self.assertIn("mistralai/devstral-2512", ids)
            self.assertIn("openai/gpt-6.5-polaris", ids)

            # Verify batch variants excluded
            self.assertNotIn("openai/o1:batch", ids)
            self.assertNotIn("openai/gpt-6.1-sol:batch", ids)

            # Verify fields populated: known release date applied
            sol = next(m for m in updated if m["id"] == "openai/gpt-6.1-sol")
            self.assertEqual(sol["release_date"], "2026-06-20")
            self.assertFalse(sol["consumer_available"])
            self.assertIn("telemetry", sol["verification_sources"])
            self.assertEqual(sol["category"], "Frontier Reasoning & Chat")

            # Verify uncatalogued model defaults safely to Unknown
            polaris = next(m for m in updated if m["id"] == "openai/gpt-6.5-polaris")
            self.assertEqual(polaris["release_date"], "Unknown")

            devstral = next(m for m in updated if m["id"] == "mistralai/devstral-2512")
            self.assertEqual(devstral["category"], "Coding")
            self.assertEqual(devstral["license_type"], "Open Weights")


class TestLiveBenchRegressionAndExactIdentity(unittest.TestCase):
    @patch("requests.get")
    def test_livebench_preserves_distinct_variants(self, mock_get):
        from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter
        from pipeline.benchmarks.base import deduplicate_and_filter_conflicts

        # CSV with two distinct effort configurations of the same model family
        csv_data = (
            "model,theory_of_mind,zebra_puzzle\n"
            "gpt-5.5-xhigh,85.0,90.0\n"
            "gpt-5.5-high,80.0,85.0\n"
            "claude-opus-4-7-xhigh-effort,88.0,92.0\n"
        )
        cat_data = {"Reasoning": ["theory_of_mind", "zebra_puzzle"]}

        mock_table = MagicMock()
        mock_table.raise_for_status.return_value = None
        mock_table.text = csv_data

        mock_cat = MagicMock()
        mock_cat.raise_for_status.return_value = None
        mock_cat.json.return_value = cat_data

        mock_get.side_effect = [mock_table, mock_cat]

        adapter = LiveBenchAdapter(release_date="2026-06-25")
        records = adapter.fetch()

        # Should parse Reasoning and Composite for each of the 3 models = 6 records
        self.assertEqual(len(records), 6)

        cids = {r["canonical_id"] for r in records}
        self.assertIn("gpt-5.5-xhigh", cids)
        self.assertIn("gpt-5.5-high", cids)
        self.assertIn("claude-opus-4-7-xhigh-effort", cids)

        # Distinct variants must not be collapsed
        self.assertNotEqual("gpt-5.5-xhigh", "gpt-5.5-high")

        # Distinct variants must survive deduplication without false conflicts
        clean, conflicts = deduplicate_and_filter_conflicts(records)
        self.assertEqual(len(conflicts), 0)
        self.assertEqual(len(clean), 6)

    def test_conflicting_measurement_rejected(self):
        from pipeline.benchmarks.base import BenchmarkRecord, deduplicate_and_filter_conflicts

        rec1: BenchmarkRecord = {
            "model_name": "GPT 5.5 (xhigh)",
            "canonical_id": "gpt-5.5-xhigh",
            "provider": "OpenAI",
            "benchmark_name": "LiveBench (Reasoning)",
            "category": "Reasoning",
            "score": 87.5,
            "score_unit": "Percentage",
            "evaluation_date": "2026-06-25",
            "source": "LiveBench",
            "source_url": "https://livebench.ai/",
            "provenance": "Independent Evaluation",
            "retrieved_at": "2026-10-11T00:00:00Z",
            "configuration": "xhigh",
        }
        rec2 = dict(rec1)
        rec2["score"] = 92.0  # Conflicting score for the exact same measurement identity

        clean, conflicts = deduplicate_and_filter_conflicts([rec1, rec2])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(len(clean), 0)


class TestStaleDataFallbackAndHealth(unittest.TestCase):
    @patch("pipeline.benchmarks.adapters.arena.ArenaAdapter.fetch")
    @patch("pipeline.benchmarks.adapters.livebench.LiveBenchAdapter.fetch")
    @patch("pipeline.benchmarks.adapters.openrouter.OpenRouterAdapter.fetch_metrics")
    def test_pipeline_preserves_snapshot_on_outage(self, mock_or, mock_lb, mock_arena):
        from pipeline.benchmarks.runner import run_benchmark_pipeline

        # Simulate remote failure
        mock_lb.return_value = []
        mock_arena.return_value = []
        mock_or.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bench = Path(tmpdir) / "benchmarks.json"
            seed_data = {
                "metadata": {
                    "last_updated": "2026-06-25T12:00:00Z",
                    "total_benchmark_records": 1,
                    "total_model_metrics": 1,
                },
                "benchmarks": [
                    {
                        "model_name": "Claude Fable 5",
                        "canonical_id": "claude-fable-5",
                        "provider": "Anthropic",
                        "benchmark_name": "LiveBench Overall Average",
                        "category": "Composite",
                        "score": 83.38,
                        "score_unit": "Percentage",
                        "evaluation_date": "2026-06-25",
                        "source": "LiveBench",
                        "source_url": "https://livebench.ai/",
                        "provenance": "Independent Evaluation",
                        "retrieved_at": "2026-06-25T12:00:00Z",
                    }
                ],
                "model_metrics": [],
                "benchmark_standards": [],
            }
            with tmp_bench.open("w", encoding="utf-8") as f:
                json.dump(seed_data, f, indent=2)

            result = run_benchmark_pipeline(benchmarks_path=tmp_bench)

            # Must preserve the snapshot, not overwrite with 0 records
            self.assertEqual(len(result["benchmarks"]), 1)
            self.assertEqual(result["metadata"]["total_benchmark_records"], 1)
            self.assertEqual(result["metadata"]["status"], "stale_fallback")
            self.assertTrue(any("Preserved" in w for w in result["metadata"]["health"]["warnings"]))


class TestDynamicBenchmarkStandards(unittest.TestCase):
    def test_dynamic_leaders_derived_from_actual_data(self):
        from pipeline.benchmarks.runner import derive_benchmark_standards

        records = [
            {
                "model_name": "Claude Fable 5.1 Max",
                "canonical_id": "claude-fable-5.1-max",
                "provider": "Anthropic",
                "benchmark_name": "LiveBench Overall Average",
                "category": "Composite",
                "score": 83.78,
                "score_unit": "Percentage",
                "evaluation_date": "2026-06-25",
                "source": "LiveBench",
                "source_url": "https://livebench.ai/",
                "provenance": "Independent Evaluation",
                "retrieved_at": "2026-10-11T00:00:00Z",
            },
            {
                "model_name": "GPT 6 Astra Max",
                "canonical_id": "gpt-6-astra-max",
                "provider": "OpenAI",
                "benchmark_name": "LiveBench Overall Average",
                "category": "Composite",
                "score": 82.50,
                "score_unit": "Percentage",
                "evaluation_date": "2026-06-25",
                "source": "LiveBench",
                "source_url": "https://livebench.ai/",
                "provenance": "Independent Evaluation",
                "retrieved_at": "2026-10-11T00:00:00Z",
            },
        ]

        standards = derive_benchmark_standards(records)
        lb_std = next(s for s in standards if s["name"] == "LiveBench")
        self.assertIn("Claude Fable 5.1 Max", lb_std["leader"])
        self.assertIn("83.78%", lb_std["leader"])
        # Static legacy leaders like o1 / Claude 3.5 Sonnet must NOT appear when data has newer leaders
        self.assertNotIn("OpenAI o1, DeepSeek-R1", lb_std["leader"])


class TestDeploymentBuildMetadata(unittest.TestCase):
    def test_build_metadata_schema_and_types(self):
        build_path = Path(__file__).parent.parent.parent / "src" / "data" / "build.json"
        self.assertTrue(build_path.exists())
        with build_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("commit_sha", data)
        self.assertIn("build_time", data)
        self.assertIsInstance(data["commit_sha"], str)
        self.assertIsInstance(data["build_time"], str)


class TestModelIdentityResolution(unittest.TestCase):
    def test_resolve_exact_internal_id(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        res = resolve_model_identity("openai/o1")
        self.assertEqual(res.match_type, MatchType.EXACT_INTERNAL_ID)
        self.assertEqual(res.matched_id, "openai/o1")
        self.assertEqual(res.confidence, 1.0)
        self.assertEqual(res.canonical_name, "OpenAI o1")
        self.assertEqual(res.provider, "OpenAI")

    def test_resolve_exact_openrouter_id(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        res = resolve_model_identity("meta-llama/llama-3.3-70b-instruct")
        self.assertEqual(res.match_type, MatchType.EXACT_OPENROUTER_ID)
        self.assertEqual(res.matched_id, "meta/llama-3-3-70b-instruct")
        self.assertEqual(res.confidence, 1.0)

    def test_resolve_exact_benchmark_name(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        res = resolve_model_identity("o3-mini-high")
        self.assertEqual(res.match_type, MatchType.EXACT_BENCHMARK_NAME)
        self.assertEqual(res.matched_id, "openai/o3-mini")
        self.assertEqual(res.configuration, "high")

    def test_resolve_verified_alias(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        res = resolve_model_identity("Claude 3.5 Sonnet")
        self.assertEqual(res.match_type, MatchType.VERIFIED_ALIAS)
        self.assertEqual(res.matched_id, "anthropic/claude-3-5-sonnet")

    def test_distinct_variant_not_collapsed(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        # Distinct version checkpoint not registered in verified aliases must NOT collapse
        res = resolve_model_identity("deepseek-v3.2")
        self.assertEqual(res.match_type, MatchType.UNRESOLVED)
        self.assertIsNone(res.matched_id)
        self.assertEqual(res.confidence, 0.0)

    def test_empty_identifier_unresolved(self):
        from pipeline.models.identity import MatchType, resolve_model_identity
        res = resolve_model_identity("   ")
        self.assertEqual(res.match_type, MatchType.UNRESOLVED)
        self.assertIsNone(res.matched_id)


class TestFeaturedFrontierAndCatalogue(unittest.TestCase):
    def test_featured_frontier_ids_presence(self):
        from pipeline.models.identity import get_featured_frontier_ids
        f_ids = get_featured_frontier_ids()
        self.assertIn("openai/o1", f_ids)
        self.assertIn("deepseek/deepseek-r1", f_ids)
        self.assertIn("anthropic/claude-3-7-sonnet", f_ids)
        self.assertIn("google/gemini-2-0-flash", f_ids)
        # Utility or low-tier models must not be featured frontier
        self.assertNotIn("openai/gpt-4o-mini", f_ids)

    def test_models_json_has_featured_frontier_boolean(self):
        models_path = Path(__file__).parent.parent.parent / "src" / "data" / "models.json"
        with models_path.open("r", encoding="utf-8") as f:
            models = json.load(f)
        self.assertGreater(len(models), 0)
        for m in models:
            self.assertIn("is_featured_frontier", m)
            self.assertIsInstance(m["is_featured_frontier"], bool)

    def test_model_identities_json_valid(self):
        ids_path = Path(__file__).parent.parent.parent / "src" / "data" / "model-identities.json"
        self.assertTrue(ids_path.exists())
        with ids_path.open("r", encoding="utf-8") as f:
            identities = json.load(f)
        self.assertIsInstance(identities, list)
        self.assertGreater(len(identities), 10)
        first = identities[0]
        self.assertIn("internal_id", first)
        self.assertIn("canonical_name", first)
        self.assertIn("provider", first)
        self.assertIn("provenance", first)


class TestBenchmarkValidationReportAndProvenance(unittest.TestCase):
    def test_validation_report_keys_and_values(self):
        bench_path = Path(__file__).parent.parent.parent / "src" / "data" / "benchmarks.json"
        with bench_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        report = data.get("metadata", {}).get("validation_report")
        self.assertIsNotNone(report)
        required_keys = {
            "retrieved_count",
            "parsed_count",
            "valid_count",
            "duplicate_count",
            "conflict_count",
            "quarantined_count",
            "retained_count",
        }
        self.assertTrue(required_keys.issubset(report.keys()))
        self.assertGreaterEqual(report["retained_count"], 0)

    def test_livebench_internal_model_id_populated(self):
        bench_path = Path(__file__).parent.parent.parent / "src" / "data" / "benchmarks.json"
        with bench_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        benchmarks = data.get("benchmarks", [])
        # Check that curated frontier model records have internal_model_id populated
        sol_records = [b for b in benchmarks if b.get("internal_model_id") == "openai/gpt-6.1-sol"]
        self.assertGreater(len(sol_records), 0)
        self.assertEqual(sol_records[0]["internal_model_id"], "openai/gpt-6.1-sol")


class TestArticleQualitySafeguards(unittest.TestCase):
    def test_clean_factual_summary_passes(self):
        from pipeline.summarizer import validate_article_summary_quality
        title = "Anthropic Releases Claude 3.7 Sonnet with Hybrid Reasoning"
        summary = "Anthropic has launched Claude 3.7 Sonnet, introducing hybrid reasoning architecture with selectable test-time compute."
        is_valid, reason = validate_article_summary_quality(title, summary)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Valid")

    def test_short_summary_rejected(self):
        from pipeline.summarizer import validate_article_summary_quality
        is_valid, reason = validate_article_summary_quality("Title", "Too short.")
        self.assertFalse(is_valid)
        self.assertIn("too short", reason.lower())

    def test_filler_placeholder_summary_rejected(self):
        from pipeline.summarizer import validate_article_summary_quality
        is_valid, reason = validate_article_summary_quality(
            "New LLM", "[Automated summary unavailable] Read more at site."
        )
        self.assertFalse(is_valid)
        self.assertIn("placeholder", reason.lower())

    def test_headline_model_mismatch_detected(self):
        from pipeline.summarizer import validate_article_summary_quality
        title = "Anthropic Announces Claude 3.7 Sonnet Frontier Model"
        summary = "Google DeepMind has introduced new Gemini models offering extended context window capabilities."
        is_valid, reason = validate_article_summary_quality(title, summary)
        self.assertFalse(is_valid)
        self.assertIn("mismatch", reason.lower())


class TestPipelineStatus(unittest.TestCase):
    def test_pipeline_status_snapshot_schema(self):
        status_path = Path(__file__).parent.parent.parent / "src" / "data" / "pipeline-status.json"
        self.assertTrue(status_path.exists())
        with status_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("generated_at", data)
        self.assertIn("overall_status", data)
        self.assertIn("feed_health", data)
        self.assertIn("rss_sources", data)
        self.assertIn("benchmark_sources", data)

    def test_pipeline_status_disabled_sources(self):
        status_path = Path(__file__).parent.parent.parent / "src" / "data" / "pipeline-status.json"
        with status_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        bench_sources = data.get("benchmark_sources", [])
        benchlm = next(s for s in bench_sources if s["name"] == "BenchLM")
        self.assertEqual(benchlm["status"], "disabled")
        self.assertIn("reason", benchlm)


class TestBenchmarkDateSemantics(unittest.TestCase):
    @patch("requests.get")
    def test_livebench_adapter_date_semantics_default_null_eval_date(self, mock_get):
        mock_table_resp = MagicMock()
        mock_table_resp.raise_for_status.return_value = None
        mock_table_resp.text = "model,theory_of_mind\no1,92.0\n"

        mock_cat_resp = MagicMock()
        mock_cat_resp.raise_for_status.return_value = None
        mock_cat_resp.json.return_value = {"Reasoning": ["theory_of_mind"]}

        mock_get.side_effect = [mock_table_resp, mock_cat_resp]

        adapter = LiveBenchAdapter(release_date="2026-06-25")
        records = adapter.fetch()
        self.assertGreater(len(records), 0)
        rec = records[0]

        # Benchmark release date and version must match release
        self.assertEqual(rec["benchmark_release_date"], "2026-06-25")
        self.assertEqual(rec["benchmark_version"], "2026-06-25")
        # evaluation_date must be None because LiveBench CSV does not provide per-model evaluation timestamps
        self.assertIsNone(rec["evaluation_date"])
        # retrieved_at must be populated with ISO datetime
        self.assertIsNotNone(rec.get("retrieved_at"))
        # notes must explain that evaluation date is not reported
        self.assertIn("Model evaluation date not reported by source", rec.get("notes", ""))

    @patch("requests.get")
    def test_livebench_adapter_preserves_explicit_row_eval_date(self, mock_get):
        mock_table_resp = MagicMock()
        mock_table_resp.raise_for_status.return_value = None
        mock_table_resp.text = "model,theory_of_mind,eval_date\no1,92.0,2026-07-01\n"

        mock_cat_resp = MagicMock()
        mock_cat_resp.raise_for_status.return_value = None
        mock_cat_resp.json.return_value = {"Reasoning": ["theory_of_mind"]}

        mock_get.side_effect = [mock_table_resp, mock_cat_resp]

        adapter = LiveBenchAdapter(release_date="2026-06-25")
        records = adapter.fetch()
        self.assertGreater(len(records), 0)
        rec = records[0]

        self.assertEqual(rec["benchmark_release_date"], "2026-06-25")
        self.assertEqual(rec["benchmark_version"], "2026-06-25")
        self.assertEqual(rec["evaluation_date"], "2026-07-01")

    def test_measurement_key_distinguishes_release_and_evaluation_dates(self):
        from pipeline.benchmarks.base import get_measurement_key

        rec1 = {
            "canonical_id": "openai/o1",
            "benchmark_name": "LiveBench (Reasoning)",
            "evaluation_date": None,
            "benchmark_release_date": "2026-06-25",
            "benchmark_version": "2026-06-25",
        }
        rec2 = {
            "canonical_id": "openai/o1",
            "benchmark_name": "LiveBench (Reasoning)",
            "evaluation_date": None,
            "benchmark_release_date": "2026-07-25",
            "benchmark_version": "2026-07-25",
        }
        rec3 = {
            "canonical_id": "openai/o1",
            "benchmark_name": "LiveBench (Reasoning)",
            "evaluation_date": "2026-06-30",
            "benchmark_release_date": "2026-06-25",
            "benchmark_version": "2026-06-25",
        }

        key1 = get_measurement_key(rec1)
        key2 = get_measurement_key(rec2)
        key3 = get_measurement_key(rec3)

        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertNotEqual(key2, key3)

    def test_benchmarks_json_date_semantics_integrity(self):
        """Validate date semantics integrity in production benchmarks.json against the adapter's supported releases."""
        from pipeline.benchmarks.adapters.livebench import VERIFIED_LIVEBENCH_RELEASES

        bench_path = Path(__file__).parent.parent.parent / "src" / "data" / "benchmarks.json"
        with bench_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        benchmarks = data.get("benchmarks", [])
        self.assertGreater(len(benchmarks), 0)
        for b in benchmarks:
            # All records must distinguish benchmark release/version from eval date
            self.assertIn("benchmark_release_date", b)
            self.assertIn("evaluation_date", b)
            self.assertIn("retrieved_at", b)
            # In LiveBench records where eval date is not reported, evaluation_date is None (no fabrication)
            if b.get("provenance") == "Independent Evaluation" and "not reported" in b.get("notes", ""):
                self.assertIsNone(b["evaluation_date"])
                self.assertIsNotNone(b["benchmark_release_date"])
                if b.get("source") == "LiveBench":
                    self.assertIn(b["benchmark_release_date"], VERIFIED_LIVEBENCH_RELEASES)

        # Check metadata health dates and distinct semantic timestamps
        metadata = data.get("metadata", {})
        health = metadata.get("health", {})
        self.assertIn("last_benchmark_release_date", health)
        self.assertIn(health["last_benchmark_release_date"], VERIFIED_LIVEBENCH_RELEASES)

        # 1. Benchmark release date: must be derived from supported release configuration (ISO date YYYY-MM-DD)
        self.assertRegex(health["last_benchmark_release_date"], r"^\d{4}-\d{2}-\d{2}$")

        # 2. Individual model evaluation date: not fabricated when upstream does not report it
        self.assertIsNone(health.get("last_successful_evaluation_date"))

        # 3. Last retrieval time: ISO datetime with timezone
        self.assertIn("last_retrieval_time", health)
        self.assertRegex(health["last_retrieval_time"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

        # 4. Last-known-good timestamp: ISO datetime with timezone
        self.assertIn("last_known_good_timestamp", health)
        self.assertRegex(health["last_known_good_timestamp"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

        # 5. Temporal distinction: release date is strictly separated from retrieval and snapshot timestamps
        self.assertNotEqual(health["last_benchmark_release_date"], health["last_retrieval_time"])
        self.assertNotEqual(health["last_benchmark_release_date"], health["last_known_good_timestamp"])

    def test_future_supported_release_date_compatibility(self):
        """Regression test ensuring a future supported release-date addition does not break date semantics validation."""
        from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter, VERIFIED_LIVEBENCH_RELEASES
        from pipeline.benchmarks.runner import run_benchmark_pipeline

        future_release = "2027-04-15"
        mock_supported = [future_release] + list(VERIFIED_LIVEBENCH_RELEASES)

        csv_content = (
            "model,design_layout,research,browser,coding,long_horizon,data_analysis,reasoning\n"
            "claude-opus-4,88.0,91.0,89.5,95.0,89.0,84.5,95.0\n"
        )

        with patch("pipeline.benchmarks.adapters.livebench.VERIFIED_LIVEBENCH_RELEASES", mock_supported), \
             patch("pipeline.benchmarks.adapters.livebench.DEFAULT_LIVEBENCH_RELEASE", future_release), \
             patch("requests.get") as mock_get, \
             patch("requests.head") as mock_head, \
             patch("pipeline.benchmarks.adapters.openrouter.OpenRouterAdapter.fetch_metrics", return_value=[]), \
             patch("pipeline.models.updater.sync_models_catalog"), \
             tempfile.TemporaryDirectory() as tmpdir:

            mock_head.return_value.ok = True

            def mock_get_handler(url, *args, **kwargs):
                resp = MagicMock()
                resp.ok = True
                resp.raise_for_status.return_value = None
                if str(url).endswith(".json"):
                    resp.json.return_value = {"Reasoning": ["reasoning"], "Agentic": ["long_horizon"]}
                else:
                    resp.text = csv_content
                return resp

            mock_get.side_effect = mock_get_handler

            adapter = LiveBenchAdapter(release_date=future_release)
            records = adapter.fetch()
            self.assertGreater(len(records), 0)
            self.assertEqual(records[0]["benchmark_release_date"], future_release)
            self.assertIsNone(records[0]["evaluation_date"])

            bench_path = Path(tmpdir) / "benchmarks.json"
            result = run_benchmark_pipeline(benchmarks_path=bench_path)

            meta_health = result["metadata"]["health"]
            # The pipeline health reflects the future supported release dynamically
            self.assertEqual(meta_health["last_benchmark_release_date"], future_release)
            self.assertIn(meta_health["last_benchmark_release_date"], mock_supported)
            self.assertIsNone(meta_health["last_successful_evaluation_date"])
            self.assertIsNotNone(meta_health["last_retrieval_time"])
            self.assertIsNotNone(meta_health["last_known_good_timestamp"])

    def test_date_semantics_temporal_dimension_distinction_and_no_fabrication(self):
        """Regression test verifying strict separation across all four temporal dimensions and no evaluation date fabrication."""
        from pipeline.benchmarks.adapters.livebench import VERIFIED_LIVEBENCH_RELEASES

        controlled_fixture = {
            "metadata": {
                "last_updated": "2026-10-30T14:00:00+00:00",
                "last_known_good_timestamp": "2026-10-25T07:51:53+00:00",
                "health": {
                    "status": "healthy",
                    "last_benchmark_release_date": "2026-10-14",
                    "last_successful_evaluation_date": None,
                    "last_retrieval_time": "2026-10-30T14:00:00+00:00",
                    "last_known_good_timestamp": "2026-10-25T07:51:53+00:00",
                    "warnings": [],
                },
            },
            "benchmarks": [
                {
                    "model_name": "Claude 3.7 Sonnet",
                    "canonical_id": "anthropic/claude-3-7-sonnet",
                    "provider": "Anthropic",
                    "benchmark_name": "LiveBench (Agentic)",
                    "category": "Agentic",
                    "score": 88.5,
                    "score_unit": "%",
                    "benchmark_release_date": "2026-10-14",
                    "evaluation_date": None,
                    "benchmark_version": "2.0",
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": "2026-10-30T14:00:00+00:00",
                    "notes": "Evaluation date not reported upstream by LiveBench table.",
                }
            ],
        }

        health = controlled_fixture["metadata"]["health"]
        bench_record = controlled_fixture["benchmarks"][0]

        # 1. Benchmark release date: identifies the benchmark release version, in verified releases
        self.assertIn(health["last_benchmark_release_date"], VERIFIED_LIVEBENCH_RELEASES)
        self.assertIn(bench_record["benchmark_release_date"], VERIFIED_LIVEBENCH_RELEASES)

        # 2. Evaluation date: None when not reported upstream, never fabricated
        self.assertIsNone(health["last_successful_evaluation_date"])
        self.assertIsNone(bench_record["evaluation_date"])

        # 3. Retrieval time: records when data was fetched
        self.assertEqual(health["last_retrieval_time"], "2026-10-30T14:00:00+00:00")
        self.assertEqual(bench_record["retrieved_at"], "2026-10-30T14:00:00+00:00")

        # 4. Last known good timestamp: tracks prior snapshot recovery point
        self.assertEqual(health["last_known_good_timestamp"], "2026-10-25T07:51:53+00:00")

        # Strict differentiation: all 4 dimensions are pairwise distinct
        release_date = health["last_benchmark_release_date"]
        eval_date = health["last_successful_evaluation_date"]
        retrieval_time = health["last_retrieval_time"]
        lkg_time = health["last_known_good_timestamp"]

        self.assertNotEqual(release_date, retrieval_time)
        self.assertNotEqual(release_date, lkg_time)
        self.assertNotEqual(retrieval_time, lkg_time)
        self.assertNotEqual(release_date, eval_date)



# ===========================================================================
# Article Enrichment, Deduplication, and Schema Tests
# ===========================================================================


class TestArticleEnrichmentAndDeduplication(unittest.TestCase):
    def test_normalize_title_strips_suffixes_and_punctuation(self):
        self.assertEqual(
            normalize_title("Anthropic releases Claude 3.5 Sonnet - The Verge"),
            "anthropic releases claude 3 5 sonnet",
        )
        self.assertEqual(
            normalize_title("OpenAI announces GPT-4o | TechCrunch"),
            "openai announces gpt 4o",
        )
        self.assertEqual(
            normalize_title("DeepMind Introduces Gemini 1.5 Pro — Google Blog"),
            "deepmind introduces gemini 1 5 pro",
        )

    def test_are_titles_duplicate_detects_syndicated_stories(self):
        t1 = "Anthropic announces Claude 3.5 Sonnet with enhanced coding - The Verge"
        t2 = "Anthropic announces Claude 3.5 Sonnet with enhanced coding | TechCrunch"
        self.assertTrue(are_titles_duplicate(t1, t2))

        t3 = "DeepMind publishes AlphaFold 3 molecular predictions technical paper"
        t4 = "DeepMind publishes AlphaFold 3 molecular predictions technical paper"
        self.assertTrue(are_titles_duplicate(t3, t4))

    def test_are_titles_duplicate_distinguishes_different_models(self):
        # Distinct model variants must not be falsely merged
        t1 = "Anthropic announces Claude 3.5 Haiku frontier model"
        t2 = "Anthropic announces Claude 3.5 Sonnet frontier model"
        self.assertFalse(are_titles_duplicate(t1, t2))

        t3 = "Google launches Gemini 1.5 Flash lightweight model"
        t4 = "Google launches Gemini 1.5 Pro multimodal architecture"
        self.assertFalse(are_titles_duplicate(t3, t4))

    def test_validate_record_handles_optional_fields_and_legacy(self):
        base_record = {
            "id": "abc1234567890def",
            "title": "OpenAI announces o3 reasoning model",
            "company": "OpenAI",
            "source": "OpenAI Blog",
            "summary": "OpenAI has officially unveiled o3, its next-generation reasoning model demonstrating state-of-the-art benchmark results.",
            "url": "https://openai.com/index/o3-announcement/",
            "date": "2026-10-10T12:00:00+00:00",
            "category": "Product Release",
            "fetched_at": "2026-10-10T12:05:00+00:00",
        }
        # Legacy record passes without optional fields
        self.assertTrue(validate_record(base_record))

        # Full enriched record passes
        enriched = dict(base_record)
        enriched["tags"] = ["OpenAI", "Reasoning", "o3"]
        enriched["key_points"] = [
            "OpenAI announced o3 reasoning model.",
            "Reaches state-of-the-art benchmark results on math and code.",
            "Developers and researchers evaluating frontier reasoning models.",
        ]
        enriched["impact_score"] = 9
        enriched["sources"] = [
            {"title": "OpenAI o3", "url": "https://openai.com/index/o3-announcement/", "publisher": "OpenAI"}
        ]
        self.assertTrue(validate_record(enriched))

        # Invalid impact_score is rejected
        invalid_score = dict(enriched)
        invalid_score["impact_score"] = 15
        self.assertFalse(validate_record(invalid_score))

        invalid_score_low = dict(enriched)
        invalid_score_low["impact_score"] = 0
        self.assertFalse(validate_record(invalid_score_low))

        # Invalid tags type is rejected
        invalid_tags = dict(enriched)
        invalid_tags["tags"] = "not a list"
        self.assertFalse(validate_record(invalid_tags))

    def test_validate_gemini_response_parses_enrichment(self):
        data_valid = {
            "summary": "DeepMind has announced AlphaFold 3 with biomolecular interactions spanning DNA and ligands.",
            "category": "Research",
            "topics": ["Research", "Healthcare & Biology"],
            "key_points": [
                "DeepMind unveils AlphaFold 3 for biomolecular prediction.",
                "Expands structural biology to DNA, RNA, and ligands.",
                "Structural biologists, pharmaceutical researchers, and chemists.",
            ],
            "tags": ["Google DeepMind", "AlphaFold", "Biology"],
            "impact_score": 9,
        }
        parsed = _validate_gemini_response(data_valid, "https://example.com/alphafold3")
        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed["key_points"]), 3)
        self.assertEqual(parsed["impact_score"], 9)
        self.assertEqual(parsed["tags"], ["Google DeepMind", "AlphaFold", "Biology"])

    def test_validate_gemini_response_handles_malformed_optional_fields_gracefully(self):
        # Even if key_points has wrong count or impact_score is string, valid summary is preserved
        data_partial = {
            "summary": "Anthropic released comprehensive system cards detailing Claude safety benchmarks.",
            "category": "News",
            "topics": ["Safety and Alignment"],
            "key_points": ["Only one point"],  # Should be omitted as it is not exactly 3
            "impact_score": "high",  # Non-integer should be omitted
        }
        parsed = _validate_gemini_response(data_partial, "https://example.com/claude-safety")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["summary"], "Anthropic released comprehensive system cards detailing Claude safety benchmarks.")
        self.assertNotIn("key_points", parsed)
        self.assertNotIn("impact_score", parsed)

    def test_merge_records_deduplicates_and_merges_sources(self):
        existing = [
            {
                "id": "article1",
                "title": "Anthropic releases Claude 3.5 Sonnet - VentureBeat",
                "company": "VentureBeat",
                "source": "VentureBeat AI",
                "summary": "Anthropic has released its updated Claude 3.5 Sonnet model with enhanced coding capabilities.",
                "url": "https://venturebeat.com/ai/2026/10/10/claude-3-5-sonnet",
                "date": "2026-10-10T14:00:00+00:00",
                "category": "News",
                "fetched_at": "2026-10-10T14:10:00+00:00",
                "sources": [
                    {
                        "title": "Anthropic releases Claude 3.5 Sonnet - VentureBeat",
                        "url": "https://venturebeat.com/ai/2026/10/10/claude-3-5-sonnet",
                        "publisher": "VentureBeat",
                    }
                ],
            }
        ]
        # Incoming secondary coverage of the exact same event
        new_incoming = [
            {
                "id": "article2",
                "title": "Anthropic releases Claude 3.5 Sonnet | TechCrunch",
                "company": "TechCrunch",
                "source": "TechCrunch AI",
                "summary": "Anthropic today announced an updated version of Claude 3.5 Sonnet.",
                "url": "https://techcrunch.com/2026/10/10/anthropic-claude-3-5-sonnet?ref=feed",
                "date": "2026-10-10T15:00:00+00:00",
                "category": "News",
                "fetched_at": "2026-10-10T15:05:00+00:00",
                "impact_score": 8,
                "tags": ["Anthropic", "Claude"],
            }
        ]

        merged = merge_records(existing, new_incoming, max_articles=10)
        # Should remain 1 primary card
        self.assertEqual(len(merged), 1)
        primary = merged[0]
        self.assertEqual(primary["id"], "article1")
        # Should have merged source from TechCrunch without duplicate URLs
        self.assertEqual(len(primary["sources"]), 2)
        urls = [s["url"] for s in primary["sources"]]
        self.assertIn("https://venturebeat.com/ai/2026/10/10/claude-3-5-sonnet", urls)
        self.assertIn("https://techcrunch.com/2026/10/10/anthropic-claude-3-5-sonnet?ref=feed", urls)
        # Preserved/upgraded impact_score and tags
        self.assertEqual(primary.get("impact_score"), 8)
        self.assertIn("Claude", primary.get("tags", []))

    def test_merge_records_preserves_distinct_stories(self):
        existing = [
            {
                "id": "story1",
                "title": "NVIDIA announces Blackwell Ultra architecture details",
                "company": "NVIDIA",
                "source": "NVIDIA News",
                "summary": "NVIDIA disclosed architectural specifications for the Blackwell Ultra server platform.",
                "url": "https://nvidianews.nvidia.com/blackwell-ultra",
                "date": "2026-10-10T10:00:00+00:00",
                "category": "Product Release",
                "fetched_at": "2026-10-10T10:05:00+00:00",
            }
        ]
        # Unrelated story
        new_incoming = [
            {
                "id": "story2",
                "title": "OpenAI expands ChatGPT Advanced Voice mode in Europe",
                "company": "Ars Technica",
                "source": "Ars Technica AI",
                "summary": "OpenAI is rolling out its Advanced Voice mode feature to users in the EU.",
                "url": "https://arstechnica.com/ai/openai-advanced-voice-europe",
                "date": "2026-10-10T11:00:00+00:00",
                "category": "News",
                "fetched_at": "2026-10-10T11:05:00+00:00",
            }
        ]
        merged = merge_records(existing, new_incoming, max_articles=10)
        self.assertEqual(len(merged), 2)


# ===========================================================================
# Recent News Date Window & Future Date Rejection Tests
# ===========================================================================


class TestRecentNewsDateFiltering(unittest.TestCase):
    def setUp(self):
        self.ref_now = datetime(2026, 10, 11, 14, 0, 0, tzinfo=UTC)

    def test_excludes_future_dated_articles(self):
        # 1 minute into the future
        future_dt = "2026-10-11T14:01:00+00:00"
        self.assertFalse(is_article_eligible_in_window(future_dt, 30.0, current_dt=self.ref_now))
        self.assertFalse(is_article_eligible_in_window(future_dt, 168.0, current_dt=self.ref_now))

        # Far future
        far_future = "2026-12-01T00:00:00+00:00"
        self.assertFalse(is_article_eligible_in_window(far_future, 30.0, current_dt=self.ref_now))
        self.assertFalse(is_article_eligible_in_window(far_future, 168.0, current_dt=self.ref_now))

    def test_thirty_hour_window_boundaries(self):
        # Exactly current time: eligible
        now_str = "2026-10-11T14:00:00+00:00"
        self.assertTrue(is_article_eligible_in_window(now_str, 30.0, current_dt=self.ref_now))

        # 29 hours ago: eligible
        within_30h = "2026-10-10T09:00:00+00:00"
        self.assertTrue(is_article_eligible_in_window(within_30h, 30.0, current_dt=self.ref_now))

        # 31 hours ago: excluded from 30h, but eligible for 7 days
        older_than_30h = "2026-10-10T07:00:00+00:00"
        self.assertFalse(is_article_eligible_in_window(older_than_30h, 30.0, current_dt=self.ref_now))
        self.assertTrue(is_article_eligible_in_window(older_than_30h, 168.0, current_dt=self.ref_now))

    def test_seven_day_window_boundaries(self):
        # 6 days ago: eligible
        six_days_ago = "2026-10-05T14:00:00+00:00"
        self.assertTrue(is_article_eligible_in_window(six_days_ago, 168.0, current_dt=self.ref_now))

        # 8 days ago: excluded
        eight_days_ago = "2026-10-03T14:00:00+00:00"
        self.assertFalse(is_article_eligible_in_window(eight_days_ago, 168.0, current_dt=self.ref_now))

    def test_invalid_or_missing_dates_excluded(self):
        self.assertFalse(is_article_eligible_in_window(None, 30.0, current_dt=self.ref_now))
        self.assertFalse(is_article_eligible_in_window("", 30.0, current_dt=self.ref_now))
        self.assertFalse(is_article_eligible_in_window("invalid-date-string", 30.0, current_dt=self.ref_now))


# ===========================================================================
# Safe Bounded TL;DR Backfill Tests
# ===========================================================================


class TestRecentTLDRBackfill(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 11, 14, 0, 0, tzinfo=UTC)

    def test_backfill_skips_articles_older_than_seven_days(self):
        old_art = {
            "id": "old1",
            "title": "Ancient Story",
            "summary": "Old summary text.",
            "url": "https://example.com/old",
            "date": (self.now - timedelta(days=30)).isoformat(),
            "category": "News",
        }
        mock_client = MagicMock()
        updated, skipped = backfill_recent_tldrs([old_art], client=mock_client, current_dt=self.now)
        self.assertEqual(updated, 0)
        self.assertEqual(mock_client.models.generate_content.call_count, 0)
        self.assertNotIn("key_points", old_art)

    def test_backfill_skips_future_dated_articles(self):
        future_art = {
            "id": "future1",
            "title": "Future Story",
            "summary": "Future summary text.",
            "url": "https://example.com/future",
            "date": (self.now + timedelta(days=2)).isoformat(),
            "category": "News",
        }
        mock_client = MagicMock()
        updated, skipped = backfill_recent_tldrs([future_art], client=mock_client, current_dt=self.now)
        self.assertEqual(updated, 0)
        self.assertEqual(mock_client.models.generate_content.call_count, 0)
        self.assertNotIn("key_points", future_art)

    def test_backfill_skips_articles_with_already_valid_tldr(self):
        valid_art = {
            "id": "valid1",
            "title": "Recent Story",
            "summary": "Recent summary.",
            "url": "https://example.com/recent",
            "date": (self.now - timedelta(days=1)).isoformat(),
            "category": "News",
            "key_points": [
                "Specific event occurred yesterday.",
                "Strategic implication for foundation models.",
                "Impacts software engineering teams globally.",
            ],
        }
        mock_client = MagicMock()
        updated, skipped = backfill_recent_tldrs([valid_art], client=mock_client, current_dt=self.now)
        self.assertEqual(updated, 0)
        self.assertEqual(mock_client.models.generate_content.call_count, 0)
        self.assertEqual(len(valid_art["key_points"]), 3)

    def test_backfill_updates_missing_tldr_and_preserves_all_metadata(self):
        recent_art = {
            "id": "rec1",
            "title": "New Reasoning Architecture Drops",
            "company": "DeepMind",
            "source": "DeepMind Blog",
            "summary": "DeepMind presented a novel reinforcement architecture.",
            "url": "https://example.com/arch",
            "date": (self.now - timedelta(days=1)).isoformat(),
            "category": "Research",
            "tags": ["DeepMind", "Reasoning"],
            "impact_score": 8,
            "sources": [{"title": "Source 1", "url": "https://example.com/arch", "publisher": "DeepMind"}],
        }

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "summary": "Ignored alternative summary.",
            "category": "Research",
            "key_points": [
                "DeepMind presented a novel reinforcement architecture.",
                "Improves benchmark accuracy by 15 percent on competitive reasoning.",
                "Engineers and researchers deploying frontier agent systems.",
            ],
            "tags": ["Extra Tag"],
            "impact_score": 5,
        })
        mock_client.models.generate_content.return_value = mock_resp

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([recent_art], f)
            tmp_path = Path(f.name)

        try:
            updated, remaining = backfill_recent_tldrs([recent_art], feed_path=tmp_path, client=mock_client, current_dt=self.now)
            self.assertEqual(updated, 1)
            # Only key_points was updated
            self.assertEqual(len(recent_art["key_points"]), 3)
            self.assertEqual(recent_art["key_points"][0], "DeepMind presented a novel reinforcement architecture.")
            # Existing metadata preserved strictly
            self.assertEqual(recent_art["summary"], "DeepMind presented a novel reinforcement architecture.")
            self.assertEqual(recent_art["tags"], ["DeepMind", "Reasoning"])
            self.assertEqual(recent_art["impact_score"], 8)
            self.assertEqual(recent_art["company"], "DeepMind")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_backfill_preserves_article_on_gemini_failure(self):
        recent_art = {
            "id": "fail1",
            "title": "Story with API Error",
            "summary": "Original summary.",
            "url": "https://example.com/fail",
            "date": (self.now - timedelta(days=1)).isoformat(),
            "category": "News",
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

        updated, remaining = backfill_recent_tldrs([recent_art], client=mock_client, current_dt=self.now)
        self.assertEqual(updated, 0)
        self.assertNotIn("key_points", recent_art)
        self.assertEqual(recent_art["summary"], "Original summary.")

    def test_backfill_handles_missing_api_key_gracefully(self):
        recent_art = {
            "id": "nokey1",
            "title": "Story Without Key",
            "summary": "Original summary.",
            "url": "https://example.com/nokey",
            "date": (self.now - timedelta(days=1)).isoformat(),
            "category": "News",
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
            updated, remaining = backfill_recent_tldrs([recent_art], current_dt=self.now)
            self.assertEqual(updated, 0)
            self.assertNotIn("key_points", recent_art)



# ===========================================================================
# Conservative Pre-Summarization Deduplication Tests
# ===========================================================================


class TestPreSummarizationDeduplication(unittest.TestCase):
    def test_presummarization_canonical_url_match_skips_gemini(self):
        existing = [
            {
                "id": "existing1",
                "title": "OpenAI announces GPT-5.5 Developer Preview",
                "company": "OpenAI",
                "source": "OpenAI Blog",
                "summary": "OpenAI has officially launched GPT-5.5 Developer Preview.",
                "summary_status": "generated",
                "url": "https://openai.com/index/gpt-5-5-preview?utm_source=rss",
                "date": "2026-10-10T14:00:00+00:00",
                "category": "Product Release",
                "sources": [
                    {
                        "title": "OpenAI announces GPT-5.5 Developer Preview",
                        "url": "https://openai.com/index/gpt-5-5-preview",
                        "publisher": "OpenAI",
                    }
                ],
            }
        ]
        # Incoming article with same canonical URL but different UTM tags / ID
        incoming = [
            {
                "id": "new_feed_art",
                "title": "OpenAI announces GPT-5.5 Developer Preview",
                "company": "OpenAI",
                "source": "OpenAI News",
                "url": "https://openai.com/index/gpt-5-5-preview?utm_campaign=newsletter&ref=feed",
                "text": "Full body text for GPT-5.5 Developer Preview.",
                "date": "2026-10-10T14:00:00+00:00",
                "category": "Product Release",
            }
        ]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles(incoming, existing_records=existing)

            # Verification: zero Gemini calls made
            self.assertEqual(metrics["skipped_cached"], 1)
            self.assertEqual(metrics["requests_sent"], 0)
            self.assertEqual(mock_client.models.generate_content.call_count, 0)
            # Reused summary and sources verified
            self.assertEqual(results[0]["summary"], "OpenAI has officially launched GPT-5.5 Developer Preview.")
            self.assertEqual(results[0]["summary_status"], "generated")

    def test_presummarization_cross_source_title_match_skips_gemini_and_merges_sources(self):
        existing = [
            {
                "id": "verge1",
                "title": "Anthropic releases Claude 3.7 Sonnet with hybrid reasoning - The Verge",
                "company": "The Verge",
                "source": "The Verge AI",
                "summary": "Anthropic introduced Claude 3.7 Sonnet offering hybrid reasoning capabilities.",
                "summary_status": "generated",
                "url": "https://www.theverge.com/2026/10/10/claude-3-7-sonnet",
                "date": "2026-10-10T15:00:00+00:00",
                "category": "News",
                "sources": [
                    {
                        "title": "Anthropic releases Claude 3.7 Sonnet with hybrid reasoning - The Verge",
                        "url": "https://www.theverge.com/2026/10/10/claude-3-7-sonnet",
                        "publisher": "The Verge",
                    }
                ],
            }
        ]
        # Incoming syndicated coverage from TechCrunch of the exact same event
        incoming = [
            {
                "id": "tc1",
                "title": "Anthropic releases Claude 3.7 Sonnet with hybrid reasoning | TechCrunch",
                "company": "TechCrunch",
                "source": "TechCrunch AI",
                "url": "https://techcrunch.com/2026/10/10/anthropic-claude-3-7-sonnet",
                "text": "Anthropic launched Claude 3.7 Sonnet today with hybrid reasoning.",
                "date": "2026-10-10T15:15:00+00:00",
                "category": "News",
            }
        ]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles(incoming, existing_records=existing)

            # Verification: zero Gemini calls made
            self.assertEqual(metrics["skipped_cached"], 1)
            self.assertEqual(metrics["requests_sent"], 0)
            self.assertEqual(mock_client.models.generate_content.call_count, 0)
            self.assertEqual(results[0]["summary"], "Anthropic introduced Claude 3.7 Sonnet offering hybrid reasoning capabilities.")
            # Verified source attribution merged
            sources_urls = [s["url"] for s in results[0]["sources"]]
            self.assertIn("https://www.theverge.com/2026/10/10/claude-3-7-sonnet", sources_urls)
            self.assertIn("https://techcrunch.com/2026/10/10/anthropic-claude-3-7-sonnet", sources_urls)

    def test_presummarization_intra_batch_duplicate_skips_redundant_call(self):
        # Two incoming articles in the SAME batch covering the same announcement
        batch = [
            {
                "id": "art_a",
                "title": "DeepSeek introduces DeepSeek-V3 Open Model - The Verge",
                "company": "The Verge",
                "source": "The Verge AI",
                "url": "https://www.theverge.com/deepseek-v3",
                "text": "DeepSeek announced DeepSeek-V3 with high performance.",
                "date": "2026-10-10T12:00:00+00:00",
                "category": "News",
            },
            {
                "id": "art_b",
                "title": "DeepSeek introduces DeepSeek-V3 Open Model | TechCrunch",
                "company": "TechCrunch",
                "source": "TechCrunch AI",
                "url": "https://techcrunch.com/deepseek-v3",
                "text": "DeepSeek released DeepSeek-V3 open model.",
                "date": "2026-10-10T12:10:00+00:00",
                "category": "News",
            },
        ]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "summary": "DeepSeek announced DeepSeek-V3 with high inference performance.",
                "category": "Product Release",
                "tags": ["DeepSeek", "DeepSeek-V3"],
            })
            mock_client.models.generate_content.return_value = mock_resp
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles(batch, existing_records=[])

            # Only 1 Gemini call for art_a; art_b reuses summary and merges attribution
            self.assertEqual(metrics["requests_sent"], 1)
            self.assertEqual(metrics["skipped_cached"], 1)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["summary"], results[1]["summary"])

    def test_presummarization_preserves_distinct_stories_with_separate_summaries(self):
        # Two distinct stories from the same company
        batch = [
            {
                "id": "story_spec",
                "title": "Google announces Gemini 2.5 Flash architecture",
                "company": "Google DeepMind",
                "source": "Google Blog",
                "url": "https://blog.google/gemini-2-5-flash",
                "text": "Architecture details for Gemini 2.5 Flash.",
                "date": "2026-10-10T10:00:00+00:00",
                "category": "Research",
            },
            {
                "id": "story_price",
                "title": "Google updates Cloud API pricing for Gemini models",
                "company": "Google DeepMind",
                "source": "Google Cloud",
                "url": "https://cloud.google.com/gemini-pricing",
                "text": "Cloud API pricing reductions for Gemini models.",
                "date": "2026-10-10T11:00:00+00:00",
                "category": "News",
            },
        ]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "summary": "Valid distinct summary for each announcement.",
                "category": "News",
            })
            mock_client.models.generate_content.return_value = mock_resp
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles(batch, existing_records=[])

            # Both stories summarized independently (no false merge)
            self.assertEqual(metrics["requests_sent"], 2)
            self.assertEqual(metrics["skipped_cached"], 0)
            self.assertEqual(len(results), 2)

    def test_presummarization_incompatible_dates_not_merged(self):
        existing = [
            {
                "id": "old_release",
                "title": "OpenAI releases o1 reasoning model",
                "company": "OpenAI",
                "source": "OpenAI Blog",
                "summary": "OpenAI released the o1 model series.",
                "summary_status": "generated",
                "url": "https://openai.com/o1",
                "date": "2024-09-12T14:00:00+00:00",
                "category": "Product Release",
            }
        ]
        # Same headline but months later (e.g. later coverage or separate milestone)
        incoming = [
            {
                "id": "new_coverage",
                "title": "OpenAI releases o1 reasoning model",
                "company": "The Verge",
                "source": "The Verge AI",
                "url": "https://theverge.com/o1-general-availability",
                "text": "OpenAI has now made o1 generally available.",
                "date": "2025-01-15T14:00:00+00:00",
                "category": "News",
            }
        ]

        with patch("pipeline.summarizer._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = json.dumps({
                "summary": "OpenAI announced general availability of the o1 reasoning model.",
                "category": "Product Release",
            })
            mock_client.models.generate_content.return_value = mock_resp
            mock_get_client.return_value = mock_client

            results, metrics = summarize_articles(incoming, existing_records=existing)

            # Incompatible date (>48h) does not falsely merge
            self.assertEqual(metrics["requests_sent"], 1)
            self.assertEqual(metrics["skipped_cached"], 0)


# ===========================================================================
# AI Relevance Gate Tests
# ===========================================================================


class TestAiRelevanceGate(unittest.TestCase):
    def test_accepts_core_ai_architectures_and_reasoning(self):
        # Direct architectural and algorithmic AI papers
        self.assertTrue(
            is_article_ai_relevant(
                title="DeepSeek announces DeepSeek-V3 architecture with multi-head latent attention",
                text="Technical details of DeepSeek-V3 FP8 training and auxiliary-loss-free balancing.",
                tags=["DeepSeek", "LLM", "Transformers"],
            )[0]
        )
        self.assertTrue(
            is_article_ai_relevant(
                title="Anthropic demonstrates agentic reasoning in Claude 3.5 Sonnet",
                text="Evaluations across SWE-bench and autonomous browser tool use.",
                tags=["Anthropic", "Claude", "Agents"],
            )[0]
        )
        self.assertTrue(
            is_article_ai_relevant(
                title="OpenAI explains flow matching and diffusion transformers in Sora",
                text="Video generation dynamics with spacetime patch tokens.",
                tags=["OpenAI", "Diffusion", "Transformers"],
            )[0]
        )

    def test_accepts_genomic_and_scientific_ai_models(self):
        self.assertTrue(
            is_article_ai_relevant(
                title="Google DeepMind unveils AlphaGenome predicting variant impact on gene expression",
                text="AlphaGenome models multimodal genomics with transformer sequence analysis.",
                tags=["DeepMind", "AlphaGenome", "Genomics"],
            )[0]
        )

    def test_rejects_unrelated_health_and_pharma(self):
        # Fake blood / stem cell synthesis
        self.assertFalse(
            is_article_ai_relevant(
                title="Scientists create synthetic human blood from stem cells to tackle donor shortages",
                text="Biomedical engineering researchers develop cellular blood substitute for trauma care.",
                tags=["Medicine", "Biology", "Health"],
            )[0]
        )
        # Weight loss GLP-1 drugs
        self.assertFalse(
            is_article_ai_relevant(
                title="New GLP-1 weight-loss injections show promising results in Phase III cardiovascular trials",
                text="Clinical trial results reveal metabolic improvements in diabetic patient cohort.",
                tags=["Pharma", "Health", "Clinical Trials"],
            )[0]
        )

    def test_rejects_unrelated_batteries_and_climate_tech(self):
        # Solid state batteries
        self.assertFalse(
            is_article_ai_relevant(
                title="Next-generation solid-state lithium-sulfur batteries promise 1,000-mile EV range",
                text="Electrochemical breakthrough stabilizes solid electrolyte interface for automotive battery packs.",
                tags=["Energy", "Batteries", "CleanTech"],
            )[0]
        )
        # Climate startup listicle
        self.assertFalse(
            is_article_ai_relevant(
                title="10 climate tech startups transforming geothermal energy and clean hydrogen",
                text="Venture capital investments in geothermal drilling and renewable hydrogen electrolysis.",
                tags=["Climate", "Startups", "CleanTech"],
            )[0]
        )

    def test_rejects_unrelated_civil_and_aerospace_infrastructure(self):
        # Underground moon city
        self.assertFalse(
            is_article_ai_relevant(
                title="Architects unveil blueprints for underground moon city inside lunar lava tubes",
                text="Structural designs for pressurised habitats protected from cosmic radiation on the Moon.",
                tags=["Space", "Architecture", "Moon"],
            )[0]
        )
        # Low-altitude drone traffic corridors
        self.assertFalse(
            is_article_ai_relevant(
                title="Automated radar corridors track low-altitude drone air traffic over Tokyo",
                text="Airspace management regulations test sensor towers for commercial package delivery.",
                tags=["Aviation", "Drones", "Logistics"],
            )[0]
        )

    def test_allows_trusted_ai_lab_research(self):
        self.assertTrue(
            is_article_ai_relevant(
                title="Scaling law empirical verification across long-context token windows",
                text="Empirical results from frontier foundation model training runs.",
                source_name="OpenAI",
            )[0]
        )

    def test_rejects_mixed_newsletter_false_positive_with_incidental_mention(self):
        # Known false positive record: "The Download: drone traffic controllers" with peripheral Anthropic footnote
        is_rel, reason = is_article_ai_relevant(
            title="The Download: drone traffic controllers",
            text="Testing radar drone traffic controller systems across commercial air corridors. In brief: Anthropic tests model.",
            source_name="MIT Technology Review",
        )
        self.assertFalse(is_rel)
        self.assertTrue("drone traffic controllers" in reason.lower() or "unrelated topic" in reason.lower())

    def test_rejects_non_ai_entertainment_and_industrial_distractors(self):
        # DistroKid music royalties
        self.assertFalse(
            is_article_ai_relevant(
                title="DistroKid announces direct digital distribution for indie musicians",
                text="Music distribution platform simplifies royalty payouts and track uploads.",
            )[0]
        )
        # Cage matches
        self.assertFalse(
            is_article_ai_relevant(
                title="Human cage matches and underground martial arts fight clubs gain popularity",
                text="Combat sports promoters organize unlicensed martial arts tournaments.",
            )[0]
        )
        # Brimstone kiln
        self.assertFalse(
            is_article_ai_relevant(
                title="Brimstone raises funding for zero-carbon cement and kiln technologies",
                text="Decarbonizing industrial concrete manufacturing with calcium silicate rocks.",
            )[0]
        )

    def test_accepts_cross_domain_ml_breakthroughs(self):
        # Space weather risk prediction
        self.assertTrue(
            is_article_ai_relevant(
                title="Machine learning model predicts geomagnetic storms and space weather risks",
                text="Researchers train deep neural networks on solar wind telemetry to forecast solar coronal mass ejections.",
            )[0]
        )
        # Skala genomic interpretation
        self.assertTrue(
            is_article_ai_relevant(
                title="Skala: deep learning for pathogenic variant effect prediction across non-coding human genomes",
                text="Transformer-based genome sequence model predicts regulatory mutation effects.",
            )[0]
        )
        # RetroChimera retrosynthesis
        self.assertTrue(
            is_article_ai_relevant(
                title="RetroChimera: AI retrosynthesis models accelerate organic molecule drug discovery",
                text="Deep neural generative models map reaction pathways for chemical synthesis.",
            )[0]
        )
        # Surrogate plasma physics
        self.assertTrue(
            is_article_ai_relevant(
                title="Surrogate models and gradient-boosting predict plasma turbulence in fusion reactors",
                text="Machine learning surrogates accelerate magnetohydrodynamic simulation runs.",
            )[0]
        )

    def test_dedicated_source_audit(self):
        # Verified feed cs.AI accepted
        self.assertTrue(
            is_article_ai_relevant(
                title="Empirical scaling laws for multi-modal reasoning architectures",
                text="Analysis of cross-entropy loss across parameter scales.",
                source_name="arXiv cs.AI",
            )[0]
        )
        # Non-AI category under broad publisher without AI terms does not bypass
        self.assertFalse(
            is_article_ai_relevant(
                title="Ergodic properties of non-Markovian Brownian motion in random media",
                text="Stochastic calculus proofs for anomalous diffusion in porous rocks.",
                source_name="arXiv math.PR",
            )[0]
        )


# ===========================================================================
# Split Mixed Newsletter Tests
# ===========================================================================


class TestSplitMixedNewsletter(unittest.TestCase):
    def test_splits_mixed_newsletter_and_extracts_ai_story(self):
        newsletter_article = {
            "title": "The Download: inside the race for AI models, plus the history of rubber dams",
            "text": "Today in The Download: We explore how tech giants are racing to build open frontier AI models with improved reasoning. In other news, civil engineers reflect on the forgotten history of inflatable rubber dams used for river irrigation.",
            "url": "https://technologyreview.com/download-ai-race-and-rubber-dams",
            "source": "MIT Technology Review",
            "company": "MIT Tech Review",
        }
        extracted = split_mixed_newsletter(newsletter_article)
        self.assertGreaterEqual(len(extracted), 1)
        first = extracted[0]
        # Extracted article should feature the AI topic
        self.assertIn("ai", first["title"].lower())
        self.assertNotIn("rubber dams", first["title"].lower())
        self.assertTrue(is_article_ai_relevant(first["title"], first["text"])[0])

    def test_leaves_single_focused_article_intact(self):
        single_article = {
            "title": "Meta releases Llama 3.3 70B Instruct open weights",
            "text": "Meta has released Llama 3.3 70B with competitive performance against industry flagship models.",
            "url": "https://ai.meta.com/blog/llama-3-3",
            "source": "Meta AI",
            "company": "Meta",
        }
        res = split_mixed_newsletter(single_article)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["title"], single_article["title"])

    def test_split_mixed_newsletter_does_not_create_fragment_urls(self):
        newsletter = {
            "title": "The Download: AI reasoning advancements, plus electric cargo ships",
            "text": "First story: Frontier models achieve new milestones in multi-step planning and tool use. Second story: Marine cargo ships transition to hydrogen fuel cells.",
            "url": "https://technologyreview.com/download-ai-and-cargo-ships",
            "source": "MIT Technology Review",
            "company": "MIT Tech Review",
        }
        segments = split_mixed_newsletter(newsletter)
        self.assertGreaterEqual(len(segments), 1)
        for seg in segments:
            # Must NOT invent #section or #fragment URLs
            self.assertNotIn("#", seg["url"])
            self.assertEqual(seg["url"], "https://technologyreview.com/download-ai-and-cargo-ships")
            # Must assign unique deterministic 16-char ID
            self.assertIn("id", seg)
            self.assertEqual(len(seg["id"]), 16)


# ===========================================================================
# Closely Related Article Deduplication Tests
# ===========================================================================


class TestCloselyRelatedArticles(unittest.TestCase):
    def test_detects_closely_related_coverage_of_same_event(self):
        art1 = {
            "title": "Anthropic releases Claude 3.7 Sonnet hybrid reasoning model",
            "company": "Anthropic",
            "date": "2026-10-10T12:00:00+00:00",
            "url": "https://techcrunch.com/claude-3-7-sonnet",
        }
        art2 = {
            "title": "Anthropic announces Claude 3.7 Sonnet with hybrid reasoning capabilities",
            "company": "Anthropic",
            "date": "2026-10-10T14:30:00+00:00",
            "url": "https://arstechnica.com/anthropic-claude-3-7-sonnet",
        }
        self.assertTrue(are_articles_closely_related(art1, art2))

    def test_distinguishes_different_model_variants_same_day(self):
        art_sonnet = {
            "title": "Anthropic launches Claude 3.5 Sonnet for enterprise developers",
            "company": "Anthropic",
            "date": "2026-10-10T12:00:00+00:00",
            "url": "https://techcrunch.com/claude-3-5-sonnet",
        }
        art_haiku = {
            "title": "Anthropic launches Claude 3.5 Haiku lightweight model",
            "company": "Anthropic",
            "date": "2026-10-10T12:00:00+00:00",
            "url": "https://techcrunch.com/claude-3-5-haiku",
        }
        # Distinct model tiers must NOT be merged
        self.assertFalse(are_articles_closely_related(art_sonnet, art_haiku))

    def test_rejects_matches_across_distant_dates(self):
        art_old = {
            "title": "DeepSeek unveils DeepSeek-R1 reasoning model",
            "company": "DeepSeek",
            "date": "2025-01-20T10:00:00+00:00",
            "url": "https://deepseek.com/r1",
        }
        art_new = {
            "title": "DeepSeek unveils DeepSeek-R1 reasoning model",
            "company": "DeepSeek",
            "date": "2025-06-15T10:00:00+00:00",
            "url": "https://theverge.com/deepseek-r1-analysis",
        }
        self.assertFalse(are_articles_closely_related(art_old, art_new))

    def test_groups_gemini_4_argon_announcement_and_media_coverage(self):
        deepmind_art = {
            "title": "Gemini 4 Argon: Built to solve your hardest problems",
            "company": "Google DeepMind",
            "date": "2026-10-14T09:00:00+00:00",
            "url": "https://deepmind.google/discover/blog/gemini-4-argon/",
        }
        verge_art = {
            "title": "Google finally releases Gemini 4 Argon",
            "company": "The Verge",
            "date": "2026-10-14T11:15:00+00:00",
            "url": "https://theverge.com/ai/gemini-4-argon-launch",
        }
        # Shared specific model compound ("Gemini 4 Argon") with release framing on same day should be merged
        self.assertTrue(are_articles_closely_related(deepmind_art, verge_art))

    def test_keeps_subdomain_robotics_and_safety_distinct(self):
        core_art = {
            "title": "Gemini 4 Argon: Built to solve your hardest problems",
            "company": "Google DeepMind",
            "date": "2026-10-14T09:00:00+00:00",
            "url": "https://deepmind.google/discover/blog/gemini-4-argon/",
        }
        robotics_art = {
            "title": "Gemini 4 Argon in Robotics: Spatial intelligence for physical manipulation",
            "company": "Google DeepMind",
            "date": "2026-10-14T09:30:00+00:00",
            "url": "https://deepmind.google/discover/blog/gemini-4-argon-robotics/",
        }
        safety_art = {
            "title": "Evaluating Gemini 4 Argon responsibly and safely",
            "company": "Google DeepMind",
            "date": "2026-10-14T09:45:00+00:00",
            "url": "https://deepmind.google/discover/blog/gemini-4-argon-safety/",
        }
        # Subdomain / aspect specializations must NOT be collapsed into the primary release
        self.assertFalse(are_articles_closely_related(core_art, robotics_art))
        self.assertFalse(are_articles_closely_related(core_art, safety_art))

    def test_distinguishes_software_release_versions_and_divergent_subjects(self):
        v141 = {
            "title": "vLLM releases v0.6.3 with optimized speculative decoding",
            "company": "vLLM",
            "date": "2026-10-14T10:00:00+00:00",
            "url": "https://github.com/vllm-project/vllm/releases/tag/v0.6.3",
        }
        v140 = {
            "title": "vLLM releases v0.6.2 with optimized speculative decoding",
            "company": "vLLM",
            "date": "2026-10-14T10:00:00+00:00",
            "url": "https://github.com/vllm-project/vllm/releases/tag/v0.6.2",
        }
        self.assertFalse(are_articles_closely_related(v141, v140))


# ===========================================================================
# LiveBench 2.0 Adapter Tests
# ===========================================================================


class TestLiveBench20Adapter(unittest.TestCase):
    def setUp(self):
        self.adapter = LiveBenchAdapter()

    def test_allowed_benchmark_categories_include_livebench_20_dimensions(self):
        self.assertIn("Agentic", ALLOWED_BENCHMARK_CATEGORIES)
        self.assertIn("Browser Use", ALLOWED_BENCHMARK_CATEGORIES)
        self.assertIn("Research", ALLOWED_BENCHMARK_CATEGORIES)
        self.assertIn("Design", ALLOWED_BENCHMARK_CATEGORIES)
        self.assertIn("Composite", ALLOWED_BENCHMARK_CATEGORIES)

    def test_parses_livebench_20_csv_format(self):
        csv_sample = (
            "model,design_layout,research,browser,coding,long_horizon,data_analysis,reasoning\n"
            "claude-opus-5-5,82.5,88.0,87.5,92.5,85.0,80.2,92.5\n"
            "gpt-6-astra,80.0,85.0,84.0,91.0,82.4,81.0,94.0\n"
        )
        records = self.adapter.parse_table_data(csv_sample, release_date="2026-10-13", generation="2.0")
        self.assertGreater(len(records), 0)

        # Check category mappings
        categories_found = {r["category"] for r in records}
        self.assertIn("Agentic", categories_found)      # mapped from long_horizon
        self.assertIn("Browser Use", categories_found)  # mapped from browser
        self.assertIn("Research", categories_found)
        self.assertIn("Design", categories_found)
        self.assertIn("Composite", categories_found)   # computed overall average

        # Check metadata
        opus_agentic = next(r for r in records if (r["canonical_id"] == "claude-opus-5-5" or r["model_name"] == "Claude Opus 5.5") and r["category"] == "Agentic")
        self.assertEqual(opus_agentic["score"], 85.0)
        self.assertEqual(opus_agentic["benchmark_version"], "2.0")
        self.assertEqual(opus_agentic["benchmark_release_date"], "2026-10-13")

    def test_safe_fallback_on_unparseable_content(self):
        # Empty or malformed CSV does not raise unhandled exception
        bad_csv = "invalid,header,without,model\n1,2,3,4\n"
        records = self.adapter.parse_table_data(bad_csv, release_date="bad_release", generation="2.0")
        self.assertEqual(records, [])


# ===========================================================================
# Reconcile Existing Feed Tests
# ===========================================================================


class TestReconcileExistingFeed(unittest.TestCase):
    def test_reconciles_feed_strips_fragments_removes_distractors_and_merges_coverage(self):
        sample_records = [
            {
                "id": "aaa111",
                "title": "Gemini 4 Argon: Built to solve your hardest problems",
                "company": "Google DeepMind",
                "source": "Google DeepMind",
                "date": "2026-10-14T09:00:00+00:00",
                "fetched_at": "2026-10-14T09:00:00+00:00",
                "url": "https://deepmind.google/blog/gemini-4-argon/#intro",
                "summary": "DeepMind releases Gemini 4 Argon frontier reasoning model.",
                "category": "Product Release",
            },
            {
                "id": "bbb222",
                "title": "Google finally releases Gemini 4 Argon",
                "company": "VentureBeat",
                "source": "VentureBeat",
                "date": "2026-10-14T11:00:00+00:00",
                "fetched_at": "2026-10-14T11:00:00+00:00",
                "url": "https://venturebeat.com/ai/gemini-4-argon#comments",
                "summary": "VentureBeat reports on Google releasing Gemini 4 Argon.",
                "category": "News",
            },
            {
                "id": "ccc333",
                "title": "The Download: drone traffic controllers",
                "company": "MIT Tech Review",
                "source": "MIT Technology Review",
                "date": "2026-10-14T12:00:00+00:00",
                "fetched_at": "2026-10-14T12:00:00+00:00",
                "url": "https://technologyreview.com/download-drones",
                "summary": "Drone air traffic control towers. In brief: Anthropic model.",
                "category": "News",
            },
            {
                "id": "ddd444",
                "title": "Machine learning predicts space weather risks",
                "company": "arXiv",
                "source": "arXiv cs.LG",
                "date": "2026-10-14T13:00:00+00:00",
                "fetched_at": "2026-10-14T13:00:00+00:00",
                "url": "https://arxiv.org/abs/2610.12345",
                "summary": "Deep surrogate models forecast solar flares and coronal mass ejections.",
                "category": "Research",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            feed_file = Path(tmpdir) / "feed.json"
            feed_file.write_text(json.dumps(sample_records, indent=2), encoding="utf-8")

            # Run 1: reconcile
            report1 = reconcile_existing_feed(feed_path=feed_file)
            self.assertEqual(report1["total_removed"], 1)
            self.assertEqual(report1["total_merged_coverage"], 1)
            self.assertEqual(report1["fragment_urls_cleaned"], 2)

            with open(feed_file, "r", encoding="utf-8") as f:
                reconciled1 = json.load(f)

            self.assertEqual(len(reconciled1), 2)
            # Fragment removed from canonical base
            self.assertEqual(reconciled1[0]["url"], "https://deepmind.google/blog/gemini-4-argon/")
            # Merged source
            self.assertIn("sources", reconciled1[0])
            self.assertEqual(len(reconciled1[0]["sources"]), 1)
            self.assertEqual(reconciled1[0]["sources"][0]["url"], "https://venturebeat.com/ai/gemini-4-argon")
            # Drone traffic distractor removed
            titles = [r["title"] for r in reconciled1]
            self.assertNotIn("The Download: drone traffic controllers", titles)
            self.assertIn("Machine learning predicts space weather risks", titles)

            # Run 2: idempotency test
            report2 = reconcile_existing_feed(feed_path=feed_file)
            self.assertEqual(report2["total_removed"], 0)
            self.assertEqual(report2["total_merged_coverage"], 0)
            self.assertEqual(report2["fragment_urls_cleaned"], 0)

            with open(feed_file, "r", encoding="utf-8") as f:
                reconciled2 = json.load(f)

            self.assertEqual(reconciled1, reconciled2)


class TestMaintenanceQualityAndHealthPass(unittest.TestCase):
    """Regression test suite for maintenance pass quality and source health fixes."""

    def test_the_verge_excluded_from_fallback_and_configured_sources(self):
        """Ensure The Verge is not present in default fallback sources or sources.json."""
        from pipeline.sources import _DEFAULT_SOURCES, load_sources

        for src in _DEFAULT_SOURCES:
            self.assertNotIn("the verge", src.get("name", "").lower())
            self.assertNotIn("theverge.com", src.get("url", "").lower())
            self.assertNotIn("the verge", src.get("publisher", "").lower())

        all_sources = load_sources(include_disabled=True)
        for src in all_sources:
            self.assertNotIn("the verge", src.get("name", "").lower())
            self.assertNotIn("theverge.com", src.get("url", "").lower())
            self.assertNotIn("the verge", src.get("publisher", "").lower())

    def test_merge_records_strictly_rejects_the_verge_primary_and_secondary(self):
        """Verify merge_records prevents The Verge from being reintroduced as primary or secondary."""
        existing = [
            {
                "id": "existing_valid",
                "title": "Google DeepMind Releases Gemini 4 Argon",
                "company": "Google DeepMind",
                "source": "Google DeepMind Blog",
                "url": "https://deepmind.google/blog/gemini-4-argon/",
                "date": "2026-10-14T10:00:00+00:00",
                "category": "Research",
                "summary": "Google DeepMind releases Gemini 4 Argon.",
                "sources": [
                    {
                        "title": "Google DeepMind Releases Gemini 4 Argon",
                        "url": "https://deepmind.google/blog/gemini-4-argon/",
                        "publisher": "Google DeepMind",
                    },
                    {
                        "title": "The Verge on Gemini 4",
                        "url": "https://www.theverge.com/ai/gemini-4-argon-launch",
                        "publisher": "The Verge",
                    },
                ],
            }
        ]

        new_articles = [
            # Verge primary article — must be completely dropped
            {
                "id": "verge_primary",
                "title": "The Verge: Gemini 4 Hands-on Review",
                "company": "The Verge",
                "source": "The Verge – AI",
                "url": "https://www.theverge.com/gemini-4-review",
                "date": "2026-10-14T11:00:00+00:00",
                "category": "News",
                "summary": "Hands-on review of Gemini 4.",
            },
            # Valid non-Verge article with Verge secondary source — Verge secondary must be stripped
            {
                "id": "techcrunch_article",
                "title": "TechCrunch Analysis of Gemini 4",
                "company": "TechCrunch",
                "source": "TechCrunch AI",
                "url": "https://techcrunch.com/gemini-4-analysis",
                "date": "2026-10-14T12:00:00+00:00",
                "category": "News",
                "summary": "TechCrunch analysis of Gemini 4.",
                "sources": [
                    {
                        "title": "TechCrunch Analysis",
                        "url": "https://techcrunch.com/gemini-4-analysis",
                        "publisher": "TechCrunch",
                    },
                    {
                        "title": "The Verge Coverage",
                        "url": "https://www.theverge.com/gemini-4-analysis",
                        "publisher": "The Verge",
                    },
                ],
            },
        ]

        merged = merge_records(existing, new_articles, filter_relevance=False)

        # 1. Verge primary was not added
        urls = [r["url"] for r in merged]
        self.assertNotIn("https://www.theverge.com/gemini-4-review", urls)

        # 2. Existing record had Verge secondary stripped
        deepmind_rec = next(r for r in merged if r["id"] == "existing_valid")
        for s in deepmind_rec["sources"]:
            self.assertNotIn("theverge.com", s["url"])
            self.assertNotIn("the verge", s.get("publisher", "").lower())

        # 3. New non-Verge article had Verge secondary stripped
        tc_rec = next(r for r in merged if r["id"] == "techcrunch_article")
        for s in tc_rec["sources"]:
            self.assertNotIn("theverge.com", s["url"])
            self.assertNotIn("the verge", s.get("publisher", "").lower())

    def test_opencode_atom_release_parsing_and_relevance(self):
        """Verify OpenCode Atom releases with bare version titles are parsed, prefixed, and preserved without network calls."""
        from pipeline.scraper import fetch_feed
        from pipeline.sources import Source

        atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Release notes from opencode</title>
  <entry>
    <id>tag:github.com,2008:Repository/123456/v1.0.50</id>
    <updated>2026-10-20T12:00:00Z</updated>
    <link rel="alternate" type="text/html" href="https://github.com/anomalyco/opencode/releases/tag/v1.0.50"/>
    <title>v1.0.50</title>
    <content type="html">&lt;p&gt;OpenCode v1.0.50 introduces enhanced terminal agent workflows and prompt caching.&lt;/p&gt;</content>
  </entry>
</feed>"""
        source: Source = {
            "name": "OpenCode Updates",
            "url": "https://github.com/anomalyco/opencode/releases.atom",
            "publisher": "OpenCode",
            "category": "Product Release",
        }

        import feedparser
        parsed = feedparser.parse(atom_xml)
        with patch("pipeline.scraper.feedparser.parse", return_value=parsed):
            articles = fetch_feed(source)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["title"], "OpenCode v1.0.50")
        self.assertEqual(articles[0]["company"], "OpenCode")
        self.assertEqual(articles[0]["url"], "https://github.com/anomalyco/opencode/releases/tag/v1.0.50")
        self.assertEqual(articles[0]["date"], "2026-10-20T12:00:00+00:00")

    def test_pipeline_status_reflects_benchmark_stale_fallback_and_degraded_overall(self):
        """Verify generate_pipeline_status derives LiveBench release from metadata and reports degraded when benchmarks are stale fallback using controlled fixture."""
        from pipeline.run import generate_pipeline_status
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_file = Path(tmpdir) / "benchmarks.json"
            bench_data = {
                "metadata": {
                    "status": "stale_fallback",
                    "last_updated": "2026-10-24T11:50:54.490204+00:00",
                    "last_known_good_timestamp": "2026-10-24T06:46:46.596051+00:00",
                    "health": {
                        "status": "stale_fallback",
                        "last_benchmark_release_date": "2026-10-13",
                        "last_retrieval_time": "2026-10-24T11:50:54.490204+00:00",
                        "last_known_good_timestamp": "2026-10-24T06:46:46.596051+00:00",
                        "warnings": [
                            "Preserved last-known-good benchmark snapshot from 2026-10-24T06:46:46.596051+00:00."
                        ],
                    },
                },
                "benchmarks": [],
            }
            bench_file.write_text(json.dumps(bench_data), encoding="utf-8")

            status_file = Path(tmpdir) / "pipeline-status.json"
            status = generate_pipeline_status(
                status_path=status_file,
                feed_records=[],
                benchmarks_path=bench_file,
            )

            # Truthful degraded status because benchmarks are in stale_fallback
            self.assertEqual(status["overall_status"], "degraded")

            # LiveBench status matches fixture metadata
            livebench = next(s for s in status["benchmark_sources"] if s["name"] == "LiveBench")
            self.assertEqual(livebench["status"], "stale_fallback")
            self.assertEqual(livebench["release"], "2026-10-13")
            self.assertEqual(livebench["last_known_good_timestamp"], "2026-10-24T06:46:46.596051+00:00")
            self.assertEqual(livebench["last_retrieval_time"], "2026-10-24T11:50:54.490204+00:00")

            # Anthropic News is disabled
            anthropic_news = next(s for s in status["rss_sources"] if s["name"] == "Anthropic News")
            self.assertEqual(anthropic_news["status"], "disabled")

            # The Verge is not present anywhere
            rss_names = [s["name"] for s in status["rss_sources"]]
            self.assertNotIn("The Verge – AI", rss_names)
            self.assertNotIn("The Verge", rss_names)

    def test_pipeline_status_reports_healthy_when_benchmarks_and_feeds_are_healthy(self):
        """Verify pipeline status reports healthy when both benchmarks and feeds are healthy."""
        from pipeline.run import generate_pipeline_status
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_file = Path(tmpdir) / "benchmarks.json"
            bench_data = {
                "metadata": {
                    "status": "healthy",
                    "last_updated": "2026-10-24T12:00:00+00:00",
                    "last_known_good_timestamp": "2026-10-24T12:00:00+00:00",
                    "health": {
                        "status": "healthy",
                        "last_benchmark_release_date": "2026-10-24",
                        "last_retrieval_time": "2026-10-24T12:00:00+00:00",
                        "last_known_good_timestamp": "2026-10-24T12:00:00+00:00",
                        "warnings": [],
                    },
                },
                "benchmarks": [],
            }
            bench_file.write_text(json.dumps(bench_data), encoding="utf-8")

            status_file = Path(tmpdir) / "pipeline-status.json"
            status = generate_pipeline_status(
                status_path=status_file,
                feed_records=[{"source": "OpenAI News", "id": "1"}],
                benchmarks_path=bench_file,
            )
            self.assertEqual(status["overall_status"], "healthy")
            livebench = next(s for s in status["benchmark_sources"] if s["name"] == "LiveBench")
            self.assertEqual(livebench["status"], "active")
            self.assertEqual(livebench["last_known_good_timestamp"], "2026-10-24T12:00:00+00:00")

    def test_successful_source_marked_active(self):
        """Verify a successful source fetch records status 'active' and populates pipeline status accordingly."""
        from pipeline.scraper import fetch_feed
        from pipeline.run import generate_pipeline_status

        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda k, d=None: {
            "title": "OpenAI announces new AI capability",
            "link": "https://openai.com/news/new-ai-cap",
            "summary": "Full article about AI and LLM reasoning.",
        }.get(k, d)
        mock_entry.title = "OpenAI announces new AI capability"
        mock_entry.link = "https://openai.com/news/new-ai-cap"
        mock_entry.summary = "Full article about AI and LLM reasoning."
        mock_entry.description = "Full article about AI and LLM reasoning."
        mock_entry.content = []
        mock_entry.published_parsed = time.gmtime()

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]

        source = {
            "name": "OpenAI News",
            "url": "https://openai.com/news/rss.xml",
            "publisher": "OpenAI",
            "category": "Product Release",
        }
        outcome = {}
        with patch("pipeline.scraper.feedparser.parse", return_value=mock_feed):
            articles = fetch_feed(source, outcome=outcome)

        self.assertEqual(len(articles), 1)
        self.assertEqual(outcome.get("status"), "active")
        self.assertIsNotNone(outcome.get("checked_at"))

        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            bench_file = Path(tmpdir) / "benchmarks.json"
            bench_file.write_text(json.dumps({"metadata": {"status": "healthy", "health": {"status": "healthy"}}}), encoding="utf-8")
            status = generate_pipeline_status(
                status_path=status_file,
                feed_records=articles,
                source_outcomes={"OpenAI News": outcome},
                benchmarks_path=bench_file,
            )
            source_status = next(s for s in status["rss_sources"] if s["name"] == "OpenAI News")
            self.assertEqual(source_status["status"], "active")
            self.assertEqual(source_status["last_checked"], outcome["checked_at"])

    def test_failed_source_with_old_retained_articles_marked_failed_and_degrades_overall(self):
        """Verify that a source with old retained articles is marked fetch_failed when current fetch fails, degrading overall status."""
        from pipeline.scraper import fetch_feed
        from pipeline.run import generate_pipeline_status

        source = {
            "name": "Google DeepMind Blog",
            "url": "https://deepmind.google/blog/rss.xml",
            "publisher": "Google DeepMind",
            "category": "Research",
        }
        outcome = {}
        with patch("pipeline.scraper.feedparser.parse", side_effect=Exception("Connection refused")):
            articles = fetch_feed(source, outcome=outcome)

        self.assertEqual(len(articles), 0)
        self.assertEqual(outcome.get("status"), "fetch_failed")
        self.assertIn("Connection refused", outcome.get("error", ""))

        # Old retained articles exist for this source
        retained = [
            {"id": "dm1", "source": "Google DeepMind Blog", "title": "AlphaFold 3", "url": "https://deepmind.google/alphafold3"}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            bench_file = Path(tmpdir) / "benchmarks.json"
            # Even with healthy benchmarks, feed failure must degrade overall_status
            bench_file.write_text(json.dumps({"metadata": {"status": "healthy", "health": {"status": "healthy"}}}), encoding="utf-8")
            status = generate_pipeline_status(
                status_path=status_file,
                feed_records=retained,
                source_outcomes={"Google DeepMind Blog": outcome},
                benchmarks_path=bench_file,
            )
            dm_status = next(s for s in status["rss_sources"] if s["name"] == "Google DeepMind Blog")
            self.assertEqual(dm_status["status"], "fetch_failed")
            self.assertEqual(dm_status["retained_articles_count"], 1)
            self.assertEqual(status["overall_status"], "degraded")

    def test_empty_response_source_distinguished_from_failure(self):
        """Verify an empty feed response (HTTP 200, 0 entries) is marked empty_response, not failed."""
        from pipeline.scraper import fetch_feed
        from pipeline.run import generate_pipeline_status

        mock_feed = MagicMock()
        mock_feed.status = 200
        mock_feed.bozo = False
        mock_feed.entries = []

        source = {
            "name": "VS Code Updates",
            "url": "https://code.visualstudio.com/feed.xml",
            "publisher": "Microsoft VS Code",
            "category": "Product Release",
        }
        outcome = {}
        with patch("pipeline.scraper.feedparser.parse", return_value=mock_feed):
            articles = fetch_feed(source, outcome=outcome)

        self.assertEqual(len(articles), 0)
        self.assertEqual(outcome.get("status"), "empty_response")

        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            bench_file = Path(tmpdir) / "benchmarks.json"
            bench_file.write_text(json.dumps({"metadata": {"status": "healthy", "health": {"status": "healthy"}}}), encoding="utf-8")
            status = generate_pipeline_status(
                status_path=status_file,
                feed_records=[],
                source_outcomes={"VS Code Updates": outcome},
                benchmarks_path=bench_file,
            )
            vs_status = next(s for s in status["rss_sources"] if s["name"] == "VS Code Updates")
            self.assertEqual(vs_status["status"], "empty_response")
            # An empty response without failures should not degrade overall_status on healthy benchmarks
            self.assertEqual(status["overall_status"], "healthy")

    def test_one_source_failure_does_not_stop_other_sources(self):
        """Verify fetch_all_sources continues to ingest other sources when one source fails."""
        from pipeline.scraper import fetch_all_sources

        mock_entry = MagicMock()
        mock_entry.get.side_effect = lambda k, d=None: {
            "title": "Hugging Face releases new model",
            "link": "https://huggingface.co/blog/new-model",
            "summary": "Post about transformers and deep learning.",
        }.get(k, d)
        mock_entry.title = "Hugging Face releases new model"
        mock_entry.link = "https://huggingface.co/blog/new-model"
        mock_entry.summary = "Post about transformers and deep learning."
        mock_entry.description = "Post about transformers and deep learning."
        mock_entry.content = []
        mock_entry.published_parsed = time.gmtime()

        good_feed = MagicMock()
        good_feed.status = 200
        good_feed.bozo = False
        good_feed.entries = [mock_entry]

        sources = [
            {"name": "Failing Source", "url": "https://failing.example.com/rss", "publisher": "Failing", "category": "News"},
            {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog/feed.xml", "publisher": "Hugging Face", "category": "News"},
        ]

        def parse_side_effect(url, **kwargs):
            if "failing" in url:
                raise ConnectionError("Host unreachable")
            return good_feed

        outcomes = {}
        with patch("pipeline.scraper.feedparser.parse", side_effect=parse_side_effect):
            articles = fetch_all_sources(sources=sources, delay=0, outcomes=outcomes)

        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["source"], "Hugging Face Blog")
        self.assertEqual(outcomes["Failing Source"]["status"], "fetch_failed")
        self.assertEqual(outcomes["Hugging Face Blog"]["status"], "active")

    def test_zero_articles_updates_health_and_preserves_feed(self):
        """Verify run_pipeline updates pipeline-status.json and preserves existing feed when no new articles are found."""
        from pipeline.run import run_pipeline

        existing_feed = [
            {
                "id": "existing1",
                "title": "Existing Valid Article",
                "company": "OpenAI",
                "source": "OpenAI News",
                "summary": "Existing summary.",
                "url": "https://openai.com/existing",
                "date": "2026-10-20T12:00:00+00:00",
                "category": "Product Release",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            feed_file = Path(tmpdir) / "feed.json"
            feed_file.write_text(json.dumps(existing_feed), encoding="utf-8")
            status_file = Path(tmpdir) / "pipeline-status.json"

            empty_outcomes = {"OpenAI News": {"status": "empty_response", "checked_at": "2026-10-24T12:00:00Z", "articles_count": 0}}

            def mock_fetch(sources=None, delay=0.5, outcomes=None):
                if outcomes is not None:
                    outcomes.update(empty_outcomes)
                return []

            with patch("pipeline.run.DEFAULT_FEED_PATH", feed_file), \
                 patch("pipeline.run.DEFAULT_STATUS_PATH", status_file), \
                 patch("pipeline.run.fetch_all_sources", side_effect=mock_fetch):
                ret = run_pipeline(sources=[{"name": "OpenAI News", "url": "https://openai.com/rss", "publisher": "OpenAI", "category": "News"}])

            self.assertEqual(ret, 0)

            # 1. Existing feed was preserved
            preserved_feed = json.loads(feed_file.read_text(encoding="utf-8"))
            self.assertEqual(len(preserved_feed), 1)
            self.assertEqual(preserved_feed[0]["id"], "existing1")

            # 2. Pipeline status was still updated
            self.assertTrue(status_file.exists())
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            openai_status = next(s for s in status_data["rss_sources"] if s["name"] == "OpenAI News")
            self.assertEqual(openai_status["status"], "empty_response")

    def test_benchmark_fallback_snapshot_preserves_distinct_last_known_good_timestamp(self):
        """Verify that when benchmark fallback snapshot is used, last_known_good_timestamp is preserved and distinct from last_retrieval_time."""
        from pipeline.benchmarks.runner import run_benchmark_pipeline

        existing_snapshot = {
            "metadata": {
                "last_updated": "2026-10-24T06:46:46.596051+00:00",
                "last_known_good_timestamp": "2026-10-24T06:46:46.596051+00:00",
                "status": "healthy",
                "health": {
                    "status": "healthy",
                    "last_retrieval_time": "2026-10-24T06:46:46.596051+00:00",
                    "last_known_good_timestamp": "2026-10-24T06:46:46.596051+00:00",
                    "warnings": [],
                },
            },
            "benchmarks": [
                {
                    "model_name": "Claude 3.7 Sonnet",
                    "canonical_id": "claude-3-7-sonnet",
                    "provider": "Anthropic",
                    "benchmark_name": "LiveBench",
                    "category": "Composite",
                    "score": 85.5,
                    "score_unit": "%",
                    "evaluation_date": None,
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": "2026-10-24T06:46:46.596051+00:00",
                }
            ],
            "model_metrics": [],
            "benchmark_standards": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "benchmarks.json"
            bench_path.write_text(json.dumps(existing_snapshot), encoding="utf-8")

            # Mock adapters to return empty
            with patch("pipeline.benchmarks.adapters.arena.ArenaAdapter.fetch", return_value=[]), \
                 patch("pipeline.benchmarks.adapters.livebench.LiveBenchAdapter.fetch", return_value=[]), \
                 patch("pipeline.benchmarks.adapters.openrouter.OpenRouterAdapter.fetch_metrics", return_value=[]), \
                 patch("pipeline.models.updater.sync_models_catalog"):
                result = run_benchmark_pipeline(benchmarks_path=bench_path)

            meta = result["metadata"]
            health = meta["health"]

            self.assertEqual(health["status"], "stale_fallback")
            # last_known_good_timestamp is preserved from the 06:46 snapshot
            self.assertEqual(health["last_known_good_timestamp"], "2026-10-24T06:46:46.596051+00:00")
            self.assertEqual(meta["last_known_good_timestamp"], "2026-10-24T06:46:46.596051+00:00")
            # last_retrieval_time is updated to the current attempt time
            self.assertNotEqual(health["last_retrieval_time"], health["last_known_good_timestamp"])
            self.assertTrue(any("2026-10-24T06:46:46.596051+00:00" in w for w in health["warnings"]))


class TestPostDeploymentAuditFix(unittest.TestCase):
    """Deterministic regression tests for post-deployment audit fixes: LiveBench 2.0 and feed outcome separation."""

    @patch("requests.head")
    def test_livebench_discover_latest_release_2026_10_14(self, mock_head):
        """Verify discovery finds official 2026-10-14 release endpoint."""
        from pipeline.benchmarks.adapters.livebench import discover_latest_release

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_head.return_value = mock_resp

        release_date, is_stale = discover_latest_release(timeout=5)
        self.assertEqual(release_date, "2026-10-14")
        self.assertFalse(is_stale)
        mock_head.assert_called_once()
        called_url = mock_head.call_args[0][0]
        self.assertIn("20261014", called_url)

    def test_livebench_parses_official_2026_10_14_table_and_categories(self):
        """Verify parsing LiveBench 2.0 format with Long Agentic mapped to Agentic."""
        from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter

        adapter = LiveBenchAdapter(release_date="2026-10-14")
        csv_data = (
            "model,long_agentic,reasoning,coding,browser,data_analysis,research,design\n"
            "claude-opus-5.5,85.0,92.2,96.3,80.0,80.3,85.0,86.9\n"
            "gpt-5.2-omni,78.5,89.0,91.2,85.0,79.0,82.0,84.5\n"
        )
        categories = {
            "Long Agentic": ["long_agentic"],
            "Reasoning": ["reasoning"],
            "Coding": ["coding"],
            "Browser": ["browser"],
            "Data Analysis": ["data_analysis"],
            "Research": ["research"],
            "Design": ["design"],
        }
        records = adapter.parse_table_data(csv_data, categories=categories, release_date="2026-10-14", generation="2.0")
        self.assertGreater(len(records), 0)

        categories_found = {r["category"] for r in records}
        self.assertIn("Agentic", categories_found)
        self.assertIn("Reasoning", categories_found)
        self.assertIn("Coding", categories_found)
        self.assertIn("Browser Use", categories_found)
        self.assertIn("Composite", categories_found)

        opus = [r for r in records if "Opus 5.5" in r["model_name"]]
        self.assertEqual(len(opus), 8)
        composite = next(r for r in opus if r["category"] == "Composite")
        self.assertGreater(composite["score"], 80.0)
        self.assertEqual(composite["benchmark_release_date"], "2026-10-14")
        self.assertEqual(composite["benchmark_version"], "2.0")

    @patch("requests.head")
    @patch("requests.get")
    def test_livebench_upstream_failure_reports_stale_fallback(self, mock_get, mock_head):
        """Verify that when upstream LiveBench endpoints fail (404/network error), adapter and runner preserve last-known-good."""
        import requests
        from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter
        from pipeline.benchmarks.runner import run_benchmark_pipeline

        mock_head.side_effect = requests.RequestException("404 Not Found")
        mock_get.side_effect = requests.RequestException("404 Not Found")

        adapter = LiveBenchAdapter(timeout=2)
        records = adapter.fetch()
        self.assertEqual(records, [])

        existing_snapshot = {
            "metadata": {
                "last_updated": "2026-10-24T06:00:00+00:00",
                "last_known_good_timestamp": "2026-10-24T06:00:00+00:00",
                "status": "healthy",
            },
            "benchmarks": [
                {
                    "model_name": "Llama 3.3 70B",
                    "canonical_id": "meta/llama-3-3-70b-instruct",
                    "category": "Composite",
                    "score": 82.0,
                    "score_unit": "%",
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": "2026-10-24T06:00:00+00:00",
                }
            ],
            "model_metrics": [],
            "benchmark_standards": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = Path(tmpdir) / "benchmarks.json"
            bench_path.write_text(json.dumps(existing_snapshot), encoding="utf-8")

            with patch("pipeline.benchmarks.adapters.openrouter.OpenRouterAdapter.fetch_metrics", return_value=[]), \
                 patch("pipeline.models.updater.sync_models_catalog"):
                result = run_benchmark_pipeline(benchmarks_path=bench_path)

            health = result["metadata"]["health"]
            self.assertEqual(health["status"], "stale_fallback")
            self.assertEqual(len(result["benchmarks"]), 1)
            self.assertEqual(result["benchmarks"][0]["model_name"], "Llama 3.3 70B")

    @patch("feedparser.parse")
    def test_fetch_feed_distinguishes_empty_response_from_no_relevant_articles(self, mock_parse):
        """Verify separation between genuinely empty feeds and feeds with non-AI filtered entries."""
        from pipeline.scraper import fetch_feed

        source = {"name": "Test Feed", "publisher": "Test Pub", "url": "https://example.com/rss", "category": "News"}

        # 1. Truly empty response (0 raw entries)
        mock_feed_empty = MagicMock()
        mock_feed_empty.entries = []
        mock_feed_empty.bozo = False
        mock_feed_empty.status = 200
        mock_parse.return_value = mock_feed_empty

        outcome_empty = {}
        articles = fetch_feed(source, outcome=outcome_empty)
        self.assertEqual(len(articles), 0)
        self.assertEqual(outcome_empty["status"], "empty_response")
        self.assertEqual(outcome_empty["raw_entries_count"], 0)
        self.assertEqual(outcome_empty["relevant_articles_count"], 0)

        # 2. Feed with entries that all fail AI relevance filter
        mock_entry = MagicMock()
        mock_entry.link = "https://example.com/unrelated-drones"
        mock_entry.title = "Local drone traffic control towers for regional deliveries"
        mock_entry.description = "Commercial aviation discussion of regional package delivery and municipal permits."
        mock_entry.published_parsed = None
        mock_entry.updated_parsed = None

        mock_feed_non_ai = MagicMock()
        mock_feed_non_ai.entries = [mock_entry]
        mock_feed_non_ai.bozo = False
        mock_feed_non_ai.status = 200
        mock_parse.return_value = mock_feed_non_ai

        outcome_non_ai = {}
        articles = fetch_feed(source, outcome=outcome_non_ai)
        self.assertEqual(len(articles), 0)
        self.assertEqual(outcome_non_ai["status"], "no_relevant_articles")
        self.assertEqual(outcome_non_ai["raw_entries_count"], 1)
        self.assertEqual(outcome_non_ai["relevant_articles_count"], 0)

        # 3. Feed with genuine AI article
        mock_ai_entry = MagicMock()
        mock_ai_entry.link = "https://example.com/frontier-llm-reasoning"
        mock_ai_entry.title = "DeepMind introduces new reasoning model with chain-of-thought architecture"
        mock_ai_entry.description = "Transformers, reinforcement learning, and LLM reasoning benchmarks."
        mock_ai_entry.published_parsed = None
        mock_ai_entry.updated_parsed = None

        mock_feed_ai = MagicMock()
        mock_feed_ai.entries = [mock_ai_entry]
        mock_feed_ai.bozo = False
        mock_feed_ai.status = 200
        mock_parse.return_value = mock_feed_ai

        outcome_ai = {}
        articles = fetch_feed(source, outcome=outcome_ai)
        self.assertEqual(len(articles), 1)
        self.assertEqual(outcome_ai["status"], "active")
        self.assertEqual(outcome_ai["raw_entries_count"], 1)
        self.assertEqual(outcome_ai["relevant_articles_count"], 1)

    def test_pipeline_status_with_no_relevant_articles_remains_healthy(self):
        """Verify that a successful fetch with 0 AI-relevant articles does not degrade overall pipeline health."""
        from pipeline.run import generate_pipeline_status

        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "pipeline-status.json"
            benchmarks_path = Path(tmpdir) / "benchmarks.json"
            benchmarks_path.write_text(json.dumps({
                "metadata": {"health": {"status": "healthy", "last_benchmark_release_date": "2026-10-14"}}
            }), encoding="utf-8")

            # Outcomse with one source having no relevant articles and one source active
            sources = [
                {"name": "General Tech", "publisher": "Tech", "url": "https://example.com/tech", "category": "News", "enabled": True},
                {"name": "OpenAI News", "publisher": "OpenAI", "url": "https://example.com/openai", "category": "Product Release", "enabled": True},
            ]
            outcomes = {
                "General Tech": {
                    "status": "no_relevant_articles",
                    "checked_at": "2026-10-25T12:00:00+00:00",
                    "raw_entries_count": 10,
                    "relevant_articles_count": 0,
                },
                "OpenAI News": {
                    "status": "active",
                    "checked_at": "2026-10-25T12:00:00+00:00",
                    "raw_entries_count": 5,
                    "relevant_articles_count": 5,
                },
            }

            status = generate_pipeline_status(
                status_path=status_path,
                feed_records=[],
                source_outcomes=outcomes,
                benchmarks_path=benchmarks_path,
                sources=sources,
            )

            self.assertEqual(status["overall_status"], "healthy")
            gen_tech = next(s for s in status["rss_sources"] if s["name"] == "General Tech")
            self.assertEqual(gen_tech["status"], "no_relevant_articles")
            self.assertEqual(gen_tech["raw_entries_count"], 10)
            self.assertEqual(gen_tech["relevant_articles_count"], 0)



class TestArenaAdapter(unittest.TestCase):
    """Tests for Arena.ai (LMSYS Chatbot Arena) benchmark adapter."""

    def test_allowed_benchmark_categories_includes_human_preference(self):
        from pipeline.benchmarks.base import ALLOWED_BENCHMARK_CATEGORIES
        self.assertIn("Human Preference", ALLOWED_BENCHMARK_CATEGORIES)

    @patch("requests.get")
    def test_arena_adapter_fetch_and_parse_success(self, mock_get):
        from pipeline.benchmarks.adapters.arena import ArenaAdapter

        def mock_hf_response(url, *args, **kwargs):
            resp = MagicMock()
            resp.ok = True
            if "config=text" in str(url):
                resp.json.return_value = {
                    "rows": [
                        {
                            "row": {
                                "model_name": "Claude 3.7 Sonnet",
                                "organization": "anthropic",
                                "rating": 1345.8,
                                "rating_lower": 1339.2,
                                "rating_upper": 1352.4,
                                "vote_count": 85000,
                                "rank": 1,
                                "leaderboard_publish_date": "2026-10-28",
                            }
                        },
                        {
                            "row": {
                                "model_name": "GPT-4o (2024-11-20)",
                                "organization": "openai",
                                "rating": 1310.2,
                                "rating_lower": 1304.1,
                                "rating_upper": 1316.3,
                                "vote_count": 92000,
                                "rank": 2,
                                "leaderboard_publish_date": "2026-10-28",
                            }
                        },
                    ]
                }
            elif "config=webdev" in str(url):
                resp.json.return_value = {
                    "rows": [
                        {
                            "row": {
                                "model_name": "Claude 3.7 Sonnet",
                                "organization": "anthropic",
                                "rating": 1380.5,
                                "rating_lower": 1372.0,
                                "rating_upper": 1389.0,
                                "vote_count": 42000,
                                "rank": 1,
                                "leaderboard_publish_date": "2026-10-28",
                            }
                        }
                    ]
                }
            else:
                resp.json.return_value = {"rows": []}
            return resp

        mock_get.side_effect = mock_hf_response

        adapter = ArenaAdapter()
        records = adapter.fetch()

        self.assertEqual(len(records), 3)

        # Check Human Preference record
        text_rec = next(r for r in records if r["category"] == "Human Preference")
        self.assertEqual(text_rec["benchmark_name"], "Chatbot Arena (Text)")
        self.assertEqual(text_rec["score"], 1345.8)
        self.assertEqual(text_rec["score_unit"], "Elo")
        self.assertIsNone(text_rec["evaluation_date"])  # No fabrication
        self.assertEqual(text_rec["benchmark_release_date"], "2026-10-28")
        self.assertEqual(text_rec["benchmark_version"], "Arena")
        self.assertEqual(text_rec["provenance"], "Independent Evaluation")
        self.assertEqual(text_rec["source"], "Chatbot Arena")
        self.assertEqual(text_rec["provider"], "Anthropic")
        self.assertIn("Rank #1", text_rec["notes"])
        self.assertIn("85,000 votes", text_rec["notes"])

        # Check Coding record
        code_rec = next(r for r in records if r["category"] == "Coding")
        self.assertEqual(code_rec["benchmark_name"], "Chatbot Arena (Coding)")
        self.assertEqual(code_rec["score"], 1380.5)
        self.assertEqual(code_rec["score_unit"], "Elo")

    @patch("requests.get")
    def test_arena_adapter_handles_network_failure_gracefully(self, mock_get):
        from pipeline.benchmarks.adapters.arena import ArenaAdapter

        mock_get.side_effect = Exception("Connection timeout")

        adapter = ArenaAdapter()
        records = adapter.fetch()

        self.assertEqual(records, [])
        self.assertTrue(adapter.is_stale)

    @patch("requests.get")
    def test_arena_adapter_skips_invalid_rows(self, mock_get):
        from pipeline.benchmarks.adapters.arena import ArenaAdapter

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {
            "rows": [
                {"row": {"model_name": "", "rating": 1200.0}},
                {"row": {"model_name": "Valid Model", "rating": None}},
                {"row": {"model_name": "Valid Model 2", "rating": "invalid_num"}},
            ]
        }
        mock_get.return_value = resp

        adapter = ArenaAdapter()
        records = adapter.fetch()
        self.assertEqual(len(records), 0)

    def test_arena_provenance_notice(self):
        from pipeline.benchmarks.adapters.arena import ArenaAdapter

        adapter = ArenaAdapter()
        adapter.active_release = "2026-10-28"
        notice = adapter.get_provenance_notice()

        self.assertEqual(notice["source"], "Chatbot Arena")
        self.assertEqual(notice["release_date"], "2026-10-28")
        self.assertIn("Active (2026-10-28 Release)", notice["status"])
        self.assertIn("Bradley-Terry / Elo", notice["description"])
        self.assertIn("LMSYS Org", notice["attribution"])


if __name__ == "__main__":
    unittest.main()





