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
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.run import (
    load_existing_feed,
    merge_records,
    prepare_article,
    validate_record,
    write_feed_atomic,
)

# ---------------------------------------------------------------------------
# Unit under test imports
# ---------------------------------------------------------------------------
from pipeline.scraper import (
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
    @patch("pipeline.benchmarks.adapters.livebench.LiveBenchAdapter.fetch")
    @patch("pipeline.benchmarks.adapters.openrouter.OpenRouterAdapter.fetch_metrics")
    def test_pipeline_preserves_snapshot_on_outage(self, mock_or, mock_lb):
        from pipeline.benchmarks.runner import run_benchmark_pipeline

        # Simulate remote failure
        mock_lb.return_value = []
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
            # In LiveBench records where eval date is not reported, evaluation_date is None
            if b.get("provenance") == "Independent Evaluation" and "not reported" in b.get("notes", ""):
                self.assertIsNone(b["evaluation_date"])
                self.assertIsNotNone(b["benchmark_release_date"])

        # Check metadata health dates
        health = data.get("metadata", {}).get("health", {})
        self.assertIn("last_benchmark_release_date", health)
        self.assertEqual(health["last_benchmark_release_date"], "2026-06-25")



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
                "title": "Anthropic releases Claude 3.5 Sonnet - The Verge",
                "company": "The Verge",
                "source": "The Verge AI",
                "summary": "Anthropic has released its updated Claude 3.5 Sonnet model with enhanced coding capabilities.",
                "url": "https://www.theverge.com/2026/10/10/claude-3-5-sonnet",
                "date": "2026-10-10T14:00:00+00:00",
                "category": "News",
                "fetched_at": "2026-10-10T14:10:00+00:00",
                "sources": [
                    {
                        "title": "Anthropic releases Claude 3.5 Sonnet - The Verge",
                        "url": "https://www.theverge.com/2026/10/10/claude-3-5-sonnet",
                        "publisher": "The Verge",
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
        self.assertIn("https://www.theverge.com/2026/10/10/claude-3-5-sonnet", urls)
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
                "company": "The Verge",
                "source": "The Verge AI",
                "summary": "OpenAI is rolling out its Advanced Voice mode feature to users in the EU.",
                "url": "https://www.theverge.com/openai-advanced-voice-europe",
                "date": "2026-10-10T11:00:00+00:00",
                "category": "News",
                "fetched_at": "2026-10-10T11:05:00+00:00",
            }
        ]
        merged = merge_records(existing, new_incoming, max_articles=10)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()




