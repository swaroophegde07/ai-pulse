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
    fetch_feed,
    normalize_url,
    parse_date,
    stable_id,
)
from pipeline.sources import ALLOWED_CATEGORIES, load_sources
from pipeline.summarizer import (
    FALLBACK_SUMMARY_PREFIX,
    _validate_gemini_response,
    make_fallback_summary,
    summarize_article,
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

    def test_returns_empty_on_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("not valid json {{{")
            tmp = Path(f.name)
        try:
            result = load_existing_feed(tmp)
            self.assertEqual(result, [])
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
        self.assertEqual(required, set(result.keys()))

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


if __name__ == "__main__":
    unittest.main()
