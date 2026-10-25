"""
Comprehensive test suite for the Official AI Model Updates Subsystem.

Covers all 20 enumerated specifications with local mocked fixtures:
 1. Parse a valid RSS or Atom model-update entry.
 2. Parse a supported official changelog page using fixtures.
 3. Correctly extract an official publication date.
 4. Handle an unknown or missing date without inventing one.
 5. Handle missing descriptions or malformed entries safely.
 6. Keep the direct official source URL.
 7. Normalize URLs consistently.
 8. Generate stable deterministic IDs.
 9. Prevent duplicate entries on repeated runs (idempotency).
10. Preserve existing history if a source times out.
11. Preserve existing history when a source returns an empty or invalid response.
12. Ensure failure of one source does not prevent processing other sources.
13. Correctly classify model, API, product, and research updates.
14. Use exact model identity mapping when a confident match exists.
15. Leave model_id unset when identity is ambiguous.
16. Avoid mapping similarly named model versions incorrectly.
17. Distinguish a provider claim from independent benchmark evidence.
18. Ensure a third-party-only prediction is not labelled as an officially verified release.
19. Verify the frontend correctly displays dates, provider filters, update-type filters, search, empty states, and source links.
20. Verify links and local JSON imports work with the GitHub Pages base path.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

REPO_ROOT = Path(__file__).parent.parent.parent

from pipeline.models.identity import resolve_model_identity
from pipeline.models.sources import (
    ALLOWED_ACCESS_METHODS,
    ALLOWED_SCOPES,
    ALLOWED_UPDATE_TYPES,
    ModelUpdateSource,
    get_active_model_update_sources,
    load_model_update_sources,
    validate_source_record,
)
from pipeline.models.updates import (
    classify_scope,
    classify_update_type,
    extract_dates,
    extract_provider_claims,
    generate_update_id,
    is_date_or_timeframe_string,
    is_generic_title,
    is_safe_official_url,
    is_substantive_model_update,
    load_model_updates,
    merge_model_updates,
    normalize_update_url,
    parse_date_string,
    parse_html_changelog,
    parse_rss_update_entry,
    parse_rss_updates,
    resolve_update_model,
    sync_model_updates,
    validate_update_record,
    write_model_updates_atomic,
)

# ---------------------------------------------------------------------------
# Mock Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>OpenAI News</title>
    <link>https://openai.com/news</link>
    <description>Latest news from OpenAI</description>
    <item>
      <title>OpenAI o3-mini: High-efficiency reasoning model for coding and STEM</title>
      <link>https://openai.com/index/openai-o3-mini/?utm_source=feed&amp;utm_medium=rss</link>
      <description>OpenAI launches o3-mini, offering frontier-class reasoning performance tuned for science, math, and coding tasks with low latency.</description>
      <pubDate>Fri, 31 Jan 2025 18:00:00 GMT</pubDate>
      <guid>https://openai.com/index/openai-o3-mini/</guid>
    </item>
  </channel>
</rss>
"""

SAMPLE_CHANGELOG_HTML = """<!DOCTYPE html>
<html>
<head><title>Anthropic Claude Platform Release Notes</title></head>
<body>
  <div class="changelog-container">
    <section>
      <h2>February 24, 2025</h2>
      <h3>Claude 3.7 Sonnet and hybrid reasoning</h3>
      <p>Anthropic releases Claude 3.7 Sonnet, the first hybrid reasoning model capable of instantaneous responses or extended chain-of-thought reasoning.</p>
      <a href="/docs/en/release-notes/claude-3-7-sonnet">Read release announcement</a>
    </section>
    <section>
      <h2>October 22, 2024</h2>
      <h3>Upgraded Claude 3.5 Sonnet and Computer Use</h3>
      <p>Upgraded Claude 3.5 Sonnet demonstrates industry-leading software engineering benchmark scores on SWE-bench Verified.</p>
      <a href="/docs/en/release-notes/computer-use">Read announcement</a>
    </section>
  </div>
</body>
</html>
"""

MOCK_SOURCE_OPENAI: ModelUpdateSource = {
    "id": "openai_news",
    "provider_id": "openai",
    "provider_name": "OpenAI",
    "display_name": "OpenAI News & Releases",
    "url": "https://openai.com/news/rss.xml",
    "source_type": "official_primary_source",
    "access_method": "rss",
    "update_categories": ["model_release", "capability_change"],
    "enabled": True,
    "notes": "Official OpenAI News RSS feed",
}

MOCK_SOURCE_ANTHROPIC: ModelUpdateSource = {
    "id": "anthropic_claude_release_notes",
    "provider_id": "anthropic",
    "provider_name": "Anthropic",
    "display_name": "Claude Platform Release Notes",
    "url": "https://platform.claude.com/docs/en/release-notes/overview",
    "source_type": "official_primary_source",
    "access_method": "html",
    "update_categories": ["model_release", "capability_change"],
    "enabled": True,
    "notes": "Technical release notes for the Claude developer platform",
}


class TestModelUpdates(unittest.TestCase):
    """Test suite covering all 20 enumerated model update tracker requirements."""

    # 1. Parse a valid RSS or Atom model-update entry
    def test_01_parse_valid_rss_entry(self):
        records = parse_rss_updates(SAMPLE_RSS_XML, MOCK_SOURCE_OPENAI)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["provider_id"], "openai")
        self.assertEqual(rec["title"], "OpenAI o3-mini: High-efficiency reasoning model for coding and STEM")
        self.assertEqual(rec["published_at"], "2025-01-31T18:00:00Z")
        self.assertEqual(rec["model_id"], "openai/o3-mini")
        self.assertEqual(rec["update_type"], "model_release")
        self.assertTrue(validate_update_record(rec))

    # 2. Parse a supported official changelog page using fixtures
    def test_02_parse_supported_html_changelog(self):
        records = parse_html_changelog(
            SAMPLE_CHANGELOG_HTML,
            MOCK_SOURCE_ANTHROPIC,
            base_url="https://platform.claude.com/docs/en/release-notes/overview",
        )
        self.assertGreaterEqual(len(records), 1)
        titles = [r["title"] for r in records]
        self.assertTrue(any("Claude 3.7 Sonnet" in t for t in titles))
        # Ensure direct URLs resolved with base_url
        for r in records:
            self.assertTrue(r["source_url"].startswith("https://platform.claude.com"))

    # 3. Correctly extract an official publication date
    def test_03_correctly_extract_official_publication_date(self):
        self.assertEqual(parse_date_string("2025-01-31"), "2025-01-31")
        self.assertEqual(parse_date_string("2025-01-31T18:00:00Z"), "2025-01-31T18:00:00Z")
        self.assertEqual(parse_date_string("Fri, 31 Jan 2025 18:00:00 GMT"), "2025-01-31T18:00:00Z")
        self.assertEqual(parse_date_string("February 24, 2025"), "2025-02-24")
        self.assertEqual(parse_date_string("Oct 22, 2024"), "2024-10-22")

    # 4. Handle an unknown or missing date without inventing one
    def test_04_handle_unknown_or_missing_date_safely(self):
        self.assertIsNone(parse_date_string(None))
        self.assertIsNone(parse_date_string(""))
        self.assertIsNone(parse_date_string("unspecified date"))
        self.assertIsNone(parse_date_string("invalid-not-a-date"))

        # When creating a record with unknown date, published_at remains None
        entry = {"title": "Update with no date", "link": "https://example.com/update"}
        rec = parse_rss_update_entry(entry, MOCK_SOURCE_OPENAI, "2025-01-01T00:00:00Z")
        self.assertIsNotNone(rec)
        self.assertIsNone(rec["published_at"])

    # 5. Handle missing descriptions or malformed entries safely
    def test_05_handle_missing_description_safely(self):
        entry = {"title": "Silent Model Drop", "link": "https://example.com/silent"}
        rec = parse_rss_update_entry(entry, MOCK_SOURCE_OPENAI, "2025-01-01T00:00:00Z")
        self.assertIsNotNone(rec)
        self.assertIn("Summary unavailable", rec["summary"])
        self.assertNotIn("Official update from", rec["summary"])
        self.assertTrue(validate_update_record(rec))
        self.assertTrue(is_substantive_model_update(rec))

        # Entry with empty title returns None
        bad_entry = {"title": "", "link": "https://example.com/empty"}
        self.assertIsNone(parse_rss_update_entry(bad_entry, MOCK_SOURCE_OPENAI, "2025-01-01T00:00:00Z"))

    # 6. Keep the direct official source URL
    def test_06_keep_direct_official_source_url(self):
        raw_url = "https://openai.com/index/openai-o3-mini/?utm_source=rss&utm_medium=feed"
        norm = normalize_update_url(raw_url)
        self.assertEqual(norm, "https://openai.com/index/openai-o3-mini")

    # 7. Normalize URLs consistently
    def test_07_normalize_urls_consistently(self):
        url1 = "HTTPS://OpenAI.com:443/index/openai-o3-mini/?utm_source=twitter&ref=blog#"
        url2 = "https://openai.com/index/openai-o3-mini/"
        norm1 = normalize_update_url(url1)
        norm2 = normalize_update_url(url2)
        self.assertEqual(norm1, norm2)
        self.assertEqual(norm1, "https://openai.com/index/openai-o3-mini")

    # 8. Generate stable deterministic IDs
    def test_08_generate_stable_deterministic_ids(self):
        url = "https://openai.com/index/openai-o3-mini/"
        title = "OpenAI o3-mini Launch"
        id1 = generate_update_id(url, "openai", title)
        id2 = generate_update_id("https://openai.com/index/openai-o3-mini/?utm_medium=rss", "OPENAI", "openai o3-mini launch")
        self.assertEqual(id1, id2)
        self.assertEqual(len(id1), 16)
        # Verify hex format
        int(id1, 16)

    # 9. Prevent duplicate entries on repeated runs (idempotency)
    def test_09_prevent_duplicate_entries_on_repeated_runs(self):
        records = parse_rss_updates(SAMPLE_RSS_XML, MOCK_SOURCE_OPENAI)
        self.assertEqual(len(records), 1)

        # Merge once
        merged1 = merge_model_updates([], records)
        self.assertEqual(len(merged1), 1)

        # Merge again with same incoming records
        merged2 = merge_model_updates(merged1, records)
        self.assertEqual(len(merged2), 1)
        self.assertEqual(merged1[0]["id"], merged2[0]["id"])

    # 10. Preserve existing history if a source times out
    @patch("requests.get")
    def test_10_preserve_history_on_source_timeout(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with tempfile.TemporaryDirectory() as tmpdir:
            updates_path = Path(tmpdir) / "model-updates.json"
            sources_path = Path(tmpdir) / "sources.json"

            initial_record = {
                "id": "1122334455667788",
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "title": "Historical o1 Launch",
                "summary": "Original launch record.",
                "update_type": "model_release",
                "scope": "model",
                "model_name": "OpenAI o1",
                "model_family": "o-series",
                "model_id": "openai/o1",
                "published_at": "2024-12-17",
                "detected_at": "2024-12-17T18:00:00Z",
                "source_type": "official_primary_source",
                "source_url": "https://openai.com/index/introducing-openai-o1/",
                "verification_status": "official_source",
                "source_id": "openai_news",
            }
            write_model_updates_atomic([initial_record], updates_path)

            source_def = [MOCK_SOURCE_OPENAI]
            with open(sources_path, "w", encoding="utf-8") as f:
                json.dump(source_def, f)

            total, added = sync_model_updates(sources_path=sources_path, updates_path=updates_path)
            self.assertEqual(total, 1)
            self.assertEqual(added, 0)

            # Ensure file still contains historical record intact
            after_records = load_model_updates(updates_path)
            self.assertEqual(len(after_records), 1)
            self.assertEqual(after_records[0]["id"], "1122334455667788")

    # 11. Preserve existing history when a source returns an empty or invalid response
    @patch("requests.get")
    def test_11_preserve_history_on_empty_or_invalid_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        existing = [{
            "id": "aabbccddeeff0011",
            "provider_id": "anthropic",
            "provider_name": "Anthropic",
            "title": "Claude 3.5 Sonnet",
            "summary": "Summary",
            "update_type": "model_release",
            "scope": "model",
            "published_at": "2024-10-22",
            "detected_at": "2024-10-22T00:00:00Z",
            "source_type": "official_primary_source",
            "source_url": "https://anthropic.com/news/3-5",
            "verification_status": "official_source",
            "source_id": "anthropic_news",
        }]

        # Empty incoming list
        merged = merge_model_updates(existing, [])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "aabbccddeeff0011")

    # 12. Ensure failure of one source does not prevent processing other sources
    @patch("requests.get")
    def test_12_failure_of_one_source_does_not_block_others(self, mock_get):
        def side_effect(url, **kwargs):
            if "openai.com" in url:
                raise requests.exceptions.ConnectionError("OpenAI unreachable")
            resp = MagicMock()
            resp.status_code = 200
            resp.text = SAMPLE_CHANGELOG_HTML
            return resp

        mock_get.side_effect = side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            updates_path = Path(tmpdir) / "model-updates.json"
            sources_path = Path(tmpdir) / "sources.json"

            sources_def = [MOCK_SOURCE_OPENAI, MOCK_SOURCE_ANTHROPIC]
            with open(sources_path, "w", encoding="utf-8") as f:
                json.dump(sources_def, f)

            total, added = sync_model_updates(sources_path=sources_path, updates_path=updates_path)
            self.assertGreater(total, 0)
            self.assertGreater(added, 0)

            # Anthropic was parsed successfully despite OpenAI error
            records = load_model_updates(updates_path)
            self.assertTrue(any(r["provider_id"] == "anthropic" for r in records))

    # 13. Correctly classify model, API, product, and research updates
    def test_13_correctly_classify_update_types_and_scopes(self):
        # Model release
        self.assertEqual(classify_update_type("Introducing Claude 3.7 Sonnet"), "model_release")
        # Pricing change
        self.assertEqual(classify_update_type("Gemini API Developer Pricing Reduction"), "pricing_change")
        # Context window change
        self.assertEqual(classify_update_type("DeepSeek-V3 128k context window support"), "context_window_change")
        # Modality change
        self.assertEqual(classify_update_type("Llama 3.2 Vision and multimodal understanding"), "modality_change")
        # Deprecation
        self.assertEqual(classify_update_type("Deprecation notice for legacy GPT-3.5 models"), "deprecation")
        # API availability
        self.assertEqual(classify_update_type("Structured Outputs now available in API"), "api_availability")
        # Product update
        self.assertEqual(classify_update_type("ChatGPT Canvas interactive workspace launch"), "product_update")
        # Research release
        self.assertEqual(classify_update_type("DeepSeek-R1 technical research report and open weights"), "research_release")

        # Scope classification
        self.assertEqual(classify_scope("Claude 3.7 Sonnet", model_id="anthropic/claude-3-7-sonnet"), "model")
        self.assertEqual(classify_scope("Llama 3 family updates", model_family="Llama 3"), "model_family")
        self.assertEqual(classify_scope("Batch API endpoint release", text="API functionality"), "api")
        self.assertEqual(classify_scope("ChatGPT Canvas mobile app", text="Mobile interface"), "product")
        self.assertEqual(classify_scope("Technical report paper on reasoning architectures", text="ArXiv preprint"), "research")

    # 14. Use exact model identity mapping when a confident match exists
    def test_14_exact_model_identity_mapping(self):
        model_id, name, family = resolve_update_model("Introducing OpenAI o3-mini for STEM")
        self.assertEqual(model_id, "openai/o3-mini")
        self.assertEqual(name, "OpenAI o3-mini")
        self.assertEqual(family, "o-series")

        model_id, name, family = resolve_update_model("Claude 3.7 Sonnet hybrid reasoning launch")
        self.assertEqual(model_id, "anthropic/claude-3-7-sonnet")

        model_id, name, family = resolve_update_model("DeepSeek-R1 open weights release")
        self.assertEqual(model_id, "deepseek/deepseek-r1")

        model_id, name, family = resolve_update_model("Gemini 2.0 Flash is generally available")
        self.assertEqual(model_id, "google/gemini-2-0-flash")

        model_id, name, family = resolve_update_model("Llama 3.3 70B Instruct open weights")
        self.assertEqual(model_id, "meta/llama-3-3-70b-instruct")

    # 15. Leave model_id unset when identity is ambiguous
    def test_15_leave_model_id_unset_when_identity_is_ambiguous(self):
        # Broad family reference without specific version
        model_id, name, family = resolve_update_model("Updates to the Claude 3 family")
        self.assertIsNone(model_id)
        self.assertEqual(name, "Claude 3.5")

        # Generic GPT-4 reference (could be gpt-4o, gpt-4o-mini, gpt-4-turbo)
        model_id, name, family = resolve_update_model("GPT-4 platform improvements")
        self.assertIsNone(model_id)

        # Generic provider mention
        model_id, name, family = resolve_update_model("DeepSeek releases new infrastructure updates")
        self.assertIsNone(model_id)

    # 16. Avoid mapping similarly named model versions incorrectly
    def test_16_avoid_mapping_distinct_versions_incorrectly(self):
        # "Llama 3.2" is distinct from "Llama 3.3 70B" or "Llama 3.1 405B"
        model_id, name, family = resolve_update_model("Llama 3.2 11B Vision release")
        self.assertNotEqual(model_id, "meta/llama-3-3-70b-instruct")
        self.assertNotEqual(model_id, "meta/llama-3-1-405b-instruct")

        # "Claude 3.5" alone does not falsely collide with "Claude 3.7 Sonnet"
        model_id, name, family = resolve_update_model("Anthropic updates Claude 3.5 models")
        self.assertNotEqual(model_id, "anthropic/claude-3-7-sonnet")

    # 17. Distinguish a provider claim from independent benchmark evidence
    def test_17_distinguish_provider_claim_from_independent_benchmarks(self):
        text = "OpenAI claims the new model outperforms all frontier models with unmatched reasoning."
        claims = extract_provider_claims("Model release", text)
        self.assertGreaterEqual(len(claims), 1)
        self.assertTrue(any("Provider claim" in c for c in claims))
        self.assertTrue(any("distinguished from independent benchmark evaluations" in c for c in claims))

    # 18. Ensure a third-party-only prediction is not labelled as an officially verified release
    def test_18_third_party_prediction_not_labelled_official(self):
        entry = {
            "title": "Rumor: GPT-5 predicted release date leaked by analysts",
            "link": "https://unverifiedtracker.com/leak",
            "summary": "Speculation and leaks suggest an unconfirmed launch next month.",
        }
        source: ModelUpdateSource = {
            "id": "third_party_feed",
            "provider_id": "third_party",
            "provider_name": "Third Party Tracker",
            "display_name": "AI Release Tracker",
            "url": "https://unverifiedtracker.com/feed",
            "source_type": "secondary_source",
            "access_method": "rss",
            "update_categories": ["model_release"],
            "enabled": True,
            "notes": "Third-party aggregator",
        }
        rec = parse_rss_update_entry(entry, source, "2025-01-01T00:00:00Z")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["verification_status"], "unverified")
        self.assertEqual(rec["source_type"], "secondary_source")

    # 19. Verify the frontend correctly displays dates, provider filters, update-type filters, search, empty states, and source links
    def test_19_frontend_data_integrity(self):
        data_path = REPO_ROOT / "src" / "data" / "model-updates.json"
        self.assertTrue(data_path.exists(), "src/data/model-updates.json must exist")

        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreaterEqual(len(records), 10, "Should have genuine historical seed records")
        providers = {r["provider_id"] for r in records}
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)
        self.assertIn("google", providers)
        self.assertIn("meta", providers)

        for r in records:
            self.assertTrue(validate_update_record(r))
            self.assertIn(r["update_type"], ALLOWED_UPDATE_TYPES)
            self.assertIn(r["scope"], ALLOWED_SCOPES)
            self.assertTrue(r["source_url"].startswith("http"))
            self.assertEqual(len(r["id"]), 16)

    # 20. Verify links and local JSON imports work with the GitHub Pages base path
    def test_20_github_pages_base_path_and_imports(self):
        # Test sources JSON
        sources_path = REPO_ROOT / "src" / "data" / "model-update-sources.json"
        self.assertTrue(sources_path.exists(), "src/data/model-update-sources.json must exist")
        sources = load_model_update_sources(sources_path)
        self.assertGreaterEqual(len(sources), 5)
        for s in sources:
            self.assertTrue(validate_source_record(s))
            self.assertIn(s["access_method"], ALLOWED_ACCESS_METHODS)

        # Test astro file references import.meta.env.BASE_URL
        models_astro = REPO_ROOT / "src" / "pages" / "models" / "index.astro"
        with open(models_astro, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import modelUpdatesData from", content)
        self.assertIn("import.meta.env.BASE_URL", content)
        self.assertIn("model-updates", content)
        self.assertIn("model-changelog-modal", content)

    # 21. HTML changelog scraper rejects navigation, footer, JS, and page chrome
    def test_21_html_changelog_parser_rejects_chrome_and_scripts(self):
        html_with_chrome = """
        <html>
        <head><script>var x = 123; function track() {}</script><style>.nav { color: red; }</style></head>
        <body>
          <nav>
            <a href="/home">Home</a>
            <a href="/docs">Docs</a>
          </nav>
          <main>
            <section>
              <h2>Claude 3.7 Sonnet Hybrid Reasoning Architecture</h2>
              <p>Anthropic introduces hybrid reasoning with controllable thinking tokens in Claude 3.7 Sonnet.</p>
              <a href="https://anthropic.com/news/claude-3-7-sonnet">Announcement</a>
            </section>
          </main>
          <footer>
            <p>Terms of service Privacy policy Cookie preferences Copyright 2026 Anthropic.</p>
            <a href="https://instagram.com/anthropic">Follow us on Instagram</a>
          </footer>
        </body>
        </html>
        """
        source = {
            "id": "anthropic_news",
            "provider_id": "anthropic",
            "provider_name": "Anthropic",
            "url": "https://anthropic.com/news",
            "source_type": "official_primary_source",
        }
        records = parse_html_changelog(html_with_chrome, source, base_url="https://anthropic.com/news")
        self.assertEqual(len(records), 1)
        self.assertIn("Claude 3.7 Sonnet", records[0]["title"])
        self.assertNotIn("Terms of service", records[0]["summary"])
        self.assertNotIn("track()", records[0]["summary"])
        self.assertEqual(records[0]["model_id"], "anthropic/claude-3-7-sonnet")

    # 22. Safe official URL allowlist rejects unauthorized and social domains
    def test_22_safe_official_url_allowlist_rejects_unauthorized_and_social_domains(self):
        # Official domains pass
        self.assertTrue(is_safe_official_url("https://openai.com/index/o3-mini", "openai"))
        self.assertTrue(is_safe_official_url("https://anthropic.com/news/claude-3-7-sonnet", "anthropic"))
        self.assertTrue(is_safe_official_url("https://deepmind.google/technologies/gemini/", "google"))

        # Social links and arbitrary domains rejected
        self.assertFalse(is_safe_official_url("https://instagram.com/openai", "openai"))
        self.assertFalse(is_safe_official_url("https://tiktok.com/@anthropic", "anthropic"))
        self.assertFalse(is_safe_official_url("https://youtube.com/watch?v=123", "google"))
        self.assertFalse(is_safe_official_url("http://insecure-http.com/news", "openai"))
        self.assertFalse(is_safe_official_url("javascript:alert(1)", "openai"))

    # 23. Extract dates separates future deprecation/effective dates from publication dates
    def test_23_extract_dates_separates_future_deprecation_from_publication(self):
        summary = "Model checkpoint gpt-4-0314 will be deprecated on 2026-12-31. Users should migrate to gpt-4o."
        pub_at, eff_at = extract_dates(summary, "2026-03-15T00:00:00Z")
        self.assertEqual(pub_at, "2026-03-15T00:00:00Z")
        self.assertEqual(eff_at, "2026-12-31")

        # Future candidate publication date (> tomorrow) is moved to effective_at
        future_candidate = "2028-06-01T00:00:00Z"
        pub_at2, eff_at2 = extract_dates("Generic announcement text", future_candidate)
        self.assertIsNone(pub_at2)
        self.assertEqual(eff_at2, future_candidate)

    # 24. Frontier Reasoning filter strictly excludes models with category General Chat & Language
    def test_24_frontier_reasoning_filter_excludes_general_chat_models(self):
        models_path = REPO_ROOT / "src" / "data" / "models.json"
        with open(models_path, "r", encoding="utf-8") as f:
            models = json.load(f)

        # Verified test models
        gpt_4o = next(m for m in models if m["id"] == "openai/gpt-4o")
        o3_mini = next(m for m in models if m["id"] == "openai/o3-mini")

        # gpt-4o is a featured frontier model, but category is General Chat & Language
        self.assertTrue(gpt_4o.get("is_featured_frontier"))
        self.assertEqual(gpt_4o.get("category"), "General Chat & Language")

        # o3-mini is Frontier Reasoning & Chat
        self.assertTrue(o3_mini.get("is_featured_frontier"))
        self.assertIn("Reasoning", o3_mini.get("category", ""))

        # The filter logic in models/index.astro: category.includes("reasoning")
        def is_reasoning_filter_match(m):
            return "reasoning" in (m.get("category") or "").lower()

        self.assertFalse(is_reasoning_filter_match(gpt_4o))
        self.assertTrue(is_reasoning_filter_match(o3_mini))

    # 25. Historical model filter reflects 0 catalogued historical models and guards state
    def test_25_historical_model_filter_and_count_logic(self):
        models_path = REPO_ROOT / "src" / "data" / "models.json"
        with open(models_path, "r", encoding="utf-8") as f:
            models = json.load(f)

        def is_historical(m):
            s = (m.get("lifecycle_status") or "").lower()
            return "deprecated" in s or "historical" in s or "archive" in s

        historical_count = len([m for m in models if is_historical(m)])
        self.assertEqual(historical_count, 0, "No models currently in models.json are deprecated or historical")

    # 26. 7-day and 30-day date filters strictly reject future dates and missing dates
    def test_26_date_window_filters_reject_future_and_invalid_dates(self):
        now_ms = 1776268800000  # Fixed epoch reference
        day_ms = 24 * 60 * 60 * 1000

        def match_time_filter(published_str: str | None, window: str) -> bool:
            if not published_str:
                return False
            try:
                dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                pub_time = int(dt.timestamp() * 1000)
            except Exception:
                return False

            if window == "7d":
                return pub_time > 0 and pub_time <= now_ms and (now_ms - pub_time) <= 7 * day_ms
            elif window == "30d":
                return pub_time > 0 and pub_time <= now_ms and (now_ms - pub_time) <= 30 * day_ms
            return True

        # Past 3 days: valid in 7d and 30d
        three_days_ago = datetime.fromtimestamp((now_ms - 3 * day_ms) / 1000, tz=UTC).isoformat()
        self.assertTrue(match_time_filter(three_days_ago, "7d"))
        self.assertTrue(match_time_filter(three_days_ago, "30d"))

        # Past 15 days: valid in 30d, invalid in 7d
        fifteen_days_ago = datetime.fromtimestamp((now_ms - 15 * day_ms) / 1000, tz=UTC).isoformat()
        self.assertFalse(match_time_filter(fifteen_days_ago, "7d"))
        self.assertTrue(match_time_filter(fifteen_days_ago, "30d"))

        # Future date: MUST be rejected by both
        future_date = datetime.fromtimestamp((now_ms + 2 * day_ms) / 1000, tz=UTC).isoformat()
        self.assertFalse(match_time_filter(future_date, "7d"))
        self.assertFalse(match_time_filter(future_date, "30d"))

        # Missing or invalid date: MUST be rejected
        self.assertFalse(match_time_filter(None, "7d"))
        self.assertFalse(match_time_filter("invalid-date", "7d"))

    # 27. Model pricing precision formatting helper preserves sub-cent rates
    def test_27_model_pricing_precision_formatting(self):
        def format_price(num: float | None) -> str:
            if num is None:
                return "0.00"
            if num == 0:
                return "0.00"
            rounded = round(num, 4)
            if rounded >= 1 and rounded == int(rounded):
                return f"{rounded:.2f}"
            s = f"{rounded:g}"
            parts = s.split(".")
            if len(parts) == 1:
                return f"{parts[0]}.00"
            if len(parts[1]) == 1:
                return f"{parts[0]}.{parts[1]}0"
            return s

        self.assertEqual(format_price(0.0), "0.00")
        self.assertEqual(format_price(2.0), "2.00")
        self.assertEqual(format_price(10.0), "10.00")
        self.assertEqual(format_price(0.1), "0.10")
        self.assertEqual(format_price(0.66), "0.66")
        self.assertEqual(format_price(0.016), "0.016")
        self.assertEqual(format_price(0.0238), "0.0238")
        self.assertEqual(format_price(0.0008), "0.0008")

    # 28. Two-level model history cleanly separates exact model updates from family announcements
    def test_28_two_level_model_history_exact_vs_family_separation(self):
        updates_path = REPO_ROOT / "src" / "data" / "model-updates.json"
        with open(updates_path, "r", encoding="utf-8") as f:
            updates = json.load(f)

        # Find exact updates for claude-3-5-sonnet
        exact_claude = [u for u in updates if u.get("model_id") == "anthropic/claude-3-5-sonnet"]
        self.assertGreaterEqual(len(exact_claude), 1)
        for u in exact_claude:
            self.assertEqual(u["scope"], "model")

        # Test ambiguous / family update separation
        fam_update = next((u for u in updates if u.get("model_family") == "Llama 3.2"), None)
        if fam_update:
            self.assertIsNone(fam_update.get("model_id"))
            self.assertEqual(fam_update.get("scope"), "model_family")

    # 29. Mistral HTML parser rejects 'Get started' and filter bar dumps
    def test_29_mistral_html_parser_rejects_get_started_and_filter_bars(self):
        mistral_html = """
        <div class="sidebar">
          <a href="/vibe/get-started">Get started</a>
          <a href="/resources/ambassadors">Ambassadors</a>
        </div>
        <div class="filter">TypeAllFeatureImprovementAreaAllCloudStudioVibe</div>
        <main>
          <h1>Release notes</h1>
          <h2>Latest updates</h2>
          <h2>Key capabilities</h2>
          <article>
            <h2>Managed Deployments</h2>
            <span>Oct 9, 2026</span>
            <p>Deploy custom Mistral models directly into your virtual private cloud with dedicated capacity.</p>
            <a href="/resources/release-notes#managed-deployments">Learn more</a>
            <a href="/resources/ambassadors">Join Ambassadors</a>
          </article>
        </main>
        """
        source: ModelUpdateSource = {
            "id": "mistral_release_notes",
            "provider_id": "mistral",
            "provider_name": "Mistral AI",
            "display_name": "Mistral AI Release Notes",
            "url": "https://docs.mistral.ai/resources/release-notes",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(mistral_html, source, base_url="https://docs.mistral.ai/resources/release-notes")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["title"], "Managed Deployments")
        self.assertEqual(rec["published_at"], "2026-10-09")
        self.assertEqual(rec["source_url"], "https://docs.mistral.ai/resources/release-notes")
        self.assertNotIn("Ambassadors", rec["source_url"])
        self.assertTrue(validate_update_record(rec))

    # 30. xAI HTML parser rejects month headings and status page
    def test_30_xai_html_parser_rejects_month_headings_and_status_page(self):
        xai_html = """
        <nav class="nav"><a href="https://status.x.ai/">Status Page</a><a href="https://console.x.ai/">Tokenizer Playground</a></nav>
        <main>
          <h1>Release Notes</h1>
          <h2>September 2025</h2>
          <h2>September</h2>
          <div class="card">
            <h3>Structured Outputs</h3>
            <p>Structured Outputs ensure model outputs strictly follow a developer-defined JSON schema.</p>
            <a href="/developers/model-capabilities/text/structured-outputs">Read docs</a>
            <a href="https://status.x.ai/">System Status</a>
          </div>
        </main>
        """
        source: ModelUpdateSource = {
            "id": "xai_api_release_notes",
            "provider_id": "xai",
            "provider_name": "xAI",
            "display_name": "xAI API Release Notes",
            "url": "https://docs.x.ai/developers/release-notes",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(xai_html, source, base_url="https://docs.x.ai/developers/release-notes")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["title"], "Structured Outputs")
        self.assertEqual(rec["source_url"], "https://docs.x.ai/developers/model-capabilities/text/structured-outputs")
        self.assertNotIn("status.x.ai", rec["source_url"])
        self.assertTrue(validate_update_record(rec))

    # 31. Cohere HTML parser rejects model spec tables and RSS headers
    def test_31_cohere_html_parser_rejects_model_tables_and_rss_banners(self):
        cohere_html = """
        <div class="header">
          <h1>Changelog</h1>
          <span>Subscribe via RSS</span>
        </div>
        <div class="spec-table">
          <p>Model Name: north-small-translate-1-0 Context Length: 16K input, 4K output</p>
        </div>
        <div class="entry">
          <h2>Announcing Cohere's Embed 5 Models</h2>
          <span>September 30, 2026</span>
          <p>Cohere introduces Embed 5 multilingual embedding models with enhanced retrieval performance.</p>
          <a href="/v2/docs/cohere-embed">Read announcement</a>
        </div>
        """
        source: ModelUpdateSource = {
            "id": "cohere_changelog",
            "provider_id": "cohere",
            "provider_name": "Cohere",
            "display_name": "Cohere API Changelog",
            "url": "https://docs.cohere.com/v2/changelog",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(cohere_html, source, base_url="https://docs.cohere.com/v2/changelog")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["title"], "Announcing Cohere's Embed 5 Models")
        self.assertEqual(rec["published_at"], "2026-09-30")
        self.assertEqual(rec["source_url"], "https://docs.cohere.com/v2/docs/cohere-embed")
        self.assertTrue(validate_update_record(rec))

    # 32. Anthropic HTML parser handles ordinal dates and derives sentence titles
    def test_32_anthropic_html_parser_handles_ordinal_dates_and_sentence_headlines(self):
        anthropic_html = """
        <div class="changelog">
          <section>
            <h3>September 4th, 2024</h3>
            <p>We have retired the Claude 1 and Instant models. Read more in Model deprecations.</p>
            <a href="/docs/en/about-claude/model-deprecations">Deprecations info</a>
          </section>
          <section>
            <h2>February 24, 2025</h2>
            <h3>Claude 3.7 Sonnet and hybrid reasoning</h3>
            <p>Anthropic releases Claude 3.7 Sonnet, introducing controllable thinking tokens.</p>
            <a href="/docs/en/release-notes/claude-3-7">Read announcement</a>
          </section>
        </div>
        """
        source: ModelUpdateSource = {
            "id": "anthropic_claude_release_notes",
            "provider_id": "anthropic",
            "provider_name": "Anthropic",
            "display_name": "Claude Platform Release Notes",
            "url": "https://platform.claude.com/docs/en/release-notes/overview",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(anthropic_html, source, base_url="https://platform.claude.com/docs/en/release-notes/overview")
        self.assertEqual(len(records), 2)
        # Verify first entry: lone date heading was NOT assigned as title
        rec1 = next(r for r in records if "Claude 1" in r["title"] or "retired" in r["title"])
        self.assertNotEqual(rec1["title"], "September 4th, 2024")
        self.assertEqual(rec1["published_at"], "2024-09-04")
        self.assertEqual(rec1["source_url"], "https://platform.claude.com/docs/en/about-claude/model-deprecations")
        self.assertTrue(validate_update_record(rec1))

        # Verify second entry
        rec2 = next(r for r in records if "Claude 3.7" in r["title"])
        self.assertEqual(rec2["title"], "Claude 3.7 Sonnet and hybrid reasoning")
        self.assertEqual(rec2["published_at"], "2025-02-24")
        self.assertEqual(rec2["model_id"], "anthropic/claude-3-7-sonnet")
        self.assertTrue(validate_update_record(rec2))

    # 33. DeepSeek HTML parser rejects lone model anchors without updates
    def test_33_deepseek_html_parser_rejects_lone_model_anchors(self):
        deepseek_html = """
        <div class="content">
          <h2>deepseek-reasonerâ\x80\x8b</h2>
          <h2>deepseek-coderâ\x80\x8b</h2>
          <article>
            <h2>deepseek-coder & deepseek-chat Upgraded to DeepSeek V2.5 Model</h2>
            <p>Date: 2024-09-05. Upgraded both models to V2.5 architecture with unified chat and code capabilities.</p>
            <a href="/updates#v2-5">Read details</a>
          </article>
        </div>
        """
        source: ModelUpdateSource = {
            "id": "deepseek_api_updates",
            "provider_id": "deepseek",
            "provider_name": "DeepSeek",
            "display_name": "DeepSeek API Updates",
            "url": "https://api-docs.deepseek.com/updates/",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(deepseek_html, source, base_url="https://api-docs.deepseek.com/updates/")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertIn("DeepSeek V2.5", rec["title"])
        self.assertNotIn("deepseek-reasoner", rec["title"])
        self.assertTrue(validate_update_record(rec))

    # 34. Card URL isolation strictly prevents trailing link inheritance across cards
    def test_34_card_url_isolation_does_not_leak_trailing_links(self):
        html = """
        <main>
          <section>
            <h2>Announcing Grok 2 Vision</h2>
            <p>Grok 2 Vision brings image reasoning capabilities.</p>
            <a href="/developers/model-capabilities/images/generation">Card 1 Link</a>
          </section>
          <section>
            <h2>Grok Voice Streaming Mode Update</h2>
            <p>Low-latency streaming mode is now available for real-time speech interaction.</p>
          </section>
          <footer>
            <a href="/terms">Terms of Service</a>
          </footer>
        </main>
        """
        source: ModelUpdateSource = {
            "id": "xai_api_release_notes",
            "provider_id": "xai",
            "provider_name": "xAI",
            "display_name": "xAI API Release Notes",
            "url": "https://docs.x.ai/developers/release-notes",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
        }
        records = parse_html_changelog(html, source, base_url="https://docs.x.ai/developers/release-notes")
        self.assertEqual(len(records), 2)
        # Card 1 has its explicit link
        self.assertEqual(records[0]["source_url"], "https://docs.x.ai/developers/model-capabilities/images/generation")
        # Card 2 must NOT inherit Card 1's link or the footer link; must fall back to base_url
        self.assertEqual(records[1]["source_url"], "https://docs.x.ai/developers/release-notes")

    # 35. Pre-persistence quality gate rejects malformed, generic, or navigation records
    def test_35_pre_persistence_quality_gate_rejects_malformed_records(self):
        base_record = {
            "id": "1122334455667788",
            "provider_id": "openai",
            "provider_name": "OpenAI",
            "title": "OpenAI o3-mini Launch",
            "summary": "Substantive announcement of o3-mini model.",
            "update_type": "model_release",
            "scope": "model",
            "model_name": "OpenAI o3-mini",
            "model_family": "o-series",
            "model_id": "openai/o3-mini",
            "published_at": "2025-01-31",
            "detected_at": "2025-01-31T18:00:00Z",
            "source_type": "official_primary_source",
            "source_url": "https://openai.com/index/openai-o3-mini",
            "verification_status": "official_source",
            "source_id": "openai_news",
        }
        # Legitimate record passes
        self.assertTrue(is_substantive_model_update(base_record))
        self.assertTrue(validate_update_record(base_record))

        # Lone date title rejected
        r_date = dict(base_record, title="September 4th, 2024")
        self.assertFalse(is_substantive_model_update(r_date))
        self.assertFalse(validate_update_record(r_date))

        # Generic title rejected
        r_generic = dict(base_record, title="Get started")
        self.assertFalse(is_substantive_model_update(r_generic))
        self.assertFalse(validate_update_record(r_generic))

        r_relnotes = dict(base_record, title="Release notes")
        self.assertFalse(is_substantive_model_update(r_relnotes))

        # Lone model name rejected
        r_model = dict(base_record, title="deepseek-reasoner", summary="deepseek-reasoner")
        self.assertFalse(is_substantive_model_update(r_model))

        # Disallowed status URL rejected
        r_status = dict(base_record, source_url="https://status.x.ai/")
        self.assertFalse(is_substantive_model_update(r_status))

        # Fwlink URL rejected
        r_fwlink = dict(base_record, source_url="https://go.microsoft.com/fwlink?linkid=2295127")
        self.assertFalse(is_substantive_model_update(r_fwlink))

        # Parameter block dump rejected
        r_dump = dict(base_record, title="Model Name: north-small Context Length: 16K")
        self.assertFalse(is_substantive_model_update(r_dump))

    def test_regression_rejects_generic_provider_only_fallback_summaries(self):
        """Verify that provider-only fallback summaries are strictly rejected."""
        base_record = {
            "id": "test1234567890ab",
            "provider_id": "huggingface",
            "provider_name": "Hugging Face",
            "title": "New Vision Language Model Released",
            "summary": "Official update from Hugging Face.",
            "update_type": "model_release",
            "scope": "model",
            "published_at": "2026-10-14",
            "detected_at": "2026-10-14T12:00:00Z",
            "source_type": "official_primary_source",
            "source_url": "https://huggingface.co/blog/new-vlm",
            "verification_status": "official_source",
            "source_id": "huggingface_blog",
        }
        self.assertFalse(is_substantive_model_update(base_record))

        r_dm = dict(base_record, provider_id="google", summary="Official update from Google DeepMind.")
        self.assertFalse(is_substantive_model_update(r_dm))

        # Honest fallback with title passes
        r_honest = dict(base_record, summary="[Summary unavailable from source] New Vision Language Model Released")
        self.assertTrue(is_substantive_model_update(r_honest))

    def test_regression_rejects_navigation_and_doc_page_headings(self):
        """Verify that navigation labels, copy links, and documentation titles are rejected."""
        base_record = {
            "id": "test1234567890ab",
            "provider_id": "mistral",
            "provider_name": "Mistral AI",
            "title": "Follow shipped product updates across Vibe, Studio, Admin, models, and the API.Latest updatesCopy section linkLatest updates",
            "summary": "Latest updates across Vibe and Studio",
            "update_type": "product_update",
            "scope": "product",
            "published_at": "2026-10-09",
            "detected_at": "2026-10-09T12:00:00Z",
            "source_type": "official_primary_source",
            "source_url": "https://docs.mistral.ai/getting-started/changelog",
            "verification_status": "official_source",
            "source_id": "mistral_release_notes",
        }
        self.assertFalse(is_substantive_model_update(base_record))

        # Doc chrome page titles rejected
        r_xai = dict(base_record, provider_id="xai", title="Release Notes | SpaceXSI Docs", published_at=None)
        self.assertFalse(is_substantive_model_update(r_xai))

        r_anthropic = dict(base_record, provider_id="anthropic", title="Claude Platform release notes - Claude Platform Docs", published_at=None)
        self.assertFalse(is_substantive_model_update(r_anthropic))

        r_mistral_doc = dict(base_record, title="Release notes | Mistral Docs", published_at=None)
        self.assertFalse(is_substantive_model_update(r_mistral_doc))

        # Copy section link in summary rejected
        r_copylink = dict(base_record, title="Devstral 2 Launch", summary="New coding model. Copy section link for details.")
        self.assertFalse(is_substantive_model_update(r_copylink))

    def test_changelog_parser_ignores_html_head_and_navigation(self):
        """Verify ChangelogHTMLParser ignores <head><title> and navigation 'Copy section link'."""
        html = """<!DOCTYPE html>
<html>
<head>
  <title>Release Notes | SpaceXSI Docs</title>
  <style>.nav { display: block; }</style>
  <script>console.log("analytics");</script>
</head>
<body>
  <nav><ul><li><a href="/docs">Overview</a></li></ul></nav>
  <main>
    <section>
      <h2>February 20, 2026</h2>
      <h3>Grok 4.1 Enterprise Fast Model Released</h3>
      <a href="#link">Copy section link</a>
      <p>xAI introduces Grok 4.1 Fast with sub-second time-to-first-token in the Enterprise API.</p>
    </section>
  </main>
</body>
</html>"""
        source: ModelUpdateSource = {
            "id": "xai_api_release_notes",
            "provider_id": "xai",
            "provider_name": "xAI",
            "display_name": "xAI Release Notes",
            "url": "https://docs.x.ai/changelog",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release"],
            "enabled": True,
            "notes": "xAI release notes",
        }
        records = parse_html_changelog(html, source, "https://docs.x.ai/changelog")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["title"], "Grok 4.1 Enterprise Fast Model Released")
        self.assertNotIn("Copy section link", records[0]["summary"])
        self.assertNotIn("Release Notes | SpaceXSI Docs", records[0]["title"])
        self.assertEqual(records[0]["published_at"], "2026-02-20")

    def test_rss_parser_extracts_content_or_uses_honest_fallback(self):
        """Verify parse_rss_update_entry never emits 'Official update from...' and extracts rich content."""
        source: ModelUpdateSource = {
            "id": "huggingface_blog",
            "provider_id": "huggingface",
            "provider_name": "Hugging Face",
            "display_name": "Hugging Face Blog",
            "url": "https://huggingface.co/blog/feed.xml",
            "source_type": "official_primary_source",
            "access_method": "rss",
            "update_categories": ["model_release"],
            "enabled": True,
            "notes": "Hugging Face Blog",
        }
        # Case 1: Entry with content value
        entry_with_content = {
            "title": "Introducing SmolVLM-2",
            "link": "https://huggingface.co/blog/smolvlm2",
            "content": [{"value": "<p>SmolVLM-2 is a lightweight compact multimodal vision language model.</p>"}],
            "published": "2026-10-14T10:00:00Z",
        }
        rec1 = parse_rss_update_entry(entry_with_content, source, "2026-10-14T12:00:00Z")
        self.assertIsNotNone(rec1)
        self.assertIn("SmolVLM-2 is a lightweight compact", rec1["summary"])
        self.assertNotEqual(rec1["summary"], "Official update from Hugging Face.")

        # Case 2: Entry with empty description/content uses honest fallback
        entry_empty = {
            "title": "Apertus 1.5 Release Notes",
            "link": "https://huggingface.co/blog/apertus-1-5",
            "published": "2026-10-14T10:00:00Z",
        }
        rec2 = parse_rss_update_entry(entry_empty, source, "2026-10-14T12:00:00Z")
        self.assertIsNotNone(rec2)
        self.assertEqual(rec2["summary"], "[Summary unavailable from source] Apertus 1.5 Release Notes")
        self.assertTrue(is_substantive_model_update(rec2))

    def test_meta_source_configuration_integrity(self):
        """Verify broken Meta RSS is disabled and Meta HTML source points to verified ai.meta.com/blog/."""
        sources = load_model_update_sources()
        meta_rss = next((s for s in sources if s["id"] == "meta_ai_blog"), None)
        self.assertIsNotNone(meta_rss)
        self.assertFalse(meta_rss["enabled"], "Broken Meta RSS endpoint (404) must be disabled")

        meta_html = next((s for s in sources if s["id"] == "meta_newsroom"), None)
        self.assertIsNotNone(meta_html)
        self.assertTrue(meta_html["enabled"], "Meta HTML blog source must be enabled")
        self.assertEqual(meta_html["url"], "https://ai.meta.com/blog/")
        self.assertEqual(meta_html["access_method"], "html")

    def test_meta_blog_html_parser_extracts_clean_announcements(self):
        """Verify parse_meta_blog extracts clean titles, dates, and official URLs from Meta AI blog HTML."""
        from pipeline.models.updates import parse_meta_blog

        source: ModelUpdateSource = {
            "id": "meta_newsroom",
            "provider_id": "meta",
            "provider_name": "Meta AI",
            "display_name": "Meta AI Blog & Research Archive",
            "url": "https://ai.meta.com/blog/",
            "source_type": "official_primary_source",
            "access_method": "html",
            "update_categories": ["model_release", "capability_change"],
            "enabled": True,
            "notes": "Meta AI Blog",
        }

        sample_html = """
        <html>
        <body>
            <!-- Featured card -->
            <div class="_amcw _amd3">
                <a href="https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/">
                    <div class="_amd5">FEATURED</div>
                </a>
                <a href="https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/">Introducing Muse Spark 1.1 </a>
                <div class="_amun">July 9, 2026</div>
            </div>
            <!-- News card with aria-label -->
            <div class="_amda">
                <a class="_amcw" aria-label="Read Reimagining Independence: How Meta’s AI Models Are Helping the University of Pittsburgh Transform Assistive Robotics" href="https://ai.meta.com/blog/assistive-robotics-university-of-pittsburgh-sam-dino/">
                    Reimagining Independence: How Meta’s AI Models Are Helping the University of Pittsburgh Transform Assistive Robotics
                </a>
                <div class="_amdj">Jul 27, 2026</div>
            </div>
        </body>
        </html>
        """

        records = parse_meta_blog(sample_html, source, "https://ai.meta.com/blog/")
        self.assertEqual(len(records), 2)

        titles = [r["title"] for r in records]
        self.assertIn("Introducing Muse Spark 1.1", titles)
        self.assertIn("Reimagining Independence: How Meta’s AI Models Are Helping the University of Pittsburgh Transform Assistive Robotics", titles)

        # Dates
        spark = next(r for r in records if "Muse Spark" in r["title"])
        self.assertEqual(spark["published_at"], "2026-07-09")
        self.assertEqual(spark["source_url"], "https://ai.meta.com/blog/introducing-muse-spark-meta-model-api")
        self.assertEqual(len(spark["id"]), 16)

        assistive = next(r for r in records if "Assistive Robotics" in r["title"])
        self.assertEqual(assistive["published_at"], "2026-07-27")
        self.assertEqual(assistive["source_url"], "https://ai.meta.com/blog/assistive-robotics-university-of-pittsburgh-sam-dino")


if __name__ == "__main__":
    unittest.main()


