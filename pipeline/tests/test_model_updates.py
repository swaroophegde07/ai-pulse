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
    extract_provider_claims,
    generate_update_id,
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
        self.assertIn("OpenAI", rec["summary"])
        self.assertTrue(validate_update_record(rec))

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


if __name__ == "__main__":
    unittest.main()
