"""
Tests for new AI Pulse features, RSS feed generation, and source registry additions.
"""

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from pipeline.sources import ALLOWED_CATEGORIES, ALLOWED_TOPICS, load_sources

REPO_ROOT = Path(__file__).parent.parent.parent


class TestSourcesRegistry(unittest.TestCase):
    def setUp(self):
        self.sources = load_sources()
        self.sources_by_id = {s.get("id"): s for s in self.sources if "id" in s}

    def test_opencode_source_present_and_valid(self):
        """OpenCode Updates must be registered as a valid product release source."""
        self.assertIn("opencode-updates", self.sources_by_id)
        src = self.sources_by_id["opencode-updates"]
        self.assertEqual(src["name"], "OpenCode Updates")
        self.assertEqual(src["publisher"], "OpenCode")
        self.assertEqual(src["category"], "Product Release")
        self.assertTrue(src["url"].startswith("https://github.com/"))
        self.assertTrue(src["url"].endswith("/releases.atom"))
        self.assertIn("Coding and Developer Tools", src.get("topics", []))
        self.assertIn("AI Agents", src.get("topics", []))

    def test_simon_willison_source_present_and_valid(self):
        """Simon Willison AI must be registered as a valid high-signal engineering opinion source."""
        self.assertIn("simon-willison-ai", self.sources_by_id)
        src = self.sources_by_id["simon-willison-ai"]
        self.assertEqual(src["name"], "Simon Willison – AI")
        self.assertEqual(src["publisher"], "Simon Willison's Weblog")
        self.assertEqual(src["category"], "Opinion")
        self.assertEqual(src["url"], "https://simonwillison.net/tags/ai.atom")
        self.assertIn("Coding and Developer Tools", src.get("topics", []))

    def test_all_sources_conform_to_allowed_categories_and_topics(self):
        """Every configured source must use allowed categories and topics."""
        for src in self.sources:
            cat = src.get("category")
            self.assertIn(cat, ALLOWED_CATEGORIES, f"Source {src.get('id')} has invalid category: {cat}")
            for topic in src.get("topics", []):
                self.assertIn(topic, ALLOWED_TOPICS, f"Source {src.get('id')} has invalid topic: {topic}")


class TestRssFeedOutput(unittest.TestCase):
    def test_feed_xml_structure(self):
        """dist/feed.xml should be valid, well-formed RSS 2.0 XML."""
        feed_path = REPO_ROOT / "dist" / "feed.xml"
        if not feed_path.exists():
            self.skipTest("dist/feed.xml not yet built")

        content = feed_path.read_text(encoding="utf-8")
        self.assertIn("<rss version=\"2.0\"", content)
        self.assertIn("<channel>", content)
        self.assertIn("AI Pulse — Curated AI Intelligence Feed", content)
        self.assertIn("https://swaroophegde07.github.io/ai-pulse/feed.xml", content)

        # Parse with standard XML parser to verify strictly valid XML
        root = ET.fromstring(content)
        self.assertEqual(root.tag, "rss")
        channel = root.find("channel")
        self.assertIsNotNone(channel)

        items = channel.findall("item")
        self.assertGreater(len(items), 0, "Feed must contain items")
        self.assertLessEqual(len(items), 50, "Feed must cap items at 50")

        for item in items:
            title = item.find("title")
            link = item.find("link")
            guid = item.find("guid")
            pub_date = item.find("pubDate")
            desc = item.find("description")

            self.assertIsNotNone(title)
            self.assertTrue(len(title.text or "") > 0)
            self.assertIsNotNone(link)
            self.assertTrue((link.text or "").startswith("http"))
            self.assertIsNotNone(guid)
            self.assertIsNotNone(pub_date)
            self.assertIsNotNone(desc)


class TestFreshnessLogic(unittest.TestCase):
    def test_freshness_window_calculation(self):
        """Freshness logic: articles within past 36 hours are fresh, future or older are not."""
        from datetime import datetime, timezone, timedelta

        now = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)
        now_ms = int(now.timestamp() * 1000)
        window_ms = 36 * 60 * 60 * 1000

        def is_fresh(iso_date: str) -> bool:
            dt = datetime.fromisoformat(iso_date)
            dt_ms = int(dt.timestamp() * 1000)
            return 0 < dt_ms <= now_ms and (now_ms - dt_ms) <= window_ms

        # 5 hours ago -> fresh
        five_h_ago = (now - timedelta(hours=5)).isoformat()
        self.assertTrue(is_fresh(five_h_ago))

        # 35 hours ago -> fresh
        thirty_five_h_ago = (now - timedelta(hours=35)).isoformat()
        self.assertTrue(is_fresh(thirty_five_h_ago))

        # 40 hours ago -> not fresh
        forty_h_ago = (now - timedelta(hours=40)).isoformat()
        self.assertFalse(is_fresh(forty_h_ago))

        # 1 hour in future -> not fresh
        future = (now + timedelta(hours=1)).isoformat()
        self.assertFalse(is_fresh(future))


if __name__ == "__main__":
    unittest.main()
