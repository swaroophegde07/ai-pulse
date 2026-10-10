"""
Central registry of RSS feed sources for AI Pulse.

Each source record contains:
  - name:      Human-readable source name shown in the UI.
  - url:       RSS or Atom feed URL (verified as plausible public feeds).
  - publisher: Company or organisation that publishes the feed.
  - category:  Default category tag applied when Gemini cannot classify.

Guidelines for adding new sources:
  1. Only add publicly accessible feeds that do not require authentication.
  2. Verify the URL is correct before committing.
  3. Choose a default category from the ALLOWED_CATEGORIES list.
  4. The pipeline will handle feed failures gracefully – a broken feed
     will not abort the entire run.
"""

from typing import TypedDict


class Source(TypedDict):
    name: str
    url: str
    publisher: str
    category: str


ALLOWED_CATEGORIES: list[str] = [
    "News",
    "Research",
    "Benchmark",
    "Product Release",
    "Opinion",
]

# ---------------------------------------------------------------------------
# RSS feed registry
# ---------------------------------------------------------------------------
# Feeds verified as plausible public RSS/Atom endpoints (October 2024).
# If a feed becomes unavailable the pipeline logs a warning and continues.
# ---------------------------------------------------------------------------

SOURCES: list[Source] = [
    {
        "name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "publisher": "Google DeepMind",
        "category": "Research",
    },
    {
        "name": "Hugging Face Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "publisher": "Hugging Face",
        "category": "News",
    },
    {
        "name": "MIT News – Artificial Intelligence",
        "url": "https://news.mit.edu/topic/artificial-intelligence2/feed",
        "publisher": "MIT",
        "category": "Research",
    },
    {
        "name": "The Gradient",
        "url": "https://thegradient.pub/rss/",
        "publisher": "The Gradient",
        "category": "Research",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
        "publisher": "VentureBeat",
        "category": "News",
    },
]
