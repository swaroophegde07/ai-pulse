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

import json
from pathlib import Path
from typing import TypedDict


class Source(TypedDict, total=False):
    id: str
    name: str
    url: str
    publisher: str
    category: str
    website: str
    source_type: str
    topics: list[str]
    access_method: str
    attribution: str
    notes: str


ALLOWED_CATEGORIES: list[str] = [
    "News",
    "Research",
    "Benchmark",
    "Product Release",
    "Opinion",
]

ALLOWED_TOPICS: list[str] = [
    "Models and Releases",
    "AI Agents",
    "Coding and Developer Tools",
    "Research and Breakthroughs",
    "Benchmarks and Evaluations",
    "Safety and Security",
    "Policy and Regulation",
    "Business and Industry",
    "Open Source",
    "Infrastructure and Hardware",
    "Applications and Products",
]

# Path to the shared sources.json
SOURCES_JSON_PATH = Path(__file__).parent.parent / "src" / "data" / "sources.json"

_DEFAULT_SOURCES: list[Source] = [
    {
        "name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
        "publisher": "OpenAI",
        "category": "Product Release",
    },
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
        "name": "GitHub AI & Copilot",
        "url": "https://github.blog/category/ai-and-ml/feed/",
        "publisher": "GitHub",
        "category": "Product Release",
    },
    {
        "name": "VS Code Updates",
        "url": "https://code.visualstudio.com/feed.xml",
        "publisher": "Microsoft VS Code",
        "category": "Product Release",
    },
    {
        "name": "Microsoft Research AI",
        "url": "https://www.microsoft.com/en-us/research/feed/",
        "publisher": "Microsoft Research",
        "category": "Research",
    },
    {
        "name": "The Gradient",
        "url": "https://thegradient.pub/rss/",
        "publisher": "The Gradient",
        "category": "Research",
    },
    {
        "name": "The Verge – AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "publisher": "The Verge",
        "category": "News",
    },
    {
        "name": "Ars Technica – AI",
        "url": "https://arstechnica.com/tag/ai/feed/",
        "publisher": "Ars Technica",
        "category": "News",
    },
]


def load_sources(file_path: Path | None = None) -> list[Source]:
    """Load configured RSS feed sources from JSON file.

    Falls back to default sources if the file cannot be read.
    """
    path = file_path or SOURCES_JSON_PATH
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return data
        except Exception:
            pass
    return list(_DEFAULT_SOURCES)


SOURCES: list[Source] = load_sources()
