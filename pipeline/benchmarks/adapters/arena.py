"""
Arena.ai / LMSYS Chatbot Arena adapter for crowdsourced human blind preference and coding leaderboards.
Retrieves Bradley-Terry Elo ratings from the official Hugging Face dataset server mirror.
"""

from __future__ import annotations

import logging
import re
from typing import Any
import requests

from pipeline.benchmarks.base import (
    BenchmarkRecord,
    SourceAdapter,
    current_iso_timestamp,
    validate_benchmark_record,
)
from pipeline.models.identity import resolve_model_identity

logger = logging.getLogger(__name__)

HF_DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "lmarena-ai/leaderboard-dataset"

# Provider organization mapping from Arena's org names
PROVIDER_ORG_MAP: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "deepseek": "DeepSeek",
    "meta": "Meta",
    "mistral": "Mistral AI",
    "xai": "xAI",
    "alibaba": "Alibaba",
    "moonshot": "Moonshot",
    "zai": "Zhipu AI",
    "zhipu": "Zhipu AI",
    "xiaomi": "Xiaomi",
    "baidu": "Baidu",
    "amazon": "Amazon",
    "microsoft": "Microsoft",
    "cohere": "Cohere",
    "nvidia": "NVIDIA",
    "tencent": "Tencent",
    "bytedance": "ByteDance",
    "01-ai": "01.AI",
    "01_ai": "01.AI",
    "reflection": "Reflection",
    "thinky": "Thinky",
    "minimax": "MiniMax",
}

CONFIGS_TO_FETCH = [
    {
        "config": "text",
        "benchmark_name": "Chatbot Arena (Text)",
        "category": "Human Preference",
    },
    {
        "config": "webdev",
        "benchmark_name": "Chatbot Arena (Coding)",
        "category": "Coding",
    },
]


def slugify_model_id(org: str, raw_name: str) -> str:
    """Create a clean canonical identifier slug from organization and raw model name."""
    clean_org = re.sub(r"[^a-zA-Z0-9_-]", "", org.lower()) or "community"
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "-", raw_name.lower()).strip("-")
    return f"{clean_org}/{clean_name}"


class ArenaAdapter(SourceAdapter):
    """Adapter for Arena.ai (LMSYS Chatbot Arena) crowdsourced Elo benchmarks."""

    name = "Chatbot Arena"
    source_url = "https://arena.ai/leaderboard"
    enabled = True

    def __init__(self, timeout: int = 15, max_records_per_config: int = 100) -> None:
        super().__init__(timeout=timeout)
        self.max_records_per_config = max_records_per_config
        self.active_release: str | None = None
        self.is_stale: bool = False

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch and parse Arena Elo ratings from official Hugging Face dataset mirror."""
        records: list[BenchmarkRecord] = []
        now_iso = current_iso_timestamp()

        for target in CONFIGS_TO_FETCH:
            cfg = target["config"]
            bench_name = target["benchmark_name"]
            cat_name = target["category"]

            try:
                url = f"{HF_DATASET_ROWS_URL}?dataset={DATASET_ID}&config={cfg}&split=latest&offset=0&limit={self.max_records_per_config}"
                logger.info("Fetching Arena leaderboard from %s", url)
                resp = requests.get(
                    url,
                    timeout=self.timeout,
                    headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                )
                if not resp.ok:
                    logger.warning("Failed to fetch Arena config %s: HTTP %s", cfg, resp.status_code)
                    continue

                payload = resp.json()
                rows = [item["row"] for item in payload.get("rows", []) if "row" in item]
                if not rows:
                    logger.warning("Empty rows returned for Arena config %s", cfg)
                    continue
                rows = rows[:self.max_records_per_config]

                for r in rows:
                    raw_name = str(r.get("model_name", "")).strip()
                    if not raw_name:
                        continue

                    raw_org = str(r.get("organization", "")).strip().lower()
                    rating_val = r.get("rating")
                    if rating_val is None:
                        continue
                    try:
                        score = round(float(rating_val), 1)
                    except (ValueError, TypeError):
                        continue

                    pub_date = r.get("leaderboard_publish_date") or None
                    if pub_date and not self.active_release:
                        self.active_release = str(pub_date)

                    rank = r.get("rank")
                    votes = r.get("vote_count")
                    r_lower = r.get("rating_lower")
                    r_upper = r.get("rating_upper")

                    # Resolve model identity
                    match = resolve_model_identity(raw_name, context_source="arena")
                    canonical_id = match.matched_id or slugify_model_id(raw_org, raw_name)
                    display_name = match.canonical_name or raw_name
                    provider = match.provider or PROVIDER_ORG_MAP.get(raw_org, raw_org.title() or "Independent")

                    # Extract configuration hint if present in name (e.g. "(High)" or "(Thinking)")
                    config_match = re.search(r"\(([^)]+)\)", raw_name)
                    configuration = config_match.group(1).strip() if config_match else None

                    notes_parts = ["Crowdsourced human preference Elo rating from blind pairwise battles."]
                    if rank is not None:
                        notes_parts.append(f"Rank #{int(rank)}.")
                    if r_lower is not None and r_upper is not None:
                        notes_parts.append(f"95% CI: [{r_lower:.1f}, {r_upper:.1f}].")
                    if votes:
                        notes_parts.append(f"{int(votes):,} votes.")
                    notes_parts.append("Model evaluation date not reported upstream by source.")

                    rec: BenchmarkRecord = {
                        "model_name": display_name,
                        "canonical_id": canonical_id,
                        "internal_model_id": match.matched_id,
                        "match_confidence": match.confidence if match.matched_id else 0.85,
                        "provider": provider,
                        "benchmark_name": bench_name,
                        "category": cat_name,
                        "score": score,
                        "score_unit": "Elo",
                        "evaluation_date": None,
                        "benchmark_release_date": str(pub_date) if pub_date else self.active_release,
                        "benchmark_version": "Arena",
                        "source": self.name,
                        "source_url": self.source_url,
                        "provenance": "Independent Evaluation",
                        "retrieved_at": now_iso,
                        "notes": " ".join(notes_parts),
                        "configuration": configuration,
                    }

                    if validate_benchmark_record(rec):
                        records.append(rec)

            except Exception as exc:
                logger.warning("Error fetching Arena data for %s: %s", cfg, exc)

        if not records:
            logger.warning("ArenaAdapter could not retrieve any benchmark records.")
            self.is_stale = True

        return records

    def get_provenance_notice(self) -> dict[str, str]:
        rel = f" ({self.active_release} Release)" if self.active_release else ""
        status_desc = f"Active{rel}"
        if self.is_stale:
            status_desc += " [Fallback/Stale snapshot]"
        return {
            "source": self.name,
            "url": self.source_url,
            "status": status_desc,
            "description": (
                "Crowdsourced blind head-to-head pairwise comparisons evaluated with Bradley-Terry / Elo rating system "
                "across general conversation, coding, and hard prompts."
            ),
            "attribution": "Chatbot Arena is an open evaluation platform founded by LMSYS Org and UC Berkeley SkyLab, hosted at arena.ai. Data distributed under Apache-2.0 / CC-BY.",
            "release_date": self.active_release or "Continuous",
        }
