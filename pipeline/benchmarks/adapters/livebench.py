"""
LiveBench adapter for contamination-free LLM benchmark results across reasoning, coding, and math.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any
import requests

from pipeline.benchmarks.base import (
    BenchmarkRecord,
    SourceAdapter,
    current_iso_timestamp,
    validate_benchmark_record,
)

logger = logging.getLogger(__name__)

# Known public release URLs for LiveBench
LIVEBENCH_RELEASE_DATE = "2026-01-08"
LIVEBENCH_TABLE_URL = f"https://livebench.ai/table_{LIVEBENCH_RELEASE_DATE.replace('-', '_')}.csv"
LIVEBENCH_CATEGORIES_URL = f"https://livebench.ai/categories_{LIVEBENCH_RELEASE_DATE.replace('-', '_')}.json"


class LiveBenchAdapter(SourceAdapter):
    """Adapter for LiveBench monthly contamination-free benchmark suite."""

    name = "LiveBench"
    source_url = "https://livebench.ai/"
    enabled = True

    def __init__(self, timeout: int = 15) -> None:
        super().__init__(timeout=timeout)

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch and parse LiveBench evaluation results."""
        logger.info("Fetching LiveBench data from %s", LIVEBENCH_TABLE_URL)
        now_iso = current_iso_timestamp()

        try:
            table_resp = requests.get(
                LIVEBENCH_TABLE_URL,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            table_resp.raise_for_status()

            cat_resp = requests.get(
                LIVEBENCH_CATEGORIES_URL,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            cat_resp.raise_for_status()
            categories: dict[str, list[str]] = cat_resp.json()
        except Exception as exc:
            logger.warning("LiveBench remote fetch failed: %s", exc)
            return []

        csv_text = table_resp.text
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            logger.warning("LiveBench CSV returned 0 rows.")
            return []

        records: list[BenchmarkRecord] = []

        # Target frontier models to display clearly
        target_models_map: dict[str, tuple[str, str]] = {
            "o1": ("OpenAI o1", "OpenAI"),
            "o3-mini": ("OpenAI o3-mini", "OpenAI"),
            "gpt-4o": ("GPT-4o", "OpenAI"),
            "gpt-4o-mini": ("GPT-4o mini", "OpenAI"),
            "claude-3-5-sonnet": ("Claude 3.5 Sonnet", "Anthropic"),
            "claude-3-7-sonnet": ("Claude 3.7 Sonnet", "Anthropic"),
            "claude-3-5-haiku": ("Claude 3.5 Haiku", "Anthropic"),
            "gemini-2.0-flash": ("Gemini 2.0 Flash", "Google DeepMind"),
            "gemini-2.0-pro": ("Gemini 2.0 Pro", "Google DeepMind"),
            "deepseek-r1": ("DeepSeek-R1", "DeepSeek"),
            "deepseek-v3": ("DeepSeek-V3", "DeepSeek"),
            "llama-3.3-70b-instruct": ("Llama 3.3 70B Instruct", "Meta AI"),
            "llama-3.1-405b-instruct": ("Llama 3.1 405B Instruct", "Meta AI"),
            "qwen-2.5-72b-instruct": ("Qwen 2.5 72B Instruct", "Alibaba Cloud"),
            "phi-4": ("Phi-4", "Microsoft"),
            "mistral-large-2411": ("Mistral Large 2", "Mistral AI"),
        }

        # Category mapping into allowed standard categories
        allowed_cat_map = {
            "Reasoning": "Reasoning",
            "Coding": "Coding",
            "Agentic Coding": "Agentic Coding",
            "Mathematics": "Mathematics",
            "Data Analysis": "Data Analysis",
            "Language": "Language",
            "Instruction Following": "Instruction Following",
        }

        for row in rows:
            raw_model = row.get("model", "").strip().lower()
            matched_info = None

            for key, (disp_name, prov) in target_models_map.items():
                if key in raw_model or raw_model.startswith(key):
                    matched_info = (key, disp_name, prov)
                    break

            if not matched_info:
                continue

            canon_key, display_name, provider = matched_info

            # 1. Compute category-level averages
            all_scores: list[float] = []

            for cat_name, task_list in categories.items():
                mapped_cat = allowed_cat_map.get(cat_name)
                if not mapped_cat:
                    continue

                cat_scores: list[float] = []
                for task in task_list:
                    val_str = row.get(task, "")
                    if val_str:
                        try:
                            score_val = float(val_str)
                            cat_scores.append(score_val)
                            all_scores.append(score_val)
                        except (ValueError, TypeError):
                            pass

                if cat_scores:
                    avg_score = round(sum(cat_scores) / len(cat_scores), 2)
                    rec: BenchmarkRecord = {
                        "model_name": display_name,
                        "canonical_id": canon_key,
                        "provider": provider,
                        "benchmark_name": f"LiveBench ({cat_name})",
                        "category": mapped_cat,
                        "score": avg_score,
                        "score_unit": "Percentage",
                        "evaluation_date": LIVEBENCH_RELEASE_DATE,
                        "source": "LiveBench",
                        "source_url": "https://livebench.ai/",
                        "provenance": "Independent Evaluation",
                        "retrieved_at": now_iso,
                        "notes": f"LiveBench {LIVEBENCH_RELEASE_DATE} release category average across {len(cat_scores)} tasks.",
                    }
                    if validate_benchmark_record(rec):
                        records.append(rec)

            # 2. Overall composite score
            if all_scores:
                overall_score = round(sum(all_scores) / len(all_scores), 2)
                overall_rec: BenchmarkRecord = {
                    "model_name": display_name,
                    "canonical_id": canon_key,
                    "provider": provider,
                    "benchmark_name": "LiveBench Overall Average",
                    "category": "Composite",
                    "score": overall_score,
                    "score_unit": "Percentage",
                    "evaluation_date": LIVEBENCH_RELEASE_DATE,
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": now_iso,
                    "notes": f"LiveBench overall average across {len(all_scores)} verified subtasks.",
                }
                if validate_benchmark_record(overall_rec):
                    records.append(overall_rec)

        logger.info("Successfully parsed %d benchmark records from LiveBench.", len(records))
        return records

    def get_provenance_notice(self) -> dict[str, str]:
        return {
            "source": self.name,
            "url": self.source_url,
            "status": "Active (Monthly Release)",
            "description": "Contamination-free evaluations with monthly updated problems across reasoning, coding, and mathematics.",
            "attribution": "LiveBench is an open-source evaluation benchmark created by researchers at Abacus.ai, NYU, and UC Berkeley. Released under Apache-2.0.",
        }
