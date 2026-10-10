"""
BenchLM adapter for benchmark rankings and model score distributions.
Requires explicit authorization due to CC BY-NC 4.0 license restrictions.
"""

from __future__ import annotations

import logging
import os
import requests

from pipeline.benchmarks.base import (
    BenchmarkRecord,
    SourceAdapter,
    current_iso_timestamp,
    validate_benchmark_record,
)

logger = logging.getLogger(__name__)

BENCHLM_LEADERBOARD_URL = "https://benchlm.ai/data/leaderboard.json"


class BenchLMAdapter(SourceAdapter):
    """Adapter for BenchLM open dataset downloads."""

    name = "BenchLM"
    source_url = "https://benchlm.ai/data"

    def __init__(self, timeout: int = 15) -> None:
        super().__init__(timeout=timeout)
        # Governed by ENABLE_BENCHLM environment variable due to CC BY-NC 4.0 license
        self.enabled = os.environ.get("ENABLE_BENCHLM", "false").lower() in ("true", "1", "yes")

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch BenchLM leaderboard data if explicitly enabled."""
        if not self.enabled:
            logger.info(
                "BenchLM adapter is disabled by default. Source data is licensed under "
                "Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). Set "
                "ENABLE_BENCHLM=true once authorization and downstream terms are validated."
            )
            return []

        logger.info("Fetching BenchLM leaderboard from %s", BENCHLM_LEADERBOARD_URL)
        now_iso = current_iso_timestamp()

        try:
            resp = requests.get(
                BENCHLM_LEADERBOARD_URL,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("BenchLM download request failed: %s", exc)
            return []

        records: list[BenchmarkRecord] = []
        rows = data.get("rows", [])

        for item in rows[:30]:
            name = item.get("name") or item.get("model")
            score = item.get("overallScore") or item.get("score")
            if not name or score is None:
                continue

            try:
                score_flt = float(score)
            except (ValueError, TypeError):
                continue

            rec: BenchmarkRecord = {
                "model_name": str(name),
                "canonical_id": str(item.get("slug") or name).lower(),
                "provider": str(item.get("provider") or "Unknown"),
                "benchmark_name": "BenchLM Composite Rating",
                "category": "Composite",
                "score": score_flt,
                "score_unit": "Index / 100",
                "evaluation_date": data.get("dateModified"),
                "source": "BenchLM",
                "source_url": "https://benchlm.ai/data",
                "provenance": "Aggregated by Source",
                "retrieved_at": now_iso,
                "notes": "BenchLM multi-benchmark composite index. Licensed under CC BY-NC 4.0.",
            }
            if validate_benchmark_record(rec):
                records.append(rec)

        logger.info("Parsed %d records from BenchLM.", len(records))
        return records

    def get_provenance_notice(self) -> dict[str, str]:
        status_label = "Active (Opt-in)" if self.enabled else "Disabled (CC BY-NC 4.0 license review required)"
        return {
            "source": self.name,
            "url": self.source_url,
            "status": status_label,
            "description": "Website JSON exports with model metadata and multi-benchmark ratings.",
            "attribution": "Dataset provided by BenchLM.ai under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). Requires explicit opt-in via ENABLE_BENCHLM=true.",
        }
