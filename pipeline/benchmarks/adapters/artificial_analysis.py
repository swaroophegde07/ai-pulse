"""
Artificial Analysis adapter for independent intelligence index, latency, and throughput metrics.
Requires commercial enterprise API credentials (ARTIFICIAL_ANALYSIS_API_KEY).
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

ARTIFICIAL_ANALYSIS_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models"


class ArtificialAnalysisAdapter(SourceAdapter):
    """Adapter for Artificial Analysis Intelligence Index and performance benchmarks."""

    name = "Artificial Analysis"
    source_url = "https://artificialanalysis.ai/data-api"

    def __init__(self, timeout: int = 15) -> None:
        super().__init__(timeout=timeout)
        self.api_key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY", "").strip()
        self.enabled = bool(self.api_key)

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch Artificial Analysis intelligence index evaluations if authenticated."""
        if not self.enabled:
            logger.info(
                "Artificial Analysis adapter is disabled: ARTIFICIAL_ANALYSIS_API_KEY is not configured. "
                "The source requires a commercial data license and API key. "
                "Set ARTIFICIAL_ANALYSIS_API_KEY to activate."
            )
            return []

        logger.info("Fetching Artificial Analysis models with provided API key.")
        now_iso = current_iso_timestamp()

        try:
            resp = requests.get(
                ARTIFICIAL_ANALYSIS_MODELS_URL,
                timeout=self.timeout,
                headers={
                    "x-api-key": self.api_key,
                    "User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            # Redact API key from error message if present
            safe_msg = str(exc)
            if self.api_key and self.api_key in safe_msg:
                safe_msg = safe_msg.replace(self.api_key, "[REDACTED]")
            logger.warning("Artificial Analysis API request failed: %s", safe_msg)
            return []

        records: list[BenchmarkRecord] = []
        models_list = data if isinstance(data, list) else data.get("models", [])

        for m in models_list:
            slug = m.get("slug") or m.get("id")
            name = m.get("name") or slug
            creator = m.get("model_creator", {}).get("name") or "Unknown"
            evals = m.get("evaluations", {})
            intel_index = evals.get("artificial_analysis_intelligence_index")

            if intel_index is not None:
                try:
                    score = float(intel_index)
                    rec: BenchmarkRecord = {
                        "model_name": name,
                        "canonical_id": str(slug).lower(),
                        "provider": creator,
                        "benchmark_name": "Artificial Analysis Intelligence Index",
                        "category": "Composite",
                        "score": score,
                        "score_unit": "Index / 100",
                        "evaluation_date": m.get("evaluated_date"),
                        "source": "Artificial Analysis",
                        "source_url": "https://artificialanalysis.ai/",
                        "provenance": "Independent Evaluation",
                        "retrieved_at": now_iso,
                        "notes": "Composite quality benchmark independently tested by Artificial Analysis.",
                    }
                    if validate_benchmark_record(rec):
                        records.append(rec)
                except (ValueError, TypeError):
                    pass

        logger.info("Parsed %d records from Artificial Analysis.", len(records))
        return records

    def get_provenance_notice(self) -> dict[str, str]:
        status_label = "Active (Commercial API)" if self.enabled else "Disabled (API Key Required)"
        return {
            "source": self.name,
            "url": self.source_url,
            "status": status_label,
            "description": "Independent intelligence index, output speed, and latency evaluations.",
            "attribution": "Metrics from Artificial Analysis (artificialanalysis.ai). Commercial API plan and agreement required for production display.",
        }
