"""
OpenRouter adapter for live model metrics, verified context lengths, and token pricing.
"""

from __future__ import annotations

import logging
from typing import Any
import requests

from pipeline.benchmarks.base import (
    ModelMetricRecord,
    SourceAdapter,
    current_iso_timestamp,
)

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"


class OpenRouterAdapter(SourceAdapter):
    """Adapter for OpenRouter model catalog and telemetry."""

    name = "OpenRouter"
    source_url = "https://openrouter.ai/models"
    enabled = True

    def __init__(self, timeout: int = 15) -> None:
        super().__init__(timeout=timeout)

    def fetch(self) -> list[Any]:
        """Fetch model metrics from OpenRouter."""
        return self.fetch_metrics()

    def fetch_metrics(self) -> list[ModelMetricRecord]:
        """Fetch and normalize pricing and context length specifications."""
        logger.info("Fetching OpenRouter models from %s", OPENROUTER_MODELS_ENDPOINT)
        try:
            resp = requests.get(
                OPENROUTER_MODELS_ENDPOINT,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as exc:
            logger.warning("OpenRouter API request failed: %s", exc)
            return []

        now_iso = current_iso_timestamp()
        metrics: list[ModelMetricRecord] = []

        # Target frontier models to index cleanly
        target_prefixes = (
            "openai/",
            "anthropic/",
            "google/",
            "deepseek/",
            "meta-llama/",
            "mistralai/",
            "microsoft/",
            "qwen/",
        )

        for item in data:
            model_id = item.get("id", "")
            if not any(model_id.startswith(p) for p in target_prefixes):
                continue

            name = item.get("name", model_id)
            context_length = item.get("context_length") or 0
            pricing = item.get("pricing") or {}

            # Pricing is per token in OpenRouter string format; convert to per 1M tokens
            prompt_str = pricing.get("prompt")
            completion_str = pricing.get("completion")

            try:
                input_per_1m = round(float(prompt_str) * 1_000_000, 4) if prompt_str else None
            except (ValueError, TypeError):
                input_per_1m = None

            try:
                output_per_1m = round(float(completion_str) * 1_000_000, 4) if completion_str else None
            except (ValueError, TypeError):
                output_per_1m = None

            provider_part = model_id.split("/")[0].capitalize()
            provider_map = {
                "Openai": "OpenAI",
                "Anthropic": "Anthropic",
                "Google": "Google DeepMind",
                "Deepseek": "DeepSeek",
                "Meta-llama": "Meta AI",
                "Mistralai": "Mistral AI",
                "Microsoft": "Microsoft",
                "Qwen": "Alibaba Cloud",
            }
            provider = provider_map.get(provider_part, provider_part)

            metrics.append(
                {
                    "model_name": name,
                    "canonical_id": model_id,
                    "provider": provider,
                    "context_length": context_length,
                    "input_price_per_1m": input_per_1m,
                    "output_price_per_1m": output_per_1m,
                    "currency": "USD",
                    "source": "OpenRouter",
                    "source_url": f"https://openrouter.ai/{model_id}",
                    "retrieved_at": now_iso,
                }
            )

        logger.info("Successfully normalized %d model metrics from OpenRouter.", len(metrics))
        return metrics

    def get_provenance_notice(self) -> dict[str, str]:
        return {
            "source": self.name,
            "url": self.source_url,
            "status": "Active (Public Catalog)",
            "description": "Public model pricing and context length specifications from OpenRouter catalog API.",
            "attribution": "Data provided by OpenRouter (openrouter.ai). Model usage telemetry reflects OpenRouter volume only and is not an objective quality ranking.",
        }
