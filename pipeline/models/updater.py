"""
Model catalog synchronization and telemetry updater.

Safely updates `src/data/models.json` by merging latest verified pricing,
context window specifications, and telemetry from OpenRouter model metrics
while preserving curated architecture details, license types, documentation URLs,
and maintaining last-known-good fallbacks if remote data is missing or partial.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODELS_PATH = REPO_ROOT / "src" / "data" / "models.json"

REQUIRED_MODEL_FIELDS = frozenset([
    "id",
    "name",
    "provider",
    "family",
    "release_date",
    "lifecycle_status",
    "context_window",
    "input_modalities",
    "output_modalities",
    "license_type",
    "api_available",
    "consumer_available",
    "documentation_url",
    "source_url",
    "last_verified",
])

# Canonical mapping from models.json ID to OpenRouter canonical ID(s)
# in order of priority preference.
OPENROUTER_ID_MAP: dict[str, list[str]] = {
    "openai/o3-mini": ["openai/o3-mini", "openai/o3-mini-high"],
    "openai/o1": ["openai/o1", "openai/o1-preview"],
    "openai/gpt-4o": ["openai/gpt-4o", "openai/gpt-4o-2024-11-20", "openai/gpt-4o-2024-08-06"],
    "openai/gpt-4o-mini": ["openai/gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18"],
    "anthropic/claude-3-7-sonnet": ["anthropic/claude-3-7-sonnet", "anthropic/claude-3.7-sonnet"],
    "anthropic/claude-3-5-sonnet": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3.5-sonnet"],
    "anthropic/claude-3-5-haiku": ["anthropic/claude-3-5-haiku", "anthropic/claude-3.5-haiku"],
    "google/gemini-2-0-flash": ["google/gemini-2.0-flash", "google/gemini-2.0-flash-exp"],
    "google/gemini-2-0-pro-exp": ["google/gemini-2.0-pro-exp", "google/gemini-2.0-pro"],
    "deepseek/deepseek-r1": ["deepseek/deepseek-r1"],
    "deepseek/deepseek-v3": ["deepseek/deepseek-v3", "deepseek/deepseek-chat"],
    "meta/llama-3-3-70b-instruct": ["meta-llama/llama-3.3-70b-instruct"],
    "meta/llama-3-1-405b-instruct": ["meta-llama/llama-3.1-405b-instruct", "meta-llama/llama-3.1-405b"],
    "microsoft/phi-4": ["microsoft/phi-4"],
    "mistral/mistral-large-2411": ["mistralai/mistral-large-2411", "mistralai/mistral-large"],
}


def validate_model_record(record: dict[str, Any]) -> bool:
    """Validate that a model record conforms to the required models.json schema."""
    if not isinstance(record, dict):
        return False
    missing = REQUIRED_MODEL_FIELDS - record.keys()
    if missing:
        logger.warning("Model record %s missing required fields %s", record.get("id"), missing)
        return False
    if not isinstance(record.get("context_window"), (int, float)) or record["context_window"] <= 0:
        logger.warning("Model record %s has invalid context window: %r", record.get("id"), record.get("context_window"))
        return False
    if not isinstance(record.get("documentation_url"), str) or not record["documentation_url"].startswith("http"):
        logger.warning("Model record %s has invalid documentation_url: %r", record.get("id"), record.get("documentation_url"))
        return False
    return True


def load_models_catalog(models_path: Path) -> list[dict[str, Any]]:
    """Load existing models catalog safely, returning valid records."""
    if not models_path.exists():
        logger.warning("Models catalog not found at %s", models_path)
        return []
    try:
        with models_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            valid = [m for m in data if validate_model_record(m)]
            return valid
    except Exception as exc:
        logger.warning("Could not load models catalog from %s: %s", models_path, exc)
    return []


def write_models_atomic(models: list[dict[str, Any]], target_path: Path) -> None:
    """Atomically write model catalog to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(models, indent=2, ensure_ascii=False)

    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent, prefix=".models_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Atomically updated models catalog at %s (%d models)", target_path, len(models))


def sync_models_catalog(
    models_path: Path | None = None,
    openrouter_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Synchronize the models catalog with live OpenRouter telemetry specifications.

    - Updates token pricing and verified context lengths from live metrics.
    - Preserves established models and maintains schema validity.
    - Updates `last_verified` date for matched models.
    - If OpenRouter metrics are unavailable or partial, preserves previous verified data.
    """
    path = models_path or DEFAULT_MODELS_PATH
    existing_models = load_models_catalog(path)

    if not existing_models:
        logger.warning("No valid existing models loaded; preserving existing file untouched.")
        return []

    if not openrouter_metrics:
        logger.info("No OpenRouter metrics provided; keeping models catalog intact.")
        return existing_models

    # Index OpenRouter metrics by canonical ID (excluding :batch variations from primary specs)
    metrics_by_id: dict[str, dict[str, Any]] = {}
    for m in openrouter_metrics:
        cid = m.get("canonical_id", "")
        if cid and not cid.endswith(":batch"):
            metrics_by_id[cid] = m

    today_str = datetime.now(UTC).strftime("%Y-%m-%d")
    updated_models: list[dict[str, Any]] = []
    updated_count = 0

    for model in existing_models:
        model_copy = dict(model)
        mid = model_copy.get("id", "")
        candidate_or_ids = OPENROUTER_ID_MAP.get(mid, [mid])

        matched_metric = None
        for or_id in candidate_or_ids:
            if or_id in metrics_by_id:
                matched_metric = metrics_by_id[or_id]
                break

        if matched_metric:
            changed = False
            # Check context length
            metric_ctx = matched_metric.get("context_length")
            if isinstance(metric_ctx, int) and metric_ctx > 0:
                if metric_ctx != model_copy.get("context_window"):
                    model_copy["context_window"] = metric_ctx
                    changed = True

            # Check pricing
            in_price = matched_metric.get("input_price_per_1m")
            out_price = matched_metric.get("output_price_per_1m")
            if in_price is not None and out_price is not None:
                existing_pricing = model_copy.get("pricing") or {}
                if (
                    existing_pricing.get("input_per_1m") != in_price
                    or existing_pricing.get("output_per_1m") != out_price
                ):
                    model_copy["pricing"] = {
                        "input_per_1m": in_price,
                        "output_per_1m": out_price,
                        "unit": "USD",
                    }
                    changed = True

            if changed:
                model_copy["last_verified"] = today_str
                updated_count += 1

        updated_models.append(model_copy)

    if updated_models:
        write_models_atomic(updated_models, path)
        logger.info(
            "Model catalog sync complete: %d models updated with live OpenRouter telemetry.",
            updated_count,
        )

    return updated_models
