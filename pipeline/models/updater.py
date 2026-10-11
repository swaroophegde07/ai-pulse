"""
Model catalog synchronization and telemetry updater.

Safely refreshes and expands `src/data/models.json` by:
1. Merging latest verified pricing and context length specifications from OpenRouter
   telemetry for existing curated frontier models while preserving verified release dates,
   licensing terms, and documentation links.
2. Dynamically discovering and appending suitable newly released frontier models,
   coding specialists, and multimodal models from top AI labs.
3. Classifying models by capability (Frontier Reasoning, Coding, Specialist, etc.).
4. Recording transparent per-field data provenance without falsely claiming unverified
   attributes (such as exact lab release date or consumer availability) are verified.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.models.identity import (
    export_identity_registry,
    get_featured_frontier_ids,
    resolve_model_identity,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODELS_PATH = REPO_ROOT / "src" / "data" / "models.json"

REQUIRED_MODEL_FIELDS = frozenset([
    "id",
    "name",
    "provider",
    "family",
    "lifecycle_status",
    "context_window",
    "license_type",
    "api_available",
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

# Verified or estimated release dates for catalog models based on provider announcements
# and benchmark release checkpoints.
KNOWN_MODEL_RELEASE_DATES: dict[str, str] = {
    # 2026 Q3 / Q2 Frontier Flagships & Updates
    "qwen/qwen3.8-max-0902": "2026-09-02",
    "deepseek/deepseek-v4-pro-0813": "2026-08-13",
    "deepseek/deepseek-v4-flash-0731": "2026-07-31",
    "openai/gpt-6.1-sol-pro": "2026-06-20",
    "openai/gpt-6.1-sol": "2026-06-20",
    "anthropic/claude-opus-5.5": "2026-06-18",
    "anthropic/claude-haiku-5.5": "2026-06-12",
    "deepseek/deepseek-v4.1-flash": "2026-06-12",
    "anthropic/claude-sonnet-5.5": "2026-06-10",
    "qwen/qwen3.8-max-prime": "2026-06-08",
    "google/gemini-3.8-flash": "2026-06-05",
    "qwen/qwen3.8-omni-flash": "2026-06-05",
    "openai/gpt-6-astra-pro": "2026-06-01",
    "openai/gpt-6-astra": "2026-06-01",
    "qwen/qwen3.8-flash": "2026-05-28",
    "google/gemini-3.7-flash": "2026-05-22",
    "openai/gpt-6-sol-pro": "2026-05-20",
    "openai/gpt-6-sol": "2026-05-20",
    "qwen/qwen3.8-27b": "2026-05-20",
    "mistralai/mistral-large-4-0": "2026-05-18",
    "openai/gpt-6-luna-pro": "2026-05-15",
    "openai/gpt-6-luna": "2026-05-15",
    "qwen/qwen3.8-2.4t-a95b": "2026-05-15",
    "google/gemini-3.6-flash": "2026-05-10",
    "google/gemini-3.5-flash": "2026-05-01",
    "google/gemini-3.5-flash-lite": "2026-05-01",
    "google/gemini-nano-banana-2.1": "2026-05-01",
    "deepseek/deepseek-v4-pro": "2026-04-23",
    "deepseek/deepseek-v4-flash": "2026-04-23",
    "deepseek/deepseek-v4-flash-vision-exp": "2026-04-20",
    "anthropic/claude-opus-5": "2026-04-20",
    "openai/gpt-5.6-sol-pro": "2026-04-18",
    "openai/gpt-5.6-sol": "2026-04-18",
    "openai/gpt-5.6-luna-pro": "2026-04-15",
    "openai/gpt-5.6-luna": "2026-04-15",
    "anthropic/claude-sonnet-5": "2026-04-15",
    "qwen/qwen3.7-max": "2026-04-10",
    "qwen/qwen3.7-plus": "2026-04-05",
    "google/gemini-3.1-pro-preview": "2026-03-18",
    "google/gemini-3.1-pro-preview-customtools": "2026-03-18",
    "openai/gpt-5.5-pro": "2026-03-15",
    "openai/gpt-5.5": "2026-03-15",
    "google/gemini-3.1-flash-lite": "2026-03-10",
    "google/gemini-3.1-flash-lite-preview": "2026-03-10",
    "google/gemini-3.1-flash-lite-image": "2026-03-10",
    "google/gemini-3.1-flash-image": "2026-03-10",
    "google/gemini-3.1-flash-image-preview": "2026-03-10",
    "anthropic/claude-opus-4.8": "2026-03-05",
    "google/gemini-3-pro-image": "2026-02-28",
    "google/gemini-3-pro-image-preview": "2026-02-28",
    "openai/gpt-5.4-image-2": "2026-02-25",
    "qwen/qwen3.7-flash": "2026-03-25",
    "openai/gpt-5.4-pro": "2026-02-20",
    "openai/gpt-5.4": "2026-02-20",
    "openai/gpt-5.4-mini": "2026-02-20",
    "openai/gpt-5.4-nano": "2026-02-20",
    "google/gemini-3-flash-preview": "2026-02-15",
    "anthropic/claude-opus-4.7": "2026-02-12",
    "openai/gpt-5.3-codex": "2026-01-25",
    "anthropic/claude-opus-4.6": "2026-01-18",
    # 2025 releases
    "mistralai/devstral-2512": "2025-12-15",
    "openai/gpt-5.2-codex": "2025-12-10",
    "openai/gpt-5.2-chat": "2025-12-10",
    "openai/gpt-5.2-pro": "2025-12-10",
    "openai/gpt-5.2": "2025-12-10",
    "openai/gpt-5.1-codex-max": "2025-11-15",
    "openai/gpt-5.1-codex": "2025-11-15",
    "openai/gpt-5.1-codex-mini": "2025-11-15",
    "openai/gpt-5.1": "2025-11-15",
    "anthropic/claude-opus-4.5": "2025-11-01",
    "anthropic/claude-opus-4.1": "2025-08-15",
    "google/gemini-2.5-pro": "2025-06-15",
    "google/gemini-2.5-pro-preview": "2025-06-05",
    "google/gemini-2.5-flash": "2025-06-10",
    "google/gemini-2.5-flash-lite": "2025-06-10",
    "google/gemini-2.5-flash-image": "2025-06-10",
}

PRIMARY_LABS = frozenset([
    "OpenAI",
    "Anthropic",
    "Google DeepMind",
    "DeepSeek",
    "Meta AI",
    "Mistral AI",
    "Alibaba Cloud",
    "Microsoft",
    "xAI",
])


def classify_model_category(canonical_id: str, model_name: str) -> str:
    """Classify model into standard categories."""
    cid_l = canonical_id.lower()
    name_l = model_name.lower()

    if any(k in cid_l or k in name_l for k in ("embed", "bge", "text-embedding", "clip")):
        return "Embedding"
    if any(k in cid_l or k in name_l for k in ("image", "flux", "dall-e", "midjourney", "stable-diffusion")):
        return "Image Generation"
    if any(k in cid_l or k in name_l for k in ("audio", "whisper", "tts", "voice")):
        return "Audio"
    if any(k in cid_l or k in name_l for k in ("coder", "coding", "codex", "devstral")):
        return "Coding"
    if any(
        k in cid_l or k in name_l
        for k in (
            "r1", "o1", "o3", "o4", "reasoning", "thinking", "opus", "sonnet-5",
            "haiku-5", "sol", "luna", "pro", "max", "flash", "gpt-6", "gpt-5", "v4"
        )
    ):
        return "Frontier Reasoning & Chat"
    return "General Chat & Language"


def infer_model_family(canonical_id: str, model_name: str, provider: str) -> str:
    """Infer model family name cleanly."""
    cid = canonical_id.lower()
    if "claude" in cid:
        if "opus" in cid:
            return "Claude Opus"
        if "sonnet" in cid:
            return "Claude Sonnet"
        if "haiku" in cid:
            return "Claude Haiku"
        return "Claude"
    if "gpt-6" in cid:
        return "GPT-6"
    if "gpt-5" in cid:
        return "GPT-5"
    if "gpt-4" in cid:
        return "GPT-4"
    if "gemini-3" in cid:
        return "Gemini 3"
    if "gemini-2" in cid:
        return "Gemini 2"
    if "deepseek-v4" in cid:
        return "DeepSeek-V4"
    if "deepseek-v3" in cid or "deepseek-r1" in cid:
        return "DeepSeek-V3"
    if "qwen3" in cid:
        return "Qwen 3"
    if "qwen2" in cid or "qwen-2" in cid:
        return "Qwen 2.5"
    if "mistral" in cid:
        return "Mistral"
    if "llama" in cid:
        return "Llama"
    if "phi" in cid:
        return "Phi"
    return provider


def should_include_new_candidate(metric: dict[str, Any]) -> bool:
    """Determine whether an OpenRouter model metric represents a suitable primary model."""
    cid = metric.get("canonical_id", "").lower()
    if not cid:
        return False

    # Exclude batch, free, nitro, duplicate hosting variants
    if any(cid.endswith(s) for s in (":batch", ":free", ":nitro", ":extended")):
        return False
    if any(k in cid for k in ("test", "debug", "internal", "eval")):
        return False

    # Only include primary labs
    provider = metric.get("provider", "")
    if provider not in PRIMARY_LABS:
        return False

    # Filter for notable frontier releases, coding models, or key lab releases
    frontier_signals = (
        "sol", "luna", "opus", "sonnet-5", "haiku-5", "gpt-6", "gpt-5.5", "gpt-5.4",
        "gpt-5.2", "gpt-5.1", "gemini-3", "gemini-2.5", "deepseek-v4", "mistral-large-4",
        "qwen3.8", "qwen3.7", "codex", "devstral", "claude-4", "nano-banana"
    )
    return any(sig in cid for sig in frontier_signals)


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
    """Synchronize and refresh the models catalog with live OpenRouter telemetry.

    1. Updates token pricing and context lengths for existing curated models.
    2. Discovers and appends newly released frontier models and coding specialists from top labs.
    3. Categorizes models and transparently records field-level verification sources.
    4. If telemetry is unavailable, preserves the existing verified catalog intact.
    """
    path = models_path or DEFAULT_MODELS_PATH
    existing_models = load_models_catalog(path)

    if not existing_models:
        logger.warning("No valid existing models loaded; preserving existing file untouched.")
        return []

    featured_ids = get_featured_frontier_ids()

    if not openrouter_metrics:
        logger.info("No OpenRouter metrics provided; ensuring is_featured_frontier flags, release dates, and order are up to date.")
        for m in existing_models:
            mid = m.get("id", "")
            id_match = resolve_model_identity(mid, context_source="CatalogSync")
            expected_flag = mid in featured_ids or (id_match.matched_id in featured_ids if id_match.matched_id else False)
            if m.get("is_featured_frontier") != expected_flag:
                m["is_featured_frontier"] = expected_flag
            if mid in KNOWN_MODEL_RELEASE_DATES and (m.get("release_date") in (None, "Unknown") or mid.startswith("anthropic/claude-opus-5") or mid.startswith("openai/gpt-6")):
                m["release_date"] = KNOWN_MODEL_RELEASE_DATES[mid]

        # Sort chronologically by release date descending so newest models appear first
        existing_models.sort(
            key=lambda m: (
                m.get("release_date") or "",
                1 if m.get("is_featured_frontier") else 0,
                m.get("name") or "",
            ),
            reverse=True,
        )
        write_models_atomic(existing_models, path)
        export_identity_registry()
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
    existing_seen_ids: set[str] = set()

    # Step 1: Update existing curated records
    for model in existing_models:
        model_copy = dict(model)
        mid = model_copy.get("id", "")
        existing_seen_ids.add(mid)

        # Set verified featured frontier flag
        id_match = resolve_model_identity(mid, context_source="CatalogSync")
        is_featured = (
            mid in featured_ids
            or (id_match.matched_id in featured_ids if id_match.matched_id else False)
            or bool(model_copy.get("is_featured_frontier", False))
        )
        model_copy["is_featured_frontier"] = is_featured

        candidate_or_ids = OPENROUTER_ID_MAP.get(mid, [mid])

        matched_metric = None
        for or_id in candidate_or_ids:
            if or_id in metrics_by_id:
                matched_metric = metrics_by_id[or_id]
                existing_seen_ids.add(or_id)
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
                ver_sources = dict(model_copy.get("verification_sources") or {})
                ver_sources["telemetry"] = f"OpenRouter ({today_str})"
                ver_sources["pricing"] = f"OpenRouter ({today_str})"
                model_copy["verification_sources"] = ver_sources
                updated_count += 1

        # Populate verified release date if missing or known
        if mid in KNOWN_MODEL_RELEASE_DATES and (model_copy.get("release_date") in (None, "Unknown") or mid.startswith("anthropic/claude-opus-5") or mid.startswith("openai/gpt-6")):
            model_copy["release_date"] = KNOWN_MODEL_RELEASE_DATES[mid]

        # Ensure category is populated
        if "category" not in model_copy:
            model_copy["category"] = classify_model_category(mid, model_copy.get("name", ""))

        updated_models.append(model_copy)

    # Step 2: Append newly discovered frontier and specialist models from OpenRouter
    appended_count = 0
    for metric in openrouter_metrics:
        cid = metric.get("canonical_id", "")
        if cid in existing_seen_ids:
            continue
        if not should_include_new_candidate(metric):
            continue

        name = metric.get("model_name", cid)
        provider = metric.get("provider", "Independent / Open")
        ctx_len = metric.get("context_length") or 128000
        in_p = metric.get("input_price_per_1m")
        out_p = metric.get("output_price_per_1m")
        pricing = (
            {"input_per_1m": in_p, "output_per_1m": out_p, "unit": "USD"}
            if in_p is not None and out_p is not None
            else None
        )

        category = classify_model_category(cid, name)
        family = infer_model_family(cid, name, provider)

        id_match = resolve_model_identity(cid, context_source="CatalogDiscover")
        is_featured = bool(id_match.matched_id and id_match.matched_id in featured_ids)

        # License inferred conservatively
        is_open = provider in ("Meta AI", "Mistral AI", "DeepSeek", "Alibaba Cloud")
        license_type = "Open Weights" if is_open else "Proprietary API"

        rel_date = KNOWN_MODEL_RELEASE_DATES.get(cid, "Unknown")

        new_record: dict[str, Any] = {
            "id": cid,
            "name": name,
            "provider": provider,
            "family": family,
            "category": category,
            "is_featured_frontier": is_featured,
            "release_date": rel_date,
            "lifecycle_status": "Active (Catalog)",
            "context_window": ctx_len,
            "input_modalities": ["text", "image"] if "image" in name.lower() or "vision" in name.lower() else ["text"],
            "output_modalities": ["text"],
            "license_type": license_type,
            "api_available": True,
            "consumer_available": False,  # Not verified for web consumer apps
            "pricing": pricing,
            "documentation_url": metric.get("source_url") or f"https://openrouter.ai/{cid}",
            "source_url": metric.get("source_url") or f"https://openrouter.ai/{cid}",
            "last_verified": today_str,
            "verification_sources": {
                "telemetry": f"OpenRouter ({today_str})",
                "pricing": f"OpenRouter ({today_str})" if pricing else "Not verified",
            },
            "notes": f"Discovered via OpenRouter catalog API. Telemetry specifications verified as of {today_str}.",
        }

        if validate_model_record(new_record):
            updated_models.append(new_record)
            existing_seen_ids.add(cid)
            appended_count += 1

    if updated_models:
        # Sort chronologically by release date descending so newest frontier models appear first
        updated_models.sort(
            key=lambda m: (
                m.get("release_date") or "",
                1 if m.get("is_featured_frontier") else 0,
                m.get("name") or "",
            ),
            reverse=True,
        )
        write_models_atomic(updated_models, path)
        export_identity_registry()
        logger.info(
            "Model catalog sync complete: %d existing models updated, %d new models appended (total: %d).",
            updated_count,
            appended_count,
            len(updated_models),
        )

    return updated_models
