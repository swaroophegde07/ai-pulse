"""
Model Identity Mapping and Registry.

Provides explicit, verified identity resolution across model catalogues and benchmark sources.
Strictly distinguishes model versions, reasoning configurations, quantised variants, and
evaluation harnesses to prevent silent identifier collisions or false merges.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class MatchType(str, Enum):
    EXACT_INTERNAL_ID = "exact_internal_id"
    EXACT_OPENROUTER_ID = "exact_openrouter_id"
    EXACT_PROVIDER_ID = "exact_provider_id"
    VERIFIED_ALIAS = "verified_alias"
    EXACT_BENCHMARK_NAME = "exact_benchmark_name"
    NORMALIZED_SLUG = "normalized_slug"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class MatchResult:
    matched_id: str | None
    confidence: float
    match_type: MatchType
    canonical_name: str | None
    provider: str | None
    family: str | None
    configuration: str | None
    is_ambiguous: bool
    notes: str


class ModelIdentityRecord(TypedDict):
    internal_id: str
    canonical_name: str
    provider: str
    family: str
    openrouter_id: str | None
    provider_id: str | None
    benchmark_names: list[str]
    verified_aliases: list[str]
    is_featured_frontier: bool
    capability_type: str
    default_configuration: str | None
    provenance: str


# Explicit, verified identity registry for primary foundation models
VERIFIED_MODEL_REGISTRY: list[ModelIdentityRecord] = [
    # --- OpenAI ---
    {
        "internal_id": "openai/o3-mini",
        "canonical_name": "OpenAI o3-mini",
        "provider": "OpenAI",
        "family": "o-series",
        "openrouter_id": "openai/o3-mini",
        "provider_id": "o3-mini-2025-01-31",
        "benchmark_names": ["o3-mini", "o3-mini-high", "o3-mini-medium", "o3-mini-low"],
        "verified_aliases": ["OpenAI o3-mini", "o3-mini", "openai/o3-mini-high"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "high",
        "provenance": "Official OpenAI Announcement",
    },
    {
        "internal_id": "openai/o1",
        "canonical_name": "OpenAI o1",
        "provider": "OpenAI",
        "family": "o-series",
        "openrouter_id": "openai/o1",
        "provider_id": "o1-2024-12-17",
        "benchmark_names": ["o1", "o1-preview", "openai-o1", "o1-2024-12-17"],
        "verified_aliases": ["OpenAI o1", "o1", "openai/o1"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official OpenAI Announcement",
    },
    {
        "internal_id": "openai/gpt-4o",
        "canonical_name": "GPT-4o (Omni)",
        "provider": "OpenAI",
        "family": "GPT-4",
        "openrouter_id": "openai/gpt-4o",
        "provider_id": "gpt-4o-2024-05-13",
        "benchmark_names": ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"],
        "verified_aliases": ["GPT-4o", "GPT-4o (Omni)", "openai/gpt-4o"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official OpenAI Announcement",
    },
    {
        "internal_id": "openai/gpt-4o-mini",
        "canonical_name": "GPT-4o mini",
        "provider": "OpenAI",
        "family": "GPT-4",
        "openrouter_id": "openai/gpt-4o-mini",
        "provider_id": "gpt-4o-mini-2024-07-18",
        "benchmark_names": ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
        "verified_aliases": ["GPT-4o mini", "openai/gpt-4o-mini"],
        "is_featured_frontier": False,
        "capability_type": "general_chat",
        "default_configuration": None,
        "provenance": "Official OpenAI Announcement",
    },
    {
        "internal_id": "openai/gpt-6.1-sol",
        "canonical_name": "GPT-6.1 Sol",
        "provider": "OpenAI",
        "family": "GPT-6",
        "openrouter_id": "openai/gpt-6.1-sol",
        "provider_id": "gpt-6.1-sol",
        "benchmark_names": ["gpt-6.1-sol-max", "gpt-6.1-sol-xhigh", "gpt-6.1-sol"],
        "verified_aliases": ["GPT-6.1 Sol", "OpenAI: GPT-6.1 Sol", "openai/gpt-6.1-sol-pro"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "max",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },
    {
        "internal_id": "openai/gpt-6-astra",
        "canonical_name": "GPT-6 Astra",
        "provider": "OpenAI",
        "family": "GPT-6",
        "openrouter_id": "openai/gpt-6-astra",
        "provider_id": "gpt-6-astra",
        "benchmark_names": ["gpt-6-astra-max", "gpt-6-astra"],
        "verified_aliases": ["GPT-6 Astra", "OpenAI: GPT-6 Astra", "openai/gpt-6-astra-pro"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "max",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },
    {
        "internal_id": "openai/gpt-5.5",
        "canonical_name": "GPT-5.5",
        "provider": "OpenAI",
        "family": "GPT-5",
        "openrouter_id": "openai/gpt-5.5",
        "provider_id": "gpt-5.5",
        "benchmark_names": ["gpt-5.5-xhigh", "gpt-5.5-high", "gpt-5.5-medium"],
        "verified_aliases": ["GPT-5.5", "OpenAI: GPT-5.5"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "xhigh",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },

    # --- Anthropic ---
    {
        "internal_id": "anthropic/claude-3-7-sonnet",
        "canonical_name": "Claude 3.7 Sonnet",
        "provider": "Anthropic",
        "family": "Claude 3.7",
        "openrouter_id": "anthropic/claude-3-7-sonnet",
        "provider_id": "claude-3-7-sonnet-20250219",
        "benchmark_names": ["claude-3-7-sonnet", "claude-3-7-sonnet-thinking"],
        "verified_aliases": ["Claude 3.7 Sonnet", "anthropic/claude-3-7-sonnet"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "hybrid_thinking",
        "provenance": "Official Anthropic Announcement",
    },
    {
        "internal_id": "anthropic/claude-3-5-sonnet",
        "canonical_name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "family": "Claude 3.5",
        "openrouter_id": "anthropic/claude-3-5-sonnet",
        "provider_id": "claude-3-5-sonnet-20241022",
        "benchmark_names": ["claude-3-5-sonnet", "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"],
        "verified_aliases": ["Claude 3.5 Sonnet", "anthropic/claude-3-5-sonnet"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Anthropic Announcement",
    },
    {
        "internal_id": "anthropic/claude-3-5-haiku",
        "canonical_name": "Claude 3.5 Haiku",
        "provider": "Anthropic",
        "family": "Claude 3.5",
        "openrouter_id": "anthropic/claude-3-5-haiku",
        "provider_id": "claude-3-5-haiku-20241022",
        "benchmark_names": ["claude-3-5-haiku", "claude-3-5-haiku-20241022"],
        "verified_aliases": ["Claude 3.5 Haiku", "anthropic/claude-3-5-haiku"],
        "is_featured_frontier": False,
        "capability_type": "general_chat",
        "default_configuration": None,
        "provenance": "Official Anthropic Announcement",
    },
    {
        "internal_id": "anthropic/claude-sonnet-5.5",
        "canonical_name": "Claude Sonnet 5.5",
        "provider": "Anthropic",
        "family": "Claude 5",
        "openrouter_id": "anthropic/claude-sonnet-5.5",
        "provider_id": "claude-sonnet-5-5",
        "benchmark_names": ["claude-sonnet-5-5-max-effort", "claude-sonnet-5-5-xhigh-effort"],
        "verified_aliases": ["Claude Sonnet 5.5", "Anthropic: Claude Sonnet 5.5"],
        "is_featured_frontier": True,
        "capability_type": "coding",
        "default_configuration": "max-effort",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },
    {
        "internal_id": "anthropic/claude-opus-5.5",
        "canonical_name": "Claude Opus 5.5",
        "provider": "Anthropic",
        "family": "Claude 5",
        "openrouter_id": "anthropic/claude-opus-5.5",
        "provider_id": "claude-opus-5-5",
        "benchmark_names": ["claude-opus-5-5-max-effort", "claude-opus-5-5-xhigh-effort"],
        "verified_aliases": ["Claude Opus 5.5", "Anthropic: Claude Opus 5.5"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "max-effort",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },

    # --- Google DeepMind ---
    {
        "internal_id": "google/gemini-2-0-flash",
        "canonical_name": "Gemini 2.0 Flash",
        "provider": "Google DeepMind",
        "family": "Gemini 2",
        "openrouter_id": "google/gemini-2.0-flash",
        "provider_id": "gemini-2.0-flash",
        "benchmark_names": ["gemini-2.0-flash", "gemini-2-0-flash"],
        "verified_aliases": ["Gemini 2.0 Flash", "google/gemini-2-0-flash"],
        "is_featured_frontier": True,
        "capability_type": "general_chat",
        "default_configuration": None,
        "provenance": "Official Google DeepMind Announcement",
    },
    {
        "internal_id": "google/gemini-2-0-pro-exp",
        "canonical_name": "Gemini 2.0 Pro Experimental",
        "provider": "Google DeepMind",
        "family": "Gemini 2",
        "openrouter_id": "google/gemini-2.0-pro-exp",
        "provider_id": "gemini-2.0-pro-exp-02-05",
        "benchmark_names": ["gemini-2.0-pro", "gemini-2-0-pro"],
        "verified_aliases": ["Gemini 2.0 Pro", "google/gemini-2-0-pro-exp"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Google DeepMind Announcement",
    },
    {
        "internal_id": "google/gemini-3.5-flash",
        "canonical_name": "Gemini 3.5 Flash",
        "provider": "Google DeepMind",
        "family": "Gemini 3",
        "openrouter_id": "google/gemini-3.5-flash",
        "provider_id": "gemini-3.5-flash",
        "benchmark_names": ["gemini-3.5-flash-high", "gemini-3.5-flash"],
        "verified_aliases": ["Gemini 3.5 Flash", "Google: Gemini 3.5 Flash"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "high",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },

    # --- DeepSeek ---
    {
        "internal_id": "deepseek/deepseek-r1",
        "canonical_name": "DeepSeek-R1",
        "provider": "DeepSeek",
        "family": "DeepSeek-R1",
        "openrouter_id": "deepseek/deepseek-r1",
        "provider_id": "deepseek-reasoner",
        "benchmark_names": ["deepseek-r1", "deepseek-reasoner"],
        "verified_aliases": ["DeepSeek-R1", "deepseek/deepseek-r1"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official DeepSeek Paper & Open Weights Release",
    },
    {
        "internal_id": "deepseek/deepseek-v3",
        "canonical_name": "DeepSeek-V3",
        "provider": "DeepSeek",
        "family": "DeepSeek-V3",
        "openrouter_id": "deepseek/deepseek-v3",
        "provider_id": "deepseek-chat",
        "benchmark_names": ["deepseek-v3", "deepseek-chat"],
        "verified_aliases": ["DeepSeek-V3", "deepseek/deepseek-v3"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official DeepSeek Paper & Open Weights Release",
    },
    {
        "internal_id": "deepseek/deepseek-v4.1-flash",
        "canonical_name": "DeepSeek-V4.1 Flash",
        "provider": "DeepSeek",
        "family": "DeepSeek-V4",
        "openrouter_id": "deepseek/deepseek-v4.1-flash",
        "provider_id": "deepseek-v4.1-flash",
        "benchmark_names": ["deepseek-v4-flash", "deepseek-v4.1-flash"],
        "verified_aliases": ["DeepSeek V4.1 Flash", "DeepSeek: DeepSeek V4.1 Flash"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },

    # --- Meta AI ---
    {
        "internal_id": "meta/llama-3-3-70b-instruct",
        "canonical_name": "Llama 3.3 70B Instruct",
        "provider": "Meta AI",
        "family": "Llama 3.3",
        "openrouter_id": "meta-llama/llama-3.3-70b-instruct",
        "provider_id": "llama-3.3-70b-instruct",
        "benchmark_names": ["llama-3.3-70b-instruct", "llama-3-3-70b-instruct"],
        "verified_aliases": ["Llama 3.3 70B", "meta/llama-3-3-70b-instruct"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Meta AI Open Weights Release",
    },
    {
        "internal_id": "meta/llama-3-1-405b-instruct",
        "canonical_name": "Llama 3.1 405B Instruct",
        "provider": "Meta AI",
        "family": "Llama 3.1",
        "openrouter_id": "meta-llama/llama-3.1-405b-instruct",
        "provider_id": "llama-3.1-405b-instruct",
        "benchmark_names": ["llama-3.1-405b-instruct", "llama-3-1-405b-instruct"],
        "verified_aliases": ["Llama 3.1 405B", "meta/llama-3-1-405b-instruct"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Meta AI Open Weights Release",
    },

    # --- Mistral AI ---
    {
        "internal_id": "mistral/mistral-large-2411",
        "canonical_name": "Mistral Large 2",
        "provider": "Mistral AI",
        "family": "Mistral Large",
        "openrouter_id": "mistralai/mistral-large-2411",
        "provider_id": "mistral-large-2411",
        "benchmark_names": ["mistral-large-2411", "mistral-large-2"],
        "verified_aliases": ["Mistral Large 2", "mistral/mistral-large-2411"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Mistral AI Announcement",
    },
    {
        "internal_id": "mistralai/mistral-large-4-0",
        "canonical_name": "Mistral Large 4",
        "provider": "Mistral AI",
        "family": "Mistral Large",
        "openrouter_id": "mistralai/mistral-large-4-0",
        "provider_id": "mistral-large-4",
        "benchmark_names": ["mistral-large-4-high", "mistral-large-4"],
        "verified_aliases": ["Mistral Large 4", "Mistral: Mistral Large 4"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": "high",
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },
    {
        "internal_id": "mistralai/devstral-2512",
        "canonical_name": "Devstral 2",
        "provider": "Mistral AI",
        "family": "Devstral",
        "openrouter_id": "mistralai/devstral-2512",
        "provider_id": "devstral-2512",
        "benchmark_names": ["devstral-2512"],
        "verified_aliases": ["Devstral 2", "Mistral: Devstral 2 2512"],
        "is_featured_frontier": False,
        "capability_type": "coding",
        "default_configuration": None,
        "provenance": "OpenRouter Telemetry",
    },

    # --- Alibaba Cloud ---
    {
        "internal_id": "qwen/qwen3.8-max-prime",
        "canonical_name": "Qwen 3.8 Max Prime",
        "provider": "Alibaba Cloud",
        "family": "Qwen 3",
        "openrouter_id": "qwen/qwen3.8-max-prime",
        "provider_id": "qwen3.8-max-prime",
        "benchmark_names": ["qwen3.8-max-prime", "qwen3.7-max"],
        "verified_aliases": ["Qwen 3.8 Max Prime", "Qwen: Qwen3.8 Max Prime"],
        "is_featured_frontier": True,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "OpenRouter Telemetry & LiveBench Ingestion",
    },

    # --- Microsoft ---
    {
        "internal_id": "microsoft/phi-4",
        "canonical_name": "Phi-4",
        "provider": "Microsoft",
        "family": "Phi",
        "openrouter_id": "microsoft/phi-4",
        "provider_id": "phi-4",
        "benchmark_names": ["phi-4"],
        "verified_aliases": ["Phi-4", "microsoft/phi-4"],
        "is_featured_frontier": False,
        "capability_type": "frontier_reasoning",
        "default_configuration": None,
        "provenance": "Official Microsoft Research Release",
    },
]

# Fast index lookups
_BY_INTERNAL_ID: dict[str, ModelIdentityRecord] = {r["internal_id"]: r for r in VERIFIED_MODEL_REGISTRY}
_BY_OPENROUTER_ID: dict[str, ModelIdentityRecord] = {
    r["openrouter_id"]: r for r in VERIFIED_MODEL_REGISTRY if r["openrouter_id"]
}
_BY_PROVIDER_ID: dict[str, ModelIdentityRecord] = {
    r["provider_id"]: r for r in VERIFIED_MODEL_REGISTRY if r["provider_id"]
}
_BY_BENCHMARK_NAME: dict[str, ModelIdentityRecord] = {}
for r in VERIFIED_MODEL_REGISTRY:
    for bn in r["benchmark_names"]:
        _BY_BENCHMARK_NAME[bn.lower()] = r

_BY_VERIFIED_ALIAS: dict[str, ModelIdentityRecord] = {}
for r in VERIFIED_MODEL_REGISTRY:
    for al in r["verified_aliases"]:
        _BY_VERIFIED_ALIAS[al.lower()] = r


def resolve_model_identity(raw_identifier: str, context_source: str = "") -> MatchResult:
    """Resolve an incoming raw model identifier against the verified identity registry.

    Enforces strict identity preservation:
    1. Exact match on internal ID
    2. Exact match on OpenRouter canonical ID
    3. Exact match on Provider ID
    4. Exact match on Benchmark name
    5. Exact match on Verified Alias
    6. Exact normalized slug match
    7. Ambiguous or Unresolved (does NOT collapse distinct model versions or configurations)
    """
    ident = raw_identifier.strip()
    ident_lower = ident.lower()

    if not ident:
        return MatchResult(
            matched_id=None,
            confidence=0.0,
            match_type=MatchType.UNRESOLVED,
            canonical_name=None,
            provider=None,
            family=None,
            configuration=None,
            is_ambiguous=False,
            notes="Empty identifier provided",
        )

    # 1. Exact match on internal ID
    if ident in _BY_INTERNAL_ID:
        rec = _BY_INTERNAL_ID[ident]
        return MatchResult(
            matched_id=rec["internal_id"],
            confidence=1.0,
            match_type=MatchType.EXACT_INTERNAL_ID,
            canonical_name=rec["canonical_name"],
            provider=rec["provider"],
            family=rec["family"],
            configuration=rec["default_configuration"],
            is_ambiguous=False,
            notes="Exact match against internal model ID",
        )

    # 2. Exact match on OpenRouter ID
    if ident in _BY_OPENROUTER_ID:
        rec = _BY_OPENROUTER_ID[ident]
        return MatchResult(
            matched_id=rec["internal_id"],
            confidence=1.0,
            match_type=MatchType.EXACT_OPENROUTER_ID,
            canonical_name=rec["canonical_name"],
            provider=rec["provider"],
            family=rec["family"],
            configuration=rec["default_configuration"],
            is_ambiguous=False,
            notes="Exact match against OpenRouter catalogue ID",
        )

    # 3. Exact match on Provider ID
    if ident in _BY_PROVIDER_ID:
        rec = _BY_PROVIDER_ID[ident]
        return MatchResult(
            matched_id=rec["internal_id"],
            confidence=1.0,
            match_type=MatchType.EXACT_PROVIDER_ID,
            canonical_name=rec["canonical_name"],
            provider=rec["provider"],
            family=rec["family"],
            configuration=rec["default_configuration"],
            is_ambiguous=False,
            notes="Exact match against provider model identifier",
        )

    # 4. Exact match on registered benchmark names
    if ident_lower in _BY_BENCHMARK_NAME:
        rec = _BY_BENCHMARK_NAME[ident_lower]
        # Extract configuration if present
        config = _extract_configuration(ident_lower) or rec["default_configuration"]
        return MatchResult(
            matched_id=rec["internal_id"],
            confidence=1.0,
            match_type=MatchType.EXACT_BENCHMARK_NAME,
            canonical_name=rec["canonical_name"],
            provider=rec["provider"],
            family=rec["family"],
            configuration=config,
            is_ambiguous=False,
            notes="Exact match against registered benchmark evaluation name",
        )

    # 5. Exact match on verified aliases
    if ident_lower in _BY_VERIFIED_ALIAS:
        rec = _BY_VERIFIED_ALIAS[ident_lower]
        return MatchResult(
            matched_id=rec["internal_id"],
            confidence=1.0,
            match_type=MatchType.VERIFIED_ALIAS,
            canonical_name=rec["canonical_name"],
            provider=rec["provider"],
            family=rec["family"],
            configuration=rec["default_configuration"],
            is_ambiguous=False,
            notes="Exact match against verified alias",
        )

    # 6. Normalized exact slug match
    norm_slug = _normalize_slug(ident_lower)
    for rec in VERIFIED_MODEL_REGISTRY:
        if norm_slug == _normalize_slug(rec["internal_id"]) or norm_slug == _normalize_slug(rec["canonical_name"]):
            return MatchResult(
                matched_id=rec["internal_id"],
                confidence=0.95,
                match_type=MatchType.NORMALIZED_SLUG,
                canonical_name=rec["canonical_name"],
                provider=rec["provider"],
                family=rec["family"],
                configuration=rec["default_configuration"],
                is_ambiguous=False,
                notes="Normalized slug exact match",
            )

    # 7. Safety check: avoid false collapse of versioned/variant models
    # If the identifier has a prefix resembling a known model family but contains
    # distinct version identifiers (e.g. "deepseek-v3.2" vs "deepseek-v3" or "gpt-5.5-xhigh" vs "gpt-5.5"),
    # do NOT collapse! Flag as distinct unresolved / variant candidate.
    config = _extract_configuration(ident_lower)

    return MatchResult(
        matched_id=None,
        confidence=0.0,
        match_type=MatchType.UNRESOLVED,
        canonical_name=None,
        provider=None,
        family=None,
        configuration=config,
        is_ambiguous=False,
        notes=f"Unresolved model identifier '{ident}'. Preserving upstream identity without false collapse.",
    )


def _extract_configuration(identifier: str) -> str | None:
    """Extract configuration, effort, or harness modifier from identifier."""
    match = re.search(
        r"(?:-|_)(max-effort|xhigh-effort|high-effort|medium-effort|low-effort|thinking-\d+k|thinking|xhigh|high|low|medium|max|codex|preview|base|nano|mini)$",
        identifier.lower(),
    )
    return match.group(1) if match else None


def _normalize_slug(s: str) -> str:
    """Normalize string into a strict alphanumeric slug for comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_featured_frontier_ids() -> set[str]:
    """Return the set of internal model IDs marked as featured frontier models."""
    return {r["internal_id"] for r in VERIFIED_MODEL_REGISTRY if r["is_featured_frontier"]}


def export_identity_registry(target_path: Any | None = None) -> list[ModelIdentityRecord]:
    """Export the verified identity registry to JSON for consumption by Astro and frontend components."""
    from pathlib import Path
    import json
    import tempfile
    import os

    path = Path(target_path) if target_path else (Path(__file__).parent.parent.parent / "src" / "data" / "model-identities.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(VERIFIED_MODEL_REGISTRY, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".model_identities_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return VERIFIED_MODEL_REGISTRY
