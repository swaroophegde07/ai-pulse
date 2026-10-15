"""
Registry of official AI model update and release-note sources.

Provides typed models, schema validation, and loader utilities for
`src/data/model-update-sources.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_SOURCES_PATH = REPO_ROOT / "src" / "data" / "model-update-sources.json"

UpdateType = Literal[
    "model_release",
    "capability_change",
    "api_availability",
    "pricing_change",
    "context_window_change",
    "modality_change",
    "deprecation",
    "product_update",
    "research_release",
]

UpdateScope = Literal[
    "model",
    "model_family",
    "api",
    "product",
    "research",
]

AccessMethod = Literal[
    "rss",
    "atom",
    "html",
    "manual_only",
]

ALLOWED_UPDATE_TYPES: tuple[UpdateType, ...] = (
    "model_release",
    "capability_change",
    "api_availability",
    "pricing_change",
    "context_window_change",
    "modality_change",
    "deprecation",
    "product_update",
    "research_release",
)

ALLOWED_SCOPES: tuple[UpdateScope, ...] = (
    "model",
    "model_family",
    "api",
    "product",
    "research",
)

ALLOWED_ACCESS_METHODS: tuple[AccessMethod, ...] = (
    "rss",
    "atom",
    "html",
    "manual_only",
)


class ModelUpdateSource(TypedDict, total=False):
    id: str
    provider_id: str
    provider_name: str
    display_name: str
    url: str
    source_type: str
    access_method: str
    update_categories: list[str]
    enabled: bool
    notes: str


REQUIRED_SOURCE_FIELDS = frozenset([
    "id",
    "provider_id",
    "provider_name",
    "display_name",
    "url",
    "source_type",
    "access_method",
    "enabled",
])


def validate_source_record(record: dict) -> bool:
    """Validate that a source dictionary conforms to the required schema."""
    if not isinstance(record, dict):
        return False
    missing = REQUIRED_SOURCE_FIELDS - record.keys()
    if missing:
        logger.warning("Model update source missing required fields %s: %r", missing, record.get("id"))
        return False
    if record.get("access_method") not in ALLOWED_ACCESS_METHODS:
        logger.warning(
            "Source has invalid access_method '%s': %r",
            record.get("access_method"),
            record.get("id"),
        )
        return False
    if not isinstance(record.get("url"), str) or not record["url"].startswith("http"):
        logger.warning("Source has invalid URL: %r", record.get("id"))
        return False
    return True


def load_model_update_sources(path: Path | None = None) -> list[ModelUpdateSource]:
    """Load and validate all model update sources from JSON."""
    src_path = path or DEFAULT_SOURCES_PATH
    if not src_path.exists():
        logger.warning("Model update sources file not found: %s", src_path)
        return []

    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error("Model update sources JSON is not a list: %s", type(data))
            return []
        valid_sources = [s for s in data if validate_source_record(s)]
        return valid_sources
    except Exception as exc:
        logger.error("Failed to load model update sources from %s: %s", src_path, exc)
        return []


def get_active_model_update_sources(path: Path | None = None) -> list[ModelUpdateSource]:
    """Return only enabled sources whose access method supports automated ingestion."""
    sources = load_model_update_sources(path)
    return [s for s in sources if s.get("enabled", True) and s.get("access_method") != "manual_only"]
