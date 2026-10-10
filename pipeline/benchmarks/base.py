"""
Base types, schemas, and abstract adapter interfaces for benchmark intelligence.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

ALLOWED_BENCHMARK_CATEGORIES: list[str] = [
    "Reasoning",
    "Coding",
    "Agentic Coding",
    "Mathematics",
    "Data Analysis",
    "Language",
    "Instruction Following",
    "Composite",
]

ALLOWED_PROVENANCE_TYPES: list[str] = [
    "Independent Evaluation",
    "Aggregated by Source",
    "Platform Telemetry",
    "Provider Stated",
]


class BenchmarkRecord(TypedDict):
    model_name: str
    canonical_id: str
    provider: str
    benchmark_name: str
    category: str
    score: float
    score_unit: str
    evaluation_date: str | None
    source: str
    source_url: str
    provenance: str
    retrieved_at: str
    notes: str | None


class ModelMetricRecord(TypedDict):
    model_name: str
    canonical_id: str
    provider: str
    context_length: int
    input_price_per_1m: float | None
    output_price_per_1m: float | None
    currency: str
    source: str
    source_url: str
    retrieved_at: str


def validate_benchmark_record(record: dict[str, Any]) -> bool:
    """Validate that a benchmark record matches required schema and constraints."""
    required = {
        "model_name",
        "canonical_id",
        "provider",
        "benchmark_name",
        "category",
        "score",
        "score_unit",
        "source",
        "source_url",
        "provenance",
        "retrieved_at",
    }
    missing = required - record.keys()
    if missing:
        logger.warning("Benchmark record missing required fields %s: %r", missing, record)
        return False

    if not isinstance(record["score"], (int, float)):
        logger.warning("Invalid benchmark score (not numeric): %r", record)
        return False

    if record["category"] not in ALLOWED_BENCHMARK_CATEGORIES:
        logger.warning("Invalid benchmark category: %s", record.get("category"))
        return False

    if record["provenance"] not in ALLOWED_PROVENANCE_TYPES:
        logger.warning("Invalid benchmark provenance: %s", record.get("provenance"))
        return False

    return True


class SourceAdapter(ABC):
    """Abstract base class for benchmark and telemetry source adapters."""

    name: str = "BaseAdapter"
    source_url: str = ""
    enabled: bool = True

    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    @abstractmethod
    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch and return normalized benchmark records."""
        raise NotImplementedError

    def get_provenance_notice(self) -> dict[str, str]:
        """Return provenance, license, and access notice for this source."""
        return {
            "source": self.name,
            "url": self.source_url,
            "status": "Active" if self.enabled else "Disabled",
        }


def current_iso_timestamp() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()
