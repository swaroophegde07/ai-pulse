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
    "Agentic",
    "Reasoning",
    "Coding",
    "Browser Use",
    "Data Analysis",
    "Research",
    "Design",
    "Agentic Coding",
    "Mathematics",
    "Language",
    "Instruction Following",
    "Human Preference",
    "Composite",
]

ALLOWED_PROVENANCE_TYPES: list[str] = [
    "Independent Evaluation",
    "Aggregated by Source",
    "Platform Telemetry",
    "Provider Stated",
]


class BenchmarkRecord(TypedDict, total=False):
    model_name: str
    canonical_id: str
    internal_model_id: str | None
    match_confidence: float | None
    provider: str
    benchmark_name: str
    category: str
    score: float
    score_unit: str
    evaluation_date: str | None
    benchmark_release_date: str | None
    benchmark_version: str | None
    source: str
    source_url: str
    provenance: str
    retrieved_at: str
    notes: str | None
    model_version: str | None
    configuration: str | None
    harness: str | None


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


def get_measurement_key(r: dict[str, Any]) -> tuple:
    """Derive full measurement identity key preserving model identity, version, configuration, suite, and release/evaluation date."""
    return (
        r.get("source"),
        r.get("canonical_id"),
        r.get("model_name"),
        r.get("configuration"),
        r.get("benchmark_name"),
        r.get("category"),
        r.get("benchmark_release_date") or r.get("benchmark_version"),
        r.get("evaluation_date"),
    )


def deduplicate_and_filter_conflicts(
    records: list[BenchmarkRecord],
) -> tuple[list[BenchmarkRecord], list[dict[str, Any]]]:
    """Detect conflicting measurements for the same model, benchmark, category, configuration, and date.

    If two records share the exact same logical measurement identity:
    (source, canonical_id, model_name, configuration, benchmark_name, category, evaluation_date)
    but have different scores, flag them as conflicting and exclude the ambiguous
    records from public leaderboards until verified.
    """
    seen: dict[tuple, BenchmarkRecord] = {}
    conflicts: list[dict[str, Any]] = []
    conflict_keys: set[tuple] = set()

    for r in records:
        key = get_measurement_key(r)
        if key in seen:
            prev = seen[key]
            if prev["score"] != r["score"]:
                logger.warning(
                    "Conflicting benchmark score detected for %s [%s]: %s (score %.2f vs %.2f). Excluding ambiguous record.",
                    r.get("canonical_id"),
                    r.get("model_name"),
                    r.get("benchmark_name"),
                    prev["score"],
                    r["score"],
                )
                conflicts.append({
                    "identity": f"{key[1]} ({key[2]}) - {key[4]} [{key[6]}]",
                    "existing_score": prev["score"],
                    "conflicting_score": r["score"],
                    "source": r.get("source"),
                })
                conflict_keys.add(key)
        else:
            seen[key] = r

    # Filter out all ambiguous measurements whose keys experienced conflicts
    clean_records = [
        rec for rec in seen.values()
        if get_measurement_key(rec) not in conflict_keys
    ]

    return clean_records, conflicts


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
