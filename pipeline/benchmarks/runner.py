"""
Orchestrator for multi-source benchmark intelligence and model telemetry ingestion.

Coordinates LiveBench, OpenRouter telemetry, and independent benchmark harnesses.
Enforces data-quality guarantees, prevents empty-dataset regressions, derives
dynamic benchmark leaders, and keeps the model catalog synchronized.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from pipeline.benchmarks.adapters.artificial_analysis import ArtificialAnalysisAdapter
from pipeline.benchmarks.adapters.benchlm import BenchLMAdapter
from pipeline.benchmarks.adapters.livebench import LiveBenchAdapter
from pipeline.benchmarks.adapters.lmcouncil import LMCouncilAdapter
from pipeline.benchmarks.adapters.openrouter import OpenRouterAdapter
from pipeline.benchmarks.base import (
    BenchmarkRecord,
    ModelMetricRecord,
    current_iso_timestamp,
    deduplicate_and_filter_conflicts,
    validate_benchmark_record,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BENCHMARKS_PATH = REPO_ROOT / "src" / "data" / "benchmarks.json"

# Core benchmark methodology standards (no hardcoded leader strings)
BENCHMARK_METHODOLOGY_STANDARDS = [
    {
        "name": "LiveBench",
        "category": "Reasoning & Coding",
        "description": "Contamination-free benchmark with monthly released questions across reasoning, coding, and math.",
        "eval_type": "Independent Automated Harness",
        "reference_url": "https://livebench.ai/",
        "benchmark_version": "Monthly Release",
        "methodological_limitations": "Evaluates objective reasoning and automated code execution; does not measure open-ended creative writing.",
    },
    {
        "name": "SWE-bench Verified",
        "category": "Agentic Coding",
        "description": "Human-validated subset of 500 real GitHub issue resolution tasks in Python repositories.",
        "eval_type": "Unit Test Execution",
        "reference_url": "https://www.swebench.com/",
        "benchmark_version": "Verified 500",
        "methodological_limitations": "Evaluates Python repository issue resolution with unit tests; computationally demanding harness.",
    },
    {
        "name": "Chatbot Arena (LMSYS)",
        "category": "Human Preference",
        "description": "Crowdsourced blind head-to-head pairwise comparisons evaluated with Bradley-Terry / Elo rating system.",
        "eval_type": "Human Blind Preference",
        "reference_url": "https://chat.lmsys.org/",
        "benchmark_version": "Continuous Arena",
        "methodological_limitations": "Reflects human subjective preference and prompt distribution; susceptible to formatting and verbosity bias.",
    },
    {
        "name": "GPQA Diamond",
        "category": "PhD-Level Science",
        "description": "448 multiple-choice graduate-level biology, chemistry, and physics questions designed by experts to be non-Googleable.",
        "eval_type": "Expert Verification",
        "reference_url": "https://github.com/idavidrein/gpqa",
        "benchmark_version": "Diamond Subset",
        "methodological_limitations": "Multiple-choice format with limited scope; difficult for non-expert human graders.",
    },
    {
        "name": "MMLU-Pro",
        "category": "Multidisciplinary",
        "description": "Rigorous 10-choice multidisciplinary reasoning benchmark designed to overcome MMLU saturation.",
        "eval_type": "Standardized Multiple Choice",
        "reference_url": "https://github.com/TIGER-AI-Lab/MMLU-Pro",
        "benchmark_version": "MMLU-Pro v1",
        "methodological_limitations": "10-option multiple choice; tests academic and factual recall more than extended agentic reasoning.",
    },
    {
        "name": "MATH 500 / AIME",
        "category": "Mathematics",
        "description": "Advanced competition mathematics problems from AMC, AIME, and high school Olympiads.",
        "eval_type": "Formal Mathematical Deduction",
        "reference_url": "https://github.com/openai/prm800k",
        "benchmark_version": "MATH 500 / AIME 2024",
        "methodological_limitations": "Requires formal symbol manipulation and proof techniques; difficult for standard language models without test-time compute.",
    },
]


def derive_benchmark_standards(benchmarks: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    """Derive standards definitions with dynamically computed leaders from validated records."""
    standards: list[dict[str, Any]] = []

    # Map LiveBench top performers dynamically from validated records
    livebench_overall = [
        b for b in benchmarks
        if b.get("source") == "LiveBench" and b.get("category") == "Composite"
    ]
    livebench_overall.sort(key=lambda x: x.get("score", 0), reverse=True)
    livebench_leaders = [
        f"{b['model_name']} ({b['score']}%)" for b in livebench_overall[:3]
    ]

    for std in BENCHMARK_METHODOLOGY_STANDARDS:
        std_copy = dict(std)
        if std["name"] == "LiveBench":
            if livebench_leaders:
                std_copy["leader"] = ", ".join(livebench_leaders)
                std_copy["leading_models"] = [b["model_name"] for b in livebench_overall[:3]]
            else:
                std_copy["leader"] = "Awaiting validated evaluation results"
                std_copy["leading_models"] = []
        else:
            # Check if any records exist in our dataset for this benchmark
            matching = [
                b for b in benchmarks if std["name"].lower() in b.get("benchmark_name", "").lower()
            ]
            if matching:
                matching.sort(key=lambda x: x.get("score", 0), reverse=True)
                std_copy["leader"] = ", ".join(f"{b['model_name']} ({b['score']}%)" for b in matching[:3])
                std_copy["leading_models"] = [b["model_name"] for b in matching[:3]]
            else:
                std_copy["leader"] = "Independent harness (reference only)"
                std_copy["leading_models"] = []
        standards.append(std_copy)

    return standards


def load_existing_benchmarks(benchmarks_path: Path) -> dict[str, Any]:
    """Load existing benchmarks dataset for fallback and snapshot preservation."""
    if not benchmarks_path.exists():
        return {}
    try:
        with benchmarks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("Could not read existing benchmarks from %s: %s", benchmarks_path, exc)
    return {}


def write_benchmarks_atomic(data: dict[str, Any], target_path: Path) -> None:
    """Write benchmark dataset atomically using sibling tempfile."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)

    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent, prefix=".benchmarks_tmp_", suffix=".json"
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
    logger.info("Atomically updated benchmark intelligence at %s", target_path)


def run_benchmark_pipeline(benchmarks_path: Path | None = None) -> dict[str, Any]:
    """Execute the multi-source benchmark ingestion pipeline."""
    target_path = benchmarks_path or DEFAULT_BENCHMARKS_PATH
    existing = load_existing_benchmarks(target_path)
    now_iso = current_iso_timestamp()

    logger.info("=== Multi-source benchmark ingestion starting ===")

    # Initialize adapters
    openrouter = OpenRouterAdapter()
    livebench = LiveBenchAdapter()
    benchlm = BenchLMAdapter()
    art_analysis = ArtificialAnalysisAdapter()
    lmcouncil = LMCouncilAdapter()

    adapters = [openrouter, livebench, benchlm, art_analysis, lmcouncil]
    provenance_notices = [a.get_provenance_notice() for a in adapters]

    # 1. Fetch benchmark records
    all_benchmark_records: list[BenchmarkRecord] = []
    used_cached_fallback = False

    # Fetch LiveBench
    livebench_records = livebench.fetch()
    if livebench_records:
        all_benchmark_records.extend(livebench_records)
    elif existing.get("benchmarks"):
        # Preserve previous LiveBench records if remote fetch had temporary error
        old_lb = [r for r in existing["benchmarks"] if r.get("source") == "LiveBench"]
        if old_lb:
            logger.info("Preserving %d cached LiveBench records from previous snapshot.", len(old_lb))
            all_benchmark_records.extend(old_lb)
            used_cached_fallback = True

    # Fetch BenchLM (if enabled)
    benchlm_records = benchlm.fetch()
    if benchlm_records:
        all_benchmark_records.extend(benchlm_records)

    # Fetch Artificial Analysis (if authenticated)
    art_records = art_analysis.fetch()
    if art_records:
        all_benchmark_records.extend(art_records)

    # Validate and filter conflicting measurements
    valid_records = [r for r in all_benchmark_records if validate_benchmark_record(r)]
    invalid_count = len(all_benchmark_records) - len(valid_records)

    # Compute duplicate observations
    from pipeline.benchmarks.base import get_measurement_key
    seen_unique_keys = set()
    duplicate_count = 0
    for r in valid_records:
        k = get_measurement_key(r)
        if k in seen_unique_keys:
            duplicate_count += 1
        else:
            seen_unique_keys.add(k)

    validated_benchmarks, detected_conflicts = deduplicate_and_filter_conflicts(valid_records)

    if detected_conflicts:
        logger.warning(
            "Detected %d conflicting benchmark measurements. Excluded ambiguous observations from output.",
            len(detected_conflicts),
        )

    # Health check & Empty dataset protection
    health_status = "stale_fallback" if used_cached_fallback else "healthy"
    health_warnings: list[str] = []
    if used_cached_fallback:
        health_warnings.append(
            f"Preserved last-known-good benchmark snapshot from {existing.get('metadata', {}).get('last_updated', 'previous run')}."
        )

    if not validated_benchmarks:
        if existing.get("benchmarks"):
            logger.warning(
                "Fresh benchmark ingestion yielded 0 valid records while existing snapshot contains %d records. "
                "Preserving last-known-good snapshot to prevent empty dataset regression.",
                len(existing["benchmarks"]),
            )
            validated_benchmarks = existing["benchmarks"]
            health_status = "stale_fallback"
            health_warnings.append(
                f"Preserved last-known-good benchmark snapshot from {existing.get('metadata', {}).get('last_updated', 'previous run')}."
            )
        else:
            health_status = "empty"
            health_warnings.append("No benchmark records available from any source.")

    # 2. Fetch OpenRouter model telemetry (pricing and context lengths)
    openrouter_metrics = openrouter.fetch_metrics()
    if not openrouter_metrics and existing.get("model_metrics"):
        openrouter_metrics = existing["model_metrics"]

    # 3. Derive dynamic standards with actual data-driven leaders
    benchmark_standards = derive_benchmark_standards(validated_benchmarks)

    # Assemble dataset
    output_payload: dict[str, Any] = {
        "metadata": {
            "last_updated": now_iso,
            "total_benchmark_records": len(validated_benchmarks),
            "total_model_metrics": len(openrouter_metrics),
            "status": health_status,
            "validation_report": {
                "retrieved_count": len(all_benchmark_records),
                "parsed_count": len(all_benchmark_records),
                "valid_count": len(valid_records),
                "duplicate_count": duplicate_count,
                "conflict_count": len(detected_conflicts),
                "quarantined_count": len(detected_conflicts),
                "retained_count": len(validated_benchmarks),
            },
            "health": {
                "status": health_status,
                "records_parsed": len(all_benchmark_records),
                "records_rejected": invalid_count,
                "records_conflicts": len(detected_conflicts),
                "records_retained": len(validated_benchmarks),
                "last_successful_evaluation_date": (
                    validated_benchmarks[0].get("evaluation_date") if validated_benchmarks else None
                ),
                "last_retrieval_time": now_iso,
                "warnings": health_warnings,
            },
            "sources": provenance_notices,
            "disclaimer": (
                "Benchmarks reflect standardized evaluations under specific test harnesses. "
                "Usage volumes and task scores are distinct metrics and must not be conflated "
                "into a single composite ranking without verified methodology."
            ),
        },
        "benchmarks": validated_benchmarks,
        "model_metrics": openrouter_metrics,
        "benchmark_standards": benchmark_standards,
    }

    write_benchmarks_atomic(output_payload, target_path)
    logger.info("Benchmark pipeline complete: %d benchmark records, %d telemetry records (status: %s).",
                len(validated_benchmarks), len(openrouter_metrics), health_status)

    # 4. Synchronize models catalog (src/data/models.json) with OpenRouter telemetry
    try:
        from pipeline.models.updater import sync_models_catalog
        sync_models_catalog(openrouter_metrics=openrouter_metrics)
    except Exception as m_exc:
        logger.warning("Models catalog synchronization error (non-fatal): %s", m_exc)

    return output_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_benchmark_pipeline()
