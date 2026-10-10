"""
Orchestrator for multi-source benchmark intelligence and model telemetry ingestion.
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

VERIFIED_STANDARDS = [
    {
        "name": "LiveBench",
        "category": "Reasoning & Coding",
        "description": "Contamination-free benchmark with monthly released questions across reasoning, coding, and math.",
        "eval_type": "Independent Automated Harness",
        "leader": "OpenAI o1, DeepSeek-R1, Claude 3.5 Sonnet",
        "reference_url": "https://livebench.ai/",
    },
    {
        "name": "SWE-bench Verified",
        "category": "Agentic Coding",
        "description": "Human-validated subset of 500 real GitHub issue resolution tasks in Python repositories.",
        "eval_type": "Unit Test Execution",
        "leader": "Claude 3.7 Sonnet, OpenAI o3-mini",
        "reference_url": "https://www.swebench.com/",
    },
    {
        "name": "Chatbot Arena (LMSYS)",
        "category": "Human Preference",
        "description": "Crowdsourced blind head-to-head pairwise comparisons evaluated with Elo rating system.",
        "eval_type": "Human Blind Preference",
        "leader": "Gemini 2.0 Pro, GPT-4o, Claude 3.5 Sonnet",
        "reference_url": "https://chat.lmsys.org/",
    },
    {
        "name": "GPQA Diamond",
        "category": "PhD-Level Science",
        "description": "448 multiple-choice graduate-level biology, chemistry, and physics questions designed by experts to be non-Googleable.",
        "eval_type": "Expert Verification",
        "leader": "OpenAI o1, DeepSeek-R1",
        "reference_url": "https://github.com/idavidrein/gpqa",
    },
    {
        "name": "MMLU-Pro",
        "category": "Multidisciplinary",
        "description": "Rigorous 10-choice multidisciplinary reasoning benchmark designed to overcome MMLU saturation.",
        "eval_type": "Standardized Multiple Choice",
        "leader": "OpenAI o1, Gemini 2.0 Pro",
        "reference_url": "https://github.com/TIGER-AI-Lab/MMLU-Pro",
    },
    {
        "name": "MATH 500 / AIME",
        "category": "Mathematics",
        "description": "Advanced competition mathematics problems from AMC, AIME, and high school Olympiads.",
        "eval_type": "Formal Mathematical Deduction",
        "leader": "DeepSeek-R1, OpenAI o1",
        "reference_url": "https://github.com/openai/prm800k",
    },
]


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
    validated_benchmarks, detected_conflicts = deduplicate_and_filter_conflicts(valid_records)
    if detected_conflicts:
        logger.warning(
            "Detected %d conflicting benchmark measurements. Excluded ambiguous observations from output.",
            len(detected_conflicts),
        )

    # 2. Fetch OpenRouter model telemetry (pricing and context lengths)
    openrouter_metrics = openrouter.fetch_metrics()
    if not openrouter_metrics and existing.get("model_metrics"):
        openrouter_metrics = existing["model_metrics"]

    # Assemble dataset
    output_payload: dict[str, Any] = {
        "metadata": {
            "last_updated": now_iso,
            "total_benchmark_records": len(validated_benchmarks),
            "total_model_metrics": len(openrouter_metrics),
            "sources": provenance_notices,
            "disclaimer": (
                "Benchmarks reflect standardized evaluations under specific test harnesses. "
                "Usage volumes and task scores are distinct metrics and must not be conflated "
                "into a single composite ranking without verified methodology."
            ),
        },
        "benchmarks": validated_benchmarks,
        "model_metrics": openrouter_metrics,
        "benchmark_standards": VERIFIED_STANDARDS,
    }

    write_benchmarks_atomic(output_payload, target_path)
    logger.info("Benchmark pipeline complete: %d benchmark records, %d telemetry records.",
                len(validated_benchmarks), len(openrouter_metrics))

    # 3. Synchronize models catalog (src/data/models.json) with OpenRouter telemetry
    try:
        from pipeline.models.updater import sync_models_catalog
        sync_models_catalog(openrouter_metrics=openrouter_metrics)
    except Exception as m_exc:
        logger.warning("Models catalog synchronization error (non-fatal): %s", m_exc)

    return output_payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_benchmark_pipeline()
