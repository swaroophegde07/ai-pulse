"""
LiveBench adapter for contamination-free LLM benchmark results across agentic, reasoning, and coding tasks.

Supports both LiveBench 2.0 (long-horizon agentic benchmarks released 2026-10-13)
and historical LiveBench 1.0 monthly releases, preserving exact model identities,
release metadata, and evaluation configurations.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any
import requests

from pipeline.benchmarks.base import (
    BenchmarkRecord,
    SourceAdapter,
    current_iso_timestamp,
    validate_benchmark_record,
)
from pipeline.models.identity import resolve_model_identity

logger = logging.getLogger(__name__)

# Release date to underlying upstream file date mapping
RELEASE_FILE_MAPPINGS: dict[str, str] = {
    "2026-10-14": "20261014",
    "2026-10-13": "2026-09-02",
}

# Ordered list of officially verified releases from LiveBench (newest first)
VERIFIED_LIVEBENCH_RELEASES = [
    "2026-10-14",
    "2026-06-25",
    "2026-01-08",
    "2025-12-23",
    "2025-11-25",
    "2025-05-30",
    "2025-04-25",
    "2025-04-02",
    "2024-11-25",
    "2024-08-31",
    "2024-07-26",
    "2024-06-24",
]

DEFAULT_LIVEBENCH_RELEASE = VERIFIED_LIVEBENCH_RELEASES[0]


def get_release_urls(release_date: str) -> list[tuple[str, str, str]]:
    """Return candidate (table_url, categories_url, generation) tuples in priority order."""
    file_key = RELEASE_FILE_MAPPINGS.get(release_date, release_date).replace("-", "_")
    compact_date = release_date.replace("-", "")
    candidates = []

    if release_date in ("2026-10-14", "2026-10-13") or release_date >= "2026-09-01":
        # LiveBench 2.0 verified directory data structure: /data/<compact_date>/table.csv
        candidates.append((
            f"https://livebench.ai/data/{compact_date}/table.csv",
            f"https://livebench.ai/data/{compact_date}/categories.json",
            "2.0",
        ))
        # Direct date / file_key fallbacks
        candidates.append((
            f"https://livebench.ai/table_{file_key}.csv",
            f"https://livebench.ai/categories_{file_key}.json",
            "2.0",
        ))
        direct_key = release_date.replace("-", "_")
        if direct_key != file_key:
            candidates.append((
                f"https://livebench.ai/table_{direct_key}.csv",
                f"https://livebench.ai/categories_{direct_key}.json",
                "2.0",
            ))
        candidates.append((
            f"https://livebench.ai/v1/table_{file_key}.csv",
            f"https://livebench.ai/v1/categories_{file_key}.json",
            "2.0",
        ))
    else:
        # LiveBench 1.0 candidate URLs (both v1/ and root paths for historical compatibility)
        candidates.append((
            f"https://livebench.ai/v1/table_{file_key}.csv",
            f"https://livebench.ai/v1/categories_{file_key}.json",
            "1.0",
        ))
        candidates.append((
            f"https://livebench.ai/table_{file_key}.csv",
            f"https://livebench.ai/categories_{file_key}.json",
            "1.0",
        ))

    return candidates


def discover_latest_release(timeout: int = 10) -> tuple[str, bool]:
    """Discover the latest available LiveBench release.

    Returns:
        (release_date, is_fallback_or_stale)
    """
    # 1. Probe LiveBench 2.0 verified release endpoint first
    try:
        t2_url = "https://livebench.ai/data/20261014/table.csv"
        probe = requests.head(
            t2_url,
            timeout=timeout,
            headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
        )
        if probe.ok:
            logger.info("Discovered verified LiveBench 2.0 release: 2026-10-14")
            return "2026-10-14", False
    except Exception as exc:
        logger.debug("LiveBench 2.0 direct probe failed: %s", exc)

    # 2. Check dynamic JS discovery on livebench.ai
    try:
        resp = requests.get(
            "https://livebench.ai/",
            timeout=timeout,
            headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
        )
        if resp.ok:
            js_match = re.search(r'src=[\"\']([^\"\']*main\.[a-f0-9]+\.js)[\"\']', resp.text)
            if js_match:
                js_url = js_match.group(1)
                if js_url.startswith("./"):
                    js_url = "https://livebench.ai/" + js_url[2:]
                elif js_url.startswith("/"):
                    js_url = "https://livebench.ai" + js_url
                elif not js_url.startswith("http"):
                    js_url = "https://livebench.ai/" + js_url

                js_resp = requests.get(
                    js_url,
                    timeout=timeout,
                    headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                )
                if js_resp.ok:
                    if "20261014" in js_resp.text or "2026-10-14" in js_resp.text:
                        logger.info("Discovered LiveBench 2.0 from main JS bundle: 2026-10-14")
                        return "2026-10-14", False
    except Exception as exc:
        logger.debug("LiveBench dynamic release discovery error: %s", exc)

    # 3. Check verified releases in order
    for rel_date in VERIFIED_LIVEBENCH_RELEASES:
        for t_url, _, _ in get_release_urls(rel_date):
            try:
                probe = requests.head(
                    t_url,
                    timeout=timeout,
                    headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                )
                if probe.ok:
                    is_stale = rel_date != VERIFIED_LIVEBENCH_RELEASES[0]
                    return rel_date, is_stale
            except Exception:
                continue

    # Fallback to default verified snapshot with is_stale=True
    return DEFAULT_LIVEBENCH_RELEASE, True


def infer_provider_and_display_name(raw_model: str) -> tuple[str, str, str | None]:
    """Infer provider organization, formatted display name, and configuration from upstream identifier.

    Preserves exact identity without collapsing distinct model versions or evaluation configurations.
    """
    m = raw_model.strip().lower()

    # Provider mapping
    if m.startswith("claude") or "anthropic" in m:
        provider = "Anthropic"
    elif m.startswith(("gpt-", "o1", "o3", "o4", "o5", "o6")) or "openai" in m:
        provider = "OpenAI"
    elif m.startswith(("gemini", "gemma")) or "google" in m:
        provider = "Google DeepMind"
    elif m.startswith("deepseek"):
        provider = "DeepSeek"
    elif m.startswith("qwen"):
        provider = "Alibaba Cloud"
    elif m.startswith(("llama", "meta")):
        provider = "Meta AI"
    elif m.startswith(("mistral", "devstral")):
        provider = "Mistral AI"
    elif m.startswith("phi"):
        provider = "Microsoft"
    elif m.startswith("grok"):
        provider = "xAI"
    elif m.startswith("glm"):
        provider = "Zhipu AI"
    elif m.startswith(("kimi", "moonshot")):
        provider = "Moonshot AI"
    elif m.startswith(("minimax", "mimo")):
        provider = "MiniMax"
    elif m.startswith(("nemotron", "nvidia")):
        provider = "NVIDIA"
    elif m.startswith("arcee"):
        provider = "Arcee AI"
    else:
        provider = "Independent / Open"

    # Extract configuration or effort details
    config_match = re.search(
        r"(?:-|_)(xhigh-effort|high-effort|medium-effort|low-effort|thinking-\d+k|thinking|high|low|medium|xhigh|preview|codex|base|nano|mini)$",
        m,
    )
    configuration = config_match.group(1) if config_match else None

    # Format human-friendly display name while retaining complete version/variant info
    words = raw_model.replace("-", " ").replace("_", " ").split()
    formatted: list[str] = []
    for w in words:
        wl = w.lower()
        if wl in ("gpt", "glm", "amps", "api", "id", "ai", "mit", "r1", "v3", "v4", "v5", "v6"):
            formatted.append(wl.upper())
        elif re.match(r"^[vV]\d+(\.\d+)?$", w):
            formatted.append(w.upper())
        elif re.match(r"^\d+(\.\d+)?$", w):
            formatted.append(w)
        else:
            formatted.append(w.capitalize())

    display_name = " ".join(formatted)
    display_name = re.sub(r"\bGpt\b", "GPT", display_name)
    display_name = re.sub(r"\bGlm\b", "GLM", display_name)
    display_name = re.sub(r"\bDeepseek\b", "DeepSeek", display_name)
    display_name = re.sub(r"\bQwen\b", "Qwen", display_name)
    display_name = re.sub(r"\bMistral\b", "Mistral", display_name)

    return provider, display_name, configuration


class LiveBenchAdapter(SourceAdapter):
    """Adapter for LiveBench benchmark suite across reasoning, agentic coding, and long-horizon tasks."""

    name = "LiveBench"
    source_url = "https://livebench.ai/"
    enabled = True

    def __init__(self, timeout: int = 15, release_date: str | None = None) -> None:
        super().__init__(timeout=timeout)
        self.specified_release = release_date
        self.active_release = release_date or DEFAULT_LIVEBENCH_RELEASE
        self.is_stale = False
        self.generation = "2.0" if (self.active_release in ("2026-10-14", "2026-10-13") or self.active_release >= "2026-09-01") else "1.0"

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch and parse LiveBench evaluation results."""
        if not self.specified_release:
            discovered_date, is_stale = discover_latest_release(timeout=self.timeout)
            self.active_release = discovered_date
            self.is_stale = is_stale

        candidate_urls = get_release_urls(self.active_release)
        table_text: str | None = None
        categories: dict[str, list[str]] | None = None
        generation: str = "2.0" if (self.active_release in ("2026-10-14", "2026-10-13") or self.active_release >= "2026-09-01") else "1.0"

        now_iso = current_iso_timestamp()

        for t_url, c_url, gen in candidate_urls:
            try:
                logger.info("Attempting to fetch LiveBench data from %s", t_url)
                t_resp = requests.get(
                    t_url,
                    timeout=self.timeout,
                    headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                )
                t_resp.raise_for_status()

                c_resp = requests.get(
                    c_url,
                    timeout=self.timeout,
                    headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                )
                c_resp.raise_for_status()

                table_text = t_resp.text
                categories = c_resp.json()
                generation = gen
                self.generation = gen
                break
            except Exception as exc:
                logger.debug("Failed fetch for candidate (%s): %s", t_url, exc)
                continue

        if not table_text:
            logger.warning("Could not fetch remote LiveBench data for release %s. Failing safely.", self.active_release)
            return []

        return self.parse_table_data(
            table_text=table_text,
            categories=categories,
            release_date=self.active_release,
            generation=generation,
        )

    def parse_table_data(
        self,
        table_text: str,
        categories: dict[str, list[str]] | None = None,
        release_date: str = "2026-10-14",
        generation: str = "2.0",
    ) -> list[BenchmarkRecord]:
        """Parse LiveBench CSV table and category mappings into normalized BenchmarkRecords."""
        reader = csv.DictReader(io.StringIO(table_text))
        rows = list(reader)

        if not rows:
            logger.warning("LiveBench CSV returned 0 rows for release %s.", release_date)
            return []

        # Category mapping supporting both LiveBench 2.0 and LiveBench 1.0 categories
        allowed_cat_map = {
            # LiveBench 2.0 categories
            "Long-Horizon": "Agentic",
            "long_horizon": "Agentic",
            "Long Agentic": "Agentic",
            "long_agentic": "Agentic",
            "Browser": "Browser Use",
            "browser": "Browser Use",
            "Research": "Research",
            "research": "Research",
            "Design": "Design",
            "design_layout": "Design",
            "design": "Design",
            "Data Analysis": "Data Analysis",
            "data_analysis": "Data Analysis",
            "Reasoning": "Reasoning",
            "reasoning": "Reasoning",
            "Coding": "Coding",
            "coding": "Coding",
            # LiveBench 1.0 categories
            "Agentic Coding": "Agentic Coding",
            "Mathematics": "Mathematics",
            "Language": "Language",
            "IF": "Instruction Following",
            "Instruction Following": "Instruction Following",
        }

        # If categories dict was not provided (e.g. direct 2.0 CSV with column-level categories), infer from fieldnames
        if not categories:
            categories = {}
            for field in reader.fieldnames or []:
                if field in allowed_cat_map or field.title() in allowed_cat_map:
                    categories[field] = [field]

        records: list[BenchmarkRecord] = []
        is_v2 = (generation == "2.0" or release_date in ("2026-10-14", "2026-10-13") or release_date >= "2026-09-01")
        bench_version = "2.0" if is_v2 else release_date
        suite_prefix = "LiveBench 2.0" if is_v2 else "LiveBench"
        now_iso = current_iso_timestamp()

        for row in rows:
            raw_model = row.get("model", "").strip()
            if not raw_model:
                continue

            canonical_id = raw_model.lower()
            provider, display_name, config = infer_provider_and_display_name(raw_model)

            # Check if upstream source provides an explicit model evaluation date
            raw_eval_date = (
                row.get("evaluation_date")
                or row.get("eval_date")
                or row.get("date")
            )
            eval_date = raw_eval_date.strip() if raw_eval_date and raw_eval_date.strip() else None

            # Resolve model identity against verified registry
            id_match = resolve_model_identity(raw_model, context_source="LiveBench")
            internal_model_id = id_match.matched_id if id_match.confidence >= 0.9 else None
            match_confidence = id_match.confidence if internal_model_id else 0.0
            if internal_model_id and id_match.provider:
                provider = id_match.provider

            # 1. Compute category-level averages
            all_scores: list[float] = []

            for cat_name, task_list in categories.items():
                mapped_cat = allowed_cat_map.get(cat_name) or allowed_cat_map.get(cat_name.title())
                if not mapped_cat:
                    continue

                cat_scores: list[float] = []
                for task in task_list:
                    val_str = row.get(task, "")
                    if val_str:
                        try:
                            score_val = float(val_str)
                            cat_scores.append(score_val)
                            all_scores.append(score_val)
                        except (ValueError, TypeError):
                            pass

                if cat_scores:
                    avg_score = round(sum(cat_scores) / len(cat_scores), 2)
                    rec: BenchmarkRecord = {
                        "model_name": display_name,
                        "canonical_id": canonical_id,
                        "internal_model_id": internal_model_id,
                        "match_confidence": match_confidence,
                        "provider": provider,
                        "benchmark_name": f"{suite_prefix} ({mapped_cat})",
                        "category": mapped_cat,
                        "score": avg_score,
                        "score_unit": "Percentage",
                        "evaluation_date": eval_date,
                        "benchmark_release_date": release_date,
                        "benchmark_version": bench_version,
                        "source": "LiveBench",
                        "source_url": "https://livebench.ai/",
                        "provenance": "Independent Evaluation",
                        "retrieved_at": now_iso,
                        "notes": (
                            f"{suite_prefix} ({release_date}) release category average across {len(cat_scores)} tasks."
                            + (f" Model evaluation date: {eval_date}." if eval_date else " Model evaluation date not reported by source.")
                        ),
                        "configuration": config,
                    }
                    if validate_benchmark_record(rec):
                        records.append(rec)

            # 2. Overall composite score across all evaluated subtasks
            if all_scores:
                overall_score = round(sum(all_scores) / len(all_scores), 2)
                overall_rec: BenchmarkRecord = {
                    "model_name": display_name,
                    "canonical_id": canonical_id,
                    "internal_model_id": internal_model_id,
                    "match_confidence": match_confidence,
                    "provider": provider,
                    "benchmark_name": f"{suite_prefix} Overall Average",
                    "category": "Composite",
                    "score": overall_score,
                    "score_unit": "Percentage",
                    "evaluation_date": eval_date,
                    "benchmark_release_date": release_date,
                    "benchmark_version": bench_version,
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": now_iso,
                    "notes": (
                        f"{suite_prefix} overall arithmetic mean of category averages across {len(all_scores)} verified subtasks from {release_date} release."
                        + (f" Model evaluation date: {eval_date}." if eval_date else " Model evaluation date not reported by source.")
                    ),
                    "configuration": config,
                }
                if validate_benchmark_record(overall_rec):
                    records.append(overall_rec)

        logger.info(
            "Successfully parsed %d benchmark records from LiveBench %s release %s across %d models.",
            len(records),
            generation,
            release_date,
            len(rows),
        )
        return records

    def get_provenance_notice(self) -> dict[str, str]:
        status_desc = f"Active ({self.active_release} Release)"
        if self.is_stale:
            status_desc += " [Fallback/Stale snapshot]"
        return {
            "source": self.name,
            "url": self.source_url,
            "status": status_desc,
            "description": (
                "LiveBench 2.0 contamination-free long-horizon AI benchmark evaluating models on real agentic work "
                "(decks, research reports, bug fixes, apps that run for hours) alongside objective reasoning, coding, and data analysis."
                if self.generation == "2.0"
                else "Contamination-free evaluations with monthly updated problems across reasoning, coding, and mathematics."
            ),
            "attribution": "LiveBench is an open-source evaluation benchmark created by researchers at Abacus.ai, NYU, and UC Berkeley. Released under Apache-2.0.",
            "release_date": self.active_release,
            "generation": self.generation,
        }
