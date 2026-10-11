"""
LiveBench adapter for contamination-free LLM benchmark results across reasoning, coding, and math.

Discovers latest verified monthly releases and dynamically parses model evaluations
without restrictive allowlists while preserving exact model identities and configurations.
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

# Ordered list of officially verified releases from LiveBench (newest first)
VERIFIED_LIVEBENCH_RELEASES = [
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


def discover_latest_release(timeout: int = 10) -> tuple[str, bool]:
    """Discover the latest available LiveBench release.

    Returns:
        (release_date, is_fallback_or_stale)
    """
    try:
        resp = requests.get(
            "https://livebench.ai/",
            timeout=timeout,
            headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
        )
        if resp.ok:
            # Find main js file
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
                    dates_match = re.search(r'(?:const|var|let)\s+pe\s*=\s*\[([^\]]+)\]', js_resp.text)
                    if dates_match:
                        raw_dates = re.findall(r'202\d[_-]\d{2}[_-]\d{2}', dates_match.group(1))
                        valid_dates = sorted(set(d.replace("_", "-") for d in raw_dates))
                        if valid_dates:
                            latest_candidate = valid_dates[-1]
                            t_url = f"https://livebench.ai/table_{latest_candidate.replace('-', '_')}.csv"
                            probe = requests.head(
                                t_url,
                                timeout=timeout,
                                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
                            )
                            if probe.ok:
                                logger.info("Discovered latest LiveBench release: %s", latest_candidate)
                                return latest_candidate, False
    except Exception as exc:
        logger.debug("LiveBench dynamic release discovery error: %s", exc)

    # Check verified releases in order
    for rel_date in VERIFIED_LIVEBENCH_RELEASES:
        t_url = f"https://livebench.ai/table_{rel_date.replace('-', '_')}.csv"
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
    """Adapter for LiveBench monthly contamination-free benchmark suite."""

    name = "LiveBench"
    source_url = "https://livebench.ai/"
    enabled = True

    def __init__(self, timeout: int = 15, release_date: str | None = None) -> None:
        super().__init__(timeout=timeout)
        self.specified_release = release_date
        self.active_release = release_date or DEFAULT_LIVEBENCH_RELEASE
        self.is_stale = False

    def fetch(self) -> list[BenchmarkRecord]:
        """Fetch and parse LiveBench evaluation results."""
        if not self.specified_release:
            discovered_date, is_stale = discover_latest_release(timeout=self.timeout)
            self.active_release = discovered_date
            self.is_stale = is_stale

        table_url = f"https://livebench.ai/table_{self.active_release.replace('-', '_')}.csv"
        categories_url = f"https://livebench.ai/categories_{self.active_release.replace('-', '_')}.json"

        logger.info("Fetching LiveBench data for release %s from %s", self.active_release, table_url)
        now_iso = current_iso_timestamp()

        try:
            table_resp = requests.get(
                table_url,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            table_resp.raise_for_status()

            cat_resp = requests.get(
                categories_url,
                timeout=self.timeout,
                headers={"User-Agent": "AI-Pulse/1.0 (+https://github.com/swaroophegde07/ai-pulse)"},
            )
            cat_resp.raise_for_status()
            categories: dict[str, list[str]] = cat_resp.json()
        except Exception as exc:
            logger.warning("LiveBench remote fetch failed for release %s: %s", self.active_release, exc)
            return []

        csv_text = table_resp.text
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if not rows:
            logger.warning("LiveBench CSV returned 0 rows for release %s.", self.active_release)
            return []

        # Category mapping into allowed standard categories
        allowed_cat_map = {
            "Reasoning": "Reasoning",
            "Coding": "Coding",
            "Agentic Coding": "Agentic Coding",
            "Mathematics": "Mathematics",
            "Data Analysis": "Data Analysis",
            "Language": "Language",
            "IF": "Instruction Following",
            "Instruction Following": "Instruction Following",
        }

        records: list[BenchmarkRecord] = []

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
                mapped_cat = allowed_cat_map.get(cat_name)
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
                        "benchmark_name": f"LiveBench ({mapped_cat})",
                        "category": mapped_cat,
                        "score": avg_score,
                        "score_unit": "Percentage",
                        "evaluation_date": eval_date,
                        "benchmark_release_date": self.active_release,
                        "benchmark_version": self.active_release,
                        "source": "LiveBench",
                        "source_url": "https://livebench.ai/",
                        "provenance": "Independent Evaluation",
                        "retrieved_at": now_iso,
                        "notes": (
                            f"LiveBench {self.active_release} release category average across {len(cat_scores)} tasks."
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
                    "benchmark_name": "LiveBench Overall Average",
                    "category": "Composite",
                    "score": overall_score,
                    "score_unit": "Percentage",
                    "evaluation_date": eval_date,
                    "benchmark_release_date": self.active_release,
                    "benchmark_version": self.active_release,
                    "source": "LiveBench",
                    "source_url": "https://livebench.ai/",
                    "provenance": "Independent Evaluation",
                    "retrieved_at": now_iso,
                    "notes": (
                        f"LiveBench overall average across {len(all_scores)} verified subtasks from {self.active_release} release."
                        + (f" Model evaluation date: {eval_date}." if eval_date else " Model evaluation date not reported by source.")
                    ),
                    "configuration": config,
                }
                if validate_benchmark_record(overall_rec):
                    records.append(overall_rec)

        logger.info(
            "Successfully parsed %d benchmark records from LiveBench release %s across %d models.",
            len(records),
            self.active_release,
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
            "description": "Contamination-free evaluations with monthly updated problems across reasoning, coding, and mathematics.",
            "attribution": "LiveBench is an open-source evaluation benchmark created by researchers at Abacus.ai, NYU, and UC Berkeley. Released under Apache-2.0.",
            "release_date": self.active_release,
        }
