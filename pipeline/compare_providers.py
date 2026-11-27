"""Compare Gemini and the Groq fallback on live articles without touching feed.json.

Usage (from the repo root):
  python -m pipeline.compare_providers --limit 5 --output /tmp/provider-comparison.md

Requires GEMINI_API_KEY and/or GROQ_API_KEY in the environment. Providers without a
key are skipped. Each article costs at most one Gemini request and one Groq request
(plus one retry on transient errors), so keep --limit small on free tiers.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pipeline.scraper import fetch_all_sources
from pipeline.sources import SOURCES
from pipeline.summarizer import GroqSummarizer, _get_client, summarize_article

logger = logging.getLogger(__name__)


def _run_provider(name: str, article: dict, client, groq: GroqSummarizer | None) -> dict:
    started = time.perf_counter()
    if name == "gemini":
        result = summarize_article(article, client=client, retries=0)
    else:
        result = summarize_article(article, client=None, fallback=groq)
    result["_latency_s"] = time.perf_counter() - started
    return result


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _render_report(rows: list[tuple[dict, dict[str, dict]]], providers: list[str]) -> str:
    lines = ["# Summarizer provider comparison", ""]
    lines.append(f"Providers: {', '.join(providers)} · Articles: {len(rows)}")
    lines.append("")

    totals = {p: {"generated": 0, "latency": 0.0} for p in providers}
    for _, results in rows:
        for provider, res in results.items():
            totals[provider]["latency"] += res["_latency_s"]
            if res.get("summary_status") == "generated" and res.get("summary_provider") == provider:
                totals[provider]["generated"] += 1

    lines += ["| Provider | Generated | Avg latency |", "|---|---|---|"]
    for provider in providers:
        avg = totals[provider]["latency"] / max(1, len(rows))
        lines.append(f"| {provider} | {totals[provider]['generated']}/{len(rows)} | {avg:.1f}s |")
    lines.append("")

    for index, (article, results) in enumerate(rows, start=1):
        lines.append(f"## {index}. {article.get('title', '(untitled)')}")
        lines.append(f"{article.get('company', '')} · {article.get('url', '')}")
        lines.append("")
        lines += ["| Provider | Status | Model | Category | Impact | Latency |", "|---|---|---|---|---|---|"]
        for provider, res in results.items():
            lines.append(
                "| {p} | {s} | {m} | {c} | {i} | {l:.1f}s |".format(
                    p=provider,
                    s=res.get("summary_status", ""),
                    m=_escape_cell(str(res.get("summary_model", "-"))),
                    c=res.get("category", ""),
                    i=res.get("impact_score", "-"),
                    l=res["_latency_s"],
                )
            )
        lines.append("")
        for provider, res in results.items():
            lines.append(f"**{provider}:** {res.get('summary', '')}")
            for point in res.get("key_points") or []:
                lines.append(f"- {point}")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=5, help="Number of live articles to compare (default 5).")
    parser.add_argument("--source", help="Only use sources whose name contains this text (case-insensitive).")
    parser.add_argument("--output", type=Path, help="Write the markdown report here instead of stdout.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    providers: list[str] = []
    client = None
    try:
        client = _get_client()
        providers.append("gemini")
    except ValueError:
        logger.warning("GEMINI_API_KEY not set; skipping Gemini.")
    groq = GroqSummarizer.from_env()
    if groq is not None:
        groq.max_requests = max(groq.max_requests, args.limit * (groq.retries + 1))
        providers.append("groq")
    else:
        logger.warning("GROQ_API_KEY not set; skipping Groq.")
    if not providers:
        logger.error("Set GEMINI_API_KEY and/or GROQ_API_KEY to run the comparison.")
        return 2

    sources = [s for s in SOURCES if s.get("enabled", True)]
    if args.source:
        sources = [s for s in sources if args.source.lower() in s["name"].lower()]
    articles = [a for a in fetch_all_sources(sources) if len(a.get("text") or "") >= 200][: max(0, args.limit)]
    if not articles:
        logger.error("No live articles with enough text were found.")
        return 1

    rows = []
    for article in articles:
        results = {p: _run_provider(p, article, client, groq) for p in providers}
        rows.append((article, results))

    report = _render_report(rows, providers)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
        logger.info("Wrote comparison report to %s", args.output)
    else:
        sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
