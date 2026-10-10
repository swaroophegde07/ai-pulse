"""
LMCouncil adapter and provenance descriptor.
Documents integration constraints (Disallow: /api/ in robots.txt and absence of public API).
"""

from __future__ import annotations

import logging

from pipeline.benchmarks.base import (
    BenchmarkRecord,
    SourceAdapter,
)

logger = logging.getLogger(__name__)


class LMCouncilAdapter(SourceAdapter):
    """Adapter for LMCouncil benchmarks portal."""

    name = "LM Council"
    source_url = "https://lmcouncil.ai/benchmarks"
    enabled = False  # Automated scraping prohibited by robots.txt and lack of public API

    def __init__(self, timeout: int = 10) -> None:
        super().__init__(timeout=timeout)

    def fetch(self) -> list[BenchmarkRecord]:
        """Return empty list and log transparent architectural explanation."""
        logger.info(
            "LMCouncil automated API scraping is disabled: lmcouncil.ai/robots.txt disallows /api/ "
            "and does not offer a public documented REST endpoint. "
            "Attribution and external portal links are provided in the UI instead."
        )
        return []

    def get_provenance_notice(self) -> dict[str, str]:
        return {
            "source": self.name,
            "url": self.source_url,
            "status": "External Reference Only (No Public API)",
            "description": "Comparative benchmark rankings across reasoning, coding, and mathematical benchmarks.",
            "attribution": "LM Council (lmcouncil.ai). Automated ingestion is disabled in compliance with robots.txt directives and lack of a public API.",
        }
