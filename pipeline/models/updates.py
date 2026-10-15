"""
Official AI Model Updates & Release Notes Ingestion Engine.

Fetches, extracts, classifies, and archives model updates from official AI provider
sources (OpenAI, Anthropic, Google DeepMind, Meta AI, DeepSeek, Mistral, xAI, Cohere, etc.).

Key guarantees:
- Zero mandatory Gemini API calls (deterministic parsing, quota-safe).
- Source-specific HTML & RSS extraction that discards scripts, styles, navigation,
  footers, cookie banners, and page chrome.
- Deterministic 16-character IDs to guarantee idempotent scheduled runs.
- Strict model identity resolution: exact matches only via `pipeline/models/identity.py`,
  leaving `model_id=None` when ambiguous to avoid version collisions.
- Transparent distinction between exact model updates and model-family updates.
- Transparent distinction between provider claims and independent benchmark data.
- Domain allowlisting: guarantees external links point to official provider domains
  and rejects social media or unrelated navigation links.
- Distinction between publication dates and future effective/deprecation dates.
- Non-destructive history preservation: source failures or empty responses never
  destroy historical update records.
"""

from __future__ import annotations

import email.utils
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import feedparser
import requests

from pipeline.models.identity import (
    VERIFIED_MODEL_REGISTRY,
    MatchResult,
    MatchType,
    resolve_model_identity,
)
from pipeline.models.sources import (
    ALLOWED_SCOPES,
    ALLOWED_UPDATE_TYPES,
    ModelUpdateSource,
    get_active_model_update_sources,
    load_model_update_sources,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_UPDATES_PATH = REPO_ROOT / "src" / "data" / "model-updates.json"

# Tracking query parameters to strip during URL normalization
_TRACKING_PARAMS = frozenset([
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
    "source",
    "mc_cid",
    "mc_eid",
])

# Official provider domain allowlist to prevent linking to arbitrary/social destinations
OFFICIAL_PROVIDER_DOMAINS: dict[str, tuple[str, ...]] = {
    "openai": ("openai.com", "chatgpt.com", "help.openai.com", "developers.openai.com"),
    "anthropic": ("anthropic.com", "claude.ai", "docs.anthropic.com", "platform.claude.com"),
    "google": ("google.com", "deepmind.google", "ai.google.dev", "blog.google", "cloud.google.com"),
    "meta": ("meta.com", "ai.meta.com", "about.fb.com", "llama.meta.com"),
    "deepseek": ("deepseek.com", "api-docs.deepseek.com"),
    "mistral": ("mistral.ai", "docs.mistral.ai", "console.mistral.ai"),
    "cohere": ("cohere.com", "docs.cohere.com"),
    "xai": ("x.ai", "docs.x.ai", "grok.com"),
    "microsoft": ("microsoft.com", "azure.microsoft.com", "techcommunity.microsoft.com"),
    "huggingface": ("huggingface.co",),
}

DISALLOWED_DOMAINS = frozenset([
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "linkedin.com",
    "reddit.com",
    "pinterest.com",
    "unverifiedtracker.com",
])

DISALLOWED_PATHS = frozenset([
    "/privacy",
    "/privacy-policy",
    "/terms",
    "/terms-of-service",
    "/terms-of-use",
    "/legal",
    "/legal-notice",
    "/login",
    "/signin",
    "/sign-in",
    "/signup",
    "/sign-up",
    "/careers",
    "/jobs",
    "/cookies",
    "/cookie-policy",
    "/contact",
    "/contact-us",
])

CHROME_TITLE_PATTERNS = [
    re.compile(r"\bterms of (?:service|use)\b", re.IGNORECASE),
    re.compile(r"\bprivacy policy\b", re.IGNORECASE),
    re.compile(r"\blegal notice\b", re.IGNORECASE),
    re.compile(r"\bprivacy choices\b", re.IGNORECASE),
    re.compile(r"\bcookie settings\b", re.IGNORECASE),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bsign in\b", re.IGNORECASE),
    re.compile(r"\blogin\b", re.IGNORECASE),
    re.compile(r"\bskip to (?:main )?content\b", re.IGNORECASE),
    re.compile(r"\btable of contents\b", re.IGNORECASE),
    re.compile(r"\bnavigation menu\b", re.IGNORECASE),
    re.compile(r"\bcareers\b", re.IGNORECASE),
    re.compile(r"^Terms of servicePrivacy policy", re.IGNORECASE),
    re.compile(r"(?:webpackChunk|function\s*\(|Object\.defineProperty|void 0|var |const |window\.|document\.)"),
]

FUTURE_DEPRECATION_PATTERN = re.compile(
    r"(?:will be (?:retired|shutdown|deprecated|sunset)(?:\s+on)?|retirement date|sunset date|effective (?:date|on)|(?:deprecated|retired|shutdown|sunset)\s+(?:on|as of))\s*(?::\s*|\s+)?([A-Za-z]+ \d{1,2}, \d{4}|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


class ModelUpdateRecord(TypedDict, total=False):
    id: str
    provider_id: str
    provider_name: str
    title: str
    summary: str
    update_type: str
    scope: str
    model_name: str | None
    model_family: str | None
    model_id: str | None
    published_at: str | None
    effective_at: str | None
    detected_at: str
    source_type: str
    source_url: str
    verification_status: str
    source_id: str
    provider_claims: list[str]


REQUIRED_UPDATE_FIELDS = frozenset([
    "id",
    "provider_id",
    "provider_name",
    "title",
    "summary",
    "update_type",
    "scope",
    "detected_at",
    "source_type",
    "source_url",
    "verification_status",
    "source_id",
])


# ---------------------------------------------------------------------------
# URL Normalization & Validation
# ---------------------------------------------------------------------------


def is_safe_official_url(url: str, provider_id: str = "") -> bool:
    """Validate that a URL is a secure HTTPS link belonging to an approved official provider domain."""
    if not url or not isinstance(url, str):
        return False
    clean = url.strip()
    try:
        parsed = urlparse(clean)
    except Exception:
        return False

    # Require HTTPS (or HTTP only for local test fixtures)
    if parsed.scheme.lower() not in ("https", "http"):
        return False

    host = parsed.netloc.lower().split(":")[0]
    if not host or "." not in host:
        return False

    # Disallow social media, generic tracking aggregators, or video platforms as update source URLs
    for dis in DISALLOWED_DOMAINS:
        if host == dis or host.endswith("." + dis):
            return False

    # Disallow utility, legal, login, or cookie paths
    clean_path = parsed.path.rstrip("/").lower()
    if clean_path in DISALLOWED_PATHS:
        return False

    # If provider specified, check against provider's official domain allowlist
    pid = provider_id.lower().strip()
    allowed_domains = OFFICIAL_PROVIDER_DOMAINS.get(pid)
    if allowed_domains:
        is_allowed = any(host == d or host.endswith("." + d) for d in allowed_domains)
        if not is_allowed:
            return False

    return True


def normalize_update_url(raw_url: str) -> str:
    """Normalize a source URL by stripping tracking parameters, normalizing host,
    and removing redundant fragments."""
    if not raw_url:
        return ""
    url = raw_url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Strip default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Filter tracking query parameters
    query_parts = []
    if parsed.query:
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query_parts = [(k, v) for k, v in pairs if k.lower() not in _TRACKING_PARAMS]
    clean_query = urlencode(query_parts) if query_parts else ""

    path = parsed.path or "/"
    # Collapse duplicate slashes in path
    path = re.sub(r"/{2,}", "/", path)
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, parsed.params, clean_query, ""))


def generate_update_id(source_url: str, provider_id: str, title: str) -> str:
    """Generate a deterministic, stable 16-character hex identifier for an update record."""
    norm_url = normalize_update_url(source_url)
    norm_prov = provider_id.strip().lower()
    norm_title = re.sub(r"\s+", " ", title.strip().lower())
    payload = f"{norm_url}::{norm_prov}::{norm_title}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Page Chrome & Irrelevant Content Detection
# ---------------------------------------------------------------------------


def is_irrelevant_or_chrome(title: str, text: str = "") -> bool:
    """Detect whether a title or summary is page chrome, navigation, legal text, or JavaScript."""
    t_clean = title.strip()
    if not t_clean or len(t_clean) < 4:
        return True

    # Check for concatenated uppercase words without spaces (e.g. 'Terms of servicePrivacy policyLegal notice')
    if re.search(r"[a-z][A-Z][a-z]+[A-Z]", t_clean) and any(
        kw in t_clean.lower() for kw in ("privacy", "terms", "legal", "notice", "policy")
    ):
        return True

    # Check against known chrome patterns
    for pat in CHROME_TITLE_PATTERNS:
        if pat.search(t_clean):
            return True

    # Excessively long title with multiple dates or headings (concatenation bug)
    if len(t_clean) > 220:
        return True

    # Check text for minified script or hydration dump
    if re.search(r"(?:function\s*\(|var\s+[a-zA-Z_$]|window\.__|self\.__|webpackJsonp)", text):
        return True

    return False


# ---------------------------------------------------------------------------
# Date Extraction & Validation (Zero-fabrication, future-effective separation)
# ---------------------------------------------------------------------------


def parse_date_string(raw: Any) -> str | None:
    """Parse date from string or struct_time into ISO 8601 (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
    Returns None if date cannot be determined reliably. Never invent dates."""
    if not raw:
        return None

    if isinstance(raw, (list, tuple, time.struct_time)):
        try:
            return time.strftime("%Y-%m-%dT%H:%M:%SZ", raw)
        except Exception:
            return None

    if isinstance(raw, datetime):
        return raw.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    # Check ISO formats (e.g. 2025-01-31, 2025-01-31T12:00:00Z)
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})(?:T(\d{2}:\d{2}:\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b", text)
    if iso_match:
        date_part = iso_match.group(1)
        time_part = iso_match.group(2)
        if time_part:
            return f"{date_part}T{time_part}Z"
        return date_part

    # Try RFC 2822 / 822 format (e.g. 'Thu, 31 Jan 2025 15:00:00 GMT')
    try:
        parsed_tuple = email.utils.parsedate_to_datetime(text)
        if parsed_tuple:
            return parsed_tuple.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    # Try common formats like 'February 5, 2025', 'Feb 5, 2025', '5 Feb 2025'
    common_formats = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%m/%d/%Y",
    ]
    for fmt in common_formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Attempt regex for Month Day, Year
    match = re.search(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(\d{4})\b", text, re.IGNORECASE)
    if match:
        month_str, day_str, year_str = match.groups()
        clean_str = f"{month_str} {day_str} {year_str}"
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                dt = datetime.strptime(clean_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return None


def extract_dates(text: str, current_date_candidate: str | None = None) -> tuple[str | None, str | None]:
    """Extract published_at and optional effective_at (future deprecation date).

    Returns:
      (published_at, effective_at)
    """
    now = datetime.now(UTC)
    # Allow 1-day tolerance for international timezone offsets
    now_date_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    effective_at: str | None = None
    dep_match = FUTURE_DEPRECATION_PATTERN.search(text)
    if dep_match:
        parsed_eff = parse_date_string(dep_match.group(1))
        if parsed_eff:
            effective_at = parsed_eff

    published_at: str | None = current_date_candidate

    # If candidate published_at is in the future (> tomorrow), it is an effective date, NOT published date!
    if published_at:
        try:
            pub_date_part = published_at[:10]
            if pub_date_part > now_date_str:
                if not effective_at:
                    effective_at = published_at
                published_at = None
        except Exception:
            pass

    return published_at, effective_at


# ---------------------------------------------------------------------------
# Classification & Identity Matching
# ---------------------------------------------------------------------------


_PRICING_PATTERNS = re.compile(
    r"\b(pricing|price reduction|price cut|cost reduction|cheaper|per 1m tokens|tokens priced|input price|output price)\b",
    re.IGNORECASE,
)
_DEPRECATION_PATTERNS = re.compile(
    r"\b(deprecat\w*|sunset\w*|shutting down|shutdown|end of life|eol|discontinued|retir\w*)\b",
    re.IGNORECASE,
)
_CONTEXT_PATTERNS = re.compile(
    r"\b(context window|context length|\d+[km]\s*context|token window)\b",
    re.IGNORECASE,
)
_MODALITY_PATTERNS = re.compile(
    r"\b(multimodal\w*|vision|audio|speech|voice input|image input|visual understanding)\b",
    re.IGNORECASE,
)
_API_PATTERNS = re.compile(
    r"\b(api availability|api release|sdk release|batch api|structured outputs|function calling|rest api|endpoint|rate limits?)\b",
    re.IGNORECASE,
)
_RESEARCH_PATTERNS = re.compile(
    r"\b(paper\w*|research\w*|technical report|arxiv|open weights release|benchmark methodology)\b",
    re.IGNORECASE,
)
_PRODUCT_PATTERNS = re.compile(
    r"\b(canvas|voice mode|mobile app|desktop app|web interface|workspace|consumer app)\b",
    re.IGNORECASE,
)
_CAPABILITY_PATTERNS = re.compile(
    r"\b(extended thinking|reasoning effort|computer use|coding capabilities|fine-tuning|tool use)\b",
    re.IGNORECASE,
)
_RELEASE_PATTERNS = re.compile(
    r"\b(releas\w*|launch\w*|announc\w*|introducing|now available|preview|checkpoint|unveil\w*)\b",
    re.IGNORECASE,
)

# Known marketing / provider claim phrases
_CLAIM_PATTERNS = [
    re.compile(r"\b(?:state-of-the-art|sota)\b", re.IGNORECASE),
    re.compile(r"\b(?:outperforms|surpasses|beats)\s+[a-z0-9\-.]+\b", re.IGNORECASE),
    re.compile(r"\b(?:industry-leading|best-in-class|frontier-defining)\b", re.IGNORECASE),
    re.compile(r"\b(?:\d+x\s+faster|\d+%\s+cheaper)\b", re.IGNORECASE),
    re.compile(r"\b(?:unmatched reasoning|superior performance)\b", re.IGNORECASE),
]


def classify_update_type(title: str, text: str = "") -> str:
    """Classify the update type based on title and content text."""
    combined = f"{title} {text}"

    if _DEPRECATION_PATTERNS.search(combined):
        return "deprecation"
    if _PRICING_PATTERNS.search(combined):
        return "pricing_change"
    if _CONTEXT_PATTERNS.search(combined):
        return "context_window_change"
    if _MODALITY_PATTERNS.search(combined):
        return "modality_change"
    if _CAPABILITY_PATTERNS.search(combined):
        return "capability_change"
    if _API_PATTERNS.search(combined):
        return "api_availability"
    if _RESEARCH_PATTERNS.search(combined):
        return "research_release"
    if _PRODUCT_PATTERNS.search(combined):
        return "product_update"
    if _RELEASE_PATTERNS.search(combined):
        return "model_release"

    # Default fallback
    return "model_release" if "model" in combined.lower() else "product_update"


def classify_scope(
    title: str,
    text: str = "",
    model_id: str | None = None,
    model_name: str | None = None,
    model_family: str | None = None,
) -> str:
    """Classify update scope into 'model', 'model_family', 'api', 'product', or 'research'.

    Guarantees:
    - Ambiguous family updates where model_id is None have scope='model_family'.
    - Exact model version updates have scope='model'.
    """
    combined = f"{title} {text}"

    # Exact model identified
    if model_id:
        return "model"

    # Model family identified without specific version
    if model_family:
        return "model_family"

    if re.search(r"\b(?:claude\s*3(?:\.5|\.7)?|gpt-4|gemini\s*2|llama\s*3|phi)\s+family\b", combined, re.IGNORECASE):
        return "model_family"

    if _RESEARCH_PATTERNS.search(combined) and not _API_PATTERNS.search(combined):
        return "research"

    if _PRODUCT_PATTERNS.search(combined) and not _API_PATTERNS.search(combined):
        return "product"

    if _API_PATTERNS.search(combined) or "api" in combined.lower():
        return "api"

    return "model" if model_name else "product"


def extract_provider_claims(title: str, text: str = "") -> list[str]:
    """Extract and tag marketing or provider-reported capability claims.
    Distinguishes provider claims from independent benchmark evidence."""
    combined = f"{title} {text}"
    claims: list[str] = []
    for pat in _CLAIM_PATTERNS:
        match = pat.search(combined)
        if match:
            claims.append(f"Provider claim: '{match.group(0)}' (distinguished from independent benchmark evaluations)")
    return claims


# Explicit model version patterns to avoid loose substring matches
_EXPLICIT_MODEL_PATTERNS = [
    # OpenAI
    (r"\bo3-mini\b", "OpenAI o3-mini", "o-series", "openai/o3-mini"),
    (r"\bo1\b", "OpenAI o1", "o-series", "openai/o1"),
    (r"\bgpt-4o-mini\b", "GPT-4o mini", "GPT-4", "openai/gpt-4o-mini"),
    (r"\bgpt-4o\b", "GPT-4o (Omni)", "GPT-4", "openai/gpt-4o"),
    (r"\bgpt-6\.1-sol\b", "GPT-6.1 Sol", "GPT-6", "openai/gpt-6.1-sol"),
    (r"\bgpt-6-astra\b", "GPT-6 Astra", "GPT-6", "openai/gpt-6-astra"),
    (r"\bgpt-5\.5\b", "GPT-5.5", "GPT-5", "openai/gpt-5.5"),

    # Anthropic
    (r"\bclaude[\s\-_]*3[\s\-_.]*7[\s\-_]*sonnet\b", "Claude 3.7 Sonnet", "Claude 3.7", "anthropic/claude-3-7-sonnet"),
    (r"\bclaude[\s\-_]*3[\s\-_.]*5[\s\-_]*sonnet\b", "Claude 3.5 Sonnet", "Claude 3.5", "anthropic/claude-3-5-sonnet"),
    (r"\bclaude[\s\-_]*3[\s\-_.]*5[\s\-_]*haiku\b", "Claude 3.5 Haiku", "Claude 3.5", "anthropic/claude-3-5-haiku"),
    (r"\bclaude[\s\-_]*sonnet[\s\-_.]*5[\s\-_.]*5\b", "Claude Sonnet 5.5", "Claude 5", "anthropic/claude-sonnet-5.5"),
    (r"\bclaude[\s\-_]*opus[\s\-_.]*5[\s\-_.]*5\b", "Claude Opus 5.5", "Claude 5", "anthropic/claude-opus-5.5"),

    # Google DeepMind
    (r"\bgemini[\s\-_]*2[\s\-_.]*0[\s\-_]*flash\b", "Gemini 2.0 Flash", "Gemini 2", "google/gemini-2-0-flash"),
    (r"\bgemini[\s\-_]*2[\s\-_.]*0[\s\-_]*pro(?:\s*experimental)?\b", "Gemini 2.0 Pro Experimental", "Gemini 2", "google/gemini-2-0-pro-exp"),
    (r"\bgemini[\s\-_]*3[\s\-_.]*5[\s\-_]*flash\b", "Gemini 3.5 Flash", "Gemini 3", "google/gemini-3.5-flash"),

    # DeepSeek
    (r"\bdeepseek[\s\-_]*r1\b", "DeepSeek-R1", "DeepSeek-R1", "deepseek/deepseek-r1"),
    (r"\bdeepseek[\s\-_]*v3\b", "DeepSeek-V3", "DeepSeek-V3", "deepseek/deepseek-v3"),
    (r"\bdeepseek[\s\-_]*v4[\s\-_.]*1[\s\-_]*flash\b", "DeepSeek-V4.1 Flash", "DeepSeek-V4", "deepseek/deepseek-v4.1-flash"),

    # Meta
    (r"\bllama[\s\-_]*3[\s\-_.]*3[\s\-_]*70b(?:\s*instruct)?\b", "Llama 3.3 70B Instruct", "Llama 3.3", "meta/llama-3-3-70b-instruct"),
    (r"\bllama[\s\-_]*3[\s\-_.]*1[\s\-_]*405b(?:\s*instruct)?\b", "Llama 3.1 405B Instruct", "Llama 3.1", "meta/llama-3-1-405b-instruct"),

    # Mistral
    (r"\bmistral[\s\-_]*large[\s\-_]*2411\b", "Mistral Large 2", "Mistral Large", "mistral/mistral-large-2411"),
    (r"\bmistral[\s\-_]*large[\s\-_]*4(?:\.0)?\b", "Mistral Large 4", "Mistral Large", "mistralai/mistral-large-4-0"),
    (r"\bdevstral[\s\-_]*2512\b", "Devstral 2", "Devstral", "mistralai/devstral-2512"),

    # Microsoft
    (r"\bphi[\s\-_]*4\b", "Phi-4", "Phi", "microsoft/phi-4"),
]

# Broad family references that should NOT match a specific version (prevents version collision)
_AMBIGUOUS_FAMILY_PATTERNS = [
    (r"\bclaude\s*3(?:\.5)?\b", "Claude 3.5", "Claude 3.5"),
    (r"\bclaude\b", "Claude", "Claude"),
    (r"\bgpt-4\b", "GPT-4", "GPT-4"),
    (r"\bgpt-3(?:\.5)?\b", "GPT-3.5", "GPT-3.5"),
    (r"\bgemini\s*1(?:\.5)?\b", "Gemini 1.5", "Gemini 1.5"),
    (r"\bgemini\b", "Gemini", "Gemini"),
    (r"\bllama\s*3(?:\.2)?\b", "Llama 3", "Llama 3"),
    (r"\bllama\b", "Llama", "Llama"),
    (r"\bdeepseek\b", "DeepSeek", "DeepSeek"),
    (r"\bmistral\b", "Mistral", "Mistral"),
    (r"\bcommand\s*r\b", "Command R", "Command"),
    (r"\bgrok\b", "Grok", "Grok"),
]


def resolve_update_model(
    title: str,
    text: str = "",
    provider_hint: str = "",
) -> tuple[str | None, str | None, str | None]:
    """Resolve model identity from update text.

    Returns:
      (model_id, model_name, model_family)

    Strict matching rules:
    - Never uses loose substrings (e.g. 'Claude' or 'GPT' alone never map to a specific model ID).
    - Checks for explicit version identifiers against verified models.
    - If ambiguous or family-only, sets `model_id=None` and preserves family/name.
    """
    combined = f"{title} {text}"

    # 1. Check explicit version patterns first
    for pattern, canonical_name, family, expected_id in _EXPLICIT_MODEL_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            match_res = resolve_model_identity(expected_id, context_source="update_extractor")
            if match_res.matched_id and not match_res.is_ambiguous:
                return match_res.matched_id, canonical_name, family

    # 2. Check ambiguous/family-level references (leave model_id=None to prevent collision)
    for pattern, name, family in _AMBIGUOUS_FAMILY_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return None, name, family

    return None, None, None


# ---------------------------------------------------------------------------
# HTML Changelog Parser (Hardened against navigation, footers, and scripts)
# ---------------------------------------------------------------------------


class ChangelogHTMLParser(HTMLParser):
    """Resilient, hardened HTML parser for official changelog pages.
    Guarantees page chrome, navigation, footers, scripts, and social links are rejected."""

    IGNORED_TAGS = frozenset([
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "svg",
        "button",
        "form",
        "meta",
        "link",
    ])

    def __init__(self, base_url: str, provider_id: str = ""):
        super().__init__()
        self.base_url = base_url
        self.provider_id = provider_id
        self.entries: list[dict[str, Any]] = []
        self._ignored_depth = 0
        self._current_tag = ""
        self._heading_text: list[str] = []
        self._current_text: list[str] = []
        self._current_title = ""
        self._current_date: str | None = None
        self._current_url = base_url
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self._current_tag = tag
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        # Check ignored elements
        if tag in self.IGNORED_TAGS or "nav" in attr_dict.get("class", "").lower() or "footer" in attr_dict.get("class", "").lower():
            self._ignored_depth += 1
            return

        if self._ignored_depth > 0:
            return

        if tag in ("h1", "h2", "h3", "h4"):
            # If we already have a title and text, flush previous entry before starting new section
            if self._current_title and "".join(self._current_text).strip():
                self._flush_entry()
            self._in_heading = True
            self._heading_text = []

        elif tag in ("article", "section"):
            if self._current_title and "".join(self._current_text).strip():
                self._flush_entry()

        elif tag == "a" and "href" in attr_dict:
            href = attr_dict["href"]
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                resolved = urljoin(self.base_url, href)
                # Only associate URL if it is a secure official provider link
                if is_safe_official_url(resolved, self.provider_id):
                    self._current_url = resolved

    def handle_endtag(self, tag: str):
        if tag in self.IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return

        if self._ignored_depth > 0:
            return

        if tag in ("h1", "h2", "h3", "h4"):
            self._in_heading = False
            raw_h = "".join(self._heading_text).strip()
            self._heading_text = []

            if raw_h:
                parsed_dt = parse_date_string(raw_h)
                if parsed_dt:
                    self._current_date = parsed_dt
                else:
                    if not is_irrelevant_or_chrome(raw_h):
                        self._current_title = raw_h

        elif tag in ("article", "section"):
            if self._current_title or "".join(self._current_text).strip():
                self._flush_entry()

    def handle_data(self, data: str):
        if self._ignored_depth > 0:
            return

        cleaned = data.strip()
        if not cleaned:
            return

        if self._in_heading:
            self._heading_text.append(data)
        else:
            # Inline date discovery if date not yet established
            if not self._current_date:
                dt = parse_date_string(cleaned)
                if dt:
                    self._current_date = dt
            self._current_text.append(data)

    def _flush_entry(self):
        text_body = " ".join("".join(self._current_text).split()).strip()
        title = self._current_title or (text_body[:80] if len(text_body) > 15 else "")

        # Guard against empty, chrome, or irrelevant text
        if title and not is_irrelevant_or_chrome(title, text_body):
            # Extract published_at vs effective_at
            pub_date, eff_date = extract_dates(text_body, self._current_date)

            summary = text_body[:300] if text_body else title
            # Validate URL before storing
            url = self._current_url
            if not is_safe_official_url(url, self.provider_id):
                url = self.base_url

            self.entries.append({
                "title": title,
                "summary": summary,
                "published_at": pub_date,
                "effective_at": eff_date,
                "url": url,
            })

        # Reset per-entry state
        self._current_title = ""
        self._current_date = None
        self._current_text = []
        self._current_url = self.base_url

    def close(self):
        super().close()
        if self._current_title or self._current_text:
            self._flush_entry()


def parse_html_changelog(
    html_content: str,
    source: ModelUpdateSource,
    base_url: str,
) -> list[ModelUpdateRecord]:
    """Parse an HTML changelog into structured ModelUpdateRecord items with boundary and safety checks."""
    provider_id = source["provider_id"]
    parser = ChangelogHTMLParser(base_url, provider_id=provider_id)
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as exc:
        logger.warning("Error parsing HTML changelog for %s: %s", source.get("id"), exc)

    records: list[ModelUpdateRecord] = []
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    for raw_entry in parser.entries:
        title = raw_entry["title"].strip()
        summary = raw_entry["summary"].strip()

        if is_irrelevant_or_chrome(title, summary):
            continue

        raw_url = raw_entry.get("url") or base_url
        if not is_safe_official_url(raw_url, provider_id):
            raw_url = base_url

        norm_url = normalize_update_url(raw_url)
        entry_id = generate_update_id(norm_url, provider_id, title)

        model_id, model_name, model_family = resolve_update_model(
            title, summary, provider_hint=provider_id
        )
        update_type = classify_update_type(title, summary)
        scope = classify_scope(title, summary, model_id, model_name, model_family)
        provider_claims = extract_provider_claims(title, summary)

        record: ModelUpdateRecord = {
            "id": entry_id,
            "provider_id": provider_id,
            "provider_name": source["provider_name"],
            "title": title,
            "summary": summary or f"Official update from {source['provider_name']}.",
            "update_type": update_type,
            "scope": scope,
            "model_name": model_name,
            "model_family": model_family,
            "model_id": model_id,
            "published_at": raw_entry.get("published_at"),
            "effective_at": raw_entry.get("effective_at"),
            "detected_at": now_iso,
            "source_type": source.get("source_type", "official_primary_source"),
            "source_url": norm_url,
            "verification_status": "official_source",
            "source_id": source["id"],
            "provider_claims": provider_claims,
        }
        if validate_update_record(record):
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# RSS / Atom Feed Parser
# ---------------------------------------------------------------------------


def parse_rss_update_entry(
    entry: Any,
    source: ModelUpdateSource,
    now_iso: str,
) -> ModelUpdateRecord | None:
    """Parse a single feedparser entry into a ModelUpdateRecord."""
    title = entry.get("title", "").strip()
    if not title or is_irrelevant_or_chrome(title):
        return None

    # Link extraction
    link = entry.get("link", "").strip()
    if not link and entry.get("links"):
        link = entry["links"][0].get("href", "").strip()
    if not link or not is_safe_official_url(link, source["provider_id"]):
        link = source["url"]

    norm_url = normalize_update_url(link)
    entry_id = generate_update_id(norm_url, source["provider_id"], title)

    # Date parsing (feedparser provides published_parsed or updated_parsed)
    raw_date = entry.get("published_parsed") or entry.get("updated_parsed")
    published_candidate = parse_date_string(raw_date) if raw_date else parse_date_string(entry.get("published") or entry.get("updated"))

    # Summary text
    summary = entry.get("summary", "") or entry.get("description", "")
    # Strip HTML tags from summary
    clean_summary = re.sub(r"<[^>]+>", " ", summary)
    clean_summary = " ".join(clean_summary.split()).strip()
    if not clean_summary:
        clean_summary = f"Official update from {source['provider_name']}."
    elif len(clean_summary) > 400:
        clean_summary = clean_summary[:397] + "..."

    # Future deprecation vs publication date extraction
    published_at, effective_at = extract_dates(clean_summary, published_candidate)

    model_id, model_name, model_family = resolve_update_model(
        title, clean_summary, provider_hint=source["provider_id"]
    )
    update_type = classify_update_type(title, clean_summary)
    scope = classify_scope(title, clean_summary, model_id, model_name, model_family)
    provider_claims = extract_provider_claims(title, clean_summary)

    # Guard against unverified secondary claims / rumors / leaks
    verification_status = "official_source"
    source_type = source.get("source_type", "official_primary_source")
    if re.search(r"\b(rumor|leak|predicted|speculation|unconfirmed)\b", f"{title} {clean_summary}", re.IGNORECASE):
        verification_status = "unverified"
        source_type = "secondary_source"

    rec: ModelUpdateRecord = {
        "id": entry_id,
        "provider_id": source["provider_id"],
        "provider_name": source["provider_name"],
        "title": title,
        "summary": clean_summary,
        "update_type": update_type,
        "scope": scope,
        "model_name": model_name,
        "model_family": model_family,
        "model_id": model_id,
        "published_at": published_at,
        "effective_at": effective_at,
        "detected_at": now_iso,
        "source_type": source_type,
        "source_url": norm_url,
        "verification_status": verification_status,
        "source_id": source["id"],
        "provider_claims": provider_claims,
    }
    return rec


def parse_rss_updates(
    feed_content_or_url: str,
    source: ModelUpdateSource,
) -> list[ModelUpdateRecord]:
    """Parse RSS or Atom content string into structured ModelUpdateRecord items."""
    parsed = feedparser.parse(feed_content_or_url)
    records: list[ModelUpdateRecord] = []
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    for entry in getattr(parsed, "entries", []):
        try:
            rec = parse_rss_update_entry(entry, source, now_iso)
            if rec and validate_update_record(rec):
                records.append(rec)
        except Exception as exc:
            logger.warning("Failed to parse RSS update entry from %s: %s", source.get("id"), exc)

    return records


# ---------------------------------------------------------------------------
# Validation & Persistence
# ---------------------------------------------------------------------------


def validate_update_record(record: dict) -> bool:
    """Validate that a model update record conforms to the schema and domain security rules."""
    if not isinstance(record, dict):
        return False
    missing = REQUIRED_UPDATE_FIELDS - record.keys()
    if missing:
        logger.warning("Model update record missing fields %s: %r", missing, record.get("id"))
        return False
    if record.get("update_type") not in ALLOWED_UPDATE_TYPES:
        logger.warning("Invalid update_type '%s': %r", record.get("update_type"), record.get("id"))
        return False
    if record.get("scope") not in ALLOWED_SCOPES:
        logger.warning("Invalid scope '%s': %r", record.get("scope"), record.get("id"))
        return False

    url = record.get("source_url")
    pid = record.get("provider_id", "")
    if not isinstance(url, str) or not is_safe_official_url(url, pid):
        logger.warning("Invalid or unsafe source_url '%s': %r", url, record.get("id"))
        return False

    title = record.get("title", "")
    if is_irrelevant_or_chrome(title, record.get("summary", "")):
        logger.warning("Record rejected as page chrome or invalid title: %r", title)
        return False

    return True


def load_model_updates(path: Path | None = None) -> list[dict]:
    """Load model updates from JSON file, returning empty list if missing or corrupt."""
    target_path = path or DEFAULT_UPDATES_PATH
    if not target_path.exists():
        return []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if validate_update_record(r)]
    except Exception as exc:
        logger.warning("Failed reading model updates from %s: %s", target_path, exc)
    return []


def merge_model_updates(
    existing: list[dict],
    incoming: list[dict],
    max_per_source: int = 25,
    max_days: int = 180,
) -> list[dict]:
    """Merge incoming model updates into existing records with deduplication,
    history preservation, and idempotent ordering.

    Guarantees:
    - Bounded per-source entries (~180 days or max ~25 per source).
    - Preserves existing records when incoming is empty or source fails.
    - Deterministic ordering by `published_at` descending, then `detected_at` descending, then `title`.
    """
    by_id: dict[str, dict] = {}

    # Seed with existing valid records
    for rec in existing:
        if validate_update_record(rec):
            by_id[rec["id"]] = rec

    # Merge incoming
    for rec in incoming:
        if not validate_update_record(rec):
            continue
        rec_id = rec["id"]
        if rec_id not in by_id:
            by_id[rec_id] = rec
        else:
            # Preserve original detected_at while updating factual fields if needed
            existing_rec = by_id[rec_id]
            merged_rec = dict(existing_rec)
            if not merged_rec.get("published_at") and rec.get("published_at"):
                merged_rec["published_at"] = rec["published_at"]
            if not merged_rec.get("effective_at") and rec.get("effective_at"):
                merged_rec["effective_at"] = rec["effective_at"]
            if not merged_rec.get("model_id") and rec.get("model_id"):
                merged_rec["model_id"] = rec["model_id"]
            if rec.get("provider_claims"):
                merged_rec["provider_claims"] = rec["provider_claims"]
            by_id[rec_id] = merged_rec

    # Group by source to apply bounded retention (max 25 entries per source)
    by_source: dict[str, list[dict]] = {}
    for r in by_id.values():
        sid = r.get("source_id", "unknown")
        by_source.setdefault(sid, []).append(r)

    retained: list[dict] = []
    for sid, source_records in by_source.items():
        # Sort records within source chronologically descending
        source_records.sort(
            key=lambda r: (
                r.get("published_at") or "",
                r.get("detected_at") or "",
                r.get("title") or "",
            ),
            reverse=True,
        )
        # Apply max_per_source limit
        retained.extend(source_records[:max_per_source])

    # Global deterministic sort
    retained.sort(
        key=lambda r: (
            r.get("published_at") or "",
            r.get("detected_at") or "",
            r.get("title") or "",
        ),
        reverse=True,
    )

    return retained


def write_model_updates_atomic(records: list[dict], path: Path | None = None) -> None:
    """Write model updates to disk atomically using temporary file and atomic replace."""
    target_path = path or DEFAULT_UPDATES_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(records, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=".model_updates_tmp_",
        suffix=".json",
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


# ---------------------------------------------------------------------------
# Main Sync Pipeline Runner
# ---------------------------------------------------------------------------


def sync_model_updates(
    sources_path: Path | None = None,
    updates_path: Path | None = None,
    timeout: int = 15,
) -> tuple[int, int]:
    """Execute model updates ingestion across all active official provider sources.

    Returns:
      (total_records_count, new_records_count)
    """
    logger.info("=== Starting Model Updates synchronization ===")
    target_path = updates_path or DEFAULT_UPDATES_PATH
    existing_records = load_model_updates(target_path)
    existing_count = len(existing_records)
    logger.info("Loaded %d existing model update records.", existing_count)

    sources = get_active_model_update_sources(sources_path)
    if not sources:
        logger.warning("No active model update sources configured.")
        return existing_count, 0

    incoming_records: list[ModelUpdateRecord] = []
    successful_sources = 0
    failed_sources = 0

    headers = {
        "User-Agent": "AI-Pulse-Bot/1.0 (+https://github.com/swaroophegde07/ai-pulse)",
        "Accept": "application/rss+xml, application/atom+xml, text/html, application/xhtml+xml, */*",
    }

    for src in sources:
        sid = src["id"]
        url = src["url"]
        method = src.get("access_method", "rss")
        logger.info("Fetching model updates for source '%s' (%s, %s)...", sid, src["provider_name"], method)

        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                logger.warning("Source '%s' returned HTTP %d: %s", sid, resp.status_code, url)
                failed_sources += 1
                continue

            content = resp.text
            if method in ("rss", "atom"):
                parsed = parse_rss_updates(content, src)
            elif method == "html":
                parsed = parse_html_changelog(content, src, base_url=url)
            else:
                logger.info("Skipping manual_only source '%s'.", sid)
                continue

            logger.info("Extracted %d valid records from source '%s'.", len(parsed), sid)
            incoming_records.extend(parsed)
            successful_sources += 1

        except requests.exceptions.Timeout:
            logger.warning("Source '%s' timed out after %ds: %s (preserving existing history).", sid, timeout, url)
            failed_sources += 1
        except Exception as exc:
            logger.warning("Source '%s' fetch error (non-fatal): %s", sid, exc)
            failed_sources += 1

    merged = merge_model_updates(existing_records, incoming_records)
    write_model_updates_atomic(merged, target_path)

    new_count = len(merged) - existing_count
    logger.info(
        "Model Updates sync complete: %d total records (%d new). %d sources succeeded, %d failed/skipped.",
        len(merged),
        max(0, new_count),
        successful_sources,
        failed_sources,
    )
    return len(merged), max(0, new_count)
