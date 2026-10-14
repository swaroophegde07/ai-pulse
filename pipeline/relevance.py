"""
Relevance gate and newsletter parser for AI Pulse.

Ensures the feed only includes stories where artificial intelligence, machine
learning, foundation models, or autonomous agents are substantively involved in the
development, rather than merely mentioned in passing.

Rejects unrelated stories about health, pharmaceuticals, batteries, climate technology,
space exploration, or general tech unless AI is central to the breakthrough.
Separates mixed newsletters (e.g. MIT Technology Review's 'The Download') into
individual stories so AI-relevant segments can be ingested without unrelated baggage.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Core AI terminology regexes with word boundaries
_AI_CORE_TERMS = re.compile(
    r"\b("
    r"artificial\s+intelligence|"
    r"machine\s+learning|"
    r"deep\s+learning|"
    r"neural\s+networks?|"
    r"deep\s+neural|"
    r"foundation\s+models?|"
    r"large\s+language\s+models?|"
    r"small\s+language\s+models?|"
    r"vision-?language\s+models?|"
    r"diffusion\s+language\s+models?|"
    r"llms?|slms?|vlms?|"
    r"generative\s+ai|genai|"
    r"transformers?|"
    r"attention\s+mechanisms?|sparse\s+attention|self-attention|"
    r"mixture\s+of\s+experts|moe|"
    r"state\s+space\s+models?|mamba|ssms?|"
    r"diffusion\s+models?|diffusion|"
    r"flow\s+matching|"
    r"pre-?training|fine-?tuning|distillation|pruning|quantization|"
    r"ai\s+agents?|agentic\s+ai|agentic\s+reasoning|agentic\s+coding|agentic|"
    r"autonomous\s+agents?|self-evolving\s+agents?|tool-using\s+agents?|"
    r"computer-use\s+agents?|computer-use|"
    r"reinforcement\s+learning|rlhf|rlaif|self-play|"
    r"reasoning\s+traces?|chain-of-thought|"
    r"prompt\s+engineering|in-context\s+learning|"
    r"ai\s+safety|ai\s+alignment|ai\s+evaluations?|ai\s+benchmarks?|"
    r"ai\s+governance|ai\s+regulation|ai\s+policy|"
    r"synthetic\s+data|synthid|"
    r"humanoid\s+robots?|physical\s+robotics|robotics\s+ai|"
    r"computer\s+vision|natural\s+language\s+processing|nlp|"
    r"vector\s+databases?|retrieval-?augmented\s+generation|rag|"
    r"embeddings?|text\s+embeddings?|"
    r"text-to-speech|voice\s+cloning|speech\s+recognition|asr|tts|"
    r"vision-?language|multimodal\s+models?|multimodal|"
    r"open\s+models?|open-?weights?|"
    r"vibe-cod(ing|ed)|copilot|codex|"
    r"alphagenome|alphafold|alphazero|"
    r"agi|artificial\s+general\s+intelligence"
    r")\b",
    re.IGNORECASE,
)

# Known frontier models, dedicated AI labs, and frameworks
_AI_NAMED_ENTITIES = re.compile(
    r"\b("
    r"openai|chatgpt|gpt-[3456]|o1|o3|o4|sora|"
    r"anthropic|claude|"
    r"google\s+deepmind|deepmind|gemini|gemma|"
    r"meta\s+ai|llama|"
    r"deepseek|"
    r"mistral\s+ai|mistral|mixtral|codestral|devstral|"
    r"qwen|"
    r"hugging\s+face|"
    r"github\s+copilot|copilot|"
    r"cursor|"
    r"midjourney|stable\s+diffusion|"
    r"perplexity|"
    r"livebench|swe-bench|mmlu|humaneval|"
    r"megatron|cuopt|doca\s+gpunetio|tensorrt"
    r")\b",
    re.IGNORECASE,
)

# Clear non-AI / distractor topics when AI is absent
_UNRELATED_DISTRACTORS = re.compile(
    r"\b("
    r"fake\s+blood|synthetic\s+blood|dried\s+blood|blood\s+substitutes?|hemorrhagic|"
    r"glp-1|weight-loss\s+drugs?|ozempic|wegovy|tirzepatide|biological\s+age(?!\s+clock\s+model)|"
    r"solid-state\s+batter(?:y|ies)|lithium-ion|carbon\s+dioxide\s+batter(?:y|ies)|compressed\s+co2|future\s+of\s+batteries|"
    r"moon\s+city|cities\s+on\s+the\s+moon|lunar\s+base|lunar\s+surface|moon\s+city\s+problems|"
    r"delivery\s+drone\s+air\s+traffic|drone\s+air\s+traffic|unmanned\s+aircraft\s+traffic|"
    r"portable\s+rubber\s+dams?|rubber\s+dams?|flood\s+defen[sc]e|"
    r"rainforests?|natural\s+forest\s+regeneration|seed-dispersers?|tropical\s+ecosystems?|"
    r"climate\s+tech\s+companies\s+to\s+watch|climate\s+tech\s+companies\s+list|"
    r"geothermal\s+power’s\s+promise|geothermal\s+power|"
    r"cage\s+matches|distrokid"
    r")\b",
    re.IGNORECASE,
)

# AI explicitly in title: e.g. "AI", "AI's", "AI-powered", "industrial AI", "AGI"
_AI_WORD_BOUNDARY = re.compile(r"\b(AI('s|s)?|AGI)\b|AI-[A-Za-z]+|[A-Za-z]+-AI\b")

# Dedicated primary AI sources where published research is inherently AI-focused
_DEDICATED_AI_SOURCES = frozenset([
    "arXiv cs.AI (Artificial Intelligence)",
    "arXiv cs.LG (Machine Learning)",
    "Google DeepMind Blog",
    "OpenAI News",
    "Anthropic News",
    "Hugging Face Blog",
    "The Gradient",
])


def is_article_ai_relevant(
    title: str,
    text: str = "",
    summary: str = "",
    tags: list[str] | None = None,
    source_name: str = "",
) -> tuple[bool, str]:
    """Determine whether an article has artificial intelligence as a central focus.

    Returns:
        (is_relevant, reason)
    """
    clean_title = html.unescape(title or "").strip()
    clean_text = html.unescape(text or "").strip()
    clean_summary = html.unescape(summary or "").strip()
    combined_body = f"{clean_text} {clean_summary}".strip()

    if not clean_title:
        return False, "Empty title"

    # Distractor check
    distractor_match_title = _UNRELATED_DISTRACTORS.search(clean_title)

    # Check for direct AI presence in title
    has_ai_word_in_title = bool(_AI_WORD_BOUNDARY.search(clean_title))
    has_ai_core_in_title = bool(_AI_CORE_TERMS.search(clean_title))
    has_ai_entity_in_title = bool(_AI_NAMED_ENTITIES.search(clean_title))
    has_title_ai = has_ai_word_in_title or has_ai_core_in_title or has_ai_entity_in_title

    # If the title is strictly about an unrelated distractor and has no AI terms at all -> REJECT
    if distractor_match_title and not has_title_ai:
        return False, f"Title focuses on unrelated topic ({distractor_match_title.group(0)}) without AI centrality"

    # If from dedicated AI research lab/archive and no unrelated distractor
    is_dedicated_source = (
        source_name in _DEDICATED_AI_SOURCES
        or any(d in source_name.lower() for d in ["openai", "anthropic", "deepmind", "hugging face", "arxiv", "the gradient"])
    )
    if is_dedicated_source and not distractor_match_title:
        return True, f"Verified publication from dedicated AI source ({source_name})"

    # Positive title AI signal
    if has_title_ai:
        # If title also matched a distractor (e.g. AI-designed viruses vs fake blood), verify AI substantive presence
        if distractor_match_title:
            body_ai = len(_AI_CORE_TERMS.findall(combined_body)) + len(_AI_WORD_BOUNDARY.findall(combined_body))
            if body_ai >= 2 or has_ai_word_in_title:
                return True, "Cross-domain breakthrough with substantive AI centrality"
            return False, f"Distractor topic with only passing AI reference ({distractor_match_title.group(0)})"
        return True, "Title substantively centers on AI / machine learning"

    # For titles without explicit AI terminology (e.g. "Do text embeddings perfectly encode text?"):
    body_ai_count = (
        len(_AI_WORD_BOUNDARY.findall(combined_body))
        + len(_AI_CORE_TERMS.findall(combined_body))
        + len(_AI_NAMED_ENTITIES.findall(combined_body))
    )

    if distractor_match_title:
        # Title has distractor. Even if body mentions AI in passing, reject unless AI heavily dominates.
        return False, f"Body only mentions AI peripherally in unrelated story ({distractor_match_title.group(0)})"

    # Check body substantive threshold for articles
    if body_ai_count >= 2:
        return True, f"Substantive AI discussion throughout text ({body_ai_count} signals)"

    # If tags explicitly assert AI
    if tags:
        ai_tags = [t for t in tags if any(k in t.lower() for k in ["model", "agent", "coding", "llm", "ai", "research", "machine learning"])]
        if ai_tags and body_ai_count >= 1:
            return True, f"AI validated via category tags ({', '.join(ai_tags)}) and text signals"

    # Default: not substantively AI-related
    return False, "AI is not substantively involved in this development"


def split_mixed_newsletter(article: dict[str, Any]) -> list[dict[str, Any]]:
    """Separate mixed newsletters (e.g. 'The Download: <part1> and <part2>') into individual stories.

    Filters out unrelated portions and produces clean, focused articles for the AI-relevant parts.
    If no part is AI-relevant, returns an empty list.
    """
    title = article.get("title", "")
    url = article.get("url", "")
    text = article.get("text", "")
    summary = article.get("summary", "")

    # Look for newsletter header pattern like "The Download: <story1> and <story2>"
    download_match = re.match(r"^(The Download:\s*)(.+)$", title, re.IGNORECASE)
    if not download_match:
        # Not a recognized newsletter format; return single article if AI-relevant, else []
        is_rel, _ = is_article_ai_relevant(title, text, summary, article.get("tags"), article.get("source", ""))
        return [article] if is_rel else []

    prefix = download_match.group(1)
    body_topics = download_match.group(2).strip()

    # Split on " and ", ", plus ", ";", etc. to identify distinct newsletter segments
    segments: list[str] = []
    parts = re.split(r",?\s+(?:and|plus)\s+|;\s*", body_topics, flags=re.IGNORECASE)
    for p in parts:
        cleaned = p.strip().strip(". ")
        if cleaned:
            segments.append(cleaned)

    if len(segments) <= 1:
        is_rel, _ = is_article_ai_relevant(title, text, summary, article.get("tags"), article.get("source", ""))
        return [article] if is_rel else []

    text_paras = [p.strip() for p in (text or summary).split("\n\n") if p.strip()]
    extracted_articles: list[dict[str, Any]] = []

    for seg_idx, segment in enumerate(segments):
        seg_title = f"{prefix}{segment}"
        matching_paras = []
        seg_words = set(re.findall(r"\b\w{4,}\b", segment.lower()))
        for p in text_paras:
            p_words = set(re.findall(r"\b\w{4,}\b", p.lower()))
            if seg_words & p_words:
                matching_paras.append(p)

        seg_text = "\n\n".join(matching_paras) if matching_paras else text

        # Evaluate relevance of this individual segment
        is_rel, reason = is_article_ai_relevant(seg_title, seg_text, tags=article.get("tags"))
        if is_rel:
            import hashlib
            from pipeline.scraper import normalize_url, stable_id

            canon_base = normalize_url(url)
            seg_slug = re.sub(r"[^\w]+", "-", segment.lower()).strip("-")
            child_url = f"{canon_base}#{seg_slug}"
            child_id = stable_id(child_url)

            child_art = dict(article)
            child_art["id"] = child_id
            child_art["title"] = seg_title
            child_art["url"] = child_url
            child_art["raw_url"] = child_url
            if matching_paras:
                child_art["text"] = seg_text
            extracted_articles.append(child_art)
        else:
            logger.info("Discarded non-AI newsletter segment: '%s' (%s)", seg_title, reason)

    return extracted_articles
