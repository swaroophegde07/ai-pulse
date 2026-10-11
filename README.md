# AI Pulse

> A minimalist AI intelligence dashboard aggregating news, research, model releases, and benchmarks from verified public sources. Summaries generated with Google Gemini.

[![Live Website](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-black?style=flat-square)](https://swaroophegde07.github.io/ai-pulse/)
[![Repository](https://img.shields.io/badge/GitHub-swaroophegde07%2Fai--pulse-181717?style=flat-square&logo=github)](https://github.com/swaroophegde07/ai-pulse)

* **Live Dashboard:** [https://swaroophegde07.github.io/ai-pulse/](https://swaroophegde07.github.io/ai-pulse/)
* **Repository:** [https://github.com/swaroophegde07/ai-pulse](https://github.com/swaroophegde07/ai-pulse)

---

## Overview

The artificial intelligence landscape moves faster than traditional news cycles. Tracking major foundation model releases, algorithmic breakthroughs, and benchmark evaluations across disparate company blogs and labs creates noise and fragmentation.

**AI Pulse** solves this by centralizing high-signal RSS feeds from frontier AI labs and reputable tech publications into a single, clean static dashboard. A lightweight Python pipeline ingests new articles twice daily, extracts content, filters duplicate or tracking parameters, and uses the official Google Gemini SDK to generate concise, objective 2–3 sentence summaries.

---

## Live Demo

Explore the live site at **[swaroophegde07.github.io/ai-pulse](https://swaroophegde07.github.io/ai-pulse/)**.

Visitors can:
* Read the latest curated AI industry developments and research findings.
* Search articles instantly by title, publisher, summary, or category.
* Filter coverage by topic: **News**, **Research**, **Benchmark**, **Product Release**, and **Opinion**.
* Explore the **Model Tracker** for frontier model architectures (GPT-4o, Claude 3.5/3.7, Gemini 2.0, Llama 3.3, DeepSeek-R1, Phi-4).
* Access the **Research Feed** directory of primary AI research labs and publications.
* Review the **Benchmark Board** tracking standardized evaluation suites (SWE-bench, Chatbot Arena, GPQA Diamond, MMLU-Pro).

---

## Features

* **Verified RSS Feeds**: Ingestion across primary labs and AI journalism outlets including OpenAI, Google DeepMind, Hugging Face, GitHub AI, VS Code, Microsoft Research, The Gradient, The Verge AI, and Ars Technica.
* **Factual AI Summaries**: Uses the official `google-genai` Python SDK to generate 2–3 sentence summaries. Built-in guardrails strip conversational filler, preserve source metadata, and fall back gracefully if an API call fails.
* **Client-Side Search & Filtering**: Fast, zero-latency in-browser search across titles, summaries, and publishers with dynamic category tabs and empty state handling.
* **Dedicated Intelligence Hubs**:
  * `/news/` — Complete, searchable chronological news archive.
  * `/models/` — Frontier foundation model landscape and model release tracking.
  * `/research/` — Research laboratory directory and technical publications feed.
  * `/benchmarks/` — Evaluation methodology reference and benchmark coverage.
* **Robust Data Pipeline**: Normalizes URLs, removes UTM tracking tags, generates stable SHA-256 IDs, prunes demo entries from production metrics, and writes JSON atomically.
* **Serverless & Database-Free**: Operates entirely with static site generation (Astro) and a Git-committed JSON datastore. No external database or persistent server required.

---

## How It Works

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[14 Public RSS Feeds] --> B[Python Scraper]
        B --> C[Gemini Summarizer + Quality Guardrails]
        C --> D[feed.json]
    end

    subgraph Benchmark & Telemetry
        E[OpenRouter API Models] --> F[Model Identity Registry]
        G[LiveBench Official Releases] --> F
        F --> H[Benchmark Runner & Validation Report]
        H --> I[benchmarks.json & model-identities.json]
    end

    subgraph Model Catalogue
        F --> J[Model Updater]
        J --> K[models.json - 91 Catalogued, 23 Featured]
    end

    subgraph Status & Build
        L[Pipeline Status Reporter] --> M[pipeline-status.json & build.json]
    end

    D --> N[Astro SSG Build]
    I --> N
    K --> N
    M --> N
    N --> O[GitHub Pages Dashboard]
```

1. **Ingest News**: The Python ingestion script reads verified feeds from `src/data/sources.json` across 14 research and news publishers.
2. **Deduplicate & Clean**: URLs are canonicalized, UTM tracking queries stripped, future dates clamped, and existing articles detected via deterministic SHA-256 IDs.
3. **Summarize with Quality Guardrails**: New articles are passed to Google Gemini (`google-genai` SDK) to produce structured summaries. A quality validator rejects filler intros, bracketed placeholders, and headline-to-model brand mismatches, falling back gracefully to clean excerpts without burning quota. Merged into `src/data/feed.json`.
4. **Model Identity Mapping**: `pipeline/models/identity.py` standardizes models across OpenRouter and benchmark datasets using an exact 4-tier resolution engine, preventing false version collapsing.
5. **Ingest Benchmarks & Validate**: The benchmark pipeline (`pipeline/benchmarks/runner.py`) fetches and validates multi-source evaluations (LiveBench evaluations under Apache-2.0, OpenRouter pricing & context telemetry). All records are verified, conflict-checked, and exported with a full validation report to `src/data/benchmarks.json`.
6. **Catalog Models**: Foundation model architecture specs, release dates, open weights licenses, and pricing are curated in `src/data/models.json` (91 models, 23 featured flagships).
7. **Status & Build Metadata**: `pipeline/run.py` generates `src/data/pipeline-status.json` and `src/data/build.json` tracking source health, git revisions, and deployment timestamps.
8. **Build & Deploy**: Astro compiles the static HTML/CSS bundle into `dist/` and GitHub Actions deploys the site to GitHub Pages.

---

## Technology Stack

* **Frontend:** [Astro](https://astro.build/) (Static Site Generation), [TypeScript](https://www.typescriptlang.org/), [Tailwind CSS](https://tailwindcss.com/)
* **Pipeline:** [Python 3.10+](https://www.python.org/), [feedparser](https://github.com/kurtmckee/feedparser), [google-genai](https://github.com/google-gemini/deprecations) (Official Google GenAI SDK)
* **Automation:** [GitHub Actions](https://github.com/features/actions) (Scheduled cron + manual workflow dispatch)
* **Hosting:** [GitHub Pages](https://pages.github.com/)
* **Testing:** [pytest](https://docs.pytest.org/), [pytest-cov](https://pytest-cov.readthedocs.io/), `astro check`

---

## Local Setup

### Prerequisites

* **Node.js** 18.17.1+ or 20+
* **Python** 3.10+
* **Google AI Studio API Key**: Required only for running the Gemini summarizer locally ([Get a free key](https://aistudio.google.com/app/apikey))

### 1. Clone the Repository

```bash
git clone https://github.com/swaroophegde07/ai-pulse.git
cd ai-pulse
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and provide your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

*(Note: The frontend runs without an API key because it reads pre-built data from `src/data/feed.json`.)*

### 3. Install Dependencies

Install Node.js dependencies:
```bash
npm install
```

Install Python dependencies:
```bash
pip install -r requirements-dev.txt
```

### 4. Run the Ingestion Pipeline (Optional)

To ingest fresh RSS items and generate summaries locally:

```bash
python -m pipeline.run
```

### 5. Start the Development Server

```bash
npm run dev
```

Open [http://localhost:4321](http://localhost:4321) in your browser.

To verify the production build locally:

```bash
npm run build
npm run preview
```

---

## Running Tests

Unit tests mock the Gemini API and do not require network access or a live API key.

```bash
# Run Python pipeline tests
python -m pytest pipeline/tests/ -v

# Run Astro type check
npm run check

# Run production build validation
npm run build
```

---

## Automated Updates

The project uses a GitHub Actions workflow defined in [`.github/workflows/update-and-deploy.yml`](.github/workflows/update-and-deploy.yml).

* **Twice-Daily Schedule:** Runs at `07:00 UTC` and `19:00 UTC` daily.
* **Manual Dispatch:** Can be triggered on demand via **Actions → Update and Deploy → Run workflow**.
* **Deployment Pipeline:**
  1. Runs pipeline test suite (`pytest`).
  2. Runs `python -m pipeline.run` with repository secret `GEMINI_API_KEY`.
  3. Commits updated `src/data/feed.json` using `[skip ci]` to prevent recursive workflow triggers.
  4. Runs `npm run check` and `npm run build`.
  5. Deploys the static `dist/` directory to GitHub Pages via `actions/deploy-pages`.

---

## Project Structure

```text
ai-pulse/
├── .github/workflows/
│   └── update-and-deploy.yml    # Ingestion, validation, build, and deployment workflow
├── pipeline/
│   ├── run.py                   # Unified pipeline orchestrator & status reporter
│   ├── scraper.py               # Feed parsing, URL normalization & deduplication
│   ├── sources.py               # 14 verified source registries & category definitions
│   ├── summarizer.py            # Gemini SDK summarization, quality validator & rate-limiter
│   ├── models/                  # Model intelligence & synchronization subsystem
│   │   ├── identity.py          # Exact canonical model identity mapping & alias registry
│   │   └── updater.py           # Model catalogue synchronizer & featured frontier tagging
│   ├── benchmarks/              # Multi-source benchmark & telemetry subsystem
│   │   ├── base.py              # Schema validation, internal identity linking & adapter specs
│   │   ├── runner.py            # Benchmark execution orchestrator & validation reporting
│   │   └── adapters/            # Adapters (LiveBench, OpenRouter, BenchLM, etc.)
│   ├── requirements.txt         # Production Python dependencies
│   └── tests/
│       └── test_pipeline.py     # Comprehensive test suite (118 passing unit tests)
├── src/
│   ├── components/              # Header, Footer, NewsCard, CategoryFilter
│   ├── data/
│   │   ├── feed.json            # Ingested article datastore
│   │   ├── sources.json         # Configured RSS feed list
│   │   ├── models.json          # 91 catalogued models with 23 featured frontier flagships
│   │   ├── model-identities.json# Exported canonical identity & alias registry
│   │   ├── benchmarks.json      # 552 LiveBench evaluations & OpenRouter telemetry
│   │   ├── pipeline-status.json # Source health snapshot across feeds & external APIs
│   │   └── build.json           # Git commit revision, build date & environment metadata
│   ├── layouts/
│   │   └── Layout.astro         # HTML document shell
│   └── pages/
│       ├── index.astro          # Dashboard homepage with Today's AI Brief & Priority Intel
│       ├── news/index.astro     # Searchable news archive with live pipeline health badge
│       ├── models/index.astro   # Dual-tier Featured Frontier cards & complete filterable catalogue
│       ├── research/index.astro # Research lab directory & publication feed
│       └── benchmarks/index.astro # Benchmark Board, validation metrics & head-to-head comparison
├── astro.config.mjs             # Astro SSG configuration
├── tailwind.config.mjs          # Tailwind CSS styling config
└── requirements-dev.txt         # Dev & test Python dependencies
```

---

## Data Quality, Identity & Provenance Governance

### 1. Canonical Model Identity Mapping (`pipeline/models/identity.py`)
To prevent data contamination between external providers and benchmark suites, model resolution operates via a strict 4-tier hierarchy:
1. **Exact Internal Model ID**: Matches curated system keys (e.g. `anthropic/claude-3-7-sonnet`).
2. **Exact OpenRouter Model ID**: Matches canonical provider IDs (e.g. `anthropic/claude-3.7-sonnet`).
3. **Exact Benchmark Canonical ID**: Matches evaluation suite keys (e.g. `claude-3-7-sonnet`).
4. **Verified Exact Alias**: Matches approved historical checkpoint tags without substring ambiguity.

> [!IMPORTANT]
> Unmapped models or versioned checkpoints (e.g. `gpt-5.5` vs `gpt-5.5-xhigh`, or `deepseek-v3` vs `deepseek-v3.2`) are **never** collapsed together via fuzzy substrings. They retain distinct records or remain tagged as unmapped variants with transparent attribution.

### 2. Dual-Tier Model Interface (`/models/`)
- **Featured Frontier Flagships**: Curated high-impact reasoning, coding, and multimodal models (23 models) displayed with provider badges, context window formatting, verified token pricing, licensing, and documentation links.
- **Complete Model Catalogue**: Exhaustive table of 91 models with client-side category filter tabs (`All`, `Frontier Reasoning`, `Coding`, `Multimodal`, `Open Weights`, `Historical`) and instant real-time text search.

### 3. Empirical Benchmark Validation (`/benchmarks/`)
- **Upstream Provenance**: Ingests official LiveBench releases under Apache-2.0 licensing (current release: `2026-06-25` containing 552 evaluation observations).
- **Validation Report**: Tracks retrieved, parsed, valid, duplicate, and quarantined records directly in `benchmarks.json` and renders transparently on the dashboard.
- **Dynamic Standards**: Leaders for Reasoning, Coding, and Mathematics are computed dynamically from actual verified records rather than hardcoded assumptions.

### 4. Article Quality Safeguards (`pipeline/summarizer.py`)
- Automated summaries are checked for minimum character count, bracketed placeholders, and repetitive filler phrases.
- Brand mismatch protection verifies that the summary discusses the primary entity named in the headline.
- Articles failing validation gracefully fall back to clean excerpts (`unverified_excerpt`) without burning API quota.
- Homepage displays calendar-aware "Today's AI Brief" with fallback indicators when same-day volume is limited.

### 5. Automated Health & Status Monitoring (`src/data/pipeline-status.json`)
- Continuous tracking of all 14 RSS feed endpoints, OpenRouter telemetry, and LiveBench updates.
- Explicit status logging for disabled adapters (BenchLM, Artificial Analysis, LM Council) explaining licensing constraints and API key requirements.

---

## Project Status & Roadmap

### Implemented & Live
- [x] Automated RSS feed ingestion across 14 verified providers with future-date clamping and UTC normalization.
- [x] Objective Gemini-powered article summarization with strict quality validation and quota circuit breakers.
- [x] Calendar-aware Today's AI Brief with Priority Intelligence age badges on the homepage.
- [x] Dual-tier model catalogue: 23 Featured Frontier flagships and 91 complete models with 6 category filter tabs.
- [x] Multi-Source Benchmark & Intelligence Dashboard (`/benchmarks/`) with transparent Data Integrity & Validation Reporting.
- [x] Official LiveBench ingestion (552 observations) without fuzzy substring collisions or version collapses.
- [x] Automated OpenRouter API telemetry integration and exact model identity mapping.
- [x] Transparent pipeline status and source health monitoring (`pipeline-status.json` and footer badges).
- [x] Build revision metadata (`build.json`) capturing git commit hashes and build timestamps.
- [x] Comprehensive test suite (**118 passing unit tests** covering feeds, summarizer, benchmarks, model identity, and pipeline health).
- [x] 100% clean TypeScript diagnostics (`astro check`: 0 errors, 0 warnings, 0 hints across 17 files).
- [x] Static site generation producing clean, verified HTML across all 5 routes (`/`, `/news/`, `/models/`, `/research/`, `/benchmarks/`).

### Planned Features
- [ ] Direct arXiv `cs.AI` and `cs.LG` automated daily preprint clustering.
- [ ] Semantic vector search over historical articles using Gemini embeddings.
- [ ] Optional daily email digest dispatch.

---

## Feedback & Bug Reporting

AI Pulse welcomes community contributions, bug reports, and data quality corrections. All feedback is organized through dedicated GitHub Issue Forms:

* **[Report a Bug](https://github.com/swaroophegde07/ai-pulse/issues/new?template=bug_report.yml)**: Technical errors, broken filters, layout overflow, or mobile defects (applies repository label `bug`).
* **[Suggest a Feature](https://github.com/swaroophegde07/ai-pulse/issues/new?template=feature_request.yml)**: Propose new RSS feeds, topic filters, or dashboard features (applies repository label `enhancement`).
* **[Report Incorrect Data](https://github.com/swaroophegde07/ai-pulse/issues/new?template=data_correction.yml)**: Report inaccurate model specs, benchmark scores, or article summaries (requests repository label `data-quality`).
* **[Share General Feedback](https://github.com/swaroophegde07/ai-pulse/issues/new?template=general_feedback.yml)**: Usability feedback, design thoughts, and general suggestions (requests repository label `feedback`).

Visitors can also access the dedicated web portal at **[/feedback/](https://swaroophegde07.github.io/ai-pulse/feedback/)**.

> [!NOTE]
> **Repository Label Setup Notice**: Adding labels to GitHub Issue Form YAML templates (`.github/ISSUE_TEMPLATE/*.yml`) does **not** create those labels in GitHub.
> While `bug` and `enhancement` exist in standard GitHub repositories, the custom labels `data-quality` and `feedback` must be created manually by the repository owner in **GitHub > Issues > Labels** ([Manage Labels](https://github.com/swaroophegde07/ai-pulse/labels)). Until created, issues submitted via these templates will still open normally without error, but GitHub will skip attaching uncreated labels.

---

## Security and Private Reporting

All GitHub issues filed through public templates are publicly visible to all repository visitors.

> **DO NOT** report security-sensitive vulnerabilities, private API keys, credentials, or personal information in public GitHub issues.

To report a security vulnerability or sensitive defect, please contact the maintainer privately via GitHub profile contact options or submit a confidential advisory if GitHub Private Vulnerability Reporting is active on the repository.

---

## Contributing

Pull requests and community suggestions are welcome!

When proposing code or data changes:
1. Ensure all feeds added to `src/data/sources.json` are public and unauthenticated.
2. Maintain data pipeline integrity: run `pytest pipeline/tests/ -v`.
3. Verify Astro templates and TypeScript diagnostics: run `npm run check`.
4. Verify production build: run `npm run build`.

Submit changes by creating a branch and opening a pull request on the repository.

---

## License

No license has been specified for this repository yet. All rights are reserved by the repository owner unless otherwise stated.
