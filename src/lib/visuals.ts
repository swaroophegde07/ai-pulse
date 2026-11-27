// Shared helpers for generated covers and build-time chart aggregation.

export interface Datum {
  label: string;
  value: number;
  hue?: number;
  title?: string;
}

const BRAND_HUES: Record<string, number> = {
  openai: 165,
  anthropic: 22,
  "google deepmind": 217,
  google: 217,
  "meta ai": 190,
  meta: 190,
  microsoft: 320,
  "microsoft research": 320,
  nvidia: 88,
  "hugging face": 42,
  "arxiv (cornell university)": 356,
  arxiv: 356,
  "mit technology review": 340,
  "ars technica": 14,
  "simon willison's weblog": 265,
  deepseek: 252,
  "alibaba cloud": 46,
  "mistral ai": 4,
  xai: 0,
  opencode: 280,
  "the gradient": 300,
};

export const CATEGORY_HUES: Record<string, number> = {
  News: 199,
  Research: 262,
  Benchmark: 38,
  "Product Release": 160,
  Opinion: 350,
};

const BRAND_MARKS: Record<string, string> = {
  "arxiv (cornell university)": "arXiv",
  arxiv: "arXiv",
  openai: "OAI",
  nvidia: "NV",
  "hugging face": "HF",
  "mit technology review": "MIT",
};

export function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function brandHue(name: string | null | undefined): number {
  const key = (name || "").trim().toLowerCase();
  if (key in BRAND_HUES) return BRAND_HUES[key];
  return hashString(key || "ai-pulse") % 360;
}

export function brandMark(name: string | null | undefined): string {
  const raw = (name || "").trim();
  const key = raw.toLowerCase();
  if (key in BRAND_MARKS) return BRAND_MARKS[key];
  const words = raw
    .replace(/\(.*?\)/g, "")
    .replace(/['’]s\b/g, "")
    .split(/[\s\-_/:.]+/)
    .filter((w) => /[A-Za-z0-9]/.test(w));
  if (words.length === 0) return "AI";
  if (words.length === 1) {
    const w = words[0];
    return (w[0].toUpperCase() + (w[1] || "")).slice(0, 2);
  }
  return (words[0][0] + words[1][0]).toUpperCase();
}

function toDayKey(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * Daily counts for the `days` days ending at the newest valid, non-future date.
 * Anchoring on the newest date keeps the chart meaningful when the build runs
 * a while after the last ingestion.
 */
export function dailyCounts(
  dates: (string | null | undefined)[],
  days: number,
  nowMs: number = Date.now()
): Datum[] {
  const valid = dates
    .map((d) => (d ? new Date(d).getTime() : NaN))
    .filter((ms) => !Number.isNaN(ms) && ms <= nowMs);
  if (valid.length === 0) return [];
  const endKey = toDayKey(Math.max(...valid));
  const endMs = new Date(`${endKey}T00:00:00Z`).getTime();
  const buckets = new Map<string, number>();
  for (let i = days - 1; i >= 0; i--) {
    buckets.set(toDayKey(endMs - i * DAY_MS), 0);
  }
  for (const ms of valid) {
    const key = toDayKey(ms);
    if (buckets.has(key)) buckets.set(key, (buckets.get(key) || 0) + 1);
  }
  return [...buckets.entries()].map(([label, value]) => ({ label, value }));
}

/** Monthly counts for the `months` months ending at the newest valid, non-future date. */
export function monthlyCounts(
  dates: (string | null | undefined)[],
  months: number,
  nowMs: number = Date.now()
): Datum[] {
  const valid = dates
    .map((d) => (d && /^\d{4}-\d{2}/.test(d) ? new Date(d).getTime() : NaN))
    .filter((ms) => !Number.isNaN(ms) && ms <= nowMs);
  if (valid.length === 0) return [];
  const end = new Date(Math.max(...valid));
  const buckets = new Map<string, number>();
  for (let i = months - 1; i >= 0; i--) {
    const d = new Date(Date.UTC(end.getUTCFullYear(), end.getUTCMonth() - i, 1));
    buckets.set(d.toISOString().slice(0, 7), 0);
  }
  for (const ms of valid) {
    const key = new Date(ms).toISOString().slice(0, 7);
    if (buckets.has(key)) buckets.set(key, (buckets.get(key) || 0) + 1);
  }
  return [...buckets.entries()].map(([label, value]) => ({ label, value }));
}

export function countBy<T>(items: T[], keyFn: (item: T) => string | null | undefined): Datum[] {
  const counts = new Map<string, number>();
  for (const item of items) {
    const key = keyFn(item);
    if (!key) continue;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value || a.label.localeCompare(b.label));
}

export function formatShortDate(dayKey: string): string {
  const d = new Date(`${dayKey}T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

export function formatMonth(monthKey: string): string {
  const d = new Date(`${monthKey}-01T00:00:00Z`);
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" });
}
