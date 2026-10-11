/**
 * Lightweight, progressive client-side relative publication date enhancer for AI Pulse.
 *
 * Presentation format examples:
 * - 30m ago
 * - 2h ago
 * - Yesterday
 * - Oct 7
 * - Sep 18, 2026
 * - Date unknown
 *
 * Guarantees:
 * - Uses actual article publication date (never retrieval or build timestamp).
 * - Preserves machine-readable datetime and absolute date in title/tooltip.
 * - Handles local calendar day boundaries for "Yesterday".
 * - Never shows elapsed time ("X ago") for future timestamps.
 * - Leaves unrelated <time> elements untouched.
 */

export function formatRelativePublicationDate(
  isoDate: string | null | undefined,
  now: Date = new Date()
): string {
  if (!isoDate || typeof isoDate !== "string" || !isoDate.trim()) {
    return "Date unknown";
  }

  const pubDate = new Date(isoDate);
  if (isNaN(pubDate.getTime())) {
    return "Date unknown";
  }

  const diffMs = now.getTime() - pubDate.getTime();

  // If timestamp is in the future (> 1 minute ahead), format as calendar date without elapsed "ago"
  if (diffMs < -60000) {
    const month = pubDate.toLocaleString("en-US", { month: "short" });
    const day = pubDate.getDate();
    return pubDate.getFullYear() === now.getFullYear()
      ? `${month} ${day}`
      : `${month} ${day}, ${pubDate.getFullYear()}`;
  }

  const diffSec = Math.max(0, Math.floor(diffMs / 1000));
  const diffMin = Math.floor(diffSec / 60);
  const diffHours = Math.floor(diffMin / 60);

  // Under 1 minute
  if (diffMin < 1) {
    return "Just now";
  }

  // Under 60 minutes
  if (diffMin < 60) {
    return `${diffMin}m ago`;
  }

  // Compare local calendar dates
  const nowDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const pubDay = new Date(pubDate.getFullYear(), pubDate.getMonth(), pubDate.getDate()).getTime();
  const oneDayMs = 86400000;

  // Under 24h on same calendar day
  if (nowDay === pubDay && diffHours < 24) {
    return `${diffHours}h ago`;
  }

  // Previous calendar day
  if (nowDay - pubDay === oneDayMs) {
    return "Yesterday";
  }

  // Same calendar year
  const month = pubDate.toLocaleString("en-US", { month: "short" });
  const day = pubDate.getDate();
  if (pubDate.getFullYear() === now.getFullYear()) {
    return `${month} ${day}`;
  }

  // Different calendar year
  return `${month} ${day}, ${pubDate.getFullYear()}`;
}

export function enhanceRelativePublicationTimes(): void {
  const elements = document.querySelectorAll<HTMLElement>("time[data-article-time='true']");
  if (!elements.length) return;

  const now = new Date();
  elements.forEach((el) => {
    const rawDate = el.getAttribute("datetime") || el.getAttribute("data-pubdate");
    if (!rawDate) return;

    const formatted = formatRelativePublicationDate(rawDate, now);
    if (formatted && formatted !== "Date unknown") {
      el.textContent = formatted;
    }
  });
}

// Auto-run if executed in a browser document
if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enhanceRelativePublicationTimes, { once: true });
  } else {
    enhanceRelativePublicationTimes();
  }
}
