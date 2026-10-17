// src/pages/feed.xml.ts — Public RSS 2.0 Feed for AI Pulse
import type { APIRoute } from "astro";
import feedData from "../data/feed.json";

interface FeedRecord {
  id: string;
  title: string;
  company: string;
  source: string;
  summary: string;
  url: string;
  date: string | null;
  category: string;
  fetched_at: string;
  _demo?: boolean;
  topics?: string[];
  key_points?: string[];
}

function escapeXml(unsafe: string): string {
  return (unsafe || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export const GET: APIRoute = async () => {
  const articles = (feedData as FeedRecord[])
    .filter((a) => !a._demo)
    .sort((a, b) => {
      const da = a.date ? new Date(a.date).getTime() : 0;
      const db = b.date ? new Date(b.date).getTime() : 0;
      if (da !== db) return db - da;
      const fa = a.fetched_at ? new Date(a.fetched_at).getTime() : 0;
      const fb = b.fetched_at ? new Date(b.fetched_at).getTime() : 0;
      return fb - fa;
    })
    .slice(0, 50);

  const siteUrl = "https://swaroophegde07.github.io/ai-pulse";
  const lastBuildDate = articles[0]?.date
    ? new Date(articles[0].date).toUTCString()
    : new Date().toUTCString();

  const itemsXml = articles
    .map((article) => {
      const pubDate = article.date
        ? new Date(article.date).toUTCString()
        : new Date(article.fetched_at).toUTCString();
      const escapedTitle = escapeXml(article.title);
      const escapedUrl = escapeXml(article.url);
      const escapedSummary = escapeXml(article.summary);
      const escapedCompany = escapeXml(article.company || article.source || "AI Pulse");
      const escapedCategory = escapeXml(article.category || "AI News");

      return `    <item>
      <title>${escapedTitle}</title>
      <link>${escapedUrl}</link>
      <guid isPermaLink="false">${escapeXml(article.id)}</guid>
      <pubDate>${pubDate}</pubDate>
      <description>${escapedSummary}</description>
      <category>${escapedCategory}</category>
      <dc:creator>${escapedCompany}</dc:creator>
    </item>`;
    })
    .join("\n");

  const rssFeed = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>AI Pulse — Curated AI Intelligence Feed</title>
    <link>${siteUrl}/</link>
    <description>Minimalist AI intelligence dashboard aggregating news about AI companies, foundation models, preprints, and benchmarks.</description>
    <language>en-us</language>
    <lastBuildDate>${lastBuildDate}</lastBuildDate>
    <atom:link href="${siteUrl}/feed.xml" rel="self" type="application/rss+xml"/>
${itemsXml}
  </channel>
</rss>`;

  return new Response(rssFeed, {
    status: 200,
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
    },
  });
};
