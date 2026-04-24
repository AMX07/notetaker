"""Render a job's result.md as an editorial HTML article.

Styling follows the tokens in DESIGN.md: Fraunces + Geist Sans + Geist Mono,
warm paper palette, Cron Orange (#FF4700) accent, 640px reading column with
optional 840px image breakouts. Self-contained single-document output (CSS
inlined, fonts loaded from Bunny Fonts).
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True}).enable(
    ["table", "strikethrough"]
)

_BUNNY_FONTS_HREF = (
    "https://fonts.bunny.net/css?family=fraunces:300,400,400i,500,500i,600,700"
    "|geist:400,500,600|geist-mono:400,500&display=swap"
)

_CSS = r"""
:root {
  --bg:       #FAF7F2;
  --surface:  #F2EDE4;
  --ink:      #1A1714;
  --muted:    #6B6258;
  --rule:     #D9D2C4;
  --accent:   #FF4700;

  --font-display: "Fraunces", Georgia, serif;
  --font-body:    "Fraunces", Georgia, serif;
  --font-ui:      "Geist", -apple-system, sans-serif;
  --font-mono:    "Geist Mono", "Menlo", monospace;

  --reading-width: 640px;
  --breakout-width: 840px;
  --page-max: 960px;

  --size-xs:   0.75rem;
  --size-sm:   0.875rem;
  --size-base: 1rem;
  --size-md:   1.0625rem;
  --size-lg:   1.25rem;
  --size-xl:   1.5rem;
  --size-2xl:  2.25rem;
  --size-3xl:  3.5rem;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:      #14110E;
    --surface: #1F1B17;
    --ink:     #EDE8DF;
    --muted:   #908679;
    --rule:    #2B2620;
    --accent:  #FF5A1F;
  }
}
:root[data-theme="dark"] {
  --bg:      #14110E;
  --surface: #1F1B17;
  --ink:     #EDE8DF;
  --muted:   #908679;
  --rule:    #2B2620;
  --accent:  #FF5A1F;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html { font-size: 17px; }

body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-body);
  font-size: var(--size-md);
  line-height: 1.65;
  font-feature-settings: "liga", "kern";
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  transition: background 300ms ease, color 300ms ease;
}

.page {
  max-width: var(--page-max);
  margin: 0 auto;
  padding: 40px 24px 96px;
}

/* ——— Header ——— */
.page-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--rule);
  margin-bottom: 64px;
}
.page-head .brand {
  font-family: var(--font-display);
  font-style: italic;
  font-weight: 400;
  font-size: var(--size-md);
  color: var(--ink);
  text-decoration: none;
}
.page-head .brand:hover { color: var(--accent); }
.page-head .meta {
  font-family: var(--font-ui);
  font-size: var(--size-xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}

/* ——— Article ——— */
.article {
  max-width: var(--reading-width);
  margin: 0 auto;
}

/* Marginalia rail (desktop ≥1000px). `.article-wrap--with-rail` adds the
   rail column next to the prose; without the modifier the wrap is a
   single-column container so the image-breakout rule below can fire. */
.article-wrap {
  max-width: var(--reading-width);
  margin: 0 auto;
}
.article-wrap--with-rail {
  display: block;
  max-width: var(--reading-width);
}
.article-wrap--with-rail .article { margin: 0; }
@media (min-width: 1000px) {
  .article-wrap--with-rail {
    display: grid;
    grid-template-columns: var(--reading-width) 280px;
    gap: 60px;
    max-width: calc(var(--reading-width) + 280px + 60px);
    align-items: start;
  }
}

.marginalia {
  border-left: 1px solid var(--rule);
  padding: 0 0 0 24px;
  font-family: var(--font-ui);
  position: sticky;
  top: 32px;
  max-height: calc(100vh - 64px);
  overflow-y: auto;
}
.marginalia .m-label {
  font-size: var(--size-xs);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 16px;
}
.marginalia ul { list-style: none; margin: 0; padding: 0; }
.marginalia li {
  padding: 10px 0;
  border-bottom: 1px dotted var(--rule);
}
.marginalia li:last-child { border-bottom: none; }
.marginalia a {
  display: grid;
  grid-template-columns: 68px 1fr;
  gap: 12px;
  align-items: baseline;
  text-decoration: none;
  color: inherit;
  transition: color 200ms ease;
}
.marginalia a:hover { color: var(--accent); }
.marginalia .m-time {
  font-family: var(--font-mono);
  font-size: var(--size-xs);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0;
  text-transform: none;
}
.marginalia .m-label-heading {
  font-family: var(--font-body);
  font-size: var(--size-sm);
  line-height: 1.4;
  color: var(--ink);
  text-wrap: pretty;
}

@media (max-width: 1000px) {
  .article-wrap {
    grid-template-columns: 1fr;
    max-width: var(--reading-width);
    gap: 40px;
  }
  .marginalia {
    border-left: none;
    padding: 24px 0 0;
    border-top: 1px solid var(--rule);
    position: static;
    max-height: none;
    overflow-y: visible;
    order: 2;
  }
  .article-wrap .article { order: 1; }
}

/* Scroll-margin so anchored h2 isn't hidden under sticky header (if any). */
.article h2[id] { scroll-margin-top: 24px; }

.article > h1:first-of-type,
.article > h1:first-child {
  font-family: var(--font-display);
  font-weight: 300;
  font-size: var(--size-3xl);
  line-height: 1.1;
  letter-spacing: -0.01em;
  margin-bottom: 16px;
  font-variation-settings: "opsz" 144, "SOFT" 70;
  text-wrap: balance;
}

.article p.byline {
  font-family: var(--font-ui);
  font-size: var(--size-sm);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
  margin-bottom: 48px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--rule);
}

.article p.editorial-intro {
  font-family: var(--font-display);
  font-style: italic;
  font-size: var(--size-lg);
  line-height: 1.5;
  color: var(--muted);
  margin: 0 0 24px;
  padding-left: 20px;
  border-left: 2px solid var(--accent);
  text-wrap: pretty;
}

/* Drop cap on the first paragraph after the byline. Pure CSS — no JS. */
.article p.byline + p::first-letter,
.article p.byline + p::first-line { font-kerning: normal; }
.article p.byline + p::first-letter {
  font-family: var(--font-display);
  font-weight: 400;
  font-style: normal;
  font-size: 4.6rem;
  line-height: 0.85;
  float: left;
  padding: 0.15em 0.12em 0 0;
  color: var(--ink);
  font-variation-settings: "opsz" 144, "SOFT" 50;
}

.article h1,
.article h2,
.article h3,
.article h4 {
  font-family: var(--font-display);
  font-weight: 500;
  letter-spacing: -0.005em;
  text-wrap: balance;
}

.article h2 {
  font-size: var(--size-2xl);
  line-height: 1.15;
  margin-top: 64px;
  margin-bottom: 16px;
  font-weight: 400;
}

.article h3 {
  font-size: var(--size-lg);
  line-height: 1.3;
  margin-top: 40px;
  margin-bottom: 8px;
}

.article h4 {
  font-size: var(--size-md);
  line-height: 1.4;
  margin-top: 32px;
  margin-bottom: 8px;
}

.article p {
  font-family: var(--font-body);
  font-size: var(--size-md);
  line-height: 1.65;
  margin-bottom: 24px;
  color: var(--ink);
  hyphens: auto;
  -webkit-hyphens: auto;
}

.article em { font-style: italic; }
.article strong { font-weight: 600; }

.article a {
  color: inherit;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
  transition: color 200ms ease;
}
.article a:hover {
  color: var(--accent);
  text-decoration-thickness: 2px;
}

.article ul, .article ol {
  margin: 0 0 24px 24px;
  padding-left: 8px;
}
.article li {
  margin-bottom: 8px;
  padding-left: 4px;
}
.article li::marker { color: var(--muted); }

.article blockquote {
  border-left: 2px solid var(--muted);
  padding: 0 0 0 24px;
  margin: 24px 0;
  color: var(--ink);
  font-family: var(--font-display);
  font-style: italic;
}
.article blockquote p {
  font-family: inherit;
  font-style: inherit;
  margin-bottom: 8px;
}
.article blockquote p:last-child { margin-bottom: 0; }

.article code {
  font-family: var(--font-mono);
  font-size: 0.95em;
  background: var(--surface);
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid var(--rule);
}

.article pre {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 2px solid var(--accent);
  padding: 16px 20px;
  margin: 24px 0;
  overflow-x: auto;
  border-radius: 2px;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  line-height: 1.6;
}
.article pre code {
  background: transparent;
  border: none;
  padding: 0;
  font-size: inherit;
}

/* Ornament hr — centered asterism instead of a thin line, editorial convention. */
.article hr {
  border: none;
  margin: 48px auto;
  width: 120px;
  max-width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 1em;
  overflow: visible;
  position: relative;
}
.article hr::before {
  content: "\2042";
  font-family: var(--font-display);
  font-size: var(--size-lg);
  color: var(--muted);
  letter-spacing: 0.4em;
  line-height: 1;
}

.article table {
  width: 100%;
  border-collapse: collapse;
  margin: 24px 0;
  font-family: var(--font-ui);
  font-size: var(--size-sm);
  font-variant-numeric: tabular-nums;
}
.article table th,
.article table td {
  border-bottom: 1px solid var(--rule);
  padding: 10px 12px;
  text-align: left;
  vertical-align: top;
}
.article table th {
  font-weight: 500;
  color: var(--muted);
  font-size: var(--size-xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ——— Images ——— */
.article figure,
.article p > img:only-child,
.article img {
  display: block;
  margin: 32px auto;
  max-width: 100%;
  height: auto;
  border-radius: 2px;
  background: var(--surface);
  border: 1px solid var(--rule);
  padding: 6px;
}

.article figure {
  padding: 0;
  background: transparent;
  border: none;
}
.article figure img {
  padding: 6px;
  margin: 0 auto;
  display: block;
}
.article figure figcaption {
  font-family: var(--font-display);
  font-style: italic;
  font-size: var(--size-sm);
  color: var(--muted);
  text-align: center;
  margin-top: 10px;
}

/* Image breakout on wide viewports — only when there is no marginalia rail.
   When `.article-wrap--with-rail` is present the reading column is constrained
   by the grid cell, so images must stay at 100% width. */
@media (min-width: 1000px) {
  .article-wrap:not(.article-wrap--with-rail) .article figure,
  .article-wrap:not(.article-wrap--with-rail) .article p > img:only-child {
    width: var(--breakout-width);
    max-width: calc(100vw - 48px);
    margin-left: calc((var(--reading-width) - var(--breakout-width)) / 2);
  }
}

/* Inside the marginalia rail layout images stay within the 640px column. */
.article-wrap--with-rail .article figure,
.article-wrap--with-rail .article figure img,
.article-wrap--with-rail .article img,
.article-wrap--with-rail .article p > img:only-child {
  width: 100%;
  max-width: 100%;
  margin-left: auto;
  margin-right: auto;
}

/* ——— Footer ——— */
.page-foot {
  max-width: var(--reading-width);
  margin: 96px auto 0;
  padding-top: 24px;
  border-top: 1px solid var(--rule);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--font-ui);
  font-size: var(--size-xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}
.page-foot .brand {
  font-family: var(--font-display);
  font-style: italic;
  font-size: var(--size-sm);
  text-transform: none;
  letter-spacing: 0;
  color: var(--ink);
  text-decoration: none;
}
.page-foot a { color: inherit; text-decoration: underline; text-underline-offset: 2px; }
.page-foot a:hover { color: var(--accent); }

.theme-toggle {
  background: transparent;
  border: 1px solid var(--rule);
  color: var(--muted);
  padding: 6px 10px;
  font-family: var(--font-ui);
  font-size: var(--size-xs);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 2px;
  transition: border-color 200ms ease, color 200ms ease;
}
.theme-toggle:hover { border-color: var(--ink); color: var(--ink); }

.page-head .theme-toggle { margin-left: 16px; }
.page-head .head-right { display: flex; align-items: center; gap: 16px; }

@media (prefers-reduced-motion: reduce) {
  * {
    transition: none !important;
    animation: none !important;
  }
}

@media (max-width: 700px) {
  html { font-size: 16px; }
  .page { padding: 24px 20px 72px; }
  .article > h1:first-of-type { font-size: 2.25rem; }
  .article h2 { font-size: 1.75rem; }
  .article figure,
  .article p > img:only-child { width: 100%; margin-left: auto; margin-right: auto; }
}
"""


_THEME_SCRIPT = r"""
(function () {
  try {
    var saved = localStorage.getItem("notetaker-theme");
    if (saved === "dark" || saved === "light") {
      document.documentElement.dataset.theme = saved;
    }
  } catch (e) {}

  window.__toggleTheme = function () {
    var root = document.documentElement;
    var current = root.dataset.theme;
    if (!current) {
      var prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      current = prefersDark ? "dark" : "light";
    }
    var next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("notetaker-theme", next); } catch (e) {}
  };
})();
"""


_BYLINE_RE = re.compile(r"^Source video:\s*(.+)$", re.MULTILINE)

_H2_WITH_BODY_RE = re.compile(
    r"^##\s+(?P<heading>.+?)\s*$\n+(?P<body>(?:(?!^#)[^\n]+\n?)+)",
    re.MULTILINE,
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _slugify(text: str) -> str:
    """Produce an HTML anchor id from heading text."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_RE.sub("-", text).strip("-")
    return slug or "section"


def _normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _WORD_RE.findall(text.lower())


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _build_toc(
    md_text: str, transcript_path: Path | None
) -> tuple[list[dict], dict[str, str]]:
    """Build a marginalia TOC from H2 headings + transcript word timestamps.

    For each H2 heading in the markdown we take the first ~6 words of the
    paragraph after it and look for a matching phrase in the transcript.
    If a match is found, the heading gets a timestamp in the marginalia.

    Returns (toc_entries, slug_for_heading) where:
      toc_entries = [{"heading": str, "slug": str, "timestamp": str, "start_s": float}]
      slug_for_heading[heading_text] = anchor slug
    """
    slug_for_heading: dict[str, str] = {}
    entries: list[dict] = []

    if not transcript_path or not transcript_path.exists():
        # Still slugify so h2s can be anchored, just no timestamps.
        for m in re.finditer(r"^##\s+(.+?)\s*$", md_text, re.MULTILINE):
            h = m.group(1).strip()
            slug_for_heading.setdefault(h, _slugify(h))
        return entries, slug_for_heading

    try:
        transcript = json.loads(transcript_path.read_text())
    except Exception:
        return entries, slug_for_heading

    # Flatten transcript into a searchable (word, start_seconds) list.
    all_words: list[tuple[str, float]] = []
    for chunk in transcript if isinstance(transcript, list) else []:
        chunk_start = float(chunk.get("start", 0.0))
        words = chunk.get("words") or []
        if words:
            for w in words:
                txt = w.get("text") or w.get("word") or ""
                start = float(w.get("start", chunk_start))
                for norm in _normalize_words(txt):
                    all_words.append((norm, start))
        else:
            # Fall back to per-chunk granularity.
            for norm in _normalize_words(chunk.get("text", "")):
                all_words.append((norm, chunk_start))

    if not all_words:
        return entries, slug_for_heading

    only_words = [w for w, _ in all_words]

    def _find_start_seconds(probe_words: list[str]) -> float | None:
        """Slide-match the probe sequence against the transcript word stream.
        Exact match required on the first word; at least 3 of the next 5 must
        also match to count as a hit."""
        if not probe_words:
            return None
        needle0 = probe_words[0]
        probe = probe_words[:6]
        best_hits = 0
        best_start: float | None = None
        for i, w in enumerate(only_words):
            if w != needle0:
                continue
            window = only_words[i : i + len(probe)]
            hits = sum(1 for a, b in zip(probe, window) if a == b)
            if hits > best_hits:
                best_hits = hits
                best_start = all_words[i][1]
                if hits == len(probe):
                    break
        if best_hits >= max(3, len(probe) // 2):
            return best_start
        return None

    for m in _H2_WITH_BODY_RE.finditer(md_text):
        heading_raw = m.group("heading").strip()
        body_raw = m.group("body").strip()
        slug = _slugify(heading_raw)
        slug_for_heading[heading_raw] = slug

        probe = _normalize_words(body_raw)
        if not probe:
            probe = _normalize_words(heading_raw)
        start_s = _find_start_seconds(probe)
        if start_s is None:
            continue
        entries.append(
            {
                "heading": heading_raw,
                "slug": slug,
                "timestamp": _format_timestamp(start_s),
                "start_s": start_s,
            }
        )

    # Also slugify headings that didn't match so anchors still work.
    for m in re.finditer(r"^##\s+(.+?)\s*$", md_text, re.MULTILINE):
        h = m.group(1).strip()
        slug_for_heading.setdefault(h, _slugify(h))

    return entries, slug_for_heading


def _inject_h2_ids(html_body: str, slug_for_heading: dict[str, str]) -> str:
    """Add id="..." to every rendered <h2> so marginalia anchors work."""

    def repl(match: re.Match) -> str:
        attrs = match.group("attrs") or ""
        inner = match.group("inner")
        # Already has id= — leave alone.
        if re.search(r"\bid\s*=", attrs):
            return match.group(0)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        slug = slug_for_heading.get(plain) or _slugify(plain)
        return f'<h2{attrs} id="{slug}">{inner}</h2>'

    return re.sub(
        r"<h2(?P<attrs>[^>]*)>(?P<inner>.+?)</h2>",
        repl,
        html_body,
        flags=re.DOTALL,
    )


def _render_marginalia(entries: list[dict]) -> str:
    if not entries:
        return ""
    items = []
    for e in entries:
        heading_esc = html.escape(e["heading"])
        slug = html.escape(e["slug"])
        ts = html.escape(e["timestamp"])
        items.append(
            f'<li><a href="#{slug}">'
            f'<span class="m-time">{ts}</span>'
            f'<span class="m-label-heading">{heading_esc}</span>'
            f"</a></li>"
        )
    return (
        '<aside class="marginalia" aria-label="Table of contents">'
        '<div class="m-label">In this lecture</div>'
        f'<ul>{"".join(items)}</ul>'
        "</aside>"
    )


def _extract_byline(md_text: str) -> tuple[str, str | None]:
    """Pull the first-line `Source video:` metadata out of the markdown.

    Returns (remaining_markdown, source_filename_or_None).
    """
    match = _BYLINE_RE.search(md_text)
    if not match:
        return md_text, None
    source = match.group(1).strip()
    # Remove the matched line and any surrounding horizontal rules.
    cleaned = _BYLINE_RE.sub("", md_text, count=1)
    cleaned = re.sub(r"^(---\s*\n){2,}", "---\n", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*---\s*\n\s*---\s*", "", cleaned)
    return cleaned, source


def _format_duration(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m} min"


def render_article(
    md_text: str,
    title: str,
    source_filename: str | None = None,
    duration_seconds: float | int | None = None,
    created_at: str | None = None,
    base_href: str | None = None,
    transcript_path: Path | None = None,
    brand_text: str = "Notetaker",
    brand_href: str = "/",
    footer_extra_html: str | None = None,
    intro_html: str | None = None,
) -> str:
    """Render markdown as a standalone editorial HTML document.

    Args:
        md_text: The raw markdown content of result.md.
        title: Page title (used in <title> and as fallback H1).
        source_filename: The original video filename.
        duration_seconds: Video duration for the byline.
        created_at: ISO timestamp, shown in page header.
        base_href: If set, adds <base href=...> so relative asset paths resolve.
            Use for the live view where assets live under the API. Omit for
            static export where paths like `assets/foo.jpg` should resolve
            relative to the HTML file.
        transcript_path: Path to the job's transcript.json, used to build the
            marginalia rail with per-section timestamps. If omitted or missing,
            the article renders without a marginalia rail.
        brand_text: Site/brand label shown in the header and footer.
        brand_href: URL the brand label links to.
        footer_extra_html: Raw HTML inserted into the footer's middle slot.
            Caller controls the content (e.g., live view uses a download link,
            static export uses a takedown / copyright notice). Trusted input —
            not escaped.
        intro_html: Optional plain-text editorial intro rendered as a styled
            italic paragraph above the byline. Escaped for safety.

    Returns:
        Complete HTML document as a string.
    """
    body_md, extracted_source = _extract_byline(md_text)
    source = source_filename or extracted_source

    # Promote first heading to title if H1 is present; we don't inject our own
    # H1 because the markdown typically starts with one. If it doesn't, we
    # prepend the passed title.
    stripped = body_md.lstrip()
    if not stripped.startswith("# "):
        body_md = f"# {title}\n\n{body_md}"

    toc_entries, slug_map = _build_toc(body_md, transcript_path)
    body_html = _md.render(body_md)
    body_html = _inject_h2_ids(body_html, slug_map)
    marginalia_html = _render_marginalia(toc_entries)

    # Inject a byline paragraph right after the first h1.
    byline_bits = []
    if source:
        # Humanize sanitized upload filenames ("Some_Title.mp4" → "Some Title")
        # and skip when the filename is just the lecture title restated.
        display = Path(source).stem.replace("_", " ").strip()
        if display and display.casefold() != (title or "").casefold():
            byline_bits.append(html.escape(display))
    dur = _format_duration(duration_seconds)
    if dur:
        byline_bits.append(dur)
    if created_at:
        byline_bits.append(html.escape(str(created_at)[:10]))
    byline_html = ""
    if byline_bits:
        byline_html = (
            '<p class="byline">' + " &middot; ".join(byline_bits) + "</p>"
        )
    intro_p = ""
    if intro_html:
        intro_p = f'<p class="editorial-intro">{html.escape(intro_html)}</p>'
    insertion = intro_p + byline_html
    if insertion:
        body_html = re.sub(
            r"(</h1>)",
            r"\1" + insertion,
            body_html,
            count=1,
        )

    title_esc = html.escape(title or brand_text)
    brand_text_esc = html.escape(brand_text)
    brand_href_esc = html.escape(brand_href)
    base_tag = f'<base href="{html.escape(base_href)}">' if base_href else ""
    footer_middle = footer_extra_html or ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_esc} &middot; {brand_text_esc}</title>
{base_tag}
<link rel="preconnect" href="https://fonts.bunny.net">
<link href="{_BUNNY_FONTS_HREF}" rel="stylesheet">
<style>{_CSS}</style>
<script>{_THEME_SCRIPT}</script>
</head>
<body>
<div class="page">
  <header class="page-head">
    <a class="brand" href="{brand_href_esc}">{brand_text_esc}</a>
    <span class="head-right">
      <span class="meta">{html.escape(str(created_at)[:10]) if created_at else ""}</span>
      <button class="theme-toggle" onclick="window.__toggleTheme()">Light / Dark</button>
    </span>
  </header>

  <div class="article-wrap{ ' article-wrap--with-rail' if marginalia_html else '' }">
    <article class="article">
      {body_html}
    </article>
    {marginalia_html}
  </div>

  <footer class="page-foot">
    <a class="brand" href="{brand_href_esc}">{brand_text_esc}</a>
    <span>{footer_middle}</span>
    <button class="theme-toggle" onclick="window.__toggleTheme()">Light / Dark</button>
  </footer>
</div>
</body>
</html>
"""


def render_article_for_job(job_dir: Path, job: dict, job_id: str) -> str:
    """Convenience: render a live in-app view from a job dir on disk."""
    md_path = job_dir / "result.md"
    md_text = md_path.read_text(encoding="utf-8")
    transcript_path = job_dir / "transcript.json"
    footer_extra = f'<a href="/api/download/{html.escape(job_id)}">Download .md</a>'
    return render_article(
        md_text=md_text,
        title=job.get("title") or md_path.stem,
        source_filename=job.get("video_filename"),
        duration_seconds=job.get("duration_seconds"),
        created_at=job.get("created_at"),
        base_href=f"/api/view/{job_id}/",
        transcript_path=transcript_path,
        footer_extra_html=footer_extra,
    )
