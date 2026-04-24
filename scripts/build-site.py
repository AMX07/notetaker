#!/usr/bin/env python3
"""Generate a static HTML site from completed notetaker jobs.

Per-lecture pages are rendered by `src.htmlview.render_article` so the static
export and the live `/api/view/{job_id}` view share one design system. The
index page is hand-assembled here with the same tokens.

Usage:
    uv run python scripts/build-site.py \\
        --config scripts/site-config.json \\
        --output output/site
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import from the src package — this script must run via `uv run python
# scripts/build-site.py` from the repo root so Python can resolve `src.`.
from src.htmlview import render_article, _BUNNY_FONTS_HREF, _CSS, _THEME_SCRIPT


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Lecture:
    job_id: str
    title: str
    slug: str
    youtube_url: Optional[str]
    result_md_path: Path
    assets_dir: Optional[Path]
    completed_at: datetime
    duration_seconds: Optional[float]
    created_at: Optional[str]

    @property
    def url_path(self) -> str:
        return f"/{self.slug}/"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def slugify(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    dashed = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return dashed[:80] or "lecture"


def load_job_url_mapping(mapping_path: Path) -> dict[str, str]:
    """Read job-ids.txt — each line is '<job_id>\\t<url>'."""
    if not mapping_path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in mapping_path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            mapping[parts[0].strip()] = parts[1].strip()
    return mapping


def discover_lectures(jobs_dir: Path, url_mapping: dict[str, str]) -> list[Lecture]:
    lectures: list[Lecture] = []
    for job_dir in sorted(jobs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        result_md = job_dir / "result.md"
        checkpoint = job_dir / "checkpoint.json"
        if not (result_md.exists() and checkpoint.exists()):
            continue
        cp = json.loads(checkpoint.read_text())
        title = cp.get("title") or cp.get("video_filename") or job_dir.name
        assets = job_dir / "assets"
        lectures.append(
            Lecture(
                job_id=job_dir.name,
                title=title,
                slug=slugify(title),
                youtube_url=url_mapping.get(job_dir.name),
                result_md_path=result_md,
                assets_dir=assets if assets.exists() else None,
                completed_at=datetime.fromtimestamp(result_md.stat().st_mtime),
                duration_seconds=cp.get("duration_seconds"),
                created_at=cp.get("created_at"),
            )
        )
    lectures.sort(key=lambda lec: lec.completed_at, reverse=True)
    return lectures


# ---------------------------------------------------------------------------
# Index page (hand-assembled, uses the same tokens as htmlview)
# ---------------------------------------------------------------------------


INDEX_CSS_EXTRA = r"""
.page-head .meta { display: none; }

.site-intro {
  margin: 0 auto 64px;
  max-width: 640px;
}
.site-intro h1 {
  font-family: var(--font-display);
  font-weight: 300;
  font-size: var(--size-3xl);
  line-height: 1.1;
  letter-spacing: -0.01em;
  font-variation-settings: "opsz" 144, "SOFT" 70;
  margin: 0 0 16px;
  text-wrap: balance;
}
.site-intro p {
  font-family: var(--font-body);
  font-size: var(--size-md);
  line-height: 1.6;
  color: var(--muted);
  margin: 0 0 24px;
  text-wrap: pretty;
}

.lecture-list {
  max-width: 640px;
  margin: 0 auto;
  list-style: none;
  padding: 0;
}
.lecture-list li {
  padding: 32px 0;
  border-bottom: 1px solid var(--rule);
}
.lecture-list li:last-child { border-bottom: 0; }
.lecture-list .hero-item {
  padding-bottom: 48px;
  margin-bottom: 16px;
  border-bottom: 2px solid var(--ink);
}
.lecture-list .lecture-link {
  display: block;
  color: var(--ink);
  text-decoration: none;
  font-family: var(--font-display);
  font-weight: 400;
  font-size: var(--size-xl);
  line-height: 1.25;
  letter-spacing: -0.005em;
}
.lecture-list .hero-item .lecture-link {
  font-size: var(--size-2xl);
  font-weight: 500;
}
.lecture-list .lecture-link:hover { color: var(--accent); }
.lecture-list .meta-row {
  display: flex;
  gap: 16px;
  margin-top: 10px;
  font-family: var(--font-ui);
  font-size: var(--size-xs);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
}
.lecture-list .meta-row a {
  color: var(--muted);
  text-decoration: underline;
  text-underline-offset: 2px;
}
.lecture-list .meta-row a:hover { color: var(--accent); }
.lecture-list .hero-tag {
  color: var(--accent);
  font-weight: 500;
}
"""


def render_index(lectures: list[Lecture], hero: Optional[Lecture], config: dict) -> str:
    site_title = config.get("site_title", "Lecture Notes")
    tagline = config.get("tagline", "")
    takedown = config.get("takedown_email", "")
    copyright_notice = config.get("copyright_notice", "")

    items_html: list[str] = []

    def _item(lec: Lecture, *, is_hero: bool) -> str:
        classes = "hero-item" if is_hero else ""
        meta_bits: list[str] = []
        if is_hero:
            meta_bits.append('<span class="hero-tag">Hero lecture</span>')
        if lec.youtube_url:
            meta_bits.append(
                f'<a href="{_html.escape(lec.youtube_url)}" target="_blank" rel="noopener">Original on YouTube</a>'
            )
        meta = (
            f'<div class="meta-row">{" ".join(meta_bits)}</div>' if meta_bits else ""
        )
        return (
            f'<li class="{classes}">'
            f'<a class="lecture-link" href="{lec.url_path}">{_html.escape(lec.title)}</a>'
            f"{meta}"
            f"</li>"
        )

    if hero is not None:
        items_html.append(_item(hero, is_hero=True))
    for lec in lectures:
        if hero is not None and lec.job_id == hero.job_id:
            continue
        items_html.append(_item(lec, is_hero=False))

    footer_extra_parts: list[str] = []
    if copyright_notice:
        footer_extra_parts.append(_html.escape(copyright_notice))
    if takedown:
        footer_extra_parts.append(
            f'Takedown: <a href="mailto:{_html.escape(takedown)}">{_html.escape(takedown)}</a>'
        )
    footer_middle = " &middot; ".join(footer_extra_parts)

    site_title_esc = _html.escape(site_title)
    tagline_esc = _html.escape(tagline)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{site_title_esc}</title>
<meta name="description" content="{tagline_esc}">
<meta property="og:title" content="{site_title_esc}">
<meta property="og:description" content="{tagline_esc}">
<meta property="og:type" content="website">
<link rel="preconnect" href="https://fonts.bunny.net">
<link href="{_BUNNY_FONTS_HREF}" rel="stylesheet">
<style>{_CSS}{INDEX_CSS_EXTRA}</style>
<script>{_THEME_SCRIPT}</script>
</head>
<body>
<div class="page">
  <header class="page-head">
    <a class="brand" href="/">{site_title_esc}</a>
    <span class="head-right">
      <button class="theme-toggle" onclick="window.__toggleTheme()">Light / Dark</button>
    </span>
  </header>

  <section class="site-intro">
    <h1>{site_title_esc}</h1>
    <p>{tagline_esc}</p>
  </section>

  <ul class="lecture-list">
    {"".join(items_html)}
  </ul>

  <footer class="page-foot">
    <a class="brand" href="/">{site_title_esc}</a>
    <span>{footer_middle}</span>
    <button class="theme-toggle" onclick="window.__toggleTheme()">Light / Dark</button>
  </footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Per-lecture pages (delegate to htmlview.render_article)
# ---------------------------------------------------------------------------


def build_footer_middle(config: dict) -> str:
    takedown = config.get("takedown_email", "")
    copyright_notice = config.get("copyright_notice", "")
    parts: list[str] = []
    if copyright_notice:
        parts.append(_html.escape(copyright_notice))
    if takedown:
        parts.append(
            f'Takedown: <a href="mailto:{_html.escape(takedown)}">{_html.escape(takedown)}</a>'
        )
    return " &middot; ".join(parts)


def render_lecture_page(
    lecture: Lecture, config: dict, *, is_hero: bool
) -> str:
    md_text = lecture.result_md_path.read_text(encoding="utf-8")
    transcript_path = lecture.result_md_path.parent / "transcript.json"

    intro = config.get("hero_editorial_intro") if is_hero else None

    return render_article(
        md_text=md_text,
        title=lecture.title,
        source_filename=None,
        duration_seconds=lecture.duration_seconds,
        created_at=lecture.created_at,
        base_href=None,
        transcript_path=transcript_path if transcript_path.exists() else None,
        brand_text=config.get("site_title", "Notetaker"),
        brand_href="../",
        footer_extra_html=build_footer_middle(config),
        intro_html=intro,
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(config_path: Path, jobs_dir: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text())

    mapping_path = Path(
        config.get("job_url_mapping", "/tmp/notetaker-ingest/job-ids.txt")
    )
    url_mapping = load_job_url_mapping(mapping_path)

    lectures = discover_lectures(jobs_dir, url_mapping)
    if not lectures:
        print(
            f"[build-site] No completed lectures found in {jobs_dir}",
            file=sys.stderr,
        )
        sys.exit(2)

    hero_job_id = config.get("hero_job_id")
    hero = (
        next((lec for lec in lectures if lec.job_id == hero_job_id), None)
        if hero_job_id
        else None
    )

    # Guardrail: refuse to rmtree anything that isn't clearly under the repo.
    repo_root = Path(__file__).resolve().parent.parent
    try:
        output_dir.resolve().relative_to(repo_root)
    except ValueError:
        print(
            f"[build-site] Refusing to build outside repo root: {output_dir}",
            file=sys.stderr,
        )
        sys.exit(3)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Index
    index_html = render_index(lectures, hero, config)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(f"[build-site] Wrote {output_dir / 'index.html'}")

    # Per-lecture pages
    for lec in lectures:
        lec_dir = output_dir / lec.slug
        lec_dir.mkdir(parents=True, exist_ok=True)
        is_hero = hero is not None and lec.job_id == hero.job_id
        page_html = render_lecture_page(lec, config, is_hero=is_hero)
        (lec_dir / "index.html").write_text(page_html, encoding="utf-8")
        if lec.assets_dir and lec.assets_dir.exists():
            shutil.copytree(
                lec.assets_dir, lec_dir / "assets", dirs_exist_ok=True
            )
        print(
            f"[build-site] Wrote {lec_dir / 'index.html'} ({lec.title})"
        )

    print(
        f"\n[build-site] Done. {len(lectures)} lectures rendered to {output_dir}/"
    )
    if hero:
        print(f"[build-site] Hero: {hero.title}")
    else:
        print("[build-site] No hero designated (set hero_job_id in config)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=Path("scripts/site-config.json")
    )
    parser.add_argument(
        "--jobs-dir", type=Path, default=Path("output/jobs")
    )
    parser.add_argument("--output", type=Path, default=Path("output/site"))
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[build-site] Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    build(args.config, args.jobs_dir, args.output)


if __name__ == "__main__":
    main()
