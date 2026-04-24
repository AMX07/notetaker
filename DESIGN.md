# Design System — Notetaker

## Product Context

- **What this is:** A web app that converts MP4 video lectures into well-structured markdown, preserving the speaker's voice with minimal edits. Upload a video, get back a reader-friendly document. Backend pipeline uses Whisper for STT and a Claude Opus 4.6 agent that orchestrates Haiku cleanup, Sonnet vision + assembly, and a self-review loop.
- **Who it's for:** Primarily the operator/builder running notetaker against their own lectures. Secondary surface: anyone who receives a generated markdown file and opens it in a reader. The Archive (sibling project, see `~/sneak-in/DESIGN.md`) is the public-facing reader for the same content.
- **Space/industry:** Personal tooling / long-form reading. Adjacent to Otter / Granola / Descript / Fireflies in category, but deliberately NOT aligned with SaaS meeting-productivity visual language. Closer to Readwise, Matter, craigmod.com, Stripe Press.
- **Project type:** Web tool with three surfaces — Upload form, Processing view, Result view. Not a marketing site. Not a dashboard.
- **Memorable thing:** When someone uses notetaker for the first time, the first feeling should be *"oh, this tool respects the speaker's voice."* Not *"another AI tool."*
- **Sibling alignment:** This design system inherits all tokens from `~/sneak-in/DESIGN.md` (the Archive) so notetaker and Archive share one visual identity. Additions in this file are notetaker-specific surfaces the Archive does not have (upload, processing, job list).

## Aesthetic Direction

- **Direction:** Editorial / Reader. Reference posture: Craig Mod's essays (craigmod.com), Robin Sloan's archive, ia.net, Stripe Press.
- **Decoration level:** Minimal. Typography carries everything. No cards, no blobs, no gradients. A 1%-opacity paper-grain background texture is optional.
- **Mood:** Warm paper, book-like, serious, unhurried. The tool feels like a print shop, not a dashboard.
- **Reference sites (posture, not copy):** craigmod.com, robinsloan.com, ia.net, stripepress.com

## Typography

- **Display / Hero / H1-H2:** Fraunces (variable; use `opsz` for optical sizing, turn `soft` axis up slightly for human feel). Google Fonts.
- **Body prose:** Fraunces Text (same family, body-optimized `opsz`). Serif body is a deliberate choice — the result view is a reader, not an app.
- **UI / Metadata / Captions / Buttons:** Geist Sans (tabular-nums enabled for dates, timestamps, durations, file sizes). Google Fonts.
- **Code / Timestamps:** Geist Mono. Google Fonts.
- **Loading strategy:** `<link rel="preconnect" href="https://fonts.googleapis.com">` + standard Google Fonts `@import` with `display=swap`. Critical styles inlined.
- **Scale (rem, base 17px):**
  - xs: 0.75 (12.75px)
  - sm: 0.875 (14.875px)
  - base: 1 (17px)
  - md: 1.0625 (18px) — reserved for body paragraph
  - lg: 1.25 (21.25px)
  - xl: 1.5 (25.5px)
  - 2xl: 2.25 (38.25px)
  - 3xl: 3.5 (59.5px) — hero H1 only
- **Line height:** 1.65 body, 1.2 display, 1.4 UI.
- **Tracking:** -0.01em on display sizes ≥ 2.25rem, 0 elsewhere.

## Color

- **Approach:** Restrained. Accent appears rarely and means something when it does.
- **Light mode (default):**
  - Background: `#FAF7F2` — warm off-white, paper tone
  - Surface (inline code bg, subtle blocks): `#F2EDE4`
  - Ink (body text): `#1A1714` — near-black with brown warmth, not `#000`
  - Muted (metadata, captions, footnotes): `#6B6258`
  - Accent: `#FF4700` — **Cron Orange**, inherited from the Archive (`~/sneak-in/DESIGN.md`). Reference to the *Grid Systems* book cover, cited in the hero lecture itself
  - Link: inherit ink color + underline, turns accent on hover
- **Dark mode:**
  - Background: `#14110E`
  - Surface: `#1F1B17`
  - Ink: `#EDE8DF`
  - Muted: `#908679`
  - Accent: `#FF5A1F` (reduce saturation ~10% for comfort at night)
- **Semantic (minimal, use sparingly):** success `#2E7D4F`, warning `#B5730E`, error `#C23B22`, info `#3E6B99`. Only for system messages (pipeline alerts, API errors), not decoration.
- **Dark mode strategy:** CSS custom properties + `prefers-color-scheme`, with a manual toggle in the footer.

## Spacing

- **Base unit:** 8px.
- **Density:** Spacious. Paragraph spacing ≈ 1.25× line-height so paragraphs read as blocks.
- **Scale:** 2xs(4) xs(8) sm(16) md(24) lg(40) xl(64) 2xl(96) 3xl(128).
- **Vertical rhythm:** all vertical spacing is a multiple of 8.

## Layout

- **Approach:** Single-column editorial with asymmetric breakouts. The Result view adds an optional right-side marginalia rail for timestamps.
- **Max content width:** 640px for prose (reading optimum, ~72 characters per line at 18px body).
- **Image handling:**
  - Inline images (slides, sketches, figures extracted from frames) render at content width by default.
  - Images with captions can break out to 840px on wide viewports; caption hangs in left margin.
  - No cards around images. No drop shadows. No border-radius > 2px.
- **Grid:** no visible grid.
- **Max page width:** 960px (content + margin breakouts + marginalia rail).
- **Header:** minimal. Site name (Fraunces italic, small), one link ("New conversion" or "Archive"). No nav menu. ~64px tall.
- **Footer:** site name, light/dark toggle, build/version, copyright. ~80px tall. No columns.
- **Border radius:** 2px max (images, inline code). Zero rounding on buttons and most UI.

## Motion

- **Approach:** Minimal-functional.
- **Easing:** `ease-out` for enter, `ease-in` for exit, `ease-in-out` for move. Default `ease`.
- **Duration:** micro 80ms (link underline), short 200ms (color transitions), medium 300ms (theme toggle, panel reveal). No `long` durations — this is a reader, not a presentation.
- **Never:** scroll-driven animation, entry animations on page load, hover-lift, bounce, parallax, fade-in-on-scroll.
- **Processing indicator exception:** the typewriter reveal in the Processing view is not animation-for-decoration; it is the product communicating what it is actually doing. Runs at ~180wpm, uses a steady caret, no bounce.

## Anti-slop rules

Banned from this design system and any page rendered against it:
- Purple / violet gradients anywhere
- 3-column feature grid with icons in colored circles
- Centered-everything layouts
- Gradient buttons
- Generic stock-photo hero sections
- `system-ui` / `-apple-system` as the display or body font (signals "gave up on typography")
- "Built for founders" / "Designed for builders" marketing copy patterns
- Inter, Roboto, Montserrat, Poppins, Space Grotesk as primary faces
- Hover-lift on any element
- Rounded "pill" buttons
- Spinners on the Processing view — use the typewriter reveal instead

## Component Notes

### Shared with the Archive
- **Hero H1 (lecture title / page title):** Fraunces 3xl, soft axis 70, weight 400, -0.01em tracking, line-height 1.1. Single line preferred; wraps naturally on narrow viewports.
- **Byline / Metadata (below H1):** Geist Sans sm, muted color, tabular-nums. Format: *`Speaker · Duration · YYYY`*.
- **Body paragraphs:** Fraunces Text md (1.0625rem), 1.65 line-height. First paragraph after H2 can drop-cap optionally (out of scope for v1).
- **H2 (section breaks):** Fraunces 2xl, weight 500, 48px top margin, 16px bottom.
- **H3:** Fraunces lg, weight 500, 32px top, 8px bottom.
- **Inline code:** Geist Mono 0.95em, surface bg, 2px radius, 2px 6px padding.
- **Links:** Ink color + 1px underline, 2px offset from baseline. Hover: accent color + underline thickens to 2px. No animation on color.
- **Images:** Full content width by default, optional 840px breakout. Caption in Geist Sans sm, muted, italic, left-aligned, 8px above image.
- **Blockquotes:** 2px solid muted border on left, 24px left padding, 24px vertical margin. No color change.

### Notetaker-specific

- **Upload dropzone:** a thin top rule, and under it, a single line of Fraunces italic — *"Drop a lecture here, or click to browse."* Not a bordered card. Not a dashed box with an icon. When active (drag-over), the italic line shifts to accent color and the rule thickens from 1px to 2px. When a file is staged, the italic line is replaced with the filename in Geist Sans base + size/duration in Geist Mono muted. Reject states (wrong MIME type, over size limit) use the `error` semantic color inline, never as a toast.
- **Title + language inputs:** bottom-rule inputs only — no box, no surface. Label above in Geist Sans xs, letter-spacing 0.18em, uppercase, muted. Input in Fraunces Text base. Placeholder in Fraunces italic, muted. Focus: rule transitions to accent color over 200ms.
- **Primary action button ("Convert to markdown"):** Geist Sans sm, 500 weight, tracked 0.06em, ink background + paper-color text. 10px 20px padding, 2px radius. Hover: swaps to accent background with paper-color text. A small Fraunces `¶` pilcrow in accent color may precede the label as structural punctuation.
- **Ghost button ("Load example"):** same type/sizing, transparent background, muted border, ink text. Hover: border darkens to ink.
- **Processing view (the wait):** no spinner. Show a single line `Step N of 4 · <stage name>` in Geist Sans xs, uppercase, tracked, accent color. Below it, a live typewriter reveal of the current transcript segment being processed, set in Fraunces Text md. The typewriter uses a 2px accent caret with a 0.9s steady blink. Stage names: `Extracting audio`, `Transcribing`, `Cleaning`, `Assembling`. When the agent review loop runs, append `· Reviewing (iteration N/15)` to the stage line.
- **Alert (success / warning / error):** surface background, 1px muted border, 2px accent left-border for warning and error variants. Type: Fraunces Text base. A Fraunces mark precedes the text: `✓` (success), `!` (warning), `×` (error), all in accent color. No colored backgrounds. No toasts that auto-dismiss — all alerts stay until the user acknowledges.
- **Result view — prose column:** Fraunces Text md at 640px max width. Paragraphs 30px bottom margin. Drop-cap on the first paragraph (Fraunces 3xl, floated, 76px size, line-height 60px). Pilcrow marks (¶) in accent color precede structural paragraph breaks identified by the assembler.
- **Result view — marginalia rail (desktop only, ≥ 900px viewport):** 280px column to the right of the prose column, separated by a 1px muted vertical rule at 24px padding. Contents: timestamp entries in Geist Mono xs (muted) + short label in Fraunces Text sm (ink). Each entry has a 1px dotted muted bottom rule. Below 900px viewport, the rail collapses inline between paragraphs, styled as a small Geist Sans xs muted line with the timestamp + label.
- **Result view — code blocks:** Geist Mono 14px/22px, surface background, 1px muted border, 2px accent left-border (signals "this is what the speaker was showing"), 18px 22px padding, 2px radius.
- **Result view — figure (preserved frame):** 12px padding on a surface background, 1px muted border, 2px radius. Figcaption in Fraunces italic sm, muted, centered, 10px above image.
- **Job list / history (future):** single-column list of past conversions, like the Archive's lecture index. Each entry: H3 title (Fraunces lg) + source filename (Geist Mono sm muted) + status (Geist Sans sm). 40px between entries. No cards, no table chrome, no row striping.
- **Status pill (in job list or header):** Geist Sans xs, 0.16em tracking, uppercase, 4px radius (2px exception for pills only), 4px 10px padding, ink border, muted fill. Active/running state: accent border + accent text.

## Accessibility

- All text on background passes WCAG AA at normal size. Ink on paper (`#1A1714` on `#FAF7F2`) is 14.6:1.
- Accent is reserved for structural emphasis and interaction; it never carries meaning on its own. Error/warning alerts carry both color and a Fraunces mark so they are distinguishable without color.
- Focus rings: 2px accent outline with 2px offset on all interactive elements. Never removed.
- Respect `prefers-reduced-motion`: typewriter reveal falls back to static text, theme toggle instant, all 200ms color transitions drop to 0ms.
- Minimum touch target 44×44px on any viewport that matches `pointer: coarse`.

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-24 | Initial design system created by `/design-consultation` | Editorial / Reader aesthetic chosen because notetaker's output is long-form editorial transcripts intended for reading; SaaS meeting-productivity visual language would be actively counterproductive (research: Otter/Granola/Descript/Fireflies all converge on that slop). |
| 2026-04-24 | Adopted all tokens from `~/sneak-in/DESIGN.md` (Archive) for sibling-product consistency | Two proposals independently landed on the same editorial direction. Archive's tokens are calibrated (Cron Orange tribute, Geist designed pair, explicit type scale). Unified visual identity across notetaker (tool) and Archive (reader) is stronger than two near-identical-but-different systems. |
| 2026-04-24 | Accent: FF4700 (Cron Orange) | Inherited from Archive. Direct reference to the *Grid Systems* book cover, cited in the Garry Tan hero lecture — a tribute designers (primary audience for the Archive; secondary for notetaker) will clock. |
| 2026-04-24 | Serif body (Fraunces Text) for Result view | The result view IS a reader; it should read like a book. Serif body is unusual for web apps but correct for this surface. |
| 2026-04-24 | Typewriter progress instead of spinner | The tool's job is to produce prose; the Processing view should communicate that by writing prose in front of the user at realistic reading speed, not by animating a neutral geometry. |
| 2026-04-24 | Marginalia rail for Result view on ≥ 900px viewports | Notetaker's in-app Result view has structured metadata (timestamps aligned to transcript spans) that the Archive's published pages do not. The rail treats timestamps as scholar's apparatus, visually reinforcing "this is a reading tool, not a meeting tool." |
| 2026-04-24 | No spinners, no toasts, no hover-lift | All three are category clichés for SaaS productivity tools. Notetaker is not that. |
