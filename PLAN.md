# MVP Launch Plan — Notetaker

## Context
Portfolio project for job search. Small user base. Goal: demonstrate engineering quality, not scale.

---

## Priority 1 — Must-have for MVP launch

### 1. Enable the built-in FastAPI docs (`/docs`)
**Why:** Instant, professional API documentation. Shows you know OpenAPI.
**Work:**
- Add `title`, `description`, `version` metadata to the FastAPI app
- Add `summary` / `response_model` annotations to each route
- Ensure `/docs` and `/redoc` are accessible (FastAPI enables them by default, verify they're not disabled)

### 2. Add a `/health` endpoint
**Why:** Standard practice for any deployed service. Required by every orchestrator (Docker, Railway, Render, etc.).
**Work:**
- Add `GET /health` returning `{"status": "ok", "version": "0.2.0", "jobs_active": N}`
- Add `HEALTHCHECK` instruction to the Dockerfile

### 3. Harden the Dockerfile for production
**Why:** Running as root in a container is a red flag for any reviewer.
**Work:**
- Add a non-root `notetaker` user, switch with `USER`
- Add `HEALTHCHECK` (from item 2)
- Add `.dockerignore` to exclude `.git`, `output/`, `tests/`, etc.

### 4. Add job expiration / cleanup
**Why:** Without this, disk fills up. Shows you think about operational concerns.
**Work:**
- Add a background task (FastAPI `on_event("startup")` or `lifespan`) that periodically removes job directories older than a configurable TTL (default 24h)
- Show remaining time in the job status response

### 5. Add basic rate limiting
**Why:** Prevents abuse, shows security awareness.
**Work:**
- Add `slowapi` or a simple middleware: limit uploads to e.g. 5/hour per IP
- Return `429 Too Many Requests` with a `Retry-After` header

### 6. Add request timeouts
**Why:** Prevents jobs from hanging forever, which would block the 2-job semaphore.
**Work:**
- Add a wall-clock timeout (e.g. 30 min) to the pipeline runner; cancel and mark failed if exceeded
- Add `timeout` parameter to `subprocess.run` calls for ffmpeg (e.g. 5 min)

### 7. Improve error UX on the frontend
**Why:** Users need to understand what went wrong and what to do about it.
**Work:**
- Map backend error codes to user-friendly messages (e.g. 413 → "Video is too large (max 2 GB)", 429 → "Too many requests, try again in X minutes")
- Add a "try again" button that resets the form cleanly
- Show which pipeline stage failed (audio extraction, transcription, etc.)

---

## Priority 2 — High-impact polish (makes the portfolio shine)

### 8. Add a demo / landing section
**Why:** Visitors who don't have a video to upload need to see what the tool produces.
**Work:**
- Add a "See an example" link/button on the main page
- Ship a short sample markdown output in `static/` and display it in a modal or separate view
- Optionally: embed a 30s screen recording GIF showing the upload → result flow

### 9. Add unit + integration tests
**Why:** Any serious portfolio project should have tests. Current coverage is smoke-only.
**Work:**
- Add `pytest` config in `pyproject.toml` (`[tool.pytest.ini_options]`)
- Unit tests for pure functions: `segmenter.find_natural_pause()`, `audio.probe_video()`, `assembler.copy_referenced_frames()`, filename sanitization
- Integration test: mock the OpenAI/Anthropic APIs, feed a small fixture through the pipeline, assert markdown output structure
- Add a `Makefile` or `justfile` with `test`, `lint`, `format` targets

### 10. Add CI with GitHub Actions
**Why:** Green badge on the repo = instant credibility.
**Work:**
- `.github/workflows/ci.yml`: install deps, run `ruff check`, run `ruff format --check`, run `pytest` (unit tests only, skip API smoke tests)
- Badge in README

### 11. Improve the README for portfolio visitors
**Why:** The README is the first thing a hiring manager sees.
**Work:**
- Add a hero section: one-line description + screenshot/GIF
- Add a "Tech Stack" section with badges (Python, FastAPI, Claude, Whisper, ffmpeg, Docker)
- Add "Architecture" section with a simplified diagram (link to ARCHITECTURE.md for details)
- Add a "What I learned" or "Design decisions" section (brief, 3-4 bullets)
- Add deploy instructions (Docker one-liner, Railway/Render button)

### 12. Add a simple deploy target
**Why:** "Click here to see it live" is the most powerful line on a portfolio project.
**Work:**
- Add a `railway.toml` or `render.yaml` for one-click deploy
- Or document the Docker deploy: `docker run -p 8000:8000 --env-file .env notetaker`
- Add deploy badge / link in README

---

## Priority 3 — Nice-to-have (if time permits)

### 13. Add dark mode
**Work:** CSS `prefers-color-scheme` media query + toggle button. Mostly CSS variables.

### 14. Add progress estimation
**Work:** Estimate total time from video duration, show "~X minutes remaining" on the progress bar.

### 15. Add optional webhook/email notification
**Work:** Accept an optional `webhook_url` or `email` field; POST result URL on completion.

### 16. Add job history with SQLite
**Work:** Replace in-memory dict with SQLite. Show past jobs in a sidebar. Survives restarts.

---

## Implementation Order (suggested)

**Phase 1 — Backend hardening (items 1-6):** ~1 session
These are all backend-only, independent changes that can be done in parallel.

**Phase 2 — Frontend + UX (items 7-8):** ~1 session
Improve the user-facing experience.

**Phase 3 — Testing + CI (items 9-10):** ~1 session
Add tests, add CI, get the green badge.

**Phase 4 — Portfolio polish (items 11-12):** ~1 session
README, deploy target, make it shine for hiring managers.

**Phase 5 — Nice-to-haves (items 13-16):** only if time permits.
