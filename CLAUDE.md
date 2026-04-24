# Notetaker Project

## Overview
Web application that converts MP4 video lectures into markdown files. Upload a video, get back a well-structured document preserving the speaker's voice with minimal edits.

## Project Structure
```
notetaker/                      (repo root)
├── pyproject.toml              # Project config, dependencies (uv)
├── CLAUDE.md                   # This file - dev instructions
├── src/                        # Main package
│   ├── __init__.py
│   ├── app.py                  # FastAPI backend, routes, job management
│   ├── audio.py                # ffmpeg: MP4→MP3, video probing
│   ├── transcribe.py           # OpenAI Whisper API transcription
│   ├── frames.py               # ffmpeg: hybrid frame extraction
│   ├── segmenter.py            # Time-window segmentation + frame alignment
│   ├── llm.py                  # Opus 4.6 agent: cleanup, vision, assembly, review
│   └── assembler.py            # Markdown generation + image management
├── static/                     # Frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                      # Smoke tests
│   └── test_api_connections.py
├── docs/
│   └── ARCHITECTURE.md
└── output/jobs/                # Runtime job data (gitignored)
```

## Quick Start
```bash
uv sync

# Set API keys
export OPENAI_API_KEY="your-key"      # For Whisper transcription
export ANTHROPIC_API_KEY="your-key"   # For LLM processing

# Start server
uv run notetaker
# Open http://localhost:8000
```

## Pipeline
1. **Audio extraction** — ffmpeg converts MP4 → MP3 (compact for API)
2. **Transcription** — OpenAI Whisper API with word-level timestamps
3. **Frame extraction** — ffmpeg hybrid: interval (10s) + scene change detection
4. **Segmentation** — Group transcript + frames into ~3-min segments at natural pauses
5. **LLM Stage 1: Grammar cleanup** — Haiku model, minimal edits only
6. **LLM Stage 2: Visual analysis** — Sonnet with vision, classify frames, extract code/math
7. **LLM Stage 3: Structure + assembly** — Sonnet determines headings, assembles markdown

## Key Design Decisions

### Minimal Transcript Editing
The grammar cleanup stage uses a fast model (Haiku) with strict instructions to ONLY:
- Fix grammar errors
- Remove filler words (um, uh)
- Fix transcription errors
- Format code refs with backticks

It must NOT rewrite, paraphrase, or restructure anything.

### Smart Visual Handling
- Code/math frames → extracted as text (code blocks, LaTeX)
- Diagrams → kept as images
- Speaker-referenced visuals (histograms, terminal output, plots) → kept as images
- Talking head frames → skipped entirely

### Output Format
Markdown (.md) with title-case headings, first-person voice preserved.

## System Requirements
- **ffmpeg** + **ffprobe** (system install)
- **OPENAI_API_KEY** for Whisper STT
- **ANTHROPIC_API_KEY** for Claude LLM (or AWS Bedrock credentials)

## Code Style
- Use `uv` for package management (not pip)
- Python 3.10+
- Type hints throughout
- Ruff for linting

## Design System
Always read `DESIGN.md` before making any visual or UI decisions (fonts, colors, spacing, layout, motion, component styling). All tokens live there; do not redefine them ad hoc in `static/style.css` or `static/app.js`.

Notetaker's design system is unified with the sibling Archive project (`~/sneak-in/DESIGN.md`) — same palette, typography, and anti-slop rules. `DESIGN.md` documents additions specific to notetaker's tool surfaces (upload, processing, result view, job list) that the Archive does not have.

Do not deviate from `DESIGN.md` without explicit user approval. In `/qa` and `/design-review` mode, flag any code that doesn't match it.

## gstack
- Use the `/browse` skill from gstack for all web browsing.
- Never use `mcp__claude-in-chrome__*` tools.
- Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.
