# Notetaker

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Convert video lectures into well-structured markdown notes — preserving the speaker's voice with minimal edits.**

Inspired by Andrej Karpathy's challenge to create an automated note taker: https://x.com/karpathy/status/1760740503614836917

## Demo

<table>
<tr>
<td align="center"><strong>Video</strong></td>
<td align="center"><strong>Notes</strong></td>
</tr>
<tr>
<td><img src="image.png" alt="Video lecture" width="500" /></td>
<td><img src="demo-notes.png" alt="Generated markdown notes" width="500" /></td>
</tr>
</table>

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, uvicorn |
| LLM | Claude Opus 4.6 (orchestrator), Sonnet (vision + assembly), Haiku (cleanup) |
| Speech-to-text | OpenAI Whisper API |
| Video processing | ffmpeg + ffprobe |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker, Render |

## Features

- **Lossless translation** — preserves the speaker's voice, no paraphrasing
- **Parallel pipeline** — audio + frame extraction run concurrently, Whisper chunks transcribe in parallel
- **Opus 4.6 agent** — orchestrates cleanup, visual analysis, and assembly, then reviews its own output
- **Smart visual handling** — extracts code/math from frames, keeps diagrams as images, skips talking-head frames
- **Checkpointed jobs** — resume interrupted jobs from where they left off
- **Rate limiting & job cleanup** — production-ready with automatic expiration
- **API docs** — interactive OpenAPI docs at `/docs`

## Quick Start

```bash
uv sync

# Set API keys in .env or environment
export OPENAI_API_KEY="..."       # Whisper transcription
export ANTHROPIC_API_KEY="..."    # Claude API (or use AWS Bedrock below)

# Or use AWS Bedrock:
# export AWS_REGION="us-east-1"
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."

# Start server
uv run notetaker
# Open http://localhost:8000
```

### Docker

```bash
docker build -t notetaker .
docker run -p 8000:8000 --env-file .env notetaker
```

## Architecture

```
 Upload ──► ffmpeg ──┬──► Audio (MP3) ──► Whisper API ──► Transcript
                     └──► Frames (JPG) ─────────────────────┐
                                                             ▼
                                              Opus 4.6 Agent Loop
                                            ┌────────────────────┐
                                            │ 1. Haiku: grammar  │
                                            │ 2. Sonnet: vision  │
                                            │ 3. Sonnet: assembly│
                                            │ 4. Opus: review    │
                                            └────────┬───────────┘
                                                     ▼
                                              Markdown Output
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## Design Decisions

- **Minimal editing** — the grammar cleanup stage uses Haiku with strict instructions to only fix errors, never rewrite. This preserves authenticity.
- **Agent-based orchestration** — Opus 4.6 plans the work, delegates to cheaper models, and reviews the output. This keeps costs down while maintaining quality.
- **Hybrid frame extraction** — interval-based (every 10s) + scene-change detection, deduplicated. Catches both regular slides and visual transitions.
- **Checkpoint/resume** — every pipeline stage saves progress to disk. A failed job can resume from exactly where it stopped.

## Project Structure

```
src/           — Python package (app, pipeline, LLM agent)
static/        — Frontend (HTML, CSS, JS)
tests/         — Unit tests + API smoke tests
docs/          — Architecture documentation
output/jobs/   — Runtime job data (gitignored)
```

## Development

```bash
uv sync --dev

# Run tests (no API keys needed)
uv run pytest -m "not smoke" -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Requirements

- Python 3.10+
- **ffmpeg** + **ffprobe** (system install)
- **OPENAI_API_KEY** for Whisper STT
- **ANTHROPIC_API_KEY** or **AWS credentials** for Claude (Opus 4.6, Sonnet, Haiku)

## License

MIT
