# Notetaker — System Architecture

## Overview

Notetaker is a web application that converts MP4 video lectures into structured markdown documents. Users upload a video through a browser UI, and the server runs a multi-stage pipeline that transcribes, extracts frames, and uses LLMs to produce a clean, well-structured document.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Browser (Client)                       │
│                                                          │
│  Upload MP4 ──→ Poll Progress ──→ Download .md           │
│  POST /api/convert  GET /api/status  GET /api/download   │
└────────┬──────────────┬──────────────────┬───────────────┘
         │              │                  │
┌────────▼──────────────▼──────────────────▼───────────────┐
│                FastAPI Backend (app.py)                    │
│                                                          │
│  • Routes: /api/convert, /api/status, /api/download      │
│  • Job manager: in-memory dict, rebuilt from disk         │
│  • Background thread per job                             │
│  • Checkpoint controller: save/load/resume               │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                Pipeline (per job)                          │
│                                                          │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │ audio.py │──→│transcribe.py│──→│    frames.py     │  │
│  │ ffmpeg   │   │OpenAI Whisper│  │    ffmpeg         │  │
│  │→ MP3     │   │→ transcript  │  │→ JPG frames       │  │
│  └──────────┘   └─────────────┘   └──────────────────┘  │
│       │                │                   │             │
│       └────────────────┼───────────────────┘             │
│                        ▼                                 │
│              ┌──────────────────┐                        │
│              │  segmenter.py    │                        │
│              │  group transcript│                        │
│              │  + frames into   │                        │
│              │  time segments   │                        │
│              └────────┬─────────┘                        │
│                       ▼                                  │
│              ┌──────────────────┐                        │
│              │    llm.py        │                        │
│              │                  │                        │
│              │ Stage 1: Cleanup │──→ Anthropic (Haiku)   │
│              │   grammar only   │                        │
│              │                  │                        │
│              │ Stage 2: Vision  │──→ Anthropic (Sonnet)  │
│              │   classify frames│                        │
│              │   extract code   │                        │
│              │                  │                        │
│              │ Stage 3: Assembly│──→ Anthropic (Sonnet)  │
│              │   add headings   │                        │
│              │   build markdown │                        │
│              └────────┬─────────┘                        │
│                       ▼                                  │
│              ┌──────────────────┐                        │
│              │  assembler.py    │                        │
│              │  → result.md     │                        │
│              └──────────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Pipeline Stages

| Step | Module | Input | Output | External Dep |
|------|--------|-------|--------|-------------|
| 1. Audio | `audio.py` | MP4 file | MP3 (64kbps mono) | ffmpeg |
| 2. Transcribe | `transcribe.py` | MP3 | `transcript.json` | OpenAI Whisper API |
| 3. Frames | `frames.py` | MP4 file | JPG frames | ffmpeg |
| 4. Segment | `segmenter.py` | transcript + frames | `segments.json` | — |
| 5. LLM | `llm.py` | segments | `llm_results/*.json` | Anthropic API |
| 6. Assemble | `assembler.py` | LLM results | `result.md` | — |

## Checkpoint System

Each pipeline step saves its output and updates `checkpoint.json`. On resume:

```
For each step:
  if checkpoint says "completed" → load saved output, skip
  if checkpoint says "failed" or "pending" → execute step
```

This means:
- Transcription fails → audio + frames aren't redone
- LLM fails on segment 15 → segments 1-14 aren't reprocessed
- Server restarts → completed jobs are immediately downloadable

### Job directory layout
```
output/jobs/<job_id>/
├── checkpoint.json      # Step status tracking
├── video.mp4            # Uploaded file
├── audio.mp3            # Step 1
├── transcript.json      # Step 2
├── frames/              # Step 3
├── segments.json        # Step 4
├── llm_results/         # Step 5 (per-segment)
└── result.md            # Step 6 (final output)
```

## Component Responsibilities

### `app.py` — Web server + job orchestration
- Receives uploads, creates job directories
- Runs pipeline in background threads
- Manages checkpoints (save, load, resume)
- Restores job state from disk on startup
- Serves static frontend + API endpoints

### `audio.py` — Audio extraction
- Probes video metadata via ffprobe (duration, resolution, fps)
- Extracts audio track as MP3 via ffmpeg (64kbps mono for small size)

### `transcribe.py` — Speech-to-text
- Calls OpenAI Whisper API with `verbose_json` for word-level timestamps
- Splits audio into <=25MB chunks if needed, merges with offset timestamps

### `frames.py` — Frame extraction
- Hybrid strategy: interval-based (every 10s) + scene-change detection
- Deduplicates frames within 2s of each other
- Resizes to max 1024px wide (reduces API costs)

### `segmenter.py` — Grouping
- Divides transcript into ~3-minute segments at natural pauses
- Aligns frames to segments by timestamp

### `llm.py` — LLM processing (3 stages)
- **Stage 1 (Haiku)**: Grammar cleanup — minimal edits only, preserves speaker voice
- **Stage 2 (Sonnet + vision)**: Classify frames as code/math/diagram/talking_head, extract text from code/math frames
- **Stage 3 (Sonnet)**: Determine headings/subheadings, assemble final markdown with code blocks and image references

### `assembler.py` — Output generation
- Creates final markdown document with title and attribution
- Copies referenced frame images to output assets directory

## Data Flow

```
MP4 ──ffmpeg──→ MP3 ──OpenAI API──→ TranscriptSegments
MP4 ──ffmpeg──→ ExtractedFrames
                    │                        │
                    └──── segmenter ──────────┘
                              │
                       VideoSegments
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
                 Cleanup   Vision    Assembly
                 (Haiku)  (Sonnet)  (Sonnet)
                    │         │          │
                    └─────────┼──────────┘
                              │
                         result.md
```

## External Dependencies

| Dependency | Purpose | Required |
|-----------|---------|----------|
| ffmpeg + ffprobe | Audio extraction, frame extraction | Yes (system) |
| OpenAI API | Whisper speech-to-text | Yes (OPENAI_API_KEY) |
| Anthropic API | Claude for cleanup, vision, assembly | Yes (ANTHROPIC_API_KEY) |
