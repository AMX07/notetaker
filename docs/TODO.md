# TODO: Notetaker Development Roadmap

## Immediate Next Steps

### 1. Test Full Convert Pipeline
```bash
export ANTHROPIC_API_KEY="your-key"
uv run notetaker convert "https://youtube.com/watch?v=PaCmpygFfXo" \
  -t "The spelled-out intro to language modeling: building makemore" \
  -o output/makemore_part1_auto.md \
  -v
```

**Success criteria:**
- [ ] No errors during execution
- [ ] Output matches quality of manual `makemore_part1_lossless.ipynb`
- [ ] All 23 chapters present
- [ ] Voice preserved (first-person, asides)

### 2. Add Code Extraction
Currently, code isn't automatically extracted from source notebooks. Need to:
- [ ] Implement heuristics to map code cells to chapters
- [ ] Options: keyword matching, cell comments, manual mapping file
- [ ] Test with `makemore_part1_bigrams (1).ipynb`

### 3. Refine Claude Prompts
After testing, iterate on prompts in `notetaker/llm.py`:
- [ ] Tune cleanup aggressiveness
- [ ] Improve code integration placement
- [ ] Add handling for edge cases (diagrams, whiteboard moments)

## Phase 2: More Karpathy Videos

### Add Chapter Data
```python
# In notetaker/presets/karpathy.py
MAKEMORE_PART2_CHAPTERS = """
00:00:00 intro
...
""".strip()
```

Videos needing chapters:
- [ ] Makemore Part 2: MLP
- [ ] Makemore Part 3: BatchNorm
- [ ] Makemore Part 4: Backprop Ninja
- [ ] Makemore Part 5: WaveNet
- [ ] Let's build GPT
- [ ] Micrograd

### Validate Against lecture.md
The Tokenizer video has an existing `lecture.md`. Use it to validate our pipeline:
```bash
uv run notetaker convert "https://youtube.com/watch?v=zduSFxRajkE" \
  -o output/tokenizer_auto.md

# Compare
diff output/tokenizer_auto.md reference/lecture.md
```

## Phase 3: Enhancements

### Code Integration Improvements
- [ ] Parse original notebook for code cells
- [ ] Match code to chapters via:
  - Comment markers (`# Section: intro`)
  - Keyword detection
  - Manual mapping YAML file
- [ ] Handle code that spans multiple chapters

### Image/Frame Extraction
- [ ] Detect moments needing visuals (diagrams, whiteboard)
- [ ] Extract frames using ffmpeg
- [ ] Embed as base64 or save to assets/
- [ ] Only include when they add info not in text

### Output Formats
- [ ] Jupyter notebook (.ipynb) output option
- [ ] HTML output with syntax highlighting
- [ ] PDF via pandoc

### CLI Improvements
- [ ] `--preset karpathy` flag to auto-detect video
- [ ] `--dry-run` to preview without API calls
- [ ] `--chapters-only` to just extract/verify chapters
- [ ] Progress bars for long videos

## Phase 4: Quality Assurance

### Automated Testing
```bash
# Add to pyproject.toml [project.optional-dependencies]
dev = ["pytest>=7.0.0"]
```

Tests needed:
- [ ] `test_transcript.py`: Fetch, parse, extract video ID
- [ ] `test_chapters.py`: Parse timestamps, load presets
- [ ] `test_segmenter.py`: Split by chapters, keyword fallback
- [ ] `test_llm.py`: Mock Claude responses
- [ ] `test_assembler.py`: Markdown generation

### Integration Tests
- [ ] Full pipeline on Makemore Part 1 (compare to manual output)
- [ ] Full pipeline on Tokenizer (compare to lecture.md)

## Known Issues

### youtube-transcript-api Changes
API changed in v1.2.3+ - must instantiate class:
```python
api = YouTubeTranscriptApi()
fetched = api.fetch(video_id)
```
**Status**: Fixed in `transcript.py`

### yt-dlp Not Required
Chapter fetching works without yt-dlp if video has preset.
**Status**: Handled via fallback chain

### Long Transcripts
2-hour videos may hit Claude context limits.
**TODO**: Implement chunking in `llm.py`

## File Inventory

```
notetaker/
├── docs/
│   ├── PROCESS.md      # This conversion process doc
│   └── TODO.md         # This file
├── notetaker/
│   ├── cli.py          # ✅ Working
│   ├── transcript.py   # ✅ Working
│   ├── chapters.py     # ✅ Working (presets + yt-dlp)
│   ├── segmenter.py    # ✅ Working
│   ├── llm.py          # ⚠️ Needs end-to-end testing
│   ├── assembler.py    # ⚠️ Needs end-to-end testing
│   └── presets/
│       └── karpathy.py # ✅ Makemore P1 + Tokenizer chapters
└── output/
    ├── makemore_part1_blog.ipynb     # Old approach
    └── makemore_part1_lossless.ipynb # Manual lossless (reference)
```
