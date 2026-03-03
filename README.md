# Notetaker

Convert video lectures into lossless text-based markdown, preserving the speaker's teaching voice.

## Features

- **Lossless translation**: Preserves the speaker's actual words, not paraphrased summaries
- **Automatic transcript fetching**: Pulls transcripts directly from YouTube
- **Chapter-aware segmentation**: Splits content by video chapters
- **Claude-powered cleanup**: Removes filler words while preserving voice
- **Karpathy presets**: Built-in support for Andrej Karpathy's "Neural Networks: Zero to Hero" series

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd notetaker

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Fetch Transcript
```bash
uv run notetaker transcript "https://youtube.com/watch?v=PaCmpygFfXo"
```

### View Chapters
```bash
uv run notetaker chapters "https://youtube.com/watch?v=PaCmpygFfXo"
```

### Convert Video to Markdown
```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Convert a video
uv run notetaker convert "https://youtube.com/watch?v=PaCmpygFfXo" \
  --title "Building makemore" \
  --output lecture.md
```

### Options
```
uv run notetaker convert --help

Options:
  -c, --chapters PATH   Path to chapters file (optional)
  -n, --notebook PATH   Source Jupyter notebook for code extraction
  -o, --output PATH     Output markdown file path
  -t, --title TEXT      Document title
  -m, --model TEXT      Claude model to use (default: claude-sonnet-4-20250514)
  -v, --verbose         Verbose output
```

## Supported Videos

### Andrej Karpathy's Series (Built-in Presets)
| Video | Status |
|-------|--------|
| Makemore Part 1: Bigrams | ✅ Full chapter data |
| Makemore Part 2: MLP | 🔲 TODO |
| Makemore Part 3: BatchNorm | 🔲 TODO |
| Makemore Part 4: Backprop Ninja | 🔲 TODO |
| Makemore Part 5: WaveNet | 🔲 TODO |
| Let's build GPT | 🔲 TODO |
| GPT Tokenizer | ✅ Full chapter data |
| Micrograd | 🔲 TODO |

## How It Works

1. **Fetch transcript** from YouTube using `youtube-transcript-api`
2. **Get chapters** from presets, yt-dlp, or manual file
3. **Segment transcript** by chapter timestamps
4. **Clean with Claude**: Remove filler, fix errors, format code references
5. **Assemble markdown** with chapter headers and cleaned text

## Output Quality

The goal is "lossless" translation - you should be able to learn everything from the text that you could from the video. The output:

- Keeps first-person voice ("I would like to", "let's")
- Preserves personality and teaching asides
- Removes only filler words ("um", "uh")
- Formats code references with backticks
- Uses video chapter titles as section headers

## Requirements

- Python 3.10+
- `ANTHROPIC_API_KEY` environment variable (for LLM processing)
- Optional: `yt-dlp` for auto-fetching chapters from videos without presets

## License

MIT
