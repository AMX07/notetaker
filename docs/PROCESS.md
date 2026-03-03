# Video-to-Text Conversion Process

This document details the complete process for converting video lectures into lossless text-based markdown.

## The Problem: Lossy vs Lossless

### Lossy Approach (What NOT to Do)
```markdown
## 2. Loading and Exploring the Dataset

Our dataset is `names.txt` - a very large dataset of names that I found randomly
on a government website. There are about 32,000 names in it.

We'll load it by reading the entire file as a string, then splitting it into
individual words.
```

**Problems:**
- Third-person voice ("Our dataset", "We'll load")
- Paraphrased/summarized content
- Lost teaching personality
- Feels like documentation, not a lecture

### Lossless Approach (What TO Do)
```markdown
## reading and exploring the dataset

The first thing is I would like to basically load up the dataset `names.txt`.
So we're going to open up `names.txt` for reading and we're going to read in
everything into a massive string. And then because it's a massive string we'd
only like the individual words and put them in the list, so let's call split
lines on that string to get all of our words as a Python list of strings.

So basically we can look at for example the first 10 words and we have that
it's a list of emma, olivia, eva and so on. This list actually makes me feel
that this is probably sorted by frequency but okay.
```

**Why this works:**
- First-person voice preserved ("I would like to", "we're going to")
- Natural asides kept ("This list actually makes me feel...")
- Reads like the speaker is teaching you directly
- Section header matches video chapter exactly (lowercase)

## The Reference: lecture.md

The gold standard is Andrej Karpathy's [lecture.md](https://github.com/karpathy/minbpe/blob/master/lecture.md) from the minbpe repo. Key patterns:

1. **Direct translation** - Not paraphrased, actual words from video
2. **Conversational tone** - "Hi everyone", "let's", "I'd like to"
3. **Minimal cleanup** - Only removes filler ("um", "uh")
4. **Code integrated inline** - Where it appears in the video
5. **Images sparse** - Only when they add visual understanding

## Step-by-Step Process

### Step 1: Fetch Transcript
```python
from notetaker.transcript import fetch_transcript

transcript = fetch_transcript("https://youtube.com/watch?v=PaCmpygFfXo")
# Returns: [{"text": "hi everyone", "start": 0.0, "duration": 2.5}, ...]
```

The transcript comes with timestamps, which we use for chapter segmentation.

### Step 2: Get Chapters
```python
from notetaker.chapters import fetch_chapters_from_youtube

chapters = fetch_chapters_from_youtube("https://youtube.com/watch?v=PaCmpygFfXo")
# Returns: [Chapter(title="intro", start_seconds=0), ...]
```

Chapter sources (priority order):
1. Built-in presets (for known videos)
2. yt-dlp metadata
3. Manual chapter file

### Step 3: Segment Transcript
```python
from notetaker.segmenter import segment_transcript_by_chapters

segments = segment_transcript_by_chapters(transcript, chapters)
# Returns: {"intro": "transcript text...", "reading dataset": "..."}
```

### Step 4: Clean with Claude
```python
from notetaker.llm import clean_transcript_chunk

cleaned = clean_transcript_chunk(raw_text)
```

**Cleanup rules:**
- Remove filler: "um", "uh", "sort of" (when meaningless)
- Fix transcription errors
- Format code references: `names.txt`, `torch.tensor`
- Add paragraph breaks every 3-5 sentences

**Preserve:**
- First-person voice
- Personality and humor
- Teaching asides
- Natural flow

### Step 5: Assemble Markdown
```python
from notetaker.llm import assemble_chapter

markdown = assemble_chapter(
    chapter_title="reading and exploring the dataset",
    transcript_text=cleaned,
    code_snippets=["words = open('names.txt').read().splitlines()"]
)
```

Output structure:
```markdown
## chapter-title

[Cleaned transcript text with paragraph breaks]

```python
# Code where the speaker introduces it
```

[More transcript text]
```

## Quality Checklist

After conversion, verify:

1. **Losslessness**: Can you learn everything from text that you could from video?
2. **Voice**: Does it sound like the speaker teaching you directly?
3. **Code**: Are code blocks placed where they're discussed?
4. **Skimmability**: Can you scan headers to find topics?

## Transcript API Notes

The `youtube-transcript-api` changed in v1.2.3+. Must instantiate:

```python
# Old (broken)
YouTubeTranscriptApi.get_transcript(video_id)

# New (correct)
api = YouTubeTranscriptApi()
fetched = api.fetch(video_id, languages=['en'])
```

## Chapter Mapping Strategy

When timestamps aren't available, use keyword markers:

| Chapter | Search Keywords |
|---------|-----------------|
| intro | "hi everyone", "what is makemore" |
| reading dataset | "load up the dataset", "names.txt" |
| exploring bigrams | "bi-gram language model", "zip" |
| counting (dict) | "dictionary", "b dot get" |
| ... | ... |

## Claude Prompts

### Cleanup Prompt
```
You are a transcript editor. Your job is to clean up video lecture transcripts
while PRESERVING the speaker's voice and teaching style.

RULES:
1. KEEP first-person voice ("I would like to", "let's")
2. KEEP personality and natural asides
3. REMOVE filler words: "um", "uh", "sort of" (when meaningless)
4. FORMAT code references with backticks
5. ADD paragraph breaks for readability
6. DO NOT paraphrase or rewrite
```

### Assembly Prompt
```
Given a chapter title, cleaned transcript, and code snippets, create a markdown
section that:
1. Uses chapter title as ## heading (lowercase)
2. Places transcript with natural paragraph breaks
3. Inserts code blocks where speaker introduces them
4. Preserves speaker's voice completely
```
