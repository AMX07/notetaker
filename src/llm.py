"""LLM agent for video-to-markdown processing.

Opus 4.6 orchestrates worker tools (cleanup, visual analysis, assembly)
and reviews the output, requesting revisions if needed.
"""

import base64
import difflib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .assembler import create_markdown_document, save_markdown, copy_referenced_frames
from .segmenter import VideoSegment

logger = logging.getLogger("notetaker")

_RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)

_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    reraise=True,
)


def _call_anthropic(client, **kwargs):
    """Call client.messages.create with retry on transient errors."""
    return _api_retry(client.messages.create)(**kwargs)


# ---------------------------------------------------------------------------
# Data classes for LLM output
# ---------------------------------------------------------------------------


@dataclass
class VisualContent:
    """Result of analyzing a single video frame."""

    frame_path: Path
    timestamp: float
    category: str  # "code", "math", "diagram", "visual_reference", "text", "talking_head", "other"
    extracted_text: str | None
    description: str | None
    include_as_image: bool


@dataclass
class ProcessedSegment:
    """A segment after all LLM processing."""

    index: int
    cleaned_text: str
    visuals: list[VisualContent]


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------


def get_client(use_bedrock: Optional[bool] = None):
    """Get Anthropic client, auto-detecting Bedrock vs direct API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Please install anthropic: uv add anthropic")

    if use_bedrock is None:
        has_aws = bool(os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
        has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

        if has_aws:
            use_bedrock = True
        elif has_anthropic_key:
            use_bedrock = False
        else:
            raise ValueError(
                "No API configured. Set either:\n"
                "  - ANTHROPIC_API_KEY for direct Anthropic API, or\n"
                "  - AWS_REGION + AWS credentials for Bedrock"
            )

    if use_bedrock:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if not region:
            raise ValueError("AWS_REGION or AWS_DEFAULT_REGION must be set for Bedrock")
        return anthropic.AnthropicBedrock(aws_region=region)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return anthropic.Anthropic(api_key=api_key)


def get_model_id(model: str, use_bedrock: bool) -> str:
    """Convert model name to appropriate ID for the client."""
    if not use_bedrock:
        return model

    # Use cross-region inference profiles (us.anthropic.*) — required for on-demand
    bedrock_models = {
        "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1",
        "claude-sonnet-4-6": "arn:aws:bedrock:us-east-1:744741211997:inference-profile/us.anthropic.claude-sonnet-4-6",
        "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
        "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-3-5-sonnet-20241022": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku-20241022": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    }
    return bedrock_models.get(model, f"us.anthropic.{model}-v1:0")


def _is_bedrock(client) -> bool:
    return "Bedrock" in type(client).__name__


# ---------------------------------------------------------------------------
# Worker prompts
# ---------------------------------------------------------------------------

CLEANUP_SYSTEM_PROMPT = """You are a transcript copy-editor. You make ONLY the following corrections:

1. Fix grammar errors (subject-verb agreement, tense, articles).
2. Add missing punctuation: insert commas, periods, and other punctuation where the speaker's natural pauses and clause boundaries indicate them. Split run-on sentences at natural clause boundaries.
3. Remove meaningless filler: "um", "uh", "you know", "like" (when filler), "right" (when filler). Reduce excessive sentence-initial "So" — keep the first in a sequence but remove repetitive ones.
4. Remove word-level stutters, false starts, and immediate self-repetitions (e.g., "we are building X as part of X will do Y" → "we are building X which will do Y").
5. Fix obvious transcription errors (wrong homophones, garbled words).
6. Format code/technical references with backticks: `variable_name`, `torch.tensor`.
7. Add paragraph breaks every 3-5 sentences for readability.

CRITICAL — DO NOT:
- Rewrite or paraphrase — keep the speaker's own words and phrasing.
- Change the speaker's word choices, vocabulary, or level of formality.
- Remove personality, humor, or teaching asides.
- Add content or explanations that were not spoken.
- Change first-person voice.
- Add formatting like bullet points, bold, or headings.
- Remove meaningful repetition used for emphasis (only remove accidental self-repetition from speech).

Your output must be the cleaned transcript text and nothing else."""


VISUAL_ANALYSIS_PROMPT = """You are analyzing video frames from a lecture. You will receive a transcript excerpt showing what the speaker is discussing, followed by the video frames to analyze.

For each frame, output a JSON array with one object per frame.

For each frame determine:
1. **category**: one of "code", "math", "diagram", "visual_reference", "text", "talking_head", "other"
   - "visual_reference": use when the frame shows content the speaker explicitly references (e.g., terminal output, browser window, application UI, plot, histogram, training progress) but doesn't fit "code", "math", or "diagram".
2. **extracted_text**: For "code" frames, extract the code exactly. For "math" frames, write LaTeX. For "text" frames, extract key text. For others, null.
3. **description**: For "diagram" and "visual_reference" frames, a 1-2 sentence description of what the frame shows. For others, null.
4. **include_as_image**: Set to true ONLY for truly graphical content that cannot be represented as text:
   - Graphs, plots, charts, and histograms
   - Diagrams, flowcharts, and architectural drawings
   - Mathematical visualizations (not formulas — those go in extracted_text as LaTeX)
   Set to false for EVERYTHING else, including:
   - Talking-head frames (speaker's face)
   - Code on screen (extract as text instead)
   - Jupyter notebook cells and outputs (extract as text instead)
   - Terminal/console output (extract as text instead)
   - Browser windows, GitHub pages, documentation pages
   - Slides with mostly text content
   - Any frame whose useful content can be captured as extracted_text

5. **deduplication**: If multiple frames in this segment show essentially the same content (e.g., same web page, same code file, same slide with minor scrolling), set include_as_image to true ONLY for the most complete/representative frame. Set the others to false.

Output ONLY a JSON array, no explanation. Example:
[
  {"category": "code", "extracted_text": "import torch\\nx = torch.tensor([1,2,3])", "description": null, "include_as_image": false},
  {"category": "talking_head", "extracted_text": null, "description": null, "include_as_image": false},
  {"category": "visual_reference", "extracted_text": null, "description": "Terminal showing training loss decreasing over epochs", "include_as_image": true}
]"""


STRUCTURE_PROMPT = """You are assembling a lecture transcript into a well-structured markdown document.

You receive a sequence of cleaned transcript segments, each with visual content analysis (extracted code, math, diagram descriptions).

Your job:
1. Determine where to place headings (## h2) and subheadings (### h3) based on topic transitions.
   - Use the speaker's own words when possible for heading text.
   - Use title case for headings (capitalize major words, lowercase articles/prepositions like "a", "the", "in", "of", "for").
   - Place a heading when the speaker moves to a new major topic.
   - Only use subheadings if a section exceeds ~500 words with sub-topic changes.

2. Integrate visual content:
   - Insert ```python (or appropriate language) code blocks where the speaker discusses code.
   - Insert $$ LaTeX blocks where math formulas are discussed.
   - Insert ![description](path) for diagrams AND visual references marked with IMAGE: — these are frames the speaker explicitly referenced that readers need to see to follow along.
   - Use the extracted text from visual analysis — do NOT invent content.
   - Only include images for truly graphical content (graphs, plots, diagrams). Do NOT include images of code, notebook cells, terminal output, or text-heavy screens — those should already be extracted as text.
   - When multiple IMAGE: markers appear with very similar descriptions, include ONLY the last one.

3. Code explanation formatting:
   - When the speaker explains code step-by-step or line-by-line, break it into logical sections.
   - Each logical section (imports, data loading, model definition, training loop, etc.) gets its OWN code block.
   - The speaker's explanation for each section goes as regular prose IMMEDIATELY BEFORE its code block.
   - Do NOT create one giant code block containing all the code.
   - Do NOT put the speaker's explanations as comments inside code blocks.
   - The pattern is: explanation paragraph, then code block, then explanation paragraph, then code block.
   - When the speaker incrementally builds a class or function (adding one method/feature at a time), do NOT repeat the entire unchanged code each time. Show only the new or changed portion, using a comment like `# ... (previous code unchanged)` for context. Only show the full code when it first appears or when the final complete version is needed.
   - Example:

     First we import the necessary libraries and set up our data path.

     ```python
     import torch
     from pathlib import Path
     data_dir = Path("./data")
     ```

     Then we define the model architecture with two hidden layers.

     ```python
     model = nn.Sequential(
         nn.Linear(784, 256),
         nn.ReLU(),
         nn.Linear(256, 10),
     )
     ```

4. Preserve the speaker's voice completely. Do not paraphrase.

5. Formatting:
   - Ensure a blank line between every paragraph.
   - Ensure a blank line before and after every code block, math block, and image.
   - Do NOT collapse multiple paragraphs into a single dense block of text.

6. Output ONLY the final markdown content. No meta-commentary."""


REVISION_SYSTEM_PROMPT = """You are revising specific segments of a lecture transcript based on reviewer feedback.

You will receive:
- The current cleaned transcript text for one or more segments
- Specific revision instructions from a reviewer

Apply the requested revisions while preserving the speaker's voice. Output ONLY the revised text."""


# ---------------------------------------------------------------------------
# Worker functions (called by tool executors)
# ---------------------------------------------------------------------------


def clean_segment(
    segment_text: str,
    cleanup_model: str = "claude-haiku-4-5-20251001",
    use_bedrock: Optional[bool] = None,
) -> str:
    """Clean a transcript segment with minimal edits using a fast model."""
    client = get_client(use_bedrock=use_bedrock)
    model_id = get_model_id(cleanup_model, _is_bedrock(client))

    message = _call_anthropic(
        client,
        model=model_id,
        max_tokens=4096,
        system=CLEANUP_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<transcript>\n{segment_text}\n</transcript>"}],
    )
    return message.content[0].text


def _encode_frame(frame_path: Path) -> str:
    """Read and base64-encode a frame image."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(suffix, "image/jpeg")


def analyze_segment_visuals(
    segment: VideoSegment,
    vision_model: str = "claude-sonnet-4-6",
    use_bedrock: Optional[bool] = None,
) -> list[VisualContent]:
    """Analyze frames for a segment, classifying and extracting text."""
    if not segment.frames:
        return []

    client = get_client(use_bedrock=use_bedrock)
    model_id = get_model_id(vision_model, _is_bedrock(client))

    content: list[dict] = [
        {"type": "text", "text": f"<transcript>\n{segment.text}\n</transcript>\n\nAnalyze these frames from the lecture:"},
    ]

    for i, frame in enumerate(segment.frames):
        minutes = int(frame.timestamp // 60)
        seconds = int(frame.timestamp % 60)
        content.append(
            {
                "type": "text",
                "text": f'\n<frame index="{i + 1}" timestamp="{minutes:02d}:{seconds:02d}">',
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _media_type(frame.path),
                    "data": _encode_frame(frame.path),
                },
            }
        )
        content.append({"type": "text", "text": "</frame>"})

    message = _call_anthropic(
        client,
        model=model_id,
        max_tokens=4096,
        system=VISUAL_ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    response_text = message.content[0].text
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        response_text = response_text.rsplit("```", 1)[0]

    try:
        results = json.loads(response_text)
    except json.JSONDecodeError:
        return [
            VisualContent(
                frame_path=f.path,
                timestamp=f.timestamp,
                category="talking_head",
                extracted_text=None,
                description=None,
                include_as_image=False,
            )
            for f in segment.frames
        ]

    visuals = []
    for i, frame in enumerate(segment.frames):
        if i < len(results):
            r = results[i]
            visuals.append(
                VisualContent(
                    frame_path=frame.path,
                    timestamp=frame.timestamp,
                    category=r.get("category", "other"),
                    extracted_text=r.get("extracted_text"),
                    description=r.get("description"),
                    include_as_image=r.get("include_as_image", False),
                )
            )
        else:
            visuals.append(
                VisualContent(
                    frame_path=frame.path,
                    timestamp=frame.timestamp,
                    category="talking_head",
                    extracted_text=None,
                    description=None,
                    include_as_image=False,
                )
            )

    return visuals


def _deduplicate_images(segments: list[ProcessedSegment]) -> set[tuple[int, str]]:
    """Find near-duplicate images across segments. Returns set of (seg_index, frame_name) to suppress."""
    # Collect all include_as_image visuals sorted by timestamp
    all_images: list[tuple[int, VisualContent]] = []
    for seg in segments:
        for v in seg.visuals:
            if v.include_as_image and v.category != "talking_head":
                all_images.append((seg.index, v))
    all_images.sort(key=lambda x: x[1].timestamp)

    # Walk through and suppress earlier duplicates (keep later frame — may show more content)
    suppressed: set[tuple[int, str]] = set()
    i = 0
    while i < len(all_images):
        j = i + 1
        # Find the end of a run of near-duplicates
        while j < len(all_images):
            seg_i, vi = all_images[j - 1]
            seg_j, vj = all_images[j]
            desc_i = (vi.description or "").lower()
            desc_j = (vj.description or "").lower()
            if (
                vi.category == vj.category
                and (vj.timestamp - vi.timestamp) < 30.0
                and desc_i
                and desc_j
                and difflib.SequenceMatcher(None, desc_i, desc_j).ratio() > 0.5
            ):
                j += 1
            else:
                break
        # Suppress all but the last in the run
        for k in range(i, j - 1):
            seg_idx, v = all_images[k]
            suppressed.add((seg_idx, v.frame_path.name))
        i = j

    return suppressed


def _format_code_for_assembly(
    code: str, seg_index: int, recent_code: list[tuple[int, str]]
) -> str:
    """Return the code annotation for the assembly prompt.

    If the code is ≥60% similar (line-level) to a recently seen code block,
    this is an incremental update — only the added/changed lines are returned
    with a header directing the assembly model to use '# ... (previous code
    unchanged)' markers. This prevents the assembled document from repeating
    the full class/function multiple times when a speaker builds it up step
    by step.

    Returns the full code prefixed with a space when it is new/distinct.
    """
    for prev_seg_idx, prev_code in reversed(recent_code[-3:]):
        prev_lines = prev_code.splitlines()
        curr_lines = code.splitlines()
        ratio = difflib.SequenceMatcher(None, prev_lines, curr_lines).ratio()
        if ratio < 0.6:
            continue

        # Compute the changed/added lines using opcode-level diffing
        added: list[str] = []
        removed: list[str] = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, prev_lines, curr_lines
        ).get_opcodes():
            if tag in ("insert", "replace"):
                added.extend(curr_lines[j1:j2])
            if tag in ("delete", "replace"):
                removed.extend(prev_lines[i1:i2])

        if not added and not removed:
            return f" [DUPLICATE CODE from segment {prev_seg_idx + 1} — do not repeat in output]"

        header = (
            f" [INCREMENTAL UPDATE extending segment {prev_seg_idx + 1} code"
            f" — show only new/changed lines; use '# ... (previous code unchanged)'"
            f" to mark unchanged parts]"
        )
        if added:
            shown = added[:8]
            header += "\nNew/changed lines:\n" + "\n".join(f"+ {line}" for line in shown)
            if len(added) > 8:
                header += f"\n... ({len(added) - 8} more additions)"
        return header

    # No matching prior code — show in full
    return f" {code}"


def _build_assembly_prompt(segments: list[ProcessedSegment]) -> str:
    """Build the assembly prompt from processed segments."""
    suppressed = _deduplicate_images(segments)

    # Track recent code blocks to detect incremental class/function building.
    # list of (seg_index, code_text); capped at 5 entries.
    recent_code: list[tuple[int, str]] = []

    parts = []
    for seg in segments:
        parts.append(f"--- Segment {seg.index + 1} ---")
        parts.append(f"Transcript:\n{seg.cleaned_text}")

        if seg.visuals:
            visual_notes = []
            for v in seg.visuals:
                if v.category == "talking_head":
                    continue
                note = f"[{v.category}]"
                if v.extracted_text:
                    if v.category == "code":
                        note += _format_code_for_assembly(v.extracted_text, seg.index, recent_code)
                        recent_code.append((seg.index, v.extracted_text))
                        if len(recent_code) > 5:
                            recent_code.pop(0)
                    else:
                        note += f" {v.extracted_text}"
                if v.description:
                    note += f" Description: {v.description}"
                if v.include_as_image and (seg.index, v.frame_path.name) not in suppressed:
                    note += f" IMAGE: {v.frame_path.name}"
                visual_notes.append(note)
            if visual_notes:
                parts.append("Visual content:\n" + "\n".join(visual_notes))

        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent tool definitions
# ---------------------------------------------------------------------------

AGENT_TOOLS = [
    {
        "name": "clean_segments",
        "description": (
            "Clean transcript grammar for a batch of segment indices. "
            "Removes filler words, fixes grammar, formats code references. "
            "Does NOT rewrite or paraphrase. Internally parallelized."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Segment indices (0-based) to clean.",
                }
            },
            "required": ["indices"],
        },
    },
    {
        "name": "analyze_visuals",
        "description": (
            "Analyze video frames for a batch of segment indices. "
            "Classifies frames as code/math/diagram/text/talking_head and extracts content. "
            "SKIP segments where all frames are likely talking-head (no visual content changes). "
            "Internally parallelized."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Segment indices to analyze. Skip pure talking-head segments.",
                }
            },
            "required": ["indices"],
        },
    },
    {
        "name": "assemble_document",
        "description": (
            "Assemble cleaned segments + visual analysis into structured markdown. "
            "Determines heading placement, integrates code blocks and images. "
            "Call AFTER clean_segments and analyze_visuals are complete. "
            "Provide structure_hints to guide heading placement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Segment indices to include (usually all, in order).",
                },
                "structure_hints": {
                    "type": "string",
                    "description": (
                        "Hints for heading placement. E.g.: "
                        "'Major topic change at segment 5 (theory to code). "
                        "Segments 0-2 are introduction.'"
                    ),
                },
            },
            "required": ["indices"],
        },
    },
    {
        "name": "review_document",
        "description": (
            "Read the assembled result.md for review. "
            "Without arguments: returns document metadata (total length, headings list, number of chunks). "
            "With chunk=N: returns the Nth chunk (~20k chars, split at heading boundaries). "
            "Call this after assemble_document. Review ALL chunks for thorough quality control."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk": {
                    "type": "integer",
                    "description": (
                        "Chunk number to retrieve (1-indexed). "
                        "Omit to get document overview with total chunks and headings list."
                    ),
                }
            },
        },
    },
    {
        "name": "revise_segments",
        "description": (
            "Re-process specific segments with revision instructions. "
            "Use this after reviewing the document to fix specific issues. "
            "The revised segments will replace the cached versions, "
            "then call assemble_document again to rebuild."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Segment indices that need revision.",
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Specific revision instructions. E.g.: "
                        "'Segment 5: the code block is incomplete, re-extract. "
                        "Segment 8: awkward transition, clean up the opening sentence.'"
                    ),
                },
            },
            "required": ["indices", "instructions"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are an expert document-assembly orchestrator converting a video lecture into a high-quality markdown document.

<context>
You have {num_segments} segments from a video titled "{title}".

Here is a summary of each segment:

{segment_summaries}
</context>

<tools>
1. clean_segments(indices) — Grammar cleanup (fast Haiku model). Run on ALL segments.
2. analyze_visuals(indices) — Frame analysis (Sonnet vision). Skip segments with only talking-head frames — check the frame summary to decide.
3. assemble_document(indices, structure_hints) — Structure + assemble markdown (Sonnet). Provide detailed structure_hints based on your content understanding.
4. review_document() — Get document overview (total length, headings list, number of chunks). review_document(chunk=N) — Read chunk N for detailed review.
5. revise_segments(indices, instructions) — Re-process specific segments with your feedback.
</tools>

<workflow>
Phase A — Processing:
1. Call clean_segments with ALL segment indices. Also call analyze_visuals for segments with interesting frames. Where possible, call both tools in the same response so they run in parallel.
2. Examine the tool results. Formulate structure_hints: identify where major topic transitions occur, where sections begin/end, based on the segment previews.
3. Call assemble_document with all indices and your structure_hints.

Phase B — Review:
4. Call review_document() (no arguments) to get the document overview: total length, number of chunks, and heading list.
5. Call review_document(chunk=N) for each chunk (1 through num_chunks). Call multiple chunks in the same response for parallel retrieval.
6. Evaluate the entire document against these criteria:
   - Are headings well-placed, descriptive, and properly capitalized in title case (using the speaker's own words)?
   - Is code/math properly extracted and formatted?
   - When the speaker explains code step-by-step, is it broken into separate code blocks with explanation prose before each? (Not one giant code block, not comments inside code.)
   - Is the speaker's voice preserved (no paraphrasing)?
   - Are there awkward transitions between segments?
   - Is any content missing or duplicated?
   - Are images limited to truly graphical content only (graphs, plots, diagrams, flowcharts)? Images of code, notebooks, terminal output, browser pages, or text-heavy slides should NOT appear — their content belongs in text/code blocks.
   - When the speaker references a visual ("look at this graph", "as shown here"), is the referenced graphic included?
7. If the document is good, respond with "APPROVED".
8. If revisions are needed, call revise_segments with specific instructions, then call assemble_document again, then review again. Repeat until no new revisions are needed.
</workflow>

When done, respond with ONLY the text "APPROVED" — the final document is already saved.
Your value is in planning (which segments need visuals, structure hints) and reviewing (catching quality issues)."""


# ---------------------------------------------------------------------------
# Segment summary builder
# ---------------------------------------------------------------------------


def _build_segment_summaries(segments: list[VideoSegment]) -> str:
    """Build compact summary of each segment for the agent's context."""
    lines = []
    for seg in segments:
        m_start = int(seg.start // 60)
        s_start = int(seg.start % 60)
        m_end = int(seg.end // 60)
        s_end = int(seg.end % 60)

        preview = seg.text[:150].replace("\n", " ")
        if len(seg.text) > 150:
            preview += "..."

        frame_count = len(seg.frames)
        frame_sources = set(f.source for f in seg.frames) if seg.frames else set()
        frame_info = f"{frame_count} frames"
        if frame_sources:
            frame_info += f" ({', '.join(sorted(frame_sources))})"

        lines.append(
            f"Segment {seg.index} [{m_start:02d}:{s_start:02d} - {m_end:02d}:{s_end:02d}]: "
            f"{frame_info}\n"
            f'  Preview: "{preview}"'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool execution functions (with disk caching + internal parallelism)
# ---------------------------------------------------------------------------


def _execute_clean_segments(
    indices: list[int],
    segments: list[VideoSegment],
    job_dir: Path,
    use_bedrock: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Execute clean_segments tool — parallelized with disk caching."""
    cache_dir = job_dir / "cache" / "cleaned"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    to_process = []

    for idx in indices:
        cache_file = cache_dir / f"segment_{idx}.txt"
        if cache_file.exists():
            results[idx] = cache_file.read_text()
        else:
            to_process.append(idx)

    if not to_process:
        return {"status": "success", "cleaned_count": len(indices), "cached": len(indices)}

    completed = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(clean_segment, segments[idx].text, use_bedrock=use_bedrock): idx
            for idx in to_process
        }
        for future in as_completed(futures):
            idx = futures[future]
            cleaned = future.result()
            cache_file = cache_dir / f"segment_{idx}.txt"
            cache_file.write_text(cleaned)
            results[idx] = cleaned
            completed += 1
            if on_progress:
                on_progress(
                    "Cleaning segments",
                    f"{completed}/{len(to_process)}",
                    int(completed / len(to_process) * 30),
                    latest_text=cleaned,
                )

    return {
        "status": "success",
        "cleaned_count": len(indices),
        "cached": len(indices) - len(to_process),
        "processed": len(to_process),
    }


def _execute_analyze_visuals(
    indices: list[int],
    segments: list[VideoSegment],
    job_dir: Path,
    use_bedrock: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Execute analyze_visuals tool — parallelized with disk caching."""
    cache_dir = job_dir / "cache" / "visuals"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    to_process = []

    for idx in indices:
        cache_file = cache_dir / f"segment_{idx}.json"
        if cache_file.exists():
            results[idx] = json.loads(cache_file.read_text())
        else:
            to_process.append(idx)

    if not to_process:
        return {"status": "success", "analyzed_count": len(indices), "cached": len(indices)}

    completed = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(analyze_segment_visuals, segments[idx], use_bedrock=use_bedrock): idx
            for idx in to_process
        }
        for future in as_completed(futures):
            idx = futures[future]
            visuals = future.result()
            visual_data = [
                {
                    "frame_path": str(v.frame_path),
                    "timestamp": v.timestamp,
                    "category": v.category,
                    "extracted_text": v.extracted_text,
                    "description": v.description,
                    "include_as_image": v.include_as_image,
                }
                for v in visuals
            ]
            cache_file = cache_dir / f"segment_{idx}.json"
            cache_file.write_text(json.dumps(visual_data, indent=2))
            results[idx] = visual_data
            completed += 1
            if on_progress:
                on_progress(
                    "Analyzing visuals",
                    f"{completed}/{len(to_process)}",
                    30 + int(completed / len(to_process) * 30),
                )

    # Return summary for the agent
    summary = {}
    for idx in indices:
        data = results[idx]
        categories = [v["category"] for v in data]
        summary[str(idx)] = {
            "frame_count": len(data),
            "categories": list(set(categories)),
            "has_code": any(c == "code" for c in categories),
            "has_math": any(c == "math" for c in categories),
            "has_diagram": any(c == "diagram" for c in categories),
            "has_visual_reference": any(c == "visual_reference" for c in categories),
            "all_talking_head": all(c == "talking_head" for c in categories),
        }

    return {
        "status": "success",
        "analyzed_count": len(indices),
        "cached": len(indices) - len(to_process),
        "processed": len(to_process),
        "segment_summaries": summary,
    }


def _execute_assemble_document(
    indices: list[int],
    structure_hints: str,
    segments: list[VideoSegment],
    job_dir: Path,
    title: str,
    source_video: str,
    use_bedrock: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Execute assemble_document tool — loads cached results and assembles."""
    cache_cleaned_dir = job_dir / "cache" / "cleaned"
    cache_visuals_dir = job_dir / "cache" / "visuals"

    processed_segments = []
    for idx in indices:
        # Load cleaned text
        cleaned_file = cache_cleaned_dir / f"segment_{idx}.txt"
        if cleaned_file.exists():
            cleaned_text = cleaned_file.read_text()
        else:
            cleaned_text = segments[idx].text  # fallback to raw

        # Load visual analysis
        visuals = []
        visual_file = cache_visuals_dir / f"segment_{idx}.json"
        if visual_file.exists():
            visual_data = json.loads(visual_file.read_text())
            visuals = [
                VisualContent(
                    frame_path=Path(v["frame_path"]),
                    timestamp=v["timestamp"],
                    category=v["category"],
                    extracted_text=v.get("extracted_text"),
                    description=v.get("description"),
                    include_as_image=v.get("include_as_image", False),
                )
                for v in visual_data
            ]

        processed_segments.append(
            ProcessedSegment(
                index=idx,
                cleaned_text=cleaned_text,
                visuals=visuals,
            )
        )

    if on_progress:
        on_progress("Assembling document", "Structuring...", 70)

    # Assembly with structure hints
    enhanced_system = STRUCTURE_PROMPT
    if structure_hints:
        enhanced_system += f"\n\nSTRUCTURE HINTS from the orchestrator:\n{structure_hints}"

    client = get_client(use_bedrock=use_bedrock)
    model_id = get_model_id("claude-sonnet-4-6", _is_bedrock(client))

    batch_size = 15
    overlap = 1

    if len(processed_segments) <= batch_size:
        prompt = _build_assembly_prompt(processed_segments)
        message = _call_anthropic(
            client,
            model=model_id,
            max_tokens=8192,
            system=enhanced_system,
            messages=[{"role": "user", "content": prompt}],
        )
        markdown = message.content[0].text
    else:
        parts = []
        i = 0
        while i < len(processed_segments):
            end = min(i + batch_size, len(processed_segments))
            batch = processed_segments[i:end]
            context = ""
            if parts:
                last_lines = parts[-1].strip().split("\n")[-10:]
                context = (
                    "CONTEXT: The document so far ends with:\n"
                    + "\n".join(last_lines)
                    + "\n\nContinue from here. Do not repeat the above.\n\n"
                )
            prompt = context + _build_assembly_prompt(batch)
            message = _call_anthropic(
                client,
                model=model_id,
                max_tokens=8192,
                system=enhanced_system,
                messages=[{"role": "user", "content": prompt}],
            )
            parts.append(message.content[0].text)
            i = end - overlap if end < len(processed_segments) else end
        markdown = "\n\n".join(parts)

    if on_progress:
        on_progress("Assembling document", "Finalizing...", 85)

    # Post-process: copy frames, create final document
    frames_dir = job_dir / "frames"
    markdown = copy_referenced_frames(markdown, frames_dir, job_dir)
    final_doc = create_markdown_document(
        title=title,
        content=markdown,
        source_video=source_video,
    )

    result_path = job_dir / "result.md"
    save_markdown(final_doc, result_path)

    return {
        "status": "success",
        "markdown_length": len(final_doc),
        "output_path": str(result_path),
    }


REVIEW_CHUNK_SIZE = 20000  # ~20k chars per chunk for review


def _split_document_into_chunks(content: str, chunk_size: int = REVIEW_CHUNK_SIZE) -> list[str]:
    """Split document at heading boundaries into chunks of ~chunk_size chars.

    Finds all ## and ### heading positions and groups sections so each chunk
    stays under chunk_size. If the document is shorter than chunk_size,
    returns a single chunk.
    """
    if len(content) <= chunk_size:
        return [content]

    # Find heading positions (## or ###)
    heading_pattern = re.compile(r"^#{2,3}\s+", re.MULTILINE)
    heading_positions = [m.start() for m in heading_pattern.finditer(content)]

    if not heading_positions:
        # No headings — split at chunk_size boundaries
        return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

    # Group sections into chunks that stay under chunk_size
    chunks: list[str] = []
    current_start = 0

    for pos in heading_positions:
        # If adding this section would exceed chunk_size, finalize current chunk
        if pos - current_start >= chunk_size and pos > current_start:
            chunks.append(content[current_start:pos])
            current_start = pos

    # Append remaining content
    if current_start < len(content):
        chunks.append(content[current_start:])

    return chunks


def _execute_review_document(job_dir: Path, chunk: int | None = None) -> dict:
    """Execute review_document tool — supports chunked reading for full review.

    Without chunk parameter: returns document metadata (length, headings, chunk count).
    With chunk=N: returns the Nth chunk (1-indexed) of ~20k chars split at headings.
    """
    result_path = job_dir / "result.md"
    if not result_path.exists():
        return {"status": "error", "error": "No result.md found. Call assemble_document first."}

    content = result_path.read_text()
    chunks = _split_document_into_chunks(content)

    if chunk is None:
        # Return overview metadata
        headings = re.findall(r"^(#{2,3})\s+(.+)$", content, re.MULTILINE)
        heading_list = [f"{h[0]} {h[1]}" for h in headings]
        return {
            "status": "success",
            "total_length": len(content),
            "num_chunks": len(chunks),
            "chunk_lengths": [len(c) for c in chunks],
            "headings": heading_list,
        }

    # Return specific chunk (1-indexed)
    if chunk < 1 or chunk > len(chunks):
        return {"status": "error", "error": f"Chunk {chunk} out of range (1-{len(chunks)})"}

    return {
        "status": "success",
        "chunk": chunk,
        "total_chunks": len(chunks),
        "content": chunks[chunk - 1],
        "length": len(chunks[chunk - 1]),
    }


def _execute_revise_segments(
    indices: list[int],
    instructions: str,
    segments: list[VideoSegment],
    job_dir: Path,
    use_bedrock: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Execute revise_segments tool — re-process with reviewer feedback."""
    cache_cleaned_dir = job_dir / "cache" / "cleaned"
    cache_cleaned_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(use_bedrock=use_bedrock)
    model_id = get_model_id("claude-sonnet-4-6", _is_bedrock(client))

    revised_count = 0

    def revise_one(idx: int) -> tuple[int, str]:
        current_file = cache_cleaned_dir / f"segment_{idx}.txt"
        current_text = current_file.read_text() if current_file.exists() else segments[idx].text

        message = _call_anthropic(
            client,
            model=model_id,
            max_tokens=4096,
            system=REVISION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<current_text>\n{current_text}\n</current_text>\n\n"
                        f"<revision_instructions>\n{instructions}\n</revision_instructions>"
                    ),
                }
            ],
        )
        return idx, message.content[0].text

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(revise_one, idx) for idx in indices]
        for future in as_completed(futures):
            idx, revised_text = future.result()
            cache_file = cache_cleaned_dir / f"segment_{idx}.txt"
            cache_file.write_text(revised_text)
            revised_count += 1
            if on_progress:
                on_progress(
                    "Revising segments",
                    f"{revised_count}/{len(indices)}",
                    90,
                    latest_text=revised_text,
                )

    return {
        "status": "success",
        "revised_count": revised_count,
        "indices": indices,
    }


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _dispatch_tool(
    tool_name: str,
    tool_input: dict,
    segments: list[VideoSegment],
    job_dir: Path,
    title: str,
    source_video: str,
    use_bedrock: Optional[bool],
    on_progress: Optional[Callable],
) -> dict:
    """Route a tool call to the appropriate execution function."""
    if tool_name == "clean_segments":
        return _execute_clean_segments(
            indices=tool_input["indices"],
            segments=segments,
            job_dir=job_dir,
            use_bedrock=use_bedrock,
            on_progress=on_progress,
        )
    elif tool_name == "analyze_visuals":
        return _execute_analyze_visuals(
            indices=tool_input["indices"],
            segments=segments,
            job_dir=job_dir,
            use_bedrock=use_bedrock,
            on_progress=on_progress,
        )
    elif tool_name == "assemble_document":
        return _execute_assemble_document(
            indices=tool_input["indices"],
            structure_hints=tool_input.get("structure_hints", ""),
            segments=segments,
            job_dir=job_dir,
            title=title,
            source_video=source_video,
            use_bedrock=use_bedrock,
            on_progress=on_progress,
        )
    elif tool_name == "review_document":
        return _execute_review_document(
            job_dir=job_dir,
            chunk=tool_input.get("chunk"),
        )
    elif tool_name == "revise_segments":
        return _execute_revise_segments(
            indices=tool_input["indices"],
            instructions=tool_input["instructions"],
            segments=segments,
            job_dir=job_dir,
            use_bedrock=use_bedrock,
            on_progress=on_progress,
        )
    else:
        return {"status": "error", "error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Agent loop (Opus 4.6 orchestrator + reviewer)
# ---------------------------------------------------------------------------


def run_agent_loop(
    video_segments: list[VideoSegment],
    job_dir: Path,
    title: str,
    source_video: str,
    orchestrator_model: str = "claude-opus-4-6",
    max_iterations: int = 15,
    use_bedrock: Optional[bool] = None,
    on_progress: Optional[Callable] = None,
) -> str:
    """Run the Opus 4.6 agent loop that orchestrates processing and reviews output.

    The agent plans tool calls, executes them, reviews the assembled document,
    and requests revisions if needed.

    Args:
        video_segments: VideoSegments from the segmenter.
        job_dir: Job working directory for caching.
        title: Document title.
        source_video: Source video filename.
        orchestrator_model: Model for the orchestrator (Opus 4.6).
        max_iterations: Safety limit on agent loop iterations.
        use_bedrock: API routing.
        on_progress: Callback(stage, detail, pct) for progress updates.

    Returns:
        Path to the final result.md as string.
    """
    client = get_client(use_bedrock=use_bedrock)
    model_id = get_model_id(orchestrator_model, _is_bedrock(client))

    summaries = _build_segment_summaries(video_segments)
    system = AGENT_SYSTEM_PROMPT.format(
        num_segments=len(video_segments),
        title=title,
        segment_summaries=summaries,
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"Process all {len(video_segments)} segments into a markdown document. "
                f'The video is titled "{title}". Plan your approach, then use the tools.'
            ),
        }
    ]

    for iteration in range(max_iterations):
        logger.info(f"Agent loop iteration {iteration + 1}/{max_iterations}")

        response = _call_anthropic(
            client,
            model=model_id,
            max_tokens=16384,
            system=system,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        # Check if agent is done (returned text, no more tool calls)
        if response.stop_reason == "end_turn":
            logger.info("Agent returned final response")
            result_path = job_dir / "result.md"
            if result_path.exists():
                return result_path.read_text()
            # Fallback: extract text from response
            text_blocks = [b.text for b in response.content if b.type == "text"]
            if text_blocks:
                return "\n".join(text_blocks)
            break

        if response.stop_reason != "tool_use":
            logger.warning(f"Unexpected stop_reason: {response.stop_reason}")
            break

        # Extract tool calls
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            break

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": response.content})

        # Log what the agent is doing
        tool_names = [tc.name for tc in tool_calls]
        logger.info(f"Agent calling tools: {tool_names}")

        # Execute tool calls (parallel if multiple)
        tool_results = []
        if len(tool_calls) == 1:
            tc = tool_calls[0]
            try:
                result = _dispatch_tool(
                    tc.name,
                    tc.input,
                    video_segments,
                    job_dir,
                    title,
                    source_video,
                    use_bedrock,
                    on_progress,
                )
            except Exception as e:
                logger.error(f"Tool {tc.name} failed: {e}", exc_info=True)
                result = {"status": "error", "error": str(e)}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result),
                }
            )
        else:
            with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
                futures = {
                    executor.submit(
                        _dispatch_tool,
                        tc.name,
                        tc.input,
                        video_segments,
                        job_dir,
                        title,
                        source_video,
                        use_bedrock,
                        on_progress,
                    ): tc
                    for tc in tool_calls
                }
                for future in as_completed(futures):
                    tc = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        logger.error(f"Tool {tc.name} failed: {e}", exc_info=True)
                        result = {"status": "error", "error": str(e)}
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )

        messages.append({"role": "user", "content": tool_results})

    # Safety: check if result exists even if loop exhausted
    result_path = job_dir / "result.md"
    if result_path.exists():
        return result_path.read_text()

    raise RuntimeError("Agent loop exhausted without producing a result")
