"""Speech-to-text transcription via OpenAI Whisper API."""

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("notetaker")

_whisper_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    reraise=True,
)


@dataclass
class TranscriptWord:
    """A single transcribed word with timing."""

    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    """A sentence-level transcript segment."""

    text: str
    start: float
    end: float
    words: list[TranscriptWord]


def split_audio_for_api(audio_path: Path, max_size_mb: float = 25.0) -> list[Path]:
    """Split an audio file into chunks that fit the OpenAI 25MB limit.

    Uses ffmpeg to split at roughly equal durations. Each chunk gets a
    sequential suffix like _chunk001.mp3.

    Returns list of chunk paths (or just [audio_path] if already small enough).
    """
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return [audio_path]

    # Get duration via ffprobe
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    total_duration = float(result.stdout.strip())

    # Calculate chunk duration to stay under limit
    num_chunks = int(file_size_mb / max_size_mb) + 1
    chunk_duration = total_duration / num_chunks

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = audio_path.with_stem(f"{audio_path.stem}_chunk{i:03d}")
        cmd = [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(chunk_duration),
            "-y",
            str(chunk_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        chunks.append(chunk_path)

    return chunks


def _get_chunk_duration(chunk_path: Path) -> float:
    """Get duration of an audio file via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(chunk_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _transcribe_chunk(
    chunk_path: Path,
    time_offset: float,
    language: str | None,
) -> list[TranscriptSegment]:
    """Transcribe a single audio chunk and apply time offset."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    with open(chunk_path, "rb") as f:
        kwargs: dict = {
            "model": "whisper-1",
            "file": f,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
        }
        if language:
            kwargs["language"] = language

        response = _whisper_retry(client.audio.transcriptions.create)(**kwargs)

    segments = []
    for seg in response.segments:
        words = []
        if response.words:
            for w in response.words:
                if seg.start <= w.start < seg.end:
                    words.append(
                        TranscriptWord(
                            text=w.word.strip(),
                            start=w.start + time_offset,
                            end=w.end + time_offset,
                        )
                    )
        segments.append(
            TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start + time_offset,
                end=seg.end + time_offset,
                words=words,
            )
        )
    return segments


def transcribe_audio(
    audio_path: Path,
    language: str | None = None,
    max_workers: int = 4,
) -> list[TranscriptSegment]:
    """Transcribe an audio file using the OpenAI Whisper API.

    Handles files >25MB by splitting into chunks and sending them
    to the API in parallel.

    Args:
        audio_path: Path to audio file (MP3 recommended).
        language: Language code (e.g. "en"). None for auto-detect.
        max_workers: Max parallel Whisper API calls.

    Returns:
        List of TranscriptSegment with word-level timestamps.
    """
    chunks = split_audio_for_api(audio_path)

    if len(chunks) == 1:
        return _transcribe_chunk(chunks[0], 0.0, language)

    # Pre-compute time offsets from chunk durations
    chunk_durations = [_get_chunk_duration(p) for p in chunks]
    time_offsets = [0.0]
    for dur in chunk_durations[:-1]:
        time_offsets.append(time_offsets[-1] + dur)

    # Parallel transcription
    indexed_results: list[tuple[int, list[TranscriptSegment]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_transcribe_chunk, chunk_path, offset, language): i
            for i, (chunk_path, offset) in enumerate(zip(chunks, time_offsets))
        }
        for future in as_completed(futures):
            idx = futures[future]
            indexed_results.append((idx, future.result()))

    # Clean up chunk files
    for chunk_path in chunks:
        if chunk_path != audio_path:
            chunk_path.unlink(missing_ok=True)

    # Sort by chunk index and flatten
    indexed_results.sort(key=lambda x: x[0])
    return [seg for _, segs in indexed_results for seg in segs]


def segments_to_json(segments: list[TranscriptSegment], path: Path) -> None:
    """Serialize transcript segments to a JSON file."""
    data = [asdict(seg) for seg in segments]
    path.write_text(json.dumps(data, indent=2))


def segments_from_json(path: Path) -> list[TranscriptSegment]:
    """Deserialize transcript segments from a JSON file."""
    data = json.loads(path.read_text())
    return [
        TranscriptSegment(
            text=seg["text"],
            start=seg["start"],
            end=seg["end"],
            words=[TranscriptWord(**w) for w in seg["words"]],
        )
        for seg in data
    ]


def segments_to_text(
    segments: list[TranscriptSegment],
    include_timestamps: bool = False,
) -> str:
    """Convert transcript segments to plain text."""
    lines = []
    for seg in segments:
        if include_timestamps:
            minutes, seconds = divmod(int(seg.start), 60)
            hours, minutes = divmod(minutes, 60)
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
            lines.append(f"{timestamp} {seg.text}")
        else:
            lines.append(seg.text)
    return "\n".join(lines)
