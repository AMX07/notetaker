"""Transcript Extraction Tool

This module extracts transcript text from a YouTube transcript HTML export (often saved from
Chrome devtools / copy-paste) where the HTML might contain HTML-escaped markup like:
  &lt;yt-formatted-string class="segment-text"&gt;...&lt;/yt-formatted-string&gt;

Key behavior:
- Extracts only segment text nodes ("segment-text"), ignoring timestamps.
- Removes any timestamp-like prefix that accidentally appears inside text.
- Provides a small CLI that can auto-detect a likely input HTML file in /mnt/data.

Usage:
  python transcript_extract.py /path/to/transcripts.html

If no path is provided, it will try to find a .html file in /mnt/data.
"""

from __future__ import annotations

from pathlib import Path
import html
import re
import sys
from typing import List, Optional


# Matches transcript segment text nodes.
SEGMENT_RE = re.compile(
    r'<yt-formatted-string[^>]*\bclass="[^"]*\bsegment-text\b[^"]*"[^>]*>(.*?)</yt-formatted-string>',
    re.IGNORECASE | re.DOTALL,
)

# Safety: strips common timestamp prefixes if they end up in the text itself.
LEADING_TS_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s+")

# Safety: collapses whitespace.
WS_RE = re.compile(r"\s+")


def eprint(message: str) -> None:
    """Write an error message to stderr.

    Some sandboxed environments stub/override print() and may not support the
    'file=' keyword argument, so we write directly to sys.stderr.
    """
    sys.stderr.write(message)
    if not message.endswith("\n"):
        sys.stderr.write("\n")


def extract_transcript_from_html_string(raw_html: str) -> List[str]:
    """Extract transcript segments from an HTML string.

    The input may contain HTML-escaped markup (e.g., "&lt;yt-formatted-string ...&gt;").

    Returns:
        A list of transcript segment strings (no timestamps).
    """
    # Some exports contain HTML-escaped markup; unescape once to restore tags.
    unescaped = html.unescape(raw_html)

    segments: List[str] = []
    for inner in SEGMENT_RE.findall(unescaped):
        # Remove any nested tags within the segment text.
        text = re.sub(r"<[^>]+>", "", inner)
        text = html.unescape(text).replace("\xa0", " ")
        text = WS_RE.sub(" ", text).strip()

        # Extra safety: strip leading timestamps if present in the text.
        text = LEADING_TS_RE.sub("", text).strip()

        if text:
            segments.append(text)

    return segments


def extract_transcript_from_html_file(html_path: str | Path) -> List[str]:
    """Extract transcript segments from a file path."""
    p = Path(html_path)
    if not p.exists():
        raise FileNotFoundError(_missing_file_message(p))

    raw = p.read_text(encoding="utf-8", errors="ignore")
    return extract_transcript_from_html_string(raw)


def _missing_file_message(p: Path) -> str:
    """Build a helpful FileNotFoundError message with suggestions."""
    hint_dir = Path("/mnt/data")
    candidates: List[str] = []
    if hint_dir.exists():
        # Prefer transcript-ish names first, then any .html
        preferred = sorted(hint_dir.glob("*transcript*.html"))
        if not preferred:
            preferred = sorted(hint_dir.glob("*.html"))
        candidates = [str(x) for x in preferred[:10]]

    msg = [f"No such file: {str(p)!r}."]
    if candidates:
        msg.append("\nAvailable .html candidates in /mnt/data (showing up to 10):")
        msg.extend([f"  - {c}" for c in candidates])
        msg.append(
            "\nTip: pass the correct path explicitly, e.g.\n  python transcript_extract.py '/mnt/data/<file>.html'"
        )
    else:
        msg.append(
            "\nTip: ensure the HTML file is uploaded to /mnt/data, then pass its path explicitly."
        )

    return "\n".join(msg)


def _auto_detect_input_file() -> Optional[Path]:
    """Try to find a likely transcript HTML file in /mnt/data."""
    data_dir = Path("/mnt/data")
    if not data_dir.exists():
        return None

    # Prefer transcript-ish names.
    preferred = sorted(data_dir.glob("*transcript*.html"))
    if preferred:
        return preferred[0]

    # Otherwise, pick the newest .html if any.
    html_files = sorted(
        data_dir.glob("*.html"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    return html_files[0] if html_files else None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv:
        in_path = Path(argv[0])
    else:
        detected = _auto_detect_input_file()
        if detected is None:
            eprint(
                "Could not auto-detect an input .html file in /mnt/data. "
                "Please pass the path explicitly."
            )
            return 2
        in_path = detected

    try:
        transcript = extract_transcript_from_html_file(in_path)
    except FileNotFoundError as e:
        eprint(str(e))
        return 2

    # Print first 10 lines to stdout as a quick check.
    for line in transcript[:10]:
        print(line)

    # Also write the full transcript next to the input file.
    out_path = in_path.with_name("extracted_transcript.txt")
    out_path.write_text("\n".join(transcript), encoding="utf-8")
    print(f"\nWrote {len(transcript)} segments to: {out_path}")

    return 0


# -----------------------------
# Tests
# -----------------------------

import unittest
import tempfile
from contextlib import redirect_stderr
from io import StringIO


class TestTranscriptExtraction(unittest.TestCase):
    def test_extracts_segment_text_and_ignores_timestamps(self) -> None:
        # Simulates an export where the yt-formatted-string tags are HTML-escaped.
        sample = (
            "<div>"
            "&lt;yt-formatted-string class=\"segment-timestamp\"&gt;0:00&lt;/yt-formatted-string&gt;"
            "&lt;yt-formatted-string class=\"segment-text\"&gt; Hello&nbsp;world! &lt;/yt-formatted-string&gt;"
            "&lt;yt-formatted-string class=\"segment-text\"&gt;0:01 Second line&lt;/yt-formatted-string&gt;"
            "</div>"
        )
        out = extract_transcript_from_html_string(sample)
        self.assertEqual(out, ["Hello world!", "Second line"])  # timestamp prefix stripped

    def test_returns_empty_list_when_no_segments(self) -> None:
        self.assertEqual(extract_transcript_from_html_string("<html></html>"), [])

    def test_missing_file_error_is_helpful(self) -> None:
        missing = Path("/mnt/data/DOES_NOT_EXIST.html")
        with self.assertRaises(FileNotFoundError) as ctx:
            extract_transcript_from_html_file(missing)
        self.assertIn("No such file", str(ctx.exception))

    def test_file_roundtrip(self) -> None:
        sample = (
            "&lt;yt-formatted-string class=\"segment-text\"&gt;Line one&lt;/yt-formatted-string&gt;"
            "&lt;yt-formatted-string class=\"segment-text\"&gt;Line two&lt;/yt-formatted-string&gt;"
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "transcripts.html"
            p.write_text(sample, encoding="utf-8")
            out = extract_transcript_from_html_file(p)
            self.assertEqual(out, ["Line one", "Line two"])

    def test_eprint_writes_to_stderr_without_print_file_kw(self) -> None:
        buf = StringIO()
        with redirect_stderr(buf):
            eprint("hello")
            eprint("world\n")
        self.assertEqual(buf.getvalue(), "hello\nworld\n")


if __name__ == "__main__":
    raise SystemExit(main())
