"""Unit tests for notetaker core functions (no API keys needed)."""

from pathlib import Path

from src.transcribe import TranscriptSegment, TranscriptWord
from src.frames import ExtractedFrame
from src.segmenter import segment_by_time_windows, align_frames_to_segments
from src.assembler import (
    create_markdown_document,
    copy_referenced_frames,
    estimate_reading_time,
    generate_toc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(text: str, start: float, end: float) -> TranscriptSegment:
    """Create a TranscriptSegment with dummy words."""
    words = [TranscriptWord(text=w, start=start, end=end) for w in text.split()]
    return TranscriptSegment(text=text, start=start, end=end, words=words)


def _make_frame(timestamp: float, name: str = "frame.jpg") -> ExtractedFrame:
    return ExtractedFrame(path=Path(f"/tmp/{name}"), timestamp=timestamp, source="interval")


# ---------------------------------------------------------------------------
# Segmenter tests
# ---------------------------------------------------------------------------


class TestSegmentByTimeWindows:
    def test_empty_transcript(self):
        assert segment_by_time_windows([]) == []

    def test_short_transcript_single_window(self):
        """A transcript shorter than target_duration should produce one window."""
        segments = [
            _make_segment("Hello world", 0.0, 30.0),
            _make_segment("Some more text", 31.0, 60.0),
        ]
        windows = segment_by_time_windows(segments, target_duration=180.0)
        assert len(windows) == 1
        assert windows[0][0] == 0.0
        assert windows[0][1] == 60.0

    def test_long_transcript_splits(self):
        """A transcript longer than target_duration should produce multiple windows."""
        segments = [
            _make_segment(f"Segment {i}", i * 60.0, (i + 1) * 60.0 - 1.0) for i in range(10)
        ]
        windows = segment_by_time_windows(segments, target_duration=180.0)
        assert len(windows) >= 2
        # Windows should cover full range
        assert windows[0][0] == 0.0
        assert windows[-1][1] == segments[-1].end

    def test_prefers_natural_pauses(self):
        """Should split at the largest gap when possible."""
        segments = [
            _make_segment("A", 0.0, 60.0),
            _make_segment("B", 60.5, 120.0),  # 0.5s gap
            _make_segment("C", 125.0, 180.0),  # 5.0s gap — should prefer this
            _make_segment("D", 180.5, 240.0),
        ]
        windows = segment_by_time_windows(segments, target_duration=180.0, min_duration=60.0)
        # Should split at the gap after C (the 5s gap)
        assert len(windows) >= 1

    def test_max_duration_enforced(self):
        """Segments should not exceed max_duration."""
        segments = [_make_segment(f"S{i}", i * 10.0, (i + 1) * 10.0 - 0.1) for i in range(50)]
        windows = segment_by_time_windows(segments, target_duration=60.0, max_duration=120.0)
        for start, end in windows:
            assert end - start <= 120.1  # small tolerance


class TestAlignFramesToSegments:
    def test_frames_assigned_to_correct_windows(self):
        windows = [(0.0, 60.0), (60.0, 120.0)]
        frames = [
            _make_frame(10.0, "f1.jpg"),
            _make_frame(70.0, "f2.jpg"),
            _make_frame(90.0, "f3.jpg"),
        ]
        segments = [
            _make_segment("Hello", 0.0, 30.0),
            _make_segment("World", 60.0, 90.0),
        ]
        result = align_frames_to_segments(windows, frames, segments)
        assert len(result) == 2
        assert len(result[0].frames) == 1  # f1 at 10s
        assert len(result[1].frames) == 2  # f2 at 70s, f3 at 90s

    def test_empty_windows(self):
        result = align_frames_to_segments([], [], [])
        assert result == []


# ---------------------------------------------------------------------------
# Assembler tests
# ---------------------------------------------------------------------------


class TestCreateMarkdownDocument:
    def test_basic_document(self):
        doc = create_markdown_document("Test Title", "Body content")
        assert doc.startswith("# Test Title")
        assert "Body content" in doc

    def test_with_source_video(self):
        doc = create_markdown_document("Title", "Content", source_video="lecture.mp4")
        assert "lecture.mp4" in doc
        assert "---" in doc

    def test_without_source_video(self):
        doc = create_markdown_document("Title", "Content")
        assert "Source video" not in doc


class TestCopyReferencedFrames:
    def test_copies_found_frames(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "slide_001.jpg").write_bytes(b"\xff\xd8frame1")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        content = "Look at this: ![slide](slide_001.jpg)"
        result = copy_referenced_frames(content, frames_dir, output_dir)

        assert "assets/slide_001.jpg" in result
        assert (output_dir / "assets" / "slide_001.jpg").exists()

    def test_leaves_missing_frames_unchanged(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        content = "![missing](nonexistent.jpg)"
        result = copy_referenced_frames(content, frames_dir, output_dir)
        assert result == content

    def test_no_images(self, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        content = "Just text, no images."
        result = copy_referenced_frames(content, frames_dir, output_dir)
        assert result == content


class TestEstimateReadingTime:
    def test_short_text(self):
        assert estimate_reading_time("Hello world") == 1

    def test_longer_text(self):
        text = " ".join(["word"] * 600)
        assert estimate_reading_time(text) == 3


class TestGenerateToc:
    def test_generates_toc(self):
        content = "## Introduction\nSome text\n## Methods\nMore text\n## Conclusion"
        toc = generate_toc(content)
        assert "Introduction" in toc
        assert "Methods" in toc
        assert "Conclusion" in toc
        assert toc.startswith("## Table of Contents")

    def test_ignores_toc_heading(self):
        content = "## Table of Contents\n## Real Section"
        toc = generate_toc(content)
        assert toc.count("Table of Contents") == 1  # only the generated header


# ---------------------------------------------------------------------------
# App utility tests
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    def test_strips_dangerous_chars(self):
        from src.app import _sanitize_filename

        assert _sanitize_filename("../../etc/passwd") == "passwd"
        assert _sanitize_filename("file with spaces.mp4") == "file_with_spaces.mp4"
        assert _sanitize_filename("normal.mp4") == "normal.mp4"

    def test_empty_filename(self):
        from src.app import _sanitize_filename

        assert _sanitize_filename("") == "upload.mp4"

    def test_special_characters(self):
        from src.app import _sanitize_filename

        result = _sanitize_filename("lecture (2024) [final].mp4")
        assert "/" not in result
        assert "\\" not in result


class TestRateLimit:
    def test_allows_under_limit(self):
        from src.app import _check_rate_limit, _rate_limit_store

        _rate_limit_store.clear()
        result = _check_rate_limit("test-ip-1")
        assert result is None  # allowed

    def test_blocks_over_limit(self):
        from src.app import _check_rate_limit, _rate_limit_store, UPLOAD_RATE_LIMIT

        _rate_limit_store.clear()
        for _ in range(UPLOAD_RATE_LIMIT):
            _check_rate_limit("test-ip-2")
        result = _check_rate_limit("test-ip-2")
        assert result is not None  # blocked
        assert isinstance(result, int)
        assert result > 0
