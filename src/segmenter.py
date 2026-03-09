"""Segment transcript into time windows and align with extracted frames."""

from dataclasses import dataclass

from .transcribe import TranscriptSegment
from .frames import ExtractedFrame


@dataclass
class VideoSegment:
    """A segment of the video with aligned transcript and frames."""

    index: int
    start: float  # seconds
    end: float  # seconds
    transcript_segments: list[TranscriptSegment]
    frames: list[ExtractedFrame]

    @property
    def text(self) -> str:
        """Full transcript text for this segment."""
        return " ".join(seg.text for seg in self.transcript_segments)

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_by_time_windows(
    transcript_segments: list[TranscriptSegment],
    target_duration: float = 180.0,
    min_duration: float = 60.0,
    max_duration: float = 300.0,
) -> list[tuple[float, float]]:
    """Divide transcript into time windows using natural pause boundaries.

    Walks through segments accumulating duration. When the target is
    exceeded, looks backward for the largest silence gap to split at.

    Args:
        transcript_segments: Whisper transcript segments.
        target_duration: Target segment duration in seconds (default 3 min).
        min_duration: Minimum segment duration.
        max_duration: Maximum segment duration.

    Returns:
        List of (start_time, end_time) tuples.
    """
    if not transcript_segments:
        return []

    # Compute gaps between consecutive segments
    gaps: list[tuple[int, float]] = []  # (index_after_which_to_split, gap_size)
    for i in range(len(transcript_segments) - 1):
        gap = transcript_segments[i + 1].start - transcript_segments[i].end
        gaps.append((i, max(0.0, gap)))

    windows: list[tuple[float, float]] = []
    window_start_idx = 0
    window_start_time = transcript_segments[0].start

    i = 0
    while i < len(transcript_segments):
        current_duration = transcript_segments[i].end - window_start_time

        if current_duration >= target_duration:
            # Find the largest gap in the last 30 seconds of this window
            best_gap_idx = None
            best_gap_size = -1.0

            for gap_idx, gap_size in gaps:
                if gap_idx < window_start_idx:
                    continue
                if gap_idx > i:
                    break
                gap_time = transcript_segments[gap_idx].end
                # Prefer gaps near the target duration, weighted by gap size
                if (gap_time - window_start_time) >= min_duration and gap_size > best_gap_size:
                    best_gap_size = gap_size
                    best_gap_idx = gap_idx

            if best_gap_idx is not None:
                split_at = best_gap_idx
            else:
                # No good gap found, split at current position
                split_at = i

            window_end_time = transcript_segments[split_at].end
            windows.append((window_start_time, window_end_time))

            window_start_idx = split_at + 1
            if window_start_idx < len(transcript_segments):
                window_start_time = transcript_segments[window_start_idx].start
                i = window_start_idx
            else:
                break
            continue

        # Force split if we exceed max_duration
        if current_duration >= max_duration:
            window_end_time = transcript_segments[i].end
            windows.append((window_start_time, window_end_time))

            window_start_idx = i + 1
            if window_start_idx < len(transcript_segments):
                window_start_time = transcript_segments[window_start_idx].start
            i = window_start_idx
            continue

        i += 1

    # Add final window
    if window_start_idx < len(transcript_segments):
        windows.append(
            (
                window_start_time,
                transcript_segments[-1].end,
            )
        )

    return windows


def align_frames_to_segments(
    windows: list[tuple[float, float]],
    frames: list[ExtractedFrame],
    transcript_segments: list[TranscriptSegment],
) -> list[VideoSegment]:
    """Align extracted frames and transcript segments to time windows.

    Each frame and transcript segment is assigned to the VideoSegment
    whose time window contains it.

    Args:
        windows: Time window boundaries from segment_by_time_windows.
        frames: All extracted frames.
        transcript_segments: All transcript segments.

    Returns:
        List of VideoSegment with aligned transcript and frames.
    """
    video_segments = []

    for idx, (start, end) in enumerate(windows):
        # Collect transcript segments in this window
        segs = [s for s in transcript_segments if s.start >= start and s.start < end]

        # Collect frames in this window
        window_frames = [f for f in frames if f.timestamp >= start and f.timestamp < end]

        video_segments.append(
            VideoSegment(
                index=idx,
                start=start,
                end=end,
                transcript_segments=segs,
                frames=window_frames,
            )
        )

    return video_segments
