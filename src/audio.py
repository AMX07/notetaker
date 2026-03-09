"""Extract audio from video files and probe video metadata using ffmpeg."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    """Metadata about a video file."""

    duration_seconds: float
    width: int
    height: int
    fps: float
    audio_codec: str | None


def probe_video(video_path: Path) -> VideoInfo:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    if not video_stream:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse fps from r_frame_rate (e.g. "30/1" or "30000/1001")
    fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

    return VideoInfo(
        duration_seconds=float(data["format"]["duration"]),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        audio_codec=audio_stream["codec_name"] if audio_stream else None,
    )


def extract_audio(
    video_path: Path,
    output_path: Path | None = None,
    audio_format: str = "mp3",
    bitrate: str = "64k",
) -> Path:
    """Convert video to MP3 (or other format) using ffmpeg.

    Default bitrate of 64k keeps file sizes small for API upload.
    A 1-hour video at 64kbps mono → ~30MB MP3.

    Args:
        video_path: Path to MP4 file.
        output_path: Where to save audio. Defaults to same dir with new extension.
        audio_format: Output format (mp3 recommended for compact size).
        bitrate: Audio bitrate (64k keeps files under 25MB API limit for most lectures).

    Returns:
        Path to the extracted audio file.
    """
    if output_path is None:
        output_path = video_path.with_suffix(f".{audio_format}")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-ab",
        bitrate,  # bitrate
        "-y",  # overwrite
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    return output_path
