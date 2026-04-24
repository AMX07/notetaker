"""Extract frames from video files using ffmpeg."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedFrame:
    """A frame extracted from a video."""

    path: Path
    timestamp: float  # seconds into the video
    source: str  # "interval" or "scene_change"


def frames_to_json(frames: list[ExtractedFrame], path: Path) -> None:
    """Serialize extracted frames list to a JSON file."""
    data = [{"path": str(f.path), "timestamp": f.timestamp, "source": f.source} for f in frames]
    path.write_text(json.dumps(data, indent=2))


def frames_from_json(path: Path) -> list[ExtractedFrame]:
    """Deserialize extracted frames list from a JSON file."""
    data = json.loads(path.read_text())
    return [
        ExtractedFrame(path=Path(f["path"]), timestamp=f["timestamp"], source=f["source"])
        for f in data
    ]


def extract_frames_interval(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 10.0,
    max_width: int = 1024,
    quality: int = 5,
) -> list[ExtractedFrame]:
    """Extract frames at regular intervals using ffmpeg.

    Args:
        video_path: Path to video file.
        output_dir: Directory to save frames.
        interval_seconds: Seconds between frames.
        max_width: Max frame width (downscale for API cost savings).
        quality: JPEG quality 1-31 (lower is better).

    Returns:
        List of ExtractedFrame sorted by timestamp.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "interval_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval_seconds},scale='min({max_width},iw)':-1",
        "-qscale:v",
        str(quality),
        "-y",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=1800)

    frames = []
    for f in sorted(output_dir.glob("interval_*.jpg")):
        # Frame number is 1-indexed from ffmpeg
        match = re.search(r"interval_(\d+)\.jpg", f.name)
        if match:
            frame_num = int(match.group(1))
            timestamp = (frame_num - 1) * interval_seconds
            frames.append(ExtractedFrame(path=f, timestamp=timestamp, source="interval"))

    return frames


def extract_frames_scene_change(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.3,
    max_width: int = 1024,
    quality: int = 5,
) -> list[ExtractedFrame]:
    """Extract frames at scene changes using ffmpeg.

    Args:
        video_path: Path to video file.
        output_dir: Directory to save frames.
        threshold: Scene change threshold 0.0-1.0 (lower = more sensitive).
        max_width: Max frame width.
        quality: JPEG quality.

    Returns:
        List of ExtractedFrame sorted by timestamp.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use showinfo filter to get timestamps, and select filter for scene detection
    # We need to get the timestamps of scene changes, so use -showinfo and parse
    pattern = str(output_dir / "scene_%06d.jpg")

    # Run with showinfo to capture timestamps from stderr
    cmd_with_info = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        (f"select='gt(scene\\,{threshold})',scale='min({max_width},iw)':-1,showinfo"),
        "-vsync",
        "vfr",
        "-qscale:v",
        str(quality),
        "-y",
        pattern,
    ]
    result = subprocess.run(cmd_with_info, capture_output=True, text=True, timeout=1800)

    # Parse timestamps from showinfo output in stderr
    frames = []
    scene_files = sorted(output_dir.glob("scene_*.jpg"))
    timestamps = []

    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            match = re.search(r"pts_time:\s*([\d.]+)", line)
            if match:
                timestamps.append(float(match.group(1)))

    for i, f in enumerate(scene_files):
        ts = timestamps[i] if i < len(timestamps) else 0.0
        frames.append(ExtractedFrame(path=f, timestamp=ts, source="scene_change"))

    return frames


def extract_frames_hybrid(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 10.0,
    scene_threshold: float = 0.3,
    dedup_window: float = 2.0,
    max_width: int = 1024,
    quality: int = 5,
) -> list[ExtractedFrame]:
    """Hybrid frame extraction: regular interval + scene changes, deduplicated.

    Combines interval-based and scene-change extraction, then removes
    scene-change frames that are within dedup_window seconds of an
    interval frame.

    Args:
        video_path: Path to video file.
        output_dir: Directory to save frames.
        interval_seconds: Seconds between interval frames.
        scene_threshold: Scene change sensitivity (0-1).
        dedup_window: Min seconds between final frames.
        max_width: Max frame width.
        quality: JPEG quality.

    Returns:
        Deduplicated list of ExtractedFrame sorted by timestamp.
    """
    interval_dir = output_dir / "interval"
    scene_dir = output_dir / "scene"

    with ThreadPoolExecutor(max_workers=2) as executor:
        interval_future = executor.submit(
            extract_frames_interval,
            video_path,
            interval_dir,
            interval_seconds,
            max_width,
            quality,
        )
        scene_future = executor.submit(
            extract_frames_scene_change,
            video_path,
            scene_dir,
            scene_threshold,
            max_width,
            quality,
        )
        interval_frames = interval_future.result()
        scene_frames = scene_future.result()

    # Deduplicate: keep all interval frames, add scene frames only if
    # they're not within dedup_window of any interval frame
    interval_timestamps = {f.timestamp for f in interval_frames}
    deduplicated_scene = []
    for sf in scene_frames:
        too_close = any(abs(sf.timestamp - it) < dedup_window for it in interval_timestamps)
        if not too_close:
            deduplicated_scene.append(sf)

    all_frames = interval_frames + deduplicated_scene
    all_frames.sort(key=lambda f: f.timestamp)
    return all_frames
