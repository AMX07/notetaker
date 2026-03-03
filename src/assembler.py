"""Assemble final markdown output and manage frame assets."""

import re
import shutil
from pathlib import Path


def create_markdown_document(
    title: str,
    content: str,
    source_video: str | None = None,
) -> str:
    """Create the final markdown document with header.

    Args:
        title: Document title (h1).
        content: Assembled markdown content from LLM.
        source_video: Source video filename for attribution.

    Returns:
        Complete markdown document as string.
    """
    lines = [f"# {title}", ""]

    if source_video:
        lines.append("---")
        lines.append("")
        lines.append(f"Source video: {source_video}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(content)

    return "\n".join(lines)


def save_markdown(content: str, output_path: str | Path) -> Path:
    """Save markdown content to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def copy_referenced_frames(
    content: str,
    frames_dir: Path,
    output_dir: Path,
) -> str:
    """Copy frames referenced in markdown to output directory and update paths.

    Scans markdown for ![...](...) image references, copies those frames
    to output_dir/assets/, and updates the paths in the markdown.

    Args:
        content: Markdown content with image references.
        frames_dir: Directory containing extracted frames.
        output_dir: Output directory (assets/ will be created inside).

    Returns:
        Updated markdown with corrected image paths.
    """
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def replace_image_path(match: re.Match) -> str:
        alt_text = match.group(1)
        image_name = match.group(2)

        # Try to find the frame in frames_dir (search recursively)
        source = None
        for candidate in frames_dir.rglob(Path(image_name).name):
            source = candidate
            break

        if source and source.exists():
            dest = assets_dir / source.name
            shutil.copy2(source, dest)
            return f"![{alt_text}](assets/{source.name})"

        return match.group(0)  # leave unchanged if not found

    updated = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image_path, content)
    return updated


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes."""
    words = len(text.split())
    return max(1, words // wpm)


def generate_toc(content: str) -> str:
    """Generate table of contents from headings in markdown content."""
    toc_lines = ["## Table of Contents", ""]

    for line in content.split("\n"):
        if line.startswith("## ") and not line.startswith("## Table of Contents"):
            title = line[3:].strip()
            anchor = title.lower().replace(" ", "-")
            anchor = re.sub(r"[^a-z0-9\-]", "", anchor)
            toc_lines.append(f"- [{title}](#{anchor})")

    return "\n".join(toc_lines)
