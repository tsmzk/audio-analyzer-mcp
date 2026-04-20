"""Output and error formatting helpers.

- Convert AudioFrame lists to CSV or JSON for MCP tool responses
- Turn exceptions from the analysis pipeline into actionable error messages
"""

from __future__ import annotations

import csv
import io
import json

from audio_analyzer_mcp.constants import (
    AnalysisError,
    AudioFrame,
    DownloadError,
)


def _frames_to_csv(frames: list[AudioFrame]) -> str:
    """Convert AudioFrame list to CSV string."""
    output = io.StringIO()
    fieldnames = [
        "timestamp", "time_sec", "rms_db", "rms_norm",
        "pitch_hz", "spectral_centroid", "is_speech", "volume_spike",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for frame in frames:
        writer.writerow(frame.to_dict())
    return output.getvalue()


def _frames_to_json(frames: list[AudioFrame]) -> str:
    """Convert AudioFrame list to JSON string."""
    data = [f.to_dict() for f in frames]
    return json.dumps(data, ensure_ascii=False)


def _format_frames(frames: list[AudioFrame], fmt: str) -> str:
    """Format frames as CSV or JSON."""
    if fmt == "json":
        return _frames_to_json(frames)
    return _frames_to_csv(frames)


def _format_error(error: Exception) -> str:
    """Format an exception into an actionable error message."""
    if isinstance(error, DownloadError):
        return (
            f"Download Error: {error}\n\n"
            "Suggestions:\n"
            "- Check the YouTube URL is correct and the video is public\n"
            "- Ensure yt-dlp and ffmpeg are installed\n"
            "- Try updating yt-dlp: pip install -U yt-dlp"
        )
    elif isinstance(error, AnalysisError):
        return (
            f"Analysis Error: {error}\n\n"
            "Suggestions:\n"
            "- Ensure the file exists and is a valid audio/video format\n"
            "- Check that librosa and soundfile are installed\n"
            "- For large files, analysis may take several minutes"
        )
    elif isinstance(error, ValueError):
        return f"Input Error: {error}"
    else:
        return f"Unexpected Error: {type(error).__name__}: {error}"
