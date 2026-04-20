"""YouTube audio download via yt-dlp.

Handles:
- yt-dlp / ffmpeg availability checks
- YouTube URL validation
- Duration probing (for progress reporting)
- Audio-only download and WAV conversion
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from audio_analyzer_mcp.constants import (
    DOWNLOAD_TIMEOUT_SEC,
    DownloadError,
    SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# Callback signature: (message, progress_fraction_0_to_1)
ProgressCallback = Callable[[str, float], None]


def _emit(cb: Optional[ProgressCallback], message: str, fraction: float) -> None:
    """Forward a progress event to the callback, swallowing callback errors.

    Progress callbacks are best-effort — if the consumer raises we don't want
    to abort the analysis pipeline.
    """
    if cb is None:
        return
    try:
        cb(message, fraction)
    except Exception:
        logger.debug("progress callback raised", exc_info=True)


def _check_yt_dlp() -> None:
    """Verify yt-dlp is available."""
    if not shutil.which("yt-dlp"):
        raise DownloadError(
            "yt-dlp not found. Install it with: pip install yt-dlp"
        )


def _check_ffmpeg() -> None:
    """Verify ffmpeg is available."""
    if not shutil.which("ffmpeg"):
        raise DownloadError(
            "ffmpeg not found. Install it with: brew install ffmpeg (Mac) "
            "or apt install ffmpeg (Linux)"
        )


def _validate_youtube_url(url: str) -> None:
    """Basic validation that the URL looks like a YouTube link."""
    url_lower = url.strip().lower()
    valid_prefixes = (
        "https://www.youtube.com/watch",
        "https://youtube.com/watch",
        "https://youtu.be/",
        "https://m.youtube.com/watch",
        "http://www.youtube.com/watch",
        "http://youtube.com/watch",
        "http://youtu.be/",
    )
    if not any(url_lower.startswith(p) for p in valid_prefixes):
        raise DownloadError(
            f"Invalid YouTube URL: {url}\n"
            "Expected: https://www.youtube.com/watch?v=XXXXX or https://youtu.be/XXXXX"
        )


def _probe_youtube_duration(youtube_url: str) -> Optional[int]:
    """Ask yt-dlp for the video duration in seconds without downloading.

    Returns None if the probe fails — this is best-effort metadata used only
    for progress reporting, so failures must not abort the actual download.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--print", "%(duration)s",
                "--skip-download",
                "--no-playlist",
                "--no-warnings",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return int(float(line))
    except (ValueError, IndexError):
        return None


def download_youtube_audio(
    youtube_url: str,
    output_dir: str,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """Download audio-only from YouTube and convert to WAV.

    Args:
        youtube_url: Public YouTube video URL.
        output_dir: Directory to save the WAV file.
        progress_cb: Optional (message, fraction) callback for progress reporting.

    Returns:
        Path to the downloaded WAV file.
    """
    _check_yt_dlp()
    _check_ffmpeg()
    _validate_youtube_url(youtube_url)

    output_template = str(Path(output_dir) / "audio.%(ext)s")
    wav_path = str(Path(output_dir) / "audio.wav")

    _emit(progress_cb, "Probing video metadata...", 0.02)
    duration_sec = _probe_youtube_duration(youtube_url)
    if duration_sec is not None:
        mins = duration_sec // 60
        secs = duration_sec % 60
        logger.info("Video duration: %d:%02d", mins, secs)
        _emit(
            progress_cb,
            f"Video is {mins}:{secs:02d} — starting download",
            0.05,
        )
    else:
        _emit(progress_cb, "Starting download (duration unknown)", 0.05)

    logger.info("Downloading audio from: %s", youtube_url)
    cmd = [
        "yt-dlp",
        "-x",                        # Extract audio only
        "--audio-format", "wav",     # Convert to WAV
        "--audio-quality", "0",      # Best quality
        "--no-playlist",
        "--no-check-certificates",
        "-o", output_template,
        youtube_url,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT_SEC,
    )

    if result.returncode != 0:
        raise DownloadError(
            f"yt-dlp failed (exit code {result.returncode}):\n{result.stderr[-500:]}"
        )

    # Find the output file (yt-dlp may produce different extensions)
    output_dir_path = Path(output_dir)
    audio_files = sorted(output_dir_path.glob("audio.*"))
    if not audio_files:
        raise DownloadError("Download completed but no audio file was found.")

    actual_path = audio_files[0]
    _emit(progress_cb, "Download complete", 0.15)

    # Convert to WAV if needed (yt-dlp sometimes produces other formats)
    if actual_path.suffix.lower() != ".wav":
        logger.info("Converting %s → WAV...", actual_path.suffix)
        _emit(progress_cb, f"Converting {actual_path.suffix} → WAV", 0.17)
        convert_cmd = [
            "ffmpeg", "-y", "-i", str(actual_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1",  # Mono, target sample rate
            wav_path,
        ]
        conv_result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if conv_result.returncode != 0:
            raise DownloadError(f"ffmpeg conversion failed:\n{conv_result.stderr[-500:]}")
        actual_path.unlink()
        return wav_path

    return str(actual_path)
