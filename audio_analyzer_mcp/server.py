"""Audio Analyzer MCP Server.

Provides tools for extracting audio features (volume, pitch, speech)
from YouTube videos or local audio/video files using librosa.

Tools:
    analyze_youtube_audio  — Download and analyze audio from a YouTube URL
    analyze_local_audio    — Analyze audio from a local file
    detect_highlights      — Find volume spikes and high-energy moments (shortcut for short-form video selection)
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Callable

from mcp.server.fastmcp import Context, FastMCP

from audio_analyzer_mcp.analyzer import analyze_local, analyze_youtube
from audio_analyzer_mcp.formatters import _format_error, _format_frames
from audio_analyzer_mcp.highlights import _detect_highlight_moments
from audio_analyzer_mcp.models import (
    AnalyzeLocalAudioInput,
    AnalyzeYouTubeAudioInput,
    DetectHighlightsInput,
)
from audio_analyzer_mcp.progress import _run_with_heartbeat

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("audio_analyzer_mcp")

# ------------------------------------------------------------------
# Server instance
# ------------------------------------------------------------------

mcp = FastMCP("audio_analyzer_mcp")


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


@mcp.tool(
    name="analyze_youtube_audio",
    annotations={
        "title": "Analyze YouTube Audio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def analyze_youtube_audio(
    params: AnalyzeYouTubeAudioInput,
    ctx: Context,
) -> str:
    """Analyze audio from a YouTube video.

    Downloads the audio track and extracts per-second features:
    - rms_db: Volume in decibels (higher = louder)
    - rms_norm: Volume normalized 0-100
    - pitch_hz: Voice pitch in Hz (higher = more excited)
    - spectral_centroid: Voice sharpness (higher = sharper)
    - is_speech: Whether someone is speaking
    - volume_spike: Sudden volume increase (highlight candidate)

    Output is CSV or JSON, one row per second. Use time_sec to match
    with subtitle data from youtube-transcript MCP.

    Args:
        params: youtube_url and output format.

    Returns:
        CSV or JSON string with per-second audio analysis.
    """
    try:
        def _runner(cb: Callable[[str, float], None]):
            return analyze_youtube(
                params.youtube_url,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)
        return _format_frames(frames, params.format)
    except Exception as exc:
        logger.error("analyze_youtube_audio failed: %s", exc, exc_info=True)
        return _format_error(exc)


@mcp.tool(
    name="analyze_local_audio",
    annotations={
        "title": "Analyze Local Audio File",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def analyze_local_audio_tool(
    params: AnalyzeLocalAudioInput,
    ctx: Context,
) -> str:
    """Analyze audio from a local file.

    Supports audio files (wav, mp3, flac, ogg) and video files
    (mp4, mkv, avi, mov, webm) — the audio track is automatically extracted.

    Output is the same per-second CSV/JSON as analyze_youtube_audio.

    Args:
        params: file_path and output format.

    Returns:
        CSV or JSON string with per-second audio analysis.
    """
    try:
        def _runner(cb: Callable[[str, float], None]):
            return analyze_local(
                params.file_path,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)
        return _format_frames(frames, params.format)
    except Exception as exc:
        logger.error("analyze_local_audio failed: %s", exc, exc_info=True)
        return _format_error(exc)


@mcp.tool(
    name="detect_highlights",
    annotations={
        "title": "Detect Audio Highlights",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def detect_highlights(
    params: DetectHighlightsInput,
    ctx: Context,
) -> str:
    """Detect highlight moments from a YouTube video's audio.

    Analyzes the full audio track and returns the most notable moments,
    ranked by a combination of:
    - Volume spikes (sudden jumps in loudness)
    - Overall loudness (top percentile)
    - High pitch (excitement indicator)
    - Voice sharpness (e.g. ツッコミ, exclamations)

    Each highlight includes a score, timestamp, and reasons.
    Use with subtitle data to see what was said at each moment.

    Args:
        params: youtube_url, top_n (max results), min_gap_sec (minimum gap between results).

    Returns:
        JSON array of highlight moments with scores.
    """
    try:
        def _runner(cb: Callable[[str, float], None]):
            return analyze_youtube(
                params.youtube_url,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)
        highlights = _detect_highlight_moments(
            frames,
            top_n=params.top_n,
            min_gap_sec=params.min_gap_sec,
        )

        # Add summary stats
        speech_frames = [f for f in frames if f.is_speech]
        summary = {
            "total_duration_sec": len(frames),
            "speech_seconds": len(speech_frames),
            "silence_seconds": len(frames) - len(speech_frames),
            "total_volume_spikes": sum(1 for f in frames if f.volume_spike),
            "highlights": highlights,
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("detect_highlights failed: %s", exc, exc_info=True)
        return _format_error(exc)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run()


if __name__ == "__main__":
    main()
