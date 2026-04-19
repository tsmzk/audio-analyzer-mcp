"""Audio Analyzer MCP Server.

Provides tools for extracting audio features (volume, pitch, speech)
from YouTube videos or local audio/video files using librosa.

Tools:
    analyze_youtube_audio  — Download and analyze audio from a YouTube URL
    analyze_local_audio    — Analyze audio from a local file
    detect_highlights      — Find volume spikes and high-energy moments (shortcut for short-form video selection)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import sys
from typing import Callable, Optional

import numpy as np
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from audio_analyzer_mcp.audio_client import (
    AudioAnalysisError,
    AudioFrame,
    AnalysisError,
    DownloadError,
    analyze_local,
    analyze_youtube,
    SAMPLE_RATE as DEFAULT_SAMPLE_RATE,
    HOP_LENGTH as DEFAULT_HOP_LENGTH,
    FRAME_LENGTH as DEFAULT_FRAME_LENGTH,
)

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
# Input models
# ------------------------------------------------------------------


def _validate_frame_vs_hop(frame_length: int, hop_length: int) -> None:
    """Ensure frame_length is at least hop_length.

    librosa's internal resampling fails opaquely when frame_length < hop_length
    (e.g. ParameterError: Target size must be at least input size). Reject here
    with a clearer message.
    """
    if frame_length < hop_length:
        raise ValueError(
            f"frame_length ({frame_length}) must be >= hop_length ({hop_length}). "
            "Typical values: frame_length=4096, hop_length=2048."
        )


class AnalyzeYouTubeAudioInput(BaseModel):
    """Input for analyzing audio from a YouTube video."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    youtube_url: str = Field(
        ...,
        description=(
            "Public YouTube video URL. "
            "Formats: https://www.youtube.com/watch?v=XXXXX or https://youtu.be/XXXXX"
        ),
        min_length=10,
        max_length=500,
    )
    format: str = Field(
        default="csv",
        description=(
            "Output format: 'csv' (1 row per second, good for data processing) "
            "or 'json' (structured, good for programmatic use). Default: csv"
        ),
    )
    sample_rate: int = Field(
        default=DEFAULT_SAMPLE_RATE,
        description=(
            f"Sample rate in Hz. Lower = faster but less precise pitch. "
            f"Default: {DEFAULT_SAMPLE_RATE}. Use 22050 for max quality, 8000 for speed."
        ),
        ge=4000,
        le=44100,
    )
    hop_length: int = Field(
        default=DEFAULT_HOP_LENGTH,
        description=(
            f"Hop length in samples. Higher = faster, fewer frames per second. "
            f"Default: {DEFAULT_HOP_LENGTH}. Use 512 for fine granularity, 2048+ for speed."
        ),
        ge=128,
        le=8192,
    )
    frame_length: int = Field(
        default=DEFAULT_FRAME_LENGTH,
        description=(
            f"Frame length in samples. Default: {DEFAULT_FRAME_LENGTH}."
        ),
        ge=512,
        le=16384,
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_frame_hop(self) -> "AnalyzeYouTubeAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        return self


class AnalyzeLocalAudioInput(BaseModel):
    """Input for analyzing a local audio/video file."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    file_path: str = Field(
        ...,
        description=(
            "Absolute path to an audio or video file. "
            "Supported: wav, mp3, flac, ogg, mp4, mkv, avi, mov, webm, etc."
        ),
        min_length=1,
        max_length=1000,
    )
    format: str = Field(
        default="csv",
        description="Output format: 'csv' or 'json'. Default: csv",
    )
    sample_rate: int = Field(
        default=DEFAULT_SAMPLE_RATE,
        description=f"Sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}.",
        ge=4000, le=44100,
    )
    hop_length: int = Field(
        default=DEFAULT_HOP_LENGTH,
        description=f"Hop length in samples. Default: {DEFAULT_HOP_LENGTH}.",
        ge=128, le=8192,
    )
    frame_length: int = Field(
        default=DEFAULT_FRAME_LENGTH,
        description=f"Frame length in samples. Default: {DEFAULT_FRAME_LENGTH}.",
        ge=512, le=16384,
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_frame_hop(self) -> "AnalyzeLocalAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        return self


class DetectHighlightsInput(BaseModel):
    """Input for detecting highlight moments from a YouTube video."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    youtube_url: str = Field(
        ...,
        description="Public YouTube video URL.",
        min_length=10,
        max_length=500,
    )
    top_n: int = Field(
        default=20,
        description=(
            "Maximum number of highlight moments to return. Default: 20. "
            "Highlights are ranked by volume spike intensity."
        ),
        ge=1,
        le=100,
    )
    min_gap_sec: int = Field(
        default=10,
        description=(
            "Minimum gap in seconds between reported highlights. "
            "Prevents clustering of nearby spikes. Default: 10"
        ),
        ge=1,
        le=300,
    )
    sample_rate: int = Field(
        default=DEFAULT_SAMPLE_RATE,
        description=f"Sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}.",
        ge=4000, le=44100,
    )
    hop_length: int = Field(
        default=DEFAULT_HOP_LENGTH,
        description=f"Hop length in samples. Default: {DEFAULT_HOP_LENGTH}.",
        ge=128, le=8192,
    )
    frame_length: int = Field(
        default=DEFAULT_FRAME_LENGTH,
        description=f"Frame length in samples. Default: {DEFAULT_FRAME_LENGTH}.",
        ge=512, le=16384,
    )

    @model_validator(mode="after")
    def validate_frame_hop(self) -> "DetectHighlightsInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        return self


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


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


def _detect_highlight_moments(
    frames: list[AudioFrame],
    top_n: int = 20,
    min_gap_sec: int = 10,
) -> list[dict]:
    """Detect highlight moments from analyzed frames.

    Looks for:
    1. Volume spikes (sudden loudness jumps)
    2. High-volume speech moments (top percentile)
    3. High-pitch moments (excitement indicators)

    Returns ranked list of highlights with scores.
    """
    candidates: list[dict] = []

    # Speech-only frames
    speech_frames = [f for f in frames if f.is_speech]
    if not speech_frames:
        return []

    # Stats for scoring
    rms_values = [f.rms_db for f in speech_frames]
    pitch_values = [f.pitch_hz for f in speech_frames if f.pitch_hz > 0]
    rms_p90 = float(np.percentile(rms_values, 90))
    rms_p95 = float(np.percentile(rms_values, 95))
    pitch_p90 = float(np.percentile(pitch_values, 90)) if pitch_values else 200.0

    for frame in frames:
        if not frame.is_speech:
            continue

        score = 0.0
        reasons = []

        # Volume spike (strongest signal)
        if frame.volume_spike:
            score += 50.0
            reasons.append("volume_spike")

        # Very loud (top 5%)
        if frame.rms_db >= rms_p95:
            score += 30.0
            reasons.append("very_loud")
        elif frame.rms_db >= rms_p90:
            score += 15.0
            reasons.append("loud")

        # High pitch (excitement)
        if frame.pitch_hz >= pitch_p90 and frame.pitch_hz > 0:
            score += 20.0
            reasons.append("high_pitch")

        # High spectral centroid (sharp/bright voice = ツッコミ等)
        centroid_values = [f.spectral_centroid for f in speech_frames if f.spectral_centroid > 0]
        if centroid_values:
            centroid_p90 = float(np.percentile(centroid_values, 90))
            if frame.spectral_centroid >= centroid_p90:
                score += 10.0
                reasons.append("sharp_voice")

        if score > 0:
            candidates.append({
                "timestamp": frame.timestamp,
                "time_sec": frame.time_sec,
                "score": round(score, 1),
                "rms_db": frame.rms_db,
                "rms_norm": frame.rms_norm,
                "pitch_hz": frame.pitch_hz,
                "reasons": reasons,
            })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # De-duplicate (enforce minimum gap)
    selected: list[dict] = []
    used_times: set[int] = set()
    for c in candidates:
        t = c["time_sec"]
        if any(abs(t - ut) < min_gap_sec for ut in used_times):
            continue
        selected.append(c)
        used_times.add(t)
        if len(selected) >= top_n:
            break

    return selected


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


# ------------------------------------------------------------------
# Progress bridging
# ------------------------------------------------------------------

# Interval between heartbeat progress events while the analysis thread is
# inside a phase that doesn't emit its own progress (notably librosa.pyin).
# Claude Desktop times out MCP tool calls at ~4 minutes of silence; sending a
# heartbeat every 15s keeps the client-side timeout reset with comfortable
# margin.
HEARTBEAT_INTERVAL_SEC = 15.0


def _make_progress_bridge(
    ctx: Context,
    loop: asyncio.AbstractEventLoop,
) -> Callable[[str, float], None]:
    """Return a thread-safe progress callback that forwards to `ctx`.

    The analysis runs inside `asyncio.to_thread`, so callbacks fire on a
    worker thread and cannot `await` directly. We schedule the async
    `ctx.report_progress` / `ctx.info` coroutines onto the main event loop.
    """

    def _callback(message: str, fraction: float) -> None:
        progress = max(0.0, min(1.0, fraction)) * 100.0
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=progress, total=100.0, message=message),
                loop,
            )
            asyncio.run_coroutine_threadsafe(ctx.info(message), loop)
        except RuntimeError:
            # Event loop is gone (request cancelled). Nothing to do.
            pass

    return _callback


async def _run_with_heartbeat(
    ctx: Context,
    runner: Callable[[Callable[[str, float], None]], Any],
) -> Any:
    """Run a blocking `runner(progress_cb)` in a thread, emitting heartbeats.

    `runner` is invoked with a progress callback it can pass down into the
    analysis pipeline. While it runs, a heartbeat task keeps the MCP client
    alive by re-emitting the last seen progress every HEARTBEAT_INTERVAL_SEC.
    """
    loop = asyncio.get_running_loop()
    last_state: dict[str, Any] = {"message": "Starting...", "fraction": 0.0}
    bridge = _make_progress_bridge(ctx, loop)

    def _cb(message: str, fraction: float) -> None:
        last_state["message"] = message
        last_state["fraction"] = fraction
        bridge(message, fraction)

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            msg = f"Still working: {last_state['message']}"
            try:
                await ctx.report_progress(
                    progress=max(0.0, min(1.0, float(last_state["fraction"]))) * 100.0,
                    total=100.0,
                    message=msg,
                )
            except Exception:
                return

    hb_task = asyncio.create_task(_heartbeat())
    try:
        return await asyncio.to_thread(runner, _cb)
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass


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
