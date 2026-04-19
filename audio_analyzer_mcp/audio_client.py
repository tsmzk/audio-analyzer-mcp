"""Audio analysis client.

Handles:
- YouTube audio download via yt-dlp (audio-only, no video)
- Local audio/video file loading
- Per-second audio feature extraction via librosa:
  - RMS volume (dB)
  - Pitch / fundamental frequency (Hz) via pyin
  - Spectral centroid (voice sharpness)
  - Speech detection (voiced frames + volume threshold)
  - Volume spike detection (sudden loudness jumps)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import librosa
import numpy as np

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

logger = logging.getLogger(__name__)

# yt-dlp download config
DOWNLOAD_TIMEOUT_SEC = 600  # 10 min max for long videos

# librosa analysis config
SAMPLE_RATE = 8000        # Lower SR for speed — sufficient for voice pitch (65-2000Hz)
HOP_LENGTH = 2048         # ~256ms per frame at 8000Hz — 4 frames/sec (fast, 1-sec aggregation用)
FRAME_LENGTH = 4096       # ~512ms window

# Thresholds
SILENCE_THRESHOLD_DB = -50.0    # Below this = no speech
SPIKE_THRESHOLD_DB = 10.0       # dB jump to flag as spike
SPIKE_LOOKBACK_SEC = 3          # Compare against last N seconds


class AudioAnalysisError(Exception):
    """Base error for audio analysis operations."""


class DownloadError(AudioAnalysisError):
    """Raised when audio download fails."""


class AnalysisError(AudioAnalysisError):
    """Raised when librosa analysis fails."""


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

class AudioFrame:
    """Analysis data for a single 1-second frame."""

    __slots__ = (
        "time_sec", "timestamp", "rms_db", "rms_norm",
        "pitch_hz", "spectral_centroid", "is_speech", "volume_spike",
    )

    def __init__(
        self,
        time_sec: int,
        rms_db: float,
        rms_norm: float,
        pitch_hz: float,
        spectral_centroid: float,
        is_speech: bool,
        volume_spike: bool,
    ):
        self.time_sec = time_sec
        self.timestamp = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
        self.rms_db = rms_db
        self.rms_norm = rms_norm
        self.pitch_hz = pitch_hz
        self.spectral_centroid = spectral_centroid
        self.is_speech = is_speech
        self.volume_spike = volume_spike

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "time_sec": self.time_sec,
            "rms_db": self.rms_db,
            "rms_norm": self.rms_norm,
            "pitch_hz": self.pitch_hz,
            "spectral_centroid": self.spectral_centroid,
            "is_speech": self.is_speech,
            "volume_spike": self.volume_spike,
        }


# ------------------------------------------------------------------
# YouTube download
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Audio analysis
# ------------------------------------------------------------------

def analyze_audio_file(
    file_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """Analyze an audio or video file and extract per-second features.

    Args:
        file_path: Path to an audio file (WAV, MP3, FLAC, etc.)
                   or a video file (MP4, MKV, etc. — audio track is extracted).
        sample_rate: Sample rate in Hz. Lower = faster but less precise.
                     Default 8000. Use 22050 for maximum quality.
        hop_length: Hop length in samples. Higher = faster, fewer frames per second.
                    Default 2048. Use 512 for finer granularity.
        frame_length: Frame length in samples. Default 4096.

    Returns:
        List of AudioFrame objects, one per second of audio.
    """
    path = Path(file_path)
    if not path.exists():
        raise AnalysisError(f"File not found: {file_path}")

    logger.info("Loading audio: %s", path.name)
    _emit(progress_cb, f"Loading audio: {path.name}", 0.20)
    try:
        y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:
        raise AnalysisError(f"Failed to load audio: {exc}") from exc

    duration = librosa.get_duration(y=y, sr=sr)
    logger.info("Duration: %d:%02d (%d seconds)", int(duration // 60), int(duration % 60), int(duration))
    _emit(
        progress_cb,
        f"Duration: {int(duration // 60)}:{int(duration % 60):02d}",
        0.25,
    )

    # --- Feature extraction (fast) ---
    _emit(progress_cb, "Extracting features (RMS, spectral centroid, ZCR)...", 0.30)

    # RMS (volume) — very fast
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Spectral centroid (voice sharpness) — fast
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # Zero-crossing rate (speech vs silence heuristic) — fast
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

    # --- Pitch estimation (pyin) ---
    logger.info("Estimating pitch...")
    _emit(progress_cb, "Estimating pitch (slowest phase)...", 0.40)
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    _emit(progress_cb, "Pitch estimation complete", 0.80)

    # --- Aggregate per second ---
    total_seconds = int(duration)
    results: list[AudioFrame] = []

    # Global RMS stats for normalization
    rms_db_valid = rms_db[rms_db > -80.0]
    if len(rms_db_valid) > 0:
        rms_db_min = float(np.percentile(rms_db_valid, 1))
        rms_db_max = float(np.percentile(rms_db_valid, 99))
    else:
        rms_db_min, rms_db_max = -60.0, 0.0
    rms_db_range = max(rms_db_max - rms_db_min, 1.0)

    logger.info("Processing %d seconds...", total_seconds)
    _emit(progress_cb, f"Aggregating per-second frames ({total_seconds}s)...", 0.82)
    # Emit aggregation progress at ~10% increments to keep clients alive
    # without flooding the notification channel.
    next_report_sec = max(total_seconds // 10, 1)
    for sec in range(total_seconds):
        t_start = float(sec)
        t_end = float(sec + 1)

        # Mask for this second
        rms_mask = (rms_times >= t_start) & (rms_times < t_end)

        # RMS (volume in dB)
        sec_rms = rms_db[rms_mask]
        avg_rms_db = float(np.nanmean(sec_rms)) if len(sec_rms) > 0 else -80.0

        # Normalized volume (0-100)
        rms_norm = max(0.0, min(100.0, (avg_rms_db - rms_db_min) / rms_db_range * 100.0))

        # Pitch (Hz)
        pitch_mask = (pitch_times >= t_start) & (pitch_times < t_end)
        sec_pitch = f0[pitch_mask]
        valid_pitch = sec_pitch[~np.isnan(sec_pitch)] if len(sec_pitch) > 0 else np.array([])
        avg_pitch = float(np.nanmean(valid_pitch)) if len(valid_pitch) > 0 else 0.0

        # Spectral centroid
        sec_centroid = spectral_centroids[rms_mask]
        avg_centroid = float(np.nanmean(sec_centroid)) if len(sec_centroid) > 0 else 0.0

        # Speech detection (RMS + ZCR based, no pitch needed)
        sec_zcr = zcr[rms_mask]
        avg_zcr = float(np.nanmean(sec_zcr)) if len(sec_zcr) > 0 else 0.0
        # Speech: above silence threshold + moderate ZCR (not pure noise)
        # + spectral centroid in speech range (300-5000 Hz typical)
        is_speech = (
            avg_rms_db > SILENCE_THRESHOLD_DB
            and 0.01 < avg_zcr < 0.30
            and 300.0 < avg_centroid < 5000.0
        )

        # Volume spike detection
        volume_spike = False
        if sec >= SPIKE_LOOKBACK_SEC and len(results) >= SPIKE_LOOKBACK_SEC:
            prev_avg = np.mean([r.rms_db for r in results[-SPIKE_LOOKBACK_SEC:]])
            if avg_rms_db - prev_avg > SPIKE_THRESHOLD_DB:
                volume_spike = True

        results.append(AudioFrame(
            time_sec=sec,
            rms_db=round(avg_rms_db, 1),
            rms_norm=round(rms_norm, 1),
            pitch_hz=round(avg_pitch, 1),
            spectral_centroid=round(avg_centroid, 1),
            is_speech=is_speech,
            volume_spike=volume_spike,
        ))

        if sec and sec % next_report_sec == 0 and total_seconds > 0:
            frac = 0.82 + 0.15 * (sec / total_seconds)
            _emit(progress_cb, f"Aggregated {sec}/{total_seconds}s", frac)

    logger.info("Analysis complete: %d frames", len(results))
    _emit(progress_cb, f"Analysis complete: {len(results)} frames", 1.0)
    return results


# ------------------------------------------------------------------
# High-level entry points
# ------------------------------------------------------------------

def analyze_youtube(
    youtube_url: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """Download and analyze audio from a YouTube video.

    Args:
        youtube_url: Public YouTube video URL.
        sample_rate: Sample rate in Hz. Default 8000.
        hop_length: Hop length in samples. Default 2048.
        frame_length: Frame length in samples. Default 4096.
        progress_cb: Optional (message, fraction) callback for progress reporting.

    Returns:
        List of AudioFrame objects, one per second.
    """
    with tempfile.TemporaryDirectory(prefix="audio_analyzer_") as tmpdir:
        wav_path = download_youtube_audio(youtube_url, tmpdir, progress_cb=progress_cb)
        return analyze_audio_file(
            wav_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            frame_length=frame_length,
            progress_cb=progress_cb,
        )


def analyze_local(
    file_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """Analyze audio from a local file.

    Args:
        file_path: Path to audio or video file.
        sample_rate: Sample rate in Hz. Default 8000.
        hop_length: Hop length in samples. Default 2048.
        frame_length: Frame length in samples. Default 4096.
        progress_cb: Optional (message, fraction) callback for progress reporting.

    Returns:
        List of AudioFrame objects, one per second.
    """
    return analyze_audio_file(
        file_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_length=frame_length,
        progress_cb=progress_cb,
    )
