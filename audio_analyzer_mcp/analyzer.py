"""Audio analysis using librosa.

Extracts per-second features from an audio or video file:
- RMS volume (dB)
- Pitch / fundamental frequency (Hz) via pyin
- Spectral centroid (voice sharpness)
- Speech detection (voiced frames + volume threshold)
- Volume spike detection (sudden loudness jumps)

High-level entry points:
- analyze_youtube: download + analyze a YouTube URL
- analyze_local: analyze a local audio/video file
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from audio_analyzer_mcp.constants import (
    AnalysisError,
    AudioFrame,
    FRAME_LENGTH,
    HOP_LENGTH,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_DB,
    SPIKE_LOOKBACK_SEC,
    SPIKE_THRESHOLD_DB,
)
from audio_analyzer_mcp.downloader import (
    ProgressCallback,
    _emit,
    download_youtube_audio,
)

logger = logging.getLogger(__name__)


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
