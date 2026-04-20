"""Shared constants, exceptions, and the AudioFrame data structure.

Pulled out so every module (downloader, analyzer, models, formatters,
highlights, server) can depend on a single, stable source of truth.
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------
# yt-dlp download config
# ------------------------------------------------------------------

DOWNLOAD_TIMEOUT_SEC = 600  # 10 min max for long videos

# ------------------------------------------------------------------
# librosa analysis config
# ------------------------------------------------------------------

SAMPLE_RATE = 8000        # Lower SR for speed — sufficient for voice pitch (65-2000Hz)
HOP_LENGTH = 2048         # ~256ms per frame at 8000Hz — 4 frames/sec (fast, 1-sec aggregation用)
FRAME_LENGTH = 4096       # ~512ms window

# ------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------

SILENCE_THRESHOLD_DB = -50.0    # Below this = no speech
SPIKE_THRESHOLD_DB = 10.0       # dB jump to flag as spike
SPIKE_LOOKBACK_SEC = 3          # Compare against last N seconds


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


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
