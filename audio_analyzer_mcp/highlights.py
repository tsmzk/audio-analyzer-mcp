"""Highlight-moment detection from analyzed audio frames.

Ranks moments by a combination of volume spikes, overall loudness, high
pitch (excitement), and spectral sharpness (ツッコミ-like exclamations),
then deduplicates nearby picks via a minimum time gap.
"""

from __future__ import annotations

import numpy as np

from audio_analyzer_mcp.constants import AudioFrame


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
