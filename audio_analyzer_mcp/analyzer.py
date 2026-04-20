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

長尺(20分超)ファイルは内部で CHUNK_DURATION_SEC 刻みに分割して順に処理し、
最後にフレームを結合 → rms_norm を全体で再計算する。
"""

from __future__ import annotations

import logging
# tempfile: 一時ディレクトリ/ファイルを作る標準ライブラリ。
# YouTubeから落としたWAVファイルを一時置き場に置き、解析後に自動削除する。
import tempfile
from pathlib import Path
from typing import Optional

# librosa: 音声解析のデファクトPythonライブラリ。
#   - ファイル読み込み(librosa.load)
#   - 特徴量抽出(RMS, スペクトル重心, ピッチ…)
#   - 時間/フレーム変換
# を一通り提供する。
import librosa

# numpy(np): 数値配列を高速に扱う標準ライブラリ。
# librosaの出力は全て numpy配列(ndarray)なので必須。
import numpy as np

from audio_analyzer_mcp.constants import (
    AUTO_CHUNK_THRESHOLD_SEC,
    AnalysisError,
    AudioFrame,
    CHUNK_DURATION_SEC,
    CHUNK_OVERLAP_SEC,
    FRAME_LENGTH,
    HOP_LENGTH,
    NYQUIST_SAFETY,
    PITCH_FMAX_HZ,
    PITCH_FMIN_HZ,
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


def _resolve_pitch_range(sample_rate: int) -> tuple[float, float]:
    """pyin に渡す fmin/fmax を sample_rate に合わせて安全に決める。

    fmax は Nyquist 周波数 (sr/2) を超えられない。
    さらに pyin は内部で少し余裕を取るので、NYQUIST_SAFETY (=2.1) で割る。

    サンプルレートが低すぎて有効な範囲が作れない場合は AnalysisError を投げる。
    """
    fmin = float(PITCH_FMIN_HZ)
    fmax = min(float(PITCH_FMAX_HZ), sample_rate / NYQUIST_SAFETY)

    if fmax <= fmin:
        raise AnalysisError(
            f"sample_rate={sample_rate}Hz is too low for pitch detection. "
            f"At this rate the usable fmax is {fmax:.1f}Hz, which is below "
            f"fmin={fmin:.1f}Hz. Increase sample_rate to at least "
            f"{int(PITCH_FMIN_HZ * NYQUIST_SAFETY) + 1}Hz (recommended: 8000)."
        )
    return fmin, fmax


def _compute_pyin_max_transition_rate(
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
    resolution: float = 0.1,
) -> float:
    """pyin の内部 `transition_local` が落ちない max_transition_rate を計算する。

    librosa.pyin は内部で
        n_pitch_bins     = 12 * n_bins_per_semitone * log2(fmax/fmin) + 1
        transition_width = round(max_transition_rate * 12 * hop/sr) * n_bins_per_semitone + 1
    を計算し、`transition_width <= n_pitch_bins` を要求する。
    ここが崩れると `Target size (N) must be at least input size (M)` で落ちる。

    デフォルトの `max_transition_rate=35.92` は `hop_length/sr` が小さい(細かい解析)
    前提で決められており、本プロジェクトの既定 (hop=2048, sr=8000) では制約を超えてしまう。
    そこで、制約を満たす最大値を逆算し、デフォルトと比較して小さい方を採用する。
    """
    default_max = 35.92  # librosa.pyin のデフォルト
    # 制約を満たす最大値: max_transition_rate * hop/sr < log2(fmax/fmin)
    # round() と +1 の余裕を取り、安全率 0.9 を掛ける。
    log2_ratio = float(np.log2(fmax / fmin))
    safe_upper = log2_ratio * sr / hop_length * 0.9
    return float(min(default_max, safe_upper))


def _run_pyin_safe(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """librosa.pyin を呼び出すラッパー。失敗時は NaN 埋めの配列で代用する。

    加えて、内部整合性(transition_width <= n_pitch_bins)を守るため、
    max_transition_rate をパラメータから動的計算して渡す。

    Returns:
        (f0, pitch_times) の組。pitch_times はフレーム中心の秒。
    """
    fmin, fmax = _resolve_pitch_range(sr)
    max_transition_rate = _compute_pyin_max_transition_rate(
        sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    try:
        f0, _voiced_flag, _voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            max_transition_rate=max_transition_rate,
        )
    except Exception as exc:
        # librosa の想定外内部例外をユーザー向けに格下げ。
        # ピッチなし扱いで処理を継続する(RMS/重心/発話判定はそのまま動く)。
        logger.warning(
            "librosa.pyin failed (%s); falling back to no-pitch mode for this chunk.",
            exc,
        )
        n_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
        f0 = np.full(n_frames, np.nan)

    pitch_times = librosa.frames_to_time(
        np.arange(len(f0)), sr=sr, hop_length=hop_length
    )
    return f0, pitch_times


def _analyze_waveform(
    y: np.ndarray,
    sr: int,
    *,
    hop_length: int,
    frame_length: int,
    time_offset_sec: int = 0,
    progress_cb: Optional[ProgressCallback] = None,
    progress_span: tuple[float, float] = (0.25, 1.0),
    chunk_label: str = "",
) -> list[AudioFrame]:
    """numpy 波形配列を 1秒刻みの AudioFrame リストに変換する(メイン解析コア)。

    ファイル I/O を含まないので、チャンク処理からも直接呼べる。
    time_offset_sec を指定すると、各 AudioFrame の time_sec にその値が加算される
    (チャンクのグローバル時刻復元用)。

    rms_norm はここでは暫定値のまま返す。全体の rms_db 分布が分かった後、
    analyze_audio_file 側で再計算する。
    """
    p_lo, p_hi = progress_span
    p_span = max(1e-6, p_hi - p_lo)

    def _frac(local: float) -> float:
        """チャンク内進捗(0.0〜1.0)をグローバル進捗に写像。"""
        return p_lo + max(0.0, min(1.0, local)) * p_span

    prefix = f"[{chunk_label}] " if chunk_label else ""

    # ── RMS(音量)──
    # RMS = Root Mean Square。波形の「平均的な強さ」を表す指標。
    # librosaは「フレーム(短い区間)ごと」にRMSを計算して配列で返す。
    # [0] を付けているのは「チャンネル0」を取り出すため(モノラルだが一応)。
    _emit(progress_cb, f"{prefix}Extracting features (RMS, centroid, ZCR)...", _frac(0.05))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    # 振幅の生値 → dB値 に変換(ref=np.max で最大値を0dBの基準にする)。
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # ── スペクトル重心(声の鋭さ)── 高いほど高音成分が多い=鋭い声。
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # ── ゼロクロッシング率(ZCR)── 発話検出の補助指標。
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

    # ── ピッチ推定(pyin)──  全体の処理時間の大半はここ。
    _emit(progress_cb, f"{prefix}Estimating pitch (slowest phase)...", _frac(0.15))
    f0, pitch_times = _run_pyin_safe(y, sr, hop_length, frame_length)
    _emit(progress_cb, f"{prefix}Pitch estimation complete", _frac(0.70))

    # ===== 1秒ごとに集約 =====
    duration = float(len(y)) / float(sr)
    total_seconds = int(duration)
    results: list[AudioFrame] = []

    if total_seconds <= 0:
        # 音声が 1 秒未満でも落ちないようにガード。
        return results

    _emit(
        progress_cb,
        f"{prefix}Aggregating per-second frames ({total_seconds}s)...",
        _frac(0.75),
    )

    next_report_sec = max(total_seconds // 10, 1)

    for sec in range(total_seconds):
        t_start = float(sec)
        t_end = float(sec + 1)

        rms_mask = (rms_times >= t_start) & (rms_times < t_end)

        sec_rms = rms_db[rms_mask]
        avg_rms_db = float(np.nanmean(sec_rms)) if len(sec_rms) > 0 else -80.0

        pitch_mask = (pitch_times >= t_start) & (pitch_times < t_end)
        sec_pitch = f0[pitch_mask]
        valid_pitch = sec_pitch[~np.isnan(sec_pitch)] if len(sec_pitch) > 0 else np.array([])
        avg_pitch = float(np.nanmean(valid_pitch)) if len(valid_pitch) > 0 else 0.0

        sec_centroid = spectral_centroids[rms_mask]
        avg_centroid = float(np.nanmean(sec_centroid)) if len(sec_centroid) > 0 else 0.0

        sec_zcr = zcr[rms_mask]
        avg_zcr = float(np.nanmean(sec_zcr)) if len(sec_zcr) > 0 else 0.0
        is_speech = (
            avg_rms_db > SILENCE_THRESHOLD_DB
            and 0.01 < avg_zcr < 0.30
            and 300.0 < avg_centroid < 5000.0
        )

        # ── 音量スパイク判定 ──
        # 直近 SPIKE_LOOKBACK_SEC 秒の平均より SPIKE_THRESHOLD_DB 以上大きければスパイク。
        # チャンク先頭は past history がないので判定しない(オーバーラップで解消済み)。
        volume_spike = False
        if sec >= SPIKE_LOOKBACK_SEC and len(results) >= SPIKE_LOOKBACK_SEC:
            prev_avg = np.mean([r.rms_db for r in results[-SPIKE_LOOKBACK_SEC:]])
            if avg_rms_db - prev_avg > SPIKE_THRESHOLD_DB:
                volume_spike = True

        results.append(AudioFrame(
            time_sec=sec + time_offset_sec,
            rms_db=round(avg_rms_db, 1),
            # rms_norm はここでは 0 を入れておき、呼び出し側で全体分布から再計算する。
            rms_norm=0.0,
            pitch_hz=round(avg_pitch, 1),
            spectral_centroid=round(avg_centroid, 1),
            is_speech=is_speech,
            volume_spike=volume_spike,
        ))

        if sec and sec % next_report_sec == 0 and total_seconds > 0:
            # チャンク内進捗: 0.75 → 0.97 の区間をリニアに
            _emit(
                progress_cb,
                f"{prefix}Aggregated {sec}/{total_seconds}s",
                _frac(0.75 + 0.22 * (sec / total_seconds)),
            )

    _emit(progress_cb, f"{prefix}Chunk complete: {len(results)} frames", _frac(1.0))
    return results


def _recompute_rms_norm(frames: list[AudioFrame]) -> None:
    """全体の rms_db 分布から rms_norm(0〜100)を再計算して埋める(in-place)。

    チャンクごとに計算すると境界で 0/100 がずれてしまうため、
    全チャンク集約後に1回だけ呼び出して統一する。
    """
    if not frames:
        return

    rms_db_array = np.array([f.rms_db for f in frames])
    valid = rms_db_array[rms_db_array > -80.0]
    if len(valid) > 0:
        rms_db_min = float(np.percentile(valid, 1))
        rms_db_max = float(np.percentile(valid, 99))
    else:
        rms_db_min, rms_db_max = -60.0, 0.0
    rms_db_range = max(rms_db_max - rms_db_min, 1.0)

    for f in frames:
        norm = (f.rms_db - rms_db_min) / rms_db_range * 100.0
        f.rms_norm = round(max(0.0, min(100.0, norm)), 1)


def _probe_duration(path: Path) -> float:
    """ファイルの長さ(秒)だけを取得する。ここでは波形をロードしない。"""
    try:
        # librosa.get_duration は recent version ではファイルパスだけで動く。
        return float(librosa.get_duration(path=str(path)))
    except TypeError:
        # 古い librosa は filename= のみ対応。
        return float(librosa.get_duration(filename=str(path)))
    except Exception as exc:
        raise AnalysisError(f"Failed to probe audio duration: {exc}") from exc


def _load_segment(
    path: Path,
    *,
    sample_rate: int,
    offset_sec: float,
    duration_sec: Optional[float],
) -> tuple[np.ndarray, int]:
    """ファイルの一部(または全体)だけをモノラルで読み込む。"""
    try:
        # duration=None を渡すとファイル末尾まで読み込む。
        y, sr = librosa.load(
            str(path),
            sr=sample_rate,
            mono=True,
            offset=float(offset_sec),
            duration=duration_sec,
        )
    except Exception as exc:
        raise AnalysisError(f"Failed to load audio: {exc}") from exc
    return y, sr


def analyze_audio_file(
    file_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """音声/動画ファイルを 1秒刻みで解析し、AudioFrame のリストを返す。

    長尺(AUTO_CHUNK_THRESHOLD_SEC 超)は CHUNK_DURATION_SEC ずつに分割し、
    CHUNK_OVERLAP_SEC のオーバーラップを入れて順次処理する。

    Args:
        start_sec: 解析開始秒(None なら先頭)
        end_sec  : 解析終了秒(None ならファイル末尾)
    """
    path = Path(file_path)
    if not path.exists():
        raise AnalysisError(f"File not found: {file_path}")

    # Nyquist 整合性のチェック(pyin 呼び出し前に早期エラー化)。
    _resolve_pitch_range(sample_rate)

    logger.info("Probing duration: %s", path.name)
    _emit(progress_cb, f"Probing duration: {path.name}", 0.18)

    file_duration = _probe_duration(path)
    if file_duration <= 0.0:
        raise AnalysisError(f"Audio file has zero or unknown duration: {file_path}")

    # ── 解析対象の時間範囲を正規化 ──
    effective_start = max(0.0, float(start_sec) if start_sec is not None else 0.0)
    effective_end = (
        min(float(end_sec), file_duration) if end_sec is not None else file_duration
    )
    if effective_end <= effective_start:
        raise AnalysisError(
            f"Invalid time range: start_sec={effective_start} must be less than "
            f"end_sec={effective_end} (file duration: {file_duration:.1f}s)."
        )
    effective_duration = effective_end - effective_start

    _emit(
        progress_cb,
        (
            f"Analyzing {int(effective_duration // 60)}:"
            f"{int(effective_duration % 60):02d}"
            f" (range {int(effective_start)}s–{int(effective_end)}s)"
        ),
        0.22,
    )

    if effective_duration <= AUTO_CHUNK_THRESHOLD_SEC:
        # ── 単発パス(短尺)──
        logger.info(
            "Single-pass analysis (%.1fs <= threshold %ds)",
            effective_duration,
            AUTO_CHUNK_THRESHOLD_SEC,
        )
        y, sr = _load_segment(
            path,
            sample_rate=sample_rate,
            offset_sec=effective_start,
            duration_sec=effective_duration,
        )
        frames = _analyze_waveform(
            y,
            sr,
            hop_length=hop_length,
            frame_length=frame_length,
            time_offset_sec=int(effective_start),
            progress_cb=progress_cb,
            progress_span=(0.25, 0.97),
        )
        _recompute_rms_norm(frames)
        _emit(progress_cb, f"Analysis complete: {len(frames)} frames", 1.0)
        return frames

    # ── チャンク分割パス(長尺)──
    # dict[time_sec -> AudioFrame] で「先に登録されたもの勝ち」方式で結合。
    # 先行チャンクは自分の末尾付近を十分な past history つきで計算しているので、
    # その frames を正として採用し、後続チャンクの先頭 OVERLAP 秒は discard。
    merged: dict[int, AudioFrame] = {}

    stride = max(1, CHUNK_DURATION_SEC - CHUNK_OVERLAP_SEC)
    # 何チャンクできるか(進捗表示用、端数は切り上げ)。
    n_chunks = max(1, int(np.ceil((effective_duration - CHUNK_OVERLAP_SEC) / stride)))
    if n_chunks < 1:
        n_chunks = 1

    logger.info(
        "Chunked analysis: duration=%.1fs, chunks=%d (chunk=%ds, overlap=%ds)",
        effective_duration,
        n_chunks,
        CHUNK_DURATION_SEC,
        CHUNK_OVERLAP_SEC,
    )
    _emit(
        progress_cb,
        f"Long audio detected — splitting into {n_chunks} chunks of "
        f"{CHUNK_DURATION_SEC // 60} min (overlap {CHUNK_OVERLAP_SEC}s)",
        0.24,
    )

    # チャンクループの進捗は 0.25 → 0.97 に均等配分。
    p_start_global = 0.25
    p_end_global = 0.97

    chunk_idx = 0
    seg_start = effective_start
    while seg_start < effective_end:
        seg_duration = min(float(CHUNK_DURATION_SEC), effective_end - seg_start)
        chunk_label = f"{chunk_idx + 1}/{n_chunks}"

        # このチャンクに割り当てる進捗区間
        lo = p_start_global + (p_end_global - p_start_global) * (chunk_idx / max(n_chunks, 1))
        hi = p_start_global + (p_end_global - p_start_global) * ((chunk_idx + 1) / max(n_chunks, 1))

        _emit(
            progress_cb,
            f"Loading chunk {chunk_label} ({int(seg_start)}s–"
            f"{int(seg_start + seg_duration)}s)",
            lo,
        )
        y, sr = _load_segment(
            path,
            sample_rate=sample_rate,
            offset_sec=seg_start,
            duration_sec=seg_duration,
        )

        chunk_frames = _analyze_waveform(
            y,
            sr,
            hop_length=hop_length,
            frame_length=frame_length,
            time_offset_sec=int(seg_start),
            progress_cb=progress_cb,
            progress_span=(lo, hi),
            chunk_label=chunk_label,
        )

        # 「先勝ち」でマージ → オーバーラップ領域は前チャンクの値が残る。
        for f in chunk_frames:
            if f.time_sec not in merged:
                merged[f.time_sec] = f

        chunk_idx += 1
        if seg_start + seg_duration >= effective_end - 1e-6:
            break
        seg_start += stride

    # 並べ替えて返す。rms_norm は全体で再計算。
    sorted_frames = [merged[t] for t in sorted(merged.keys())]
    _recompute_rms_norm(sorted_frames)

    logger.info("Chunked analysis complete: %d frames from %d chunks", len(sorted_frames), chunk_idx)
    _emit(
        progress_cb,
        f"Analysis complete: {len(sorted_frames)} frames from {chunk_idx} chunks",
        1.0,
    )
    return sorted_frames


# ------------------------------------------------------------------
# 高レベルエントリーポイント(server.py から呼ばれる薄いラッパー)
# ------------------------------------------------------------------


def analyze_youtube(
    youtube_url: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """YouTubeから音声を落として解析する(ダウンロード + 解析をひとまとめに)。"""

    with tempfile.TemporaryDirectory(prefix="audio_analyzer_") as tmpdir:
        wav_path = download_youtube_audio(youtube_url, tmpdir, progress_cb=progress_cb)
        return analyze_audio_file(
            wav_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            frame_length=frame_length,
            start_sec=start_sec,
            end_sec=end_sec,
            progress_cb=progress_cb,
        )


def analyze_local(
    file_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """ローカルの音声/動画ファイルを解析する。"""
    return analyze_audio_file(
        file_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_length=frame_length,
        start_sec=start_sec,
        end_sec=end_sec,
        progress_cb=progress_cb,
    )
