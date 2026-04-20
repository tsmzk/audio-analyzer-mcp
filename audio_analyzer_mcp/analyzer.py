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
    """音声/動画ファイルを1秒単位で解析し、AudioFrame のリストを返す。

    処理の流れ:
      1. ファイルをWAVとして読み込む(librosaが音声波形ndarrayに変換)
      2. 高速な特徴量(RMS, スペクトル重心, ZCR)を計算
      3. ピッチ(f0)を計算(pyin — 遅いが精度が高い)
      4. 1秒ごとに集約 + 発話判定 + スパイク判定
      5. AudioFrame のリストとして返す
    """
    # Pathオブジェクトで扱うとOS依存の区切り文字や存在チェックが楽。
    path = Path(file_path)
    # ファイルがなければ先にわかりやすい例外を投げる。
    # (librosa に渡すと内部で難解なエラーになるのを防ぐ)
    if not path.exists():
        raise AnalysisError(f"File not found: {file_path}")

    logger.info("Loading audio: %s", path.name)
    _emit(progress_cb, f"Loading audio: {path.name}", 0.20)

    try:
        # librosa.load: 音声ファイル → (波形データ, サンプリングレート)
        #   y : numpy配列(各サンプルの音の振幅値が並んでいる)
        #   sr: 実際のサンプリングレート(sample_rate指定通りになる)
        #   mono=True: ステレオでもモノラル(1ch)に混ぜる
        y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as exc:
        # `raise ... from exc` で元例外を保持しつつ新しい例外に包む。
        # (デバッグ時、両方のトレースバックが残って便利)
        raise AnalysisError(f"Failed to load audio: {exc}") from exc

    # 音声の長さ(秒)。波形の長さ÷サンプリングレート、をライブラリがやってくれる。
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info("Duration: %d:%02d (%d seconds)", int(duration // 60), int(duration % 60), int(duration))
    _emit(
        progress_cb,
        f"Duration: {int(duration // 60)}:{int(duration % 60):02d}",
        0.25,
    )

    # ===== 特徴量抽出(高速フェーズ) =====
    _emit(progress_cb, "Extracting features (RMS, spectral centroid, ZCR)...", 0.30)

    # ── RMS(音量)──
    # RMS = Root Mean Square。波形の「平均的な強さ」を表す指標。
    # librosaは「フレーム(短い区間)ごと」にRMSを計算して配列で返す。
    # [0] を付けているのは「チャンネル0」を取り出すため(モノラルだが一応)。
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # 各RMS値が「何秒時点の値か」を計算(これで時刻と対応づけられる)。
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    # 振幅の生値 → dB値 に変換(ref=np.max で最大値を0dBの基準にする)。
    # dBはログスケールなので人間の「音の大きさ感覚」に近い。
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # ── スペクトル重心(声の鋭さ)──
    # 周波数スペクトルの「重心(中央)」。高いほど高音成分が多い=鋭い声。
    spectral_centroids = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    # ── ゼロクロッシング率(ZCR)──
    # 波形が 0 を何回横切るか。声は適度(0.01〜0.3)、ノイズは高い、無音は低い傾向。
    # 発話検出に使う簡易指標。
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]

    # ===== ピッチ推定(低速フェーズ) =====
    # pyin: 基本周波数(=ピッチ)を高精度で推定するアルゴリズム。
    # 全体の処理時間の大半はここ。そのため進捗を細かく通知する。
    logger.info("Estimating pitch...")
    _emit(progress_cb, "Estimating pitch (slowest phase)...", 0.40)

    # タプル(複数戻り値)の展開代入。
    # 先頭のアンダースコア `_voiced_flag` は「受け取るけど使わない変数」の慣例。
    # fmin/fmax: 探索するピッチ範囲。C2 ≈ 65Hz, C7 ≈ 2093Hz(人の声の範囲をカバー)。
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    _emit(progress_cb, "Pitch estimation complete", 0.80)

    # ===== 1秒ごとに集約 =====
    # librosaの結果は「フレーム単位」(このプロジェクト設定だと秒4コマ程度)。
    # これを「1秒1行」の AudioFrame にまとめ直す。
    total_seconds = int(duration)
    results: list[AudioFrame] = []

    # ── 音量正規化の基準値を決める ──
    # dB値のうち、ほぼ無音(-80dB以下)は除いて有効範囲だけ見る。
    # さらに上下1%を捨てて(np.percentileで)外れ値に強い min/max を取る。
    rms_db_valid = rms_db[rms_db > -80.0]
    if len(rms_db_valid) > 0:
        rms_db_min = float(np.percentile(rms_db_valid, 1))
        rms_db_max = float(np.percentile(rms_db_valid, 99))
    else:
        # 全部無音なら適当な範囲に(割り算でゼロ除算しないためのフォールバック)。
        rms_db_min, rms_db_max = -60.0, 0.0
    # 分母が0にならないよう下限1.0を確保。
    rms_db_range = max(rms_db_max - rms_db_min, 1.0)

    logger.info("Processing %d seconds...", total_seconds)
    _emit(progress_cb, f"Aggregating per-second frames ({total_seconds}s)...", 0.82)

    # 進捗通知は総秒数の約10%刻みで出す(通知を出しすぎないため)。
    # `max(x, 1)` で0にならないようガード(1秒未満の音でクラッシュしないように)。
    next_report_sec = max(total_seconds // 10, 1)

    # ── メインループ: 1秒ごとに集約 ──
    for sec in range(total_seconds):
        t_start = float(sec)
        t_end = float(sec + 1)

        # ── この1秒に該当するフレームの「マスク」を作る ──
        # numpyのブール配列(Trueのところだけ取り出す使い方)を作る。
        # `(条件1) & (条件2)` で各要素について「両方True」の配列ができる。
        rms_mask = (rms_times >= t_start) & (rms_times < t_end)

        # RMS(dB)の平均値を取る。
        # `array[mask]` でTrue位置の要素だけ取り出せる(numpyの便利機能)。
        sec_rms = rms_db[rms_mask]
        # np.nanmean: NaN(欠損値)を無視して平均。len()==0 の時は無音扱い。
        avg_rms_db = float(np.nanmean(sec_rms)) if len(sec_rms) > 0 else -80.0

        # 0〜100の正規化音量: (値 - 最小) / 範囲 * 100
        # max/min でクランプして範囲外(負/100超)にならないように。
        rms_norm = max(0.0, min(100.0, (avg_rms_db - rms_db_min) / rms_db_range * 100.0))

        # ── ピッチ(Hz)の平均 ──
        pitch_mask = (pitch_times >= t_start) & (pitch_times < t_end)
        sec_pitch = f0[pitch_mask]
        # pyinは発声がない部分を NaN にする。~np.isnan(...) で「NaNでない」位置だけ残す。
        # `~` はビット反転(numpy配列では要素ごとに True/False を反転)。
        valid_pitch = sec_pitch[~np.isnan(sec_pitch)] if len(sec_pitch) > 0 else np.array([])
        avg_pitch = float(np.nanmean(valid_pitch)) if len(valid_pitch) > 0 else 0.0

        # ── スペクトル重心の平均 ──
        sec_centroid = spectral_centroids[rms_mask]
        avg_centroid = float(np.nanmean(sec_centroid)) if len(sec_centroid) > 0 else 0.0

        # ── 発話判定(シンプルなヒューリスティック)──
        # 3つの条件のANDで「発話らしい」と判定:
        #   1. 音量が無音しきい値(-50dB)より大きい
        #   2. ZCRが 0.01〜0.30(音声の典型レンジ。ノイズは高すぎ、無音は低すぎ)
        #   3. スペクトル重心が 300〜5000Hz(人の声の特性)
        sec_zcr = zcr[rms_mask]
        avg_zcr = float(np.nanmean(sec_zcr)) if len(sec_zcr) > 0 else 0.0
        is_speech = (
            avg_rms_db > SILENCE_THRESHOLD_DB
            and 0.01 < avg_zcr < 0.30
            and 300.0 < avg_centroid < 5000.0
        )

        # ── 音量スパイク判定 ──
        # 直近 SPIKE_LOOKBACK_SEC 秒の平均より SPIKE_THRESHOLD_DB 以上大きければスパイク。
        # 開始直後(過去データがない)は判定しない。
        volume_spike = False
        if sec >= SPIKE_LOOKBACK_SEC and len(results) >= SPIKE_LOOKBACK_SEC:
            # results[-N:] は「末尾N要素」のリスト(スライス)。
            prev_avg = np.mean([r.rms_db for r in results[-SPIKE_LOOKBACK_SEC:]])
            if avg_rms_db - prev_avg > SPIKE_THRESHOLD_DB:
                volume_spike = True

        # 1秒分の結果を AudioFrame にまとめて追加。
        # round(値, 1) で小数第1位までに丸める(CSV/JSONを読みやすくするため)。
        results.append(AudioFrame(
            time_sec=sec,
            rms_db=round(avg_rms_db, 1),
            rms_norm=round(rms_norm, 1),
            pitch_hz=round(avg_pitch, 1),
            spectral_centroid=round(avg_centroid, 1),
            is_speech=is_speech,
            volume_spike=volume_spike,
        ))

        # ~10%ごとに進捗を出す。
        # `sec and sec % next_report_sec == 0` で「0秒目は出さない、次は倍数のとき出す」を実現。
        if sec and sec % next_report_sec == 0 and total_seconds > 0:
            # 進捗バー位置: 82%(集約開始)から最大97%までリニアに進める。
            frac = 0.82 + 0.15 * (sec / total_seconds)
            _emit(progress_cb, f"Aggregated {sec}/{total_seconds}s", frac)

    logger.info("Analysis complete: %d frames", len(results))
    _emit(progress_cb, f"Analysis complete: {len(results)} frames", 1.0)
    return results


# ------------------------------------------------------------------
# 高レベルエントリーポイント(server.py から呼ばれる薄いラッパー)
# ------------------------------------------------------------------


def analyze_youtube(
    youtube_url: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """YouTubeから音声を落として解析する(ダウンロード + 解析をひとまとめに)。"""

    # `with tempfile.TemporaryDirectory() as tmpdir:` はコンテキストマネージャ構文。
    # ブロックを抜けた瞬間に tmpdir が自動削除される(例外発生時も同様)。
    # 「一時ファイルの後始末忘れ」を防ぐPythonのお決まりパターン。
    with tempfile.TemporaryDirectory(prefix="audio_analyzer_") as tmpdir:
        wav_path = download_youtube_audio(youtube_url, tmpdir, progress_cb=progress_cb)
        return analyze_audio_file(
            wav_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
            frame_length=frame_length,
            progress_cb=progress_cb,
        )
    # with ブロックを抜けたこの時点で tmpdir(とその中のWAV)は消えている。


def analyze_local(
    file_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    frame_length: int = FRAME_LENGTH,
    progress_cb: Optional[ProgressCallback] = None,
) -> list[AudioFrame]:
    """ローカルの音声/動画ファイルを解析する(単に analyze_audio_file を呼び直すだけ)。

    一見不要に見えるが、
      - 関数名を「YouTube版と対になる」ようにして API を揃える
      - 将来ローカル特有の前処理(例: 動画→音声抽出)を足したい時に拡張しやすい
    という意図で残している。
    """
    return analyze_audio_file(
        file_path,
        sample_rate=sample_rate,
        hop_length=hop_length,
        frame_length=frame_length,
        progress_cb=progress_cb,
    )
