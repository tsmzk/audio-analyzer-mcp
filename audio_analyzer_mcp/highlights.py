"""Highlight-moment detection from analyzed audio frames.

Ranks moments by a combination of volume spikes, overall loudness, high
pitch (excitement), and spectral sharpness (ツッコミ-like exclamations),
then deduplicates nearby picks via a minimum time gap.
"""

from __future__ import annotations

# numpy: 数値計算の標準ライブラリ。`np` と略してインポートするのが慣例。
# ここではパーセンタイル計算(上位10%のdB値は?)に使う。
import numpy as np

from audio_analyzer_mcp.constants import AudioFrame


def _detect_highlight_moments(
    frames: list[AudioFrame],
    top_n: int = 20,      # 返すハイライトの最大件数
    min_gap_sec: int = 10,  # ハイライト同士が最低何秒離れていないといけないか
) -> list[dict]:
    """解析済みフレーム列から「盛り上がりそうな瞬間」を抽出する。

    仕組み:
      1. 各フレームに対して「スコア」を計算(音量/ピッチ/鋭さで加点)
      2. スコア降順で並べる
      3. 近接したものを間引く(min_gap_sec で距離を確保)
      4. 上位 top_n 件を返す
    """
    # candidates は「各フレームの採点結果」を貯めるリスト。
    # 型ヒント `list[dict]` は「辞書が入るリスト」の意。
    candidates: list[dict] = []

    # ── 1. 発話フレームだけ抽出 ──
    # リスト内包表記 + if条件。`for f in frames if f.is_speech` は
    # 「is_speech が True のフレームだけ拾う」の短縮形。
    speech_frames = [f for f in frames if f.is_speech]
    # 発話がないならハイライトは出せない。空リストを即返す(early return)。
    if not speech_frames:
        return []

    # ── 2. スコア基準となる統計値を計算 ──
    # パーセンタイル: 値を昇順に並べたときの「下からN%の位置の値」。
    # 例: rms_p90 = 「発話dB値の下から90%の位置の値」 = 上位10%のライン。
    rms_values = [f.rms_db for f in speech_frames]
    # ピッチが取れなかった(0のまま)フレームは統計から除外。
    pitch_values = [f.pitch_hz for f in speech_frames if f.pitch_hz > 0]

    rms_p90 = float(np.percentile(rms_values, 90))  # 上位10%の音量ライン
    rms_p95 = float(np.percentile(rms_values, 95))  # 上位5%の音量ライン

    # ピッチ値がない場合のフォールバック値 200Hz(人の平均的な声の高さ)。
    # `A if 条件 else B` は三項演算子(条件分岐式)。
    pitch_p90 = float(np.percentile(pitch_values, 90)) if pitch_values else 200.0

    # ── 3. 各フレームを採点 ──
    for frame in frames:
        # 非発話フレームは採点対象外(BGMだけで盛り上げ判定しないため)。
        # `continue` は「このループの残りをスキップして次の繰り返しへ」。
        if not frame.is_speech:
            continue

        score = 0.0
        reasons = []  # 「なぜスコアを上げたか」を人間に説明するための理由リスト

        # (a) 音量スパイク(最強のシグナル) = +50点
        if frame.volume_spike:
            score += 50.0
            reasons.append("volume_spike")

        # (b) 大音量 = +30点、やや大音量 = +15点
        if frame.rms_db >= rms_p95:
            score += 30.0
            reasons.append("very_loud")
        elif frame.rms_db >= rms_p90:
            # `elif` は else if。上の条件が偽のときだけ評価される。
            score += 15.0
            reasons.append("loud")

        # (c) 高いピッチ(興奮している声) = +20点
        if frame.pitch_hz >= pitch_p90 and frame.pitch_hz > 0:
            score += 20.0
            reasons.append("high_pitch")

        # (d) 声の鋭さ(スペクトル重心が高い = ツッコミ・叫び) = +10点
        #
        # NOTE: centroid_p90 の計算がループ内にあるのは少し無駄
        # (毎フレーム同じ値を再計算している)。
        # 改善の余地だが、動作には影響しないので今は放置。
        centroid_values = [f.spectral_centroid for f in speech_frames if f.spectral_centroid > 0]
        if centroid_values:
            centroid_p90 = float(np.percentile(centroid_values, 90))
            if frame.spectral_centroid >= centroid_p90:
                score += 10.0
                reasons.append("sharp_voice")

        # 0点のフレームは候補に入れない(該当理由なし)。
        if score > 0:
            # `candidates.append({...})` で辞書を1件追加。
            candidates.append({
                "timestamp": frame.timestamp,
                "time_sec": frame.time_sec,
                "score": round(score, 1),  # round(値, 桁数) で小数点以下を丸める
                "rms_db": frame.rms_db,
                "rms_norm": frame.rms_norm,
                "pitch_hz": frame.pitch_hz,
                "reasons": reasons,
            })

    # ── 4. スコア降順でソート ──
    # `list.sort(key=関数)` で、各要素を関数に通した値で並べ替える。
    # `lambda x: x["score"]` は「辞書 x を受け取って x["score"] を返す」無名関数。
    # `reverse=True` で降順(大きい順)。
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # ── 5. 近接した候補を間引き、上位 top_n 件を選ぶ ──
    selected: list[dict] = []
    used_times: set[int] = set()  # 既に選んだ瞬間の時刻(秒)集合

    for c in candidates:
        t = c["time_sec"]
        # ここが「近接スパイクの除外」ロジック。
        # `any(...)` は「1つでも True があれば True」の関数。
        # 既選択の時刻との差が min_gap_sec 未満なら、この候補は捨てる。
        if any(abs(t - ut) < min_gap_sec for ut in used_times):
            continue

        selected.append(c)
        used_times.add(t)

        # 件数が上限に達したら打ち切り。
        if len(selected) >= top_n:
            break

    return selected
