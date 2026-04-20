"""Shared constants, exceptions, and the AudioFrame data structure.

Pulled out so every module (downloader, analyzer, models, formatters,
highlights, server) can depend on a single, stable source of truth.
"""

# `from __future__ import annotations` は「型ヒントを文字列として評価する」宣言。
# これを書いておくと、`dict[str, Any]` のような新しい書き方が古いPythonでも動く。
# 実用上は「型ヒントはコメント扱い」と考えてOK（実行時には無視される）。
from __future__ import annotations

# `typing.Any` は「どんな型でもよい」を表す型ヒント。to_dict() の戻り値の値の型に使う。
from typing import Any

# ------------------------------------------------------------------
# yt-dlp ダウンロード設定
# ------------------------------------------------------------------

# yt-dlp のダウンロードがこの秒数を超えたら打ち切る。
# 長い動画（数十分〜1時間）でも終わるよう、10分の余裕を持たせている。
DOWNLOAD_TIMEOUT_SEC = 600  # 10 min max for long videos

# ------------------------------------------------------------------
# librosa 解析設定
# ------------------------------------------------------------------

# ── 音声解析の基本用語 ──
# sample_rate: 1秒間に音を何回サンプリング(数値化)するか。単位はHz。
#   CD音質は44100Hz。高いほど高音まで正確だが、処理時間・メモリが増える。
# hop_length:  解析の「1コマ」を何サンプルずつずらすか。小さいほど細かい(=遅い)。
# frame_length:解析の「1コマ」が何サンプル分を見るか(=1回に見る窓の幅)。

SAMPLE_RATE = 8000        # 人の声のピッチ範囲(65〜2000Hz)には8000Hzで十分。速度優先。
HOP_LENGTH = 2048         # 8000Hzで約256ms/コマ → 1秒あたり約4コマ。
FRAME_LENGTH = 4096       # 約512msの窓で音を観測する。

# ------------------------------------------------------------------
# しきい値(判定基準)
# ------------------------------------------------------------------

# RMS(音量)の単位 dB はログスケール。0dBが最大、負に大きくなるほど小さい音。
SILENCE_THRESHOLD_DB = -50.0    # これ未満は「無音」とみなす(発話ではない)
SPIKE_THRESHOLD_DB = 10.0       # 直近からこのdB以上跳ねたら「音量スパイク」扱い
SPIKE_LOOKBACK_SEC = 3          # スパイク判定で「直近何秒」と比較するか


# ------------------------------------------------------------------
# 例外クラス
# ------------------------------------------------------------------
#
# Pythonではエラー(例外)も「クラス」として定義する。
# 独自の例外クラスを作ることで、呼び出し側が「ダウンロード失敗か解析失敗か」
# を `except DownloadError:` のように区別できる。
# `class Child(Parent):` は「ParentクラスのすべてをChildが引き継ぐ」の意味。
# `pass` は「中身は何もない」を表す。例外クラスは中身なしで十分。


class AudioAnalysisError(Exception):
    """音声解析関連のすべての例外のベース(親)クラス。

    `Exception` を継承することで、Pythonが扱う通常の例外と同じ振る舞いになる。
    このクラスを直接 raise することは少なく、下の2つを使う。
    """


class DownloadError(AudioAnalysisError):
    """YouTubeダウンロードに失敗したときに投げる例外。"""


class AnalysisError(AudioAnalysisError):
    """librosa での解析に失敗したときに投げる例外。"""


# ------------------------------------------------------------------
# データ構造
# ------------------------------------------------------------------


class AudioFrame:
    """音声1秒分の解析結果を保持するクラス。

    1秒ごとに AudioFrame を1つ作り、それをリストにして返す設計。
    10分の動画なら 600個の AudioFrame ができる。
    """

    # `__slots__` は「このクラスで使う属性名を固定する」宣言。
    # 普通のPythonクラスは属性を動的に追加できるが、__slots__ を指定すると
    # メモリ消費が減り、属性アクセスも少し速くなる。長い動画で数千個作るので効く。
    __slots__ = (
        "time_sec", "timestamp", "rms_db", "rms_norm",
        "pitch_hz", "spectral_centroid", "is_speech", "volume_spike",
    )

    # `__init__` は「コンストラクタ」。`AudioFrame(time_sec=0, ...)` と呼ぶとこれが走る。
    # 第1引数 `self` はこのインスタンス自身を指す。他言語の `this` に相当。
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
        # f-string: `f"..."` の中で `{変数}` と書くと変数の値が埋め込まれる。
        # `:02d` は「2桁のゼロ埋め10進整数」。例: 5 → "05"、120//60=2 → "02"。
        self.timestamp = f"{time_sec // 60:02d}:{time_sec % 60:02d}"
        self.rms_db = rms_db              # 音量(dB) 生値
        self.rms_norm = rms_norm          # 音量を0〜100に正規化した値(扱いやすい)
        self.pitch_hz = pitch_hz          # 声の高さ(Hz)。男性声 ~120Hz、女性声 ~220Hz
        self.spectral_centroid = spectral_centroid  # 声の鋭さ(高音成分が多いと大きい)
        self.is_speech = is_speech        # この1秒が「発話」かどうかのbool
        self.volume_spike = volume_spike  # 急な音量アップ(盛り上がり候補)

    def to_dict(self) -> dict[str, Any]:
        """この AudioFrame を辞書(dict)に変換する。

        CSV/JSON出力のとき、辞書のほうがライブラリに渡しやすいので用意している。
        """
        # 辞書リテラル `{"key": value, ...}` を返すだけ。
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
