"""Pydantic input models for MCP tool parameters.

Each model validates the arguments for one MCP tool so the server handler
receives already-sanitised input (stripped strings, bounds-checked ints,
lowercased format, etc.).
"""

from __future__ import annotations

# Pydantic は「入力データの型と制約をクラスで宣言 → 受け取った値を自動検証」してくれるライブラリ。
# 手書きで `if not isinstance(x, int): raise ...` を書かずに済むのが利点。
#   BaseModel       : 全モデルが継承する親クラス
#   ConfigDict      : モデル全体の設定(extra禁止、空白除去など)を入れる辞書
#   Field           : 各フィールドの詳細設定(デフォルト値、説明、上下限)
#   field_validator : 個別フィールドのカスタム検証デコレータ
#   model_validator : モデル全体(フィールド同士の関係)を検証するデコレータ
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `as 別名` でインポート時に名前を付け替え。
# constants.SAMPLE_RATE を DEFAULT_SAMPLE_RATE として扱い、
# 下のフィールドのデフォルト値として使う(定数であることを明示)。
from audio_analyzer_mcp.constants import (
    FRAME_LENGTH as DEFAULT_FRAME_LENGTH,
    HOP_LENGTH as DEFAULT_HOP_LENGTH,
    SAMPLE_RATE as DEFAULT_SAMPLE_RATE,
)


def _validate_frame_vs_hop(frame_length: int, hop_length: int) -> None:
    """Ensure frame_length is at least hop_length.

    librosa's internal resampling fails opaquely when frame_length < hop_length
    (e.g. ParameterError: Target size must be at least input size). Reject here
    with a clearer message.
    """
    # frame_length(窓の幅)が hop_length(ずらし幅)より小さいと、
    # librosa 内部のリサンプリングがわかりにくいエラーで落ちる。
    # ここで先に弾いて、ユーザーに親切なメッセージを返す。
    if frame_length < hop_length:
        # `raise` で例外を発生させる。Pydantic が拾って「入力エラー」として返してくれる。
        raise ValueError(
            f"frame_length ({frame_length}) must be >= hop_length ({hop_length}). "
            "Typical values: frame_length=4096, hop_length=2048."
        )


# ------------------------------------------------------------------
# YouTube音声解析ツールの入力モデル
# ------------------------------------------------------------------
#
# 書き方の流れ:
#   1. `class XxxInput(BaseModel):` で Pydanticモデルを宣言
#   2. `model_config` でモデル全体の挙動を設定
#   3. 各フィールドを `name: 型 = Field(...)` で宣言
#   4. 必要なら `@field_validator` / `@model_validator` で追加検証
#
# 型の書き方: `youtube_url: str` は「このフィールドは文字列型」の意。
class AnalyzeYouTubeAudioInput(BaseModel):
    """Input for analyzing audio from a YouTube video."""

    # モデル全体の設定。
    #   str_strip_whitespace=True : 文字列の前後空白を自動で除去
    #   validate_assignment=True  : 後から属性を書き換えたときも検証する
    #   extra="forbid"            : 未定義のフィールドが来たらエラー(タイプミス防止)
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # `Field(...)` の最初の `...`(Ellipsis)は「必須フィールド」の意味。
    # デフォルト値がない = 呼び出し元は必ず指定しないといけない。
    youtube_url: str = Field(
        ...,
        description=(
            "Public YouTube video URL. "
            "Formats: https://www.youtube.com/watch?v=XXXXX or https://youtu.be/XXXXX"
        ),
        min_length=10,   # 10文字未満はURLとしてあり得ない
        max_length=500,  # 500文字超は明らかにおかしい(攻撃対策も兼ねる)
    )
    format: str = Field(
        default="csv",  # 省略時は "csv"
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
        # ge = "greater than or equal" (以上), le = "less than or equal" (以下)
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

    # `@field_validator("format")` は「formatフィールドの検証関数です」という印(デコレータ)。
    # `@classmethod` はインスタンスではなくクラスそのものを第1引数に受け取る関数の印。
    # Pydantic の慣例で、field_validator は classmethod として定義する。
    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        # 入力を小文字化・前後空白除去してから比較。"JSON" や "Csv" も受け入れる。
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        # 返した値がそのままフィールドに入る(正規化後の値を保存できる)。
        return v.lower().strip()

    # model_validator(mode="after"): 全フィールドの型検証が終わった後に走る。
    # フィールド同士の関係を検証するときに使う(ここでは frame と hop の関係)。
    @model_validator(mode="after")
    def validate_frame_hop(self) -> "AnalyzeYouTubeAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        # after mode では自分自身を return する約束。
        return self


# ------------------------------------------------------------------
# ローカルファイル解析ツールの入力モデル
# ------------------------------------------------------------------
# 構造はほぼ YouTube用と同じ。違いは youtube_url → file_path のみ。
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


# ------------------------------------------------------------------
# ハイライト検出ツールの入力モデル
# ------------------------------------------------------------------
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
    # ハイライト固有のパラメータ ↓
    top_n: int = Field(
        default=20,  # 省略時は上位20件返す
        description=(
            "Maximum number of highlight moments to return. Default: 20. "
            "Highlights are ranked by volume spike intensity."
        ),
        ge=1,
        le=100,
    )
    min_gap_sec: int = Field(
        default=10,  # 近接したスパイクを1件にまとめるための最小間隔(秒)
        description=(
            "Minimum gap in seconds between reported highlights. "
            "Prevents clustering of nearby spikes. Default: 10"
        ),
        ge=1,
        le=300,
    )
    # 以下は他モデルと共通の解析パラメータ
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
