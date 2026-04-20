"""Pydantic input models for MCP tool parameters.

Each model validates the arguments for one MCP tool so the server handler
receives already-sanitised input (stripped strings, bounds-checked ints,
lowercased format, etc.).
"""

from __future__ import annotations

from typing import Optional

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
    NYQUIST_SAFETY,
    PITCH_FMAX_HZ,
    SAMPLE_RATE as DEFAULT_SAMPLE_RATE,
)

# ピッチ探索(pyin)で fmax=PITCH_FMAX_HZ を確保するための最小サンプルレート。
# Nyquist 余裕 NYQUIST_SAFETY を掛けて、切り上げ整数に。
# 例: 2093Hz * 2.1 ≈ 4395 → ge=4400 を採用。
MIN_SAMPLE_RATE = int(PITCH_FMAX_HZ * NYQUIST_SAFETY) + 5
MIN_SAMPLE_RATE = ((MIN_SAMPLE_RATE + 99) // 100) * 100  # 100Hz 単位に丸め


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


def _validate_time_range(start_sec: Optional[float], end_sec: Optional[float]) -> None:
    """start_sec / end_sec が両方指定された場合の前後関係をチェック。"""
    if start_sec is not None and start_sec < 0:
        raise ValueError(f"start_sec ({start_sec}) must be >= 0.")
    if end_sec is not None and end_sec <= 0:
        raise ValueError(f"end_sec ({end_sec}) must be > 0.")
    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        raise ValueError(
            f"end_sec ({end_sec}) must be greater than start_sec ({start_sec})."
        )


# ------------------------------------------------------------------
# YouTube音声解析ツールの入力モデル
# ------------------------------------------------------------------
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
            f"Default: {DEFAULT_SAMPLE_RATE}. "
            f"Minimum {MIN_SAMPLE_RATE}Hz is required so the Nyquist frequency "
            f"stays above the pitch detector's fmax ({int(PITCH_FMAX_HZ)}Hz). "
            "Use 22050 for max quality, 8000 for speed."
        ),
        ge=MIN_SAMPLE_RATE,
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
    start_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. Start of the analysis range in seconds. "
            "If omitted, analysis starts from the beginning. "
            "Combine with end_sec to analyze a specific segment of a long video."
        ),
        ge=0.0,
    )
    end_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. End of the analysis range in seconds. "
            "If omitted, analysis goes through the end of the file."
        ),
        gt=0.0,
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "AnalyzeYouTubeAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        _validate_time_range(self.start_sec, self.end_sec)
        return self


# ------------------------------------------------------------------
# ローカルファイル解析ツールの入力モデル
# ------------------------------------------------------------------
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
        description=(
            f"Sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}. "
            f"Minimum {MIN_SAMPLE_RATE}Hz (Nyquist >= pitch fmax)."
        ),
        ge=MIN_SAMPLE_RATE, le=44100,
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
    start_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. Start of the analysis range in seconds. Default: from beginning."
        ),
        ge=0.0,
    )
    end_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. End of the analysis range in seconds. Default: to end of file."
        ),
        gt=0.0,
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "AnalyzeLocalAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        _validate_time_range(self.start_sec, self.end_sec)
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
        description=(
            f"Sample rate in Hz. Default: {DEFAULT_SAMPLE_RATE}. "
            f"Minimum {MIN_SAMPLE_RATE}Hz (Nyquist >= pitch fmax)."
        ),
        ge=MIN_SAMPLE_RATE, le=44100,
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
    start_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. Start of the analysis range in seconds. Default: from beginning."
        ),
        ge=0.0,
    )
    end_sec: Optional[float] = Field(
        default=None,
        description=(
            "Optional. End of the analysis range in seconds. Default: to end of file."
        ),
        gt=0.0,
    )

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "DetectHighlightsInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        _validate_time_range(self.start_sec, self.end_sec)
        return self
