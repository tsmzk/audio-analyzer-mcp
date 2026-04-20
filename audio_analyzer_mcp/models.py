"""Pydantic input models for MCP tool parameters.

Each model validates the arguments for one MCP tool so the server handler
receives already-sanitised input (stripped strings, bounds-checked ints,
lowercased format, etc.).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    if frame_length < hop_length:
        raise ValueError(
            f"frame_length ({frame_length}) must be >= hop_length ({hop_length}). "
            "Typical values: frame_length=4096, hop_length=2048."
        )


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
            f"Default: {DEFAULT_SAMPLE_RATE}. Use 22050 for max quality, 8000 for speed."
        ),
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

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v.lower().strip() not in ("csv", "json"):
            raise ValueError("format must be 'csv' or 'json'")
        return v.lower().strip()

    @model_validator(mode="after")
    def validate_frame_hop(self) -> "AnalyzeYouTubeAudioInput":
        _validate_frame_vs_hop(self.frame_length, self.hop_length)
        return self


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
