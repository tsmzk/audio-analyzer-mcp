"""Audio Analyzer MCP Server.

Provides tools for extracting audio features (volume, pitch, speech)
from YouTube videos or local audio/video files using librosa.

Tools:
    analyze_youtube_audio   — Download and analyze audio from a YouTube URL
    analyze_local_audio     — Analyze audio from a local file
    detect_highlights       — Find volume spikes / high-energy moments from a YouTube URL
    detect_highlights_local — Same ranking, but read from a local audio/video file
                              (workaround when YouTube DL is blocked by bot detection)
"""

# ------------------------------------------------------------------
# このファイルの役割
# ------------------------------------------------------------------
# MCP(Model Context Protocol)は「AIアシスタント(Claudeなど)に外部ツールを
# 提供する仕組み」。このファイルは以下を担当する:
#   1. MCPサーバーのインスタンス作成
#   2. 3つのツール(MCPクライアントから呼べる関数)の定義
#   3. エントリーポイント(サーバー起動)
#
# 具体的な処理(ダウンロード、解析、ハイライト検出、進捗通知…)は
# 他のモジュールに委譲し、ここでは「ツールの外形」だけ定義する。
# ------------------------------------------------------------------

from __future__ import annotations

import json
import logging
# sys.stderr: 標準エラー出力。MCPはstdoutを通信に使うので、ログはstderrに出す。
import sys
from typing import Callable

# FastMCP: MCPサーバーを簡単に作るためのフレームワーク。
# Context: 各ツール呼び出しごとのコンテキスト(進捗報告・ログ送信などの手段を持つ)。
from mcp.server.fastmcp import Context, FastMCP

from audio_analyzer_mcp.analyzer import analyze_local, analyze_youtube
from audio_analyzer_mcp.formatters import _format_error, _format_frames
from audio_analyzer_mcp.highlights import build_highlight_summary
from audio_analyzer_mcp.models import (
    AnalyzeLocalAudioInput,
    AnalyzeYouTubeAudioInput,
    DetectHighlightsInput,
    DetectHighlightsLocalInput,
)
from audio_analyzer_mcp.progress import _run_with_heartbeat

# ------------------------------------------------------------------
# ロギング設定
# ------------------------------------------------------------------
# basicConfig はプロセス全体のログ設定。
#   level=INFO        : INFO以上のログを出す(DEBUGは出さない)
#   format=...        : ログの見た目フォーマット
#   stream=sys.stderr : 標準エラーに出す(stdoutはMCP通信に使うため避ける)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("audio_analyzer_mcp")

# ------------------------------------------------------------------
# サーバーインスタンス
# ------------------------------------------------------------------

# FastMCP("名前") でサーバーを1つ作る。
# この `mcp` オブジェクトを通して、下で @mcp.tool デコレータでツールを登録する。
mcp = FastMCP("audio_analyzer_mcp")


# ------------------------------------------------------------------
# ツール定義
# ------------------------------------------------------------------
#
# @mcp.tool(...) デコレータ:
#   関数を「MCPツールとして登録」する印。
#   annotations はツールのメタ情報で、
#     readOnlyHint   : 読み取り専用(副作用なし)
#     destructiveHint: 破壊的操作か(削除等)
#     idempotentHint : 冪等か(同じ入力→同じ出力、副作用同じ)
#     openWorldHint  : 外部世界(インターネット等)にアクセスするか
#
# 関数シグネチャ(params, ctx):
#   params : Pydanticモデル(models.py で定義)。入力は自動バリデーションされる
#   ctx    : Context。進捗報告・ログ送信に使う
#   戻り値 : 文字列(CSV や JSON)。MCPクライアントに返される


@mcp.tool(
    name="analyze_youtube_audio",
    annotations={
        "title": "Analyze YouTube Audio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,  # YouTubeにアクセスするのでTrue
    },
)
async def analyze_youtube_audio(
    params: AnalyzeYouTubeAudioInput,
    ctx: Context,
) -> str:
    """Analyze audio from a YouTube video.

    Downloads the audio track and extracts per-second features:
    - rms_db: Volume in decibels (higher = louder)
    - rms_norm: Volume normalized 0-100
    - pitch_hz: Voice pitch in Hz (higher = more excited)
    - spectral_centroid: Voice sharpness (higher = sharper)
    - is_speech: Whether someone is speaking
    - volume_spike: Sudden volume increase (highlight candidate)

    Output is CSV or JSON, one row per second. Use time_sec to match
    with subtitle data from youtube-transcript MCP.

    Args:
        params: youtube_url and output format.

    Returns:
        CSV or JSON string with per-second audio analysis.
    """
    try:
        # 内側関数 _runner を定義して _run_with_heartbeat に渡すパターン。
        # _runner は「進捗コールバック cb を受け取って analyze_youtube を呼ぶ」だけの関数。
        # こうすることで、別スレッドから進捗通知を流せる(progress.py 参照)。
        def _runner(cb: Callable[[str, float], None]):
            return analyze_youtube(
                params.youtube_url,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                start_sec=params.start_sec,
                end_sec=params.end_sec,
                progress_cb=cb,
            )

        # `await` で重い処理の完了を待つ(裏ではハートビートが流れている)。
        frames = await _run_with_heartbeat(ctx, _runner)
        # 結果を CSV/JSON 文字列に整形して返す。
        return _format_frames(frames, params.format)
    except Exception as exc:
        # どの例外でもユーザー向けメッセージに変換して返す。
        # ここで「ログには詳細を、ユーザーには親切な文章を」と役割分担している。
        logger.error("analyze_youtube_audio failed: %s", exc, exc_info=True)
        return _format_error(exc)


@mcp.tool(
    name="analyze_local_audio",
    annotations={
        "title": "Analyze Local Audio File",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def analyze_local_audio_tool(
    params: AnalyzeLocalAudioInput,
    ctx: Context,
) -> str:
    """Analyze audio from a local file.

    Supports audio files (wav, mp3, flac, ogg) and video files
    (mp4, mkv, avi, mov, webm) — the audio track is automatically extracted.

    Output is the same per-second CSV/JSON as analyze_youtube_audio.

    Args:
        params: file_path and output format.

    Returns:
        CSV or JSON string with per-second audio analysis.
    """
    # ↑の analyze_youtube_audio とほぼ同じ構造。違いは analyze_local を呼ぶだけ。
    try:
        def _runner(cb: Callable[[str, float], None]):
            return analyze_local(
                params.file_path,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                start_sec=params.start_sec,
                end_sec=params.end_sec,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)
        return _format_frames(frames, params.format)
    except Exception as exc:
        logger.error("analyze_local_audio failed: %s", exc, exc_info=True)
        return _format_error(exc)


@mcp.tool(
    name="detect_highlights",
    annotations={
        "title": "Detect Audio Highlights",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def detect_highlights(
    params: DetectHighlightsInput,
    ctx: Context,
) -> str:
    """Detect highlight moments from a YouTube video's audio.

    Analyzes the full audio track and returns the most notable moments,
    ranked by a combination of:
    - Volume spikes (sudden jumps in loudness)
    - Overall loudness (top percentile)
    - High pitch (excitement indicator)
    - Voice sharpness (e.g. ツッコミ, exclamations)

    Each highlight includes a score, timestamp, and reasons.
    Use with subtitle data to see what was said at each moment.

    Args:
        params: youtube_url, top_n (max results), min_gap_sec (minimum gap between results).

    Returns:
        JSON array of highlight moments with scores.
    """
    try:
        # 1. まず解析 → 2. ハイライト抽出 → 3. サマリーと一緒にJSONで返す、という流れ。
        def _runner(cb: Callable[[str, float], None]):
            return analyze_youtube(
                params.youtube_url,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                start_sec=params.start_sec,
                end_sec=params.end_sec,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)

        summary = build_highlight_summary(
            frames,
            top_n=params.top_n,
            min_gap_sec=params.min_gap_sec,
        )
        # indent=2 で人間が読みやすく整形。ensure_ascii=False で日本語もそのまま出力。
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("detect_highlights failed: %s", exc, exc_info=True)
        return _format_error(exc)


@mcp.tool(
    name="detect_highlights_local",
    annotations={
        "title": "Detect Audio Highlights (Local File)",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        # ローカルファイルを読むだけなので外部アクセスはしない。
        "openWorldHint": False,
    },
)
async def detect_highlights_local(
    params: DetectHighlightsLocalInput,
    ctx: Context,
) -> str:
    """Detect highlight moments from a LOCAL audio or video file.

    Same ranking logic as detect_highlights, but reads from a local file path
    instead of downloading from YouTube. Use this when:
    - YouTube download is blocked (bot detection / PO Token issues)
    - You already have a local recording (e.g., OBS capture)
    - You want to analyze a specific local mp4/mp3

    Supports audio files (wav, mp3, flac, ogg) and video files
    (mp4, mkv, avi, mov, webm) — the audio track is automatically extracted.

    Returns the same JSON structure as detect_highlights: a summary with
    total_duration_sec / speech_seconds / silence_seconds / total_volume_spikes
    and a `highlights` array. Each highlight includes rank, time_sec, time_hms,
    score, reasons, rms_db, rms_norm, pitch_hz, and spectral_centroid.

    Required args:
        file_path: absolute path to the audio/video file.
        top_n: max number of highlights to return (1-100).
        min_gap_sec: minimum gap between highlights in seconds (1-300).
    """
    # ── detect_highlights とほぼ同じ形。違いは analyze_local を呼ぶことだけ。──
    try:
        def _runner(cb: Callable[[str, float], None]):
            return analyze_local(
                params.file_path,
                sample_rate=params.sample_rate,
                hop_length=params.hop_length,
                frame_length=params.frame_length,
                start_sec=params.start_sec,
                end_sec=params.end_sec,
                progress_cb=cb,
            )

        frames = await _run_with_heartbeat(ctx, _runner)

        summary = build_highlight_summary(
            frames,
            top_n=params.top_n,
            min_gap_sec=params.min_gap_sec,
        )
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("detect_highlights_local failed: %s", exc, exc_info=True)
        return _format_error(exc)


# ------------------------------------------------------------------
# エントリーポイント
# ------------------------------------------------------------------


def main() -> None:
    """MCPサーバーを stdio transport で起動する。

    stdio transport: 標準入出力(stdin/stdout)でMCPクライアントと通信するモード。
    Claude Desktop などローカルMCPクライアントはこの方式で接続してくる。
    """
    mcp.run()


# `if __name__ == "__main__":` は「このファイルが直接実行された時だけ走る」のお約束。
# 他のモジュールから `import audio_analyzer_mcp.server` された時は実行されない。
if __name__ == "__main__":
    main()
