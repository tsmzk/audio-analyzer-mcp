"""Output and error formatting helpers.

- Convert AudioFrame lists to CSV or JSON for MCP tool responses
- Turn exceptions from the analysis pipeline into actionable error messages
"""

from __future__ import annotations

# Python標準ライブラリ(追加インストール不要):
#   csv  : CSV(カンマ区切り)を読み書きするモジュール
#   io   : メモリ上で「ファイルっぽいもの」を扱うためのモジュール
#   json : JSON形式の文字列化/パースを行うモジュール
import csv
import io
import json

from audio_analyzer_mcp.constants import (
    AnalysisError,
    AudioFrame,
    DownloadError,
)


def _frames_to_csv(frames: list[AudioFrame]) -> str:
    """AudioFrame のリストを CSV文字列に変換する。

    関数名の先頭の `_` は「このモジュール内部だけで使う」という慣習的マーク。
    Python自体に強制力はないが、「外からは呼ばないで」と伝える記号。
    """
    # StringIO は「メモリ上のテキストファイル」。ファイル名を使わずに、
    # `csv.DictWriter` がファイルに書くのと同じ操作をメモリ上で行える。
    output = io.StringIO()

    # CSV の列名(ヘッダー行)。書き込み順もこの通りになる。
    fieldnames = [
        "timestamp", "time_sec", "rms_db", "rms_norm",
        "pitch_hz", "spectral_centroid", "is_speech", "volume_spike",
    ]

    # DictWriter: 辞書(dict)をそのまま1行として書き込めるCSVライター。
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()  # 1行目にヘッダーを書く

    # `for frame in frames:` でリストを1つずつ取り出して処理。
    for frame in frames:
        # frame.to_dict() で {列名: 値} の辞書を作り、1行として書き込む。
        writer.writerow(frame.to_dict())

    # `getvalue()` で StringIO に書き込まれた全テキストを文字列として取り出す。
    return output.getvalue()


def _frames_to_json(frames: list[AudioFrame]) -> str:
    """AudioFrame のリストを JSON文字列に変換する。"""
    # リスト内包表記: `[式 for 要素 in イテラブル]`
    # → 各 frame を to_dict() した結果を集めたリストを作る、の短い書き方。
    #   通常のfor文で書くと:
    #     data = []
    #     for f in frames:
    #         data.append(f.to_dict())
    data = [f.to_dict() for f in frames]

    # json.dumps: Pythonのオブジェクト → JSON文字列。
    # ensure_ascii=False を付けないと日本語が \u30c4... のようにエスケープされる。
    return json.dumps(data, ensure_ascii=False)


def _format_frames(frames: list[AudioFrame], fmt: str) -> str:
    """フォーマット指定(csv/json)に応じて上の2つを使い分ける玄関口。"""
    if fmt == "json":
        return _frames_to_json(frames)
    # デフォルトは CSV(数値処理しやすいため、MCPツールの既定値もこちら)
    return _frames_to_csv(frames)


def _format_error(error: Exception) -> str:
    """例外オブジェクトを「ユーザーが読んで対処できるメッセージ」に整形する。

    `isinstance(error, DownloadError)` は「error が DownloadError(またはその子)の
    インスタンスか」の判定。例外の種類で分岐して、それぞれに合う対処法を添える。
    """
    if isinstance(error, DownloadError):
        return (
            # 括弧の中で文字列リテラルを並べると自動で連結される。改行を `\n` で入れている。
            f"Download Error: {error}\n\n"
            "This is most often caused by YouTube's bot detection (e.g. "
            "'Sign in to confirm you're not a bot', PO Token required).\n\n"
            "Suggestions:\n"
            "- If you already have a local recording of this video (e.g. OBS capture), "
            "use `detect_highlights_local` or `analyze_local_audio` instead — it skips "
            "the YouTube download entirely.\n"
            "- Check the YouTube URL is correct and the video is public.\n"
            "- Ensure yt-dlp and ffmpeg are installed and on PATH.\n"
            "- Try updating yt-dlp: pip install -U yt-dlp"
        )
    elif isinstance(error, AnalysisError):
        return (
            f"Analysis Error: {error}\n\n"
            "Suggestions:\n"
            "- Ensure the file exists and is a valid audio/video format\n"
            "- Check that librosa and soundfile are installed\n"
            "- For large files, analysis may take several minutes"
        )
    elif isinstance(error, ValueError):
        # Pydanticのバリデーション失敗などは ValueError 系になる。
        return f"Input Error: {error}"
    else:
        # 想定外の例外。type(error).__name__ はクラス名の文字列(例: "KeyError")。
        return f"Unexpected Error: {type(error).__name__}: {error}"
