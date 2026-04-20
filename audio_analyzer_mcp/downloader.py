"""YouTube audio download via yt-dlp.

Handles:
- yt-dlp / ffmpeg availability checks
- YouTube URL validation
- Duration probing (for progress reporting)
- Audio-only download and WAV conversion
"""

from __future__ import annotations

# 標準ライブラリ:
#   logging    : ログ出力(print より本格的。レベル・フォーマットを制御できる)
#   shutil     : ファイル/シェル関連ユーティリティ。ここでは外部コマンド存在確認に使用
#   subprocess : 外部コマンド(yt-dlp, ffmpeg)をPythonから呼び出す
#   pathlib    : ファイルパスをオブジェクト的に扱うモジュール(文字列連結より安全)
import logging
import shutil
import subprocess
from pathlib import Path

# 型ヒント:
#   Callable[[引数の型...], 戻り値の型] = 「関数を受け取る」の印
#   Optional[X] = X または None。呼び出し側が「指定しない」を許す用途
from typing import Callable, Optional

from audio_analyzer_mcp.constants import (
    DOWNLOAD_TIMEOUT_SEC,
    DownloadError,
    SAMPLE_RATE,
)

# `__name__` はこのモジュールの名前文字列(例: "audio_analyzer_mcp.downloader")。
# モジュール別ロガーを作ることで、どのモジュールから出たログか見分けやすくなる。
logger = logging.getLogger(__name__)

# 進捗コールバックの型を名前付きで定義しておく(後で何度も使うため)。
# 引数は (メッセージ文字列, 進捗率0.0〜1.0) の2つ。
ProgressCallback = Callable[[str, float], None]


def _emit(cb: Optional[ProgressCallback], message: str, fraction: float) -> None:
    """進捗コールバックに状況を通知する(None許容・失敗しても止まらない)。

    進捗通知は「あれば便利」程度のもの。コールバック側でエラーが出ても
    解析本体は止めたくないので、try/except で握りつぶしている。
    """
    if cb is None:
        return
    try:
        cb(message, fraction)
    except Exception:
        # `exc_info=True` を付けると、トレースバック(エラーの詳細)もログに出る。
        logger.debug("progress callback raised", exc_info=True)


def _check_yt_dlp() -> None:
    """yt-dlp コマンドが実行可能か確認する。見つからなければ即エラー。"""
    # shutil.which は Unix の `which` 相当。PATH から実行ファイルを探す。
    # 見つかれば絶対パス、なければ None。
    if not shutil.which("yt-dlp"):
        raise DownloadError(
            "yt-dlp not found. Install it with: pip install yt-dlp"
        )


def _check_ffmpeg() -> None:
    """ffmpeg コマンドが実行可能か確認する。yt-dlp が音声変換に使う。"""
    if not shutil.which("ffmpeg"):
        raise DownloadError(
            "ffmpeg not found. Install it with: brew install ffmpeg (Mac) "
            "or apt install ffmpeg (Linux)"
        )


def _validate_youtube_url(url: str) -> None:
    """URLが YouTube っぽい形をしているか簡易チェック。"""
    # .strip() で前後の空白除去、.lower() で小文字化。
    url_lower = url.strip().lower()

    # YouTubeの各種URL形式。タプル(変更不可のリスト)で定義。
    valid_prefixes = (
        "https://www.youtube.com/watch",
        "https://youtube.com/watch",
        "https://youtu.be/",
        "https://m.youtube.com/watch",
        "http://www.youtube.com/watch",
        "http://youtube.com/watch",
        "http://youtu.be/",
    )
    # any(): 「どれか1つでもTrueならTrue」。
    # `url_lower.startswith(p) for p in valid_prefixes` はジェネレータ式(遅延評価)。
    # どれかにマッチすればOK、全部マッチしなければエラー。
    if not any(url_lower.startswith(p) for p in valid_prefixes):
        raise DownloadError(
            f"Invalid YouTube URL: {url}\n"
            "Expected: https://www.youtube.com/watch?v=XXXXX or https://youtu.be/XXXXX"
        )


def _probe_youtube_duration(youtube_url: str) -> Optional[int]:
    """動画の長さ(秒)を yt-dlp に聞く。ダウンロードはしない。

    戻り値が Optional[int] なのは「取得失敗してもNoneで続行したい」ため。
    進捗表示用のオマケ情報なので、失敗しても解析は続ける。
    """
    try:
        # subprocess.run(): 外部コマンドを実行し、終わるまで待つ。
        #   capture_output=True : 標準出力・エラー出力をキャプチャ(result.stdout で取得)
        #   text=True           : 出力を bytes ではなく str(文字列) で取る
        #   timeout=60          : 60秒で打ち切る
        result = subprocess.run(
            [
                "yt-dlp",
                "--print", "%(duration)s",  # 秒数だけ出力
                "--skip-download",           # ダウンロードしない
                "--no-playlist",
                "--no-warnings",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        # タイムアウトや yt-dlp 不在などで例外が出たら None を返す。
        return None

    # returncode は外部コマンドの終了コード。0以外は失敗扱い。
    if result.returncode != 0:
        return None

    # 標準出力の1行目を取り出す。
    # splitlines() は改行で分割したリストを返す。空なら空文字列のまま。
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        # "123.4" のように小数で返ることがあるので、float → int の順で変換。
        return int(float(line))
    except (ValueError, IndexError):
        # 数値変換できない or 空リストアクセス時。
        return None


def download_youtube_audio(
    youtube_url: str,
    output_dir: str,
    # `*,` より後の引数は「キーワード引数でしか渡せない」という印。
    # `download_youtube_audio(url, dir, progress_cb=cb)` のみ可。
    # 位置引数としての誤用を防ぐ。
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> str:
    """YouTubeから音声だけダウンロードし、WAV形式で保存する。

    Returns:
        ダウンロード後のWAVファイルのパス。
    """
    # 実行前チェック(必要なコマンドが揃っているか、URLは妥当か)。
    _check_yt_dlp()
    _check_ffmpeg()
    _validate_youtube_url(youtube_url)

    # Pathオブジェクトは `/` 演算子でパスを連結できる(OS依存の区切り文字を自動処理)。
    # output_template は yt-dlp用テンプレート(%(ext)s は拡張子に展開される)。
    output_template = str(Path(output_dir) / "audio.%(ext)s")
    wav_path = str(Path(output_dir) / "audio.wav")

    _emit(progress_cb, "Probing video metadata...", 0.02)

    duration_sec = _probe_youtube_duration(youtube_url)
    # `is not None` は「値が None でない」の安全な書き方。
    # (0秒も None も「偽」なので、if duration_sec: だと 0秒動画を弾いてしまう)
    if duration_sec is not None:
        mins = duration_sec // 60  # `//` は整数除算(小数切り捨て)
        secs = duration_sec % 60   # `%`  は剰余(余り)
        logger.info("Video duration: %d:%02d", mins, secs)
        _emit(
            progress_cb,
            f"Video is {mins}:{secs:02d} — starting download",
            0.05,
        )
    else:
        _emit(progress_cb, "Starting download (duration unknown)", 0.05)

    logger.info("Downloading audio from: %s", youtube_url)

    # yt-dlp の実行コマンドを組み立てる。
    cmd = [
        "yt-dlp",
        "-x",                        # 音声のみ抽出(映像は捨てる)
        "--audio-format", "wav",     # WAV形式に変換(librosaが直接読める)
        "--audio-quality", "0",      # 0 = 最高品質
        "--no-playlist",
        "--no-check-certificates",
        "-o", output_template,       # 出力ファイル名テンプレート
        youtube_url,
    ]

    # ダウンロード実行。長尺動画も考慮して DOWNLOAD_TIMEOUT_SEC (10分) までは待つ。
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT_SEC,
    )

    if result.returncode != 0:
        # stderr[-500:] は「末尾500文字」。エラー全文は長すぎるので末尾だけ表示。
        raise DownloadError(
            f"yt-dlp failed (exit code {result.returncode}):\n{result.stderr[-500:]}"
        )

    # ── ダウンロードされたファイルを探す ──
    # yt-dlp は環境によって拡張子が .wav / .m4a / .webm など変わることがある。
    # `glob("audio.*")` で "audio.xxx" に該当するファイルをすべて取得。
    output_dir_path = Path(output_dir)
    audio_files = sorted(output_dir_path.glob("audio.*"))
    if not audio_files:
        raise DownloadError("Download completed but no audio file was found.")

    actual_path = audio_files[0]
    _emit(progress_cb, "Download complete", 0.15)

    # ── WAV でなければ ffmpeg で変換 ──
    # .suffix はファイルの拡張子(例: ".m4a")。小文字化して比較。
    if actual_path.suffix.lower() != ".wav":
        logger.info("Converting %s → WAV...", actual_path.suffix)
        _emit(progress_cb, f"Converting {actual_path.suffix} → WAV", 0.17)

        convert_cmd = [
            "ffmpeg", "-y", "-i", str(actual_path),  # -y: 既存ファイルを上書き
            "-ar", str(SAMPLE_RATE),  # サンプリングレート指定
            "-ac", "1",                # モノラル1chに変換(ステレオの片方は不要)
            wav_path,
        ]
        conv_result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if conv_result.returncode != 0:
            raise DownloadError(f"ffmpeg conversion failed:\n{conv_result.stderr[-500:]}")

        # 元の(非WAV)ファイルは削除してディスクを空ける。
        actual_path.unlink()
        return wav_path

    # 既にWAVだった場合はそのままパスを返す。
    return str(actual_path)
