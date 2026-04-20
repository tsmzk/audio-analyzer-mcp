"""Progress reporting bridge between the analysis thread and the MCP client.

The analysis pipeline runs in a worker thread via `asyncio.to_thread`, so its
progress callbacks fire off the event loop and cannot `await` directly. This
module marshals those callbacks back to the loop and keeps the MCP client
alive with periodic heartbeats while slow phases (notably librosa.pyin) run.
"""

# ------------------------------------------------------------------
# このファイルが何をするか(重要・やや難しい)
# ------------------------------------------------------------------
# このプロジェクトでは、音声解析(librosa)が「重くて時間がかかる同期処理」である。
# 一方、MCP(Model Context Protocol)との通信は「非同期(asyncio)」で動いている。
#
# 2つの世界が混在するため、次のような問題が起こる:
#   (A) 重い処理を素直に書くとイベントループが止まり、MCPクライアントが「応答なし」扱いで切断する
#   (B) 重い処理を別スレッドに逃がせば(A)は解決するが、進捗通知(非同期関数)を別スレッドから呼べない
#   (C) さらに、長時間無言だと Claude Desktop が「タイムアウト」とみなして切る(約4分で死ぬ)
#
# 対処:
#   - (A) の対応: `asyncio.to_thread(runner, ...)` で重い処理を別スレッドに逃がす
#   - (B) の対応: 別スレッドからの通知を `run_coroutine_threadsafe` で
#                イベントループに戻す「橋渡し関数」を作る(= _make_progress_bridge)
#   - (C) の対応: 15秒ごとに「まだ生きてます」イベントを送る「ハートビート」を流す
# ------------------------------------------------------------------

from __future__ import annotations

# asyncio: Pythonの「非同期処理」の標準ライブラリ。
#   `async def 関数`  : 非同期関数(コルーチン)の定義
#   `await 式`        : 「結果が出るまで待つ(が、他の処理は進める)」
#   イベントループ     : 複数の非同期処理を切り替えながら1スレッドで回す仕組み
import asyncio

from typing import Any, Callable

# MCP の Context オブジェクト。進捗・情報ログをクライアントへ送るメソッドを持つ。
from mcp.server.fastmcp import Context

# ハートビート間隔(秒)。
# Claude Desktop は約4分「何も通信がない」とタイムアウトさせる。
# 15秒ごとに送れば、タイムアウトまでかなり余裕を持って生存通知できる。
HEARTBEAT_INTERVAL_SEC = 15.0


def _make_progress_bridge(
    ctx: Context,
    loop: asyncio.AbstractEventLoop,
) -> Callable[[str, float], None]:
    """「別スレッド→メインループへの通知」を行う関数を作って返す。

    ここは「関数を作って返す」=高階関数のパターン。
    内側の _callback が外側の ctx と loop を覚えている(= クロージャ)。
    """

    def _callback(message: str, fraction: float) -> None:
        # 進捗率 0〜1 を 0〜100 に変換。範囲外(0未満/1超)もクランプして安全に。
        # min(1.0, max(0.0, x)) のイディオムで [0,1] に収める。
        progress = max(0.0, min(1.0, fraction)) * 100.0

        try:
            # run_coroutine_threadsafe: 「別スレッドから」イベントループに
            # コルーチン(非同期関数)の実行を投げ込むAPI。
            # これがあるので、別スレッドから ctx.report_progress/info を呼べる。
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=progress, total=100.0, message=message),
                loop,
            )
            asyncio.run_coroutine_threadsafe(ctx.info(message), loop)
        except RuntimeError:
            # RuntimeError が出るケース: ユーザーがリクエストをキャンセルして
            # イベントループが閉じた後。もう通知先がないので何もしない。
            pass

    return _callback


async def _run_with_heartbeat(
    ctx: Context,
    runner: Callable[[Callable[[str, float], None]], Any],
) -> Any:
    """重い処理を別スレッドで走らせつつ、裏でハートビートを流す。

    引数 runner は「進捗コールバックを受け取って、重い処理を実行する関数」。
    例:
        def _runner(cb):
            return analyze_youtube(url, progress_cb=cb)
        result = await _run_with_heartbeat(ctx, _runner)
    """
    # 現在動いているイベントループを取得(`ctx.*` を別スレッドから呼ぶ際の宛先になる)。
    loop = asyncio.get_running_loop()

    # 最後に受け取った進捗を共有するための辞書。
    # 辞書をラッパーに使うのは「内側関数から値を書き換えるため」
    # (Pythonでは、内側関数が外側のローカル変数を「再代入」するのは一手間なので辞書が楽)。
    last_state: dict[str, Any] = {"message": "Starting...", "fraction": 0.0}

    bridge = _make_progress_bridge(ctx, loop)

    def _cb(message: str, fraction: float) -> None:
        """重い処理から呼ばれるコールバック。状態を覚えつつ、実際の通知は bridge に丸投げ。"""
        last_state["message"] = message
        last_state["fraction"] = fraction
        bridge(message, fraction)

    async def _heartbeat() -> None:
        """15秒ごとに「まだ動いてます」を通知するタスク。

        これをバックグラウンドで回すことで、解析中のクライアント側タイムアウトを防ぐ。
        """
        # while True + await asyncio.sleep はクライアントタイムアウトまで無限ループ。
        # 外側の hb_task.cancel() で強制終了される想定。
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)

            msg = f"Still working: {last_state['message']}"
            try:
                # bridge(直接) ではなく await で直接通知。
                # こちらはメインループ上で走るので run_coroutine_threadsafe は不要。
                await ctx.report_progress(
                    progress=max(0.0, min(1.0, float(last_state["fraction"]))) * 100.0,
                    total=100.0,
                    message=msg,
                )
            except Exception:
                # 通知先が死んでいたら、ハートビートループを静かに終了。
                return

    # `asyncio.create_task` はコルーチンを「別タスクとして」登録する。
    # 下の to_thread と並行に動き、15秒ごとにハートビートを送る。
    hb_task = asyncio.create_task(_heartbeat())
    try:
        # asyncio.to_thread: 同期関数を別スレッドで実行し、その結果を await で待てる形にする。
        # この行で「重い処理の完了を待つ」。ハートビートは裏で勝手に動いている。
        return await asyncio.to_thread(runner, _cb)
    finally:
        # try/finally: 例外が出ても必ず実行されるブロック。
        # 重い処理が終わった(or 失敗した)後、ハートビートを止めてクリーンアップ。
        hb_task.cancel()
        try:
            # キャンセル完了を待つ。CancelledError が飛んでくるので受け止める。
            await hb_task
        except (asyncio.CancelledError, Exception):
            # キャンセル時の例外はここで握りつぶす(想定内のため)。
            pass
