"""Progress reporting bridge between the analysis thread and the MCP client.

The analysis pipeline runs in a worker thread via `asyncio.to_thread`, so its
progress callbacks fire off the event loop and cannot `await` directly. This
module marshals those callbacks back to the loop and keeps the MCP client
alive with periodic heartbeats while slow phases (notably librosa.pyin) run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from mcp.server.fastmcp import Context

# Interval between heartbeat progress events while the analysis thread is
# inside a phase that doesn't emit its own progress (notably librosa.pyin).
# Claude Desktop times out MCP tool calls at ~4 minutes of silence; sending a
# heartbeat every 15s keeps the client-side timeout reset with comfortable
# margin.
HEARTBEAT_INTERVAL_SEC = 15.0


def _make_progress_bridge(
    ctx: Context,
    loop: asyncio.AbstractEventLoop,
) -> Callable[[str, float], None]:
    """Return a thread-safe progress callback that forwards to `ctx`.

    The analysis runs inside `asyncio.to_thread`, so callbacks fire on a
    worker thread and cannot `await` directly. We schedule the async
    `ctx.report_progress` / `ctx.info` coroutines onto the main event loop.
    """

    def _callback(message: str, fraction: float) -> None:
        progress = max(0.0, min(1.0, fraction)) * 100.0
        try:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=progress, total=100.0, message=message),
                loop,
            )
            asyncio.run_coroutine_threadsafe(ctx.info(message), loop)
        except RuntimeError:
            # Event loop is gone (request cancelled). Nothing to do.
            pass

    return _callback


async def _run_with_heartbeat(
    ctx: Context,
    runner: Callable[[Callable[[str, float], None]], Any],
) -> Any:
    """Run a blocking `runner(progress_cb)` in a thread, emitting heartbeats.

    `runner` is invoked with a progress callback it can pass down into the
    analysis pipeline. While it runs, a heartbeat task keeps the MCP client
    alive by re-emitting the last seen progress every HEARTBEAT_INTERVAL_SEC.
    """
    loop = asyncio.get_running_loop()
    last_state: dict[str, Any] = {"message": "Starting...", "fraction": 0.0}
    bridge = _make_progress_bridge(ctx, loop)

    def _cb(message: str, fraction: float) -> None:
        last_state["message"] = message
        last_state["fraction"] = fraction
        bridge(message, fraction)

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            msg = f"Still working: {last_state['message']}"
            try:
                await ctx.report_progress(
                    progress=max(0.0, min(1.0, float(last_state["fraction"]))) * 100.0,
                    total=100.0,
                    message=msg,
                )
            except Exception:
                return

    hb_task = asyncio.create_task(_heartbeat())
    try:
        return await asyncio.to_thread(runner, _cb)
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass
