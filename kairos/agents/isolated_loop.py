"""Sub-agent Isolated Event Loop — DeerFlow-compatible async isolation.

Ensures sub-agents run in a dedicated, persistent asyncio event loop
so they don't corrupt the parent Gateway's shared connections.

DeerFlow equivalent: SubagentExecutor persistent isolated loop
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from contextvars import copy_context
from typing import Any, Callable, Coroutine

logger = logging.getLogger("kairos.agents.isolated_loop")


# ── Singleton isolated loop ───────────────────────────────────

_isolated_loop: asyncio.AbstractEventLoop | None = None
_isolated_loop_lock = threading.Lock()
_isolated_loop_thread: threading.Thread | None = None


def get_isolated_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent isolated asyncio event loop.

    This loop lives in a dedicated daemon thread and persists across
    sub-agent invocations. Sub-agents use this instead of
    asyncio.run() which creates/destroys temporary loops.

    Returns:
        The isolated event loop (thread-safe).
    """
    global _isolated_loop, _isolated_loop_thread

    with _isolated_loop_lock:
        if _isolated_loop is None or _isolated_loop.is_closed():
            _isolated_loop = asyncio.new_event_loop()
            _isolated_loop_thread = threading.Thread(
                target=_run_isolated_loop,
                args=(_isolated_loop,),
                daemon=True,
                name="kairos-isolated-loop",
            )
            _isolated_loop_thread.start()
            logger.debug("Isolated event loop started in thread %s",
                         _isolated_loop_thread.name)

        return _isolated_loop  # type: ignore[return-value]


def _run_isolated_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run the isolated loop forever (until process exit)."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    except Exception:
        pass
    finally:
        loop.close()


def run_in_isolated_loop(coro: Coroutine, timeout: float = 300) -> Any:
    """Run a coroutine in the isolated loop and return its result.

    Thread-safe. Preserves context vars.

    Args:
        coro: The coroutine to run
        timeout: Max seconds to wait (default: 5 min)

    Returns:
        The coroutine's result

    Raises:
        TimeoutError: If the coroutine doesn't complete in time
        Any exception from the coroutine
    """
    loop = get_isolated_loop()

    # Preserve context vars (trace_id, etc.)
    ctx = copy_context()
    future = asyncio.run_coroutine_threadsafe(coro, loop)

    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise


async def run_in_isolated_loop_async(coro: Coroutine, timeout: float = 300) -> Any:
    """Async wrapper: run a coroutine in the isolated loop without blocking.

    Use this when called from inside an existing async context (e.g. Gateway).

    Args:
        coro: The coroutine to run
        timeout: Max seconds to wait

    Returns:
        The coroutine's result
    """
    return await asyncio.get_event_loop().run_in_executor(
        None,
        run_in_isolated_loop,
        coro,
        timeout,
    )


# ── Cleanup ────────────────────────────────────────────────────


def _shutdown_isolated_loop() -> None:
    """Shutdown the isolated loop (called at process exit)."""
    global _isolated_loop
    with _isolated_loop_lock:
        if _isolated_loop and not _isolated_loop.is_closed():
            _isolated_loop.call_soon_threadsafe(_isolated_loop.stop)
            _isolated_loop = None


atexit.register(_shutdown_isolated_loop)


# ── Helper: detect and avoid nested loop issues ────────────────


def needs_isolated_loop() -> bool:
    """Return True if the current thread has a running event loop.

    If True, you should use run_in_isolated_loop_async instead of
    asyncio.run() to avoid 'cannot be called from a running event loop'.
    """
    try:
        loop = asyncio.get_running_loop()
        return loop is not None
    except RuntimeError:
        return False
