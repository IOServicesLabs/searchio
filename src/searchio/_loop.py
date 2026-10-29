"""Event-loop selection (``fast_loop``).

``asyncio.run`` with the stock loop, or on winloop / uvloop -- libuv-backed
loops with cheaper socket and callback dispatch -- when ``fast_loop`` is on
and one is installed. A missing package is the stock loop, never an error.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Coroutine


def fast_loop_factory():
    """The installed libuv loop class's ``new_event_loop``, else None."""
    try:
        if sys.platform == "win32":
            import winloop as mod  # type: ignore[import-not-found]
        else:
            import uvloop as mod  # type: ignore[import-not-found]
    except ImportError:
        return None
    return mod.new_event_loop


def run(coro: Coroutine[Any, Any, Any], *, fast: bool | None = None) -> Any:
    if fast is None:
        from .config import get_settings

        fast = get_settings().fast_loop
    factory = fast_loop_factory() if fast else None
    if factory is None:
        return asyncio.run(coro)
    if sys.version_info >= (3, 11):
        with asyncio.Runner(loop_factory=factory) as runner:
            return runner.run(coro)
    loop = factory()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            # asyncio.run's teardown: cancel what is left, then let it unwind.
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
