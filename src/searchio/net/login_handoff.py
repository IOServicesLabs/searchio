"""Delegated headed login handoff (firing 40) — the agent never touches credentials.

Design 3 of the agent-driven gap designs: when a loaded session hits a
gate only a real login clears (Facebook Marketplace's silent
``feed_units: null`` is the measured case — see
:mod:`searchio.providers.facebook`), the recovery is not a scripted
credential submission. Scripted login from a fresh automation profile is
the single most reliable way to earn a checkpoint, and a checkpointed
account returns LESS than no account. So the flow is delegated and
human-executed:

1. the agent detects the gate (``session_gate == "empty_feed"`` on the
   fallback items — detection already exists in the provider);
2. :func:`request_human_login` opens a HEADED browser on the login page —
   a window a human can type into, which the engine (no rasterizer, no
   input path) deliberately is not;
3. a human logs in by hand. The poll loop watches for the session cookie;
   it never sees a password, and neither does any artifact;
4. :func:`apply_to_engine` injects the exported Playwright
   ``storage_state`` artifact into the engine/sidecar via the existing
   ``storage_state_set`` verb (cookies + per-origin localStorage) and
   optionally persists it with ``session_save``;
5. headless work resumes with a session the risk engine trusts.

The whole thing is one awaitable tool call an agent can park on: pass an
``asyncio.Event`` as ``cancel`` to abort (human walked away), and a small
``timeout_s`` for tests. The headed browser is patchright — the same
Chromium SwarmIO's sidecar drives — imported lazily so searchio's venv
keeps working without it; a missing backend raises
:class:`~searchio.errors.LoginHandoffUnavailable` with the actionable
reason instead of an import error at startup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import LoginHandoffCancelled, LoginHandoffError, LoginHandoffTimeout, LoginHandoffUnavailable

DEFAULT_LOGIN_URL = "https://www.facebook.com/login"
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_POLL_INTERVAL_S = 2.0
CLOSE_TIMEOUT_S = 10.0
EXPORT_TIMEOUT_S = 30.0


@dataclass
class HandoffResult:
    """What a completed handoff produced. ``state`` is the Playwright
    ``storage_state`` artifact: cookies (httpOnly included) + per-origin
    localStorage. It contains session proof, never credentials."""

    state: dict[str, Any]
    login_url: str
    elapsed_s: float

    @property
    def cookies(self) -> int:
        return len(self.state.get("cookies") or [])

    @property
    def origins(self) -> int:
        return len(self.state.get("origins") or [])


class HeadedDriver(Protocol):
    """The seam the handoff talks to. Duck-typed on purpose: tests script a
    fake (open/poll/export/close), production uses :class:`PatchrightDriver`.
    Anything with these four methods is a valid driver."""

    async def open(self, url: str) -> None: ...
    async def poll_logged_in(self) -> bool: ...
    async def export_state(self) -> dict[str, Any]: ...
    async def close(self) -> None: ...


async def request_human_login(
    *,
    driver: HeadedDriver | None = None,
    login_url: str = DEFAULT_LOGIN_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    cancel: asyncio.Event | None = None,
) -> HandoffResult:
    """Open a headed login window and wait for a human to log in by hand.

    Returns the exported session artifact once the probe cookie appears.
    Raises :class:`LoginHandoffTimeout` when nobody logged in within
    ``timeout_s``, :class:`LoginHandoffCancelled` when ``cancel`` is set,
    and always closes the browser on the way out (the window is for the
    human, not a resource to leak).
    """
    drv: HeadedDriver = driver if driver is not None else PatchrightDriver()
    started = time.monotonic()
    deadline = started + timeout_s

    def remaining() -> float:
        return max(deadline - time.monotonic(), 0.01)

    try:
        # Every driver call is bounded by the same deadline (bug 83): the
        # deadline used to bite only BETWEEN polls, so an open() on a wedged
        # browser or a poll that never returned hung the handoff forever.
        try:
            await asyncio.wait_for(drv.open(login_url), remaining())
        except asyncio.TimeoutError:
            raise LoginHandoffTimeout(
                f"login page did not open within {timeout_s:.0f}s: {login_url}"
            ) from None
        while True:
            if cancel is not None and cancel.is_set():
                raise LoginHandoffCancelled(f"login handoff cancelled at {login_url}")
            try:
                if await asyncio.wait_for(drv.poll_logged_in(), remaining()):
                    break
            except asyncio.TimeoutError:
                pass  # the deadline check below reports it
            except Exception:
                # A probe that hiccups while the page navigates mid-login is
                # not terminal — the deadline is the only timeout that bites.
                pass
            if time.monotonic() >= deadline:
                raise LoginHandoffTimeout(
                    f"no login within {timeout_s:.0f}s of opening {login_url}"
                )
            await asyncio.sleep(min(poll_interval_s, remaining()))
        try:
            state = await asyncio.wait_for(drv.export_state(), EXPORT_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise LoginHandoffError("logged in, but the session export did not "
                                    f"complete within {EXPORT_TIMEOUT_S:.0f}s") from None
        # The export is what gets injected: a state without the session
        # cookie (a race with the jar, a driver bug) is not a success.
        if cancel is not None and cancel.is_set():
            raise LoginHandoffCancelled(f"login handoff cancelled at {login_url}")
        state_ok = getattr(drv, "state_ok", None)
        # A driver without its own check still cannot call an empty export a
        # session (bug 156): no cookies is nothing to inject.
        valid = state_ok(state) if state_ok is not None else bool(
            isinstance(state, dict) and state.get("cookies"))
        if not valid:
            raise LoginHandoffError(
                "logged in, but the exported state carries no session cookie "
                "for the login host -- nothing to inject")
        return HandoffResult(
            state=state, login_url=login_url, elapsed_s=time.monotonic() - started
        )
    finally:
        try:
            # Bounded (bug 156 rider): a wedged browser used to hold the
            # handoff open long after its deadline.
            await asyncio.wait_for(drv.close(), CLOSE_TIMEOUT_S)
        except Exception:
            pass  # the window is best-effort cleanup; never mask the result


async def apply_to_engine(
    sidecar: Any,
    state: dict[str, Any],
    *,
    save_path: str = "",
) -> dict[str, Any]:
    """Inject a handoff artifact into an engine/sidecar session and resume.

    ``sidecar`` is anything with ``SidecarClient.call`` — the Rust engine
    and the patchright sidecar speak the same verb. ``storage_state_set``
    loads cookies (httpOnly included) and per-origin localStorage into the
    LIVE jar; ``session_save`` persists the artifact to disk when a path
    is given (operator-managed session files survive a sidecar restart).
    """
    res = await sidecar.call("storage_state_set", {"storage_state": state})
    if not res.get("ok"):
        raise LoginHandoffError(f"engine rejected the session: {res.get('error')}")
    out: dict[str, Any] = {"applied": res, "saved": None}
    if save_path:
        saved = await sidecar.call("session_save", {"path": save_path})
        if not saved.get("ok"):
            raise LoginHandoffError(f"session did not persist: {saved.get('error')}")
        out["saved"] = saved
    return out


def empty_feed_gated(items: Any) -> bool:
    """Whether a search result set carries the silent empty-feed gate tag.

    Accepts provider ``Item`` models (``it.meta``) and plain dicts
    (``it["meta"]``) so an agent working over raw envelopes can use it too.
    """
    for it in items or []:
        meta = getattr(it, "meta", None)
        if meta is None and isinstance(it, dict):
            meta = it.get("meta")
        if isinstance(meta, dict) and meta.get("session_gate") == "empty_feed":
            return True
    return False


class PatchrightDriver:
    """The production driver: a headed patchright Chromium the human types into.

    patchright is imported lazily at :meth:`open` — searchio's own venv
    usually lacks it (SwarmIO's does), and the module must stay importable
    without. ``probe_cookie``/``probe_host`` decide what "logged in" means;
    the defaults are Facebook's session cookie, which appears the moment a
    login succeeds and never appears for a logged-out visit.
    """

    def __init__(
        self,
        *,
        probe_cookie: str = "c_user",
        probe_host: str = "facebook.com",
        proxy: str = "",
        launch_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._probe_cookie = probe_cookie
        self._probe_host = probe_host
        self._proxy = proxy
        self._launch_kwargs = dict(launch_kwargs or {})
        self._pw: Any = None
        self._browser: Any = None
        self._context: Any = None

    async def open(self, url: str) -> None:
        try:
            from patchright.async_api import async_playwright
        except ImportError as exc:
            raise LoginHandoffUnavailable(
                "patchright is not importable in this interpreter — run the "
                "handoff under the environment that drives SwarmIO's sidecar "
                "(it has patchright + Chromium), or pass driver= to "
                "request_human_login"
            ) from exc
        self._pw = await async_playwright().start()
        kwargs: dict[str, Any] = {"headless": False, **self._launch_kwargs}
        if self._proxy:
            kwargs["proxy"] = {"server": self._proxy}
        self._browser = await self._pw.chromium.launch(**kwargs)
        self._context = await self._browser.new_context()
        page = await self._context.new_page()
        await page.goto(url)

    async def poll_logged_in(self) -> bool:
        cookies = await self._context.cookies()
        return any(
            c.get("name") == self._probe_cookie
            and _host_matches(c.get("domain") or "", self._probe_host)
            for c in cookies
        )

    async def export_state(self) -> dict[str, Any]:
        return await self._context.storage_state()

    def state_ok(self, state: dict[str, Any]) -> bool:
        """The export carries the probe cookie for the probe host."""
        return any(
            c.get("name") == self._probe_cookie
            and _host_matches(c.get("domain") or "", self._probe_host)
            for c in (state.get("cookies") or [])
        )

    async def close(self) -> None:
        # Both halves always run: a browser that refuses to close must not
        # leave the playwright driver process behind (iteration 47 rider).
        try:
            if self._browser is not None:
                await self._browser.close()
        finally:
            self._browser = None
            if self._pw is not None:
                pw, self._pw = self._pw, None
                await pw.stop()


def _host_matches(cookie_domain: str, probe_host: str) -> bool:
    """``.facebook.com`` / ``m.facebook.com`` match ``facebook.com``;
    ``facebook.com.evil.test`` and ``notfacebook.com`` do not (a substring
    test counted any cookie whose domain merely CONTAINED the host)."""
    d = cookie_domain.lstrip(".").lower()
    h = probe_host.lstrip(".").lower()
    return d == h or d.endswith("." + h)
