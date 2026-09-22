"""Delegated headed login handoff (firing 40).

The human step is scripted with a FakeDriver for the unit tests. The wire
test drives the real se-serve binary: a handoff artifact (host-only
127.0.0.1 cookie) goes through apply_to_engine, a page fetch on that host
must then carry the cookie, and session_save must persist the artifact to
disk. Provider tests drive _handoff_and_retry against a fake sidecar.

Skipped automatically when the engine binary has not been built yet.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from searchio.errors import (
    LoginHandoffCancelled,
    LoginHandoffError,
    LoginHandoffTimeout,
    LoginHandoffUnavailable,
)
from searchio.net import login_handoff
from searchio.net.login_handoff import (
    HandoffResult,
    PatchrightDriver,
    apply_to_engine,
    empty_feed_gated,
    request_human_login,
)
from searchio.net.sidecar import SidecarClient

ENGINE_BIN = (
    Path(__file__).parents[2]
    / "searchio-engine"
    / "target"
    / "debug"
    / "se-serve.exe"
)

_FACEBOOK_STATE = {
    "cookies": [
        {
            "name": "c_user",
            "value": "6159",
            "domain": ".facebook.com",
            "path": "/",
            "secure": True,
            "expires": 1893456000,
        }
    ],
    "origins": [
        {
            "origin": "https://www.facebook.com",
            "localStorage": [{"name": "m", "value": "v"}],
        }
    ],
}


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeDriver:
    """Scripted stand-in for the human. Poll results play out in order and
    the last one repeats; probe exceptions fire for the first ``exc_times``
    polls (a page navigating mid-login). Records open/close so the tests
    can assert the window is always cleaned up."""

    def __init__(
        self,
        polls: list[bool],
        *,
        state: dict | None = None,
        exc: Exception | None = None,
        exc_times: int = 0,
    ) -> None:
        self._polls = list(polls)
        self._state = state if state is not None else _FACEBOOK_STATE
        self._exc = exc
        self._exc_times = exc_times
        self.opened_with: str | None = None
        self.polls_made = 0
        self.closed = False

    async def open(self, url: str) -> None:
        self.opened_with = url

    async def poll_logged_in(self) -> bool:
        self.polls_made += 1
        if self._exc_times > 0:
            self._exc_times -= 1
            assert self._exc is not None
            raise self._exc
        i = min(self.polls_made - 1, len(self._polls) - 1)
        return self._polls[max(i, 0)]

    async def export_state(self) -> dict:
        return self._state

    async def close(self) -> None:
        self.closed = True


# ── request_human_login ──────────────────────────────────────────────────────


async def test_human_login_succeeds_once_probe_cookie_appears():
    drv = FakeDriver([False, False, True])
    res = await request_human_login(
        driver=drv,
        login_url="https://example.test/login",
        timeout_s=5.0,
        poll_interval_s=0.01,
    )
    assert isinstance(res, HandoffResult)
    assert res.state is drv._state
    assert res.login_url == "https://example.test/login"
    assert res.cookies == 1
    assert res.origins == 1
    assert res.elapsed_s >= 0
    assert drv.opened_with == "https://example.test/login"
    assert drv.closed is True, "the headed window is for the human; it must not leak"


async def test_human_login_times_out_and_still_closes():
    drv = FakeDriver([False])
    with pytest.raises(LoginHandoffTimeout, match="no login within"):
        await request_human_login(driver=drv, timeout_s=0.15, poll_interval_s=0.05)
    assert drv.closed is True


async def test_human_login_is_cancellable():
    cancel = asyncio.Event()

    class _WalkawayDriver(FakeDriver):
        async def poll_logged_in(self) -> bool:
            cancel.set()
            return False

    drv = _WalkawayDriver([False])
    with pytest.raises(LoginHandoffCancelled, match="cancelled"):
        await request_human_login(
            driver=drv, cancel=cancel, timeout_s=5.0, poll_interval_s=0.01
        )
    assert drv.closed is True


async def test_probe_hiccups_mid_login_are_tolerated():
    # A poll that throws while the page navigates must not end the wait —
    # the deadline is the only timeout that bites.
    drv = FakeDriver([False, True], exc=RuntimeError("navigating"), exc_times=2)
    res = await request_human_login(driver=drv, timeout_s=5.0, poll_interval_s=0.01)
    assert res.cookies == 1
    assert drv.polls_made >= 3


async def test_patchright_missing_is_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "patchright", None)
    with pytest.raises(LoginHandoffUnavailable, match="patchright"):
        await request_human_login(timeout_s=1.0, poll_interval_s=0.01)


async def test_patchright_driver_close_is_safe_when_never_opened():
    drv = PatchrightDriver()
    await drv.close()  # nothing started; must not raise


# ── apply_to_engine ──────────────────────────────────────────────────────────


class FakeSidecar:
    def __init__(self, results: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._results = results or {}

    async def call(self, verb: str, params: dict | None = None) -> dict:
        self.calls.append((verb, dict(params or {})))
        return self._results.get(verb, {"ok": True})


async def test_apply_injects_state_and_persists_when_asked():
    sc = FakeSidecar()
    out = await apply_to_engine(sc, _FACEBOOK_STATE, save_path="sess.json")
    assert [v for v, _ in sc.calls] == ["storage_state_set", "session_save"]
    assert sc.calls[0][1]["storage_state"] is _FACEBOOK_STATE
    assert sc.calls[1][1] == {"path": "sess.json"}
    assert out["applied"]["ok"] is True
    assert out["saved"]["ok"] is True


async def test_apply_without_save_path_skips_session_save():
    sc = FakeSidecar()
    out = await apply_to_engine(sc, {"cookies": [], "origins": []})
    assert [v for v, _ in sc.calls] == ["storage_state_set"]
    assert out["saved"] is None


async def test_apply_rejection_is_a_handoff_error():
    sc = FakeSidecar(results={"storage_state_set": {"ok": False, "error": "bad artifact"}})
    with pytest.raises(LoginHandoffError, match="bad artifact"):
        await apply_to_engine(sc, {"cookies": []})
    assert [v for v, _ in sc.calls] == ["storage_state_set"]


# ── gate glue ────────────────────────────────────────────────────────────────


def test_empty_feed_gated_reads_models_and_dicts():
    assert empty_feed_gated([{"meta": {"session_gate": "empty_feed"}}]) is True
    assert empty_feed_gated([SimpleNamespace(meta={"session_gate": "empty_feed"})]) is True
    assert empty_feed_gated([{"meta": {"session_gate": "empty_market"}}]) is False
    assert empty_feed_gated([{"meta": {}}]) is False
    assert empty_feed_gated([]) is False
    assert empty_feed_gated(None) is False


# ── provider wiring ──────────────────────────────────────────────────────────

_FB_HTML = """
<html><body>
  <a href="/marketplace/item/777001"><span>$250</span><span>Used kayak</span><span>Seattle, WA</span></a>
  <a href="/marketplace/item/777002"><span>$80</span><span>Climbing rope</span><span>Bellevue, WA</span></a>
</body></html>
"""


class _FakeProviderSidecar:
    """goto serves a static marketplace page; storage_state_set reports the
    cookie count like the engine does; everything else is ok:true."""

    def __init__(self, html: str) -> None:
        self.html = html
        self.calls: list[tuple[str, dict]] = []

    async def call(self, verb: str, params: dict | None = None) -> dict:
        self.calls.append((verb, dict(params or {})))
        if verb == "storage_state_set":
            state = (params or {}).get("storage_state") or {}
            return {"ok": True, "cookies": len(state.get("cookies") or [])}
        return {"ok": True}

    async def goto(self, url: str, tab_id: str | None = None) -> dict:
        self.calls.append(("goto", {"url": url, "tab_id": tab_id}))
        return {"ok": True, "status": 200}

    async def eval_js(self, js: str, tab_id: str | None = None, timeout: float | None = None):
        raise RuntimeError("no js in this fake")

    async def read_html(self, tab_id: str | None = None, selector: str | None = None) -> str:
        return self.html


def _provider_ctx(**overrides):
    settings = SimpleNamespace(
        facebook_login_url="https://example.test/login",
        handoff_timeout_s=12.0,
        facebook_session="",
    )
    for k, v in overrides.items():
        setattr(settings, k, v)
    return SimpleNamespace(settings=settings)


async def test_provider_handoff_retry_revives_session_and_tags_items(monkeypatch):
    from searchio.providers.facebook import FacebookMarketplace

    async def fake_handoff(**kw):
        assert kw["login_url"] == "https://example.test/login"
        assert kw["timeout_s"] == 12.0
        return HandoffResult(
            state=_FACEBOOK_STATE, login_url=kw["login_url"], elapsed_s=1.0
        )

    monkeypatch.setattr(login_handoff, "request_human_login", fake_handoff)

    prov = FacebookMarketplace()
    prov._session_broken = True  # _retry_anonymous proved the gate just before
    sc = _FakeProviderSidecar(_FB_HTML)
    items = await prov._handoff_and_retry(
        sc, "https://facebook.test/marketplace/search?query=kayak", want=5, ctx=_provider_ctx()
    )
    assert len(items) == 2
    assert all(it.meta["session"] == "handoff" for it in items)
    assert all("session_gate" not in it.meta for it in items)
    assert prov._session_state == "handoff:1c/1o"
    assert prov.search_gate == ""
    verbs = [v for v, _ in sc.calls]
    assert "storage_state_set" in verbs
    assert "close_tab" in verbs, "the retry tab is closed like every other load"
    assert "session_save" not in verbs, "no facebook_session configured -> nothing persisted"


async def test_provider_handoff_failure_leaves_the_fallback(monkeypatch):
    from searchio.providers.facebook import FacebookMarketplace

    async def fake_handoff(**kw):
        raise LoginHandoffTimeout("nobody came")

    monkeypatch.setattr(login_handoff, "request_human_login", fake_handoff)

    prov = FacebookMarketplace()
    sc = _FakeProviderSidecar(_FB_HTML)
    items = await prov._handoff_and_retry(
        sc, "https://facebook.test/marketplace/search?query=kayak", want=5, ctx=_provider_ctx()
    )
    assert items == []
    assert [v for v, _ in sc.calls] == [], "a failed handoff must not touch the sidecar"
    assert prov._session_broken is False  # unchanged; nothing claimed a revival


# ── wire test against the real se-serve ──────────────────────────────────────


async def test_handoff_artifact_rides_into_engine_pages_and_persists(tmp_path):
    """The engine half of the flow over the wire: apply_to_engine injects a
    handoff artifact (host-only 127.0.0.1 cookie) into the real se-serve, a
    page fetch on that host then carries it, and session_save persists the
    artifact to disk where _ensure_session would read it back."""
    if not ENGINE_BIN.exists():
        pytest.skip(f"se-serve binary not built: {ENGINE_BIN}")
    port = free_port()
    proc = subprocess.Popen(
        [str(ENGINE_BIN), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail(f"se-serve did not come up at {base}")

    seen: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Cookie", ""))
            body = b"<html><body>ok</body></html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    sc = SidecarClient(url=base, autostart=False)
    save_path = tmp_path / "session.json"
    try:
        state = {
            "cookies": [
                {
                    "name": "handoff_proof",
                    "value": "yes",
                    "domain": "127.0.0.1",
                    "path": "/",
                    "expires": 1893456000,
                }
            ],
            "origins": [],
        }
        out = await apply_to_engine(sc, state, save_path=str(save_path))
        assert out["applied"]["ok"] is True, out
        assert out["applied"]["cookies"] == 1

        # cookies key on HOST, not port: the jar cookie must ride a goto to
        # any 127.0.0.1 origin once it is in the live jar.
        page = f"http://127.0.0.1:{srv.server_address[1]}/"
        res = await sc.goto(page, tab_id="handoff")
        assert res.get("ok") is True, res
        assert any("handoff_proof=yes" in c for c in seen), (
            f"injected handoff cookie must ride the goto: {seen!r}"
        )

        # session_save persisted the artifact where apply asked.
        saved = json.loads(save_path.read_text(encoding="utf-8"))
        assert any(c["name"] == "handoff_proof" for c in saved["cookies"]), saved
    finally:
        await sc.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── iteration 47 riders (DeepSeek backends + handoff review) ─────────────────


async def test_probe_host_needs_a_dot_boundary():
    # `probe_host in cookie_domain` was a substring test: a c_user cookie
    # from facebook.com.evil.test or notfacebook.com counted as logged in.
    from types import SimpleNamespace

    from searchio.net.login_handoff import PatchrightDriver

    drv = PatchrightDriver()
    for dom, ok in [(".facebook.com", True), ("m.facebook.com", True),
                    ("facebook.com.evil.test", False), ("notfacebook.com", False)]:
        async def cookies(dom=dom):
            return [{"name": "c_user", "domain": dom, "value": "1"}]
        drv._context = SimpleNamespace(cookies=cookies)
        assert await drv.poll_logged_in() is ok, dom


async def test_close_stops_playwright_even_when_the_browser_refuses():
    from types import SimpleNamespace

    from searchio.net.login_handoff import PatchrightDriver

    stopped = []

    async def bad_close():
        raise RuntimeError("browser wedged")

    async def stop():
        stopped.append(True)

    drv = PatchrightDriver()
    drv._browser = SimpleNamespace(close=bad_close)
    drv._pw = SimpleNamespace(stop=stop)
    with pytest.raises(RuntimeError):
        await drv.close()
    assert stopped and drv._pw is None and drv._browser is None


async def test_open_that_hangs_is_bounded_by_the_deadline():
    # Bug 83 (iteration 47): the deadline only bit BETWEEN polls -- a
    # driver.open() (page.goto on a wedged browser) or a poll that never
    # returned hung the handoff past timeout_s forever.
    class Hanging(FakeDriver):
        async def open(self, url):
            self.opened_with = url
            await asyncio.sleep(3600)

    drv = Hanging([True])
    with pytest.raises(LoginHandoffTimeout):
        await asyncio.wait_for(request_human_login(driver=drv, timeout_s=0.2, poll_interval_s=0.01), 5.0)
    assert drv.closed


async def test_exported_state_without_the_session_cookie_is_refused():
    # Bug 83, second half: the poll saw the cookie but the export is what gets
    # injected -- an export that lacks the session cookie for the login host
    # (a race, a cleared jar, a driver bug) must not be reported as success.
    class Validating(FakeDriver):
        def state_ok(self, state):
            return any(c.get("name") == "c_user" for c in state.get("cookies") or [])

    drv = Validating([True], state={"cookies": [{"name": "other", "domain": ".facebook.com"}], "origins": []})
    with pytest.raises(LoginHandoffError, match="session cookie"):
        await request_human_login(driver=drv, timeout_s=1.0, poll_interval_s=0.01)
    assert drv.closed
    ok = Validating([True])
    res = await request_human_login(driver=ok, timeout_s=1.0, poll_interval_s=0.01)
    assert res.cookies == 1
    # The production driver validates with its probe, dot-boundary included.
    pd = PatchrightDriver()
    assert pd.state_ok(_FACEBOOK_STATE) is True
    assert pd.state_ok({"cookies": [{"name": "c_user", "domain": "facebook.com.evil.test"}]}) is False



# ── iteration 69: Muse second pass ──────────────────────────────────────────


async def test_empty_export_is_not_a_success_without_state_ok():
    # Bug 156: a driver without `state_ok` skipped export validation, so an
    # export with no cookies at all came back as a successful handoff.
    from searchio.errors import LoginHandoffError

    drv = FakeDriver([True], state={"cookies": [], "origins": []})
    with pytest.raises(LoginHandoffError):
        await request_human_login(login_url="https://x.example/login", timeout_s=5, driver=drv,
                                  poll_interval_s=0.01)


async def test_wedged_close_does_not_hang_the_handoff():
    # Rider: drv.close() in the finally was unbounded -- a wedged browser
    # held the handoff open long after its deadline.
    import asyncio

    class Wedged(FakeDriver):
        async def close(self):
            await asyncio.sleep(60)
    drv = Wedged([True])
    from searchio.net import login_handoff as lh
    orig = lh.CLOSE_TIMEOUT_S
    lh.CLOSE_TIMEOUT_S = 0.2
    try:
        t0 = asyncio.get_running_loop().time()
        await request_human_login(login_url="https://x.example/login", timeout_s=5, driver=drv,
                                  poll_interval_s=0.01)
        assert asyncio.get_running_loop().time() - t0 < 2.0
    finally:
        lh.CLOSE_TIMEOUT_S = orig


async def test_cancel_during_export_is_honoured():
    import asyncio

    from searchio.errors import LoginHandoffCancelled

    cancel = asyncio.Event()

    class SlowExport(FakeDriver):
        async def export_state(self):
            cancel.set()
            await asyncio.sleep(0.05)
            return self._state
    drv = SlowExport([True])
    with pytest.raises(LoginHandoffCancelled):
        await request_human_login(login_url="https://x.example/login", timeout_s=5, driver=drv,
                                  poll_interval_s=0.01, cancel=cancel)
