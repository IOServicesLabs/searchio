"""Seeded browser sessions: the searchio half of docs/SESSIONS.md.

The cloud exports login jars into a directory and drains writeback/; these
tests pin the searchio side of the contract, acceptance criteria first
(SESSIONS.md section 5):

1.  no directory  -> behaviour identical to a build without the feature
2.  jar present, page NOT gated -> fetch stays anonymous, no cookie sent
3.  jar present, page gated -> same-tier retry carries the jar under the
    site's pinned UA, and no browser launches where the site allows it
4.  rotated jars write back in the platform's shape; logged-out jars do not
5.  a deleted jar stops being used within one refresh
6.  no cookie VALUE appears in any log line, note, or stats payload

The network edge is a scripted gate (the trust-cookie suite's stubbed
_open_stream pattern): everything the feature touches -- the _fetch loop
policy, _tier0_document, _session_headers, the store, write-back, tier-2
injection -- is production code. Attach requires https by design, so the
loop-level tests speak https URLs at the ladder while the gate answers
without a socket; one tier-1 test runs a real curl_cffi pass over an http
loopback with only the https gate monkeypatched (that gate has its own
store-level pin below).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from searchio.config import Settings
from searchio.errors import Blocked, TransientError
from searchio.net import persona as persona_mod
from searchio.net.ladder import Ladder, _HopCookies
from searchio.net.sessions import (
    PUBLIC_SUFFIXES,
    SessionStore,
    _match_records,
    _parse_set_cookie,
)

HTML = "text/html"
GOOD = "<html><body>" + ("Plenty of real readable article content here. " * 20) + "</body></html>"
URL = "https://linked.example/page"
ROTATED = "ROT7F3D-SEED"  # distinctive: AC6 greps for it in logs/notes/stats

PLANTED = {
    "name": "li_at",
    "value": "SEED-VALUE",
    "domain": ".linked.example",
    "path": "/",
    "expires": 1893456000,
    "httpOnly": True,
    "secure": True,
    "sameSite": "Lax",
}


def _jar_doc(cookies=None, platform="linkedin", label="recruiter-1"):
    doc = {"cookies": [dict(c) for c in (cookies if cookies is not None else [PLANTED])],
           "origins": []}
    if platform is not None:
        doc["swarmio"] = {"platform": platform, "label": label,
                          "updated_at": "2026-09-22T16:40:00Z", "note": ""}
    return doc


def _plant(dirpath: Path, filename="linked.example.json", **kw) -> Path:
    """Write one jar into ``dirpath`` and return the DIRECTORY (the store's
    address -- the cloud owns the filenames)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / filename).write_text(json.dumps(_jar_doc(**kw)))
    return dirpath


@pytest.fixture
def make_settings(tmp_path):
    n = itertools.count()
    sessions_n = itertools.count()

    def _make(sessions_dir: str | None = None, **kw) -> Settings:
        d = tmp_path / f"state-{next(n)}"
        d.mkdir(exist_ok=True)
        if sessions_dir is None:
            sessions_dir = str(tmp_path / f"sessions-{next(sessions_n)}")
        kw.setdefault("cache_enabled", False)
        kw.setdefault("max_tier", 2)
        return Settings(
            state_dir=d,
            sessions_dir=sessions_dir,
            robots_policy="off",
            per_domain_rps=1000.0,  # keep tests fast; pacing is tested separately
            per_domain_burst=1000,
            sidecar_autostart=False,
            **kw,
        )

    return _make


class _FakeStreamResponse:
    """The slice of httpx.Response _tier0_document + _read_capped touch."""

    def __init__(self, url: str, status: int, body: bytes,
                 headers: dict | None = None, set_cookie: list | None = None):
        pairs = list((headers or {}).items()) + [("content-type", HTML)]
        for sc in set_cookie or []:
            pairs.append(("set-cookie", sc))
        self._headers = httpx.Headers(pairs)
        self.status_code = status
        self.url = url
        self.charset_encoding = None
        self._body = body

    @property
    def headers(self) -> httpx.Headers:
        return self._headers

    async def aiter_bytes(self, _n: int):
        yield self._body

    async def aclose(self) -> None:
        return None


class _Gate:
    """Scripted network edge: wall without the planted cookie, prose with it.

    ``wall_always`` walls even jar-carrying requests (the "session retry
    still gated" shape that hands the climb to tier 2). Records every
    request's headers -- the tests' only assertion surface.
    """

    def __init__(self, wall_status=403, wall_always=False, rotate_to=None,
                 new_name=None, expire_planted=False, expire_second=False):
        self.wall_status = wall_status
        self.wall_always = wall_always
        self.rotate_to = rotate_to
        self.new_name = new_name
        self.expire_planted = expire_planted
        self.expire_second = expire_second
        self.requests: list[dict] = []

    async def __call__(self, url, headers, timeout):
        hdrs = dict(headers)
        cookie = hdrs.get("Cookie") or ""
        has_jar = PLANTED["name"] in cookie
        self.requests.append({"url": url, "headers": hdrs, "cookie": cookie,
                              "ua": hdrs.get("User-Agent") or "",
                              "has_jar": has_jar})
        walled = self.wall_status is not None and (not has_jar or self.wall_always)
        if walled:
            return _FakeStreamResponse(
                url, self.wall_status,
                b"<html><head><title>Just a moment...</title></head></html>",
                {"server": "cloudflare"})
        set_cookies = []
        if has_jar and self.rotate_to:
            set_cookies.append(
                f'{PLANTED["name"]}={self.rotate_to}; Path=/; HttpOnly; '
                "Secure; Max-Age=999999")
        if has_jar and self.new_name:
            set_cookies.append(f"{self.new_name}=newbie; Path=/")
        if has_jar and self.expire_planted:
            set_cookies.append(f'{PLANTED["name"]}=gone; Path=/; Max-Age=0')
        if has_jar and self.expire_second:
            set_cookies.append('session_id=dead; Path=/; Expires=Wed, 01 Jan 2020 00:00:00 GMT')
        body = (GOOD + f"<p>jar={'yes' if has_jar else 'no'}</p>").encode()
        return _FakeStreamResponse(url, 200, body, set_cookie=set_cookies)


class _SessionLadder(Ladder):
    """Real ladder on a scripted network edge. Tier 1 is a scripted
    'unavailable' by default so climbs reach tier 2 deterministically
    without real DNS; the dedicated tier-1 test below uses the production
    curl_cffi path."""

    def __init__(self, settings, gate: _Gate, tier1_transient=True, **kw):
        super().__init__(settings, **kw)
        self.gate = gate
        self._open_stream = gate
        self._tier1_transient = tier1_transient

    async def _tier1(self, url, *, referer="", session=None):
        if self._tier1_transient:
            raise TransientError("tier1 unavailable in test")
        return await super()._tier1(url, referer=referer, session=session)


class _JarSidecar:
    """Records verbs; serves a canned page for fetch; answers
    storage_state_get with the WHOLE browser jar (other sites included --
    the write-back filter is what must keep them out)."""

    def __init__(self, state: dict, set_ok=True):
        self.calls: list[str] = []
        self.state = state
        self.set_ok = set_ok
        self.injected: dict | None = None
        self.available = True

    async def call(self, verb, params=None):
        self.calls.append(verb)
        return {"ok": True}

    async def fetch(self, url, **kw):
        self.calls.append("fetch")
        return {"ok": True, "status": 200, "html": GOOD,
                "headers": {"content-type": HTML}, "final_url": url}

    async def storage_state_set(self, state):
        self.calls.append("storage_state_set")
        self.injected = state
        return self.set_ok

    async def storage_state_get(self):
        self.calls.append("storage_state_get")
        return self.state

    async def close(self):
        return None


_BROWSER_STATE = {
    "cookies": [
        {"name": "li_at", "value": ROTATED + "-BROWSER", "domain": ".linked.example",
         "path": "/", "expires": 1893456000, "httpOnly": True, "secure": True},
        {"name": "session_id", "value": "browser-new-name", "domain": ".linked.example",
         "path": "/", "expires": -1, "httpOnly": True, "secure": True},
        {"name": "other_site", "value": "leak-check", "domain": ".unrelated.example",
         "path": "/", "expires": 1893456000, "httpOnly": True, "secure": True},
    ],
    "origins": [],
}


def _writebacks(sessions_dir: str) -> list[Path]:
    wb = Path(sessions_dir) / "writeback"
    return sorted(wb.glob("*.json")) if wb.is_dir() else []


def _pin_for(lad: Ladder, key: str = "linked.example"):
    return lad.sessions.pinned_ua(lad.sessions.for_host(key))


# ── AC1: no directory means today-behaviour ────────────────────────────────────

class TestNoDirectory:
    async def test_unset_dir_is_silent_noop(self, make_settings):
        s = make_settings(sessions_dir="")
        lad = _SessionLadder(s, _Gate())
        try:
            assert lad.sessions.enabled is False
            with pytest.raises(Blocked):
                await lad.fetch(URL, use_cache=False, max_tier=0)
            assert len(lad.gate.requests) == 1  # anonymous attempt, no retry
            assert "session_retry" not in lad._stats
            assert _writebacks("") == []
        finally:
            await lad.close()

    async def test_absent_directory_never_errors(self, make_settings, tmp_path):
        missing = str(tmp_path / "not-there")
        s = make_settings(sessions_dir=missing)
        lad = _SessionLadder(s, _Gate())
        try:
            with pytest.raises(Blocked):
                await lad.fetch(URL, use_cache=False, max_tier=0)
            assert lad.sessions.errors == 0
            assert not Path(missing).exists()  # the store never creates it
        finally:
            await lad.close()


# ── AC2: jar present, page not gated -> anonymous ──────────────────────────────

class TestAnonymousByDefault:
    async def test_plain_page_sends_no_cookie_despite_jar(self, make_settings):
        gate = _Gate(wall_status=None)  # serves everyone
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            r = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r.tier == 0 and "Plenty of real" in r.body
            assert len(gate.requests) == 1
            assert gate.requests[0]["cookie"] == ""
            assert gate.requests[0]["has_jar"] is False
            assert r.escalations == []
            assert _writebacks(s.sessions_dir) == []
        finally:
            await lad.close()


# ── AC3: gated -> same-tier retry with the jar, one identity, no browser ───────

class TestAttachOnAWall:
    async def test_wall_retries_same_tier_with_jar(self, make_settings):
        gate = _Gate()
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            r = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r.tier == 0 and "Plenty of real" in r.body
            assert [q["has_jar"] for q in gate.requests] == [False, True]
            assert "SEED-VALUE" in gate.requests[1]["cookie"]
            assert r.escalations == ["tier0:http_403", "tier0:session_retry"]
        finally:
            await lad.close()

    async def test_retry_never_boots_a_browser(self, make_settings):
        gate = _Gate()
        s = make_settings()
        _plant(Path(s.sessions_dir))
        sidecar = _JarSidecar(_BROWSER_STATE)
        lad = _SessionLadder(s, gate, sidecar=sidecar)
        try:
            r = await lad.fetch(URL, use_cache=False)
            assert r.tier == 0 and r.status == 200
            assert sidecar.calls == []  # tier 2 never ran
        finally:
            await lad.close()

    async def test_pinned_ua_is_one_stable_identity(self, make_settings):
        gate = _Gate()
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            await lad.fetch(URL, use_cache=False, max_tier=0)
            await lad.fetch(URL, use_cache=False, max_tier=0)
            pin = _pin_for(lad)
            assert pin == persona_mod.for_domain("linked.example").ua
            jar_reqs = [q for q in gate.requests if q["has_jar"]]
            assert len(jar_reqs) == 2
            assert all(q["ua"] == pin for q in jar_reqs)
            anon = gate.requests[0]
            assert anon["ua"] != pin or anon["ua"] == pin  # persona may collide;
            # the load-bearing part: the PIN, not the persona, rides the jar
            # (asserted above), and it persists:
            pins = json.loads((Path(s.state_dir) / "session_pins.json").read_text())
            assert pins["linked.example"] == pin
        finally:
            await lad.close()

    async def test_pin_survives_restart(self, make_settings):
        gate = _Gate()
        s1 = make_settings()
        _plant(Path(s1.sessions_dir))
        lad1 = _SessionLadder(s1, gate)
        try:
            await lad1.fetch(URL, use_cache=False, max_tier=0)
            pin = _pin_for(lad1)
        finally:
            await lad1.close()
        # Same sessions volume AND same state dir: the pin store lives in
        # ours, not in the cloud's file (SESSIONS.md 3.4).
        s2 = Settings(state_dir=s1.state_dir, sessions_dir=s1.sessions_dir,
                      cache_enabled=False, robots_policy="off", max_tier=0,
                      per_domain_rps=1000.0, per_domain_burst=1000,
                      sidecar_autostart=False)
        lad2 = _SessionLadder(s2, gate)
        try:
            assert _pin_for(lad2) == pin
        finally:
            await lad2.close()

    async def test_401_login_wall_also_triggers(self, make_settings):
        gate = _Gate(wall_status=401)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            r = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r.tier == 0 and r.status == 200
            assert [q["has_jar"] for q in gate.requests] == [False, True]
            assert r.escalations == ["tier0:auth_required_401", "tier0:session_retry"]
        finally:
            await lad.close()

    async def test_429_never_attaches_an_identity(self, make_settings):
        gate = _Gate(wall_status=429)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            with pytest.raises(TransientError):
                await lad.fetch(URL, use_cache=False, max_tier=0)
            assert len(gate.requests) == 1  # no retry at a throttle
            assert gate.requests[0]["cookie"] == ""
            assert "session_retry" not in lad._stats
        finally:
            await lad.close()

    async def test_jar_attached_body_is_never_cached(self, make_settings):
        # Bug 138's shape one tier up: a private page must not answer a later
        # anonymous fetch from the shared cache. Both fetches allow the cache;
        # if the jar-attached 200 from fetch 1 leaked into it, fetch 2 would
        # come back from_cache with zero gate traffic.
        gate = _Gate()
        s = make_settings(cache_enabled=True)
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            r1 = await lad.fetch(URL, use_cache=True, max_tier=0)
            r2 = await lad.fetch(URL, use_cache=True, max_tier=0)
            assert r1.from_cache is False and r2.from_cache is False
            assert r2.status == 200
            assert [q["has_jar"] for q in gate.requests] == [False, True, False, True]
        finally:
            await lad.close()


# ── AC3 (tier 1): the impersonation tier follows the pin ───────────────────────

class TestTier1:
    async def test_tier1_retry_carries_jar_under_pinned_ua(self, make_settings,
                                                           monkeypatch):
        pytest.importorskip("curl_cffi")
        _LiveHandler.Requests = []
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _LiveHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            s = make_settings(max_tier=1)
            # The ladder resolves a jar by the fetch URL's host, so the
            # loopback pass needs a loopback-keyed jar. It speaks http, so
            # the cookie must not be Secure -- _session_headers derives the
            # header with the REAL scheme, and a Secure cookie drops off
            # https (production only attaches over https, where it rides).
            _plant(Path(s.sessions_dir), filename="127.0.0.1.json",
                   cookies=[dict(PLANTED, domain="127.0.0.1", secure=False)])
            lad = _SessionLadder(s, _Gate(), tier1_transient=False)
            # The https gate is pinned at store level below; this test's
            # subject is the tier-1 header/impersonation path, so open the
            # gate for the loopback only.
            monkeypatch.setattr(
                type(lad.sessions), "attachable",
                lambda self, sess, url: bool(
                    self.cookie_header(sess, urlsplit(url).hostname or "", "https")))
            seen_impersonate: list[str] = []
            monkeypatch.setattr(
                persona_mod, "impersonate_for_ua",
                lambda ua, default: seen_impersonate.append(ua) or default)
            try:
                r = await lad.fetch(f"{base}/page", use_cache=False, force_tier=1)
                assert r.tier == 1 and r.status == 200
                reqs = _LiveHandler.Requests
                assert [q["has_jar"] for q in reqs] == [False, True]
                pin = _pin_for(lad, "127.0.0.1")
                assert reqs[1]["ua"] == pin
                assert pin in seen_impersonate  # bug-141 chain followed the pin
            finally:
                await lad.close()
        finally:
            srv.shutdown()


class _LiveHandler(BaseHTTPRequestHandler):
    Requests: list[dict] = []  # class attr; reset per server by the test

    def log_message(self, *a):
        pass

    def do_GET(self):
        cookie = self.headers.get("Cookie") or ""
        has_jar = PLANTED["name"] in cookie
        self.__class__.Requests.append({
            "cookie": cookie, "has_jar": has_jar,
            "ua": self.headers.get("User-Agent") or ""})
        if not has_jar:
            body = b"<html><title>Just a moment...</title></html>"
            self.send_response(403)
            self.send_header("Server", "cloudflare")
        else:
            body = GOOD.encode()
            self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── AC4: write-back ────────────────────────────────────────────────────────────

class TestWriteback:
    async def test_rotated_jar_writes_back_in_platform_shape(self, make_settings):
        gate = _Gate(rotate_to=ROTATED)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            await lad.fetch(URL, use_cache=False, max_tier=0)
            files = _writebacks(s.sessions_dir)
            assert len(files) == 1
            doc = json.loads(files[0].read_text())
            assert doc["platform"] == "linkedin" and doc["label"] == "recruiter-1"
            names = [c["name"] for c in doc["cookies"]]
            assert names == ["li_at"]  # the whole jar, one cookie
            c = doc["cookies"][0]
            assert c["value"] == ROTATED
            assert c["http_only"] is True and "httpOnly" not in c  # snake_case
            assert c["secure"] is True and isinstance(c["expires"], int)
            assert c["domain"] == "linked.example" and c["path"] == "/"
        finally:
            await lad.close()

    async def test_new_name_on_rotation_does_not_enter_jar(self, make_settings):
        gate = _Gate(rotate_to=ROTATED, new_name="fresh_id")
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            await lad.fetch(URL, use_cache=False, max_tier=0)
            files = _writebacks(s.sessions_dir)
            assert len(files) == 1
            doc = json.loads(files[0].read_text())
            assert [c["name"] for c in doc["cookies"]] == ["li_at"]
        finally:
            await lad.close()

    async def test_logged_out_jar_is_not_written_back(self, make_settings):
        gate = _Gate(expire_planted=True)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            r = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r.status == 200  # the 200 still served; only the write-back is refused
            assert _writebacks(s.sessions_dir) == []
        finally:
            await lad.close()

    async def test_expired_records_omitted_from_writeback(self, make_settings):
        second = dict(PLANTED, name="session_id", expires=1893456000)
        gate = _Gate(rotate_to=ROTATED, expire_second=True)
        s = make_settings()
        _plant(Path(s.sessions_dir), cookies=[PLANTED, second])
        lad = _SessionLadder(s, gate)
        try:
            await lad.fetch(URL, use_cache=False, max_tier=0)
            files = _writebacks(s.sessions_dir)
            assert len(files) == 1
            doc = json.loads(files[0].read_text())
            # session_id was tombstoned by the response; a jar carrying an
            # expired record would be dropped WHOLE by the cloud's drain.
            assert [c["name"] for c in doc["cookies"]] == ["li_at"]
        finally:
            await lad.close()

    async def test_untouched_response_writes_nothing(self, make_settings):
        gate = _Gate(new_name="analytics_only")  # 200 with an unknown name only
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            await lad.fetch(URL, use_cache=False, max_tier=0)
            assert _writebacks(s.sessions_dir) == []
        finally:
            await lad.close()

    async def test_missing_swarmio_block_attaches_but_skips_writeback(self,
                                                                      make_settings):
        gate = _Gate(rotate_to=ROTATED)
        s = make_settings()
        _plant(Path(s.sessions_dir), platform=None)
        lad = _SessionLadder(s, gate)
        try:
            r = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r.status == 200  # no platform address -> the jar still attaches
            assert _writebacks(s.sessions_dir) == []
        finally:
            await lad.close()


# ── AC5: a deleted jar stops being used within one refresh ─────────────────────

class TestDeletion:
    async def test_deleted_jar_stops_attaching(self, make_settings):
        gate = _Gate()
        s = make_settings()
        d = _plant(Path(s.sessions_dir))
        jar = d / "linked.example.json"
        lad = _SessionLadder(s, gate)
        try:
            r1 = await lad.fetch(URL, use_cache=False, max_tier=0)
            assert r1.status == 200
            jar.unlink()
            with pytest.raises(Blocked):
                await lad.fetch(URL, use_cache=False, max_tier=0)
            # Third request (fetch 2's anonymous attempt) carries no cookie.
            assert gate.requests[2]["has_jar"] is False
            assert len(gate.requests) == 3
        finally:
            await lad.close()


# ── AC6: no cookie value in logs, notes, or stats ──────────────────────────────

class TestNoValuesLeak:
    async def test_rotated_value_absent_from_logs_notes_stats(self, make_settings,
                                                              caplog):
        gate = _Gate(rotate_to=ROTATED)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _SessionLadder(s, gate)
        try:
            with caplog.at_level(logging.DEBUG):
                await lad.fetch(URL, use_cache=False, max_tier=0)
            assert _writebacks(s.sessions_dir)  # the rotation really happened
            assert ROTATED not in caplog.text
            assert ROTATED not in json.dumps(lad.stats())
            assert all(ROTATED not in n for n in lad._capture_notes)
        finally:
            await lad.close()


# ── Browser tier: injection and read-back ──────────────────────────────────────

class TestBrowserTier:
    async def test_injects_then_writes_back_the_rotated_jar(self, make_settings):
        gate = _Gate(wall_always=True)  # the session retry stays gated
        s = make_settings()
        _plant(Path(s.sessions_dir))
        sidecar = _JarSidecar(_BROWSER_STATE)
        lad = _SessionLadder(s, gate, sidecar=sidecar)
        try:
            r = await lad.fetch(URL, use_cache=False)
            assert r.tier == 2 and r.status == 200
            # Inject -> fetch -> read back -> close, in that order.
            assert sidecar.calls == ["storage_state_set", "fetch",
                                     "storage_state_get", "close_tab"]
            doc = sidecar.injected
            assert doc["origins"] == []  # cookies only, never localStorage
            assert [c["name"] for c in doc["cookies"]] == ["li_at"]
            files = _writebacks(s.sessions_dir)
            assert len(files) == 1
            out = json.loads(files[0].read_text())
            names = [c["name"] for c in out["cookies"]]
            assert "li_at" in names and "session_id" in names  # browser replace
            assert "other_site" not in names  # another site's jar stayed out
            by_name = {c["name"]: c for c in out["cookies"]}
            assert by_name["li_at"]["value"] == ROTATED + "-BROWSER"
            assert "session_id" in by_name and "expires" not in by_name["session_id"]
        finally:
            await lad.close()

    async def test_injection_refusal_never_fails_the_fetch(self, make_settings):
        gate = _Gate(wall_always=True)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        sidecar = _JarSidecar(_BROWSER_STATE, set_ok=False)
        lad = _SessionLadder(s, gate, sidecar=sidecar)
        try:
            r = await lad.fetch(URL, use_cache=False)
            assert r.tier == 2 and r.status == 200  # anonymous pass, no exception
            assert sidecar.calls == ["storage_state_set", "fetch", "close_tab"]
            assert _writebacks(s.sessions_dir) == []  # nothing was injected
        finally:
            await lad.close()


# ── Store units ────────────────────────────────────────────────────────────────

class TestStore:
    def test_domain_rule_no_name_filtering(self, tmp_path):
        jar = [
            dict(PLANTED),  # .linked.example parent scope
            {"name": "session_id", "value": "v", "domain": ".linked.example"},
            {"name": "sub_only", "value": "v", "domain": "sub.linked.example"},
            {"name": "sibling", "value": "v", "domain": "auth.linked.example"},
            {"name": "orphan", "value": "v", "domain": ""},  # belongs nowhere
        ]
        store = SessionStore(str(_plant(tmp_path / "s", cookies=jar)), tmp_path / "st")
        sess = store.for_host("www.linked.example")
        assert sess is not None and sess.key == "linked.example"
        # Parent -> child matches; the browser rule. NO name filtering: a
        # login name like li_at/session_id survives (the clearance rule
        # would drop them -- that is the whole reason this store exists).
        header = store.cookie_header(sess, "www.linked.example", "https")
        assert "li_at=SEED-VALUE" in header and "session_id=v" in header
        # Child/sibling scopes stay on their own host: not on the parent,
        # not on a sibling subdoomain (bug 155's rule).
        parent = store.cookie_header(sess, "linked.example", "https")
        assert "li_at=SEED-VALUE" in parent
        assert "sub_only" not in header and "sub_only" not in parent
        assert "sibling" not in header and "sibling" not in parent
        assert "orphan" not in header
        # ...and a child-scoped jar never matches the bare parent.
        store2 = SessionStore(str(_plant(
            tmp_path / "s2", cookies=[{"name": "a", "value": "v",
                                       "domain": "sub.linked.example"}])),
            tmp_path / "st2")
        assert store2.for_host("linked.example") is None
        assert store.for_host("other.example") is None

    def test_attachable_requires_https(self, tmp_path):
        d = _plant(tmp_path / "s")
        store = SessionStore(str(d), tmp_path / "st")
        sess = store.for_host("linked.example")
        assert store.attachable(sess, "https://linked.example/x") is True
        assert store.attachable(sess, "http://linked.example/x") is False

    def test_mtime_reload_and_malformed_file(self, tmp_path):
        d = _plant(tmp_path / "s")
        p = d / "linked.example.json"
        store = SessionStore(str(d), tmp_path / "st")
        assert store.for_host("linked.example") is not None
        # Content changed -> mtime moved -> re-parsed on the next lookup.
        p.write_text(json.dumps(_jar_doc(cookies=[dict(PLANTED, value="V2")])))
        st = p.stat().st_mtime + 5
        os.utime(p, (st, st))
        sess = store.for_host("linked.example")
        assert "V2" in store.cookie_header(sess, "linked.example", "https")
        # Malformed JSON: treated absent, counted, never raised.
        p.write_text("{not json")
        st = p.stat().st_mtime + 5
        os.utime(p, (st, st))
        errors_before = store.errors
        assert store.for_host("linked.example") is None
        assert store.errors == errors_before + 1

    def test_deleted_file_dropped_on_rescan(self, tmp_path):
        d = _plant(tmp_path / "s")
        p = d / "linked.example.json"
        store = SessionStore(str(d), tmp_path / "st")
        assert store.for_host("linked.example") is not None
        p.unlink()
        assert store.for_host("linked.example") is None
        assert store.sites == 0

    def test_pinned_ua_deterministic(self, tmp_path):
        d = _plant(tmp_path / "s")
        a = SessionStore(str(d), tmp_path / "st1")
        b = SessionStore(str(d), tmp_path / "st2")
        sa, sb = a.for_host("linked.example"), b.for_host("linked.example")
        assert a.pinned_ua(sa) == b.pinned_ua(sb) == \
            persona_mod.for_domain("linked.example").ua

    def test_public_suffixes_match_hopcookies(self):
        # The table is duplicated from Ladder's hop-cookie jar (sessions.py
        # cannot import ladder -- ladder imports sessions); this pin catches
        # a drift on either side.
        assert PUBLIC_SUFFIXES == _HopCookies._PUBLIC_SUFFIXES

    def test_match_records_implements_relevant_cookies_rule(self):
        jar = [{"name": "n", "value": "v", "domain": d}
               for d in (".linked.example", "sub.linked.example", ".example", "")]
        recs = _match_records(jar, "www.linked.example")
        assert [r["domain"] for r in recs] == [".linked.example", ".example"]
        # No name filter: every name that domain-matches is kept.
        assert all(r["name"] == "n" for r in recs)


class TestParseSetCookie:
    def test_max_age_wins_and_tombstones(self):
        now = time.time()
        rec = _parse_set_cookie(
            "li_at=x; Path=/; Expires=Wed, 01 Jan 2041 00:00:00 GMT; Max-Age=1000; Secure; HttpOnly",
            "linked.example")
        assert rec["name"] == "li_at" and rec["value"] == "x"
        assert rec["expires"] is not None and rec["expires"] > now
        assert rec["expires"] < now + 2000  # Max-Age, not the 2041 date
        assert rec["secure"] is True and rec["http_only"] is True
        # No Domain attribute -> host-only record scoped to the setting host.
        assert rec["host_only"] is True and rec["domain"] == "linked.example"
        assert _parse_set_cookie("li_at=x; Max-Age=0", "linked.example")["delete"]
        assert _parse_set_cookie(
            "li_at=x; Expires=Wed, 01 Jan 2020 00:00:00 GMT",
            "linked.example")["delete"]

    def test_domain_validation_and_garbage(self):
        assert _parse_set_cookie("a=b; Domain=other.com", "linked.example") is None
        assert _parse_set_cookie("a=b; Domain=com", "linked.example") is None
        host_only = _parse_set_cookie("a=b", "linked.example")
        assert host_only["host_only"] is True and host_only["domain"] == "linked.example"
        assert _parse_set_cookie("no-equals", "h") is None
        assert _parse_set_cookie("", "h") is None
        assert _parse_set_cookie("=v", "h") is None
