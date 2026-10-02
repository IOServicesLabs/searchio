"""Chain oracles: three end-to-end proofs about the ladder's fetch chain
that no single existing suite pins.

1.  A sessioned body (docs/SESSIONS.md jar, not just a tenant cookie jar)
    reaches NO cache variant -- bug 138's shape one tier up. The write
    guard (``session is None`` at both cache.put sites) and the tenant
    variant key are each unit-pinned; this is the end-to-end proof that a
    private body fetched under one tenant can never answer another tenant,
    the default tenant, or the same tenant later -- while an open page
    still caches per variant (isolation without over-correction).

2.  Concurrent fetches over ONE seeded session close cleanly. The store's
    methods hold one lock, so nothing tears; what the ladder owns -- every
    fetch succeeds, every stored/written value is a rotation the origin
    actually served, the planted name survives -- is pinned here. The
    documented limit these tests quantify: tier-2 passes share the
    browser's ONE profile jar, so one pass's injection resets another
    pass's in-flight cookies, and write-back is last-writer-wins with no
    merge. Nothing serializes two passes over the same session; that
    contract question is the suite's, not these tests', to settle.

3.  The chain returns the HIGHEST serving tier's bytes, stamped with the
    identity of the backend that actually served them. A loopback origin
    serves an empty-mount shell to request 1 (tier 0) and full prose to
    request 2 (tier 1): the result must carry the tier-1 body under tier
    1's identity, never the shell. Plus the rendered-stamp matrix: a
    rendered=True request answered by a cheap tier claims no browser pass;
    a patchright primary does claim it; an engine challenge does not
    (bug 36).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from searchio.config import Settings
from searchio.errors import TransientError
from searchio.net.ladder import Ladder

HTML = "text/html"
GOOD = "<html><body>" + ("Plenty of real readable article content here. " * 20) + "</body></html>"
URL = "https://linked.example/page"
PUB = "https://pub.example/page"
SEED = "SEED-VALUE"
ROT = "ROT-BY-REQ"

PLANTED = {
    "name": "li_at",
    "value": SEED,
    "domain": ".linked.example",
    "path": "/",
    "expires": 1893456000,
    "httpOnly": True,
    "secure": True,
    "sameSite": "Lax",
}


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


def _plant(dirpath: Path) -> Path:
    """Write the seeded jar into ``dirpath`` and return the DIRECTORY (the
    store's address -- the cloud owns the filenames)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "linked.example.json").write_text(json.dumps({
        "cookies": [dict(PLANTED)],
        "origins": [],
        "swarmio": {"platform": "linkedin", "label": "recruiter-1",
                    "updated_at": "2026-09-22T16:40:00Z", "note": ""},
    }))
    return dirpath


def _writebacks(sessions_dir: str) -> list[Path]:
    wb = Path(sessions_dir) / "writeback"
    return sorted(wb.glob("*.json")) if wb.is_dir() else []


class _FakeStreamResponse:
    """The slice of httpx.Response the cheap tiers touch (same contract as
    the sessions suite's fake)."""

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


class _NonceGate:
    """Scripted network edge with per-request nonces.

    Gated hosts wall every anonymous request (and, in ``wall_always`` mode,
    jar-carrying ones too, so the climb reaches the browser tier). Every
    response carries its request ordinal, so a body served from cache
    cannot impersonate a fresh one. Jar-carrying requests rotate the
    planted cookie to their own nonce: every rotation is attributable, and
    the final jar state has a finite allowed set. The ``asyncio.sleep(0)``
    at entry lets concurrent fetches interleave their request streams.
    """

    def __init__(self, gated=frozenset(), wall_always=False):
        self.gated = set(gated)
        self.wall_always = wall_always
        self.requests: list[dict] = []

    async def __call__(self, url, headers, timeout):
        await asyncio.sleep(0)
        hdrs = dict(headers)
        host = urlsplit(url).hostname or ""
        cookie = hdrs.get("Cookie") or ""
        has_jar = PLANTED["name"] in cookie
        n = len(self.requests) + 1
        self.requests.append({"n": n, "host": host, "cookie": cookie,
                              "has_jar": has_jar})
        if host in self.gated and (not has_jar or self.wall_always):
            return _FakeStreamResponse(
                url, 403,
                b"<html><head><title>Just a moment...</title></head></html>",
                {"server": "cloudflare"})
        set_cookies = []
        if has_jar:
            set_cookies.append(
                f'{PLANTED["name"]}={ROT}-{n}; Path=/; HttpOnly; Secure; Max-Age=999999')
        body = (GOOD + f"<p>req={n} jar={'yes' if has_jar else 'no'}</p>").encode()
        return _FakeStreamResponse(url, 200, body, set_cookie=set_cookies)


class _ChainLadder(Ladder):
    """Real ladder on the scripted edge; tier 1 scripted transient so a
    climb that escapes its max_tier dies deterministically instead of
    reaching real DNS (the sessions suite's pattern)."""

    def __init__(self, settings, gate: _NonceGate, **kw):
        super().__init__(settings, **kw)
        self.gate = gate
        self._open_stream = gate

    async def _tier1(self, url, *, referer="", session=None):
        raise TransientError("tier1 unavailable in test")


class _ModeledJarSidecar:
    """The slice of a sidecar the session browser pass uses, modeling ONE
    shared profile jar the way a real browser owns it: ``storage_state_set``
    REPLACES the jar, a page rotates cookies during ``fetch``, and
    ``storage_state_get`` reads whatever is there now -- so a concurrent
    pass's injection resets another pass's in-flight cookies, exactly the
    hazard the concurrency test quantifies. No ``cookies`` verb on
    purpose: clearance capture then errors out instantly instead of
    spending its 6s settle retry (the sessions suite's fake contract)."""

    def __init__(self, seed: list[dict]):
        self.jar = [dict(c) for c in seed]
        self.calls: list[str] = []
        self.injection_count = 0
        self.state_get_count = 0
        self.fetch_count = 0
        self.available = True

    async def call(self, verb, params=None):
        self.calls.append(verb)
        return {"ok": True}

    async def fetch(self, url, **kw):
        self.calls.append("fetch")
        self.fetch_count += 1
        await asyncio.sleep(0)
        for c in self.jar:
            if c["name"] == PLANTED["name"]:
                c["value"] = f"BROWSER-{self.fetch_count}"
        return {"ok": True, "status": 200, "html": GOOD,
                "headers": {"content-type": HTML}, "final_url": url}

    async def storage_state_set(self, state):
        self.calls.append("storage_state_set")
        self.injection_count += 1
        self.jar = [dict(c) for c in state.get("cookies", [])]
        return True

    async def storage_state_get(self):
        self.calls.append("storage_state_get")
        self.state_get_count += 1
        return {"cookies": [dict(c) for c in self.jar], "origins": []}

    async def close(self):
        return None


def _seed_records() -> list[dict]:
    """The planted cookie as a Playwright storage_state record (the shape
    both injection and the modeled jar speak)."""
    return [{"name": PLANTED["name"], "value": SEED, "domain": ".linked.example",
             "path": "/", "expires": PLANTED["expires"],
             "httpOnly": True, "secure": True}]


class _StaticSidecar:
    """Tier-2 fake serving canned prose. No storage verbs (nothing in its
    tests carries a session); no ``cookies`` verb (keeps clearance capture
    on its instant-error path instead of the 6s settle)."""

    def __init__(self):
        self.calls: list[str] = []
        self.available = True

    async def call(self, verb, params=None):
        self.calls.append(verb)
        return {"ok": True}

    async def fetch(self, url, **kw):
        self.calls.append("fetch")
        return {"ok": True, "status": 200,
                "html": GOOD + "<p>TIER2-SIDECAR-MARKER</p>",
                "headers": {"content-type": HTML}, "final_url": url}

    async def close(self):
        return None


class _UnusableSidecar:
    """Tier 2 must never run in the climb tests: the shell/full edge
    answers at tier 0/1, so reaching the browser IS the bug under test.
    Any verb raising AssertionError makes that failure loud and named."""

    available = True

    async def call(self, verb, params=None):
        raise AssertionError(f"tier 2 must not run in this test (call {verb})")

    async def fetch(self, url, **kw):
        raise AssertionError("tier 2 must not run in this test (fetch)")

    async def close(self):
        return None


# ── Oracle 1: a sessioned body reaches no cache variant ───────────────────────

class TestSessionedBodiesNeverEnterTheCache:
    async def test_private_body_reaches_no_variant(self, make_settings):
        # Bug 138's shape one tier up, proven end to end with cache_enabled:
        # the open page must still cache per variant while the gated page's
        # jar-attached body must be unreachable from every variant.
        gate = _NonceGate(gated={"linked.example"})
        s = make_settings(cache_enabled=True)
        _plant(Path(s.sessions_dir))
        lad = _ChainLadder(s, gate)
        try:
            # Open page, three tenants: each miss caches under its own key.
            r1 = await lad.fetch(PUB, use_cache=True, max_tier=0)
            ra = await lad.fetch(PUB, use_cache=True, session="tenant-a", max_tier=0)
            rb = await lad.fetch(PUB, use_cache=True, session="tenant-b", max_tier=0)
            assert (r1.from_cache, ra.from_cache, rb.from_cache) == (False,) * 3
            rd = await lad.fetch(PUB, use_cache=True, max_tier=0)
            ra2 = await lad.fetch(PUB, use_cache=True, session="tenant-a", max_tier=0)
            assert rd.from_cache and "req=1" in rd.body      # shared variant
            assert ra2.from_cache and "req=2" in ra2.body    # tenant variant

            # Gated page under the seeded session: two rounds across three
            # tenants. Every answer is a fresh gate response; a cached
            # private body would repeat an earlier marker or skip the gate.
            seen: list[int] = []
            for _round in range(2):
                for tenant in ("", "tenant-a", "tenant-b"):
                    r = await lad.fetch(URL, use_cache=True, session=tenant,
                                        max_tier=0)
                    assert r.status == 200 and r.from_cache is False
                    assert "jar=yes" in r.body
                    seen.append(int(re.search(r"req=(\d+)", r.body).group(1)))
            assert len(set(seen)) == 6  # six distinct gate answers

            # The direct oracle: no variant holds the private body; the
            # open page's shared variant is the positive control.
            assert lad.cache.get(PUB, variant="") is not None
            assert lad.cache.get(URL, variant="") is None
            assert lad.cache.get(URL, variant="s:tenant-a") is None
            assert lad.cache.get(URL, variant="s:tenant-b") is None

            # Every retry carried the seeded jar -- the jar is the HOST's
            # login, shared across tenants by SESSIONS.md design -- and
            # every anonymous attempt was unauthenticated. After the first
            # rotation the retries carry the rotated value (write-back
            # working as designed), so the oracle is set membership, not
            # the literal seed.
            jar_reqs = [q for q in gate.requests if q["host"] == "linked.example"]
            assert len(jar_reqs) == 12  # 6 fetches x (anonymous wall, jar retry)
            assert [q["has_jar"] for q in jar_reqs] == [False, True] * 6
            allowed = {SEED} | {f"{ROT}-{q['n']}" for q in jar_reqs if q["has_jar"]}
            for q in jar_reqs:
                if q["has_jar"]:
                    m = re.search(rf'{PLANTED["name"]}=([^;]+)', q["cookie"])
                    assert m and m.group(1) in allowed
        finally:
            await lad.close()


# ── Oracle 2: concurrent fetches over one session close cleanly ───────────────

class TestConcurrentSessionFetches:
    async def test_cheap_tier_rotations_close(self, make_settings):
        # Six fetches race the same gated page; each retry rotates li_at to
        # its own request nonce. What the ladder owns: every fetch succeeds,
        # the store holds exactly one li_at whose value is one of the six
        # served rotations (last writer wins -- no merge, no tear), and
        # every write-back file is one of those observed states.
        gate = _NonceGate(gated={"linked.example"})
        s = make_settings()
        _plant(Path(s.sessions_dir))
        lad = _ChainLadder(s, gate)
        try:
            results = await asyncio.gather(
                *[lad.fetch(URL, use_cache=False, max_tier=0) for _ in range(6)])
            assert all(r.status == 200 for r in results)
            nonces = {f"{ROT}-{q['n']}" for q in gate.requests if q["has_jar"]}
            assert len(nonces) == 6  # one rotation per fetch, each attributable
            assert lad._stats.get("session_rotated") == 6
            sess = lad.sessions.for_host("linked.example")
            li = [c for c in lad.sessions.site_records(sess)
                  if c["name"] == PLANTED["name"]]
            assert len(li) == 1 and li[0]["value"] in nonces
            files = _writebacks(s.sessions_dir)
            assert 1 <= len(files) <= 6
            for f in files:
                doc = json.loads(f.read_text())
                vals = {c["value"] for c in doc["cookies"]
                        if c["name"] == PLANTED["name"]}
                assert len(vals) == 1 and vals <= nonces
            assert lad.sessions.errors == 0
        finally:
            await lad.close()

    async def test_browser_tier_passes_close_over_one_profile_jar(self, make_settings):
        # Four fetches climb to tier 2 (the gate walls even jar-carrying
        # requests) and race across the browser's ONE profile jar: every
        # pass injects the seed and the page rotates li_at per pass, so a
        # pass's injection demonstrably resets another's in-flight cookies.
        # The closure properties are what the ladder owns: all four passes
        # succeed, inject, read back, and close their own tab; the stored
        # jar and every write-back file hold a value the modeled browser
        # actually served (a pass's rotation, or the seed a clobber
        # restored) -- never a torn or invented one.
        gate = _NonceGate(gated={"linked.example"}, wall_always=True)
        s = make_settings()
        _plant(Path(s.sessions_dir))
        sidecar = _ModeledJarSidecar(_seed_records())
        lad = _ChainLadder(s, gate, sidecar=sidecar)
        try:
            results = await asyncio.gather(
                *[lad.fetch(URL, use_cache=False) for _ in range(4)])
            assert all(r.tier == 2 and r.status == 200 for r in results)
            assert sidecar.injection_count == 4
            assert sidecar.state_get_count == 4  # one write-back read per pass
            assert sidecar.calls.count("close_tab") == 4
            assert lad._stats.get("session_browser_injected") == 4
            allowed = {f"BROWSER-{i}" for i in range(1, 5)} | {SEED}
            sess = lad.sessions.for_host("linked.example")
            li = [c for c in lad.sessions.site_records(sess)
                  if c["name"] == PLANTED["name"]]
            assert len(li) == 1 and li[0]["value"] in allowed
            files = _writebacks(s.sessions_dir)
            assert 1 <= len(files) <= 4
            for f in files:
                doc = json.loads(f.read_text())
                vals = {c["value"] for c in doc["cookies"]
                        if c["name"] == PLANTED["name"]}
                assert vals <= allowed
            assert lad.sessions.errors == 0
        finally:
            await lad.close()


# ── Oracle 3: the chain returns the highest tier's bytes, honestly stamped ────

SHELL = ('<html><head><title>Dashboard</title></head><body>'
         '<div id="root"></div><script src="/bundle.js"></script>'
         "<!-- " + "shell-padding " * 60 + "--></body></html>")
#: visible text is one word (classify: empty_mount via the id="root" mount),
#: so tier 0 can never pass it and the climb must bring tier 1's bytes back.
FULL = GOOD + "<p>TIER1-UNIQUE-CONTENT-MARKER</p>"


class _OracleHandler(BaseHTTPRequestHandler):
    """Request-index edge: request 1 gets the empty-mount shell, later
    requests get full prose. The climb's ORDER is the oracle -- which
    tier's bytes end up in the result."""

    Requests: list[dict] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        n = len(self.__class__.Requests) + 1
        self.__class__.Requests.append({"n": n})
        body = (SHELL if n == 1 else FULL).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestPerTierContentOracle:
    async def test_climb_returns_the_tier_that_served_real_content(self, make_settings):
        pytest.importorskip("curl_cffi")
        _OracleHandler.Requests = []
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _OracleHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            s = make_settings()
            lad = Ladder(s, sidecar=_UnusableSidecar())
            try:
                r = await lad.fetch(f"{base}/page", use_cache=False)
                assert r.tier == 1 and r.via == "impersonate"
                assert "TIER1-UNIQUE-CONTENT-MARKER" in r.body
                assert "shell-padding" not in r.body
                assert r.rendered is False and r.from_cache is False
                assert r.escalations == ["tier0:empty_mount"]
            finally:
                await lad.close()
        finally:
            srv.shutdown()

    async def test_rendered_request_answered_by_a_cheap_tier_claims_no_browser(
            self, make_settings):
        pytest.importorskip("curl_cffi")
        _OracleHandler.Requests = []
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _OracleHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            s = make_settings()
            lad = Ladder(s, sidecar=_UnusableSidecar())
            try:
                r = await lad.fetch(f"{base}/page", use_cache=False, rendered=True)
                assert r.tier == 1 and r.rendered is False
                assert "TIER1-UNIQUE-CONTENT-MARKER" in r.body
                assert "tier2_rendered" not in lad._stats
            finally:
                await lad.close()
        finally:
            srv.shutdown()

    async def test_rendered_stamp_claimed_by_a_real_browser_pass(self, make_settings):
        # Patchright primary: the challenge sidecar de-duplicates to it
        # (same kind, no separate URL), and a patchright pass IS a browser
        # -- the stamp may claim it, and the tier-2 bytes are the sidecar's.
        # Both knobs pinned: Settings reads .env (the development checkout
        # configures an engine primary), and this scenario IS the kind
        # pairing -- an inherited knob would silently test a different one.
        s = make_settings(sidecar_engine=False, sidecar_challenge="patchright")
        lad = Ladder(s, sidecar=_StaticSidecar())
        try:
            r = await lad.fetch("https://stamp.example/page", use_cache=False,
                                force_tier=2, rendered=True)
            assert r.tier == 2 and r.rendered is True
            assert "TIER2-SIDECAR-MARKER" in r.body
            assert lad._stats.get("tier2_rendered") == 1
        finally:
            await lad.close()

    async def test_engine_challenge_does_not_claim_the_browser_stamp(self, make_settings):
        # Engine on both knobs: the challenge de-duplicates to the engine
        # primary, whose tier-2 answer is parse-only -- bug 36's stamp must
        # stay False even though the caller paid for rendered=True.
        s = make_settings(sidecar_engine=True, sidecar_challenge="engine")
        sidecar = _StaticSidecar()
        sidecar.binary = Path(__file__)  # _primary_kind's honest tell: engine
        lad = Ladder(s, sidecar=sidecar)
        try:
            r = await lad.fetch("https://stamp.example/page", use_cache=False,
                                force_tier=2, rendered=True)
            assert r.tier == 2 and r.rendered is False
            assert "TIER2-SIDECAR-MARKER" in r.body
            assert "tier2_rendered" not in lad._stats
        finally:
            await lad.close()
