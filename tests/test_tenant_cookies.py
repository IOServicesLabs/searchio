"""Tenant cookie isolation (iteration 62).

The HTTP server is one process shared by every agent that talks to it, and
the ladder's cheap tier kept ONE cookie jar: a Set-Cookie banked by tenant
A's read rode along on tenant B's read of the same host. Every test here
bit RED before its fix.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder

PROSE = "<html><body>" + "<p>Real prose about kayaks and rivers and paddles.</p>" * 30 + "<p>COOKIE={c}</p></body></html>"


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        body = PROSE.format(c=self.headers.get("Cookie") or "none").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        if self.path.startswith("/set"):
            self.send_header("Set-Cookie", "tenant_marker=" + self.path.split("=", 1)[-1] + "; Path=/")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _echo(body: str) -> str:
    return body.split("COOKIE=", 1)[1].split("<", 1)[0] if "COOKIE=" in body else "<none>"


@pytest.fixture
def lad(tmp_path):
    s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=0, robots_policy="off")
    return Ladder(s)


class TestTenantsDoNotShareCookies:
    async def test_other_tenant_does_not_see_the_cookie(self, lad, site):
        # Bug 135: one long-lived cheap-tier client deposited every response
        # cookie into ONE jar, so tenant B's read of the same host carried
        # tenant A's session cookie.
        try:
            await lad.fetch(f"{site}/set?v=A", use_cache=False, session="tenant-a")
            r = await lad.fetch(f"{site}/echo", use_cache=False, session="tenant-b")
            assert "tenant_marker" not in _echo(r.body), _echo(r.body)
        finally:
            await lad.close()

    async def test_same_tenant_keeps_its_cookie(self, lad, site):
        try:
            await lad.fetch(f"{site}/set?v=A", use_cache=False, session="tenant-a")
            r = await lad.fetch(f"{site}/echo", use_cache=False, session="tenant-a")
            assert "tenant_marker=A" in _echo(r.body), _echo(r.body)
        finally:
            await lad.close()

    async def test_default_session_is_its_own_tenant(self, lad, site):
        try:
            await lad.fetch(f"{site}/set?v=D", use_cache=False)
            r = await lad.fetch(f"{site}/echo", use_cache=False, session="tenant-b")
            assert "tenant_marker" not in _echo(r.body)
            r = await lad.fetch(f"{site}/echo", use_cache=False)
            assert "tenant_marker=D" in _echo(r.body)
        finally:
            await lad.close()

    async def test_tenant_fetch_starts_at_the_isolated_tiers(self, tmp_path, site):
        # Bug 136 (caught live by api.tenant_cookie_isolation after a
        # challenge rescue on the same host): the domain profile had LEARNED
        # to start at the browser tier, so every later read of that host --
        # every tenant's -- went through the one shared browser cookie store
        # even for plain pages the cheap tiers serve fine. A tenant-scoped
        # fetch now starts at tier 0 and climbs only when actually refused.
        from urllib.parse import urlsplit

        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=2, robots_policy="off",
                     sidecar_autostart=False)
        lad = Ladder(s)
        dom = urlsplit(site).netloc
        try:
            lad.profiles.record_block(dom, 0, "")
            lad.profiles.record_block(dom, 1, "")
            assert lad.profiles.get(dom).min_tier == 2  # the learned preference
            r = await lad.fetch(f"{site}/echo", use_cache=False, session="tenant-a")
            assert r.tier == 0, r.tier
        finally:
            await lad.close()

    async def test_cached_body_is_not_served_across_tenants(self, tmp_path, site):
        # Bug 138 (DeepSeek concurrency pass, confirmed by probe): the page
        # cache was keyed by URL alone, so a body fetched under tenant A's
        # cookies (A's private page) was served from cache to tenant B and
        # to the default tenant. A tenant-scoped fetch now caches under its
        # own key; the default tenant's shared cache is unchanged.
        s = Settings(state_dir=tmp_path, cache_enabled=True, max_tier=0, robots_policy="off",
                     sidecar_autostart=False)
        lad = Ladder(s)
        try:
            await lad.fetch(f"{site}/set?v=A", use_cache=False, session="tenant-a")
            ra = await lad.fetch(f"{site}/private", session="tenant-a")
            assert "tenant_marker=A" in _echo(ra.body)
            rb = await lad.fetch(f"{site}/private", session="tenant-b")
            assert "tenant_marker" not in _echo(rb.body), (rb.via, _echo(rb.body))
            rd = await lad.fetch(f"{site}/private")
            assert "tenant_marker" not in _echo(rd.body), (rd.via, _echo(rd.body))
            ra2 = await lad.fetch(f"{site}/private", session="tenant-a")
            assert ra2.via == "cache" and "tenant_marker=A" in _echo(ra2.body)
        finally:
            await lad.close()

    async def test_jars_are_bounded(self, lad, site):
        # Rider: a jar per session id from the network must not grow without
        # bound -- an agent minting a fresh id per call is the normal case.
        try:
            for i in range(300):
                await lad.fetch(f"{site}/set?v={i}", use_cache=False, session=f"s{i}")
            assert len(lad._jars) <= 256
        finally:
            await lad.close()
