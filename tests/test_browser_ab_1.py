"""Browser-system batch 1 (#189-#198): one class per knob.

Every knob is off by default, and each class pins BOTH arms: off = the
behaviour before the knob, on = the change. bench/ab.py measures them; these
only prove each arm does what its config comment says.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest

from searchio.config import Settings
from tests._ab_baseline import BASELINE
from searchio.errors import TransientError
from searchio.net import ladder as ladder_mod
from searchio.net.cache import Cache
from searchio.net.clearance import ClearanceStore, cookie_expiry, parent_scoped
from searchio.net.ladder import Ladder

GOOD = "<html><body><article><p>" + ("readable prose sentence here. " * 60) + "</p></article></body></html>"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,
        per_domain_burst=1000,
        sidecar_autostart=False,
        **BASELINE,
    )


class _Sidecar:
    """SidecarClient stand-in. ``jars`` is the sequence of cookie-jar reads."""

    available = True

    def __init__(self, jars=None, engine=False):
        self.calls: list[str] = []
        self.jars = list(jars or [[]])
        self.cookie_reads = 0
        if engine:
            self.binary = "se-serve"

    async def fetch(self, url, **kw):
        self.calls.append(url)
        return {"ok": True, "status": 200, "html": GOOD, "url": url}

    async def goto(self, url, **kw):
        return {"ok": True, "status": 200}

    async def cookies(self):
        i = min(self.cookie_reads, len(self.jars) - 1)
        self.cookie_reads += 1
        return self.jars[i]

    async def user_agent(self):
        return "BrowserUA/1.0"

    async def close_tab(self, *a, **kw):
        return None

    async def close(self):
        return None


class NoHttpLadder(Ladder):
    async def _tier0(self, url, *, referer=""):
        raise TransientError("no network in unit tests")

    async def _tier1(self, url, *, referer=""):
        raise TransientError("no network in unit tests")


@pytest.fixture
def sleeps(monkeypatch):
    """Record asyncio.sleep calls made by the ladder; never really wait."""
    seen: list[float] = []
    real = asyncio.sleep

    async def fake(d, *a, **kw):
        seen.append(d)
        await real(0)

    monkeypatch.setattr(ladder_mod.asyncio, "sleep", fake)
    return seen


CF = {"name": "cf_clearance", "value": "tok", "domain": ".bg.example"}


class TestClearanceCaptureBg:
    async def test_off_holds_the_page_for_the_settle(self, settings, sleeps):
        lad = NoHttpLadder(settings, sidecar=_Sidecar())
        try:
            await lad.fetch("https://bg.example/a", force_tier=2, use_cache=False)
            assert 6.0 in sleeps
        finally:
            await lad.close()

    async def test_on_engine_pass_skips_the_settle(self, settings, sleeps):
        settings.clearance_capture_bg = True
        settings.sidecar_engine = True
        sc = _Sidecar(engine=True)
        lad = NoHttpLadder(settings, sidecar=sc)
        try:
            res = await lad.fetch("https://bg.example/a", force_tier=2, use_cache=False)
            assert res.tier == 2
            assert 6.0 not in sleeps
            assert sc.cookie_reads == 1
            assert lad.stats()["tiers"].get("clearance_capture_settle_skipped") == 1
            assert not lad._bg
        finally:
            await lad.close()

    async def test_on_browser_pass_settles_in_the_background(self, settings, sleeps):
        settings.clearance_capture_bg = True
        sc = _Sidecar(jars=[[], [CF]])  # the cookie lands after load, as Akamai's does
        lad = NoHttpLadder(settings, sidecar=sc)
        try:
            await lad.fetch("https://bg.example/a", force_tier=2, use_cache=False)
            assert lad.clearance.get("bg.example") is None  # page returned first
            assert len(lad._bg) == 1
            await asyncio.gather(*list(lad._bg))
            assert 6.0 in sleeps
            assert lad.clearance.get("bg.example").cookies == {"cf_clearance": "tok"}
        finally:
            await lad.close()

    async def test_on_banks_inline_when_the_cookie_is_already_there(self, settings, sleeps):
        settings.clearance_capture_bg = True
        lad = NoHttpLadder(settings, sidecar=_Sidecar(jars=[[CF]]))
        try:
            await lad.fetch("https://bg.example/a", force_tier=2, use_cache=False)
            assert lad.clearance.get("bg.example") is not None
            assert 6.0 not in sleeps and not lad._bg
        finally:
            await lad.close()


class _Tier1Ladder(Ladder):
    """Tier 0 must not be asked; tier 1 answers; records what ran."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ran: list[int] = []

    async def _tier0(self, url, *, referer=""):
        self.ran.append(0)
        raise TransientError("no network in unit tests")

    async def _tier1(self, url, *, referer=""):
        self.ran.append(1)
        return 200, {}, GOOD, "text/html", url


class TestClearanceStartTier:
    def _pinned(self, lad, dom, cookies):
        lad.profiles.record_block(dom, 1, "cloudflare")  # min_tier -> 2
        lad.profiles.should_probe = lambda: False
        lad.clearance.put(dom, cookies, "BrowserUA/1.0")

    async def test_off_starts_at_the_browser(self, settings):
        sc = _Sidecar()
        lad = _Tier1Ladder(settings, sidecar=sc)
        self._pinned(lad, "st.example", {"cf_clearance": "tok"})
        try:
            res = await lad.fetch("https://st.example/p", use_cache=False)
            assert res.tier == 2 and lad.ran == []
        finally:
            await lad.close()

    async def test_on_starts_at_tier1_with_a_banked_challenge_cookie(self, settings):
        settings.clearance_start_tier = True
        sc = _Sidecar()
        lad = _Tier1Ladder(settings, sidecar=sc)
        self._pinned(lad, "st.example", {"cf_clearance": "tok"})
        try:
            res = await lad.fetch("https://st.example/p", use_cache=False)
            assert res.tier == 1 and lad.ran == [1] and sc.calls == []
            assert lad.stats()["tiers"].get("clearance_start_tier1") == 1
        finally:
            await lad.close()

    async def test_on_a_trust_only_row_does_not_lower_the_start(self, settings):
        settings.clearance_start_tier = True
        lad = _Tier1Ladder(settings, sidecar=_Sidecar())
        self._pinned(lad, "st.example", {"_pxvid": "v"})
        try:
            res = await lad.fetch("https://st.example/p", use_cache=False)
            assert res.tier == 2 and lad.ran == []
        finally:
            await lad.close()


class TestClearanceSkipTier0:
    async def test_withholds_only_challenge_rows_and_only_when_on(self, settings):
        lad = NoHttpLadder(settings, sidecar=_Sidecar())
        try:
            lad.clearance.put("w.example", {"cf_clearance": "tok"}, "UA")
            lad.clearance.put("t.example", {"_pxvid": "v"}, "UA")
            assert lad._tier0_withholds("w.example") is False
            lad.s.clearance_skip_tier0 = True
            assert lad._tier0_withholds("w.example") is True
            assert lad._tier0_withholds("t.example") is False
            assert lad._tier0_withholds("none.example") is False
        finally:
            await lad.close()


class TestClearanceCookieExpiry:
    def test_expiry_is_the_earliest_challenge_cookie(self):
        now = time.time()
        jar = [
            {"name": "cf_clearance", "value": "a", "domain": ".x.example", "expires": now + 7200},
            {"name": "__cf_bm", "value": "b", "domain": ".x.example", "expires": now + 1800},
            {"name": "_ga", "value": "c", "domain": ".x.example", "expires": now + 5},
            {"name": "datadome", "value": "d", "domain": ".other.example", "expires": now + 1},
        ]
        assert cookie_expiry(jar, "x.example") == pytest.approx(now + 1800)
        assert cookie_expiry([{"name": "cf_clearance", "domain": "x.example", "expires": -1}],
                             "x.example") == 0.0

    def test_off_uses_the_flat_ttl(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", ttl_s=1500)
        st.put("x.example", {"cf_clearance": "a"}, "UA", expires=time.time() + 7200)
        st._mem["x.example"].captured -= 2000  # older than the flat TTL
        assert st.get("x.example") is None

    def test_on_outlives_the_flat_ttl_until_the_cookie_expires(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", ttl_s=1500, max_ttl_s=86400)
        st.put("x.example", {"cf_clearance": "a"}, "UA", expires=time.time() + 7200)
        st._mem["x.example"].captured -= 2000
        assert st.get("x.example") is not None

    def test_on_a_short_cookie_dies_before_the_flat_ttl(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", ttl_s=1500, max_ttl_s=86400)
        st.put("x.example", {"cf_clearance": "a"}, "UA", expires=time.time() - 1)
        assert st.get("x.example") is None

    def test_on_caps_a_year_long_cookie(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", ttl_s=1500, max_ttl_s=3600)
        st.put("x.example", {"cf_clearance": "a"}, "UA", expires=time.time() + 3e7)
        st._mem["x.example"].captured -= 4000
        assert st.get("x.example") is None

    def test_expiry_survives_the_disk_and_an_old_schema_migrates(self, tmp_path):
        path = tmp_path / "c.sqlite3"
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE clearance (domain TEXT PRIMARY KEY, cookies TEXT NOT NULL, "
                   "user_agent TEXT NOT NULL DEFAULT '', captured REAL NOT NULL, "
                   "uses INTEGER NOT NULL DEFAULT 0)")
        db.execute("INSERT INTO clearance VALUES (?,?,?,?,0)",
                   ("old.example", json.dumps({"cf_clearance": "o"}), "UA", time.time()))
        db.commit()
        db.close()
        st = ClearanceStore(path, max_ttl_s=86400)
        assert st.errors == 0
        assert st.get("old.example").expires == 0.0
        exp = time.time() + 7200
        st.put("x.example", {"cf_clearance": "a"}, "UA", expires=exp)
        st.close()
        again = ClearanceStore(path, max_ttl_s=86400)
        assert again.get("x.example").expires == pytest.approx(exp)


class TestClearanceDomainScope:
    JAR = [
        {"name": "cf_clearance", "value": "wide", "domain": ".site.example"},
        {"name": "__cf_bm", "value": "host", "domain": "shop.site.example"},
    ]

    def test_parent_scoped_takes_only_dotted_parent_cookies(self):
        assert parent_scoped(self.JAR, "shop.site.example") == {
            "site.example": {"cf_clearance": "wide"}}
        # the host's own scope is not a parent; a public suffix never is
        assert parent_scoped([{"name": "cf_clearance", "value": "x", "domain": ".example"}],
                             "site.example") == {}

    def test_off_a_sibling_subdomain_misses(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3")
        st.put("shop.site.example", {"cf_clearance": "wide"}, "UA",
               scoped=parent_scoped(self.JAR, "shop.site.example"))
        assert st.get("blog.site.example") is None

    def test_on_a_sibling_subdomain_and_the_apex_hit(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", domain_scope=True)
        st.put("shop.site.example", {"cf_clearance": "wide", "__cf_bm": "host"}, "UA",
               scoped=parent_scoped(self.JAR, "shop.site.example"))
        assert st.get("blog.site.example").cookies == {"cf_clearance": "wide"}
        assert st.get("site.example").cookies == {"cf_clearance": "wide"}
        assert st.get("shop.site.example").cookies == {"cf_clearance": "wide", "__cf_bm": "host"}
        assert st.get("site.other") is None

    def test_on_a_block_on_the_sibling_drops_the_shared_row(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", domain_scope=True)
        st.put("shop.site.example", {"cf_clearance": "wide"}, "UA",
               scoped=parent_scoped(self.JAR, "shop.site.example"))
        st.record_use("blog.site.example")
        assert st._mem[".site.example"].uses == 1
        st.drop("blog.site.example")
        assert st.get("blog.site.example") is None
        assert st.get("shop.site.example") is not None  # its own row stays


class TestRobotsPersonaHeaders:
    class _Resp:
        status_code = 200
        charset_encoding = "utf-8"

        async def aiter_bytes(self, n):
            yield b"User-agent: *\nAllow: /\n"

        async def aclose(self):
            return None

    async def _run(self, settings):
        seen: dict = {}
        lad = NoHttpLadder(settings, sidecar=_Sidecar())

        async def open_stream(url, headers, timeout, **kw):
            seen.update({"headers": dict(headers)})
            return self._Resp()

        lad._open_stream = open_stream
        try:
            status, body = await lad._raw_get("https://r.example/robots.txt")
            assert status == 200 and "Allow" in body
        finally:
            await lad.close()
        return seen["headers"]

    async def test_off_sends_no_headers_of_its_own(self, settings):
        assert await self._run(settings) == {}

    async def test_on_sends_the_hosts_persona(self, settings):
        from searchio.net import persona

        settings.robots_persona_headers = True
        h = await self._run(settings)
        want = persona.for_domain("r.example", session="")
        assert "Mozilla/5.0" in h["User-Agent"] and "python-httpx" not in h["User-Agent"]
        assert h["Accept-Language"] == want.accept_language


class TestSidecarBootKnobs:
    async def test_the_ladder_sets_the_module_flag(self, settings):
        from searchio.net import sidecar as sc

        settings.sidecar_fast_boot = True
        lad = NoHttpLadder(settings, sidecar=_Sidecar())
        try:
            assert sc._FAST_BOOT[0] is True
        finally:
            await lad.close()
            sc._FAST_BOOT[0] = False

    async def test_warm_spawns_only_an_engine_and_only_when_on(self, settings):
        calls: list[str] = []

        class Eng(_Sidecar):
            async def ensure(self):
                calls.append("ensure")

        settings.sidecar_autostart = True
        lad = NoHttpLadder(settings, sidecar=Eng(engine=True))
        try:
            await lad.warm([])
            assert calls == []
            lad.s.warm_sidecar = True
            await lad.warm([])
            assert calls == ["ensure"]
        finally:
            await lad.close()


class TestCacheKnobs:
    BIG = {"body": "<p>" + "the same sentence again and again. " * 400 + "</p>", "n": 1}

    def _size(self, c):
        return c._db.execute("SELECT SUM(LENGTH(payload)) FROM entries").fetchone()[0]

    def test_off_stores_plain_json_text(self, tmp_path):
        c = Cache(tmp_path / "c.sqlite3")
        c.put("https://a.example/", self.BIG)
        (payload,) = c._db.execute("SELECT payload FROM entries").fetchone()
        assert isinstance(payload, str) and c.get("https://a.example/") == self.BIG

    def test_on_round_trips_smaller(self, tmp_path):
        plain = Cache(tmp_path / "p.sqlite3")
        plain.put("https://a.example/", self.BIG)
        z = Cache(tmp_path / "z.sqlite3", compress=True)
        z.put("https://a.example/", self.BIG)
        assert z.get("https://a.example/") == self.BIG
        assert self._size(z) * 5 < self._size(plain)

    def test_rows_cross_the_knob_both_ways(self, tmp_path):
        path = tmp_path / "c.sqlite3"
        plain = Cache(path)
        plain.put("https://old.example/", self.BIG)
        plain.close()
        z = Cache(path, compress=True)
        assert z.get("https://old.example/") == self.BIG
        z.put("https://new.example/", self.BIG)
        z.close()
        back = Cache(path)  # knob off again: compressed rows still read
        assert back.get("https://new.example/") == self.BIG

    def test_purge_on_open_removes_expired_rows(self, tmp_path):
        path = tmp_path / "c.sqlite3"
        c = Cache(path, ttl_s=60)
        c.put("https://old.example/", {"a": 1})
        c.put("https://new.example/", {"a": 2})
        c._db.execute("UPDATE entries SET created = created - 3600 WHERE url LIKE '%old%'")
        c._db.commit()
        c.close()
        assert Cache(path, ttl_s=60).stats()["entries"] == 2
        assert Cache(path, ttl_s=60, purge_on_open=True).stats()["entries"] == 1

    def test_max_mb_evicts_the_oldest_first(self, tmp_path):
        c = Cache(tmp_path / "c.sqlite3", max_mb=1)
        blob = {"body": "x" * 200_000}
        for i in range(8):  # 1.6 MB of payload
            c.put(f"https://e.example/{i}", blob)
            c._db.execute("UPDATE entries SET created = ? WHERE url = ?",
                          (1000.0 + i, f"https://e.example/{i}"))
        c._db.commit()
        c.ttl_s = 10**12
        assert c._evict() > 0
        assert self._size(c) <= 1024 * 1024
        assert c.get("https://e.example/0") is None
        assert c.get("https://e.example/7") == blob
