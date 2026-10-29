"""Fan-out speed knobs (bench/ab.py A/B-tests each one live).

Every knob is off by default; these tests pin what each one does when on,
and that the off position is the old behaviour.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from searchio.config import Settings
from searchio.models import Capability, Doc, FetchResult, Query
from searchio.net import ladder as ladder_mod
from searchio.net.robots import RobotsCache
from searchio.providers.base import Provider
from searchio.providers.engines import run_hedged
from searchio.providers.registry import Registry
from searchio.router import Router


class Timed(Provider):
    """Answers after ``delay`` seconds, counting calls."""

    def __init__(self, name: str, delay: float, docs: int = 2, fail: bool = False):
        super().__init__()
        self.name = name
        self.caps = frozenset({Capability.WEB})
        self.delay = delay
        self.n_docs = docs
        self.fail = fail
        self.calls = 0

    async def search(self, q, ctx):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("down")
        return [Doc(url=f"https://{self.name}.com/{i}", title=f"{self.name} {i}",
                    source=self.name) for i in range(self.n_docs)]


def _router(tmp_path, providers, **kw):
    # Deadline tests use thin (2-doc) answers: the doc floor is off unless asked.
    kw.setdefault("deadline_min_docs", 0)
    s = Settings(state_dir=tmp_path, cache_enabled=False, **kw)
    return Router(object(), settings=s, registry=Registry(providers))


Q = Query(text="x", intent="web", k=10)


class TestRunHedged:
    @staticmethod
    def attempt(delay, ok, tag, log):
        async def run():
            log.append(f"start:{tag}")
            await asyncio.sleep(delay)
            return ([Doc(url=f"https://{tag}.com", source=tag)], "") if ok else (None, f"err:{tag}")
        return run

    async def test_off_is_sequential(self):
        log: list[str] = []
        docs, errs = await run_hedged(
            [self.attempt(0.05, False, "a", log), self.attempt(0.0, True, "b", log)], 0)
        assert docs[0].source == "b" and errs == ["err:a"]
        assert log == ["start:a", "start:b"]

    async def test_slow_primary_is_hedged(self):
        log: list[str] = []
        t0 = time.monotonic()
        docs, errs = await run_hedged(
            [self.attempt(1.0, True, "a", log), self.attempt(0.0, True, "b", log)], 50)
        assert docs[0].source == "b"
        assert time.monotonic() - t0 < 0.5, "the hedge answered; the slow primary was not awaited"

    async def test_fast_primary_never_starts_the_hedge(self):
        log: list[str] = []
        docs, _ = await run_hedged(
            [self.attempt(0.0, True, "a", log), self.attempt(0.0, True, "b", log)], 200)
        assert docs[0].source == "a" and log == ["start:a"]

    async def test_failed_primary_falls_back_at_once_and_errors_keep_order(self):
        log: list[str] = []
        t0 = time.monotonic()
        docs, errs = await run_hedged(
            [self.attempt(0.0, False, "a", log), self.attempt(0.0, False, "b", log)], 5000)
        assert docs is None and errs == ["err:a", "err:b"]
        assert time.monotonic() - t0 < 1.0, "a failure does not wait out the hedge timer"


class TestEarlyReturn:
    async def test_off_waits_for_everyone(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("bing", 0.3)])
        res = await r.search(Q)
        assert set(res.used) == {"duckduckgo", "bing"} and res.late == []

    async def test_quorum_then_grace_returns_without_the_straggler(self, tmp_path):
        slow = Timed("bing", 1.0)
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("wikipedia", 0.0), slow],
                    fanout_grace_ms=50, fanout_quorum=2)
        t0 = time.monotonic()
        res = await r.search(Q)
        assert time.monotonic() - t0 < 0.6
        assert set(res.used) == {"duckduckgo", "wikipedia"} and res.late == ["bing"]
        assert "bing" not in res.failed, "late is not failed"
        # The straggler finishes detached and its health still lands.
        await r.drain()
        assert slow.health.calls == 1 and slow.health.consecutive_failures == 0

    async def test_failures_do_not_count_toward_the_quorum(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.0, fail=True), Timed("bing", 0.2)],
                    fanout_grace_ms=10, fanout_quorum=1)
        res = await r.search(Q)
        assert res.used == ["bing"], "one failure must not trigger the early return"


class TestSingleflightAndQueryCache:
    async def test_identical_concurrent_searches_share_one_fanout(self, tmp_path):
        p = Timed("duckduckgo", 0.05)
        r = _router(tmp_path, [p], singleflight=True)
        a, b = await asyncio.gather(r.search(Q), r.search(Q))
        assert p.calls == 1
        assert [d.url for d in a.docs] == [d.url for d in b.docs]
        a.docs[0].title = "mutated"
        assert b.docs[0].title != "mutated", "each caller owns its copy"

    async def test_query_cache_answers_a_repeat_and_fresh_bypasses(self, tmp_path):
        p = Timed("duckduckgo", 0.0)
        r = _router(tmp_path, [p], query_cache_ttl_s=60)
        await r.search(Q)
        await r.search(Q)
        assert p.calls == 1
        await r.search(Q, fresh=True)
        assert p.calls == 2

    async def test_empty_answers_are_not_cached(self, tmp_path):
        p = Timed("duckduckgo", 0.0, docs=0)
        r = _router(tmp_path, [p], query_cache_ttl_s=60)
        await r.search(Q)
        await r.search(Q)
        assert p.calls == 2


class TestRobots:
    async def test_persisted_answer_skips_the_fetch_in_a_new_process(self, tmp_path):
        calls: list[str] = []

        async def fetch(url):
            calls.append(url)
            return 200, "User-agent: *\nDisallow: /private\n"

        path = tmp_path / "robots.sqlite"
        first = RobotsCache(fetch, store_path=path)
        assert (await first.check("https://a.com/private/x")).allowed is False
        first.close()
        second = RobotsCache(fetch, store_path=path)
        info = await second.check("https://a.com/private/y")
        assert info.allowed is False and info.checked
        assert calls == ["https://a.com/robots.txt"], "the second cache read the store"
        second.close()

    async def test_unreadable_robots_is_not_persisted(self, tmp_path):
        async def fetch(url):
            return 503, ""

        path = tmp_path / "robots.sqlite"
        c = RobotsCache(fetch, store_path=path)
        await c.check("https://a.com/")
        c.close()
        assert RobotsCache(fetch, store_path=path)._from_store("https://a.com") is None

    async def test_nonblocking_warn_does_not_wait_and_learns_for_next_time(self):
        gate = asyncio.Event()

        async def fetch(url):
            await gate.wait()
            return 200, "User-agent: *\nDisallow: /p\n"

        c = RobotsCache(fetch, policy="warn", nonblocking=True)
        info = await asyncio.wait_for(c.check("https://a.com/p"), 0.5)
        assert info.allowed and info.reason == "robots_pending"
        gate.set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert (await c.check("https://a.com/p")).allowed is False

    async def test_enforce_always_waits(self):
        async def fetch(url):
            await asyncio.sleep(0.01)
            return 200, "User-agent: *\nDisallow: /p\n"

        c = RobotsCache(fetch, policy="enforce", nonblocking=True)
        assert (await c.check("https://a.com/p")).allowed is False


class TestLadderSingleflightAndDns:
    async def test_identical_concurrent_fetches_share_one_climb(self, tmp_path, monkeypatch):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, cache_enabled=False,
                                         singleflight=True, sidecar_autostart=False))
        n = {"calls": 0}

        async def fake(self, url, **kw):
            n["calls"] += 1
            await asyncio.sleep(0.02)
            return FetchResult(url=url, final_url=url, status=200, body="ok")

        monkeypatch.setattr(ladder_mod.Ladder, "_fetch", fake)
        a, b = await asyncio.gather(lad.fetch("https://a.com/"), lad.fetch("https://a.com/"))
        assert n["calls"] == 1 and a.body == b.body == "ok"
        await lad.close()

    async def test_a_cancelled_leader_does_not_cancel_its_waiter(self, tmp_path, monkeypatch):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, cache_enabled=False,
                                         singleflight=True, sidecar_autostart=False))

        async def fake(self, url, **kw):
            await asyncio.sleep(0.05)
            return FetchResult(url=url, final_url=url, status=200, body="ok")

        monkeypatch.setattr(ladder_mod.Ladder, "_fetch", fake)
        leader = asyncio.ensure_future(lad.fetch("https://a.com/"))
        await asyncio.sleep(0)
        waiter = asyncio.ensure_future(lad.fetch("https://a.com/"))
        await asyncio.sleep(0.01)
        leader.cancel()
        assert (await waiter).body == "ok"
        await lad.close()

    def test_dns_cache_reuses_only_vetted_answers(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def gai(host, port, *a):
            calls["n"] += 1
            return [(2, 1, 6, "", ("93.184.216.34", port))]

        monkeypatch.setattr(ladder_mod.socket, "getaddrinfo", gai)
        monkeypatch.setattr(ladder_mod, "_DNS_TTL", [60.0])
        monkeypatch.setattr(ladder_mod, "_DNS_CACHE", {})
        ladder_mod._resolve_checked("example.com", 443)
        ladder_mod._resolve_checked("example.com", 443)
        assert calls["n"] == 1

        def poisoned(host, port, *a):
            return [(2, 1, 6, "", ("169.254.169.254", port))]

        monkeypatch.setattr(ladder_mod.socket, "getaddrinfo", poisoned)
        with pytest.raises(Exception):
            ladder_mod._resolve_checked("evil.example", 443)
        assert ("evil.example", 443) not in ladder_mod._DNS_CACHE


class _Walled(Provider):
    def __init__(self, name="duckduckgo"):
        super().__init__()
        self.name = name
        self.caps = frozenset({Capability.WEB})
        self.calls = 0

    async def search(self, q, ctx):
        from searchio.errors import ProviderError

        self.calls += 1
        raise ProviderError(self.name, "a: blocked(None); b: blocked(None)", walled=True)


class TestWallsAndDegradation:
    async def test_fast_trip_opens_the_breaker_on_the_first_wall(self, tmp_path):
        w = _Walled()
        r = _router(tmp_path, [w, Timed("bing", 0.0)], fast_trip_on_block=True,
                    query_cache_ttl_s=0)
        await r.search(Q)
        await r.search(Q)
        assert w.calls == 1 and w.health.open

    async def test_off_needs_three_strikes(self, tmp_path):
        w = _Walled()
        r = _router(tmp_path, [w, Timed("bing", 0.0)], fast_trip_on_block=False,
                    query_cache_ttl_s=0)
        for _ in range(3):
            await r.search(Q)
        assert w.calls == 3 and w.health.open

    async def test_sibling_skip_stops_after_a_wall(self):
        log: list[str] = []
        a = TestRunHedged.attempt
        docs, errs = await run_hedged(
            [lambda: _ret(None, "html: blocked(None)", log, "a"), a(0.0, True, "b", log)],
            0, stop=lambda e: "blocked(" in e)
        assert docs is None and log == ["start:a"]

    async def test_bing_skips_html_after_learning_it_is_degraded_too(self, tmp_path, monkeypatch):
        from searchio.providers import engines as eng
        from searchio.providers.base import ProviderContext

        fetched: list[str] = []

        class L:
            async def fetch(self, url, **kw):
                fetched.append("rss" if "format=rss" in url else "html")
                return FetchResult(url=url, final_url=url, status=200, body="")

        monkeypatch.setattr(eng, "parse_bing_rss", lambda b, k: [("u", "t", "s", "")] * 3)
        monkeypatch.setattr(eng, "parse_bing", lambda b, k: [("u", "t", "s", "")] * 3)
        monkeypatch.setattr(eng, "looks_degraded", lambda q, rows: True)
        monkeypatch.setattr(eng.random, "randrange", lambda n: 1)  # never probe
        b = eng.Bing()
        ctx = ProviderContext(ladder=L(), settings=Settings(state_dir=tmp_path, degraded_skip=True))
        for _ in range(5):
            with pytest.raises(Exception):
                await b.search(Query(text="zzz"), ctx)
        assert fetched == ["rss", "html"] * 3 + ["rss", "rss"]

    async def test_adaptive_timeout_tracks_latency(self, tmp_path):
        p = Timed("duckduckgo", 0.0)
        r = _router(tmp_path, [p], adaptive_timeout=True, provider_timeout_s=12.0)
        assert r._timeout_for(p) == 12.0, "no history: the configured ceiling"
        for _ in range(5):
            p.health.record(True, 300.0)
        assert 1.9 < r._timeout_for(p) < 3.0


async def _ret(docs, err, log, tag):
    log.append(f"start:{tag}")
    return docs, err


class TestLocalThrottle:
    async def test_bounded_acquire_fails_fast_instead_of_sleeping(self):
        from searchio.errors import Throttled
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=0.75, burst=1)
        await lim.acquire("d.com")
        lim.record_block("d.com")
        t0 = time.monotonic()
        with pytest.raises(Throttled):
            await lim.acquire("d.com", 0.5)
        assert time.monotonic() - t0 < 0.2, "nothing slept: the bound refused at once"

    async def test_unbounded_acquire_still_waits(self):
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=50, burst=1)
        await lim.acquire("d.com")
        await asyncio.wait_for(lim.acquire("d.com"), 1.0)

    async def test_router_bounds_only_provider_calls(self, tmp_path):
        from searchio.net.ladder import MAX_LIMITER_WAIT

        seen = []

        class Peek(Timed):
            async def search(self, q, ctx):
                seen.append(MAX_LIMITER_WAIT.get())
                return await super().search(q, ctx)

        r = _router(tmp_path, [Peek("duckduckgo", 0.0)], serp_max_wait_s=1.5)
        await r.search(Q)
        assert seen == [1.5] and MAX_LIMITER_WAIT.get() is None


class TestBatch3:
    async def test_health_survives_a_new_router(self, tmp_path):
        w = _Walled()
        r1 = _router(tmp_path, [w, Timed("bing", 0.0)], persist_health=True,
                     query_cache_ttl_s=0)
        await r1.search(Q)
        assert w.health.open
        r1._health.close()
        fresh = _Walled()
        r2 = _router(tmp_path, [fresh, Timed("bing", 0.0)], persist_health=True,
                     query_cache_ttl_s=0)
        assert fresh.health.open, "the breaker came back open in the new process"
        await r2.search(Q)
        assert fresh.calls == 0
        r2._health.close()

    async def test_normalized_keys_share_one_answer(self, tmp_path):
        p = Timed("duckduckgo", 0.0)
        r = _router(tmp_path, [p], normalize_query_keys=True)
        await r.search(Query(text="Rust  Async", intent="web"))
        await r.search(Query(text="rust async", intent="web"))
        assert p.calls == 1

    async def test_backup_provider_starts_when_the_fanout_is_slow(self, tmp_path):
        slow = [Timed("duckduckgo", 1.0), Timed("bing", 1.0)]
        spare = Timed("wikipedia", 0.0)
        r = _router(tmp_path, slow + [spare], fanout=2, backup_provider_ms=50,
                    fanout_grace_ms=50, fanout_quorum=1)
        t0 = time.monotonic()
        res = await r.search(Q)
        assert "wikipedia" in res.used and time.monotonic() - t0 < 0.6
        await r.drain()

    async def test_backup_stays_idle_when_the_fanout_answered(self, tmp_path):
        spare = Timed("wikipedia", 0.0)
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("bing", 0.0), spare],
                    fanout=2, backup_provider_ms=200)
        await r.search(Q)
        assert spare.calls == 0

    async def test_stale_serp_answers_and_refreshes(self, tmp_path, monkeypatch):
        from searchio.net.ladder import SERP_SWR

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, sidecar_autostart=False,
                                         robots_policy="off", cache_ttl_s=1))
        calls = {"n": 0}

        async def tier(self, tier, url, **kw):
            calls["n"] += 1
            return 200, {"content-type": "text/html"}, "<html><body>" + "word " * 60 + f"v{calls['n']}</body></html>", "text/html", url

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        await lad.fetch("https://s.com/q")
        lad.cache._db.execute("UPDATE entries SET created = created - 5")
        lad.cache._db.commit()
        tok = SERP_SWR.set(3600.0)
        try:
            res = await lad.fetch("https://s.com/q")
        finally:
            SERP_SWR.reset(tok)
        assert res.via == "cache-stale" and "v1" in res.body
        for _ in range(50):
            if not lad._bg:
                break
            await asyncio.sleep(0.01)
        assert calls["n"] == 2
        assert "v2" in (await lad.fetch("https://s.com/q")).body
        await lad.close()

    async def test_endpoint_learning_leads_with_the_last_winner(self):
        from searchio.providers.engines import _EndpointOrder

        o = _EndpointOrder()
        log: list[int] = []

        def mk(i, ok):
            async def run():
                log.append(i)
                return ([Doc(url=f"https://{i}.com")], "") if ok else (None, "x")
            return run

        await run_hedged(o.wrap(o.order(2, True), [mk(0, False), mk(1, True)]), 0)
        assert o.lead == 1
        log.clear()
        await run_hedged(o.wrap(o.order(2, True), [mk(0, True), mk(1, True)]), 0)
        assert log == [1], "the fallback that answered now leads"


class TestSoftDeadline:
    async def test_a_slow_provider_is_left_behind_after_the_deadline(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("bing", 0.05), Timed("arxiv", 2.0)],
                    fanout_deadline_ms=200)
        t0 = time.monotonic()
        res = await r.search(Q)
        assert time.monotonic() - t0 < 0.8
        assert set(res.used) == {"duckduckgo", "bing"}, "everything inside the deadline is kept"
        assert res.late == ["arxiv"]
        await r.drain()

    async def test_no_answer_yet_keeps_waiting(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.3), Timed("bing", 0.3)],
                    fanout_deadline_ms=50)
        res = await r.search(Q)
        assert res.used, "the deadline never returns empty-handed while providers run"


class TestStreaming:
    async def test_first_yield_arrives_with_the_fastest_provider(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("bing", 0.6)])
        t0 = time.monotonic()
        parts = []
        async for part in r.search_stream(Q):
            parts.append((time.monotonic() - t0, part))
        first_t, first = parts[0]
        assert first_t < 0.4 and first.used == ["duckduckgo"] and first.late == ["bing"]
        assert set(parts[-1][1].used) == {"duckduckgo", "bing"} and parts[-1][1].late == []

    async def test_last_yield_matches_search(self, tmp_path):
        provs = [Timed("duckduckgo", 0.0), Timed("bing", 0.02)]
        r = _router(tmp_path, provs, query_cache_ttl_s=0, fanout_deadline_ms=0)
        last = None
        async for part in r.search_stream(Q):
            last = part
        full = await r.search(Q)
        assert [d.url for d in last.docs] == [d.url for d in full.docs]


class TestParseOffloop:
    async def test_large_body_parses_off_the_loop_thread(self):
        import threading
        from types import SimpleNamespace

        from searchio.providers.engines import _parse

        seen = []

        def parser(body, k):
            seen.append(threading.current_thread() is threading.main_thread())
            return [k]

        on = SimpleNamespace(settings=SimpleNamespace(parse_offloop=True))
        off = SimpleNamespace(settings=SimpleNamespace(parse_offloop=False))
        big = "x" * 30_000
        assert await _parse(on, parser, big, 3) == [3]
        assert await _parse(on, parser, "small", 3) == [3]
        assert await _parse(off, parser, big, 3) == [3]
        assert seen == [False, True, True]


class TestWriteBehind:
    async def test_write_lands_and_close_waits_for_it(self, tmp_path, monkeypatch):
        s = Settings(state_dir=tmp_path, max_tier=0, robots_policy="off",
                     cache_write_behind=True)
        lad = ladder_mod.Ladder(s)

        async def tier(self, tier, url, **kw):
            return 200, {"content-type": "text/html"}, "<html><body>" + "word " * 60 + "</body></html>", "text/html", url

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        res = await lad.fetch("https://wb.com/a")
        assert res.tier == 0
        await lad.close()
        from searchio.net.cache import Cache

        c = Cache(s.cache_path(), s.cache_ttl_s, True)
        assert c.get("https://wb.com/a", lad._cache_variant(False)) is not None
        c.close()


class TestPersistRates:
    def test_learned_rate_survives_a_restart_and_recovers(self, tmp_path):
        from searchio.net import ratelimit
        from searchio.net.ratelimit import DomainLimiter, RateStore

        a = DomainLimiter(rps=0.75)
        a.record_block("duckduckgo.com")
        a.record_block("duckduckgo.com")
        a.record_success("ok.com")
        store = RateStore(tmp_path / "rates.sqlite")
        store.save(a)
        store.close()
        assert set(a.learned()) == {"duckduckgo.com"}, "only hosts below the default persist"

        b = DomainLimiter(rps=0.75)
        assert RateStore(tmp_path / "rates.sqlite").load_into(b) == 1
        assert b.rate_for("duckduckgo.com") < 0.75
        assert b.rate_for("other.com") == 0.75

        # An hour later the host is back at the default.
        db = RateStore(tmp_path / "rates.sqlite")
        db._db.execute("UPDATE rates SET at = at - 3600")
        db._db.commit()
        c = DomainLimiter(rps=0.75)
        assert db.load_into(c) == 0 and c.rate_for("duckduckgo.com") == 0.75
        assert ratelimit.RECOVER_RPS_PER_S > 0

    async def test_seeded_host_fails_fast_under_max_wait(self):
        from searchio.errors import Throttled
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=0.75)
        lim.seed("duckduckgo.com", 0.1)
        with pytest.raises(Throttled):
            await lim.acquire("duckduckgo.com", max_wait=1.5)


class TestPrefetch:
    async def test_top_hits_are_fetched_in_the_background(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=2))
        fetched: list[str] = []

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url=f"https://h{i}.com/") for i in range(4)])

        async def fake_fetch(url, **kw):
            fetched.append(url)

        monkeypatch.setattr(eng.router, "search", fake_search)
        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        await eng.search("x")
        await eng.router.drain()
        assert fetched == ["https://h0.com/", "https://h1.com/"]
        assert not eng.router._background
        await eng.close()

    async def test_off_fetches_nothing(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=0))
        fetched: list[str] = []

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url="https://h.com/")])

        async def fake_fetch(url, **kw):
            fetched.append(url)

        monkeypatch.setattr(eng.router, "search", fake_search)
        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        await eng.search("x")
        await eng.router.drain()
        assert fetched == []
        await eng.close()


def test_fast_loop_runs_a_coroutine_with_or_without_the_package():
    from searchio._loop import run

    async def f():
        await asyncio.sleep(0)
        return 7

    assert run(f(), fast=True) == 7
    assert run(f(), fast=False) == 7


# ── batch 5 ──────────────────────────────────────────────────────────────────


class _UrlLadder:
    """Records fetched URLs; answers each with a canned body."""

    def __init__(self, body: str):
        self.body = body
        self.urls: list[str] = []

    async def fetch(self, url, **kw):
        from types import SimpleNamespace

        self.urls.append(url)
        return SimpleNamespace(body=self.body, final_url=url, status=200)


def _ctx(ladder, **kw):
    from searchio.providers.base import ProviderContext

    return ProviderContext(ladder=ladder, settings=Settings(cache_enabled=False, **kw))


class TestBatch5Urls:
    async def test_arxiv_https_skips_the_redirect(self):
        from searchio.providers.apis import ArxivProvider

        feed = "<feed><entry><id>http://arxiv.org/abs/1</id><title>T</title></entry></feed>"
        for on, scheme in ((False, "http://"), (True, "https://")):
            lad = _UrlLadder(feed)
            await ArxivProvider().search(Q, _ctx(lad, arxiv_https=on))
            assert lad.urls[0].startswith(scheme + "export.arxiv.org/")

    async def test_lean_fields_select_only_what_the_parsers_read(self):
        from searchio.providers.apis import Crossref, HackerNews, OpenAlex

        cases = [
            (OpenAlex, '{"results": []}', "&select=id,doi,title"),
            (Crossref, '{"status": "ok", "message": {"items": []}}', "&select=URL,title"),
            (HackerNews, '{"hits": []}', "&attributesToRetrieve="),
        ]
        for prov, body, marker in cases:
            off, on = _UrlLadder(body), _UrlLadder(body)
            await prov().search(Q, _ctx(off))
            await prov().search(Q, _ctx(on, lean_api_fields=True))
            assert marker not in off.urls[0] and marker in on.urls[0], prov.name


class TestMarginalia:
    async def test_opt_in_and_parse(self):
        from searchio.providers.apis import Marginalia

        assert not Marginalia().configured(Settings())
        assert Marginalia().configured(Settings(marginalia_enabled=True))
        lad = _UrlLadder('{"results": [{"url": "https://a.org/x", "title": "A &amp; B",'
                         ' "description": "<b>d</b>"}, {"url": "", "title": "no url"}]}')
        docs = await Marginalia().search(Query(text="repot a monstera", intent="web", k=5),
                                         _ctx(lad, marginalia_enabled=True))
        assert [(d.url, d.title, d.snippet) for d in docs] == [("https://a.org/x", "A & B", "d")]
        assert lad.urls[0].startswith("https://api.marginalia.nu/public/search/repot%20a%20monstera")

    async def test_refusal_envelope_is_an_error(self):
        from searchio.errors import ProviderError
        from searchio.providers.apis import Marginalia

        with pytest.raises(ProviderError):
            await Marginalia().search(Q, _ctx(_UrlLadder('{"error": "rate limited"}')))


class TestBreakerBackoff:
    def test_cooldown_doubles_per_trip_and_resets_on_success(self):
        from searchio.providers.base import Health

        h = Health(backoff_base_s=15.0)
        seen = []
        for _ in range(5):
            h.trip()
            seen.append(h.cooldown_s)
        assert seen == [15.0, 30.0, 60.0, 120.0, 120.0]
        h.record(True, 100)
        assert h.trips == 0 and not h.open
        assert Health().cooldown_s == Health.COOLDOWN_S, "off = the fixed cooldown"

    def test_short_cooldown_reopens_for_a_probe_sooner(self):
        from searchio.providers.base import Health

        h = Health(backoff_base_s=15.0)
        h.trip()
        h.opened_at -= 16
        assert not h.open, "past the 15 s first cooldown: half-open"
        legacy = Health()
        legacy.trip()
        legacy.opened_at -= 16
        assert legacy.open

    async def test_router_hands_the_base_to_every_provider_and_fast_trip_counts(self, tmp_path):
        from searchio.errors import ProviderError

        class Walled(Timed):
            async def search(self, q, ctx):
                raise ProviderError(self.name, "blocked(widget)", walled=True)

        w = Walled("duckduckgo", 0)
        r = _router(tmp_path, [w, Timed("bing", 0)], breaker_backoff_s=15.0,
                    query_cache_ttl_s=0)
        assert w.health.backoff_base_s == 15.0
        await r.search(Q)
        assert w.health.open and w.health.trips == 1 and w.health.cooldown_s == 15.0


class TestStarvedProbe:
    def _starved(self, tmp_path, **kw):
        ddg, bing, wiki = Timed("duckduckgo", 0), Timed("bing", 0), Timed("wikipedia", 0)
        kw.setdefault("breaker_backoff_s", 0.0)  # the fixed 120 s: both still cooling
        kw.setdefault("starved_probe_s", 0.0)
        r = _router(tmp_path, [ddg, bing, wiki], **kw)
        now = time.monotonic()
        for p, ago in ((ddg, 40.0), (bing, 20.0)):
            p.health.consecutive_failures = 3
            p.health.opened_at = now - ago
        return r, ddg, bing

    def test_off_fans_out_without_a_generalist(self, tmp_path):
        r, _, _ = self._starved(tmp_path)
        assert [p.name for p in r.select(Q)] == ["wikipedia"]

    def test_longest_cooling_generalist_is_let_through_once_per_interval(self, tmp_path):
        r, ddg, _ = self._starved(tmp_path, starved_probe_s=10.0)
        assert "duckduckgo" in [p.name for p in r.select(Q)]
        assert ddg.health.last_probe > 0
        assert "bing" in [p.name for p in r.select(Q)], "ddg was just probed; bing is next"
        third = {p.name for p in r.select(Q)}
        assert not {"duckduckgo", "bing"} & third, "both probed inside the interval"

    def test_a_fresh_trip_is_not_probed(self, tmp_path):
        r, _, _ = self._starved(tmp_path, starved_probe_s=60.0)
        assert [p.name for p in r.select(Q)] == ["wikipedia"]


class TestSerpTierHints:
    def test_unseen_hinted_host_starts_at_its_tier(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, serp_tier_hints=True))
        assert lad._hinted_start("html.duckduckgo.com", 0, 1) == 1
        assert lad._hinted_start("html.duckduckgo.com", 0, 0) == 0, "never above the ceiling"
        assert lad._hinted_start("example.com", 0, 1) == 0
        lad.profiles.record_success("lite.duckduckgo.com", 0)
        assert lad._hinted_start("lite.duckduckgo.com", 0, 1) == 0, "the profile's own answer rules"

    async def test_fetch_starts_at_the_hint(self, tmp_path, monkeypatch):
        tiers = []

        async def tier(self, t, url, **kw):
            tiers.append(t)
            body = "<html><body>" + "word " * 60 + "</body></html>"
            return 200, {"content-type": "text/html"}, body, "text/html", url

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        for on, want in ((False, 0), (True, 1)):
            tiers.clear()
            lad = ladder_mod.Ladder(Settings(state_dir=tmp_path / str(on), serp_tier_hints=on,
                                             cache_enabled=False, robots_policy="off", max_tier=1))
            await lad.fetch("https://html.duckduckgo.com/html/?q=x")
            assert tiers[0] == want
            await lad.close()


class TestOffloopImports:
    async def test_preimport_runs_on_a_worker(self, tmp_path):
        import sys

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, offloop_imports=True))
        assert lad._preimport is not None
        await asyncio.wrap_future(lad._preimport)
        assert "selectolax.parser" in sys.modules
        await lad.close()

    async def test_off_submits_nothing(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path))
        assert lad._preimport is None
        await lad.close()


@pytest.mark.parametrize("on", [False, True])
def test_lean_imports_skips_httpx_cli_chain(on):
    import os
    import subprocess
    import sys

    env = {**os.environ, "SEARCHIO_LEAN_IMPORTS": "true" if on else "false"}
    code = ("import sys, searchio, httpx; m = sys.modules.get('httpx._main'); "
            "print(m is not None, 'click' in sys.modules, callable(httpx.main))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout.split()
    loaded, click, has_main = out
    assert has_main == "True", "httpx.main stays callable either way"
    assert (loaded, click) == (("False", "False") if on else ("True", "True"))


# ── batch 6 ──────────────────────────────────────────────────────────────────


class TestRateLimitWait:
    def test_shapes(self):
        from searchio.net.ladder import _rate_limit_wait

        now = time.time()
        assert _rate_limit_wait(429, {"Retry-After": "30"}) == 30
        assert _rate_limit_wait(429, {}) == 60
        assert 15 < _rate_limit_wait(403, {"x-ratelimit-remaining": "0",
                                           "x-ratelimit-reset": str(now + 20)}) <= 20
        assert _rate_limit_wait(403, {}) is None, "a bare 403 is a wall, not a quota"
        assert _rate_limit_wait(403, {"x-ratelimit-remaining": "3"}) is None
        assert _rate_limit_wait(200, {"Retry-After": "5"}) is None
        assert _rate_limit_wait(429, {"Retry-After": "99999"}) == 3600, "clamped"

    async def test_quota_403_ends_the_climb_without_pinning_the_profile(self, tmp_path, monkeypatch):
        from searchio.errors import RateLimited

        tiers = []

        async def tier(self, t, url, **kw):
            tiers.append(t)
            return (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(time.time() + 30),
                          "content-type": "application/json"},
                    '{"message": "API rate limit exceeded"}', "application/json", url)

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, ratelimit_aware=True, cache_enabled=False,
                                         robots_policy="off", max_tier=1))
        with pytest.raises(RateLimited) as ei:
            await lad.fetch("https://api.github.com/search/repositories?q=x")
        assert tiers == [0], "tier 1 draws the same per-IP quota"
        assert 20 < ei.value.retry_after <= 30
        assert lad.profiles.get("api.github.com").min_tier == 0
        await lad.close()

    async def test_router_holds_for_the_stated_time(self, tmp_path):
        from searchio.errors import RateLimited

        class Limited(Timed):
            async def search(self, q, ctx):
                raise RateLimited("rate_limited", retry_after=42)

        p = Limited("github", 0)
        r = _router(tmp_path, [p, Timed("duckduckgo", 0)], ratelimit_aware=True,
                    query_cache_ttl_s=0)
        await r.search(Q)
        assert p.health.open and p.health.cooldown_s == 42

    async def test_stackexchange_backoff_is_honored_without_a_request(self):
        from searchio.errors import RateLimited
        from searchio.providers.apis import StackExchange

        lad = _UrlLadder('{"items": [], "backoff": 10, "quota_remaining": 250}')
        se = StackExchange()
        await se.search(Q, _ctx(lad, ratelimit_aware=True))
        with pytest.raises(RateLimited):
            await se.search(Q, _ctx(lad, ratelimit_aware=True))
        assert len(lad.urls) == 1, "the paused call sent nothing"


class TestApiRateTable:
    async def test_documented_rates_seed_the_limiter(self, tmp_path):
        from searchio.net.ladder import API_RATES

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, api_rate_table=True))
        assert lad.limiter.rate_for("en.wikipedia.org") == API_RATES["en.wikipedia.org"][0]
        arxiv = API_RATES["export.arxiv.org"][0]
        assert lad.limiter.rate_for("export.arxiv.org") == arxiv
        for _ in range(20):
            lad.limiter.record_success("export.arxiv.org")
        assert lad.limiter.rate_for("export.arxiv.org") == arxiv, "never probed past arXiv's own rate"
        await lad.close()


class TestQueryCacheSwr:
    async def test_stale_entry_answers_now_and_refreshes_once(self, tmp_path):
        p = Timed("duckduckgo", 0.0)
        r = _router(tmp_path, [p], query_cache_ttl_s=60, query_cache_swr_s=600)
        await r.search(Q)
        key = next(iter(r._qcache))
        exp, res = r._qcache[key]
        r._qcache[key] = (time.monotonic() - 1, res)
        t0 = time.monotonic()
        out = await r.search(Q)
        assert out.docs and time.monotonic() - t0 < 0.05
        await r.drain()
        assert p.calls == 2, "one background refresh"
        assert r._qcache[key][0] > time.monotonic()


class TestPersistQueryCache:
    async def test_a_new_router_answers_a_repeat_without_calling(self, tmp_path):
        a = Timed("duckduckgo", 0.0)
        r1 = _router(tmp_path, [a], query_cache_ttl_s=300, persist_query_cache=True)
        first = await r1.search(Q)
        r1._qstore.close()
        b = Timed("duckduckgo", 0.0)
        r2 = _router(tmp_path, [b], query_cache_ttl_s=300, persist_query_cache=True)
        again = await r2.search(Q)
        assert b.calls == 0
        assert [d.url for d in again.docs] == [d.url for d in first.docs]
        r2._qstore.close()


class TestLateMerge:
    async def test_straggler_lands_in_the_cached_entry(self, tmp_path):
        fast, slow = Timed("duckduckgo", 0.0), Timed("wikipedia", 0.3)
        r = _router(tmp_path, [fast, slow], query_cache_ttl_s=300, fanout_deadline_ms=50,
                    late_merge=True)
        first = await r.search(Q)
        assert first.late == ["wikipedia"]
        await r.drain()
        again = await r.search(Q)
        assert {d.source for d in again.docs} == {"duckduckgo", "wikipedia"}
        assert again.late == [] and slow.calls == 1

    async def test_off_keeps_the_partial_entry(self, tmp_path):
        r = _router(tmp_path, [Timed("duckduckgo", 0.0), Timed("wikipedia", 0.3)],
                    query_cache_ttl_s=300, fanout_deadline_ms=50, late_merge=False)
        await r.search(Q)
        await r.drain()
        again = await r.search(Q)
        assert {d.source for d in again.docs} == {"duckduckgo"}


class TestDeadlineDocFloor:
    async def test_thin_answer_waits_for_more_up_to_the_cap(self, tmp_path):
        provs = [Timed("duckduckgo", 0.0, docs=2), Timed("wikipedia", 0.3)]
        r = _router(tmp_path, provs, query_cache_ttl_s=0, fanout_deadline_ms=50,
                    deadline_min_docs=10, fanout_deadline_max_ms=2000)
        res = await r.search(Q)
        assert set(res.used) == {"duckduckgo", "wikipedia"} and not res.late

    async def test_cap_still_returns(self, tmp_path):
        provs = [Timed("duckduckgo", 0.0, docs=2), Timed("wikipedia", 2.0)]
        r = _router(tmp_path, provs, query_cache_ttl_s=0, fanout_deadline_ms=50,
                    deadline_min_docs=10, fanout_deadline_max_ms=300)
        t0 = time.monotonic()
        res = await r.search(Q)
        assert time.monotonic() - t0 < 1.0 and res.late == ["wikipedia"]
        await r.drain()

    async def test_no_spin_once_the_deadline_has_passed_unanswered(self, tmp_path, monkeypatch):
        import asyncio as aio

        real = aio.wait
        calls = {"n": 0}

        async def counting(*a, **kw):
            calls["n"] += 1
            return await real(*a, **kw)

        monkeypatch.setattr(aio, "wait", counting)
        r = _router(tmp_path, [Timed("duckduckgo", 0.3), Timed("wikipedia", 0.3)],
                    query_cache_ttl_s=0, fanout_deadline_ms=20)
        await r.search(Q)
        assert calls["n"] < 10, f"spun {calls['n']} times waiting past the deadline"


class TestHalfOpenSingle:
    def test_probe_in_flight_takes_the_provider_out(self, tmp_path):
        ddg, bing = Timed("duckduckgo", 0), Timed("bing", 0)
        r = _router(tmp_path, [ddg, bing], halfopen_single=True)
        ddg.health.consecutive_failures = 3
        ddg.health.opened_at = time.monotonic() - 999
        assert ddg.health.half_open
        ddg.health.probing = True
        assert "duckduckgo" not in [p.name for p in r.select(Q)]
        ddg.health.probing = False
        assert "duckduckgo" in [p.name for p in r.select(Q)]


class TestDegradedHold:
    async def test_degraded_failure_is_held_longer(self, tmp_path):
        from searchio.errors import ProviderError

        class Degraded(Timed):
            async def search(self, q, ctx):
                raise ProviderError(self.name, "degraded_results; degraded_results")

        b = Degraded("bing", 0)
        r = _router(tmp_path, [b, Timed("duckduckgo", 0)], degraded_hold_s=600,
                    query_cache_ttl_s=0)
        await r.search(Q)
        assert b.health.open and b.health.cooldown_s == 600


class TestStripResultTrackers:
    @pytest.mark.parametrize("url,want", [
        ("https://a.com/x?msockid=ab12&utm_source=bing", "https://a.com/x"),
        ("https://a.com/x?id=5&gclid=z&p=2#top", "https://a.com/x?id=5&p=2#top"),
        ("https://a.com/x?ref=nav&source=home", "https://a.com/x?ref=nav&source=home"),
        ("https://a.com/x?Q=a%20b&B=1", "https://a.com/x?Q=a%20b&B=1"),
        ("https://a.com/x", "https://a.com/x"),
    ])
    def test_only_click_trackers_go(self, url, want):
        from searchio.fuse import strip_trackers

        assert strip_trackers(url) == want

    @pytest.mark.parametrize("on", [False, True])
    async def test_result_urls_are_cleaned_only_when_on(self, tmp_path, on):
        class Tracked(Timed):
            async def search(self, q, ctx):
                return [Doc(url="https://a.com/p?id=1&msockid=9", title="a", source=self.name)]

        r = _router(tmp_path, [Tracked("duckduckgo", 0), Timed("bing", 0)],
                    strip_result_trackers=on, query_cache_ttl_s=0)
        res = await r.search(Q)
        urls = [d.url for d in res.docs]
        assert ("https://a.com/p?id=1" in urls) is on, urls


class TestSearchWallRescue:
    """search_wall_rescue: no general web engine answered -> one browser-backed
    provider is asked before the reference filler goes back."""

    @staticmethod
    def _provs():
        browser = Timed("sidecar_search", 0, docs=3)
        browser.cost_per_1k = 5.0
        return [Timed("duckduckgo", 0, fail=True), Timed("bing", 0, fail=True),
                Timed("wikipedia", 0), browser], browser

    @pytest.mark.parametrize("on", [False, True])
    async def test_rescue_only_when_on(self, tmp_path, on):
        provs, browser = self._provs()
        r = _router(tmp_path, provs, search_wall_rescue=on, query_cache_ttl_s=0)
        res = await r.search(Q)
        assert (browser.calls == 1) is on
        assert ("sidecar_search" in res.used) is on, res

    async def test_an_answering_web_engine_costs_no_browser(self, tmp_path):
        provs, browser = self._provs()
        provs[1] = Timed("bing", 0)
        r = _router(tmp_path, provs, search_wall_rescue=True, query_cache_ttl_s=0)
        res = await r.search(Q)
        assert browser.calls == 0 and "bing" in res.used

    async def test_an_explicit_expensive_call_is_not_doubled(self, tmp_path):
        provs, browser = self._provs()
        r = _router(tmp_path, provs, search_wall_rescue=True, query_cache_ttl_s=0, fanout=4)
        await r.search(Q, allow_expensive=True)
        assert browser.calls == 1


# ── batch 7 ──────────────────────────────────────────────────────────────────

_PAGE = ("<html><head><title>T &amp; U</title></head><body>"
         + "<p>" + "word " * 400 + "</p></body></html>")


class _Res:
    def __init__(self, url, body):
        self.final_url, self.body, self.via = url, body, "tier0"


def _read_engine(tmp_path, monkeypatch, **kw):
    from searchio.engine import Engine

    eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, **kw))
    fetched: list[str] = []

    async def fake_fetch(url, **_kw):
        fetched.append(url)
        return _Res(url, _PAGE)

    monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
    return eng, fetched


class TestMemoExtractImport:
    def test_failed_import_is_tried_once(self, monkeypatch):
        import builtins

        from searchio import extract
        from searchio.config import set_settings

        tries = {"n": 0}
        real = builtins.__import__

        def fake(name, *a, **kw):
            if name == "trafilatura":
                tries["n"] += 1
                raise ImportError("broken install")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake)
        monkeypatch.setattr(extract, "_TRAF", None)
        set_settings(Settings(memo_extract_import=True))
        try:
            for _ in range(3):
                assert extract.to_markdown(_PAGE)
            assert tries["n"] == 1
        finally:
            set_settings(None)


class TestPrefetchExtract:
    async def test_read_after_prefetch_skips_extraction(self, tmp_path, monkeypatch):
        from searchio import extract
        from searchio.router import RouteResult

        eng, _ = _read_engine(tmp_path, monkeypatch, prefetch_top_k=1, prefetch_extract=True)

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url="https://h.com/a")])

        monkeypatch.setattr(eng.router, "search", fake_search)
        await eng.search("x")
        await eng.router.drain()
        calls = {"n": 0}
        real = extract.doc_from_fetch

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(extract, "doc_from_fetch", counting)
        doc = await eng.read("https://h.com/a")
        assert calls["n"] == 0 and doc.title == "T & U" and doc.fetched_via == "tier0"
        await eng.close()

    async def test_changed_body_is_extracted_again(self, tmp_path, monkeypatch):
        from searchio.engine import _body_key

        eng, _ = _read_engine(tmp_path, monkeypatch, read_memo=True)
        eng._read_memo = __import__("collections").OrderedDict(
            {"https://h.com/a": (_body_key("old body"), Doc(url="https://h.com/a", title="stale"))})
        doc = await eng.read("https://h.com/a")
        assert doc.title == "T & U"
        await eng.close()


class TestReadMemo:
    async def test_second_read_reuses_the_doc(self, tmp_path, monkeypatch):
        from searchio import extract

        eng, _ = _read_engine(tmp_path, monkeypatch, read_memo=True)
        calls = {"n": 0}
        real = extract.doc_from_fetch

        def counting(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        monkeypatch.setattr(extract, "doc_from_fetch", counting)
        a = await eng.read("https://h.com/a")
        a.title = "mutated by the caller"
        b = await eng.read("https://h.com/a")
        assert calls["n"] == 1 and b.title == "T & U"
        await eng.close()


class TestReadOffloop:
    async def test_large_page_extracts_on_a_thread(self, tmp_path, monkeypatch):
        import threading

        from searchio import extract

        eng, _ = _read_engine(tmp_path, monkeypatch, read_offloop_bytes=100)
        seen = []
        real = extract.doc_from_fetch

        def where(*a, **kw):
            seen.append(threading.current_thread() is threading.main_thread())
            return real(*a, **kw)

        monkeypatch.setattr(extract, "doc_from_fetch", where)
        await eng.read("https://h.com/a")
        assert seen == [False]
        await eng.close()


class TestTransientHold:
    async def test_timeouts_trip_a_short_hold(self, tmp_path):
        slow = Timed("wikipedia", 5.0)
        r = _router(tmp_path, [slow], transient_hold_s=20, provider_timeout_s=0.01,
                    query_cache_ttl_s=0, adaptive_timeout=False)
        for _ in range(3):
            await r.search(Q)
        assert slow.health.open and slow.health.cooldown_s == 20

    async def test_a_wall_keeps_the_full_cooldown(self, tmp_path):
        down = Timed("wikipedia", 0, fail=True)
        r = _router(tmp_path, [down], transient_hold_s=20, query_cache_ttl_s=0,
                    breaker_backoff_s=0.0)
        for _ in range(3):
            await r.search(Q)
        assert down.health.open and down.health.cooldown_s == down.health.COOLDOWN_S


class TestPredictiveCut:
    async def test_returns_before_the_deadline_when_the_rest_is_due_late(self, tmp_path):
        fast, slow = Timed("duckduckgo", 0.0, docs=10), Timed("crossref", 0.5)
        slow.health.calls, slow.health.ema_latency_ms = 5, 2000.0
        r = _router(tmp_path, [fast, slow], predictive_cut=True, fanout_deadline_ms=300,
                    query_cache_ttl_s=0)
        t0 = time.monotonic()
        res = await r.search(Q)
        assert time.monotonic() - t0 < 0.2 and res.late == ["crossref"]
        await r.drain()

    async def test_waits_for_a_provider_due_in_time(self, tmp_path):
        fast, slow = Timed("duckduckgo", 0.0, docs=10), Timed("wikipedia", 0.1)
        slow.health.calls, slow.health.ema_latency_ms = 5, 100.0
        r = _router(tmp_path, [fast, slow], predictive_cut=True, fanout_deadline_ms=300,
                    query_cache_ttl_s=0)
        res = await r.search(Q)
        assert set(res.used) == {"duckduckgo", "wikipedia"}

    async def test_thin_answer_waits(self, tmp_path):
        fast, slow = Timed("duckduckgo", 0.0, docs=2), Timed("crossref", 0.1)
        slow.health.calls, slow.health.ema_latency_ms = 5, 2000.0
        r = _router(tmp_path, [fast, slow], predictive_cut=True, fanout_deadline_ms=300,
                    query_cache_ttl_s=0)
        res = await r.search(Q)
        assert set(res.used) == {"duckduckgo", "crossref"}


class TestFastTitle:
    def test_head_title_and_fallbacks(self):
        from searchio.config import set_settings
        from searchio.extract import title_of

        set_settings(Settings(fast_title=True))
        try:
            assert title_of(_PAGE) == "T & U"
            svg = ("<html><head><svg><title>icon</title></svg><title>Real</title></head>"
                   "<body></body></html>")
            assert title_of(svg) == "Real"
            late = "<html><head></head><body><title>Body Title</title></body></html>"
            assert title_of(late) == "Body Title"
            assert title_of("<p>no title</p>") == ""
        finally:
            set_settings(None)


# ── batch 8 ──────────────────────────────────────────────────────────────────


class TestSearchMany:
    async def test_order_kept_failures_isolated_and_capped(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=0,
                              search_many_concurrency=2))
        live = {"now": 0, "peak": 0}

        async def fake_search(q, **kw):
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
            await asyncio.sleep(0.02)
            live["now"] -= 1
            if q.text == "bad":
                raise RuntimeError("down")
            return RouteResult(docs=[Doc(url=f"https://{q.text}.com/")])

        monkeypatch.setattr(eng.router, "search", fake_search)
        out = await eng.search_many(["a", "bad", "c", "d"])
        assert [type(r).__name__ for r in out] == ["RouteResult", "RuntimeError",
                                                   "RouteResult", "RouteResult"]
        assert out[2].docs[0].url == "https://c.com/"
        assert live["peak"] == 2
        await eng.close()


class TestPersistHolds:
    async def test_a_new_router_starts_with_the_quota_hold(self, tmp_path):
        from searchio.errors import RateLimited

        class Spent(Timed):
            async def search(self, q, ctx):
                self.calls += 1
                raise RateLimited("rate_limited", retry_after=3000)

        a = Spent("openalex", 0)
        r1 = _router(tmp_path, [a, Timed("duckduckgo", 0)], ratelimit_aware=True,
                     persist_holds=True, query_cache_ttl_s=0)
        await r1.search(Q)
        r1._holds.close()
        b = Spent("openalex", 0)
        r2 = _router(tmp_path, [b, Timed("duckduckgo", 0)], ratelimit_aware=True,
                     persist_holds=True, query_cache_ttl_s=0)
        assert b.health.open and 2900 < b.health.cooldown_s <= 3000
        await r2.search(Q)
        assert b.calls == 0
        r2._holds.close()


class TestPrefetchMinKbps:
    async def test_slow_link_skips_prefetch(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=1,
                              prefetch_min_kbps=300))
        fetched = []

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url="https://h.com/")])

        async def fake_fetch(url, **kw):
            fetched.append(url)

        monkeypatch.setattr(eng.router, "search", fake_search)
        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        eng.ladder.link_kbps = 86.0
        await eng.search("x")
        await eng.router.drain()
        assert fetched == []
        eng.ladder.link_kbps = 2000.0
        await eng.search("x")
        await eng.router.drain()
        assert fetched == ["https://h.com/"]
        await eng.close()

    def test_throughput_estimate_ignores_small_bodies(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path))
        lad._note_transfer(4000, 0.5)
        assert lad.link_kbps is None
        lad._note_transfer(100 * 1024, 1.0)
        assert lad.link_kbps == 100.0


class TestCancelStragglers:
    async def test_late_provider_is_cancelled(self, tmp_path):
        fast, slow = Timed("duckduckgo", 0.0), Timed("wikipedia", 0.5)
        r = _router(tmp_path, [fast, slow], cancel_stragglers=True, fanout_deadline_ms=50,
                    query_cache_ttl_s=0, late_merge=False)
        res = await r.search(Q)
        assert res.late == ["wikipedia"] and not r._background


class TestSpottyLinkKnobs:
    async def test_http1_and_fast_connect_build_the_client(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, tier0_http2=False,
                                         connect_timeout_s=2.5, connect_retries=1))
        c = await lad._client()
        assert c.timeout.connect == 2.5 and c.timeout.read == lad.s.tier0_timeout_s
        assert c._transport._pool._http2 is False
        assert c._transport._pool._retries == 1
        await lad.close()

    async def test_stalled_request_is_hedged_on_a_fresh_client(self, tmp_path, monkeypatch):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, stall_hedge_ms=50))
        clients = []

        class R:
            closed = False

            async def aclose(self):
                R.closed = True

        async def fake_open(url, headers, timeout, client=None):
            clients.append(client)
            if client is None:
                await asyncio.sleep(5)  # the stalled connection
            return R()

        monkeypatch.setattr(lad, "_open_stream", fake_open)
        t0 = time.monotonic()
        r = await lad._open_hedged("https://h.com/", {}, 8.0)
        assert isinstance(r, R) and time.monotonic() - t0 < 1.0
        assert clients[0] is None and clients[1] is lad._hedge_http
        assert lad._stats.get("stall_hedge_won") == 1
        await lad.close()

    async def test_slow_host_waits_past_its_own_normal(self, tmp_path, monkeypatch):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, stall_hedge_ms=50))
        lad._ttfb["h.com"] = 0.2
        calls = []

        async def fake_open(url, headers, timeout, client=None):
            calls.append(client)
            await asyncio.sleep(0.15)
            return object()

        monkeypatch.setattr(lad, "_open_stream", fake_open)
        await lad._open_hedged("https://h.com/x", {}, 8.0)
        assert calls == [None], "0.15 s is normal for this host: no hedge"
        await lad.close()


class TestLinkDownGuard:
    async def test_simultaneous_network_failures_trip_nothing(self, tmp_path):
        class Dropped(Timed):
            async def search(self, q, ctx):
                self.calls += 1
                raise RuntimeError("tier0 ConnectError: [Errno 11001] getaddrinfo failed")

        provs = [Dropped("duckduckgo", 0), Dropped("wikipedia", 0), Dropped("hackernews", 0)]
        r = _router(tmp_path, provs, link_down_guard=True, query_cache_ttl_s=0)
        for _ in range(4):
            await r.search(Q)
        assert not any(p.health.open for p in provs)
        assert sum(p.health.consecutive_failures for p in provs) <= 1

    async def test_one_host_down_still_trips(self, tmp_path):
        class Refused(Timed):
            async def search(self, q, ctx):
                raise RuntimeError("tier0 ConnectError: connection refused")

        bad = Refused("wikipedia", 0)
        r = _router(tmp_path, [bad, Timed("duckduckgo", 0)], link_down_guard=True,
                    query_cache_ttl_s=0)
        for _ in range(3):
            await r.search(Q)
        assert bad.health.open


class TestOfflineStale:
    async def test_expired_answer_beats_nothing(self, tmp_path):
        p = Timed("duckduckgo", 0)
        r = _router(tmp_path, [p], query_cache_ttl_s=60, offline_stale_s=3600,
                    query_cache_swr_s=0)
        first = await r.search(Q)
        key = next(iter(r._qcache))
        r._qcache[key] = (time.monotonic() - 10, r._qcache[key][1])
        p.fail = True
        out = await r.search(Q)
        assert [d.url for d in out.docs] == [d.url for d in first.docs]
        assert "cached answer" in out.failed["router"] and "duckduckgo" in out.failed

    async def test_off_returns_the_empty_answer(self, tmp_path):
        p = Timed("duckduckgo", 0)
        r = _router(tmp_path, [p], query_cache_ttl_s=60, query_cache_swr_s=0)
        await r.search(Q)
        key = next(iter(r._qcache))
        r._qcache[key] = (time.monotonic() - 10, r._qcache[key][1])
        p.fail = True
        assert not (await r.search(Q)).docs


class TestStaleIfError:
    async def test_network_failure_serves_the_expired_page(self, tmp_path, monkeypatch):
        from searchio.errors import TransientError

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, stale_if_error_s=3600,
                                         cache_ttl_s=1))
        lad.cache.put("https://h.com/", {"url": "https://h.com/", "final_url": "https://h.com/",
                                         "status": 200, "body": "old page", "tier": 0},
                      variant=lad._cache_variant(False))
        lad.cache._db.execute("UPDATE entries SET created = created - 100")

        async def down(*a, **kw):
            raise TransientError("tier0 ConnectError: unreachable")

        monkeypatch.setattr(lad, "_fetch_shared", down)
        res = await lad.fetch("https://h.com/")
        assert res.body == "old page" and res.via == "cache-stale-error"
        await lad.close()

    async def test_a_block_is_never_masked(self, tmp_path, monkeypatch):
        from searchio.errors import Blocked

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, stale_if_error_s=3600))

        async def walled(*a, **kw):
            raise Blocked("captcha")

        monkeypatch.setattr(lad, "_fetch_shared", walled)
        with pytest.raises(Blocked):
            await lad.fetch("https://h.com/")
        await lad.close()


class TestDnsStaleIfError:
    def test_last_answer_dials_when_the_resolver_fails(self, monkeypatch):
        import socket

        monkeypatch.setattr(ladder_mod, "_DNS_STALE", [True])
        monkeypatch.setattr(ladder_mod, "_DNS_TTL", [0.0])
        monkeypatch.setattr(ladder_mod, "_DNS_LAST", {})
        good = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: good)
        assert ladder_mod._resolve_checked("h.example", 443) == good

        def fail(*a, **k):
            raise socket.gaierror(11002, "temporary failure")

        monkeypatch.setattr(socket, "getaddrinfo", fail)
        assert ladder_mod._resolve_checked("h.example", 443) == good
        with pytest.raises(socket.gaierror):
            ladder_mod._resolve_checked("never.example", 443)


class TestStaggeredConnect:
    async def test_a_stalled_dial_is_overtaken(self, monkeypatch):
        import anyio
        import socket

        dials = []

        class S:
            closed = False

            async def aclose(self):
                S.closed = True

        async def fake_connect(remote_host, remote_port, local_host=None):
            dials.append(remote_host)
            if len(dials) == 1:
                await asyncio.sleep(10)  # the lost SYN
            return S()

        monkeypatch.setattr(anyio, "connect_tcp", fake_connect)
        monkeypatch.setattr(ladder_mod, "_CONNECT_STAGGER", [0.05])
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.5", 443))]
        t0 = time.monotonic()
        s = await ladder_mod._staggered_connect(infos, 443, 8.0, None, None,
                                                ConnectionError, TimeoutError)
        assert isinstance(s, S) and time.monotonic() - t0 < 0.5
        assert dials == ["203.0.113.5", "203.0.113.5"], "same address, fresh socket"

    async def test_all_failing_raises_the_connect_error(self, monkeypatch):
        import anyio
        import socket

        async def refused(remote_host, remote_port, local_host=None):
            raise OSError("connection refused")

        monkeypatch.setattr(anyio, "connect_tcp", refused)
        monkeypatch.setattr(ladder_mod, "_CONNECT_STAGGER", [0.05])
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.5", 443))]
        with pytest.raises(ConnectionError):
            await ladder_mod._staggered_connect(infos, 443, 8.0, None, None,
                                                ConnectionError, TimeoutError)

    async def test_overall_timeout(self, monkeypatch):
        import anyio
        import socket

        async def black_hole(remote_host, remote_port, local_host=None):
            await asyncio.sleep(10)

        monkeypatch.setattr(anyio, "connect_tcp", black_hole)
        monkeypatch.setattr(ladder_mod, "_CONNECT_STAGGER", [0.02])
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.5", 443))]
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            await ladder_mod._staggered_connect(infos, 443, 0.2, None, None,
                                                ConnectionError, TimeoutError)
        assert time.monotonic() - t0 < 1.0


class TestSpecialistK:
    async def test_specialists_ask_for_fewer(self, tmp_path):
        seen = {}

        class Rec(Timed):
            async def search(self, q, ctx):
                seen[self.name] = q.k
                return await super().search(q, ctx)

        r = _router(tmp_path, [Rec("duckduckgo", 0), Rec("wikipedia", 0)],
                    specialist_k_ratio=0.6, query_cache_ttl_s=0)
        await r.search(Q)
        assert seen == {"duckduckgo": 10, "wikipedia": 6}


class TestQcacheKSuperset:
    async def test_wider_answer_serves_a_narrower_ask(self, tmp_path):
        p = Timed("duckduckgo", 0, docs=10)
        r = _router(tmp_path, [p], query_cache_ttl_s=300, qcache_k_superset=True)
        await r.search(Q)
        five = await r.search(Q.model_copy(update={"k": 5}))
        assert p.calls == 1 and len(five.docs) == 5
        await r.search(Q.model_copy(update={"k": 20}))
        assert p.calls == 2, "a wider ask is a miss"
        await r.search(Q)
        assert p.calls == 2, "the k=20 entry answers k=10"


class TestLeanSkipCrossref:
    async def test_select_only_when_not_skipped(self):
        from searchio.providers.apis import Crossref

        for skip, want in ((False, True), (True, False)):
            lad = _UrlLadder('{"status": "ok", "message": {"items": []}}')
            await Crossref().search(Q, _ctx(lad, lean_api_fields=True, lean_skip_crossref=skip))
            assert ("select=" in lad.urls[0]) is want


class TestReadStall:
    async def test_client_read_gap(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, read_stall_s=3, connect_timeout_s=0))
        c = await lad._client()
        assert c.timeout.read == 3 and c.timeout.connect == lad.s.tier0_timeout_s
        await lad.close()

    async def test_per_request_timeout_keeps_connect_and_read_knobs(self, tmp_path, monkeypatch):
        # #128: _open_stream's per-request float REPLACED the client Timeout,
        # so connect_timeout_s / read_stall_s never reached a tier-0 GET.
        import httpx

        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, robots_policy="off",
                                         connect_timeout_s=3, read_stall_s=4))
        c = await lad._client()
        seen = []

        async def send(request, **kw):
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, request=request, content=b"ok")

        monkeypatch.setattr(c, "send", send)
        r = await lad._open_stream("https://x.org/", {}, 8.0)
        await r.aclose()
        assert seen[0]["connect"] == 3 and seen[0]["read"] == 4 and seen[0]["pool"] == 8.0
        await lad.close()


class TestPrefetchYield:
    async def test_new_search_cancels_speculative_fetches(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=1,
                              prefetch_yield=True))
        cancelled = []

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url=f"https://{q.text}.com/")])

        async def slow_fetch(url, **kw):
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(url)
                raise

        monkeypatch.setattr(eng.router, "search", fake_search)
        monkeypatch.setattr(eng.ladder, "fetch", slow_fetch)
        await eng.search("a")
        await asyncio.sleep(0)
        await eng.search("b")
        await asyncio.sleep(0.01)
        assert cancelled == ["https://a.com/"]
        await eng.close()


class TestRobotsSkipApis:
    async def test_api_host_skips_robots_unless_enforced(self, tmp_path, monkeypatch):
        for policy, want_check in (("warn", False), ("enforce", True)):
            lad = ladder_mod.Ladder(Settings(state_dir=tmp_path / policy, robots_skip_apis=True,
                                             robots_policy=policy, cache_enabled=False))
            checked = []

            async def check(url, checked=checked):
                checked.append(url)
                from searchio.net.robots import RobotsInfo
                return RobotsInfo(allowed=True, checked=False, reason="t")

            async def tier(self, t, url, **kw):
                return 200, {"content-type": "application/json"}, '{"ok": 1}', "application/json", url

            monkeypatch.setattr(lad.robots, "check", check)
            monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
            await lad.fetch("https://api.crossref.org/works?query=x")
            assert bool(checked) is want_check
            await lad.close()


class TestReadSoftCap:
    async def test_slow_link_read_is_capped_and_not_cached(self, tmp_path):
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path))

        class R:
            headers = {}

            async def aiter_bytes(self, n):
                for _ in range(100):
                    yield b"x" * 65536

        tok = ladder_mod.BODY_SOFT_CAP.set((128 * 1024, []))
        try:
            raw = await lad._read_capped(R(), 10 * 1024 * 1024)
            holder = ladder_mod.BODY_SOFT_CAP.get()[1]
            assert len(raw) == 128 * 1024 and holder
            puts = []
            lad.cache.put = lambda *a, **k: puts.append(a)
            lad._cache_put("https://h.com/", {"body": "x"}, "", 0)
            assert puts == []
        finally:
            ladder_mod.BODY_SOFT_CAP.reset(tok)
        await lad.close()


# ── batch 9 ──────────────────────────────────────────────────────────────────


class TestLeanRead:
    def test_rewrites_match_only_their_pages(self):
        from searchio.leanread import lean_target

        s = Settings(lean_read_wikipedia=True, lean_read_arxiv=True, lean_read_github=True)
        assert "prop=extracts" in lean_target("https://en.wikipedia.org/wiki/Heat_pump", s)[0]
        assert lean_target("https://en.wikipedia.org/wiki/Special:Random", s) is None
        assert lean_target("https://arxiv.org/pdf/1706.03762", s)[0] == \
            "https://arxiv.org/html/1706.03762"
        assert lean_target("https://arxiv.org/pdf/1706.03762v2.pdf", s)[0] == \
            "https://arxiv.org/html/1706.03762v2"
        assert lean_target("https://github.com/tokio-rs/tokio", s)[0].endswith(
            "/tokio-rs/tokio/HEAD/README.md")
        assert lean_target("https://github.com/tokio-rs/tokio/issues", s) is None
        assert lean_target("https://github.com/topics/rust", s) is None
        assert lean_target("https://en.wikipedia.org/wiki/Heat_pump",
                           Settings(lean_read_wikipedia=False)) is None

    async def test_lean_doc_keeps_the_original_url_and_falls_back(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, lean_read_wikipedia=True))
        fetched = []

        class R:
            def __init__(self, url, body, status=200):
                self.final_url, self.body, self.via, self.status = url, body, "http", status

        async def fake_fetch(url, **kw):
            fetched.append(url)
            if "api.php" in url:
                if "Missing" in url:
                    return R(url, '{"query": {"pages": [{"missing": true}]}}')
                return R(url, '{"query": {"pages": [{"title": "Heat pump", "extract": "'
                         + "warm " * 100 + '"}]}}')
            return R(url, _PAGE)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("https://en.wikipedia.org/wiki/Heat_pump")
        assert d.url == "https://en.wikipedia.org/wiki/Heat_pump" and d.title == "Heat pump"
        assert d.fetched_via == "http+lean" and d.text.startswith("warm")
        d2 = await eng.read("https://en.wikipedia.org/wiki/Missing_page")
        assert d2.title == "T & U", "a missing extract reads the page itself"
        assert fetched[-1] == "https://en.wikipedia.org/wiki/Missing_page"
        await eng.close()


class TestRevalidate:
    async def test_304_answers_from_the_expired_entry(self, tmp_path, monkeypatch):
        sent = []
        lad = ladder_mod.Ladder(Settings(state_dir=tmp_path, revalidate_cache=True,
                                         cache_ttl_s=1, robots_policy="off", max_tier=0))
        body = "<html><body>" + "word " * 80 + "</body></html>"

        async def tier(self, t, url, **kw):
            cond = ladder_mod.COND_HEADERS.get()
            sent.append(dict(cond or {}))
            if cond:
                return 304, {}, "", "text/html", url
            return 200, {"ETag": '"v1"', "content-type": "text/html"}, body, "text/html", url

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        first = await lad.fetch("https://h.com/a")
        assert first.etag == '"v1"'
        lad.cache._db.execute("UPDATE entries SET created = created - 100")
        again = await lad.fetch("https://h.com/a")
        assert sent[-1] == {"If-None-Match": '"v1"'}
        assert again.via == "cache-revalidated" and again.body == body
        third = await lad.fetch("https://h.com/a")
        assert third.via == "cache", "the 304 restarted the entry's TTL"
        await lad.close()


class TestLeanReadMore:
    def test_mdn_and_blob_rewrites(self):
        from searchio.leanread import lean_target

        s = Settings(lean_read_mdn=True, lean_read_github_blob=True, lean_read_github=False)
        assert lean_target("https://developer.mozilla.org/en-US/docs/Web/API/WebSocket", s)[0] == \
            "https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/index.json"
        assert lean_target("https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/", s)[0] == \
            "https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/index.json"
        assert lean_target("https://github.com/o/r/blob/main/a/b.py", s)[0] == \
            "https://raw.githubusercontent.com/o/r/main/a/b.py"
        assert lean_target("https://github.com/o/r", s) is None, "repo pages ride their own knob"

    def test_mdn_build_joins_sections(self):
        from searchio.leanread import _mdn

        _url, build = _mdn("https://developer.mozilla.org/en-US/docs/Web/API/X")
        body = ('{"doc": {"title": "X", "body": [{"type": "prose", "value": {"title": null, '
                '"content": "<p>' + "alpha " * 30 + '</p>"}}, {"type": "prose", "value": '
                '{"title": "Usage", "content": "<p>' + "beta " * 30 + '</p>"}}]}}')
        title, text = build(body)
        assert title == "X" and "## Usage" in text and "<p>" not in text


class TestReadMany:
    async def test_order_kept_failures_isolated(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, read_many_concurrency=2))

        async def fake_read(url):
            if "bad" in url:
                raise RuntimeError("down")
            return Doc(url=url, text="x")

        monkeypatch.setattr(eng, "read", fake_read)
        out = await eng.read_many(["https://a/", "https://bad/", "https://c/"])
        assert [getattr(d, "url", type(d).__name__) for d in out] == \
            ["https://a/", "RuntimeError", "https://c/"]
        await eng.close()


class TestPrefetchEarly:
    async def test_generalist_first_answer_starts_the_prefetch(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.providers.registry import Registry

        ddg, wiki = Timed("duckduckgo", 0.0), Timed("wikipedia", 0.3)
        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=1,
                              prefetch_early=True, fanout_deadline_ms=1500, deadline_min_docs=0,
                              query_cache_ttl_s=0), registry=Registry([ddg, wiki]))
        started = []

        async def fake_fetch(url, **kw):
            started.append((url, time.monotonic()))

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        t0 = time.monotonic()
        await eng.search("x")
        await eng.router.drain()
        assert started and started[0][0] == "https://duckduckgo.com/0"
        assert started[0][1] - t0 < 0.2, "started before the slow specialist answered"
        await eng.close()


# ── batch 10 ─────────────────────────────────────────────────────────────────


class TestLeanReadBatch10:
    def test_rewrites(self):
        from searchio.leanread import lean_target, lean_targets

        s = Settings(lean_read_hf=True, lean_read_crates=True, lean_read_npm=True,
                     lean_read_github_issue=True, lean_read_rfc=True)
        assert lean_target("https://huggingface.co/mistralai/Mistral-7B-v0.1", s)[0] == \
            "https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/README.md"
        assert lean_target("https://huggingface.co/datasets/openai/gsm8k", s)[0] == \
            "https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md"
        assert lean_target("https://huggingface.co/docs/transformers", s) is None
        assert lean_target("https://crates.io/crates/serde", s)[0] == \
            "https://docs.rs/crate/serde/latest"
        assert [u for u, _ in lean_targets("https://www.npmjs.com/package/@types/node", s)] == \
            ["https://unpkg.com/@types/node/README.md"]
        assert lean_target("https://github.com/o/r/pull/7", s)[0] == (
            "https://api.github.com/repos/o/r/issues/7",
            "https://api.github.com/repos/o/r/issues/7/comments?per_page=100")
        for u in ("https://www.rfc-editor.org/rfc/rfc9110.html",
                  "https://datatracker.ietf.org/doc/html/rfc9110",
                  "https://tools.ietf.org/html/rfc9110#section-3"):
            assert lean_target(u, s)[0] == "https://www.rfc-editor.org/rfc/rfc9110.txt"

    def test_readme_opening_with_inline_html_is_kept(self):
        from searchio.leanread import _github

        _u, build = _github("https://github.com/o/r")
        md = '<p align="center"><img src="logo.png"></p>\n\n# R\n\n' + "text " * 30
        assert build(md)[1].startswith("<p align")
        assert build("<!DOCTYPE html><html><body>404</body></html>" + " " * 100) is None

    def test_issue_build_and_rfc_title(self):
        from searchio.leanread import _github_issue, _rfc

        _u, build = _github_issue("https://github.com/o/r/issues/3")
        title, text = build('{"title": "Bug", "state": "open", "user": {"login": "a"}, '
                            '"body": "it breaks"}',
                            '[{"user": {"login": "b"}, "body": "same here"}]')
        assert title == "Bug - o/r#3" and "it breaks" in text and "**b**" in text
        assert build('{"message": "API rate limit exceeded"}', "[]") is None
        _u, build = _rfc("https://www.rfc-editor.org/rfc/rfc1.html")
        rfc = ("﻿\n\nNetwork Working Group     S. Crocker\nRequest for Comments: 1\n\n"
               "                 Host Software\n\nAbstract\n\n" + "words " * 100)
        assert build(rfc)[0] == "RFC 1: Host Software"

    async def test_multi_part_lean_fetches_every_part(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False,
                              lean_read_github_issue=True))
        fetched = []

        class R:
            def __init__(self, url, body, status=200):
                self.final_url, self.body, self.via, self.status = url, body, "http", status

        async def fake_fetch(url, **kw):
            fetched.append(url)
            if url.endswith("/comments?per_page=100"):
                return R(url, "[]", status=403 if "/9" in url else 200)
            if "api.github.com" in url:
                return R(url, '{"title": "T", "body": "b"}')
            return R(url, _PAGE)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("https://github.com/o/r/issues/1")
        assert d.fetched_via == "http+lean" and d.url == "https://github.com/o/r/issues/1"
        d = await eng.read("https://github.com/o/r/issues/9")
        assert d.title == "T & U", "a rate-limited part reads the page"
        await eng.close()


class TestPageSwr:
    async def test_expired_page_answers_stale_then_refreshes(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_ttl_s=1, page_swr_s=3600,
                              robots_policy="off", max_tier=0, prefetch_top_k=0))
        n = {"calls": 0}

        async def tier(self, t, url, **kw):
            n["calls"] += 1
            return 200, {"content-type": "text/html"}, _PAGE, "text/html", url

        monkeypatch.setattr(ladder_mod.Ladder, "_try_tier", tier)
        await eng.read("https://h.com/a")
        eng.ladder.cache._db.execute("UPDATE entries SET created = created - 100")
        d = await eng.read("https://h.com/a")
        assert d.fetched_via == "cache-stale" and d.title == "T & U"
        await asyncio.gather(*list(eng.ladder._bg))
        assert n["calls"] == 2, "the background refetch ran"
        assert ladder_mod.SERP_SWR.get() == 0.0, "the read did not leak SWR to its caller"
        await eng.close()


class TestPreconnect:
    async def test_warms_unprefetched_hosts_once_each(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.router import RouteResult

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, prefetch_top_k=1,
                              preconnect_top_k=4, lean_read_github=True))
        got = []

        async def fake_search(q, **kw):
            return RouteResult(docs=[Doc(url=u) for u in (
                "https://a.com/1", "https://b.com/1", "https://b.com/2",
                "https://github.com/o/r", "https://c.com/x")])

        async def raw_get(url):
            got.append(url)
            return 404, ""

        async def no_fetch(url, **kw):
            return None

        monkeypatch.setattr(eng.router, "search", fake_search)
        monkeypatch.setattr(eng.ladder, "fetch", no_fetch)
        monkeypatch.setattr(eng.ladder, "_raw_get", raw_get)
        monkeypatch.setattr(eng.ladder.robots, "_fetch", raw_get)
        await eng.search("x")
        await eng.router.drain()
        await asyncio.gather(*list(eng.ladder.robots._bg), return_exceptions=True)
        assert sorted(got) == ["https://b.com/robots.txt",
                               "https://raw.githubusercontent.com/robots.txt"]
        await eng.search("x")  # rules now known: a plain GET still connects
        await eng.router.drain()
        assert len(got) == 4
        await eng.close()


class TestLeanReadBatch11:
    def test_rewrites(self):
        from searchio.leanread import lean_target, lean_targets

        s = Settings(lean_read_github_releases=True, lean_read_github_commit=True,
                     lean_read_k8s=True, lean_read_rust_book=True)
        assert lean_target("https://github.com/o/r/releases", s)[0] == \
            "https://api.github.com/repos/o/r/releases?per_page=10"
        assert lean_target("https://github.com/o/r/releases/tag/v1.2", s)[0] == \
            "https://api.github.com/repos/o/r/releases/tags/v1.2"
        assert lean_target("https://github.com/o/r/commit/4b85e6c", s)[0] == \
            "https://github.com/o/r/commit/4b85e6c.patch"
        assert [u for u, _ in lean_targets("https://kubernetes.io/docs/concepts/workloads/", s)] == [
            "https://raw.githubusercontent.com/kubernetes/website/HEAD/content/en/docs/concepts/workloads.md",
            "https://raw.githubusercontent.com/kubernetes/website/HEAD/content/en/docs/concepts/workloads/_index.md"]
        assert "/content/zh-cn/docs/" in lean_target("https://kubernetes.io/zh-cn/docs/home/x/", s)[0]
        assert lean_target("https://kubernetes.io/docs/", s) is None
        assert lean_target("https://doc.rust-lang.org/stable/book/ch16-00-concurrency.html", s)[0] == \
            "https://raw.githubusercontent.com/rust-lang/book/HEAD/src/ch16-00-concurrency.md"
        assert lean_target("https://doc.rust-lang.org/reference/items/traits.html", s)[0] == \
            "https://raw.githubusercontent.com/rust-lang/reference/HEAD/src/items/traits.md"
        assert lean_target("https://doc.rust-lang.org/book/", s) is None
        assert lean_target("https://doc.rust-lang.org/std/vec/struct.Vec.html", s) is None

    def test_builds(self):
        from searchio.leanread import _github_commit, _github_releases, _k8s

        _u, build = _github_releases("https://github.com/o/r/releases")
        title, text = build('[{"tag_name": "v1", "name": "One", "body": "notes"}]')
        assert title == "Releases - o/r" and "## One (v1)" in text and "notes" in text
        assert build('{"message": "Not Found"}') is None
        _u, build = _github_commit("https://github.com/o/r/commit/abcdef1")
        patch = "From abc Mon Sep 17 00:00:00 2001\nSubject: [PATCH] Fix it\n\n---\n diff"
        assert build(patch)[0] == "Fix it - o/r@abcdef1"
        assert build("<!DOCTYPE html><html>") is None
        (_u, build), _ = _k8s("https://kubernetes.io/docs/a/b/")
        title, text = build('---\ntitle: "Pod Lifecycle"\nweight: 30\n---\n' + "body " * 60)
        assert title == "Pod Lifecycle" and text.startswith("body")
        assert build("404: Not Found") is None


class TestLeanReadBatch12:
    def test_docs_sites_map_to_their_source(self):
        from searchio.leanread import lean_targets

        s = Settings(lean_read_github_wiki=True, lean_read_hf_docs=True, lean_read_react=True,
                     lean_read_vue=True, lean_read_node_api=True, lean_read_django=True,
                     lean_read_fastapi=True, lean_read_pydantic=True)
        raw = "https://raw.githubusercontent.com/"

        def urls(u):
            return [x for x, _ in lean_targets(u, s)]

        assert urls("https://github.com/o/r/wiki") == [raw + "wiki/o/r/Home.md"]
        assert urls("https://huggingface.co/docs/peft/main/en/quicktour") == [
            raw + "huggingface/peft/main/docs/source/en/quicktour.md",
            raw + "huggingface/peft/main/docs/source/en/quicktour.mdx",
            raw + "huggingface/peft/main/docs/source/quicktour.md",
            raw + "huggingface/peft/main/docs/source/quicktour.mdx"]
        assert urls("https://huggingface.co/docs/unknownlib/x") == []
        assert urls("https://react.dev/reference/react")[1] == \
            raw + "reactjs/react.dev/main/src/content/reference/react/index.md"
        assert urls("https://vuejs.org/guide/essentials/computed.html")[0] == \
            raw + "vuejs/docs/main/src/guide/essentials/computed.md"
        assert urls("https://nodejs.org/api/events.html") == [raw + "nodejs/node/main/doc/api/events.md"]
        assert urls("https://docs.djangoproject.com/en/5.1/topics/db/queries/")[0] == \
            raw + "django/django/stable/5.1.x/docs/topics/db/queries.txt"
        assert urls("https://docs.djangoproject.com/en/dev/ref/models/")[1] == \
            raw + "django/django/main/docs/ref/models/index.txt"
        assert urls("https://fastapi.tiangolo.com/es/tutorial/") == [
            raw + "fastapi/fastapi/master/docs/es/docs/tutorial.md",
            raw + "fastapi/fastapi/master/docs/es/docs/tutorial/index.md"]
        assert urls("https://docs.pydantic.dev/latest/concepts/models/")[0] == \
            raw + "pydantic/pydantic/main/docs/concepts/models.md"
        assert urls("https://docs.pydantic.dev/latest/api/base_model/") == []

    def test_source_titles(self):
        from searchio.leanread import _md_title

        assert _md_title("---\ntitle: useState\n---\nbody", "x") == "useState"
        assert _md_title("# Computed Properties {#computed-properties}\n", "x") == "Computed Properties"
        assert _md_title("==============\nMaking queries\n==============\n\ntext", "x") == "Making queries"
        assert _md_title("no heading", "first-steps") == "First steps"


class TestMarkdownAlternate:
    def test_alternate_and_template(self):
        from searchio.leanread import alternate_template, apply_template, markdown_alternate

        page = "https://docs.x.com/guides/auth/"
        html = ('<html><head><link rel="stylesheet" href="a.css">'
                '<link rel=alternate type=text/markdown href=/guides/auth.md></head><body>'
                '<link rel="alternate" type="text/markdown" href="/not-in-head.md"></body></html>')
        alt = markdown_alternate(page, html)
        assert alt == "https://docs.x.com/guides/auth.md"
        tpl = alternate_template(page, alt)
        assert tpl == "https://docs.x.com{path}.md"
        assert apply_template(tpl, "https://docs.x.com/guides/db") == "https://docs.x.com/guides/db.md"
        assert apply_template(tpl, "https://docs.x.com/") is None
        gh = alternate_template("https://d.com/en/a/b",
                                "https://d.com/api/article/body?pathname=/en/a/b")
        assert apply_template(gh, "https://d.com/en/c") == "https://d.com/api/article/body?pathname=/en/c"
        assert alternate_template(page, "https://docs.x.com/other-page.md") is None
        assert markdown_alternate(page, "<html><head></head></html>") is None

    async def test_one_read_teaches_the_host(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, lean_md_alternate=True))
        fetched = []
        page = ('<html><head><title>P</title><link rel="alternate" type="text/markdown" '
                'href="/docs/a.md"></head><body><p>' + "word " * 100 + "</p></body></html>")

        class R:
            def __init__(self, url, body, status=200):
                self.final_url, self.body, self.via, self.status = url, body, "http", status

        async def fake_fetch(url, **kw):
            fetched.append(url)
            if url.endswith(".md"):
                return R(url, "# Page B\n\n" + "text " * 60)
            return R(url, page)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        await eng.read("https://h.com/docs/a")
        d = await eng.read("https://h.com/docs/b")
        assert fetched == ["https://h.com/docs/a", "https://h.com/docs/b.md"]
        assert d.title == "Page B" and d.fetched_via == "http+lean" and d.url == "https://h.com/docs/b"
        eng2 = Engine(Settings(state_dir=tmp_path, cache_enabled=False, lean_md_alternate=True))
        assert eng2._alt_targets("https://h.com/docs/c")[0][0] == "https://h.com/docs/c.md", \
            "the rule persists across processes"
        await eng.close()
        await eng2.close()


class TestEditLinkRule:
    def test_rule_from_edit_link(self):
        from searchio.leanread import apply_edit_rule, edit_rule

        html = ('<a href="https://github.com/o/site/blob/main/CODE_OF_CONDUCT.md">coc</a>'
                '<a href="https://github.com/o/r/edit/main/docs/querying/basics.md">Edit</a>')
        rule = edit_rule("https://h.io/docs/prom/latest/querying/basics/", html)
        assert rule["page_prefix"] == "docs/prom/latest"
        assert apply_edit_rule(rule, "https://h.io/docs/prom/latest/querying/functions/") == [
            "https://raw.githubusercontent.com/o/r/main/docs/querying/functions.md",
            "https://raw.githubusercontent.com/o/r/main/docs/querying/functions/index.md"]
        assert apply_edit_rule(rule, "https://h.io/blog/post") == []
        assert edit_rule("https://h.io/guide/",
                         '<a href="https://github.com/o/r/blob/main/CONTRIBUTING.md">c</a>') is None
        idx = edit_rule("https://t.app/start/",
                        '<a href="https://github.com/t/docs/edit/v2/src/content/docs/start/index.mdx">e</a>')
        assert apply_edit_rule(idx, "https://t.app/start/prerequisites/")[0] == \
            "https://raw.githubusercontent.com/t/docs/v2/src/content/docs/start/prerequisites/index.mdx"

    async def test_redirected_host_learns_too(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, lean_edit_link=True))
        html = ('<html><head><title>G</title></head><body><a href="https://github.com/v/v/edit/main/'
                'docs/guide/index.md">Edit</a><p>' + "w " * 200 + "</p></body></html>")

        class R:
            def __init__(self, url, body, status=200):
                self.final_url, self.body, self.via, self.status = url, body, "http", status

        async def fake_fetch(url, **kw):
            if "raw.githubusercontent" in url:
                return R(url, "# Config\n\n" + "text " * 60)
            return R(url.replace("old.dev", "new.dev"), html)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        await eng.read("https://old.dev/guide/")
        d = await eng.read("https://old.dev/config/")
        assert d.fetched_via == "http+lean" and d.title == "Config"
        await eng.close()


class TestFastCli:
    def test_parse_forwards_only_what_it_knows(self):
        from searchio.fastcli import _parse_search

        assert _parse_search(["rust async", "-i", "code", "-k", "5", "--json"]) == \
            {"q": "rust async", "intent": "code", "k": "5"}
        assert _parse_search(["rust async"]) is None, "table output stays rich's"
        assert _parse_search(["q", "--json", "--no-cache"]) is None
        assert _parse_search(["q", "extra", "--json"]) is None
        assert _parse_search(["--json"]) is None

    def test_forward_prints_the_cli_json_shape(self, capsys):
        import http.server
        import json as js
        import threading

        from searchio.fastcli import _forward

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if "q=down" in self.path:
                    body = js.dumps({"detail": {"error": "all providers failed",
                                                "providers_failed": {"ddg": "walled"}}}).encode()
                    self.send_response(503)
                else:
                    body = js.dumps({"query": "x", "providers_used": ["ddg"], "elapsed_ms": 9,
                                     "providers_failed": {}, "providers_skipped": {},
                                     "results": [{"url": "https://a/"}]}).encode()
                    self.send_response(200)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        try:
            assert _forward(base, {"q": "x"}) == 0
            out = js.loads(capsys.readouterr().out)
            assert out["used"] == ["ddg"] and out["results"] == [{"url": "https://a/"}]
            assert _forward(base, {"q": "down"}) == 1
            assert "ddg: walled" in capsys.readouterr().err
        finally:
            srv.shutdown()
            srv.server_close()
        assert _forward(base, {"q": "x"}) is None, "no daemon: the full CLI runs"


class TestDnsRace:
    @staticmethod
    def _reply(query: bytes, ips: list[str], rcode: int = 0) -> bytes:
        import socket as so
        import struct as st

        tid = query[:2]
        q = query[12:]
        ans = b"".join(b"\xc0\x0c" + st.pack(">HHIH", 1, 1, 60, 4) + so.inet_aton(ip) for ip in ips)
        return tid + st.pack(">HHHHH", 0x8180 | rcode, 1, len(ips), 0, 0) + q + ans

    def _stub(self, ips, rcode=0):
        import socket as so
        import threading

        s = so.socket(so.AF_INET, so.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))

        def serve():
            while True:
                try:
                    data, addr = s.recvfrom(512)
                except OSError:
                    return
                s.sendto(self._reply(data, ips, rcode), addr)

        threading.Thread(target=serve, daemon=True).start()
        return s

    def test_parse_a(self):
        from searchio.net.dnsrace import _query, parse_a

        tid, q = _query("example.com")
        assert parse_a(self._reply(q, ["1.2.3.4", "5.6.7.8"]), tid) == ["1.2.3.4", "5.6.7.8"]
        assert parse_a(self._reply(q, [], rcode=3), tid) == [], "NXDOMAIN is never an answer"
        assert parse_a(self._reply(q, ["1.2.3.4"]), tid ^ 1) == [], "someone else's reply"

    def test_race_beats_a_stalled_resolver_and_defers_on_nxdomain(self, monkeypatch):
        import socket as so
        import time as ti

        from searchio.net import dnsrace

        stub = self._stub(["93.184.216.34"])
        monkeypatch.setattr(dnsrace, "_PORT", stub.getsockname()[1])
        monkeypatch.setattr(dnsrace, "_SERVERS", ["127.0.0.1"])
        monkeypatch.setattr(dnsrace, "_HOSTS", {"pinned.example"})
        monkeypatch.setattr(dnsrace, "NX_FAST", [False])  # the OFF path: the OS decides

        def slow_gai(host, port, *a, **kw):
            ti.sleep(1.0)
            return [(so.AF_INET, so.SOCK_STREAM, 6, "", ("10.9.9.9", port))]

        monkeypatch.setattr(so, "getaddrinfo", slow_gai)
        t = ti.monotonic()
        got = dnsrace.resolve("fast.example", 443)
        assert [g[4] for g in got] == [("93.184.216.34", 443)] and ti.monotonic() - t < 0.5
        assert dnsrace.resolve("pinned.example", 443)[0][4][0] == "10.9.9.9", "hosts file: OS only"
        stub.close()
        nx = self._stub([], rcode=3)
        monkeypatch.setattr(dnsrace, "_PORT", nx.getsockname()[1])
        assert dnsrace.resolve("gone.example", 443)[0][4][0] == "10.9.9.9", "NXDOMAIN: the OS decides"
        nx.close()

    def test_nx_fast_trusts_only_the_machines_own_servers(self, monkeypatch):
        import socket as so
        import time as ti

        from searchio.net import dnsrace

        nx = self._stub([], rcode=3)
        monkeypatch.setattr(dnsrace, "_PORT", nx.getsockname()[1])
        monkeypatch.setattr(dnsrace, "_HOSTS", set())
        monkeypatch.setattr(dnsrace, "NX_FAST", [True])

        def slow_gai(host, port, *a, **kw):
            ti.sleep(1.0)
            return [(so.AF_INET, so.SOCK_STREAM, 6, "", ("10.9.9.9", port))]

        monkeypatch.setattr(so, "getaddrinfo", slow_gai)
        try:
            monkeypatch.setattr(dnsrace, "_SERVERS", ["127.0.0.1"])
            t = ti.monotonic()
            with pytest.raises(so.gaierror, match="NXDOMAIN"):
                dnsrace.resolve("gone.example", 443)
            assert ti.monotonic() - t < 0.5
            # An NX from an EXTRA (public) server is not the machine's word:
            # a split-horizon name may exist only on the configured resolver.
            monkeypatch.setattr(dnsrace, "_SERVERS", [])
            monkeypatch.setattr(dnsrace, "system_servers", lambda: ["192.0.2.53"])
            got = dnsrace.resolve("intranet.example", 443, extra_servers=["127.0.0.1"])
            assert got[0][4][0] == "10.9.9.9"
        finally:
            nx.close()

    def test_servfail_fast_needs_every_system_server(self, monkeypatch):
        import socket as so
        import time as ti

        from searchio.net import dnsrace

        sf = self._stub([], rcode=2)
        monkeypatch.setattr(dnsrace, "_PORT", sf.getsockname()[1])
        monkeypatch.setattr(dnsrace, "_HOSTS", set())
        monkeypatch.setattr(dnsrace, "_SERVERS", ["127.0.0.1"])

        def slow_gai(host, port, *a, **kw):
            ti.sleep(1.0)
            return [(so.AF_INET, so.SOCK_STREAM, 6, "", ("10.9.9.9", port))]

        monkeypatch.setattr(so, "getaddrinfo", slow_gai)
        try:
            monkeypatch.setattr(dnsrace, "SERVFAIL_FAST", [False])
            assert dnsrace.resolve("broken.example", 443)[0][4][0] == "10.9.9.9", "off: the OS decides"
            monkeypatch.setattr(dnsrace, "SERVFAIL_FAST", [True])
            t = ti.monotonic()
            with pytest.raises(so.gaierror, match="SERVFAIL"):
                dnsrace.resolve("broken.example", 443)
            assert ti.monotonic() - t < 0.5
            # A second system server that has not answered keeps the OS in charge.
            monkeypatch.setattr(dnsrace, "system_servers", lambda: ["127.0.0.1", "192.0.2.53"])
            got = dnsrace.resolve("broken2.example", 443)
            assert got[0][4][0] == "10.9.9.9"
        finally:
            sf.close()

    def test_ladder_vets_raced_answers(self, tmp_path, monkeypatch):
        from searchio.errors import TargetRefused
        from searchio.net import dnsrace

        ladder_mod.Ladder(Settings(state_dir=tmp_path, dns_race=True))
        monkeypatch.setattr(dnsrace, "resolve",
                            lambda h, p, extra=None: [(2, 1, 6, "", ("169.254.169.254", p))])
        with pytest.raises(TargetRefused):
            ladder_mod._resolve_checked("metadata.example", 443)
        ladder_mod._DNS_RACE[0] = False


class TestAcceptMarkdown:
    async def test_markdown_reply_is_the_doc(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.net.ladder import ACCEPT_MD

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, accept_markdown=True,
                              lean_md_alternate=False))
        asked = []

        class R:
            def __init__(self, url, body, ctype):
                self.url = self.final_url = url
                self.body, self.content_type, self.via, self.status = body, ctype, "http", 200

        async def fake_fetch(url, **kw):
            asked.append(ACCEPT_MD.get())
            if "md-site" in url:
                return R(url, "# Guide\n\n" + "text " * 60, "text/markdown; charset=utf-8")
            return R(url, "<html><head><title>H</title></head><body><p>"
                     + "word " * 80 + "</p></body></html>", "text/html")

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("https://md-site.com/docs/guide")
        assert d.title == "Guide" and d.fetched_via == "http+md"
        h = await eng.read("https://plain.com/page")
        assert h.title == "H" and h.fetched_via == "http"
        assert asked == [True, True] and ACCEPT_MD.get() is False
        await eng.close()

    async def test_off_never_asks(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.net.ladder import ACCEPT_MD

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, lean_md_alternate=False,
                              accept_markdown=False))
        asked = []

        class R:
            final_url, via, status, content_type = "https://x.com/a", "http", 200, "text/markdown"
            body = "# Not used\n\n" + "text " * 60

        async def fake_fetch(url, **kw):
            asked.append(ACCEPT_MD.get())
            return R()

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("https://x.com/a")
        assert asked == [False] and d.fetched_via == "http"
        await eng.close()

    async def test_tier0_header_and_cache_variant(self, tmp_path, monkeypatch):
        from searchio.errors import TransientError
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False))
        seen = []

        async def fake_open(url, headers, timeout):
            seen.append(headers["Accept"])
            raise TransientError("stop")

        monkeypatch.setattr(lad, "_open_hedged", fake_open)
        assert lad._cache_variant(False) == ""
        with pytest.raises(TransientError):
            await lad._tier0_document("https://h.com/a")
        tok = lm.ACCEPT_MD.set(True)
        try:
            assert lad._cache_variant(False) == "md" and lad._cache_variant(True) == "rendered|md"
            with pytest.raises(TransientError):
                await lad._tier0_document("https://h.com/a")
        finally:
            lm.ACCEPT_MD.reset(tok)
        assert not seen[0].startswith("text/markdown") and seen[1].startswith("text/markdown")
        await lad.close()

    async def test_stub_reply_falls_back_to_html(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.net.ladder import ACCEPT_MD

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, accept_markdown=True,
                              lean_md_alternate=False))
        asked = []

        class R:
            def __init__(self, body, ctype):
                self.final_url, self.via, self.status = "https://d.org/x.html", "http", 200
                self.body, self.content_type = body, ctype

        async def fake_fetch(url, **kw):
            asked.append(ACCEPT_MD.get())
            if ACCEPT_MD.get():
                return R("[Continue to ../2.14/x.html](https://d.org/2.14/x.html)", "text/markdown")
            return R("<html><head><title>Linear</title></head><body><p>"
                     + "word " * 80 + "</p></body></html>", "text/html")

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("https://d.org/x.html")
        assert asked == [True, False] and d.title == "Linear" and d.fetched_via == "http"
        await eng.close()

    def test_plain_markdown_only_when_enabled(self):
        from searchio.engine import _markdown_reply

        class R:
            def __init__(self, body, ctype):
                self.body, self.content_type = body, ctype

        md = R("# Title (/docs/x)\n\ntext", "text/plain; charset=utf-8")
        assert not _markdown_reply(md) and _markdown_reply(md, True)
        assert _markdown_reply(R("---\ntitle: x\n---\n", "text/plain"), True)
        assert not _markdown_reply(R("User-agent: *\nDisallow: /", "text/plain"), True)
        assert _markdown_reply(R("anything", "text/markdown"))
        assert not _markdown_reply(R("<html>", "text/html"), True)


def test_anubis_interstitial_is_a_wall():
    from searchio.net.blocks import classify

    body = ('<!doctype html><html><head><title>Making sure you&#39;re not a bot!</title>'
            '<script id="anubis_challenge" type="application/json">{"rules":{}}</script>'
            '</head><body><p>Loading...</p>' + "<!-- pad -->" * 200 + '</body></html>')
    v = classify(200, {"content-type": "text/html"}, body, content_type="text/html")
    assert not v.ok and v.blocked


class TestDeadIpCache:
    def _reset(self, lm, ttl):
        lm._DEAD_IPS.clear()
        lm._DEAD_HOSTS.clear()
        lm._DEAD_HOST_ON[0] = False
        lm._DEAD_TTL[0] = ttl
        lm._LAST_DIAL_OK[0] = 0.0

    def _infos(self, *ips):
        import socket

        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ips]

    def test_marks_only_while_the_link_is_up(self):
        from searchio.net import ladder as lm

        self._reset(lm, 300)
        try:
            lm._note_dial("192.0.2.1", False)  # no dial has succeeded: maybe the link
            assert lm._DEAD_IPS == {}
            lm._note_dial("198.51.100.9", True)
            lm._note_dial("192.0.2.1", False)
            assert "192.0.2.1" in lm._DEAD_IPS
            live = lm._live_infos("h.com", self._infos("192.0.2.1", "192.0.2.2"))
            assert [i[4][0] for i in live] == ["192.0.2.2"]
            with pytest.raises(lm.TransientError, match="unreachable"):
                lm._live_infos("h.com", self._infos("192.0.2.1"))
            lm._note_dial("192.0.2.1", True)  # it answered after all
            assert lm._live_infos("h.com", self._infos("192.0.2.1"))
        finally:
            self._reset(lm, 0.0)

    def test_off_changes_nothing(self):
        from searchio.net import ladder as lm

        self._reset(lm, 0.0)
        lm._LAST_DIAL_OK[0] = 1e18
        lm._note_dial("192.0.2.1", False)
        assert lm._DEAD_IPS == {}
        infos = self._infos("192.0.2.1")
        assert lm._live_infos("h.com", infos) is infos

    async def _dial_dead(self, monkeypatch, probe_ok):
        import time

        import anyio

        from searchio.net import ladder as lm

        self._reset(lm, 300)
        lm._LAST_DIAL_OK[0] = time.monotonic()
        lm._LAST_OK_ADDR[0] = ("198.51.100.1", 443)
        monkeypatch.setattr(lm, "_resolve_checked", lambda h, p: self._infos("192.0.2.7"))
        calls = []

        class S:
            async def aclose(self):
                pass

        async def connect(remote_host, remote_port, local_host=None):
            calls.append(remote_host)
            if remote_host == "198.51.100.1" and probe_ok:
                return S()
            await anyio.sleep(5)

        monkeypatch.setattr(anyio, "connect_tcp", connect)
        monkeypatch.setattr(lm, "_PROBE_S", 0.05)
        return lm, lm._guarded_async_backend(), calls

    async def test_dial_timeout_marks_once_the_link_is_proven(self, monkeypatch):
        import time

        from httpcore import ConnectError, ConnectTimeout

        lm, backend, calls = await self._dial_dead(monkeypatch, probe_ok=True)
        try:
            with pytest.raises(ConnectTimeout):
                await backend.connect_tcp("dead.example", 443, timeout=0.05)
            assert calls == ["192.0.2.7", "198.51.100.1"], "the link-up probe ran"
            t = time.monotonic()
            with pytest.raises(ConnectError, match="unreachable"):
                await backend.connect_tcp("dead.example", 443, timeout=0.05)
            assert time.monotonic() - t < 0.04 and len(calls) == 2
        finally:
            self._reset(lm, 0.0)
            lm._LAST_OK_ADDR[0] = None

    async def test_rotating_dns_cannot_rearm_a_dead_host(self, monkeypatch):
        from httpcore import ConnectError, ConnectTimeout

        for on in (True, False):
            lm, backend, calls = await self._dial_dead(monkeypatch, probe_ok=True)
            lm._DEAD_HOST_ON[0] = on
            try:
                with pytest.raises(ConnectTimeout):
                    await backend.connect_tcp("dead.example", 443, timeout=0.05)
                # The CDN hands out a fresh address set on the next query.
                monkeypatch.setattr(lm, "_resolve_checked", lambda h, p: self._infos("192.0.2.8"))
                if on:
                    with pytest.raises(ConnectError, match="unreachable"):
                        await backend.connect_tcp("dead.example", 443, timeout=0.05)
                    assert "192.0.2.8" not in calls
                else:
                    with pytest.raises(ConnectTimeout):
                        await backend.connect_tcp("dead.example", 443, timeout=0.05)
                    assert "192.0.2.8" in calls
            finally:
                self._reset(lm, 0.0)
                lm._LAST_OK_ADDR[0] = None

    async def test_fresh_process_proves_the_link_with_the_probe(self, monkeypatch):
        from httpcore import ConnectError, ConnectTimeout

        # A filtered target (1.1.1.1 on the bench network, #129) beside one
        # that answers: the race still proves the link.
        for probes in ([("203.0.113.9", 443), ("198.51.100.1", 443)], []):
            lm, backend, calls = await self._dial_dead(monkeypatch, probe_ok=True)
            lm._LAST_OK_ADDR[0] = None  # nothing has answered in this process yet
            lm._LINK_PROBES[:] = probes
            try:
                with pytest.raises(ConnectTimeout):
                    await backend.connect_tcp("dead.example", 443, timeout=0.05)
                if probes:
                    assert "198.51.100.1" in calls
                    with pytest.raises(ConnectError, match="unreachable"):
                        await backend.connect_tcp("dead.example", 443, timeout=0.05)
                else:
                    assert calls == ["192.0.2.7"] and lm._DEAD_IPS == {}
            finally:
                self._reset(lm, 0.0)
                lm._LINK_PROBES.clear()

    async def test_link_down_marks_nothing(self, monkeypatch):
        from httpcore import ConnectTimeout

        lm, backend, calls = await self._dial_dead(monkeypatch, probe_ok=False)
        try:
            with pytest.raises(ConnectTimeout):
                await backend.connect_tcp("shared-cdn.example", 443, timeout=0.05)
            assert lm._DEAD_IPS == {}, "a drop times out every dial: no address is to blame"
        finally:
            self._reset(lm, 0.0)
            lm._LAST_OK_ADDR[0] = None

    def test_persisted_across_processes(self, tmp_path):
        import json
        import time

        from searchio.net import dnsrace
        from searchio.net import ladder as lm

        self._reset(lm, 300)
        try:
            lm._DEAD_PATH[0] = tmp_path / "dead_ips.json"
            dnsrace.LAST_UDP_OK[0] = time.monotonic()  # a fresh process: DNS proved the link
            lm._note_dial("192.0.2.5", False)
            saved = json.loads((tmp_path / "dead_ips.json").read_text())
            assert set(saved) == {"192.0.2.5"}
            lm._DEAD_IPS.clear()
            lm._load_dead(tmp_path / "dead_ips.json")
            with pytest.raises(lm.TransientError):
                lm._live_infos("h.com", self._infos("192.0.2.5"))
            (tmp_path / "dead_ips.json").write_text(json.dumps({"192.0.2.6": time.time() - 1}))
            lm._DEAD_IPS.clear()
            lm._load_dead(tmp_path / "dead_ips.json")
            assert lm._DEAD_IPS == {}, "expired entries stay dead in the file only"
        finally:
            lm._DEAD_PATH[0] = None
            dnsrace.LAST_UDP_OK[0] = 0.0
            self._reset(lm, 0.0)


class TestRedirectMemo:
    async def test_learn_confirm_apply_and_fallback(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, redirect_memo=True,
                              accept_markdown=False, lean_md_alternate=False))
        fetched = []
        page = "<html><head><title>P</title></head><body><p>" + "word " * 80 + "</p></body></html>"

        class R:
            def __init__(self, url, final, status=200):
                self.url, self.final_url, self.status = url, final, status
                self.body, self.via, self.content_type = page, "http", "text/html"

        async def fake_fetch(url, **kw):
            fetched.append(url)
            if url.startswith("https://old.org/dev/peps/"):
                return R(url, "https://peps.org/" + url.rsplit("/peps/", 1)[1])
            if "gone" in url:
                return R(url, url, 404)
            return R(url, url)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        await eng.read("https://old.org/dev/peps/pep-1/")
        assert eng._redirect_target("https://old.org/dev/peps/pep-9/") is None, "one read is not a rule"
        await eng.read("https://old.org/dev/peps/pep-2/")
        assert eng._redirect_target("https://old.org/dev/peps/pep-9/") == "https://peps.org/pep-9/"
        fetched.clear()
        d = await eng.read("https://old.org/dev/peps/pep-3/")
        assert fetched == ["https://peps.org/pep-3/"] and d.url == "https://peps.org/pep-3/"
        fetched.clear()
        d = await eng.read("https://old.org/dev/peps/gone/")  # the target misses: read as asked
        assert fetched == ["https://peps.org/gone/", "https://old.org/dev/peps/gone/"]
        assert eng._redirect_target("https://old.org/dev/peps/pep-9/") is None, "a miss forgets"
        await eng.close()

    def test_rules_skip_tail_changes_and_inserted_prefixes(self, tmp_path):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, redirect_memo=True))
        eng._redirect_learn("https://m.org/a/x", "https://m.org/login?next=/a/x")
        eng._redirect_learn("https://m.org/a/x", "https://m.org/a/x/")
        assert eng._redirects() == {}
        for u in ("https://m.org/docs/a", "https://m.org/docs/b"):
            eng._redirect_learn(u, u.replace("m.org/", "m.org/en-US/"))
        assert eng._redirect_target("https://m.org/docs/c") == "https://m.org/en-US/docs/c"
        assert eng._redirect_target("https://m.org/en-US/docs/c") is None

    async def test_https_upgrade_with_fallback(self, tmp_path, monkeypatch):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, cache_enabled=False, https_upgrade=True,
                              redirect_memo=False, accept_markdown=False, lean_md_alternate=False))
        fetched = []
        page = "<html><head><title>P</title></head><body><p>" + "word " * 80 + "</p></body></html>"

        class R:
            def __init__(self, url, status=200):
                self.url = self.final_url = url
                self.status, self.body, self.via, self.content_type = status, page, "http", "text/html"

        async def fake_fetch(url, **kw):
            fetched.append(url)
            if url.startswith("https://plain-only."):
                raise ConnectionError("no tls here")
            return R(url)

        monkeypatch.setattr(eng.ladder, "fetch", fake_fetch)
        d = await eng.read("http://a.org/x")
        assert fetched == ["https://a.org/x"] and d.url == "https://a.org/x"
        fetched.clear()
        d = await eng.read("http://plain-only.org/x")
        assert fetched == ["https://plain-only.org/x", "http://plain-only.org/x"]
        await eng.close()

    def test_host_move_applies_after_one_read(self, tmp_path):
        from searchio.engine import Engine

        on = Engine(Settings(state_dir=tmp_path / "a", cache_enabled=False, redirect_host_once=True))
        on._redirect_learn("https://g.org/doc/faq", "https://go.dev/doc/faq")
        assert on._redirect_target("https://g.org/doc/code") == "https://go.dev/doc/code"
        on._redirect_learn("https://p.org/dev/peps/pep-1/", "https://peps.org/pep-1/")
        assert on._redirect_target("https://p.org/dev/peps/pep-2/") is None, "a prefix move still needs 2"
        off = Engine(Settings(state_dir=tmp_path / "b", cache_enabled=False, redirect_host_once=False))
        off._redirect_learn("https://g.org/doc/faq", "https://go.dev/doc/faq")
        assert off._redirect_target("https://g.org/doc/code") is None


class TestCrawlDelayInterval:
    async def _run(self, tmp_path, monkeypatch, interval, n):
        import time

        from searchio.models import FetchResult
        from searchio.net import ladder as lm
        from searchio.net.robots import RobotsInfo

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="enforce",
                                 per_domain_rps=1000.0, per_domain_burst=1000,
                                 crawl_delay_interval=interval))

        async def check(url):
            return RobotsInfo(allowed=True, checked=True, crawl_delay=0.3)

        page = "<html><head><title>T</title></head><body><p>" + "word " * 80 + "</p></body></html>"
        stamps = []

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            stamps.append(time.monotonic())
            return 200, {"content-type": "text/html"}, page, "text/html", url

        monkeypatch.setattr(lad.robots, "check", check)
        monkeypatch.setattr(lad, "_try_tier", try_tier)
        t = time.monotonic()
        rs = await asyncio.gather(*(lad.fetch(f"https://slow.org/{i}", use_cache=False, max_tier=0)
                                    for i in range(n)))
        await lad.close()
        assert all(isinstance(r, FetchResult) and r.status == 200 for r in rs)
        return [s - t for s in sorted(stamps)]

    async def test_first_request_does_not_wait_and_rate_is_kept(self, tmp_path, monkeypatch):
        at = await self._run(tmp_path / "a", monkeypatch, True, 3)
        assert at[0] < 0.1, "no gap to keep before the first request"
        assert at[1] - at[0] >= 0.28 and at[2] - at[1] >= 0.28, "still one delay apart"
        old = await self._run(tmp_path / "b", monkeypatch, False, 1)
        assert old[0] >= 0.28, "the old toll: a sleep before every request"


class TestEscalateNoWait:
    async def _permits(self, tmp_path, monkeypatch, on):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 escalate_no_wait=on))
        page = "<html><head><title>T</title></head><body><p>" + "word " * 80 + "</p></body></html>"
        wall = "<html><head><title>Just a moment...</title></head><body></body></html>"
        permits = []

        async def acquire(dom, *a):
            permits.append(dom)

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            if tier == 0:
                return 403, {"content-type": "text/html", "server": "cloudflare"}, wall, "text/html", url
            return 200, {"content-type": "text/html"}, page, "text/html", url

        monkeypatch.setattr(lad.limiter, "acquire", acquire)
        monkeypatch.setattr(lad, "_try_tier", try_tier)
        r = await lad.fetch("https://walled.org/a", use_cache=False, max_tier=1)
        await lad.close()
        assert r.status == 200 and r.tier == 1
        return len(permits)

    async def test_climb_after_a_block_pays_one_permit(self, tmp_path, monkeypatch):
        assert await self._permits(tmp_path / "on", monkeypatch, True) == 1
        assert await self._permits(tmp_path / "off", monkeypatch, False) == 2


class TestSkipDeadTier:
    async def _climb(self, tmp_path, monkeypatch, on):
        from searchio.errors import Blocked
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 escalate_no_wait=False, skip_dead_tier=on))
        lad.sidecar._unavailable_reason = "browser_sidecar.py not found"
        wall = "<html><head><title>Just a moment...</title></head><body></body></html>"
        permits, tiers = [], []

        async def acquire(dom, *a):
            permits.append(dom)

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            if tier == 2:
                raise lm.SidecarUnavailable("browser_sidecar.py not found")
            return 403, {"content-type": "text/html", "server": "cloudflare"}, wall, "text/html", url

        monkeypatch.setattr(lad.limiter, "acquire", acquire)
        monkeypatch.setattr(lad, "_try_tier", try_tier)
        with pytest.raises(Blocked) as ei:
            await lad.fetch("https://walled.org/a", use_cache=False, max_tier=2)
        await lad.close()
        assert "tier2:sidecar_unavailable" in str(ei.value)
        return len(permits), tiers

    async def test_known_dead_sidecar_costs_no_permit(self, tmp_path, monkeypatch):
        assert await self._climb(tmp_path / "on", monkeypatch, True) == (2, [0, 1])
        assert await self._climb(tmp_path / "off", monkeypatch, False) == (3, [0, 1, 2])


class TestWallRefund:
    def test_refund_restores_rate_and_burst_less_one(self):
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=0.75, burst=3)
        snap = lim.snapshot("a.org")
        lim.record_block("a.org")
        assert lim.rate_for("a.org") < 0.75
        lim.refund("a.org", snap)
        b = lim._bucket("a.org")
        assert lim.rate_for("a.org") == 0.75 and 1.99 <= b.tokens <= 2.01

    async def _rate_after(self, tmp_path, monkeypatch, on, tier1_ok=True):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 wall_refund=on))
        page = "<html><head><title>T</title></head><body><p>" + "word " * 80 + "</p></body></html>"
        wall = "<html><head><title>Just a moment...</title></head><body></body></html>"

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            if tier == 0 or not tier1_ok:
                return 403, {"content-type": "text/html", "server": "cloudflare"}, wall, "text/html", url
            return 200, {"content-type": "text/html"}, page, "text/html", url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        try:
            await lad.fetch("https://walled.org/a", use_cache=False, max_tier=1)
        except Exception:  # noqa: BLE001 -- the failing climb is the point
            pass
        rate = lad.limiter.rate_for("walled.org")
        await lad.close()
        return rate

    async def test_fingerprint_wall_passed_above_costs_no_backoff(self, tmp_path, monkeypatch):
        assert await self._rate_after(tmp_path / "on", monkeypatch, True) >= 0.75
        assert await self._rate_after(tmp_path / "off", monkeypatch, False) < 0.75

    async def test_a_wall_every_tier_drew_still_backs_off(self, tmp_path, monkeypatch):
        assert await self._rate_after(tmp_path, monkeypatch, True, tier1_ok=False) < 0.75


class TestSidecarFastProbe:
    async def test_closed_local_port_is_known_fast(self):
        import asyncio
        import socket
        import time

        from searchio.net import sidecar as sc

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens there now
        t = time.monotonic()
        assert await sc._port_open(f"http://127.0.0.1:{port}") is False
        assert time.monotonic() - t < 0.5
        srv = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
        live = srv.sockets[0].getsockname()[1]
        try:
            assert await sc._port_open(f"http://127.0.0.1:{live}") is True
        finally:
            srv.close()
        assert await sc._port_open("http://sidecar.example:9000") is True  # remote: HTTP decides


class TestThinSkipTier1:
    async def _tiers(self, tmp_path, monkeypatch, on, body):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 thin_skip_tier1=on))
        tiers = []

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            return 200, {"content-type": "text/html"}, body, "text/html", url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        try:
            await lad.fetch("https://spa.org/a", use_cache=False, max_tier=1)
        except Exception:  # noqa: BLE001 -- an empty shell has no fallback
            pass
        await lad.close()
        return tiers

    async def test_js_shell_skips_tier1(self, tmp_path, monkeypatch):
        shell = ("<html><head><title>App</title></head><body><div id='root'></div>"
                 "<p>Loading app</p>" + "<script>var x=1;</script>" * 40 + "</body></html>")
        assert await self._tiers(tmp_path / "on", monkeypatch, True, shell) == [0]
        assert await self._tiers(tmp_path / "off", monkeypatch, False, shell) == [0, 1]

    async def test_a_wall_still_climbs(self, tmp_path, monkeypatch):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 thin_skip_tier1=True))
        tiers = []
        wall = "<html><head><title>Just a moment...</title></head><body></body></html>"
        page = "<html><head><title>T</title></head><body><p>" + "word " * 80 + "</p></body></html>"

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            if tier == 0:
                return 403, {"content-type": "text/html", "server": "cloudflare"}, wall, "text/html", url
            return 200, {"content-type": "text/html"}, page, "text/html", url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        r = await lad.fetch("https://walled.org/a", use_cache=False, max_tier=1)
        await lad.close()
        assert r.tier == 1 and tiers == [0, 1]


class TestMdStubPassthrough:
    async def _fetch(self, tmp_path, monkeypatch, on, md):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 md_stub_passthrough=on))
        tiers = []
        ctype = "text/markdown" if md else "text/html"

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            return 200, {"content-type": ctype}, "# Get started\n\n", ctype, url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        tok = lm.ACCEPT_MD.set(True)
        try:
            r = await lad.fetch("https://docs.example.com/a", use_cache=False, max_tier=1)
        except Exception:  # noqa: BLE001 -- the climb found nothing better
            r = None
        finally:
            lm.ACCEPT_MD.reset(tok)
        await lad.close()
        return r, tiers

    async def test_markdown_stub_returns_to_the_reader(self, tmp_path, monkeypatch):
        r, tiers = await self._fetch(tmp_path / "on", monkeypatch, True, True)
        assert tiers == [0] and r.body.startswith("# Get started") and "md_stub" in r.escalations
        r, _ = await self._fetch(tmp_path / "off", monkeypatch, False, True)
        assert r is None or "md_stub" not in r.escalations

    async def test_a_thin_html_page_still_climbs(self, tmp_path, monkeypatch):
        r, _ = await self._fetch(tmp_path, monkeypatch, True, False)
        assert r is None or "md_stub" not in r.escalations


class TestAddrsetTierMemo:
    def _res(self, tier, esc):
        from types import SimpleNamespace

        return SimpleNamespace(tier=tier, escalations=esc, from_cache=False)

    async def test_two_refusals_on_one_address_set_open_the_next_host_at_tier1(self, tmp_path):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, addrset_tier_memo=True))
        pair = ("162.159.141.224", "172.66.1.220")
        for h in ("a.com", "b.com", "c.com", "d.com"):
            lm._HOST_ADDRS["www." + h] = pair
        lm._HOST_ADDRS["www.other.org"] = ("1.2.3.4",)
        try:
            assert await lad._addrset_start("https://www.c.com/", "c.com") == 0  # nothing learned
            lad._addrset_learn("https://www.a.com/", self._res(1, ["tier0:http_403"]))
            assert await lad._addrset_start("https://www.c.com/", "c.com") == 0  # one is not a pattern
            lad._addrset_learn("https://www.b.com/", self._res(1, ["tier0:http_403"]))
            assert await lad._addrset_start("https://www.c.com/", "c.com") == 1
            assert await lad._addrset_start("https://www.other.org/", "other.org") == 0
            # a host on the set that tier 0 served makes the set mixed: off
            lad._addrset_learn("https://www.d.com/", self._res(0, []))
            assert await lad._addrset_start("https://www.c.com/", "c.com") == 0
        finally:
            for h in ("www.a.com", "www.b.com", "www.c.com", "www.d.com", "www.other.org"):
                lm._HOST_ADDRS.pop(h, None)
            await lad.close()

    async def test_a_thin_page_rescue_is_not_a_refusal(self, tmp_path):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, addrset_tier_memo=True))
        for h in ("www.e.com", "www.f.com", "www.g.com"):
            lm._HOST_ADDRS[h] = ("9.9.9.9",)
        try:
            lad._addrset_learn("https://www.e.com/", self._res(1, ["tier0:too_thin:40"]))
            lad._addrset_learn("https://www.f.com/", self._res(1, ["tier0:too_thin:40"]))
            assert await lad._addrset_start("https://www.g.com/", "g.com") == 0
        finally:
            for h in ("www.e.com", "www.f.com", "www.g.com"):
                lm._HOST_ADDRS.pop(h, None)
            await lad.close()


class TestUpgradeDnsFinal:
    async def _read(self, tmp_path, monkeypatch, on, exc):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, https_upgrade=True, upgrade_dns_final=on,
                              cache_enabled=False))
        asked = []

        async def page_fetch(url, md):
            asked.append(url)
            raise exc

        monkeypatch.setattr(eng, "_page_fetch", page_fetch)
        with pytest.raises(Exception):
            await eng.read("http://dead.example/x")
        await eng.close()
        return asked

    async def test_a_name_that_does_not_resolve_is_asked_once(self, tmp_path, monkeypatch):
        import socket

        gai = socket.gaierror(11002, "getaddrinfo failed")
        assert await self._read(tmp_path / "on", monkeypatch, True, gai) == ["https://dead.example/x"]
        assert len(await self._read(tmp_path / "off", monkeypatch, False, gai)) == 2

    async def test_a_refused_https_dial_still_falls_back_to_http(self, tmp_path, monkeypatch):
        from searchio.errors import TransientError

        asked = await self._read(tmp_path, monkeypatch, True, TransientError("connect refused"))
        assert asked == ["https://dead.example/x", "http://dead.example/x"]


class TestLeanStackExchange:
    def test_question_urls_map_to_the_api(self):
        from searchio import leanread as lr

        (q, a), _ = lr._stackexchange("https://stackoverflow.com/questions/123/some-title?noredirect=1")
        assert q == "https://api.stackexchange.com/2.3/questions/123?site=stackoverflow&filter=withbody"
        assert a.startswith("https://api.stackexchange.com/2.3/questions/123/answers?site=stackoverflow")
        (q, _), _ = lr._stackexchange("https://unix.stackexchange.com/q/42")
        assert "site=unix&" in q
        (q, _), _ = lr._stackexchange("https://superuser.com/questions/7/")
        assert "site=superuser&" in q
        assert lr._stackexchange("https://stackoverflow.com/users/1/x") is None
        assert lr._stackexchange("https://stackoverflow.com/questions/tagged/python") is None

    def test_build_renders_question_then_answers(self):
        import json

        from searchio import leanread as lr

        _, build = lr._stackexchange("https://stackoverflow.com/questions/9")
        q = json.dumps({"items": [{"title": "A &amp; B", "score": 3, "tags": ["py"],
                                   "body": "<p>How do I <code>x</code>?</p>"}]})
        a = json.dumps({"items": [{"is_accepted": True, "score": 5, "body": "<p>Use y.</p>"}]})
        title, text = build(q, a)
        assert title == "A & B" and "How do I" in text and "(accepted)" in text and "Use y." in text
        assert build(json.dumps({"items": []}), a) is None  # no such question: read the page


class TestLeanNcbi:
    def test_pubmed_maps_to_efetch_and_checks_the_pmid(self):
        from searchio import leanread as lr

        url, build = lr._pubmed("https://pubmed.ncbi.nlm.nih.gov/30817261/?from=x")
        assert url.endswith("id=30817261&rettype=abstract&retmode=text")
        body = "1. Circ Res. 2019.\n\nThe Mediterranean Diet.\n\nAbstract text.\n\nPMID: 30817261"
        assert build(body) == ("The Mediterranean Diet.", body)
        assert build(body.replace("30817261", "1")) is None
        assert build("<!doctype html><html>captcha</html>") is None
        assert lr._pubmed("https://pubmed.ncbi.nlm.nih.gov/?term=x") is None

    def test_pmc_renders_jats_and_refuses_entities(self):
        from searchio import leanread as lr

        url, build = lr._pmc("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/")
        assert url.endswith("db=pmc&id=123")
        assert lr._pmc("https://pmc.ncbi.nlm.nih.gov/articles/PMC123")[0] == url
        xml = ("<pmc-articleset><article><front><article-meta><title-group><article-title>T "
               "<italic>x</italic></article-title></title-group><abstract><p>Abs.</p></abstract>"
               "</article-meta></front><body><sec><title>Intro</title><p>Body p.</p></sec></body>"
               "</article></pmc-articleset>")
        title, text = build(xml)
        assert title == "T x" and "## Abstract" in text and "## Intro" in text and "Body p." in text
        assert build('<!DOCTYPE a [<!ENTITY e "x">]><a/>') is None
        assert build("not xml") is None


class TestLeanPublisherDoi:
    def test_walled_publisher_urls_map_to_crossref_then_openalex(self):
        from searchio import leanread as lr

        cands = lr._publisher_doi("https://www.tandfonline.com/doi/full/10.1080/1040.2017.139?scroll=top")
        (both, _), (alone, _) = cands
        assert both[0] == alone == "https://api.crossref.org/works/10.1080/1040.2017.139"
        assert both[1].startswith("https://api.openalex.org/works/doi:10.1080/1040.2017.139?select=")
        assert lr._publisher_doi("https://pubs.acs.org/doi/abs/10.1021/x.9b00975/")[1][0].endswith("10.1021/x.9b00975")
        assert lr._publisher_doi("https://www.nature.com/articles/s41586-019-1711-4") is None
        assert lr._publisher_doi("https://onlinelibrary.wiley.com/toc/1234/1") is None

    def test_build_merges_crossref_and_an_openalex_abstract(self):
        import json

        from searchio import leanread as lr

        build = lr._publisher_doi("https://www.pnas.org/doi/10.1073/p.1")[0][1]
        cr = json.dumps({"message": {"title": ["A <i>B</i>"], "container-title": ["PNAS"],
                                     "issued": {"date-parts": [[2020]]},
                                     "author": [{"given": "Ada", "family": "Lovelace"}]}})
        oa = json.dumps({"abstract_inverted_index": {"world": [1], "hello": [0]},
                         "open_access": {"oa_url": "https://x.org/p.pdf"}})
        title, text = build(cr, oa)
        assert title == "A B" and "PNAS, 2020" in text and "Ada Lovelace" in text
        assert "hello world" in text and "Open-access copy: https://x.org/p.pdf" in text
        title, text = build(cr)  # the Crossref-only candidate
        assert "## Abstract" not in text and "DOI: 10.1073/p.1" in text
        assert build("{}") is None


class TestDoiFallback:
    async def _read(self, tmp_path, monkeypatch, on, page, url="https://dl.acm.org/doi/10.1145/3686994"):
        from searchio.engine import Engine
        from searchio.models import Doc

        eng = Engine(Settings(state_dir=tmp_path, doi_fallback=on, cache_enabled=False))
        asked = []

        async def read(u):
            if isinstance(page, Exception):
                raise page
            return Doc(url=u, title="t", text=page, fetched_via="http")

        async def record(u, doi):
            asked.append(doi)
            return Doc(url=u, title="rec", text="# rec", fetched_via="http+doi")

        monkeypatch.setattr(eng, "_read", read)
        monkeypatch.setattr(eng, "_doi_record", record)
        try:
            return await eng.read(url), asked
        finally:
            await eng.close()

    async def test_a_walled_doi_page_answers_with_its_record(self, tmp_path, monkeypatch):
        from searchio.errors import Blocked

        doc, asked = await self._read(tmp_path / "a", monkeypatch, True, Blocked("http_403"))
        assert doc.fetched_via == "http+doi" and asked == ["10.1145/3686994"]
        doc, _ = await self._read(tmp_path / "b", monkeypatch, True, "x" * 31)
        assert doc.fetched_via == "http+doi"  # a near-empty page

    async def test_a_good_page_or_no_doi_never_asks(self, tmp_path, monkeypatch):
        from searchio.errors import Blocked

        doc, asked = await self._read(tmp_path / "a", monkeypatch, True, "word " * 100)
        assert doc.fetched_via == "http" and asked == []
        with pytest.raises(Blocked):
            await self._read(tmp_path / "b", monkeypatch, True, Blocked("x"), url="https://a.org/p")
        with pytest.raises(Blocked):
            await self._read(tmp_path / "c", monkeypatch, False, Blocked("x"))

    async def test_an_ssrn_abstract_page_maps_to_its_ssrn_doi(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.errors import Blocked
        from searchio.models import Doc

        eng = Engine(Settings(state_dir=tmp_path, doi_fallback=True, doi_ssrn=True, cache_enabled=False))
        asked = []

        async def read(u):
            raise Blocked("http_403")

        async def record(u, doi):
            asked.append(doi)
            return Doc(url=u, title="r", text="# r", fetched_via="http+doi")

        monkeypatch.setattr(eng, "_read", read)
        monkeypatch.setattr(eng, "_doi_record", record)
        try:
            await eng.read("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4915543")
        finally:
            await eng.close()
        assert asked == ["10.2139/ssrn.4915543"]

    def test_doi_in_url(self):
        from searchio.leanread import doi_in_url

        assert doi_in_url("https://www.emerald.com/insight/content/doi/10.1108/jeas-04/full/html") == "10.1108/jeas-04"
        assert doi_in_url("https://x.org/doi/abs/10.1000/xyz.pdf") == "10.1000/xyz"
        assert doi_in_url("https://example.com/a/b") is None


class TestLeanGitlabRxiv:
    def test_gitlab_project_pages_only(self):
        from searchio import leanread as lr

        assert lr._gitlab("https://gitlab.com/a/b/")[0] == "https://gitlab.com/a/b/-/raw/HEAD/README.md"
        assert lr._gitlab("https://gitlab.com/g/sub/p")[0].startswith("https://gitlab.com/g/sub/p/-/raw/")
        for u in ("https://gitlab.com/a/b/-/issues/1", "https://gitlab.com/explore/projects",
                  "https://gitlab.com/onlyuser", "https://gitlab.com/a/b.git"):
            assert lr._gitlab(u) is None, u
        build = lr._gitlab("https://gitlab.com/a/b")[1]
        assert build("<!DOCTYPE html><html>sign in</html>" + "x" * 100) is None
        assert build("# B" + chr(10) + "x" * 100)[0] == "a/b"

    def test_rxiv_picks_the_asked_version(self):
        import json

        from searchio import leanread as lr

        url, build = lr._rxiv("https://www.medrxiv.org/content/10.1101/2020.04.14.20062463v1.full.pdf")
        assert url == "https://api.biorxiv.org/details/medrxiv/10.1101/2020.04.14.20062463"
        body = json.dumps({"collection": [{"version": "1", "title": "Old", "abstract": "a1"},
                                          {"version": "2", "title": "New", "abstract": "a2"}]})
        assert build(body)[0] == "Old"
        assert lr._rxiv("https://www.biorxiv.org/content/10.1101/2024.01.01.5")[1](body)[0] == "New"
        assert build(json.dumps({"collection": []})) is None
        assert lr._rxiv("https://www.biorxiv.org/collection/neuroscience") is None


class TestLeanSemanticScholar:
    def test_paper_pages_map_to_the_graph_api(self):
        import json

        from searchio import leanread as lr

        sha = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
        for u in (f"https://www.semanticscholar.org/paper/Attention-Vaswani/{sha}",
                  f"https://semanticscholar.org/paper/{sha}?utm=x"):
            assert lr._semantic_scholar(u)[0].startswith(f"https://api.semanticscholar.org/graph/v1/paper/{sha}?")
        assert lr._semantic_scholar("https://www.semanticscholar.org/author/x/123") is None
        build = lr._semantic_scholar(f"https://www.semanticscholar.org/paper/{sha}")[1]
        title, text = build(json.dumps({"title": "Attention", "year": 2017, "venue": "NeurIPS",
                                        "authors": [{"name": "A. V."}], "abstract": "We propose.",
                                        "tldr": {"text": "Short."}, "externalIds": {"ArXiv": "1706.03762"}}))
        assert title == "Attention" and "NeurIPS, 2017" in text and "TLDR: Short." in text
        assert "ArXiv: 1706.03762" in text and "We propose." in text
        assert build('{"message": "Too Many Requests"}') is None


class TestPdfReadBudget:
    def test_extraction_stops_at_the_budget(self):
        from searchio.net import pdf as pdf_mod

        data = pdf_mod.build_pdf([f"page {i} " + "word " * 40 for i in range(10)])
        full, used, total = pdf_mod.extract_text(data, 50)
        assert used == total == 10
        cut, used, total = pdf_mod.extract_text(data, 50, char_budget=300)
        assert used < total and full.startswith(cut) and len(cut) >= 300


class TestPdfReuseBytes:
    async def test_extraction_uses_the_bytes_in_hand(self, tmp_path, monkeypatch):
        from searchio.net import ladder as lm
        from searchio.net import pdf as pdf_mod

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off"))
        data = pdf_mod.build_pdf(["hello pdf world"])
        calls = []

        async def fetch_bytes(url):
            calls.append(url)
            return data

        monkeypatch.setattr(lad, "_fetch_bytes", fetch_bytes)
        lad._pdf_raw["https://x.org/a.pdf"] = data
        text, _ = await lad._pdf_text("https://x.org/a.pdf")
        assert "hello" in text and calls == []
        text, _ = await lad._pdf_text("https://x.org/a.pdf")  # used once, then gone
        assert "hello" in text and calls == ["https://x.org/a.pdf"]
        await lad.close()


class TestDecisive404:
    async def _tiers(self, tmp_path, monkeypatch, on, status=404, server="nginx", cdn=False,
                     nf="<html><head><title>Not Found</title></head><body><h1>404 Not Found</h1></body></html>"):
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 decisive_404=on, decisive_404_cdn=cdn))
        tiers = []

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            return status, {"content-type": "text/html", "server": server}, nf, "text/html", url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        with pytest.raises(Exception):
            await lad.fetch("https://site.org/missing", use_cache=False, max_tier=1)
        await lad.close()
        return tiers

    async def test_a_plain_404_ends_the_climb(self, tmp_path, monkeypatch):
        assert await self._tiers(tmp_path / "on", monkeypatch, True) == [0]
        assert await self._tiers(tmp_path / "gone", monkeypatch, True, status=410) == [0]
        assert await self._tiers(tmp_path / "off", monkeypatch, False) == [0, 1]

    async def test_a_cdn_named_404_ends_the_climb_unless_it_is_a_challenge(self, tmp_path, monkeypatch):
        assert await self._tiers(tmp_path / "cf0", monkeypatch, True, server="cloudflare") == [0, 1]
        assert await self._tiers(tmp_path / "cf1", monkeypatch, True, server="cloudflare", cdn=True) == [0]
        chal = ("<html><head><title>Just a moment...</title></head><body>"
                "<div id='challenge-platform'>cf-chl</div></body></html>")
        assert await self._tiers(tmp_path / "cf2", monkeypatch, True, server="cloudflare", cdn=True,
                                 nf=chal) == [0, 1]


class TestRateLimit503:
    async def _run(self, tmp_path, monkeypatch, on, headers):
        from searchio.errors import RateLimited
        from searchio.net import ladder as lm

        lad = lm.Ladder(Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                                 ratelimit_aware=True, ratelimit_503=on))
        tiers = []
        err = "<html><head><title>Wikimedia Error</title></head><body>Too many requests</body></html>"

        async def try_tier(tier, url, referer="", rendered=False, **kw):
            tiers.append(tier)
            return 503, {"content-type": "text/html", **headers}, err, "text/html", url

        monkeypatch.setattr(lad, "_try_tier", try_tier)
        try:
            await lad.fetch("https://en.wikipedia.org/w/api.php?q", use_cache=False, max_tier=1)
            got = None
        except RateLimited as exc:
            got = exc.retry_after
        except Exception:  # noqa: BLE001 -- the climb's own exhaustion
            got = "climbed"
        await lad.close()
        return got, tiers

    async def test_503_with_retry_after_is_a_rate_limit(self, tmp_path, monkeypatch):
        assert await self._run(tmp_path / "on", monkeypatch, True, {"retry-after": "41"}) == (41.0, [0])
        got, tiers = await self._run(tmp_path / "off", monkeypatch, False, {"retry-after": "41"})
        assert got == "climbed" and tiers == [0, 1]
        got, tiers = await self._run(tmp_path / "bare", monkeypatch, True, {})
        assert got == "climbed", "a bare 503 is not a stated limit"


class TestLeanMissMemo:
    async def _eng(self, tmp_path, monkeypatch, on, exc):
        from searchio.engine import Engine

        eng = Engine(Settings(state_dir=tmp_path, lean_miss_memo=on, cache_enabled=False))
        asked = []

        async def fetch(u, **kw):
            asked.append(u)
            raise exc

        monkeypatch.setattr(eng.ladder, "fetch", fetch)
        return eng, asked

    async def test_a_404_lean_url_is_not_asked_again(self, tmp_path, monkeypatch):
        from searchio.errors import TransientError

        gone = TransientError("x: no usable content (tried ['tier0:http_404'])")
        for on, want in ((True, 1), (False, 2)):
            eng, asked = await self._eng(tmp_path / str(on), monkeypatch, on, gone)
            try:
                for _ in range(2):
                    assert await eng._lean_read("https://github.com/django/django") is None
            finally:
                await eng.close()
            assert len(asked) == want

    async def test_a_transient_failure_is_not_remembered(self, tmp_path, monkeypatch):
        from searchio.errors import TransientError

        eng, asked = await self._eng(tmp_path, monkeypatch, True,
                                     TransientError("x: (tried ['tier0:ConnectTimeout'])"))
        try:
            for _ in range(2):
                assert await eng._lean_read("https://github.com/django/django") is None
        finally:
            await eng.close()
        assert len(asked) == 2


class TestYoutubeWatchDetails:
    PAGE = ('<html><script>var ytInitialPlayerResponse = {"videoDetails": {"title": "T [x]", '
            '"author": "Chan", "lengthSeconds": "3725", "viewCount": "1234", '
            '"shortDescription": "Line one."}, "microformat": {"playerMicroformatRenderer": '
            '{"publishDate": "2024-05-01T10:00:00-07:00", "category": "Education"}}};'
            'var meta = 1;</script></html>')

    def test_the_player_json_becomes_the_text(self):
        from searchio.leanread import youtube_watch_details

        title, text = youtube_watch_details("https://www.youtube.com/watch?v=abcdefghijk", self.PAGE)
        assert title == "T [x]"
        assert "Channel: Chan" in text and "Length: 1:02:05" in text and "Views: 1,234" in text
        assert "Published: 2024-05-01" in text and text.endswith("Line one.")

    def test_other_pages_are_left_alone(self):
        from searchio.leanread import youtube_watch_details

        assert youtube_watch_details("https://www.youtube.com/@chan", self.PAGE) is None
        assert youtube_watch_details("https://www.youtube.com/watch?v=x", "<html>no json</html>") is None
        assert youtube_watch_details("https://example.com/watch?v=x", self.PAGE) is None


class TestDoiDirectMemo:
    async def test_a_walled_doi_page_is_not_climbed_again(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.errors import Blocked
        from searchio.models import Doc

        for on, want in ((True, 1), (False, 2)):
            eng = Engine(Settings(state_dir=tmp_path / str(on), doi_fallback=True,
                                  doi_direct_memo=on, thin_read_memo_s=0,
                                  blocked_read_memo_s=0, cache_enabled=False))
            pages = []

            async def read(u):
                pages.append(u)
                raise Blocked("http_403")

            async def record(u, doi):
                return Doc(url=u, title="r", text="# r", fetched_via="http+doi")

            monkeypatch.setattr(eng, "_read", read)
            monkeypatch.setattr(eng, "_doi_record", record)
            try:
                for _ in range(2):
                    doc = await eng.read("https://dl.acm.org/doi/10.1145/3686994")
                    assert doc.fetched_via == "http+doi"
            finally:
                await eng.close()
            assert len(pages) == want


class TestBlockedReadMemo:
    async def test_a_walled_read_fails_at_once_within_the_ttl(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.errors import Blocked, TransientError

        for ttl, exc, want in ((300.0, Blocked("w"), 1), (0.0, Blocked("w"), 2),
                               (300.0, TransientError("t"), 2)):
            eng = Engine(Settings(state_dir=tmp_path / f"{ttl}{type(exc).__name__}",
                                  blocked_read_memo_s=ttl, cache_enabled=False))
            calls = []

            async def inner(u, exc=exc):
                calls.append(u)
                raise exc

            monkeypatch.setattr(eng, "_read_or_doi", inner)
            try:
                for _ in range(2):
                    with pytest.raises(type(exc)):
                        await eng.read("https://walled.example/p")
            finally:
                await eng.close()
            assert len(calls) == want


class TestThinReadMemo:
    async def test_a_thin_read_is_copied_within_the_ttl(self, tmp_path, monkeypatch):
        from searchio.engine import Engine
        from searchio.models import Doc

        for ttl, text, want in ((300.0, "shell", 1), (0.0, "shell", 2), (300.0, "word " * 100, 2)):
            eng = Engine(Settings(state_dir=tmp_path / f"{ttl}{len(text)}", thin_read_memo_s=ttl,
                                  blocked_read_memo_s=0, cache_enabled=False))
            calls = []

            async def inner(u, text=text):
                calls.append(u)
                return Doc(url=u, title="t", text=text, fetched_via="http")

            monkeypatch.setattr(eng, "_read_or_doi", inner)
            try:
                docs = [await eng.read("https://shell.example/") for _ in range(2)]
            finally:
                await eng.close()
            assert len(calls) == want and docs[0].text == docs[1].text
            assert docs[0] is not docs[1]


class TestExtractCpuKnobs:
    PAGE = ("<html><head><style>p{color:red}</style><script>var big = '" + "x" * 5000 + "';</script>"
            "</head><body><article><h1>Heading here</h1>"
            + "".join(f"<p>Paragraph {i} has enough words in it to count as real article text.</p>"
                      for i in range(12))
            + "</article><div class='comments'><p>Reader comment that is not the article.</p></div>"
            "</body></html>")

    def test_same_article_with_either_knob(self, monkeypatch):
        from searchio import config
        from searchio.extract import to_markdown

        outs = []
        for kw in ({}, {"extract_no_comments": True}, {"extract_strip_scripts": True}):
            monkeypatch.setattr(config, "_settings", Settings(**kw))
            outs.append(to_markdown(self.PAGE, "https://example.com/a"))
        assert all("Paragraph 11" in o and "var big" not in o for o in outs)
        assert outs[2] == outs[0]


class TestStructuredListings:
    CAR = ('{"@type": ["Product", "Car"], "name": "Used 2022 Ford Mustang", "vehicleIdentificationNumber": '
           '"1FA6P8TH7N5135451", "vehicleTransmission": "6-Speed Manual", "mileageFromOdometer": '
           '{"value": "19,612", "unitCode": "SMI"}, "offers": {"@type": "Offer", "price": "21995.00", '
           '"priceCurrency": "USD", "url": "/cars-for-sale/vehicle/1", "seller": {"name": "World Auto", '
           '"address": {"addressLocality": "Duluth", "addressRegion": "GA"}}}}')
    # Nested two levels down, as Autotrader ships it.
    PAGE = ('<html><head><script type="application/ld+json">{"@type": "CollectionPage", "about": {"offers": '
            '{"@type": "Offer", "itemOffered": [' + CAR + ']}}}</script></head><body><article><h1>Used Mustangs</h1>'
            + "".join(f"<p>Paragraph {i} has enough words in it to count as real article text.</p>" for i in range(8))
            + "</article></body></html>")

    def _text(self, monkeypatch, page, **kw):
        from searchio import config
        from searchio.extract import doc_from_fetch

        monkeypatch.setattr(config, "_settings", Settings(**kw))
        return doc_from_fetch("https://www.autotrader.com/cars-for-sale/x", page).text

    def test_off_and_on_renders_nested_cards(self, monkeypatch):
        assert "1FA6P8TH7N5135451" not in self._text(monkeypatch, self.PAGE, structured_listings=False)
        text = self._text(monkeypatch, self.PAGE, structured_listings=True)
        row = text.split("## Listings", 1)[1]
        for part in ("Used 2022 Ford Mustang", "$21,995", "19,612 mi", "6-Speed Manual",
                     "VIN 1FA6P8TH7N5135451", "World Auto (Duluth, GA)",
                     "https://www.autotrader.com/cars-for-sale/vehicle/1"):
            assert part in row

    def test_engine_and_trim_ride_on_the_row(self, monkeypatch):
        import json

        # JSON-LD: an EngineSpecification object; mileage as a bare number.
        ld = self.CAR.replace('"vehicleTransmission"', '"vehicleEngine": {"@type": "EngineSpecification", '
                              '"name": "2.3L Inline-4 Gas Turbocharged"}, "vehicleTransmission"') \
            .replace('"19,612"', "19612")
        row = self._text(monkeypatch, self.PAGE.replace(self.CAR, ld), structured_listings=True)
        assert "19,612 mi | 6-Speed Manual | 2.3L Inline-4 Gas Turbocharged | VIN" in row
        # Generic records: an engine label, a size + cylinder split, a trim
        # beside a bare name, and a fuel word that is not an engine.
        state = "<script>window.__S = " + json.dumps([
            {"title": "2024 Ford Mustang", "trimName": "EcoBoost Fastback RWD", "price": 30097,
             "vin": "1FA6P8TH1R5111099", "engineName": "315 hp 2.3L I4", "transmission": "Automatic"},
            {"title": "2021 Ford Mustang GT Premium", "trim": "GT Premium", "price": 39998,
             "vin": "1FA6P8CF1M5118237", "engineType": "Gas", "engineSize": "5.0L", "cylinders": 8},
        ]) + ";</script>"
        on = self._generic(monkeypatch, state)
        assert "2024 Ford Mustang EcoBoost Fastback RWD | $30,097 | Automatic | 315 hp 2.3L I4 | VIN" in on
        assert "2021 Ford Mustang GT Premium | $39,998 | 5.0L 8-cyl | VIN 1FA6P8CF1M5118237" in on

    def test_asking_price_not_estimate_and_far_stores_say_so(self, monkeypatch):
        import json

        # A market estimate beside the record and the asking price in a price
        # child; trim, dealer and place one wrapper object down.
        state = "<script>window.__S = " + json.dumps([
            {"listingTitle": "2022 Ford Mustang", "vin": "1FATP8UD5N5126701", "imvPrice": 26717,
             "distance": 1863.067, "ontologyDto": {"trimName": "EcoBoost Premium Convertible RWD"},
             "priceDto": {"current": 25997, "expected": 26717, "totalPrice": 27996, "localizedTotalPrice": "$27,996"},
             "sellerDto": {"serviceProviderName": "CarMax Spokane", "city": "Spokane", "region": "WA"}},
        ]) + ";</script>"
        on = self._generic(monkeypatch, state)
        assert ("2022 Ford Mustang EcoBoost Premium Convertible RWD | $25,997 | VIN 1FATP8UD5N5126701 | "
                "CarMax Spokane (Spokane, WA) | distance 1,863") in on

    def test_app_state_twin_adds_distance_to_the_ld_row(self, monkeypatch):
        import json

        # The JSON-LD card has no distance; the page's app state holds the same
        # car (unnamed, two levels down) with one. One row, carrying both.
        twin = "<script>window.__S = " + json.dumps({"vehicle": {
            "vin": "1FA6P8TH7N5135451", "transmission": "Manual", "year": 2022, "make": {"name": "Ford"},
            "model": {"name": "Mustang"},
            "details": {"vin": "1FA6P8TH7N5135451", "dealerDistanceMiles": 96.7, "searchRadiusDistance": 100}}}) \
            + ";</script>"
        on = self._generic(monkeypatch, twin)
        rows = [r for r in on.split("## Listings", 1)[1].splitlines() if "1FA6P8TH7N5135451" in r]
        assert len(rows) == 1 and "| distance 97 |" in rows[0] and "World Auto (Duluth, GA)" in rows[0]

    def test_page_without_listing_data_is_unchanged(self, monkeypatch):
        page = self.PAGE.replace(self.CAR, '{"@type": "Organization", "name": "x"}')
        assert self._text(monkeypatch, page, structured_listings=False) == self._text(monkeypatch, page, structured_listings=True,
                                                            listing_generic=True)

    def test_every_blob_is_walked(self, monkeypatch):
        # Dedupe by id() once dropped nodes whose ids a freed earlier blob had used.
        blobs = "".join(f'<script type="application/ld+json">{{"@type": "WebPage", "n": {i}}}</script>'
                        for i in range(30))
        page = self.PAGE.replace("<head>", "<head>" + blobs)
        assert "VIN 1FA6P8TH7N5135451" in self._text(monkeypatch, page, structured_listings=True)

    def _generic(self, monkeypatch, body: str, **kw) -> str:
        page = self.PAGE.replace("<body>", f"<body>{body}")
        return self._text(monkeypatch, page, structured_listings=True, listing_generic=True, **kw)

    def test_generic_records_in_any_attribute_and_script_state(self, monkeypatch):
        import html as h
        import json

        # Attribute and key names are arbitrary: records are recognised by shape.
        attr = h.escape(json.dumps({"stockType": "Used", "year": "2024", "make": "Ford", "model": "Mustang",
                                    "trim": "GT", "price": "41164", "mileage": "24272", "vin": "1FA6P8CF2R5412852",
                                    "seller": {"dealerName": "ALM", "zip": "30060"}}))
        state = ('<script>const cars = [{"stockNumber": 1, "vin": "1FA6P8CF1M5118237", "car_year": 2021, '
                 '"makeName": "Ford", "modelName": "Mustang", "trim": "GT Premium", "basePrice": 39998.0, '
                 '"odometer": "7,827 mi", "transmission": "Automatic", "storeName": "Naperville"}];'
                 'window.__S = {"items": [{"productTitle": "Acme 2TB NVMe SSD", "salePrice": "129.99", '
                 '"sku": "A-2TB", "detailUrl": "/p/a-2tb"}]};</script>')
        on = self._generic(monkeypatch, f'<div data-zz-card="{attr}"></div>{state}')
        assert "Used 2024 Ford Mustang GT | $41,164 | 24,272 mi" in on
        assert "ALM (30060)" in on
        assert "2021 Ford Mustang GT Premium | $39,998 | 7,827 mi | Automatic | VIN 1FA6P8CF1M5118237" in on
        assert "Acme 2TB NVMe SSD | $130" in on and "/p/a-2tb" in on
        off = self._text(monkeypatch, self.PAGE.replace("<body>", f'<body><div data-zz-card="{attr}"></div>{state}'),
                         structured_listings=True, listing_generic=False)
        assert "1FA6P8CF2R5412852" not in off and "Acme" not in off

    @pytest.mark.parametrize("on", [False, True])
    def test_card_data_takes_the_cards_own_link(self, monkeypatch, on):
        import html as h
        import json

        def card(vin, n):
            data = h.escape(json.dumps({"year": "2024", "make": "Ford", "model": "Mustang", "price": "41164",
                                        "vin": vin}))
            return (f'<div data-zz="{data}"><img src="/img/{n}.jpg">'
                    f'<a data-card-href="/vehicledetail/{n}/">x</a></div>')

        body = card("1FA6P8CF2R5412852", "a1") + card("1FA6P8CF1M5118237", "b2")
        rows = [r for r in self._generic(monkeypatch, body, listing_card_href=on).splitlines()
                if "1FA6P8CF" in r]
        assert len(rows) == 2, rows
        assert ("1FA6P8CF2R5412852 | https://www.autotrader.com/vehicledetail/a1/" in rows[0]) is on
        assert ("1FA6P8CF1M5118237 | https://www.autotrader.com/vehicledetail/b2/" in rows[1]) is on

    def test_card_prints_its_distance(self, monkeypatch):
        import html as h
        import json

        def card(vin, where, extra=""):
            data = h.escape(json.dumps({"year": "2024", "make": "Ford", "model": "Mustang", "price": "41164",
                                        "vin": vin}))
            badge = h.escape(json.dumps({"title": "Great Deal"}))
            return (f'<div data-zz="{data}"><span>24,272 mi.</span><i data-badge="{badge}"></i>{extra}'
                    f'<span>{where}</span></div>')

        # The second card prints no distance: the third card's "(12 mi)" is not its.
        body = (card("1FA6P8CF2R5412852", "Duluth, GA (97 mi)") + card("1FA6P8CF1M5118237", "Marietta, GA")
                + card("1FA6P8CF0R5432551", "Chattanooga, TN (12 mi)"))
        rows = {r.split(" | VIN ")[1][:17]: r for r in
                self._generic(monkeypatch, body, listing_card_href=True).splitlines() if "1FA6P8CF" in r}
        assert "| distance 97" in rows["1FA6P8CF2R5412852"]
        assert "distance" not in rows["1FA6P8CF1M5118237"]
        assert "| distance 12" in rows["1FA6P8CF0R5432551"]

    def test_a_card_never_reaches_past_the_next_cards_data(self, monkeypatch):
        import html as h
        import json

        data = [h.escape(json.dumps({"year": "2024", "make": "Ford", "model": "Mustang", "price": p, "vin": v}))
                for p, v in (("1", "1FA6P8CF2R5412852"), ("2", "1FA6P8CF1M5118237"))]
        body = f'<div data-a="{data[0]}" data-b="{data[1]}"><a href="/vehicledetail/b2/">c</a></div>'
        rows = {r.split(" | VIN ")[1][:17]: r for r in
                self._generic(monkeypatch, body, listing_card_href=True).splitlines() if "1FA6P8CF" in r}
        assert "/vehicledetail/b2/" in rows["1FA6P8CF1M5118237"]
        assert "/vehicledetail/" not in rows["1FA6P8CF2R5412852"]

    def test_a_link_two_items_share_is_given_to_neither(self):
        from searchio import extract
        from searchio.config import Settings, set_settings

        # One data attribute per card, but both cards point at one page link
        # (a "compare" control rendered after each).
        import html as h
        import json

        set_settings(Settings(listing_card_href=True))
        try:
            cards = "".join(
                f'<div data-a="{h.escape(json.dumps({"year": "2024", "make": "Ford", "model": "Mustang", "price": p, "vin": v}))}">'
                f'<a href="/compare">c</a></div>'
                for p, v in (("1", "1FA6P8CF2R5412852"), ("2", "1FA6P8CF1M5118237")))
            rows = extract._generic_rows(cards, "https://x.com/s")
        finally:
            set_settings(Settings())
        assert len(rows) == 2 and not any("/compare" in r for r, _k in rows), rows

    @staticmethod
    def _id_rows(body, on):
        from searchio import extract
        from searchio.config import Settings, set_settings

        set_settings(Settings(listing_id_href=on))
        try:
            return {k: r for r, k in extract._generic_rows(body, "https://x.com/search?q=mustang")}
        finally:
            set_settings(Settings())

    @staticmethod
    def _script(*items):
        import json

        return "<script>window.__S = " + json.dumps({"tiles": [
            {"id": i, "stockNumber": f"S{i}", "year": "2024", "make": "Ford", "model": "Mustang",
             "price": p, "vin": v} for i, p, v in items]}) + ";</script>"

    @pytest.mark.parametrize("on", [False, True])
    def test_script_data_takes_the_link_its_id_names(self, on):
        body = self._script((458801235, "41164", "1FA6P8CF2R5412852"), (458103421, "39998", "1FA6P8CF1M5118237"))
        body += ('<a href="/details/458103421?searchId=abc&amp;i=1">b</a>'
                 '<a href="/details/458801235?searchId=abc&amp;i=0">a</a><a href="/help/1234567">h</a>')
        rows = self._id_rows(body, on)
        assert rows["1FA6P8CF2R5412852"].endswith(" | https://x.com/details/458801235?searchId=abc&i=0") is on
        assert rows["1FA6P8CF1M5118237"].endswith(" | https://x.com/details/458103421?searchId=abc&i=1") is on

    def test_an_id_only_inside_a_longer_number_or_off_site_links_nothing(self):
        body = self._script((458801235, "41164", "1FA6P8CF2R5412852"))
        body += ('<a href="/details/4588012350">a</a><a href="https://other.com/details/458801235">b</a>'
                 '<img src="/img/458801235.jpg"><span title="/details/458801235">not a link attr</span>')
        assert "http" not in self._id_rows(body, True)["1FA6P8CF2R5412852"]

    def test_an_id_in_two_pages_paths_names_neither(self):
        body = self._script((458801235, "41164", "1FA6P8CF2R5412852"))
        body += '<a href="/details/458801235">a</a><a href="/compare/458801235">b</a>'
        assert "http" not in self._id_rows(body, True)["1FA6P8CF2R5412852"]

    def test_a_slug_part_and_a_data_url_attribute_count(self):
        body = self._script((458801235, "41164", "1FA6P8CF2R5412852"))
        body += '<div data-detail-url="/used/2024-ford-mustang-458801235/"></div>'
        assert self._id_rows(body, True)["1FA6P8CF2R5412852"].endswith(
            " | https://x.com/used/2024-ford-mustang-458801235/")

    @staticmethod
    def _hist_rows(body, on=True):
        from searchio import extract
        from searchio.config import Settings, set_settings

        set_settings(Settings(listing_history=on))
        try:
            return {k: r for r, k in extract._generic_rows(body, "https://x.com/s")}
        finally:
            set_settings(Settings())

    @staticmethod
    def _state(obj):
        import json

        return "<script>window.__S = " + json.dumps(obj) + ";</script>"

    @pytest.mark.parametrize("on", [False, True])
    def test_a_sibling_history_object_reaches_its_cars_row(self, on):
        cars = [{"details": {"vin": v, "year": "2024", "make": "Ford", "model": "Mustang", "price": p},
                 "conditionHistory": {"accidentCount": a, "ownerCount": o, "isCleanTitle": True}}
                for v, p, a, o in (("1FA6P8CF2R5412852", "41164", 0, 2), ("1FA6P8CF1M5118237", "39998", 1, 1))]
        rows = self._hist_rows(self._state({"listings": cars}), on)
        assert ("history: 0 accidents, 2 owners, clean title" in rows["1FA6P8CF2R5412852"]) is on
        assert ("history: 1 accident, 1 owner, clean title" in rows["1FA6P8CF1M5118237"]) is on

    def test_history_codes_in_the_enclosing_record_are_read(self):
        rec = {"vin": "1FA6P8TH7N5135451", "year": "2022", "make": "Ford", "model": "Mustang",
               "pricing": {"listPrice": {"value": 21995}},
               "vhrPreview": ["NO_SALVAGE_TITLE", "ACCIDENTS_REPORTED", "NO_ONE_OWNER", "FREE_REPORT"]}
        row = self._hist_rows(self._state({"inventory": {"787322400": rec}}))["1FA6P8TH7N5135451"]
        assert "history: no salvage title, accidents reported, no one owner" in row
        assert "free report" not in row

    def test_a_wrapper_naming_two_vins_gives_its_history_to_neither(self):
        cars = [{"vin": v, "year": "2024", "make": "Ford", "model": "Mustang", "price": p}
                for v, p in (("1FA6P8CF2R5412852", "1"), ("1FA6P8CF1M5118237", "2"))]
        rows = self._hist_rows(self._state({"cars": cars, "summary": {"accidentCount": 3}}))
        assert not any("history:" in r for r in rows.values()), rows

    def test_look_alike_fields_are_not_history(self):
        car = {"vin": "1FA6P8CF2R5412852", "year": "2024", "make": "Ford", "model": "Mustang", "price": "1",
               "dealer": {"ownerName": "World Auto", "ownerServices": ["Body Shop"]},
               "cleanTitle": "Tungsten 2-Person Tent", "accidentFreeCount": 347, "ownerCount": "2"}
        assert "history:" not in self._hist_rows(self._state({"car": car}))["1FA6P8CF2R5412852"]

    def test_history_survives_when_the_prose_already_names_the_car(self, monkeypatch):
        state = self._state({"listings": [
            {"details": {"vin": "1FA6P8CF2R5412852", "year": "2024", "make": "Ford", "model": "Mustang",
                         "price": "41164", "url": "/listing/1FA6P8CF2R5412852/"},
             "conditionHistory": {"accidentCount": 0, "ownerCount": 1, "isCleanTitle": True}}]})
        page = self.PAGE.replace("<body>", "<body><p>2024 Ford Mustang VIN 1FA6P8CF2R5412852 for sale</p>" + state)
        text = self._text(monkeypatch, page, structured_listings=True, listing_generic=True,
                          listing_history=True)
        assert "VIN 1FA6P8CF2R5412852 | history: 0 accidents, 1 owner, clean title" in text

    def test_a_wrapped_vin_and_its_problem_flags(self):
        listing = {"id": 453062205, "price": 46997, "yearMakeModel": "2023 Ford Mustang",
                   "yearMakeModelTrim": "2023 Ford Mustang Mach 1 Fastback RWD",
                   "vin": {"value": "1FA6P8R02P5500073", "invalidVin": False},
                   "vehicleHistory": {"salvage": False, "lemon": False, "frameDamaged": True,
                                      "accidentCount": 1, "ownerCount": 2, "hasVehicleHistoryReport": True}}
        row = self._hist_rows(self._state({"listing": listing}))["1FA6P8R02P5500073"]
        assert row.startswith("2023 Ford Mustang Mach 1 Fastback RWD | $46,997"), row
        assert "history: frame damage, 1 accident, 2 owners, no salvage title/lemon record" in row

    def test_a_vin_under_vehicle_identifier(self):
        car = {"price": 46997, "make": "Ford", "model": "Mustang", "year": 2023,
               "vehicleIdentifier": "1FA6P8R02P5500073"}
        assert "VIN 1FA6P8R02P5500073" in self._hist_rows(self._state({"car": car}))["1FA6P8R02P5500073"]

    def test_history_codes_deep_inside_a_large_record(self):
        rec = {"vin": "1FA6P8TH7N5135451", "year": "2022", "make": "Ford", "model": "Mustang",
               "pricing": {"listPrice": {"value": 21995}},
               "options": [{"label": f"Option {i}", "category": {"code": i}} for i in range(120)],
               "vhrPreview": ["NO_SALVAGE_TITLE", "ACCIDENTS_REPORTED"]}
        row = self._hist_rows(self._state({"inventory": {"787322400": rec}}))["1FA6P8TH7N5135451"]
        assert "history: no salvage title, accidents reported" in row

    @pytest.mark.parametrize("fields, want", [
        ('{"total_listings":323,"result_count":323,"facets":{"totalCount":12}}', 323),
        ('{"totalCount":466,"other":{"totalCount":227}}', None),           # two totals: neither
        ('{"totalCount":150}', None),                                      # one field is not agreement
        ('{"subtotalCount":5,"grandtotalCount":5}', None),                 # longer keys are not totals
        ('window.x={totalCount = 0, ResultCount: "0", facet: {"totalCount":1}}', 0),
    ])
    def test_the_result_count_needs_agreeing_fields(self, fields, want):
        from searchio import extract

        assert extract._result_count(f"<script>{fields}</script>") == want

    @pytest.mark.parametrize("on", [False, True])
    def test_a_search_that_matched_nothing_says_so(self, monkeypatch, on):
        page = ('<html><head><title>Cars</title></head><body><p>This price does not include tax, title,'
                ' and tags.</p><script>var s={"totalCount":0,"resultCount":0};</script></body></html>')
        text = self._text(monkeypatch, page, structured_listings=True, listing_generic=True,
                          listing_result_count=on)
        assert ("No matches: the page's own data reports 0 results." in text) is on

    def test_generic_rejects_priced_non_items(self, monkeypatch):
        junk = ('<script>var a = {"shipping": [{"name": "Same Day", "price": 10}], '
                '"exp": {"name": "challenger1", "variantPrice": "control", "id": "x"}, '
                '"track": {"make": "ford", "price": "40000", "id": "t"}, '
                '"fx": [{"countryName": "Canada", "price": 382, "isActive": false}]};</script>')
        out = self._generic(monkeypatch, junk)
        for bad in ("Same Day", "challenger1", "ford | $40,000", "Canada"):
            assert bad not in out

    def test_generic_reads_items_split_into_children(self, monkeypatch):
        # GraphQL-cache shape: name under identifiers, amount under pricing(...).
        gql = ('<script>window.__S = {"p:1": {"__typename": "Product", "itemId": "1", '
               '"identifiers": {"productLabel": "18V Cordless Drill Kit", "brandName": "Acme", "itemId": "1"}, '
               '"pricing({\\"storeId\\":\\"9\\"})": {"value": 199, "original": 249}, '
               '"media": {"images": [{"url": "https://x/1.jpg"}]}, '
               '"make": {"name": "NotTheName"}, "tracking": {"clickUrl": "https://ads/beacon?x"}}, '
               # schema.org Product outside an ld+json tag: price under offers[].
               '"q": {"@type": "Product", "name": "Wireless Earbuds Pro", "url": "https://s/itm/9", '
               '"offers": [{"price": "79.00", "priceCurrency": "USD"}]}};</script>')
        out = self._generic(monkeypatch, gql)
        assert "Acme 18V Cordless Drill Kit | $199" in out
        assert "Wireless Earbuds Pro | $79 | https://s/itm/9" in out
        assert "NotTheName" not in out and "beacon" not in out

    def test_generic_prefers_the_named_item_inside_a_wrapper(self, monkeypatch):
        # The wrapper only knows the model code; its ItemCell child has the title.
        wrap = ('<script>var w = [{"Model": "AB-1TB-01", "FinalPrice": "99", "Price": "None", '
                '"ItemCell": {"Item": "9X1", "FinalPrice": "99", "Model": "AB-1TB-01", '
                '"Description": {"Title": "Acme 1TB NVMe SSD AB-1TB-01"}, "Image": {"Name": "a.jpg"}}}];</script>')
        rows = [r for r in self._generic(monkeypatch, wrap).splitlines() if "AB-1TB-01" in r]
        assert rows == ["- Acme 1TB NVMe SSD AB-1TB-01 | $99"]

    def test_generic_merges_copies_but_keeps_distinct_vins(self, monkeypatch):
        state = ('<script>var s = [{"name": "2020 Toyota Camry LE", "price": 23998, "vin": "4T1C11AKXLU391151"}, '
                 '{"name": "2020 Toyota Camry LE", "price": 23998, "vin": "4T1BF1FK6FU877630"}, '
                 '{"name": "2020 Toyota Camry LE", "price": 23998, "url": "/c/1"}];</script>')
        rows = [r for r in self._generic(monkeypatch, state).splitlines() if "Camry" in r]
        assert len(rows) == 2 and all("VIN" in r for r in rows)

    def test_rows_already_in_the_prose_are_not_repeated(self, monkeypatch):
        page = self.PAGE.replace("<h1>Used Mustangs</h1>", "<h1>Used Mustangs</h1><p>VIN 1FA6P8TH7N5135451 "
                                 "is our featured car this week, come and see it.</p>")
        assert "## Listings" not in self._text(monkeypatch, page, structured_listings=True,
                                               listing_keep_links=False)

    def test_a_prose_item_keeps_its_missing_detail_link(self, monkeypatch):
        page = self.PAGE.replace("<h1>Used Mustangs</h1>", "<h1>Used Mustangs</h1><p>VIN 1FA6P8TH7N5135451 "
                                 "is our featured car this week, come and see it.</p>")
        text = self._text(monkeypatch, page, structured_listings=True, listing_keep_links=True)
        rows = text.split("## Listings", 1)[1].strip().splitlines()
        assert rows == ["- Used 2022 Ford Mustang | https://www.autotrader.com/cars-for-sale/vehicle/1"], rows
        # A link the prose already carries is not repeated.
        linked = page.replace("come and see it.", "https://www.autotrader.com/cars-for-sale/vehicle/1")
        assert "## Listings" not in self._text(monkeypatch, linked, structured_listings=True,
                                               listing_keep_links=True)


class TestRelevanceGate:
    def _res(self):
        from searchio.router import RouteResult

        return RouteResult(docs=[
            Doc(url="https://en.wikipedia.org/wiki/List_of_Ford_factories", title="List of Ford factories",
                source="wikipedia"),
            Doc(url="https://www.autotrader.com/cars-for-sale/2023/ford/mustang/nashville-tn",
                title="Used 2023 Ford Mustang for Sale in Nashville, TN", source="duckduckgo")], used=["x"])

    def test_drops_off_topic_docs_and_counts_them(self):
        from searchio.engine import _relevance_gate

        q = Query(text="2023 Mustang GT manual low miles for sale Tennessee", intent="web", k=10)
        out = _relevance_gate(q, self._res(), 0.25)
        assert [d.source for d in out.docs] == ["duckduckgo"]
        assert out.off_topic == {"wikipedia": 1}

    def test_short_queries_are_left_alone(self):
        from searchio.engine import _relevance_gate

        res = self._res()
        assert _relevance_gate(Query(text="ford mustang", intent="web", k=10), res, 0.9) is res


class TestCommerceSkipReference:
    def _router(self, tmp_path, **kw):
        wiki = Timed("wikipedia", 0)
        wiki.caps = frozenset({Capability.WEB, Capability.REFERENCE})
        return _router(tmp_path, [Timed("duckduckgo", 0), wiki], **kw)

    def test_commercial_web_query_skips_reference(self, tmp_path):
        buy = Query(text="used Ford Mustang manual for sale Chattanooga", intent="web", k=10)
        assert "wikipedia" in [p.name for p in self._router(tmp_path).select(buy)]
        on = self._router(tmp_path, commerce_skip_reference=True)
        assert [p.name for p in on.select(buy)] == ["duckduckgo"]
        info = Query(text="Ford Mustang history", intent="web", k=10)
        assert "wikipedia" in [p.name for p in on.select(info)]

    def test_regex_is_strict(self):
        from searchio.router import _COMMERCE

        for q in ("best price sony wh-1000xm5", "cheapest flights to paris", "under $500 laptop",
                  "where to buy a used kayak", "price of iphone 16 pro"):
            assert _COMMERCE.search(q), q
        for q in ("pricing page design", "deal with merge conflicts git", "price elasticity of demand",
                  "most used python library", "buy vs build software decision"):
            assert not _COMMERCE.search(q), q
