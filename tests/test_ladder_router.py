"""Ladder escalation, learned tiers, pacing, and router behaviour.

No network. The tiers are stubbed so the *policy* is what gets tested: when
does the ladder climb, when does it refuse to climb, and what does it remember.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from searchio.config import Settings
from searchio.errors import Blocked, SearchioError, TransientError
from searchio.models import Doc, Query
from searchio.net.ladder import Ladder
from searchio.net.profiles import DomainStore
from searchio.net.ratelimit import DomainLimiter
from searchio.providers.base import Health, Provider
from searchio.providers.registry import Registry
from searchio.router import Router

HTML = "text/html"
GOOD = "<html><body>" + ("Plenty of real readable article content here. " * 20) + "</body></html>"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,  # keep tests fast; pacing is tested separately
        per_domain_burst=1000,
        sidecar_autostart=False,
    )


class FakeLadder(Ladder):
    """Ladder with the transports replaced by a scripted table of responses."""

    def __init__(self, settings, responses):
        super().__init__(settings)
        self.responses = responses  # tier -> (status, headers, body)
        self.attempts: list[int] = []

    async def _try_tier(self, tier, url, *, referer="", rendered=False):
        self.attempts.append(tier)
        entry = self.responses.get(tier)
        if entry is None:
            raise TransientError(f"tier {tier} unavailable")
        if isinstance(entry, Exception):
            raise entry
        status, headers, body = entry
        return status, headers, body, headers.get("content-type", HTML), url


class TestEscalation:
    async def test_stops_at_first_usable_tier(self, settings):
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        res = await lad.fetch("https://a.com/x")
        assert res.tier == 0 and lad.attempts == [0]
        await lad.close()

    async def test_climbs_past_a_block(self, settings):
        lad = FakeLadder(settings, {
            0: (403, {"server": "cloudflare"}, "<html>Just a moment...</html>"),
            1: (200, {"content-type": HTML}, GOOD),
        })
        res = await lad.fetch("https://b.com/x")
        assert res.tier == 1
        assert lad.attempts == [0, 1]
        assert res.escalations == ["tier0:http_403"]
        await lad.close()

    async def test_climbs_past_an_empty_shell(self, settings):
        lad = FakeLadder(settings, {
            0: (200, {"content-type": HTML}, '<html><div id="root"></div></html>'),
            1: (200, {"content-type": HTML}, GOOD),
        })
        res = await lad.fetch("https://c.com/x")
        assert res.tier == 1 and res.escalations == ["tier0:empty_mount"]
        await lad.close()

    async def test_raises_blocked_when_every_tier_refuses(self, settings):
        blocked = (403, {"set-cookie": "datadome=x"}, "<html>no</html>")
        lad = FakeLadder(settings, {0: blocked, 1: blocked, 2: blocked})
        with pytest.raises(Blocked) as exc:
            await lad.fetch("https://d.com/x")
        assert exc.value.vendor == "datadome"
        await lad.close()

    async def test_transient_error_retries_same_tier_and_does_not_escalate(self, settings):
        # The expensive mistake this guards against: a flaky connection must not
        # cost a browser launch.
        lad = FakeLadder(settings, {
            0: TransientError("connection reset"),
            1: (200, {"content-type": HTML}, GOOD),
        })
        await lad.fetch("https://e.com/x")
        assert lad.attempts[:2] == [0, 0], "tier 0 should be retried before escalating"
        assert lad.attempts == [0, 0, 1], "one retry, then the next CHEAP tier only"
        await lad.close()

    async def test_transient_never_boots_a_browser(self, settings):
        # patchright primary (the default): tiers 0/1 both flaky, and tier 2
        # would succeed -- but a network flake must not be what boots a real
        # browser. Raise honestly instead. The caller can force the browser
        # explicitly when it knows better. bench/span.py
        # ctl.timeout_not_escalated pins this live.
        settings.sidecar_engine = False
        lad = FakeLadder(settings, {
            0: TransientError("connection reset"),
            1: TransientError("timed out"),
            2: (200, {"content-type": HTML}, GOOD),
        })
        with pytest.raises(TransientError):
            await lad.fetch("https://e2.com/x")
        assert lad.attempts == [0, 0, 1, 1]
        assert 2 not in lad.attempts, "a flake climbed into a browser pass"
        await lad.close()

    async def test_transient_may_climb_to_a_cheap_engine_tier2(self, settings):
        # Engine tier 2 is not a browser boot -- it is another cheap HTTP
        # stack. Climbing to it on a flake is fine (and often wins: the
        # engine's reqwest stack dodges whatever reset the cheap tiers hit).
        settings.sidecar_engine = True
        lad = FakeLadder(settings, {
            0: TransientError("connection reset"),
            1: TransientError("timed out"),
            2: (200, {"content-type": HTML}, GOOD),
        })
        res = await lad.fetch("https://e3.com/x")
        assert res.tier == 2
        assert lad.attempts == [0, 0, 1, 1, 2]
        await lad.close()

    async def test_respects_max_tier(self, settings):
        settings.max_tier = 0
        lad = FakeLadder(settings, {0: (403, {}, "x"), 1: (200, {}, GOOD)})
        with pytest.raises(Blocked):
            await lad.fetch("https://f.com/x")
        assert lad.attempts == [0]
        await lad.close()


class TestLearning:
    async def test_remembers_required_tier_and_starts_there(self, settings):
        responses = {
            0: (403, {"server": "cloudflare"}, "<html>Just a moment...</html>"),
            1: (200, {"content-type": HTML}, GOOD),
        }
        lad = FakeLadder(settings, responses)
        await lad.fetch("https://g.com/1")
        assert lad.attempts == [0, 1]

        lad.attempts.clear()
        # Disable the occasional cheap re-probe so the assertion is deterministic.
        lad.profiles.should_probe = lambda: False
        await lad.fetch("https://g.com/2")
        assert lad.attempts == [1], "should skip the tier known to fail"
        await lad.close()

    async def test_walks_back_down_when_a_site_relaxes(self, settings):
        store = DomainStore(settings.profile_path())
        store.record_block("h.com", tier=0, vendor="cloudflare")
        assert store.get("h.com").min_tier == 1
        store.record_success("h.com", tier=0)
        assert store.get("h.com").min_tier == 0
        store.close()


class TestRateLimiter:
    async def test_backs_off_hard_on_block(self):
        lim = DomainLimiter(rps=2.0, burst=5)
        before = lim.rate_for("x.com")
        lim.record_block("x.com")
        assert lim.rate_for("x.com") == pytest.approx(before * 0.5)

    async def test_recovers_slowly_on_success(self):
        lim = DomainLimiter(rps=1.0, burst=1, max_rps=2.0)
        lim.record_block("y.com")
        low = lim.rate_for("y.com")
        for _ in range(5):
            lim.record_success("y.com")
        assert low < lim.rate_for("y.com") < 1.0, "recovery must be gradual"

    async def test_block_surrenders_burst_tokens(self):
        # Halving the refill rate while three tokens are still in hand means the
        # next three requests go out at full speed, which is what just failed.
        lim = DomainLimiter(rps=5.0, burst=5)
        await lim.acquire("z.com")
        lim.record_block("z.com")
        assert lim._bucket("z.com").tokens == 0.0


class StubProvider(Provider):
    def __init__(self, name, caps, docs=None, fail=None, delay=0.0):
        self.name = name
        self.caps = frozenset(caps)
        super().__init__()
        self._docs = docs or []
        self._fail = fail
        self._delay = delay

    async def search(self, q, ctx):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise self._fail
        return list(self._docs)


class TestRouter:
    def _router(self, settings, providers):
        return Router(object(), settings=settings, registry=Registry(providers))

    async def test_one_failing_provider_does_not_fail_the_search(self, settings):
        from searchio.models import Capability

        good = StubProvider("duckduckgo", [Capability.WEB],
                            [Doc(url="https://a.com/1", title="A", source="duckduckgo")])
        bad = StubProvider("bing", [Capability.WEB], fail=RuntimeError("boom"))
        r = self._router(settings, [good, bad])
        res = await r.search(Query(text="x", intent="web", k=5))
        assert res.docs and "bing" in res.failed and "duckduckgo" in res.used

    async def test_slow_provider_times_out_without_blocking(self, settings):
        from searchio.models import Capability

        settings.provider_timeout_s = 0.05
        fast = StubProvider("duckduckgo", [Capability.WEB],
                            [Doc(url="https://a.com/1", title="A", source="duckduckgo")])
        slow = StubProvider("bing", [Capability.WEB], delay=2.0)
        r = self._router(settings, [fast, slow])
        res = await r.search(Query(text="x", intent="web", k=5))
        assert res.failed.get("bing") == "timeout" and res.docs

    async def test_selects_specialists_for_intent(self, settings):
        from searchio.models import Capability

        providers = [
            StubProvider("duckduckgo", [Capability.WEB]),
            StubProvider("openalex", [Capability.ACADEMIC]),
            StubProvider("github", [Capability.CODE]),
        ]
        r = self._router(settings, providers)
        names = [p.name for p in r.select(Query(text="x", intent="academic"))]
        assert "openalex" in names and "github" not in names

    async def test_always_includes_a_generalist(self, settings):
        # Specialists have narrow indexes; a purely specialist fan-out answers
        # the question it recognised rather than the one that was asked.
        from searchio.models import Capability

        providers = [
            StubProvider("duckduckgo", [Capability.WEB]),
            StubProvider("openalex", [Capability.ACADEMIC]),
        ]
        r = self._router(settings, providers)
        names = [p.name for p in r.select(Query(text="x", intent="academic"))]
        assert "duckduckgo" in names

    async def test_malformed_url_does_not_crash_a_domain_scoped_search(self, settings):
        # Bug 42 (iteration 33, DeepSeek router review): _host_of ->
        # urlsplit(url).hostname RAISES ValueError on a malformed URL
        # ("http://[::1" -> "Invalid IPv6 URL"), and the domain post-filter
        # comprehension in search() was unguarded -- so ONE bad URL from ONE
        # provider crashed the WHOLE fused query (every other provider's good
        # docs lost), but only on a site:/domain-scoped search.
        from searchio.models import Capability

        p = StubProvider("bing", [Capability.WEB], [
            Doc(url="https://www.epa.gov/ok", title="OK", source="bing", rank=0),
            Doc(url="http://[::1", title="Bad", source="bing", rank=1),
        ])
        r = self._router(settings, [p])
        res = await r.search(Query(text="x", intent="web", k=5, domains=["epa.gov"]))
        assert [d.url for d in res.docs] == ["https://www.epa.gov/ok"], \
            "a malformed URL must drop out, not crash the whole search"

    async def test_domain_filter_is_case_insensitive(self, settings):
        # Bug 43 (iteration 33): urlsplit lowercases the host, but a domain
        # passed structurally (the tool's domains= param) is NOT lowercased
        # (only inline site: is, via engine._extract_operators). A mixed-case
        # domains=["EPA.gov"] then matched nothing -- a site-scoped query
        # silently returned empty.
        from searchio.models import Capability

        p = StubProvider("bing", [Capability.WEB],
                         [Doc(url="https://www.epa.gov/x", title="EPA",
                              source="bing", rank=0)])
        r = self._router(settings, [p])
        res = await r.search(Query(text="x", intent="web", k=5, domains=["EPA.gov"]))
        assert [d.url for d in res.docs] == ["https://www.epa.gov/x"], \
            "a mixed-case domain operator must still match"

    async def test_fanout_one_returns_exactly_one_provider(self, settings):
        # Bug 44 (iteration 33): the generalist injection trimmed chosen to
        # max(n-1, 1) then appended one -- so fanout=1 returned TWO providers
        # (chosen[:1] + [generalist]), busting the requested budget on k=1
        # queries. max(n-1, 0) respects it.
        from searchio.models import Capability

        providers = [
            StubProvider("openalex", [Capability.ACADEMIC]),
            StubProvider("duckduckgo", [Capability.WEB]),
        ]
        r = self._router(settings, providers)
        chosen = r.select(Query(text="x", intent="academic"), fanout=1)
        assert len(chosen) == 1, [p.name for p in chosen]

    async def test_expensive_providers_excluded_by_default(self, settings):
        from searchio.models import Capability

        cheap = StubProvider("duckduckgo", [Capability.WEB])
        pricey = StubProvider("sidecar_search", [Capability.WEB])
        pricey.cost_per_1k = 5.0
        r = self._router(settings, [cheap, pricey])
        assert "sidecar_search" not in [p.name for p in r.select(Query(text="x", intent="web"))]
        assert "sidecar_search" in [
            p.name for p in r.select(Query(text="x", intent="web"), allow_expensive=True)
        ]

    async def test_freshness_query_upweights_date_aware_providers(self, settings):
        # DDG's no-JS endpoints ignore the date filter (byte-identical SERP
        # with and without df=w, verified live). A freshness query that
        # weights every provider alike returns DDG's stale SERP on top and
        # buries the dated rows -- bench/span.py news.fresh_week.
        from searchio.models import Capability
        import datetime as dt

        ddg = StubProvider("duckduckgo", [Capability.NEWS],
                           [Doc(url="https://stale.com/old", title="Old",
                                source="duckduckgo", rank=0)])
        # A fixed date rots against freshness="week" as the calendar advances
        # (this row tipped over on 2026-09-17, nine days after 2026-09-08);
        # compute it relative like test_freshness_bound_is_enforced below.
        two_days_old = (dt.date.today() - dt.timedelta(days=2)).isoformat()
        bing = StubProvider("bing", [Capability.NEWS],
                            [Doc(url="https://fresh.com/new", title="New",
                                 source="bing", rank=0, published=two_days_old)])
        r = self._router(settings, [ddg, bing])
        stale = await r.search(Query(text="x", intent="news", k=5))
        assert stale.docs[0].url == "https://stale.com/old", "news affinity favours ddg"
        fresh = await r.search(Query(text="x", intent="news", k=5, freshness="week"))
        assert fresh.docs[0].url == "https://fresh.com/new", \
            "a dated, filter-honouring provider must win a freshness query"

    async def test_domain_restriction_is_enforced_after_the_fanout(self, settings):
        # The site: operators only reach providers that speak them; Wikipedia
        # never sees them and Bing's RSS honors them partially at best. A
        # domain-scoped query must STILL never emit an off-domain doc --
        # bench/span.py web.gov_domain caught samsclub.com in an epa.gov-only
        # query. The guarantee lives in the router, after the fan-out.
        from searchio.models import Capability

        in_scope = StubProvider(
            "bing", [Capability.WEB],
            [Doc(url="https://www.epa.gov/x", title="EPA", source="bing", rank=0)])
        off_scope = StubProvider(
            "wikipedia", [Capability.WEB],
            [Doc(url="https://en.wikipedia.org/wiki/Fuel", title="Wiki",
                 source="wikipedia", rank=0),
             Doc(url="https://www.samsclub.com/deal", title="Deal",
                 source="wikipedia", rank=1)])
        r = self._router(settings, [in_scope, off_scope])
        res = await r.search(
            Query(text="fuel economy data", intent="web", k=5,
                  domains=["epa.gov", "fueleconomy.gov"]))
        assert [d.url for d in res.docs] == ["https://www.epa.gov/x"]
        assert "wikipedia" not in res.used, \
            "a provider whose docs all filter out contributed nothing"
        # ...but it must not be INVISIBLE either: answered-off-domain is a
        # different story from errored or empty, and the caller deserves it.
        assert res.filtered == {"wikipedia": 2}

    async def test_exclude_domains_is_enforced_after_the_fanout(self, settings):
        # Same guarantee in the other direction: -site: is a hint, the
        # router's post-filter is the enforcement. Dot-boundary matching, so
        # excluding example.com also excludes www.example.com.
        from searchio.models import Capability

        prov = StubProvider(
            "duckduckgo", [Capability.WEB],
            [Doc(url="https://w3schools.com/python", title="W", source="duckduckgo",
                 rank=0),
             Doc(url="https://realpython.com/python", title="RP", source="duckduckgo",
                 rank=1)])
        r = self._router(settings, [prov])
        res = await r.search(
            Query(text="python tutorial", intent="web", k=5,
                  exclude_domains=["w3schools.com"]))
        assert [d.url for d in res.docs] == ["https://realpython.com/python"]

    async def test_freshness_bound_is_enforced_after_the_fanout(self, settings):
        # df= and filters= are hints the engines are known to ignore (DDG
        # serves a byte-identical SERP regardless); HN's freshness handling is
        # loose. A week-bound query must never emit a provably-stale doc --
        # bench/span.py tool.freshness_week caught a 2025-03 doc in one. Same
        # shape as the domain post-filter: enforce after the fan-out.
        import datetime as dt

        from searchio.models import Capability

        today = dt.date.today()
        fresh = (today - dt.timedelta(days=3)).isoformat()
        stale = (today - dt.timedelta(days=60)).isoformat()
        prov = StubProvider(
            "hackernews", [Capability.WEB, Capability.NEWS],
            [Doc(url="https://a.com/fresh", title="Fresh", source="hackernews",
                 rank=0, published=fresh),
             Doc(url="https://a.com/stale", title="Stale", source="hackernews",
                 rank=1, published=stale),
             Doc(url="https://a.com/undated", title="Undated",
                 source="hackernews", rank=2, published=None)])
        r = self._router(settings, [prov])
        res = await r.search(
            Query(text="spacex starship", intent="news", k=5, freshness="week"))
        urls = [d.url for d in res.docs]
        assert "https://a.com/stale" not in urls, "provably stale must not ship"
        assert "https://a.com/fresh" in urls
        assert "https://a.com/undated" in urls, \
            "undated docs cannot be proven stale; dropping them guts providers"
        assert res.filtered_stale == {"hackernews": 1}

    async def test_freshness_all_never_filters(self, settings):
        import datetime as dt

        from searchio.models import Capability

        ancient = (dt.date.today() - dt.timedelta(days=4000)).isoformat()
        prov = StubProvider(
            "wikipedia", [Capability.WEB],
            [Doc(url="https://a.com/old", title="Old", source="wikipedia",
                 rank=0, published=ancient)])
        r = self._router(settings, [prov])
        res = await r.search(
            Query(text="roman empire", intent="web", k=5, freshness="all"))
        assert [d.url for d in res.docs] == ["https://a.com/old"]
        assert res.filtered_stale == {}


class TestCircuitBreaker:
    def test_opens_after_consecutive_failures(self):
        h = Health()
        for _ in range(Health.TRIP_AFTER):
            h.record(False, 100, "err")
        assert h.open and h.score == 0.0

    def test_a_success_closes_it(self):
        h = Health()
        for _ in range(Health.TRIP_AFTER):
            h.record(False, 100, "err")
        h.record(True, 100)
        assert not h.open

    def test_intermittent_failures_do_not_open(self):
        h = Health()
        for _ in range(10):
            h.record(False, 100, "err")
            h.record(True, 100)
        assert not h.open


class TestThinPageFallback:
    """A page that is genuinely short must not be discarded.

    The min_text floor cannot distinguish an empty SPA shell from a real page
    with 64 characters on it, so when nothing ever actually refused us the
    ladder returns the best body it got instead of raising.
    """

    async def test_thin_page_returned_when_never_blocked(self, settings):
        # Shaped like the real case: a little visible text inside a large
        # document, not a 40-byte stub.
        thin = (
            "<html><body><h1>NOWSECURE</h1><script>"
            + ("var pad=1;" * 200)
            + "</script></body></html>"
        )
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, thin),
                                    1: (200, {"content-type": HTML}, thin),
                                    2: (200, {"content-type": HTML}, thin)})
        res = await lad.fetch("https://short.example/")
        assert res.status == 200
        assert "NOWSECURE" in res.body
        assert any(e.startswith("accepted_thin") for e in res.escalations)
        await lad.close()

    async def test_a_real_block_still_raises(self, settings):
        # The fallback must not paper over an actual refusal.
        lad = FakeLadder(settings, {
            0: (200, {"content-type": HTML}, "<html><body>hi</body></html>"),
            1: (403, {"set-cookie": "datadome=x"}, "<html>no</html>"),
            2: (403, {"set-cookie": "datadome=x"}, "<html>no</html>"),
        })
        with pytest.raises(Blocked):
            await lad.fetch("https://mixed.example/")
        await lad.close()

    async def test_empty_shell_is_not_rescued(self, settings):
        # An empty React mount is exactly what the floor exists to catch.
        shell = '<html><body><div id="root"></div></body></html>'
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, shell),
                                    1: (200, {"content-type": HTML}, shell),
                                    2: (200, {"content-type": HTML}, shell)})
        with pytest.raises(TransientError):
            await lad.fetch("https://shell.example/")
        await lad.close()


class TestTier2WarmUp:
    """A deep page on a hostile host gets an origin visit first.

    Arriving at a deep URL with no cookies and no referer is not what a real
    visitor looks like. Measured: upwork and crunchbase deep pages refused the
    browser on first contact and passed once it had seen the origin.
    """

    async def test_refusal_triggers_warmup_then_retry(self, settings):
        calls: list[tuple[str, str]] = []

        class FakeSidecar:
            available = True

            async def fetch(self, url, **kw):
                calls.append(("fetch", url))
                # Refuse until the origin has been visited.
                if any(c[0] == "goto" for c in calls):
                    return {"ok": True, "status": 200, "html": GOOD, "url": url}
                return {"ok": True, "status": 403, "html": "denied", "url": url}

            async def goto(self, url, **kw):
                calls.append(("goto", url))
                return {"ok": True, "status": 200}

            async def cookies(self):
                return []

            async def user_agent(self):
                return ""

            async def close(self):
                return None

        lad = Ladder(settings, sidecar=FakeSidecar())
        res = await lad._tier2("https://hostile.example/deep/page")
        assert res[0] == 200
        assert calls[0] == ("fetch", "https://hostile.example/deep/page")
        assert calls[1] == ("goto", "https://hostile.example/")
        assert calls[2][0] == "fetch"
        await lad.close()

    async def test_ok_response_skips_warmup(self, settings):
        calls: list[str] = []

        class FakeSidecar:
            available = True

            async def fetch(self, url, **kw):
                calls.append("fetch")
                return {"ok": True, "status": 200, "html": GOOD, "url": url}

            async def goto(self, url, **kw):
                calls.append("goto")
                return {"ok": True}

            async def cookies(self):
                return []

            async def user_agent(self):
                return ""

            async def close(self):
                return None

        lad = Ladder(settings, sidecar=FakeSidecar())
        await lad._tier2("https://fine.example/page")
        assert calls == ["fetch"], "a page that worked must not cost a warm-up"
        await lad.close()

    def test_403_inside_an_ok_envelope_counts_as_refusal(self, settings):
        lad = Ladder(settings)
        assert lad._sidecar_refused({"ok": True, "status": 403})
        assert lad._sidecar_refused({"ok": True, "status": 200, "bot_wall": True})
        assert not lad._sidecar_refused({"ok": True, "status": 200})

    def test_origin_of(self, settings):
        lad = Ladder(settings)
        assert lad._origin_of("https://a.com/deep/page?x=1") == "https://a.com/"
        assert lad._origin_of("not a url") == ""


class TestThinFallbackHasAFloor:
    """The thin fallback must not rescue an empty response.

    Observed in a benchmark run: a sidecar reply arrived in 31 ms with a
    zero-length body and was scored as a successful retrieval. A body with no
    visible text is not a thin page, it is no page.
    """

    async def test_empty_body_is_not_accepted(self, settings):
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, ""),
                                    1: (200, {"content-type": HTML}, ""),
                                    2: (200, {"content-type": HTML}, "")})
        with pytest.raises(TransientError):
            await lad.fetch("https://empty.example/")
        await lad.close()

    async def test_markup_with_no_visible_text_is_not_accepted(self, settings):
        blank = "<html><head><title></title></head><body>" + ("<div></div>" * 200) + "</body></html>"
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, blank),
                                    1: (200, {"content-type": HTML}, blank),
                                    2: (200, {"content-type": HTML}, blank)})
        with pytest.raises(TransientError):
            await lad.fetch("https://blank.example/")
        await lad.close()

    async def test_a_genuinely_small_page_still_passes(self, settings):
        # nowsecure.nl: 64 characters of text inside a large document.
        body = ("<html><body><h1>NOWSECURE</h1><p>by nodriver</p>"
                + "<script>" + ("x" * 2000) + "</script></body></html>")
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, body)})
        res = await lad.fetch("https://tiny.example/")
        assert "NOWSECURE" in res.body
        await lad.close()


class TestProxyPlumbing:
    """IP reputation is the one anti-bot input fingerprint work cannot move."""

    def test_proxy_reaches_the_sidecar_environment(self, settings, monkeypatch):
        from searchio.net.sidecar import SidecarClient

        sc = SidecarClient(proxy="http://user:pass@proxy.example:8080", autostart=False)
        assert sc.proxy == "http://user:pass@proxy.example:8080"

    def test_browser_proxy_overrides_the_general_one(self, settings):
        # Proxy plumbing is how the ladder feeds LAUNCH options to the
        # patchright client; the engine backend takes no proxy (its reqwest
        # client follows system proxy config — the documented divergence), so
        # pin the default backend for this assertion regardless of an ambient
        # SEARCHIO_SIDECAR_ENGINE dogfood run.
        settings.sidecar_engine = False
        settings.proxy = "http://cheap:1"
        settings.browser_proxy = "http://residential:2"
        lad = Ladder(settings)
        assert lad.sidecar.proxy == "http://residential:2"

    def test_general_proxy_used_when_no_browser_specific_one(self, settings):
        settings.sidecar_engine = False  # see the sibling test's comment
        settings.proxy = "http://cheap:1"
        settings.browser_proxy = ""
        lad = Ladder(settings)
        assert lad.sidecar.proxy == "http://cheap:1"

    def test_no_proxy_by_default(self, settings):
        assert Ladder(settings).sidecar.proxy == ""


class TestRenderedTier:
    """Two backends, one ladder: engine primary + patchright challenge.

    The policy under test: ``rendered=True`` routes tier 2 to the challenge
    sidecar; an unreadable engine tier-2 answer is rescued once through it;
    a patchright-primary ladder never escalates (it already IS the browser);
    and the challenge sidecar is never built for a page the primary reads.
    """

    SHELL = '<html><body><div id="root"></div></body></html>'

    class RecordingSidecar:
        """Minimal SidecarClient stand-in recording fetch/goto calls."""

        available = True

        def __init__(self, body="", status=200):
            self.calls: list[tuple] = []
            self._body = body
            self._status = status

        async def fetch(self, url, **kw):
            self.calls.append(("fetch", url))
            return {"ok": True, "status": self._status, "html": self._body, "url": url}

        async def goto(self, url, **kw):
            self.calls.append(("goto", url))
            return {"ok": True, "status": 200}

        async def cookies(self):
            return []

        async def user_agent(self):
            return ""

        async def close(self):
            return None

    class NoHttpLadder(Ladder):
        """Tiers 0/1 refuse to touch the network; tier 2 comes from fakes."""

        async def _tier0(self, url, *, referer=""):
            raise TransientError("no network in unit tests")

        async def _tier1(self, url, *, referer=""):
            raise TransientError("no network in unit tests")

    class CheapAnswersFirstLadder(Ladder):
        """Tier 0 answers everything with a perfectly readable page."""

        async def _tier0(self, url, *, referer=""):
            html = "<html><body><p>" + ("visible prose paragraph " * 30) + "</p></body></html>"
            return 200, {}, html, "text/html", url

        async def _tier1(self, url, *, referer=""):
            raise TransientError("no network in unit tests")

    def _engine_settings(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "patchright"
        return settings

    async def test_rendered_flag_routes_tier2_to_challenge(self, settings):
        self._engine_settings(settings)
        primary = self.RecordingSidecar(body=GOOD)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            res = await lad.fetch(
                "https://r.example/x", force_tier=2, rendered=True, use_cache=False
            )
            assert res.rendered is True and res.tier == 2
            assert len(challenge.calls) == 1 and not primary.calls
        finally:
            await lad.close()

    async def test_rendered_request_answered_by_cheap_tier_is_not_stamped(self, settings):
        # rendered=True is a request, not a guarantee: when a cheap tier
        # answers it, the result must NOT claim a browser pass. The
        # mislabelled "full-browser render" teaches the caller rendering was
        # spent and invites wasted escalations. Caught live by bench/combo.py.
        self._engine_settings(settings)
        primary = self.RecordingSidecar(body=GOOD)
        lad = self.CheapAnswersFirstLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch("https://c.example/x", rendered=True, use_cache=False)
            assert res.tier == 0 and res.rendered is False
            assert "tier2_rendered" not in lad.stats()["tiers"]
            assert not primary.calls, "no tier-2 pass should have happened"
            assert lad._challenge is None
        finally:
            await lad.close()

    async def test_engine_shell_auto_escalates_once(self, settings):
        self._engine_settings(settings)
        primary = self.RecordingSidecar(body=self.SHELL)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            res = await lad.fetch("https://s.example/x", force_tier=2, use_cache=False)
            assert res.rendered is True, f"escalations: {res.escalations}"
            assert len(primary.calls) == 1 and len(challenge.calls) == 1
            assert "tier2:auto_rendered" in res.escalations
        finally:
            await lad.close()

    async def test_no_escalation_when_primary_is_the_browser(self, settings):
        # patchright primary (the default config): it already IS the fidelity
        # backend, so a shell must not reach for anything extra.
        settings.sidecar_engine = False
        settings.sidecar_challenge = "patchright"
        primary = self.RecordingSidecar(body=self.SHELL)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            with pytest.raises(TransientError):
                await lad.fetch("https://n.example/x", force_tier=2, use_cache=False)
            assert len(primary.calls) == 1 and not challenge.calls
        finally:
            await lad.close()

    async def test_rescue_falls_back_when_challenge_also_shells(self, settings):
        self._engine_settings(settings)
        primary = self.RecordingSidecar(body=self.SHELL)
        challenge = self.RecordingSidecar(body=self.SHELL)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://d.example/x", force_tier=2, use_cache=False)
            assert len(challenge.calls) == 1, "rescue fires at most once"
            assert "tier2:auto_rendered" in str(exc.value)
        finally:
            await lad.close()

    async def test_same_backend_de_duplicates_the_client(self, settings):
        # patchright primary + patchright challenge (the shipped defaults):
        # an explicit rendered pass rides the SAME client; no second browser.
        settings.sidecar_engine = False
        settings.sidecar_challenge = "patchright"
        primary = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch(
                "https://q.example/x", force_tier=2, rendered=True, use_cache=False
            )
            assert res.rendered is True and lad._challenge is lad.sidecar
            assert len(primary.calls) == 1
        finally:
            await lad.close()

    async def test_normal_page_never_builds_the_challenge_sidecar(self, settings):
        self._engine_settings(settings)
        primary = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch("https://g.example/x", force_tier=2, use_cache=False)
            assert res.rendered is False and res.tier == 2
            assert lad._challenge is None, "patchright must stay unbuilt for a readable page"
        finally:
            await lad.close()

    async def test_auto_rescue_disabled_by_setting(self, settings):
        self._engine_settings(settings)
        settings.sidecar_challenge_auto = False
        primary = self.RecordingSidecar(body=self.SHELL)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            with pytest.raises(TransientError):
                await lad.fetch("https://o.example/x", force_tier=2, use_cache=False)
            assert not challenge.calls
        finally:
            await lad.close()

    async def test_thin_page_without_scripts_is_not_rescued(self, settings):
        # A genuinely tiny page (real text, no script, no mount) is thin for a
        # browser too. Rescuing it burns a full render to re-discover the same
        # characters -- first caught live by bench/span.py ctl.thin_page_fallback.
        self._engine_settings(settings)
        thin = ("<html><body><p>small but real page</p><!--"
                + "x" * 600 + "--></body></html>")
        primary = self.RecordingSidecar(body=thin)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            res = await lad.fetch("https://t.example/x", force_tier=2, use_cache=False)
            assert any(e.startswith("accepted_thin") for e in res.escalations), res.escalations
            assert res.rendered is False
            assert not challenge.calls, "no render-could-help evidence: no rescue"
            assert lad.stats()["tiers"].get("tier2_auto_render", 0) == 0
        finally:
            await lad.close()

    async def test_thin_page_with_script_evidence_is_rescued(self, settings):
        # The other half of the gate: too_thin WITH a script in the body is the
        # classic SPA-chrome signature (header/footer around an unexecuted
        # bundle), which a real browser does fix.
        self._engine_settings(settings)
        spa_chrome = ("<html><body><header>nav</header><div id=\"main\"></div>"
                      "<script>render()</script><!--" + "x" * 600
                      + "--></body></html>")
        primary = self.RecordingSidecar(body=spa_chrome)
        challenge = self.RecordingSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            res = await lad.fetch("https://u.example/x", force_tier=2, use_cache=False)
            assert res.rendered is True, f"escalations: {res.escalations}"
            assert len(primary.calls) == 1 and len(challenge.calls) == 1
            assert "tier2:auto_rendered" in res.escalations
        finally:
            await lad.close()

    async def test_rendered_requests_use_a_separate_cache_variant(self, settings):
        # bench/span.py llm.price_check caught this live: read_page cached an
        # Amazon shell, and read_page_rendered for the same URL was then
        # served the SAME shell "via cache" -- no browser pass, no fresh
        # content, and the model burned its rendered budget retrying.
        # Rendered requests cache under their own variant so a plain read
        # can never answer them (and vice versa).
        settings.cache_enabled = True
        lad = self.CheapAnswersFirstLadder(settings)
        try:
            url = "https://v.example/x"
            r1 = await lad.fetch(url)
            assert r1.tier == 0 and not r1.from_cache
            r2 = await lad.fetch(url, rendered=True)
            assert not r2.from_cache, "rendered request answered by the plain cache entry"
            r3 = await lad.fetch(url)
            assert r3.from_cache and r3.rendered is False, "plain cache still serves plain calls"
            r4 = await lad.fetch(url, rendered=True)
            assert r4.from_cache, "rendered pass cached its own variant"
        finally:
            await lad.close()

    async def test_fidelity_sidecar_engine_primary_routes_to_challenge(self, settings):
        # The patchright script's composite verbs (search_engine_results and
        # friends) belong to whichever backend can serve them: with an engine
        # primary that is the challenge sidecar -- the fidelity pairing is
        # why the second backend exists. Caught live by bench/span.py
        # web.expensive_search, where the provider used to aim the verb at
        # the engine and eat "Unknown method".
        settings.sidecar_engine = True
        settings.sidecar_challenge = "patchright"
        primary = self.RecordingSidecar()
        challenge = self.RecordingSidecar()
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            assert lad.fidelity_sidecar() is challenge
        finally:
            await lad.close()

    async def test_fidelity_sidecar_patchright_primary_is_the_primary(self, settings):
        settings.sidecar_engine = False
        primary = self.RecordingSidecar()
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            assert lad.fidelity_sidecar() is primary
        finally:
            await lad.close()

    async def test_fidelity_sidecar_engine_on_both_knobs_returns_the_primary(self, settings):
        # No patchright anywhere: the primary comes back and backend_kind()
        # says "engine", so callers fail honestly before sending the verb.
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.RecordingSidecar()
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            assert lad.fidelity_sidecar() is primary
        finally:
            await lad.close()


class _MiniSidecar:
    """Minimal SidecarClient stand-in recording fetch/goto calls."""

    available = True

    def __init__(self, body="", status=200):
        self.calls: list[tuple] = []
        self._body = body
        self._status = status

    async def fetch(self, url, **kw):
        self.calls.append(("fetch", url))
        return {"ok": True, "status": self._status, "html": self._body, "url": url}

    async def goto(self, url, **kw):
        self.calls.append(("goto", url))
        return {"ok": True, "status": 200}

    async def cookies(self):
        return []

    async def user_agent(self):
        return ""

    async def close(self):
        return None


class TestTier1ErrorHonesty:
    """Bug 35 (iteration 30, DeepSeek design review of ladder.py): _tier1's
    catch-all converted EVERY non-TransientError to TransientError -- the
    keyword-matching if/else raised the identical TransientError from both
    branches, so its own comment ("only genuine transport failures should
    count as transient") was dead code. A permanent LOGIC failure inside
    _blocking (a ValueError from the hop machinery) wore the retryable
    class, was retried, and could be climbed into a tier-2 spend that
    "answered" the fetch -- hiding the dead tier forever, the exact
    expensive mistake errors.py's taxonomy exists to prevent. The fix
    classifies by ORIGIN: curl's own RequestsError and the socket/DNS
    layer are OSErrors (transport weather, transient); anything else
    escaping _blocking is logic (permanent SearchioError, never retried,
    never climbed).
    """

    async def test_logic_error_is_not_transient(self, settings, monkeypatch):
        def boom(host, port):
            raise ValueError("hop machinery broke")

        monkeypatch.setattr("searchio.net.ladder._resolve_pin", boom)
        lad = Ladder(settings, sidecar=_MiniSidecar())
        try:
            with pytest.raises(SearchioError) as exc:
                await lad._tier1("https://logic.example/x")
            assert not isinstance(exc.value, TransientError), (
                "a permanent logic failure must not wear the retryable class")
            assert "ValueError" in str(exc.value)
        finally:
            await lad.close()

    async def test_logic_error_never_climbs_into_a_tier2_spend(self, settings, monkeypatch):
        # The blast radius pre-fix: the tier loop saw TransientError, retried
        # tier 1, then CLIMBED (engine tier 2 is cheap, so climbing stays
        # allowed) -- and tier 2 "answered" the fetch, so a permanently
        # broken tier 1 hid behind escalation spend forever.
        settings.sidecar_engine = True

        def boom(host, port):
            raise ValueError("hop machinery broke")

        monkeypatch.setattr("searchio.net.ladder._resolve_pin", boom)
        primary = _MiniSidecar(body=GOOD)
        lad = Ladder(settings, sidecar=primary)
        try:
            with pytest.raises(SearchioError) as exc:
                await lad.fetch("https://logic.example/x", force_tier=1, use_cache=False)
            assert not isinstance(exc.value, TransientError)
            assert not primary.calls, "a logic failure must never buy an escalation"
        finally:
            await lad.close()

    async def test_curl_transport_error_stays_transient(self, settings, monkeypatch):
        from curl_cffi.requests.errors import RequestsError

        def reset(url, **kw):
            raise RequestsError("connection reset by peer")

        monkeypatch.setattr("curl_cffi.requests.get", reset)
        lad = Ladder(settings, sidecar=_MiniSidecar())
        try:
            with pytest.raises(TransientError):
                # An IP literal skips _resolve_pin's getaddrinfo entirely.
                await lad._tier1("https://127.0.0.1:9/x")
        finally:
            await lad.close()

    async def test_dns_weather_stays_transient(self, settings, monkeypatch):
        def dns_flap(host, port):
            raise socket.gaierror(11001, "getaddrinfo failed")

        monkeypatch.setattr("searchio.net.ladder._resolve_pin", dns_flap)
        lad = Ladder(settings, sidecar=_MiniSidecar())
        try:
            with pytest.raises(TransientError):
                await lad._tier1("https://dns.example/x")
        finally:
            await lad.close()


class TestRenderedStampBackend:
    """Bug 36 (iteration 30, DeepSeek design review of ladder.py): the
    rendered fidelity stamp was ``rendered_pass and tier == 2`` -- with no
    check that the tier-2 client is a REAL browser. An engine challenge
    sidecar (sidecar_challenge="engine" -- the engine-on-both-knobs config
    fidelity_sidecar() already contemplates) answers rendered=True with
    parse-only output, because the ladder never sends render:true
    (iteration 26); the result was stamped rendered=True anyway, teaching
    the caller a browser pass happened that never did.
    """

    class NoHttpLadder(Ladder):
        """Tiers 0/1 refuse to touch the network; tier 2 comes from fakes."""

        async def _tier0(self, url, *, referer=""):
            raise TransientError("no network in unit tests")

        async def _tier1(self, url, *, referer=""):
            raise TransientError("no network in unit tests")

    def _engine_challenge_settings(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        return settings

    async def test_engine_challenge_rendered_pass_is_not_stamped(self, settings):
        self._engine_challenge_settings(settings)
        primary = _MiniSidecar(body=GOOD)
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch(
                "https://e.example/x", force_tier=2, rendered=True, use_cache=False
            )
            assert res.tier == 2
            assert res.rendered is False, "an engine challenge sidecar never renders"
            assert lad._challenge is lad.sidecar, "engine-on-both de-duplicates the client"
            assert "tier2_rendered" not in lad.stats()["tiers"]
        finally:
            await lad.close()

    async def test_engine_challenge_thin_fallback_is_not_stamped(self, settings):
        self._engine_challenge_settings(settings)
        thin = ("<html><body><p>small but real page</p><!--"
                + "x" * 600 + "--></body></html>")
        primary = _MiniSidecar(body=thin)
        lad = self.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch(
                "https://et.example/x", force_tier=2, rendered=True, use_cache=False
            )
            assert any(e.startswith("accepted_thin") for e in res.escalations), res.escalations
            assert res.rendered is False
        finally:
            await lad.close()

    async def test_patchright_challenge_thin_fallback_still_stamped(self, settings):
        # Regression pin for the shared best-tuple: with a REAL browser as
        # the challenge backend, a thin page it earned keeps the stamp.
        settings.sidecar_engine = True
        settings.sidecar_challenge = "patchright"
        thin = ("<html><body><p>small but real page</p><!--"
                + "x" * 600 + "--></body></html>")
        primary = _MiniSidecar()
        challenge = _MiniSidecar(body=thin)
        lad = self.NoHttpLadder(settings, sidecar=primary, challenge_sidecar=challenge)
        try:
            res = await lad.fetch(
                "https://pt.example/x", force_tier=2, rendered=True, use_cache=False
            )
            assert any(e.startswith("accepted_thin") for e in res.escalations), res.escalations
            assert res.rendered is True
        finally:
            await lad.close()


class TestThinAfterWall:
    """Iteration 31 (ladder review candidate (t)): the thin-page fallback
    refused to serve when ANY tier had been blocked -- even a tier BELOW the
    one the thin page came from. The comment's own example (nowsecure.nl:
    64 characters behind a Cloudflare wall) was defeated by its own
    condition: cheap tiers 403, the browser gets through with a real-but-
    thin page, and the fetch raised Blocked -- the browser's success
    discarded, the caller told the site refused us. A block is stale once a
    HIGHER tier got a 2xx past it; only a block at or above the thin tier
    still stands.
    """

    THIN = ("<html><body><p>small but real page</p><!--"
            + "x" * 600 + "--></body></html>")
    WALL = (403, {"server": "cloudflare"}, "<html>Attention Required</html>")

    async def test_browser_thin_success_beats_an_earlier_wall(self, settings):
        lad = FakeLadder(settings, {
            0: self.WALL, 1: self.WALL,
            2: (200, {"content-type": HTML}, self.THIN),
        })
        try:
            res = await lad.fetch("https://walled.example/x")
            assert res.tier == 2
            assert any(e.startswith("accepted_thin") for e in res.escalations), res.escalations
            assert "small but real page" in res.body
        finally:
            await lad.close()

    async def test_thin_at_a_lower_tier_than_the_wall_still_refuses(self, settings):
        # The block is the LATER word: a thin page from tier 1 does not
        # outrank tier 2's refusal (the pre-existing pin, one tier over).
        lad = FakeLadder(settings, {
            0: self.WALL,
            1: (200, {"content-type": HTML}, self.THIN),
            2: self.WALL,
        })
        try:
            with pytest.raises(Blocked):
                await lad.fetch("https://walled2.example/x")
        finally:
            await lad.close()


class TestTier2ServerError:
    """Iteration 31 (candidate (s)): a 5xx carried in an ok:true tier-2
    envelope was reported as ``tier2: browser_failed`` -- the status
    dropped, the origin's outage mislabeled as OUR browser breaking -- and
    as a TransientError it bought a same-tier retry (a second browser
    navigation at a patchright primary) that a deterministic 500 can never
    reward. Worse, a 503 WAF challenge never reached classify, whose
    http_503_challenge verdict drives the blocked bookkeeping and the
    challenge rescue. The 401/403/429 branch already surfaced the real
    status; 5xx joins it, envelope headers riding along for the vendor.
    """

    async def test_5xx_surfaces_its_status_without_a_retry(self, settings):
        settings.sidecar_engine = True
        primary = _MiniSidecar(body="<html>internal server error</html>", status=500)
        lad = TestRenderedStampBackend.NoHttpLadder(settings, sidecar=primary)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://five.example/x", force_tier=2, use_cache=False)
            assert "http_500" in str(exc.value), str(exc.value)
            assert "browser_failed" not in str(exc.value)
            # One refusal dance (fetch, origin warm-up, fetch) and out: the
            # tier loop's transient same-tier retry must not double it.
            assert [c[0] for c in primary.calls] == ["fetch", "goto", "fetch"], primary.calls
        finally:
            await lad.close()

    async def test_503_challenge_reaches_classify_as_a_wall(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge_auto = False

        class WafSidecar(_MiniSidecar):
            async def fetch(self, url, **kw):
                self.calls.append(("fetch", url))
                return {"ok": True, "status": 503, "url": url,
                        "html": "<html><title>Just a moment...</title></html>",
                        "headers": {"Server": "cloudflare", "CF-RAY": "abc",
                                    "cf-mitigated": "challenge"}}

        lad = TestRenderedStampBackend.NoHttpLadder(settings, sidecar=WafSidecar())
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://waf.example/x", force_tier=2, use_cache=False)
            assert exc.value.vendor == "cloudflare", str(exc.value)
            assert "http_503_challenge" in str(exc.value)
        finally:
            await lad.close()


class TestTier2BodyCap:
    """Iteration 31 (candidate (u)): the cheap tiers refuse a body past
    max_body_bytes on the stream (iteration 18) and the engine caps at the
    source, but a PATCHRIGHT tier-2 body arrived uncapped -- the "every tier
    refuses too_large" contract had a hole on the one backend that renders
    whatever the origin sends. The ladder now applies the same decoded-
    bytes cap to every tier-2 body, whichever backend served it.
    """

    async def test_over_cap_tier2_body_is_refused(self, settings):
        settings.sidecar_engine = True
        settings.max_body_bytes = 20_000
        big = "<html><body>" + ("real words here " * 3000) + "</body></html>"
        lad = TestRenderedStampBackend.NoHttpLadder(settings, sidecar=_MiniSidecar(body=big))
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://big.example/x", force_tier=2, use_cache=False)
            assert "too_large" in str(exc.value), str(exc.value)
            assert lad._challenge is None
        finally:
            await lad.close()

    async def test_under_cap_tier2_body_still_serves(self, settings):
        settings.sidecar_engine = True
        settings.max_body_bytes = 20_000
        lad = TestRenderedStampBackend.NoHttpLadder(settings, sidecar=_MiniSidecar(body=GOOD))
        try:
            res = await lad.fetch("https://ok.example/x", force_tier=2, use_cache=False)
            assert res.tier == 2 and "readable article content" in res.body
        finally:
            await lad.close()


class TestRetrySidecarUnavailable:
    """Iteration 31 (candidate (x)): the transient same-tier retry caught
    only TransientError on its second attempt. A sidecar that flaked once
    and was then marked sticky-unavailable raised SidecarUnavailable out of
    fetch() itself -- past the loop's bookkeeping (no tierN:sidecar_unavailable
    stamp, no exhaustion message) -- while the very same failure on a FIRST
    attempt is handled in-loop.
    """

    async def test_second_attempt_unavailable_stays_in_the_loop(self, settings):
        from searchio.errors import SidecarUnavailable

        class Seq(FakeLadder):
            async def _try_tier(self, tier, url, *, referer="", rendered=False):
                self.attempts.append(tier)
                entry = self.responses[tier].pop(0)
                if isinstance(entry, Exception):
                    raise entry
                status, headers, body = entry
                return status, headers, body, headers.get("content-type", HTML), url

        lad = Seq(settings, {2: [TransientError("flake"), SidecarUnavailable("gone")]})
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://sc.example/x", force_tier=2, use_cache=False)
            assert "sidecar_unavailable" in str(exc.value), str(exc.value)
            assert lad.attempts == [2, 2]
        finally:
            await lad.close()


class TestPdfClearanceAtTier2:
    """Iteration 31 (candidate (y)): PDF extraction re-downloads the bytes
    through the tier-0 client with the domain's BANKED clearance -- but the
    clearance a tier-2 browser pass had just earned was only banked on the
    ok path, after classify. A Cloudflare-gated PDF therefore walled the
    cheap tiers, cleared at the browser, and then had its refetch walled
    again (pdf_text:refetch_failed) because the cookie the browser held was
    never banked before the refetch. Tier 2 now banks clearance BEFORE the
    extraction refetch.
    """

    async def test_tier2_pdf_refetch_carries_the_browser_clearance(self, settings):
        from searchio.net import pdf as pdf_mod
        settings.sidecar_engine = True

        class ClearedSidecar(_MiniSidecar):
            async def cookies(self):
                return [{"name": "cf_clearance", "value": "tok", "domain": ".pdf.example"}]

            async def user_agent(self):
                return "BrowserUA/1.0"

        seen: dict = {}
        data = pdf_mod.build_pdf(["alpha clearance page"])

        class Resp:
            status_code = 200
            headers: dict = {}

            async def aiter_bytes(self, n):
                yield data

            async def aclose(self):
                return None

        lad = TestRenderedStampBackend.NoHttpLadder(
            settings, sidecar=ClearedSidecar(body="%PDF-1.4 fake bytes"))

        async def open_stream(url, headers, timeout):
            seen.update(headers)
            return Resp()

        lad._open_stream = open_stream
        try:
            res = await lad.fetch("https://pdf.example/doc.pdf", force_tier=2, use_cache=False)
            assert "alpha clearance page" in res.body
            assert "cf_clearance=tok" in seen.get("Cookie", ""), seen
            assert seen.get("User-Agent") == "BrowserUA/1.0"
        finally:
            await lad.close()


class TestBinaryContentType:
    """Bug 20: a binary body must never ship as a page, whatever tier-2 says.

    The sidecar envelope used to carry no content-type and _tier2_via
    synthesized text/html for every body; classify's body heuristics then
    passed a PDF's ASCII operator stream (bench/span.py bulk seed 113: two
    live PDFs fetched ok:t2, 446-631/5000 control bytes). Two seams pinned:
    the envelope's content-type when the backend sends one, and classify's
    body-magic sniff when it does not.
    """

    PDF = "%PDF-1.7\n" + "\x00\x01\x02obj stream BT /F1 Tf ET\n" * 400

    class PdfSidecar:
        """SidecarClient stand-in whose fetch returns a PDF body."""

        available = True

        def __init__(self, extra=None):
            self._extra = extra or {}

        async def fetch(self, url, **kw):
            return {"ok": True, "status": 200, "url": url,
                    "html": TestBinaryContentType.PDF, **self._extra}

        async def goto(self, url, **kw):
            return {"ok": True, "status": 200}

        async def cookies(self):
            return []

        async def user_agent(self):
            return ""

        async def close(self):
            return None

    async def test_envelope_content_type_reaches_classify(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.PdfSidecar(extra={"content_type": "application/pdf"})
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://x.example/doc.pdf", force_tier=2,
                                use_cache=False)
            assert "not_html:application/pdf" in str(exc.value)
        finally:
            await lad.close()

    async def test_silent_envelope_is_caught_by_the_magic_sniff(self, settings):
        # A backend that reports no content-type at all (se-serve before the
        # protocol addition, patchright today): the %PDF- magic at byte zero
        # outranks the text/html the ladder has to assume.
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.PdfSidecar()
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://x.example/doc.pdf", force_tier=2,
                                use_cache=False)
            assert "binary:application/pdf" in str(exc.value)
        finally:
            await lad.close()

    async def test_real_pages_still_pass(self, settings):
        # The sniff arbitrates text-ish lies only; an ordinary HTML page
        # through the same path must remain a clean ok.
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = TestRenderedTier.RecordingSidecar(body=GOOD)
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch("https://x.example/page", force_tier=2,
                                  use_cache=False)
            assert res.tier == 2 and "readable article content" in res.body
        finally:
            await lad.close()


class TestNavErrorPage:
    """Bug 32: the browser's network-error page is never content.

    The patchright challenge browser follows page-initiated navigations
    (page script, meta refresh) NATIVELY, and when the followed target is
    unreachable the tab commits Chromium's error page -- which the pre-fix
    sidecar then read back as the fetched page (span iteration 27 bite:
    "ERROR PAGE SERVED AS CONTENT: tier=2 final=...chrome-error://chromewebdata/
    body='<!DOCTYPE html><html dir=ltr ...'"). Three seams pinned: the fixed
    sidecar's nav_error_page token short-circuits BEFORE the origin-warmup
    retry (one fetch, no dance), the ladder belt refuses even an ok envelope
    whose landed URL is chrome-error:// (an unfixed sidecar generation), and
    an honest native follow reports where it landed via final_url.
    """

    class NavSidecar:
        """SidecarClient stand-in serving one scripted fetch envelope."""

        available = True

        def __init__(self, res):
            self._res = res
            self.fetch_calls = 0
            self.goto_calls = 0

        async def fetch(self, url, **kw):
            self.fetch_calls += 1
            return dict(self._res)

        async def goto(self, url, **kw):
            self.goto_calls += 1
            return {"ok": True, "status": 200}

        async def cookies(self):
            return []

        async def user_agent(self):
            return ""

        async def close(self):
            return None

    async def test_token_refusal_raises_once_without_the_warmup_dance(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.NavSidecar({
            "ok": False,
            "error": ("nav_error_page: a page-initiated navigation landed on "
                      "the browser error page (https://x.example/shell -> "
                      "chrome-error://chromewebdata/)"),
        })
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://x.example/shell", force_tier=2,
                                use_cache=False)
            assert "nav_error_page" in str(exc.value)
            assert primary.fetch_calls == 1, (
                "a dinoed follow is permanent -- the warmup dance must not "
                "spend a second browser navigation on it")
            assert primary.goto_calls == 0
        finally:
            await lad.close()

    async def test_unfixed_sidecars_error_page_url_is_refused_by_the_belt(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.NavSidecar({
            "ok": True, "status": 200,
            "url": "chrome-error://chromewebdata/",
            "html": '<!DOCTYPE html><html dir="ltr" lang="en"><body>err</body></html>',
        })
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://x.example/shell", force_tier=2,
                                use_cache=False)
            assert "nav_error_page" in str(exc.value)
            assert primary.fetch_calls == 1
        finally:
            await lad.close()

    async def test_final_url_preferred_over_the_envelope_url(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.NavSidecar({
            "ok": True, "status": 200,
            "url": "https://x.example/shell",
            "final_url": "https://x.example/target",
            "html": GOOD,
        })
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            res = await lad.fetch("https://x.example/shell", force_tier=2,
                                  use_cache=False)
            assert res.final_url == "https://x.example/target"
        finally:
            await lad.close()


class TestTier2NavBudget:
    """Iteration 28: the tier-2 navigation budget is caller-owned.

    The verb's per-attempt nav budget used to be patchright's fixed 30 s
    default regardless of the ladder's tier2_timeout_s -- a tight budget
    still parked the call for two 30 s attempts (the timeout-soften degrade
    retries once), and a stalled navigation outlived what the agent agreed
    to. _tier2_via now forwards a third of its tier budget as timeout_ms;
    at the 90 s default that is 30000 ms, exactly patchright's own default,
    so production behavior is unchanged.
    """

    class KwSidecar(TestRenderedTier.RecordingSidecar):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.kw: dict = {}

        async def fetch(self, url, **kw):
            self.kw = kw
            return await super().fetch(url, **kw)

    async def test_nav_budget_is_a_third_of_the_tier_budget(self, settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        settings.sidecar_challenge_url = ""
        primary = self.KwSidecar(body=GOOD)
        lad = TestRenderedTier.NoHttpLadder(settings, sidecar=primary)
        try:
            await lad.fetch("https://x.example/page", force_tier=2,
                            use_cache=False)
            assert primary.kw.get("timeout_ms") == int(
                settings.tier2_timeout_s * 1000 / 3) == 30000
        finally:
            await lad.close()


class TestSidecarIdentity:
    """/healthz cannot discriminate backends: se-serve answers it with a bare
    {"ok": true}, byte-compatible with the patchright script. A leftover
    engine on the patchright port used to be adopted sight-unseen, and every
    patchright-only verb then failed deep in provider-land with a bare
    "Unknown method" -- bench/span.py web.expensive_search caught exactly
    that in a live run. ensure() now verifies the backend's self-reported
    kind via the health verb: strictly on the default-port probe (nobody
    explicitly chose that endpoint), adaptively on an explicit url= (the
    operator's choice -- an engine behind a url is how the wire-parity suite
    drives it)."""

    @staticmethod
    def _stub_sidecar(engine_field: str | None):
        """A minimal wire-compatible responder: /healthz ok, /rpc health verb
        naming itself via the engine field (or not, when None)."""
        import http.server
        import json as jsonlib
        import threading

        class _H(http.server.BaseHTTPRequestHandler):
            def _json(self, obj):
                body = jsonlib.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._json({"ok": True})

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                result = {"ok": True}
                if engine_field is not None:
                    result["engine"] = engine_field
                self._json({"jsonrpc": "2.0", "id": 0, "result": result})

            def log_message(self, *a):  # quiet
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        return f"http://127.0.0.1:{port}", port, srv

    async def test_patchright_client_refuses_an_engine_on_the_default_port(self):
        from searchio.errors import SidecarUnavailable
        from searchio.net.sidecar import SidecarClient

        _base, port, srv = self._stub_sidecar("searchio-engine")
        # The probe path requires autostart; the (missing) script is never
        # consulted because the refusal happens at the probe.
        sc = SidecarClient(port=port, script="missing-sidecar.py", autostart=True)
        try:
            with pytest.raises(SidecarUnavailable, match="refusing to attach"):
                await sc.ensure()
            assert "SEARCHIO_SIDECAR_ENGINE=1" in sc._unavailable_reason
        finally:
            srv.shutdown()

    async def test_engine_client_refuses_a_patchright_sidecar_on_the_default_port(self):
        from pathlib import Path

        from searchio.errors import SidecarUnavailable
        from searchio.net.sidecar import SidecarClient

        _base, port, srv = self._stub_sidecar("patchright")
        sc = SidecarClient(port=port, binary=Path("se-serve.exe"), autostart=True)
        try:
            with pytest.raises(SidecarUnavailable, match="refusing to attach"):
                await sc.ensure()
            assert "unset SEARCHIO_SIDECAR_ENGINE" in sc._unavailable_reason
        finally:
            srv.shutdown()

    async def test_refuses_a_server_that_does_not_name_itself(self):
        from searchio.errors import SidecarUnavailable
        from searchio.net.sidecar import SidecarClient

        base, _port, srv = self._stub_sidecar(None)
        sc = SidecarClient(url=base, autostart=False)
        try:
            with pytest.raises(SidecarUnavailable, match="unidentified"):
                await sc.ensure()
        finally:
            srv.shutdown()

    async def test_explicit_url_adopts_whichever_known_backend_answers(self):
        from searchio.net.sidecar import SidecarClient

        # An operator-pointed URL is trusted to name the backend on purpose;
        # the client remembers the kind so consumers (SidecarSearch,
        # SidecarListings) can branch on it instead of failing at the wire.
        base, _port, srv = self._stub_sidecar("searchio-engine")
        sc = SidecarClient(url=base, autostart=False)
        try:
            assert await sc.ensure() == base
            assert await sc.backend_kind() == "engine"
        finally:
            srv.shutdown()

    async def test_matching_backend_is_adopted_and_verified_once(self):
        from searchio.net.sidecar import SidecarClient

        base, _port, srv = self._stub_sidecar("patchright")
        sc = SidecarClient(url=base, autostart=False)
        try:
            assert await sc.ensure() == base
            assert await sc.backend_kind() == "patchright"
            # The identity handshake happens once per adopted URL; later
            # ensure() calls must not re-probe (call() runs ensure() per verb,
            # so a re-probe each time would double wire traffic).
            sc._verified_url = base
            assert await sc.ensure() == base
        finally:
            srv.shutdown()


class TestRefusedTarget:
    """The SSRF target guard (bug 25, iteration 19): link-local and
    unspecified IP literals -- and non-http(s) schemes -- are never fetched,
    not at fetch entry and not on any redirect hop. Hostnames are
    deliberately not resolved (the connect-time rebinding guard is a
    documented follow-up); loopback/RFC1918 stay legal (the span suite's
    own control server is 127.0.0.1). Wire-level behavior is pinned by span
    controls 67-76."""

    @pytest.mark.parametrize("url,token", [
        ("http://169.254.169.254/latest/meta-data", "link_local"),
        ("http://169.254.0.1/", "link_local"),
        ("http://0.0.0.0/", "unspecified"),
        ("http://[::ffff:a9fe:a9a9]/", "link_local"),  # the mapped-v4 dodge
        ("http://[::ffff:169.254.169.254]/", "link_local"),
        ("http://[fe80::1]/", "link_local"),
        ("http://[::]/", "unspecified"),
        ("file:///etc/passwd", "scheme:file"),
        ("ftp://example.com/x", "scheme:ftp"),
        ("//host/path", "scheme:none"),
    ])
    def test_refused_literals(self, url, token):
        from searchio.net.ladder import _refused_target
        got = _refused_target(url)
        assert got is not None and got.startswith(token), f"{url}: {got}"

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8000/prose",   # loopback: the control suite itself
        "http://192.168.1.1/",           # RFC1918: legitimate intranet
        "http://10.0.0.5/",
        "https://example.com/",
        "https://example.com:8443/x",
    ])
    def test_allowed_targets(self, url):
        from searchio.net.ladder import _refused_target
        assert _refused_target(url) is None

    async def test_entry_gate_touches_no_tier(self, settings):
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        with pytest.raises(TransientError, match="target_refused"):
            await lad.fetch("http://169.254.169.254/latest/meta-data")
        assert lad.attempts == []


class TestHopHeaders:
    """Caller credential headers belong to the entry origin (bug 26,
    iteration 20): a redirect hop to a different host OR port must not
    carry the Cookie/Authorization it was built with. The strip set mirrors
    reqwest's remove_sensitive_headers. Wire-level behavior is pinned by
    span controls 94/95."""

    HEADERS = {"Cookie": "span_clear=tok", "Authorization": "Bearer t",
               "User-Agent": "ua", "Accept": "text/html"}
    ENTRY = "http://127.0.0.1:8000/start"

    def test_same_origin_hop_keeps_everything(self):
        from searchio.net.ladder import _hop_headers
        got = _hop_headers(self.HEADERS, self.ENTRY,
                           "http://127.0.0.1:8000/echo")
        assert got == self.HEADERS

    def test_default_port_matches_explicit(self):
        from searchio.net.ladder import _hop_headers
        got = _hop_headers(self.HEADERS, "http://example.com/start",
                           "http://example.com:80/echo")
        assert got == self.HEADERS

    @pytest.mark.parametrize("hop", [
        "http://localhost:8000/echo",       # different host string
        "http://127.0.0.1:9000/echo",       # different port: another service
    ])
    def test_cross_origin_hop_strips_credentials(self, hop):
        from searchio.net.ladder import _hop_headers
        got = _hop_headers(self.HEADERS, self.ENTRY, hop)
        assert "Cookie" not in got and "Authorization" not in got
        assert got["User-Agent"] == "ua" and got["Accept"] == "text/html"

    def test_downgrade_strips_upgrade_keeps(self):
        from searchio.net.ladder import _hop_headers
        # https entry -> http hop: cleartext must never receive the
        # credential (httpx's follow has the same directional nuance).
        got = _hop_headers(self.HEADERS, "https://example.com/start",
                           "http://example.com/echo")
        assert "Cookie" not in got and "Authorization" not in got
        # http -> https upgrade on the same authority: no exposure added,
        # and reqwest's host|port rule treats it as same -- kept.
        got = _hop_headers(self.HEADERS, self.ENTRY,
                           "https://127.0.0.1:8000/echo")
        assert got == self.HEADERS

    def test_case_insensitive_strip(self):
        from searchio.net.ladder import _hop_headers
        headers = {"cookie": "a=b", "AUTHORIZATION": "Bearer t", "X-Ok": "1"}
        got = _hop_headers(headers, self.ENTRY, "http://localhost:8000/echo")
        assert got == {"X-Ok": "1"}


class TestHopCookies:
    """The tier-1 redirect cookie jar (bug 26, iteration 20): libcurl's
    engine scopes forwarded cookies by host, but the COLLECTION side must
    still validate Domain against the setting host (RFC 6265 s5.3 -- the
    flat dict accepted and redistributed Domain=localhost set from
    127.0.0.1) and honor Secure. Path/expiry stay unscoped by design.
    Wire-level behavior is pinned by span controls 85-93."""

    def test_host_only_cookie_rides_same_host_only(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        jar.set_cookie("sid=abc; Path=/", "example.com")
        assert jar.for_host("example.com", "http") == {"sid": "abc"}
        assert jar.for_host("www.example.com", "http") == {}  # host-only
        assert jar.for_host("other.com", "http") == {}

    def test_domain_attr_widens_to_subdomains(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        jar.set_cookie("sid=abc; Domain=example.com", "www.example.com")
        assert jar.for_host("example.com", "http") == {"sid": "abc"}
        assert jar.for_host("api.example.com", "http") == {"sid": "abc"}
        assert jar.for_host("badexample.com", "http") == {}  # dot-boundary

    def test_domain_the_setter_has_no_right_to_is_rejected(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        # 127.0.0.1 setting Domain=localhost: outside its scope, reject
        # outright -- the bug-26 bite shape (span ctl.cookie_bad_domain_*).
        jar.set_cookie("span_bd=bad; Domain=localhost; Path=/", "127.0.0.1")
        assert jar.for_host("127.0.0.1", "http") == {}
        assert jar.for_host("localhost", "http") == {}

    def test_secure_cookie_never_rides_cleartext(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        jar.set_cookie("sess=s; Secure; Path=/", "example.com")
        assert jar.for_host("example.com", "https") == {"sess": "s"}
        assert jar.for_host("example.com", "http") == {}

    def test_overwrite_by_name_and_scope(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        jar.set_cookie("a=1", "example.com")
        jar.set_cookie("a=2", "example.com")
        assert jar.for_host("example.com", "http") == {"a": "2"}

    def test_malformed_headers_are_dropped_quietly(self):
        from searchio.net.ladder import _HopCookies
        jar = _HopCookies()
        jar.set_cookie("", "example.com")
        jar.set_cookie("=novalue", "example.com")
        jar.set_cookie("no-equals-sign", "example.com")
        assert jar.for_host("example.com", "http") == {}


def _fake_getaddrinfo(monkeypatch, mapping):
    """Point socket.getaddrinfo at ``mapping`` ({name: [ip, ...]}).

    The fake normalizes a BYTES host key before lookup: anyio idna-encodes
    names to bytes ahead of getaddrinfo (the iteration-21 span fixture
    bit on exactly that), and _resolve_checked passes the name through
    verbatim. Unmapped names defer to the real resolver.
    """
    import socket as socket_mod
    real = socket_mod.getaddrinfo

    def fake(host, port, *args, **kwargs):
        key = (host.decode("ascii", "ignore")
               if isinstance(host, (bytes, bytearray)) else host)
        if key in mapping:
            out = []
            for ip in mapping[key]:
                if ":" in ip:
                    out.append((socket_mod.AF_INET6, socket_mod.SOCK_STREAM,
                                0, "", (ip, port, 0, 0)))
                else:
                    out.append((socket_mod.AF_INET, socket_mod.SOCK_STREAM,
                                0, "", (ip, port)))
            return out
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(socket_mod, "getaddrinfo", fake)


class TestRefusedIp:
    """The shared address-class predicate (bug 27, iteration 21):
    link-local and unspecified addresses are undialable wherever they
    appear -- URL literal or DNS answer -- and a mapped-v4 answer is
    unwrapped before the check."""

    @pytest.mark.parametrize("ip,token", [
        ("169.254.169.254", "link_local"),
        ("169.254.0.1", "link_local"),
        ("fe80::1", "link_local"),
        ("0.0.0.0", "unspecified"),
        ("::", "unspecified"),
        ("::ffff:169.254.169.254", "link_local"),   # mapped-v4 dodge
        ("::ffff:0.0.0.0", "unspecified"),
    ])
    def test_refused_addresses(self, ip, token):
        import ipaddress
        from searchio.net.ladder import _refused_ip
        got = _refused_ip(ipaddress.ip_address(ip))
        assert got is not None and got.startswith(token), f"{ip}: {got}"

    @pytest.mark.parametrize("ip", [
        "127.0.0.1",        # loopback: the control suite itself
        "192.168.1.1",      # RFC1918: legitimate intranet
        "10.0.0.5",
        "93.184.216.34",
        "::1",
        "2606:2800:220:1:248:1893:25c8:1946",
    ])
    def test_allowed_addresses(self, ip):
        import ipaddress
        from searchio.net.ladder import _refused_ip
        assert _refused_ip(ipaddress.ip_address(ip)) is None


class TestResolveChecked:
    """The connect-time resolution guard (bug 27): every getaddrinfo
    answer passes _refused_ip and ANY refused answer poisons the whole
    set -- a legit name never answers link-local, so one poisoned record
    is the rebinding shape and the whole dial is refused."""

    def test_clean_answers_pass_through(self, monkeypatch):
        from searchio.net.ladder import _resolve_checked
        _fake_getaddrinfo(monkeypatch, {
            "ok.unit.test": ["93.184.216.34", "127.0.0.1", "192.168.1.1"]})
        infos = _resolve_checked("ok.unit.test", 80)
        assert [i[4][0] for i in infos] == [
            "93.184.216.34", "127.0.0.1", "192.168.1.1"]

    @pytest.mark.parametrize("answers", [
        ["169.254.169.254"],                       # the rebinding shape
        ["fe80::1"],
        ["0.0.0.0"],
        ["::ffff:169.254.169.254"],                # mapped-v4 answer
        ["93.184.216.34", "169.254.169.254"],      # ANY poisons the set
    ])
    def test_refused_answer_poisons_the_set(self, monkeypatch, answers):
        from searchio.errors import TargetRefused
        from searchio.net.ladder import _resolve_checked
        _fake_getaddrinfo(monkeypatch, {"evil.unit.test": answers})
        with pytest.raises(TargetRefused, match="target_refused: dns:"):
            _resolve_checked("evil.unit.test", 80)

    def test_bytes_host_key_resolves(self, monkeypatch):
        # anyio's idna2008_resolve encodes the name to bytes before
        # getaddrinfo; the guard must see through that (suite lesson).
        from searchio.net.ladder import _resolve_checked
        _fake_getaddrinfo(monkeypatch, {"ok.unit.test": ["127.0.0.1"]})
        infos = _resolve_checked(b"ok.unit.test", 80)
        assert infos[0][4][0] == "127.0.0.1"

    def test_gaierror_propagates(self, monkeypatch):
        import socket as socket_mod
        from searchio.net.ladder import _resolve_checked

        def boom(host, port, *a, **k):
            raise socket_mod.gaierror(11001, "getaddrinfo failed")
        monkeypatch.setattr(socket_mod, "getaddrinfo", boom)
        with pytest.raises(socket_mod.gaierror):
            _resolve_checked("no-such.unit.test", 80)

    def test_zone_id_stripped_before_vetting(self, monkeypatch):
        from searchio.errors import TargetRefused
        from searchio.net.ladder import _resolve_checked
        import socket as socket_mod
        real = socket_mod.getaddrinfo

        def zoned(host, port, *a, **k):
            return [(socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 0, "",
                     ("fe80::1%eth0", port, 0, 0))]
        monkeypatch.setattr(socket_mod, "getaddrinfo", zoned)
        with pytest.raises(TargetRefused, match="link_local"):
            _resolve_checked("zoned.unit.test", 80)
        monkeypatch.setattr(socket_mod, "getaddrinfo", real)


class TestGuardedBackend:
    """The tier-0 guarded transport (bug 27): the pool's network backend
    resolves through _resolve_checked BEFORE any dial and dials only the
    checked answers -- a refused name dies as TargetRefused with zero
    connect attempts, an allowed fake name is served through the checked
    dial (real DNS would NXDOMAIN it), and TargetRefused crosses the
    httpx/httpcore boundary unmapped (not a ConnectError) so the tier
    loop keeps its terminal-refusal semantics."""

    async def test_refused_answer_raises_before_any_dial(self, monkeypatch):
        from searchio.errors import TargetRefused
        from searchio.net.ladder import _guarded_async_backend
        _fake_getaddrinfo(monkeypatch,
                          {"metadata.unit.test": ["169.254.169.254"]})
        dialed = []
        import anyio
        real_connect = anyio.connect_tcp

        async def sentry(*args, **kwargs):
            dialed.append((args, kwargs))
            return await real_connect(*args, **kwargs)
        monkeypatch.setattr(anyio, "connect_tcp", sentry)

        backend = _guarded_async_backend()
        with pytest.raises(TargetRefused, match="target_refused: dns:"):
            await backend.connect_tcp("metadata.unit.test", 80, timeout=3)
        assert dialed == []

    async def test_allowed_fake_name_served_via_checked_dial(self,
                                                             monkeypatch):
        import http.server
        import threading
        import httpx
        from searchio.net.ladder import _GuardedTransport

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"guarded-dial marker"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            _fake_getaddrinfo(monkeypatch,
                              {"allowed.unit.test": ["127.0.0.1"]})
            transport = _GuardedTransport()
            async with httpx.AsyncClient(transport=transport) as client:
                r = await client.get(f"http://allowed.unit.test:{port}/")
            assert r.status_code == 200
            assert "guarded-dial marker" in r.text
        finally:
            srv.shutdown()

    async def test_target_refused_survives_the_transport_boundary(
            self, monkeypatch):
        import httpx
        from searchio.errors import TargetRefused
        from searchio.net.ladder import _GuardedTransport
        _fake_getaddrinfo(monkeypatch,
                          {"metadata.unit.test": ["169.254.169.254"]})
        transport = _GuardedTransport()
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(TargetRefused, match="target_refused"):
                await client.get("http://metadata.unit.test/")

    def test_pool_backend_actually_swapped(self):
        # The private reach this whole mechanism stands on: an httpx or
        # httpcore upgrade that reshapes the pool must fail HERE, loudly,
        # rather than silently shipping an unguarded dial.
        import httpcore
        from httpcore._backends.anyio import AnyIOBackend
        from searchio.net.ladder import _GuardedTransport
        transport = _GuardedTransport()
        pool = getattr(transport, "_pool", None)
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert isinstance(pool._network_backend, AnyIOBackend)
        assert type(pool._network_backend).__name__ == "_GuardedAsyncBackend"


class TestTier1ResolvePin:
    """The tier-1 CURLOPT_RESOLVE pin (bug 27): libcurl resolves in C,
    out of the guard's reach, so the hop loop resolves in Python first
    and hands libcurl exactly the checked addresses. Literal hosts pin
    nothing (no resolution happens; the literal guard already vetted)."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "169.254.169.254",
                                      "::1", "fe80::1"])
    def test_ip_literal_pins_nothing(self, host):
        from searchio.net.ladder import _resolve_pin
        assert _resolve_pin(host, 80) is None

    def test_entries_cover_every_checked_answer(self, monkeypatch):
        from curl_cffi import CurlOpt
        from searchio.net.ladder import _resolve_pin
        _fake_getaddrinfo(monkeypatch,
                          {"multi.unit.test": ["93.184.216.34", "::1"]})
        pin = _resolve_pin("multi.unit.test", 443)
        assert pin == {CurlOpt.RESOLVE: [
            "multi.unit.test:443:93.184.216.34",
            "multi.unit.test:443:[::1]",  # v6 answers bracketed
        ]}

    def test_refused_answer_raises_before_any_pin(self, monkeypatch):
        from searchio.errors import TargetRefused
        from searchio.net.ladder import _resolve_pin
        _fake_getaddrinfo(monkeypatch,
                          {"metadata.unit.test": ["169.254.169.254"]})
        with pytest.raises(TargetRefused, match="target_refused: dns:"):
            _resolve_pin("metadata.unit.test", 80)


class TestMetaRefreshTarget:
    """The meta-refresh follow contract (iteration 25): _meta_refresh_target
    decides whether a fetched document asks for a client-side navigation.
    Squeeze/parked pages route through <meta http-equiv="refresh"> instead of
    a 302, and a search stack that ignores the tag serves the interstitial.
    The engine (se-net) implements the SAME contract; these units pin the
    Python half's parse shapes, the control rows pin both stacks over the
    wire."""

    F = "http://127.0.0.1:9/final/landing"   # the document's final URL

    def _call(self, text, status=200, ctype="text/html; charset=utf-8",
              final=None):
        from searchio.net import ladder as ladder_mod
        return ladder_mod._meta_refresh_target(
            status, ctype, text, final or self.F)

    def test_zero_delay_absolute_follows(self):
        assert self._call(
            '<meta http-equiv="refresh" content="0; url=http://x.test/t">'
        ) == "http://x.test/t"

    def test_fractional_delay_follows(self):
        assert self._call(
            '<meta http-equiv="refresh" content="0.5; url=http://x.test/t">'
        ) == "http://x.test/t"

    def test_boundary_delay_follows(self):
        assert self._call(
            '<meta http-equiv="refresh" content="5; url=http://x.test/t">'
        ) == "http://x.test/t"

    def test_slow_refresh_served_as_is(self):
        # Over the bound the interstitial IS the document -- a search engine
        # never sleeps for a slow redirect.
        assert self._call(
            '<meta http-equiv="refresh" content="5.1; url=http://x.test/t">'
        ) is None

    def test_reload_is_a_noop(self):
        # No url= means "reload this page" (the polling pattern) -- never a hop.
        assert self._call('<meta http-equiv="refresh" content="2">') is None

    def test_case_insensitive_tag_and_attrs(self):
        assert self._call(
            "<META HTTP-EQUIV=\"Refresh\" CONTENT=\"0; URL=http://x.test/t\">"
        ) == "http://x.test/t"

    def test_single_quoted_url(self):
        assert self._call(
            "<meta http-equiv='refresh' content=\"0; url='http://x.test/t'\">"
        ) == "http://x.test/t"

    def test_bare_url_without_prefix(self):
        # Browsers accept "0; http://x" without the url= prefix.
        assert self._call(
            '<meta http-equiv="refresh" content="0; http://x.test/t">'
        ) == "http://x.test/t"

    def test_relative_resolved_against_final_url(self):
        assert self._call(
            '<meta http-equiv="refresh" content="0;url=target">'
        ) == "http://127.0.0.1:9/final/target"

    def test_first_refresh_tag_decides(self):
        # Chrome's rule: the first refresh tag wins. A reload first means a
        # later instant hop never fires.
        assert self._call(
            '<meta http-equiv="refresh" content="5">'
            '<meta http-equiv="refresh" content="0; url=http://x.test/t">'
        ) is None

    def test_garbage_content_served_as_is(self):
        assert self._call(
            '<meta http-equiv="refresh" content="never">') is None

    def test_empty_url_is_a_reload(self):
        assert self._call(
            '<meta http-equiv="refresh" content="0; url="') is None

    def test_non_200_never_hops(self):
        assert self._call(
            '<meta http-equiv="refresh" content="0; url=http://x.test/t">',
            status=404) is None

    def test_non_html_never_hops(self):
        assert self._call(
            '{"m":"<meta http-equiv=\\"refresh\\" content=\\"0; url=http://x.test/t\\">"}',
            ctype="application/json") is None

    def test_beyond_the_head_window_never_hops(self):
        # The refresh contract lives in the head; a tag past the 64 KiB
        # window is body content, not navigation.
        pad = "<!--" + ("p" * 70000) + "-->"
        assert self._call(
            pad + '<meta http-equiv="refresh" content="0; url=http://x.test/t">'
        ) is None


class TestBlockSurvivesRescueFailure:
    """Bug 74 (iteration 45, caught LIVE by ctl.blocked_classification when
    the patchright challenge sidecar died mid-rescue): tier 2 answered 403,
    the auto-render rescue `continue`d BEFORE the block verdict was recorded,
    the rescue's sidecar raised SidecarUnavailable, and exhaustion had no
    block to report -- the caller got TransientError ("try again") for a
    site that had just refused the browser. The site's verdict outranks our
    own rescue infrastructure failing; only real content outranks the site.
    """

    WALL = (403, {"server": "cloudflare"}, "<html>Attention Required</html>")
    THIN = ("<html><body><p>small but real page</p><!--" + "x" * 600 + "--></body></html>")

    class Rescue(FakeLadder):
        def __init__(self, settings, responses, rescue):
            super().__init__(settings, responses)
            self.rescue = rescue
            self.rendered_attempts = 0

        async def _try_tier(self, tier, url, *, referer="", rendered=False):
            if tier == 2 and rendered:
                self.rendered_attempts += 1
                if isinstance(self.rescue, Exception):
                    raise self.rescue
                status, headers, body = self.rescue
                return status, headers, body, headers.get("content-type", HTML), url
            return await super()._try_tier(tier, url, referer=referer, rendered=rendered)

    @staticmethod
    def _auto(settings):
        settings.sidecar_engine = True
        settings.sidecar_challenge = "patchright"
        settings.sidecar_challenge_auto = True
        return settings

    async def test_rescue_infrastructure_failure_keeps_the_block(self, settings):
        from searchio.errors import SidecarUnavailable

        lad = self.Rescue(self._auto(settings), {2: self.WALL}, SidecarUnavailable("dead"))
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://walled3.example/x", force_tier=2, use_cache=False)
            msg = str(exc.value)
            assert "tier2:http_403" in msg and "tier2:auto_rendered" in msg, msg
            assert "tier2:sidecar_unavailable" in msg, msg
            assert lad.rendered_attempts == 1
        finally:
            await lad.close()

    async def test_rescued_thin_content_still_outranks_the_block(self, settings):
        lad = self.Rescue(self._auto(settings), {2: self.WALL},
                          (200, {"content-type": HTML}, self.THIN))
        try:
            res = await lad.fetch("https://walled4.example/x", force_tier=2, use_cache=False)
            assert "small but real page" in res.body
            assert any(e.startswith("accepted_thin") for e in res.escalations), res.escalations
        finally:
            await lad.close()


class TestSecondPassLadder:
    """Iteration 50: the second DeepSeek pass over the ladder (7 chunks,
    70 raw candidates). Three real ones, each red before its fix."""

    def test_public_suffix_domain_cookie_is_rejected(self):
        # Bug 91: _HopCookies validated Domain against the setting host with
        # a dot-boundary suffix test only, so a.com could set Domain=com and
        # the cookie then rode to evil.com on the next redirect hop (RFC
        # 6265 s5.3 rule 5 + the public-suffix rule).
        from searchio.net.ladder import _HopCookies

        jar = _HopCookies()
        jar.set_cookie("sid=x; Domain=com", "a.com")
        assert jar.for_host("a.com", "http") == {} and jar.for_host("evil.com", "http") == {}
        jar.set_cookie("sid=y; Domain=co.uk", "shop.co.uk")
        assert jar.for_host("other.co.uk", "http") == {}
        jar.set_cookie("sid=z; Domain=example.co.uk", "shop.example.co.uk")
        assert jar.for_host("www.example.co.uk", "http") == {"sid": "z"}
        assert jar.for_host("evil.co.uk", "http") == {}

    async def test_thin_fallback_serves_only_thin_pages(self, settings, monkeypatch):
        # Bug 92: the "thin-but-real" fallback kept ANY non-ok 2xx verdict
        # except empty_mount/js_required -- a login wall (auth_required_*),
        # a bot-marker page, a mojibake body -- and served it as content
        # when no tier did better. Only too_thin is thin.
        from searchio.net import ladder as ladder_mod
        from searchio.net.blocks import Verdict

        body = "<html><body>" + "x" * 5000 + "</body></html>"
        verdicts = iter([Verdict(ok=False, reason="auth_required_login", text_len=400)])

        def fake_classify(status, headers, b, **kw):
            try:
                return next(verdicts)
            except StopIteration:
                return Verdict(ok=False, reason="auth_required_login", text_len=400)

        monkeypatch.setattr(ladder_mod, "classify", fake_classify)
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, body)})
        try:
            with pytest.raises(TransientError):
                await lad.fetch("https://wall.example/x", use_cache=False)
        finally:
            await lad.close()
        monkeypatch.setattr(ladder_mod, "classify",
                            lambda status, headers, b, **kw: Verdict(ok=False, reason="too_thin:40", text_len=40))
        lad = FakeLadder(settings, {0: (200, {"content-type": HTML}, body)})
        try:
            res = await lad.fetch("https://thin.example/x", use_cache=False)
            assert any(e.startswith("accepted_thin") for e in res.escalations)
        finally:
            await lad.close()

    async def test_warmup_refetch_still_refuses_the_error_page(self, settings):
        # Bug 93: the nav-error check ran only on the FIRST tier-2 fetch;
        # after a refusal triggered the origin warm-up, the refetched result
        # was never checked -- a chrome-error:// landing shipped as a 200
        # (the bug-32 dino page, through the warm-up side door).
        from searchio.errors import NavErrorPage

        calls: list[str] = []

        class FakeSidecar:
            available = True

            async def fetch(self, url, **kw):
                calls.append("fetch")
                if any(c == "goto" for c in calls):
                    return {"ok": True, "status": 200, "html": "<html>err</html>",
                            "final_url": "chrome-error://chromewebdata/"}
                return {"ok": True, "status": 403, "html": "denied", "url": url}

            async def goto(self, url, **kw):
                calls.append("goto")
                return {"ok": True, "status": 200}

            async def cookies(self):
                return []

            async def user_agent(self):
                return ""

            async def close(self):
                return None

        lad = Ladder(settings, sidecar=FakeSidecar())
        try:
            with pytest.raises(NavErrorPage):
                await lad._tier2("https://hostile.example/deep/page")
            assert calls == ["fetch", "goto", "fetch"]
        finally:
            await lad.close()


class TestLadderBacklog51:
    """Iteration 51: four items from the second-pass dropped list, adjudicated
    by reading the sites. Each red before its fix."""

    async def test_pdf_refetch_policy_refusal_is_not_a_refetch_failure(self, settings):
        # Bug 94: _pdf_text caught Exception and stamped pdf_text:refetch_failed,
        # so a TargetRefused during the byte re-fetch (a redirect to a refused
        # target) became a retryable stamp -- the ladder then climbed two more
        # tiers re-drawing the same refusal and ended in TransientError
        # instead of the permanent policy refusal it was.
        from searchio.errors import TargetRefused

        class Refusing(FakeLadder):
            async def _fetch_bytes(self, url):
                raise TargetRefused("target_refused: link-local (http://169.254.169.254/x)")

        lad = Refusing(settings, {})
        try:
            with pytest.raises(TargetRefused):
                await lad._pdf_text("https://docs.example/paper.pdf")
        finally:
            await lad.close()

    async def test_redirect_target_is_checked_against_robots(self, settings):
        # Bug 95: robots was consulted for the REQUESTED url only; a redirect
        # to a disallowed path was fetched and served under enforce, and
        # unannotated under warn.
        from types import SimpleNamespace as NS

        from searchio.net.robots import RobotsInfo

        class Redirecting(FakeLadder):
            async def _try_tier(self, tier, url, *, referer="", rendered=False):
                self.attempts.append(tier)
                return 200, {"content-type": HTML}, GOOD, HTML, "https://site.example/private/doc"

        async def check(url):
            return RobotsInfo(allowed="/private" not in url, checked=True,
                              reason="" if "/private" not in url else "disallowed_by_robots")

        for policy, expect_block in (("enforce", True), ("warn", False)):
            lad = Redirecting(settings, {})
            lad.robots = NS(check=check, permits=lambda info, p=policy: info.allowed or p != "enforce",
                            policy=policy)
            try:
                if expect_block:
                    with pytest.raises(Blocked, match="robots"):
                        await lad.fetch("https://site.example/public", use_cache=False)
                else:
                    res = await lad.fetch("https://site.example/public", use_cache=False)
                    assert any("robots" in e and "redirect" in e for e in res.escalations), res.escalations
            finally:
                await lad.close()

    async def test_failed_rescue_outranks_a_lower_tier_thin_page(self, settings):
        # Bug 96 (bug 74's sibling): a tier-1 thin page was served when tier 2
        # answered 403 and the rescue's sidecar then died -- the rescued
        # block never entered last_block_tier, so the "block at or above the
        # thin tier is the later word" rule did not see it.
        from searchio.errors import SidecarUnavailable

        WALL = (403, {"server": "cloudflare"}, "<html>Attention Required</html>")
        THIN = "<html><body><p>small but real page</p><!--" + "x" * 600 + "--></body></html>"
        settings.sidecar_engine = True
        settings.sidecar_challenge = "patchright"
        settings.sidecar_challenge_auto = True

        class Rescue(FakeLadder):
            async def _try_tier(self, tier, url, *, referer="", rendered=False):
                if tier == 2 and rendered:
                    raise SidecarUnavailable("dead")
                return await super()._try_tier(tier, url, referer=referer, rendered=rendered)

        lad = Rescue(settings, {0: WALL, 1: (200, {"content-type": HTML}, THIN), 2: WALL})
        try:
            with pytest.raises(Blocked):
                await lad.fetch("https://walled5.example/x", use_cache=False)
        finally:
            await lad.close()

    def test_domain_key_keeps_www_com(self):
        # Rider: domain_of stripped "www." unconditionally, so www.com keyed
        # its profile, limiter and clearance under "com".
        from searchio.net.ladder import domain_of

        assert domain_of("https://www.com/x") == "www.com"
        assert domain_of("https://www.example.com/") == "example.com"
        assert domain_of("https://WWW.Example.COM/") == "example.com"


class TestSidecarCallHonesty:
    """Iteration 52 (sidecar second pass): the RPC seam's own honesty."""

    @staticmethod
    def _stub(rpc_body: bytes, content_type: str = "text/plain"):
        import http.server
        import json as jsonlib
        import threading

        class _H(http.server.BaseHTTPRequestHandler):
            gets = 0

            def _send(self, body: bytes, ctype: str):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                _H.gets += 1
                self._send(jsonlib.dumps({"ok": True}).encode(), "application/json")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    method = jsonlib.loads(raw).get("method", "")
                except Exception:
                    method = ""
                if "health" in method:
                    self._send(jsonlib.dumps({"jsonrpc": "2.0", "id": 0,
                                              "result": {"ok": True, "engine": "patchright"}}).encode(),
                               "application/json")
                else:
                    self._send(rpc_body, content_type)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
        srv.handler = _H
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{srv.server_address[1]}", srv

    async def test_non_json_200_is_sidecar_unavailable_not_a_decode_crash(self):
        # Bug 97: call() parsed r.json() outside its try, so a proxy/gateway
        # page or a half-written body on a 200 escaped as a raw
        # JSONDecodeError -- not SidecarUnavailable, which is the only class
        # the ladder's tier loop knows how to classify -- and took the fetch
        # down with an unclassified exception.
        from searchio.errors import SidecarUnavailable
        from searchio.net.sidecar import SidecarClient

        base, srv = self._stub(b"<html>502 bad gateway from the proxy</html>", "text/html")
        sc = SidecarClient(url=base, autostart=False)
        try:
            with pytest.raises(SidecarUnavailable, match="malformed"):
                await sc.fetch("https://a.example/")
        finally:
            await sc.close()
            srv.shutdown()

    async def test_concurrent_first_calls_share_one_http_client(self, monkeypatch):
        # Rider: the httpx client was created lazily with a bare `is None`
        # check, so two concurrent first verbs built two clients and one
        # leaked its connection pool for the life of the process.
        import httpx as _httpx

        from searchio.net import sidecar as sidecar_mod
        from searchio.net.sidecar import SidecarClient

        made = []
        real = _httpx.AsyncClient

        class Counting(real):
            def __init__(self, *a, **kw):
                made.append(1)
                super().__init__(*a, **kw)

        monkeypatch.setattr(sidecar_mod.httpx, "AsyncClient", Counting)
        base, srv = self._stub(b'{"jsonrpc":"2.0","id":0,"result":{"ok":true,"html":"<p>x</p>"}}',
                               "application/json")
        sc = SidecarClient(url=base, autostart=False)
        try:
            await sc.ensure()
            made.clear()
            srv.handler.gets = 0
            await asyncio.gather(sc.fetch("https://a.example/"), sc.fetch("https://b.example/"))
            assert len(made) == 1, len(made)
            # Bug 98: every verb used to re-run GET /healthz with a fresh client
            # although the identity handshake was already remembered.
            assert srv.handler.gets == 0, srv.handler.gets
        finally:
            await sc.close()
            srv.shutdown()

    async def test_child_stderr_is_drained_so_a_chatty_sidecar_never_blocks(self):
        # Bug 99: the spawned sidecar's stderr was a PIPE nobody read after
        # boot. The script writes its DEBUG lines to raw stderr and the
        # Chromium tree inherits the pipe; once the OS buffer filled, the
        # next write BLOCKED and the whole sidecar wedged mid-run (the shape
        # behind bug 74's mid-rescue ReadError). First the mechanism, then
        # the fix.
        import subprocess
        import sys as _sys

        chatty = [_sys.executable, "-c",
                  "import sys; sys.stderr.write('log line ' * 60000); sys.stderr.flush()"]
        stuck = subprocess.Popen(chatty, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                stuck.wait(timeout=2)  # blocked on the full pipe
        finally:
            stuck.kill()
            stuck.wait()

        from searchio.net.sidecar import _StderrTail

        proc = subprocess.Popen(chatty, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tail = _StderrTail(proc)
        assert proc.wait(timeout=15) == 0
        tail._thread.join(timeout=5)
        assert "log line" in tail.tail() and len(tail.lines) >= 1

    def test_child_env_drops_secrets(self):
        # Bug 100: the spawned sidecar inherited dict(os.environ) whole --
        # LLM keys, search-API keys, tokens, a proxy password -- into a
        # browser process that logs its environment on some paths.
        from searchio.net.sidecar import child_env

        env = child_env({"ANTHROPIC_API_KEY": "a", "DEEPSEEK_API_KEY": "d", "SEARCHIO_SERPAPI_KEY": "s",
                         "GITHUB_TOKEN": "g", "AWS_SECRET_ACCESS_KEY": "w", "DB_PASSWORD": "p",
                         "PATH": "keep", "SWARM_SIDECAR_HTTP_TOKEN": "keep", "SE_SERVE_SESSION": "keep",
                         "SEARCHIO_STATE_DIR": "keep"})
        assert set(env) == {"PATH", "SWARM_SIDECAR_HTTP_TOKEN", "SE_SERVE_SESSION", "SEARCHIO_STATE_DIR"}, env

    async def test_read_html_verb_failure_raises_not_empty(self):
        # Bug 101: read_html turned {"ok": false, "error": "no such tab"} into
        # "" -- which the Marketplace provider parsed as zero listings and
        # then reasoned about as an empty market or a session gate.
        from searchio.errors import SidecarVerbError
        from searchio.net.sidecar import SidecarClient

        base, srv = self._stub(b'{"jsonrpc":"2.0","id":0,"result":{"ok":false,"error":"no such tab"}}',
                               "application/json")
        sc = SidecarClient(url=base, autostart=False)
        try:
            with pytest.raises(SidecarVerbError, match="no such tab"):
                await sc.read_html(tab_id="t")
        finally:
            await sc.close()
            srv.shutdown()

    def test_reap_belt_spares_a_stranger_on_a_recycled_port(self, monkeypatch):
        # Bug 102: close()'s port-verified belt killed WHOEVER listened on our
        # port. A span run's shutdown ran while the next run booted its engine
        # on the recycled port -- and killed it (the controls replay of
        # iteration 52 died on its first tier-2 fetch, ConnectError). The belt
        # may only reap PIDs from our own pre-kill tree.
        import subprocess as _sp

        from searchio.net import sidecar as sidecar_mod

        killed = []
        netstat = ("  TCP    127.0.0.1:8899     0.0.0.0:0     LISTENING       4242\n"
                   "  TCP    127.0.0.1:8899     0.0.0.0:0     LISTENING       7777\n")

        def fake_run(cmd, **kw):
            if cmd[0] == "netstat":
                return _sp.CompletedProcess(cmd, 0, stdout=netstat, stderr="")
            if cmd[0] == "taskkill":
                killed.append(cmd[2])
                return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")
            raise AssertionError(cmd)

        monkeypatch.setattr(sidecar_mod.sys, "platform", "win32")
        monkeypatch.setattr(sidecar_mod.subprocess, "run", fake_run)
        sidecar_mod._kill_listener_on_port(8899, allowed_pids={4242, 100})
        assert killed == ["4242"], killed  # 7777 is the stranger
        killed.clear()
        sidecar_mod._kill_listener_on_port(8899, None)  # no tree known: old behaviour
        assert sorted(killed) == ["4242", "7777"]

    def test_tree_pids_walks_a_cim_snapshot(self, monkeypatch):
        import subprocess as _sp

        from searchio.net import sidecar as sidecar_mod

        snapshot = "10 1\n20 10\n30 20\n40 10\n99 5\n"

        def fake_run(cmd, **kw):
            assert cmd[0] == "powershell"
            return _sp.CompletedProcess(cmd, 0, stdout=snapshot, stderr="")

        monkeypatch.setattr(sidecar_mod.sys, "platform", "win32")
        monkeypatch.setattr(sidecar_mod.subprocess, "run", fake_run)
        assert sidecar_mod._tree_pids(10) == {10, 20, 30, 40}


class TestLadderOwnsOnlyItsOwnSidecar:
    async def test_close_leaves_an_injected_sidecar_alone(self, settings):
        # Bug 103 (caught live, twice): Ladder.close() closed the sidecar it
        # was HANDED. The span suite shares one booted engine across 136
        # control Ladders; every row's close() killed it and the next row's
        # autostart quietly respawned it -- 136 engine boots per run, masked
        # until bug 98's verify-once fast path stopped the respawn and the
        # replay died on row one with ConnectError. A client the caller
        # built is the caller's to close.
        closed = []

        class Injected:
            available = True
            binary = None

            async def close(self):
                closed.append("injected")

        lad = Ladder(settings, sidecar=Injected())
        await lad.close()
        assert closed == []
        # ...while a sidecar the Ladder built itself is still closed.
        settings.sidecar_autostart = False
        own = Ladder(settings)
        own_closed = []

        async def mark():
            own_closed.append("own")
        own.sidecar.close = mark  # type: ignore[method-assign]
        await own.close()
        assert own_closed == ["own"]

