"""Circuit-open providers are named, not vanished (iteration 72, from the
--fast leg: api.search_knobs came back 200 with no results and failed={}
because the only web providers were cooling down -- an HTTP caller could
not tell that from an empty web). Red first."""

from __future__ import annotations

import time

import pytest

from searchio.config import Settings
from searchio.models import Capability, Doc, Query
from searchio.providers.registry import Registry
from searchio.router import Router

from tests.test_ladder_router import StubProvider


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False)


def _open(p):
    p.health.opened_at = time.monotonic()
    p.health.consecutive_failures = 3
    p.health.last_error = "HTTP 429"
    return p


def _router(settings, providers):
    return Router(object(), settings=settings, registry=Registry(providers))


class TestTheRouterNamesWhatItSkipped:
    async def test_everyone_cooling_down_is_not_no_provider_supports_intent(self, settings):
        # Bug 164: select() silently dropped circuit-open providers; with all
        # of them open the result said "no provider supports intent web" --
        # they do, they are cooling down -- and with SOME of them open the
        # result carried no trace at all.
        ddg = _open(StubProvider("duckduckgo", [Capability.WEB]))
        bing = _open(StubProvider("bing", [Capability.WEB]))
        res = await _router(settings, [ddg, bing]).search(Query(text="x", intent="web", k=5))
        assert res.docs == [] and res.used == []
        assert set(res.skipped) == {"duckduckgo", "bing"}, res.skipped
        assert all("circuit" in v and "429" in v for v in res.skipped.values()), res.skipped
        assert "router" not in res.failed, res.failed

    async def test_a_partly_cooling_fanout_still_names_the_skipped(self, settings):
        good = StubProvider("duckduckgo", [Capability.WEB],
                            [Doc(url="https://a.com/1", title="A", source="duckduckgo")])
        bing = _open(StubProvider("bing", [Capability.WEB]))
        res = await _router(settings, [good, bing]).search(Query(text="x", intent="web", k=5))
        assert res.docs and res.used == ["duckduckgo"]
        assert list(res.skipped) == ["bing"], res.skipped
        assert res.failed == {}, "cooling down is not a fresh failure"

    async def test_a_truly_unsupported_intent_still_says_so(self, settings):
        ddg = StubProvider("duckduckgo", [Capability.WEB])
        res = await _router(settings, [ddg]).search(Query(text="x", intent="shopping", k=5))
        assert res.skipped == {}
        assert "router" in res.failed and "shopping" in res.failed["router"]


class TestTheApiCarriesTheSkipped:
    async def test_nobody_answered_because_everyone_is_cooling_is_a_503(self, monkeypatch):
        import httpx

        import searchio.server as server
        from tests.test_server_api import FakeEng

        eng = FakeEng()
        eng.docs, eng.used = [], []
        skipped = {"bing": "circuit open: 3 consecutive failures, last 'HTTP 429', cooling for 118s"}

        async def search(text, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(docs=[], used=[], failed={}, skipped=dict(skipped), elapsed_ms=1)
        eng.search = search
        monkeypatch.setattr(server, "_engine", eng)
        transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/search", params={"q": "rust async"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["providers_skipped"] == skipped, r.text

    async def test_a_served_search_lists_the_skipped_alongside_the_used(self, monkeypatch):
        import httpx

        import searchio.server as server
        from tests.test_server_api import FakeEng

        eng = FakeEng()

        async def search(text, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(docs=list(eng.docs), used=["duckduckgo"], failed={},
                                   skipped={"bing": "circuit open"}, elapsed_ms=1)
        eng.search = search
        monkeypatch.setattr(server, "_engine", eng)
        transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/search", params={"q": "rust async"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["providers_skipped"] == {"bing": "circuit open"}, body
        assert body["results"]


class TestAnIntentOnlyBrowsersServeSaysSo:
    async def test_local_rides_the_general_web_backstop(self, settings):
        # Every provider declaring "local" is browser-backed and gated by
        # allow_expensive; the generalist backstop is what actually serves
        # it. Pinned so the message test below stays about the message.
        pricey = StubProvider("sidecar_search", [Capability.WEB, Capability.LOCAL])
        pricey.cost_per_1k = 5.0
        cheap = StubProvider("duckduckgo", [Capability.WEB])
        r = _router(settings, [cheap, pricey])
        assert [p.name for p in r.select(Query(text="pizza", intent="local"))] == ["duckduckgo"]

    async def test_a_cooling_backstop_is_named_and_the_gated_providers_are_named(self, settings):
        # Bug 165 (run 73 local.food): with the generalists cooling, a local
        # query answered "no provider supports intent local" -- three
        # providers support it, this call just did not allow the browser
        # ones, and the backstop that would have served it was cooling.
        pricey = StubProvider("sidecar_search", [Capability.WEB, Capability.LOCAL])
        pricey.cost_per_1k = 5.0
        cheap = _open(StubProvider("duckduckgo", [Capability.WEB]))
        r = _router(settings, [cheap, pricey])
        res = await r.search(Query(text="pizza", intent="local", k=5))
        assert res.used == [] and res.docs == []
        assert list(res.skipped) == ["duckduckgo"], res.skipped
        msg = res.failed.get("router", "")
        assert "sidecar_search" in msg and "allow_expensive" in msg, res.failed
        assert "no provider supports" not in msg, res.failed

    async def test_cheap_providers_for_the_intent_still_keep_the_browser_out(self, settings):
        pricey = StubProvider("sidecar_search", [Capability.WEB, Capability.LOCAL])
        pricey.cost_per_1k = 5.0
        cheap = StubProvider("duckduckgo", [Capability.WEB])
        r = _router(settings, [cheap, pricey])
        assert [p.name for p in r.select(Query(text="x", intent="web"))] == ["duckduckgo"]

    async def test_cooling_cheap_providers_do_not_admit_the_browser(self, settings):
        # A cheap provider that merely sits circuit-open must not turn every
        # web query into a browser SERP.
        pricey = StubProvider("sidecar_search", [Capability.WEB, Capability.LOCAL])
        pricey.cost_per_1k = 5.0
        cheap = _open(StubProvider("duckduckgo", [Capability.WEB]))
        r = _router(settings, [cheap, pricey])
        assert r.select(Query(text="x", intent="web")) == []
        res = await r.search(Query(text="x", intent="web", k=5))
        assert list(res.skipped) == ["duckduckgo"] and res.used == []
        assert "sidecar_search" in res.failed.get("router", ""), res.failed
