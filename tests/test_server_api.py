"""The HTTP API as an agent programs against it.

Iteration 46 of the span suite (DeepSeek design review of server.py +
swarm/orchestrator.py). Until this file existed nothing exercised the HTTP
surface at all. Every test here bit RED before its fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import searchio.server as server
from searchio.errors import Blocked, ProviderError, TargetRefused, TransientError
from searchio.models import Doc, FetchResult, Item, Price


class FakeEng:
    def __init__(self):
        self.search_kw: list[dict] = []
        self.docs = [Doc(url="https://a.com/1", title="A", snippet="s", source="stub")]
        self.used = ["stub"]
        self.failed: dict = {}
        self.items_exc: Exception | None = None
        self.fetch_exc: Exception | None = None
        self.ladder = SimpleNamespace(fetch=self._fetch,
                                      profiles=SimpleNamespace(all=lambda: []))
        self.router = SimpleNamespace(health_report=lambda: {})

    async def search(self, text, **kw):
        self.search_kw.append({"text": text, **kw})
        return SimpleNamespace(docs=list(self.docs), used=list(self.used),
                               failed=dict(self.failed), elapsed_ms=1)

    async def find_items(self, text, **kw):
        if self.items_exc:
            raise self.items_exc
        return [Item(title="T", url="https://s.com/p", price=Price(amount=1.0, currency="USD"))]

    async def find_local_items(self, text, **kw):
        if self.items_exc:
            raise self.items_exc
        return []

    async def _fetch(self, url, **kw):
        if self.fetch_exc:
            raise self.fetch_exc
        return FetchResult(url=url, final_url=url, status=200, body="<html><title>T</title><p>hi</p></html>",
                           content_type="text/html", tier=0, via="http", elapsed_ms=1)

    async def research(self, question, *, progress=None):
        raise ProviderError("planner", "unusable plan")

    def stats(self):
        return {"ok": True}


@pytest.fixture
async def api(monkeypatch):
    eng = FakeEng()
    monkeypatch.setattr(server, "_engine", eng)
    transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        yield c, eng


class TestErrorStatusesAreStable:
    """Bug 75: /items and /marketplace let SearchioError escape as a bare
    500; /read mapped an SSRF/policy refusal (TargetRefused, the caller's
    fault) to 502 Bad Gateway; /search answered 200 + [] when EVERY provider
    failed. An agent programs against status codes: a refusal, an upstream
    failure and an honest empty must be distinguishable.
    """

    async def test_items_provider_error_is_a_502_with_detail(self, api):
        c, eng = api
        eng.items_exc = ProviderError("amazon", "all product pages failed")
        r = await c.get("/items", params={"q": "x"})
        assert r.status_code == 502, r.text
        assert "all product pages failed" in r.json()["detail"]
        r = await c.get("/marketplace", params={"q": "x"})
        assert r.status_code == 502, r.text
        eng.items_exc = Blocked("http_403", vendor="datadome")
        r = await c.get("/items", params={"q": "x"})
        assert r.status_code == 451 and "datadome" in r.json()["detail"]

    async def test_read_target_refusal_is_a_400(self, api):
        c, eng = api
        eng.fetch_exc = TargetRefused("target_refused: private address 169.254.169.254")
        r = await c.get("/read", params={"url": "http://169.254.169.254/latest"})
        assert r.status_code == 400, r.text
        assert "target_refused" in r.json()["detail"]
        eng.fetch_exc = TransientError("timeout")
        r = await c.get("/read", params={"url": "https://slow.example/"})
        assert r.status_code == 502

    async def test_search_with_every_provider_failed_is_a_503(self, api):
        c, eng = api
        eng.docs, eng.used, eng.failed = [], [], {"bing": "ProviderError: degraded", "ddg": "circuit open"}
        r = await c.get("/search", params={"q": "x"})
        assert r.status_code == 503, r.text
        d = r.json()["detail"]
        assert d["providers_failed"]["bing"].startswith("ProviderError") and "all" in d["error"]
        # A provider that answered with nothing is an honest empty, still 200.
        eng.used = ["bing"]
        r = await c.get("/search", params={"q": "x"})
        assert r.status_code == 200 and r.json()["results"] == []


class TestRequestContract:
    async def test_fresh_reaches_the_engine(self, api):
        # Bug 76: `fresh` was documented as "bypass the cache" and then never
        # passed anywhere -- an agent asking for fresh results got the
        # ladder's hour-old SERP cache with a straight face.
        c, eng = api
        r = await c.get("/search", params={"q": "x", "fresh": "true"})
        assert r.status_code == 200
        assert eng.search_kw[-1].get("fresh") is True, eng.search_kw

    async def test_string_fields_are_bounded(self, api):
        c, eng = api
        assert (await c.get("/search", params={"q": "x" * 5000})).status_code == 422
        assert (await c.get("/read", params={"url": "https://a.com/" + "p" * 5000})).status_code == 422
        assert (await c.get("/search", params={"q": "x", "intent": "y" * 100})).status_code == 422
        assert (await c.get("/items", params={"q": "x", "domains": "d," * 3000})).status_code == 422
        assert (await c.get("/search", params={"q": "x", "k": 51})).status_code == 422

    async def test_unknown_intent_is_a_422_not_a_500(self, api):
        c, eng = api

        async def boom(text, **kw):
            from searchio.models import Query
            Query(text=text, intent=kw.get("intent", "web"))
        eng.search = boom
        r = await c.get("/search", params={"q": "x", "intent": "bogus"})
        assert r.status_code == 422, r.text


class TestResearchErrors:
    async def test_research_planner_failure_is_a_502(self, api):
        c, eng = api
        r = await c.post("/research", json={"question": "what is x?"})
        assert r.status_code == 502 and "unusable plan" in r.json()["detail"]

    async def test_stream_terminates_with_an_error_event(self, api):
        c, eng = api
        async with c.stream("POST", "/research/stream", json={"question": "what is x?"}) as r:
            body = (await r.aread()).decode()
        assert r.status_code == 200
        assert "event: error" in body and "unusable plan" in body


class TestSearchKnobsReachTheApi:
    """Iteration 66 probe of the agent surface with what agents actually
    send. /search had no freshness or locale parameter at all -- both exist
    on the engine and on the agent tool -- so an API caller could not ask
    for recent results or a market, and FastAPI dropped the unknown
    parameter without a word (bug 144). A whitespace-only query passed the
    length check and came back as a 503 blaming the providers; a
    capitalized intent was a 422 where folding is the obvious reading.
    """

    async def test_freshness_and_locale_reach_the_engine(self, api):
        c, eng = api
        r = await c.get("/search", params={"q": "x", "freshness": "week", "locale": "de-DE"})
        assert r.status_code == 200, r.text
        assert eng.search_kw[-1].get("freshness") == "week"
        assert eng.search_kw[-1].get("locale") == "de-DE"

    async def test_unknown_freshness_is_a_422(self, api):
        c, eng = api
        r = await c.get("/search", params={"q": "x", "freshness": "yesterday"})
        assert r.status_code == 422, r.text

    async def test_blank_query_is_the_callers_fault(self, api):
        c, eng = api
        r = await c.get("/search", params={"q": "   "})
        assert r.status_code == 422, r.text
        assert not eng.search_kw

    async def test_intent_case_is_folded(self, api):
        c, eng = api
        r = await c.get("/search", params={"q": "x", "intent": "News"})
        assert r.status_code == 200, r.text
        assert eng.search_kw[-1].get("intent") == "news"


class TestItemEndpointsNameCooling:
    """Bug 183 (the HTTP twin of bug 182): /items and /marketplace returned
    200 + {"count": 0, "items": []} when every shopping/local provider sat
    circuit-open, hiding the cooldown -- while /search answers 503 +
    providers_skipped (bug 164). They now answer 503 + providers_cooling when
    the empty is a cooldown; a genuine empty (nobody cooling) stays 200."""

    async def test_items_503s_when_all_shopping_providers_cool(self, api):
        c, eng = api

        async def empty(*a, **k):
            return []
        eng.find_items = empty
        eng.router._cooling = lambda intent: (
            {"bing": "circuit open: 3 consecutive failures, cooling for 90s"}
            if intent == "shopping" else {})
        r = await c.get("/items", params={"q": "sony headphones"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["providers_cooling"], r.text

    async def test_items_200_empty_when_nobody_is_cooling(self, api):
        c, eng = api

        async def empty(*a, **k):
            return []
        eng.find_items = empty
        eng.router._cooling = lambda intent: {}
        r = await c.get("/items", params={"q": "zxqv nonexistent widget"})
        assert r.status_code == 200 and r.json()["count"] == 0, r.text

    async def test_marketplace_503s_when_local_providers_cool(self, api):
        c, eng = api

        async def empty(*a, **k):
            return []
        eng.find_local_items = empty
        eng.router._cooling = lambda intent: (
            {"facebook_marketplace": "circuit open"} if intent == "local" else {})
        r = await c.get("/marketplace", params={"q": "kayak"})
        assert r.status_code == 503, r.text
        assert r.json()["detail"]["providers_cooling"], r.text
