"""Backend-aware SidecarListings (firing 41) — the audit slice.

The firing-41 parity audit found the one searchio caller of a
patchright-only verb: SidecarListings drove the sidecar script's composed
``research_listings`` pipeline, which the Rust engine deliberately does not
serve (SERP scraping is not protocol). The provider now branches on
``SidecarClient.backend_kind()``: on the engine it realizes the same
shopping intent with the engine's own primitives (router discovery +
rendered-page JSON-LD/meta extraction via MarketplaceItems), on patchright
it keeps the full research pipeline. These tests pin both branches and the
health-probe discrimination, including one wire test against the real
se-serve binary.

Skipped automatically when the engine binary has not been built yet (wire
test only).
"""

from __future__ import annotations

import asyncio
import http.server
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from searchio.models import Doc, Query
from searchio.net.sidecar import SidecarClient
from searchio.providers.marketplace import SidecarListings

ENGINE_BIN = (
    Path(__file__).parents[2]
    / "searchio-engine"
    / "target"
    / "debug"
    / "se-serve.exe"
)

_PRODUCT_HTML = """
<html><head><title>Sony WH-1000XM5</title></head><body>
<script type="application/ld+json">
{"@context": "https://schema.org", "@type": "Product",
 "name": "Sony WH-1000XM5 Wireless Headphones",
 "brand": {"@type": "Brand", "name": "Sony"},
 "offers": {"@type": "Offer", "price": "248.00", "priceCurrency": "USD",
            "availability": "https://schema.org/InStock"}}
</script>
<h1>Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones</h1>
<p>The WH-1000XM5 headphones rewrite the rules for distraction-free listening.
Two processors control eight microphones for unprecedented noise cancellation,
while a specially designed driver unit delivers gorgeously detailed sound.
With up to 30 hours of battery life, crystal-clear hands-free calling, and
a lightweight comfortable fit, these are the reference for wireless over-ear
headphones. This listing is the black model in sealed retail packaging.</p>
</body></html>
"""


# ── backend_kind probe ───────────────────────────────────────────────────────


async def test_backend_kind_binary_is_engine_without_network():
    client = SidecarClient(binary=Path("se-serve.exe"), autostart=False)
    assert await client.backend_kind() == "engine"


async def test_backend_kind_health_probe_discriminates_and_caches(monkeypatch):
    client = SidecarClient(url="http://127.0.0.1:1", autostart=False)
    calls = 0

    async def fake_call(verb, params=None, **kw):
        nonlocal calls
        calls += 1
        assert verb == "health"
        return {"ok": True, "engine": "searchio-engine"}

    monkeypatch.setattr(client, "call", fake_call)
    assert await client.backend_kind() == "engine"
    assert await client.backend_kind() == "engine"
    assert calls == 1, "the probe answer is cached, not re-asked per call"


async def test_backend_kind_names_patchright_and_survives_failure(monkeypatch):
    client = SidecarClient(url="http://127.0.0.1:1", autostart=False)

    async def patchright_health(verb, params=None, **kw):
        return {"ok": True, "engine": "patchright"}

    monkeypatch.setattr(client, "call", patchright_health)
    assert await client.backend_kind() == "patchright"

    failing = SidecarClient(url="http://127.0.0.1:1", autostart=False)

    async def dead_health(verb, params=None, **kw):
        raise ConnectionError("nobody listening")

    monkeypatch.setattr(failing, "call", dead_health)
    assert await failing.backend_kind() == ""


# ── provider branches on the backend ─────────────────────────────────────────


class _FakeSidecar:
    """available + backend_kind, records research_listings calls."""

    def __init__(self, kind: str) -> None:
        self.available = True
        self._kind = kind
        self.research_calls: list[dict] = []

    async def backend_kind(self) -> str:
        return self._kind

    async def research_listings(self, query: str, **kw):
        self.research_calls.append({"query": query, **kw})
        return {
            "ok": True,
            "listings": [
                {
                    "url": "https://www.ebay.com/itm/276123456789",
                    "title": "Sony WH-1000XM5",
                    "price": "$248.00",
                    "seller": "audio_deals",
                    "attributes": {"brand": "Sony", "condition": "new"},
                    "verified": True,
                }
            ],
        }


class _FakeLadder:
    """fetch() serves the JSON-LD product page for any URL."""

    def __init__(self, sidecar, html: str) -> None:
        self.sidecar = sidecar
        self.html = html

    async def fetch(self, url: str):
        return SimpleNamespace(body=self.html, final_url=url, tier=2, status=200)


def _ctx(ladder, docs):
    async def _search(q, allow_expensive=False):
        return SimpleNamespace(docs=docs, failed=[])

    return SimpleNamespace(
        ladder=ladder,
        settings=SimpleNamespace(),
        router=SimpleNamespace(search=_search),
    )


async def test_engine_backend_serves_shopping_via_marketplace_flow():
    sc = _FakeSidecar("engine")
    lad = _FakeLadder(sc, _PRODUCT_HTML)
    docs = [
        Doc(
            url="http://127.0.0.1:9/dp/B09XS7JWHH",
            title="Sony WH-1000XM5",
            snippet="$248",
            source="stub",
        )
    ]
    q = Query(text="sony wh-1000xm5", intent="shopping", k=5, domains=["127.0.0.1"])
    items = await SidecarListings().find_items(q, _ctx(lad, docs))
    assert sc.research_calls == [], "the engine path never asks for the patchright pipeline"
    assert len(items) == 1
    assert items[0].price.amount == 248.0
    assert "Sony" in items[0].title
    assert items[0].meta["backend"] == "engine"


async def test_patchright_backend_keeps_the_research_pipeline():
    sc = _FakeSidecar("patchright")
    lad = _FakeLadder(sc, _PRODUCT_HTML)
    q = Query(text="sony wh-1000xm5", intent="shopping", k=5)
    items = await SidecarListings().find_items(q, _ctx(lad, []))
    assert len(sc.research_calls) == 1
    assert sc.research_calls[0]["query"] == "sony wh-1000xm5"
    assert sc.research_calls[0]["sources"] == SidecarListings.default_sources()
    assert len(items) == 1
    assert items[0].url == "https://www.ebay.com/itm/276123456789"
    assert items[0].brand == "Sony"
    assert items[0].verified is True
    assert "backend" not in items[0].meta


# ── wire test against the real se-serve ──────────────────────────────────────


async def test_engine_backend_listings_end_to_end_over_the_wire(tmp_path):
    """SidecarListings on a real engine sidecar: the provider must answer
    shopping intent through engine primitives instead of failing on the
    unknown research_listings verb."""
    if not ENGINE_BIN.exists():
        pytest.skip(f"se-serve binary not built: {ENGINE_BIN}")

    from searchio.config import Settings
    from searchio.net.ladder import Ladder

    seen: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            body = _PRODUCT_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    settings = Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,
        per_domain_burst=1000,
    )
    client = SidecarClient(
        binary=ENGINE_BIN,
        port=_free_port(),
        boot_timeout_s=30.0,
        request_timeout_s=30.0,
    )
    lad = Ladder(settings, sidecar=client)
    docs = [
        Doc(url=f"{base}/dp/B09XS7JWHH", title="Sony WH-1000XM5", snippet="$248", source="stub")
    ]
    try:
        items = await SidecarListings().find_items(
            Query(text="sony wh-1000xm5", intent="shopping", k=5, domains=["127.0.0.1"]),
            _ctx(lad, docs),
        )
        assert len(items) == 1, f"engine-backed listings came back empty: {seen!r}"
        assert items[0].price.amount == 248.0
        assert items[0].url == f"{base}/dp/B09XS7JWHH"
        assert items[0].meta["backend"] == "engine"
        assert seen, "the ladder must actually fetch the product page"
    finally:
        await lad.close()
        await client.close()  # the test built it; the Ladder no longer closes injected clients (bug 103)
    assert client._proc is None, "ladder close() must reap the engine it started"


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestKillTree:
    """close() must reap the WHOLE sidecar tree, not just the wrapper.

    Popen.terminate reaches only the direct child; the patchright sidecar
    is python.exe -> node cli.js -> chrome.exe xN, so terminating the
    wrapper leaked headed Chromiums that kept accumulating tabs across
    runs (and the next run attached to the zombie on the shared challenge
    port and never owned it). Windows gets taskkill /T; POSIX keeps
    terminate-then-kill on the direct child.
    """

    def test_windows_tree_kills_via_taskkill(self, monkeypatch):
        from searchio.net import sidecar as sidecar_mod

        calls = []
        monkeypatch.setattr(sidecar_mod.sys, "platform", "win32")
        monkeypatch.setattr(
            sidecar_mod.subprocess, "run",
            lambda *a, **kw: calls.append(list(a[0])),
        )
        proc = SimpleNamespace(pid=4321, wait=lambda timeout=None: 0)
        sidecar_mod._kill_tree(proc)
        assert calls == [["taskkill", "/PID", "4321", "/T", "/F"]]

    def test_posix_terminates_then_kills_on_timeout(self, monkeypatch):
        import subprocess as sp

        from searchio.net import sidecar as sidecar_mod

        monkeypatch.setattr(sidecar_mod.sys, "platform", "linux")
        events = []

        def _wait(timeout=None):
            raise sp.TimeoutExpired("cmd", 10)

        proc = SimpleNamespace(
            terminate=lambda: events.append("terminate"),
            wait=_wait,
            kill=lambda: events.append("kill"),
        )
        sidecar_mod._kill_tree(proc)
        assert events == ["terminate", "kill"]


class TestChallengeSidecarLifecycle:
    """close() reaps a sidecar we SPAWNED and spares one we ADOPTED.

    The boundary that leaked in span iteration 32: a bench run draws a
    challenge port with free_port(), and when two Ladders drew the SAME
    port the second found it already healthy and ADOPTED the first's
    sidecar (self._proc stays None). close() only reaps self._proc, so the
    adopted client refused to kill "someone else's" sidecar -- but here the
    someone else was a same-run sibling with no other owner, so a patchright
    tree leaked (a bestbuy tool-check page survived the whole run). The
    spawn side must reap; the adopt side must not (a truly shared sidecar
    belongs to another process). free_port() uniqueness is what keeps every
    bench sidecar on the spawn side -- pinned below.
    """

    def test_spawned_sidecar_is_reaped_on_close(self, monkeypatch):
        from searchio.net import sidecar as sidecar_mod

        reaped = []
        monkeypatch.setattr(sidecar_mod, "_kill_tree", lambda p: reaped.append(p))
        belted = []
        monkeypatch.setattr(sidecar_mod, "_kill_listener_on_port", lambda port, *_a: belted.append(port))
        sc = sidecar_mod.SidecarClient(port=59999, autostart=True)
        proc = SimpleNamespace(pid=777, poll=lambda: None, wait=lambda timeout=None: 0)
        sc._proc = proc  # as if _spawn set it
        asyncio.run(sc.close())
        assert reaped == [proc], "a sidecar we spawned must be tree-killed"
        assert belted == [59999], "close() must verify the reap by port (belt)"
        assert sc._proc is None

    def test_adopted_sidecar_is_not_reaped_on_close(self, monkeypatch):
        from searchio.net import sidecar as sidecar_mod

        reaped = []
        monkeypatch.setattr(sidecar_mod, "_kill_tree", lambda p: reaped.append(p))
        belted = []
        monkeypatch.setattr(sidecar_mod, "_kill_listener_on_port", lambda port, *_a: belted.append(port))
        sc = sidecar_mod.SidecarClient(url="http://127.0.0.1:59998", autostart=False)
        sc._proc = None  # ensure() adopted a healthy port: nothing to reap
        asyncio.run(sc.close())
        assert reaped == [], "an adopted sidecar belongs to another process"
        assert belted == [], "the port belt must never fire for an adopted sidecar"


class TestFreePortUnique:
    """free_port() must never hand out the same port twice in a process.

    A repeat is what let a same-run sibling's challenge port collide and be
    adopted rather than spawned, stranding a patchright tree (span iteration
    32). The allocator tracks handed-out ports and re-rolls.
    """

    def test_many_calls_are_all_distinct(self):
        import sys
        from pathlib import Path

        bench = str(Path(__file__).resolve().parent.parent / "bench")
        if bench not in sys.path:
            sys.path.insert(0, bench)
        try:
            import parbench
        except Exception:  # pragma: no cover - bench not importable here
            pytest.skip("bench/parbench not importable")
        ports = [parbench.free_port() for _ in range(200)]
        assert len(set(ports)) == len(ports), "free_port handed out a duplicate"


class _FailingLadder:
    """fetch() refuses every URL -- a retailer outage / every page walled."""

    def __init__(self, sidecar, exc) -> None:
        self.sidecar = sidecar
        self.exc = exc
        self.calls = 0

    async def fetch(self, url: str):
        self.calls += 1
        raise self.exc


class _SplitLadder(_FakeLadder):
    """fetch() serves the product page for some hosts and refuses others."""

    def __init__(self, sidecar, html: str, refuse_hosts: tuple[str, ...]) -> None:
        super().__init__(sidecar, html)
        self.refuse_hosts = refuse_hosts

    async def fetch(self, url: str):
        from searchio.errors import TransientError

        if any(h in url for h in self.refuse_hosts):
            raise TransientError(f"refused {url}")
        return await super().fetch(url)


def _doc(url: str, title: str = "Sony WH-1000XM5"):
    from searchio.models import Doc

    return Doc(url=url, title=title, snippet="s", source="stub", rank=0)


class TestListingsDomainRestriction:
    """Bug 60 (iteration 40, DeepSeek marketplace.py review): domains= was
    honored loosely on one path and not at all on the other.
    MarketplaceItems.find_items matched ``a in d.domain`` -- a SUBSTRING test
    -- so domains=["amazon.com"] admitted amazon.com.mx / .br / .au (regional
    stores in other currencies; with bug 55 fixed those prices now parse
    correctly and merge_items picks the cheapest offer ACROSS currencies).
    SidecarListings emitted every row the sidecar returned with no domain
    check beyond startswith("http"). Both sites now use one dot-boundary,
    case-folded predicate -- the router's _host_allowed rule.
    """

    async def test_marketplace_items_reject_same_brand_other_tld(self):
        from searchio.models import Query
        from searchio.providers.marketplace import MarketplaceItems

        docs = [_doc("https://www.amazon.com/dp/B09XS7JWHH"),
                _doc("https://www.amazon.com.au/dp/B09XS7JWHH"),
                _doc("https://notamazon.com/dp/B09XS7JWHH")]
        lad = _FakeLadder(_FakeSidecar("engine"), _PRODUCT_HTML)
        q = Query(text="Sony WH-1000XM5", intent="shopping", k=5, domains=["amazon.com"])
        items = await MarketplaceItems().find_items(q, _ctx(lad, docs))
        hosts = sorted({i.url.split("/")[2] for i in items})
        assert hosts == ["www.amazon.com"], hosts

    async def test_sidecar_rows_are_filtered_to_requested_domains(self):
        from searchio.models import Query
        from searchio.providers.marketplace import SidecarListings

        class Sc(_FakeSidecar):
            async def research_listings(self, query: str, **kw):
                self.research_calls.append({"query": query, **kw})
                row = {"title": "Sony WH-1000XM5", "price": "$248.00", "seller": "x",
                       "attributes": {"brand": "Sony"}, "verified": True}
                return {"ok": True, "listings": [
                    {**row, "url": "https://www.ebay.com/itm/276123456789"},
                    {**row, "url": "https://www.notebay.com/itm/1"},
                    {**row, "url": "https://www.ebay.com.au/itm/2"},
                ]}

        sc = Sc("patchright")
        q = Query(text="Sony WH-1000XM5", intent="shopping", k=5, domains=["ebay.com"])
        items = await SidecarListings().find_items(q, _ctx(_FakeLadder(sc, ""), []))
        assert [i.url for i in items] == ["https://www.ebay.com/itm/276123456789"], [i.url for i in items]


class TestListingsOutageHonesty:
    """Bug 61 (iteration 40): MarketplaceItems._one swallowed every fetch
    failure to [] and find_items kept only the list results of
    gather(return_exceptions=True) -- so a page whose EXTRACTOR crashed
    silently contributed nothing, and when EVERY candidate page failed (a
    retailer outage, every page walled) the provider returned [] -- "no
    listings" -- and the router/health/breaker never learned it was down
    (the bug-53 principle one layer up). A partial failure is weather and
    still serves the pages that worked; a total failure is a ProviderError
    naming the reasons.
    """

    async def test_every_page_failing_is_a_refusal_not_empty(self):
        from searchio.errors import Blocked, ProviderError
        from searchio.models import Query
        from searchio.providers.marketplace import MarketplaceItems

        docs = [_doc("https://www.amazon.com/dp/B09XS7JWHH"),
                _doc("https://www.bestbuy.com/site/sony/6505727.p")]
        lad = _FailingLadder(_FakeSidecar("engine"), Blocked("http_403", vendor="cloudflare"))
        q = Query(text="Sony WH-1000XM5", intent="shopping", k=5)
        with pytest.raises(ProviderError, match="all 2 product pages failed"):
            await MarketplaceItems().find_items(q, _ctx(lad, docs))
        assert lad.calls == 2

    async def test_extractor_crash_is_not_silently_discarded(self, monkeypatch):
        from searchio.errors import ProviderError
        from searchio.models import Query
        from searchio.providers import marketplace as mp

        def boom(html, url, source=""):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(mp, "items_from_html", boom)
        docs = [_doc("https://www.amazon.com/dp/B09XS7JWHH")]
        lad = _FakeLadder(_FakeSidecar("engine"), _PRODUCT_HTML)
        q = Query(text="Sony WH-1000XM5", intent="shopping", k=5)
        with pytest.raises(ProviderError, match="RuntimeError"):
            await mp.MarketplaceItems().find_items(q, _ctx(lad, docs))

    async def test_partial_failure_still_serves_the_pages_that_worked(self):
        # Weather, not an outage: one page refused, one served -> items.
        from searchio.models import Query
        from searchio.providers.marketplace import MarketplaceItems

        docs = [_doc("https://www.amazon.com/dp/B09XS7JWHH"),
                _doc("https://www.bestbuy.com/site/sony/6505727.p")]
        lad = _SplitLadder(_FakeSidecar("engine"), _PRODUCT_HTML, refuse_hosts=("bestbuy",))
        q = Query(text="Sony WH-1000XM5", intent="shopping", k=5)
        items = await MarketplaceItems().find_items(q, _ctx(lad, docs))
        assert items and all("amazon.com" in i.url for i in items)

