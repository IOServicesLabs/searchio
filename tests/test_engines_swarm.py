"""SERP parsing, soft-degradation detection, and the swarm's control flow.

The swarm tests use a scripted fake client. They are about the loop's
behaviour at its edges -- budget exhaustion, a worker that never submits, a
tool that errors -- which is where a swarm either degrades gracefully or
returns nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.errors import TransientError
from searchio.models import Item, Price, SubTask
from searchio.providers.engines import (
    _is_self_redirect,
    _unwrap_bing,
    _unwrap_ddg,
    looks_degraded,
    parse_bing,
    parse_bing_rss,
    parse_ddg_html,
    parse_ddg_lite,
    query_terms,
    specific_terms,
)
from searchio.swarm.budget import Budget
from searchio.swarm.tools import ToolBridge, tool_defs
from searchio.swarm.worker import Worker


class TestUrlUnwrapping:
    def test_ddg_redirect(self):
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x"
        assert _unwrap_ddg(wrapped) == "https://example.com/a"

    def test_ddg_passthrough(self):
        assert _unwrap_ddg("https://plain.com/x") == "https://plain.com/x"

    def test_ddg_unwrap_preserves_the_targets_own_percent_encoding(self):
        # Bug 51 (iteration 37, DeepSeek engines.py review): parse_qs already
        # percent-decodes the uddg value once; a second unquote() decoded the
        # TARGET's own encoding too, so a result whose path holds a literal
        # %20 (or %26 / %2B inside a query value) was emitted as a different
        # URL and the fetch hit the wrong resource. DDG single-encodes uddg,
        # so exactly one decode is right.
        from urllib.parse import quote

        target = "https://ex.com/a%20b?q=x%26y"
        wrapped = "//duckduckgo.com/l/?uddg=" + quote(target, safe="") + "&rut=x"
        assert _unwrap_ddg(wrapped) == target

    def test_bing_ck_redirect(self):
        import base64

        target = "https://electronics.sony.com/"
        enc = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        href = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{enc}&ntb=1"
        assert _unwrap_bing(href) == target

    def test_bing_passthrough(self):
        assert _unwrap_bing("https://real.com/x") == "https://real.com/x"


class TestSelfRedirectFilter:
    def test_guard_survives_case_and_port_variants(self):
        # Hardening rider (iteration 37): the host was lower-cased but the
        # path was compared case-sensitively and an explicit port stayed in
        # the host, so DuckDuckGo.com/Y.js and duckduckgo.com:443/y.js slipped
        # past. Providers do not emit these today; the guard must not depend
        # on that (bug-17's class is exactly provider shape drift).
        for u in ("https://DuckDuckGo.com/Y.js?ad=1",
                  "https://duckduckgo.com:443/y.js?ad=1",
                  "https://WWW.BING.COM/CK/A?x"):
            assert _is_self_redirect(u), u

    """A result URL that IS the provider's own redirect/ad hop is not a result.

    bench/span.py run 080002 (forum.reddit_reach): DDG's ad block rode the
    same result__a anchor as organic results, the parser returned
    duckduckgo.com/y.js?ad_domain=... as the TOP hit, and the scenario spent
    a fetch on the ad network's tracking hop.
    """

    def test_classifier(self):
        assert _is_self_redirect("https://duckduckgo.com/y.js?ad_domain=x")
        assert _is_self_redirect("https://duckduckgo.com/l/?kh=-1")
        assert _is_self_redirect("https://lite.duckduckgo.com/y.js?ad_domain=x")
        assert _is_self_redirect("https://www.bing.com/ck/a?!&&p=abc&u=&ntb=1")
        assert not _is_self_redirect("https://duckduckgo.com/about")
        assert not _is_self_redirect("https://en.wikipedia.org/wiki/Rust")
        assert not _is_self_redirect("notaurl")

    def test_ddg_html_drops_ad_anchor(self):
        html = """<html><body>
        <div class="result results_links web-result result--ad">
          <a class="result__a"
             href="https://duckduckgo.com/y.js?ad_domain=bestreviews.com&amp;ad_type=txad"
            >Sponsored Best Reviews</a>
        </div>
        <div class="result results_links web-result">
          <a class="result__a"
             href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.com%2Fpage&amp;rut=abc"
            >Real Page</a>
          <a class="result__snippet" href="x">the snippet</a>
        </div>
        </body></html>"""
        rows = parse_ddg_html(html, 10)
        assert [r[0] for r in rows] == ["https://real.com/page"]

    def test_ddg_html_drops_unresolvable_l_link(self):
        html = """<html><body>
        <a class="result__a" href="https://duckduckgo.com/l/?kh=-1">Tracker</a>
        <a class="result__a" href="https://kept.com/a">Kept</a>
        </body></html>"""
        rows = parse_ddg_html(html, 10)
        assert [r[0] for r in rows] == ["https://kept.com/a"]

    def test_ddg_lite_drops_ad_anchor(self):
        html = """<html><body><table>
        <tr><td><a class="result-link"
           href="https://duckduckgo.com/y.js?ad_domain=x&amp;ad_type=txad">Ad</a></td></tr>
        <tr><td><a class="result-link" href="https://kept.com/b">Kept</a></td></tr>
        </table></body></html>"""
        rows = parse_ddg_lite(html, 10)
        assert [r[0] for r in rows] == ["https://kept.com/b"]

    def test_bing_drops_undecodable_ck_redirect(self):
        html = """<html><body><ol>
        <li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=abc&u=&ntb=1"
           >Broken Redirect</a></h2></li>
        <li class="b_algo"><h2><a href="https://kept.com/c">Kept</a></h2></li>
        </ol></body></html>"""
        rows = parse_bing(html, 10)
        assert [r[0] for r in rows] == ["https://kept.com/c"]

    def test_legit_provider_host_content_survives(self):
        html = """<html><body>
        <a class="result__a" href="https://duckduckgo.com/about">About DDG</a>
        </body></html>"""
        rows = parse_ddg_html(html, 10)
        assert [r[0] for r in rows] == ["https://duckduckgo.com/about"]


class TestBingRss:
    def test_parses_items(self):
        xml = """<rss><channel>
        <item><title>First Result</title><link>https://a.com/1</link>
              <description>Some description</description></item>
        <item><title><![CDATA[Second & Result]]></title><link>https://b.com/2</link>
              <description>More</description></item>
        </channel></rss>"""
        rows = parse_bing_rss(xml, 10)
        assert len(rows) == 2
        assert rows[0] == ("https://a.com/1", "First Result", "Some description", "")
        assert rows[1][1] == "Second & Result"

    def test_pubdate_becomes_iso_date(self):
        # The RSS feed is the only Bing path that knows when a result was
        # published; the date must survive onto the row for Doc.published.
        xml = """<rss><channel>
        <item><title>Dated</title><link>https://a.com/3</link>
              <description>x</description>
              <pubDate>Mon, 07 Sep 2026 08:00:00 GMT</pubDate></item>
        </channel></rss>"""
        rows = parse_bing_rss(xml, 10)
        assert rows[0][3] == "2026-09-07"

    def test_unparseable_pubdate_yields_no_date_not_raw_garbage(self):
        # Bug 171 (iteration 79, engines.py third pass; the acad.fresh_year
        # date-honesty theme): _rss_date returned the RAW pubDate truncated
        # to 16 chars when parsedate_to_datetime failed, so a garbage string
        # rode onto Doc.published -- a machine field the router parses with
        # date.fromisoformat. A non-ISO published escapes every freshness
        # bound as "unprovable" (bug 19's exact hole) and ships a fake date
        # to the agent. The fourth field must be "" (-> published=None), not
        # the raw text.
        for bad in ("not a date at all", "2026/09/07", "yesterday", "Sept 7"):
            xml = (f"<rss><channel><item><title>T</title>"
                   f"<link>https://a.com/1</link><description>d</description>"
                   f"<pubDate>{bad}</pubDate></item></channel></rss>")
            rows = parse_bing_rss(xml, 5)
            assert rows[0][3] == "", f"{bad!r} leaked as published={rows[0][3]!r}"

    def test_rows_carry_four_fields(self):
        # Every engine parser emits (url, title, snippet, published); the
        # fourth field stays "" when the source has no date.
        for parser in (parse_bing_rss,):
            rows = parser(
                "<rss><channel><item><title>T</title><link>https://a.com/1</link>"
                "<description>d</description></item></channel></rss>", 5)
            assert all(len(r) == 4 for r in rows)


class TestFreshnessPassthrough:
    """A freshness bound that never reaches the engine silently returns the
    stale SERP. bench/span.py news.fresh_week caught freshness='week' being
    dropped by both general engines; these pin the params on the wire."""

    class CaptureLadder:
        def __init__(self, body):
            self.urls: list[str] = []
            self._body = body

        async def fetch(self, url, **kw):
            self.urls.append(url)
            return SimpleNamespace(body=self._body)

    DDG_BODY = ('<a class="result__a" href="https://a.com/1">T</a>'
                '<a class="result__snippet">s</a>')
    RSS_BODY = ("<rss><channel><item><title>T</title>"
                "<link>https://a.com/1</link><description>d</description>"
                "</item></channel></rss>")

    async def test_duckduckgo_sends_df(self):
        from searchio.models import Query
        from searchio.providers.engines import DuckDuckGo

        lad = self.CaptureLadder(self.DDG_BODY)
        q = Query(text="x", intent="web", k=5, freshness="week")
        docs = await DuckDuckGo().search(q, SimpleNamespace(ladder=lad))
        assert docs and "df=w" in lad.urls[0]

    async def test_bing_sends_filters(self):
        from searchio.models import Query
        from searchio.providers.engines import Bing

        lad = self.CaptureLadder(self.RSS_BODY)
        q = Query(text="x", intent="web", k=5, freshness="day")
        docs = await Bing().search(q, SimpleNamespace(ladder=lad))
        assert docs and "filters=ex1" in lad.urls[0]

    async def test_all_sends_no_freshness_param(self):
        from searchio.models import Query
        from searchio.providers.engines import Bing, DuckDuckGo

        q = Query(text="x", intent="web", k=5, freshness="all")
        ddg_lad = self.CaptureLadder(self.DDG_BODY)
        await DuckDuckGo().search(q, SimpleNamespace(ladder=ddg_lad))
        assert "df=" not in ddg_lad.urls[0]
        bing_lad = self.CaptureLadder(self.RSS_BODY)
        await Bing().search(q, SimpleNamespace(ladder=bing_lad))
        assert "filters=" not in bing_lad.urls[0]



class TestBingAttemptFallback:
    """Bug 172 (iteration 79, engines.py third pass): the RSS->HTML attempt
    loop caught only Blocked, so a transient RSS fetch error (a timeout or
    reset -- common) propagated out and skipped the HTML fallback, needlessly
    failing Bing, a generalist backstop. Each attempt's fetch failure must
    fall through to the next; only when all fail is it a ProviderError."""

    RSS_BODY = ("<rss><channel><item><title>T</title>"
                "<link>https://a.com/1</link><description>d</description>"
                "</item></channel></rss>")
    HTML_BODY = ('<li class="b_algo"><h2><a href="https://a.com/2">HT</a></h2>'
                 '<div class="b_caption"><p>snip</p></div></li>')

    async def test_a_transient_rss_error_falls_back_to_html(self):
        from searchio.models import Query
        from searchio.providers.engines import Bing
        from searchio.errors import TransientError

        class FlakyLadder:
            def __init__(self):
                self.urls = []

            async def fetch(self, url, **kw):
                self.urls.append(url)
                if "format=rss" in url:
                    raise TransientError("connection reset")
                return SimpleNamespace(body=TestBingAttemptFallback.HTML_BODY)
        lad = FlakyLadder()
        docs = await Bing().search(Query(text="x", intent="web", k=5), SimpleNamespace(ladder=lad))
        assert docs, "the HTML attempt answered after the RSS attempt errored"
        assert len(lad.urls) == 2 and "format=rss" not in lad.urls[1]

    async def test_all_attempts_failing_is_a_clean_provider_error(self):
        from searchio.models import Query
        from searchio.providers.engines import Bing
        from searchio.errors import ProviderError, TransientError

        class DeadLadder:
            async def fetch(self, url, **kw):
                raise TransientError("network down")
        with pytest.raises(ProviderError):
            await Bing().search(Query(text="x", intent="web", k=5), SimpleNamespace(ladder=DeadLadder()))

    async def test_a_parser_crash_still_propagates(self):
        # The fetch except must not swallow a parser regression: the parse
        # call is outside the try, so a broken parser still surfaces.
        from searchio.models import Query
        from searchio.providers.engines import Bing

        class OkLadder:
            async def fetch(self, url, **kw):
                return SimpleNamespace(body=None)  # None body -> parser TypeError
        with pytest.raises((TypeError, AttributeError)):
            await Bing().search(Query(text="x", intent="web", k=5), SimpleNamespace(ladder=OkLadder()))

class TestSidecarSearch:
    """The expensive browser-backed provider against the two backends.

    search_engine_results is one of the patchright script's composite verbs;
    se-serve deliberately does not implement it (documented divergence in
    searchio-engine docs/sidecar-protocol.md). On an engine backend the
    provider must fail fast with the reason, not relay a bare wire
    "Unknown method" -- bench/span.py web.expensive_search surfaced the
    opaque version in a live run."""

    class FakeSidecar:
        def __init__(self, kind):
            self.available = True
            self._kind = kind
            self.search_calls = 0

        async def backend_kind(self):
            return self._kind

        async def search(self, *a, **kw):
            self.search_calls += 1
            return {"ok": True, "results": []}

    async def test_engine_backend_fails_fast_without_sending_the_verb(self):
        from searchio.errors import ProviderError
        from searchio.models import Query
        from searchio.providers.engines import SidecarSearch

        sc = self.FakeSidecar("engine")
        ctx = SimpleNamespace(ladder=SimpleNamespace(sidecar=sc))
        with pytest.raises(ProviderError, match="patchright-only"):
            await SidecarSearch().search(Query(text="x", intent="web", k=5), ctx)
        assert sc.search_calls == 0, "the verb must not be sent to a backend "
        "documented not to implement it"

    async def test_patchright_backend_receives_the_verb(self):
        from searchio.models import Query
        from searchio.providers.engines import SidecarSearch

        sc = self.FakeSidecar("patchright")
        ctx = SimpleNamespace(ladder=SimpleNamespace(sidecar=sc))
        docs = await SidecarSearch().search(Query(text="x", intent="web", k=5), ctx)
        assert sc.search_calls == 1 and docs == []

    async def test_sidecar_results_are_unwrapped_and_ad_hops_dropped(self):
        # Bug 52 (iteration 37): every HTML/RSS parse site applies the bug-17
        # redirector guard, but SidecarSearch emitted whatever the patchright
        # composite verb returned -- and the sidecar unwraps uddg/ck-a links
        # but never drops DDG's /y.js ad hops (its own SERP scrape carries
        # them), while a protocol-relative uddg link was DROPPED outright as
        # non-http (a real result lost). An agent then cites and fetches an
        # ad-network hop: the run-080002 shape bug 17 closed everywhere else.
        import base64
        from searchio.models import Query
        from searchio.providers.engines import SidecarSearch

        enc = base64.urlsafe_b64encode(b"https://electronics.sony.com/").decode().rstrip("=")
        raw = [
            {"url": "https://duckduckgo.com/y.js?ad_domain=x&u3=z", "title": "Ad"},
            {"url": "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x", "title": "W"},
            {"url": f"https://www.bing.com/ck/a?!&&p=abc&u=a1{enc}&ntb=1", "title": "B"},
            {"url": "https://plain.com/x", "title": "P"},
        ]

        class Sc(self.FakeSidecar):
            async def search(self, *a, **kw):
                self.search_calls += 1
                return {"ok": True, "results": raw}

        sc = Sc("patchright")
        ctx = SimpleNamespace(ladder=SimpleNamespace(sidecar=sc))
        docs = await SidecarSearch().search(Query(text="x", intent="web", k=5), ctx)
        assert [d.url for d in docs] == [
            "https://example.com/a", "https://electronics.sony.com/", "https://plain.com/x",
        ], [d.url for d in docs]
        assert [d.rank for d in docs] == [0, 1, 2], "rank follows the EMITTED order"

    async def test_engine_primary_routes_to_the_patchright_challenge_sidecar(self):
        # The composition this system converged on is engine primary +
        # patchright challenge; the patchright-only verb must reach the
        # challenge client, not die on the engine with "Unknown method".
        from searchio.models import Query
        from searchio.providers.engines import SidecarSearch

        engine_sc = self.FakeSidecar("engine")
        patch_sc = self.FakeSidecar("patchright")
        ladder = SimpleNamespace(
            sidecar=engine_sc, fidelity_sidecar=lambda: patch_sc)
        ctx = SimpleNamespace(ladder=ladder)
        await SidecarSearch().search(Query(text="x", intent="web", k=5), ctx)
        assert patch_sc.search_calls == 1
        assert engine_sc.search_calls == 0


class TestDegradation:
    """Detecting a 200 OK that is quietly a brush-off.

    Observed live against Bing from a datacenter IP: every query returned the
    dominant brand's homepage instead of results. Valid markup, correct schema,
    no error anywhere -- only relevance gives it away.
    """

    SONY_DEGRADED = [
        ("https://electronics.sony.com/", "Sony Electronics - Televisions, Audio", ""),
        ("https://www.playstation.com/en-us/", "PlayStation Official Site", ""),
        ("https://www.sony.com/", "Home - Sony Group Portal", ""),
        ("https://en.wikipedia.org/wiki/Sony", "Sony - Wikipedia", ""),
    ]
    SONY_GOOD = [
        ("https://soundguys.com/sony-wh-1000xm5-review-71783/",
         "Sony WH-1000XM5 review - SoundGuys", "earns its spot"),
        ("https://www.whathifi.com/reviews/sony-wh-1000xm5",
         "Sony WH-1000XM5 review", "We originally reviewed"),
        ("https://www.pcmag.com/reviews/sony-wh-1000xm5",
         "Sony WH-1000XM5 Review - PCMag", "top-notch"),
    ]

    def test_specific_terms_finds_model_codes(self):
        assert specific_terms(query_terms("sony wh-1000xm5 review")) == ["wh-1000xm5"]

    def test_brand_only_results_are_degraded(self):
        # Every one of these contains "sony"; none contains the model code.
        assert looks_degraded("sony wh-1000xm5 review", self.SONY_DEGRADED)

    def test_genuine_results_are_not_degraded(self):
        assert not looks_degraded("sony wh-1000xm5 review", self.SONY_GOOD)

    def test_degradation_without_a_model_code(self):
        rows = [
            ("https://www.postgresql.org/", "PostgreSQL: most advanced database", ""),
            ("https://www.postgresql.org/download/", "PostgreSQL: Downloads", ""),
            ("https://en.wikipedia.org/wiki/PostgreSQL", "PostgreSQL - Wikipedia", ""),
        ]
        assert looks_degraded("postgres index bloat vacuum", rows)

    def test_long_vocabulary_words_do_not_false_positive(self):
        # "documentation" is 13 chars but phrased "docs" on every real result.
        # Treating long words as distinctive condemned two healthy SERPs at
        # once (bench/span.py web.js_docs: ddg AND bing, same query).
        rows = [
            ("https://nextjs.org/docs/app", "App Router | Next.js", "React framework docs"),
            ("https://nextjs.org/docs/app/getting-started", "Getting Started | Next.js", ""),
            ("https://vercel.com/templates/next.js", "Next.js Templates", "app router"),
        ]
        assert not looks_degraded("nextjs app router documentation", rows)

    def test_paraphrasing_results_are_not_degraded(self):
        # The false positive that would discard perfectly good results.
        rows = [
            ("https://a.com/1", "Optimizing slow queries in your database", "speed up"),
            ("https://b.com/2", "Database performance: fixing slow queries", ""),
            ("https://c.com/3", "Why your queries are slow and how to fix them", "database"),
        ]
        assert not looks_degraded("how to fix slow database queries", rows)

    def test_too_few_results_is_not_a_verdict(self):
        assert not looks_degraded("anything at all", [("https://a.com", "x", "")])


# ── swarm ────────────────────────────────────────────────────────────────────


class FakeEngine:
    """Minimal engine surface the ToolBridge needs."""

    def __init__(self, settings):
        self.s = settings
        self.router = SimpleNamespace()
        self.ladder = SimpleNamespace(fetch=self._fetch)
        self.items_to_return: list[Item] = []
        self.docs_to_return = "default"
        self.failed_to_return: dict = {}
        self.filtered_to_return: dict = {}
        self.search_calls: list[dict] = []
        self.find_items_calls: list[dict] = []

    async def search(self, text, **kw):
        from searchio.models import Doc

        self.search_calls.append({"text": text, **kw})
        docs = self.docs_to_return
        if docs == "default":
            docs = [Doc(url="https://a.com/1", title="Result", snippet="snip", source="stub")]
        return SimpleNamespace(
            docs=list(docs),
            used=["stub"] if docs else [],
            failed=dict(self.failed_to_return),
            filtered=dict(self.filtered_to_return),
            per_provider={}, elapsed_ms=1,
        )

    async def _fetch(self, url, **kw):
        from searchio.models import FetchResult

        return FetchResult(url=url, final_url=url, status=200,
                           body="<html><body><p>Page body text.</p></body></html>",
                           tier=0, via="http")

    async def find_items(self, text, domains=None):
        self.find_items_calls.append({"text": text, "domains": domains})
        return list(self.items_to_return)


def block(**kw):
    return SimpleNamespace(**kw)


class FakeClient:
    """Scripted Anthropic client: returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.kwargs: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)


def response(content, *, stop_reason="tool_use", in_tok=100, out_tok=50):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )


def tool_use(name, inp, tid="t1"):
    return block(type="tool_use", name=name, input=inp, id=tid)


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, max_tool_calls_per_worker=4)


class TestToolBridge:
    async def test_search_returns_compact_results(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        text, err = await bridge.dispatch("web_search", {"query": "x", "intent": "web"})
        assert not err and "Result" in text and "https://a.com/1" in text

    async def test_read_page_truncates(self, settings):
        eng = FakeEngine(settings)
        long_body = "<html><body><p>" + ("word " * 20000) + "</p></body></html>"

        async def big_fetch(url, **kw):
            from searchio.models import FetchResult

            return FetchResult(url=url, status=200, body=long_body, tier=0, via="http")

        eng.ladder.fetch = big_fetch
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page", {"url": "https://a.com"})
        assert not err and "truncated" in text

    async def test_tool_error_is_returned_not_raised(self, settings):
        eng = FakeEngine(settings)

        async def boom(url, **kw):
            raise RuntimeError("blocked hard")

        eng.ladder.fetch = boom
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page", {"url": "https://a.com"})
        assert err and "blocked hard" in text

    async def test_read_page_rendered_uses_the_rendered_fetch(self, settings):
        eng = FakeEngine(settings)
        seen: dict = {}

        async def fetch(url, **kw):
            seen.update(kw)
            from searchio.models import FetchResult

            return FetchResult(
                url=url, status=200,
                body="<html><body><p>Full render body.</p></body></html>",
                tier=2, via="browser", rendered=True,
            )

        eng.ladder.fetch = fetch
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page_rendered", {"url": "https://a.com"})
        assert not err and "Full render body." in text
        assert seen.get("rendered") is True
        assert seen.get("force_tier") == 2, "the rendered tool must pay for the browser tier"
        assert "full-browser render" in text

    async def test_thin_read_page_hints_at_read_page_rendered(self, settings):
        # The escalation lesson must be inline: a shell from the cheap path
        # should tell the model exactly which tool fixes it.
        eng = FakeEngine(settings)

        async def shell_fetch(url, **kw):
            from searchio.models import FetchResult

            return FetchResult(url=url, status=200,
                               body='<html><div id="root"></div></html>',
                               tier=2, via="browser", rendered=False)

        eng.ladder.fetch = shell_fetch
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page", {"url": "https://a.com"})
        assert not err and "read_page_rendered" in text

    async def test_hard_shell_error_also_hints_at_read_page_rendered(self, settings):
        # The thin-page hint covers results that come back readable-but-empty;
        # the all-transports-empty failure must teach the same lesson, or the
        # model has no escalation path when the ladder raises instead of
        # returning a shell.
        eng = FakeEngine(settings)

        async def empty_fetch(url, **kw):
            raise TransientError(
                "no usable content at https://a.com "
                "(tried ['tier0:empty_mount', 'tier2:empty_mount'])"
            )

        eng.ladder.fetch = empty_fetch
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page", {"url": "https://a.com"})
        assert err and "read_page_rendered" in text
        assert "JavaScript shell" in text

    async def test_hard_error_hint_not_added_to_rendered_calls(self, settings):
        # read_page_rendered IS the top tier; a failure there must surface as a
        # plain error, not a suggestion to escalate to itself.
        eng = FakeEngine(settings)

        async def empty_fetch(url, **kw):
            raise TransientError("no usable content (tried ['tier2_rendered'])")

        eng.ladder.fetch = empty_fetch
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch(
            "read_page_rendered", {"url": "https://a.com"})
        assert err and "read_page_rendered on the same URL" not in text

    async def test_submit_report_captures_findings(self, settings):
        bridge = ToolBridge(FakeEngine(settings), subtask_id="s1")
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})  # retrieves a.com/1
        await bridge.dispatch("submit_report", {
            "summary": "done",
            "findings": [{"claim": "C", "confidence": "high",
                          "citations": [{"url": "https://a.com/1", "title": "T"}]}],
            "gaps": ["nothing"],
        })
        assert bridge.report is not None
        assert bridge.report.findings[0].citations[0].url == "https://a.com/1"
        assert bridge.report.subtask_id == "s1"


class TestToolBridgeEmptyHonesty:
    """The empty-answer contract: tell the agent WHY, or it rephrase-loops.

    bench/span.py llm.price_check burned its whole turn budget re-searching
    during a DuckDuckGo/Bing brush-off because "Try different keywords" was
    the only advice an attrition-empty answer ever gave (three runs red).
    """

    def _open_registry(self, *providers):
        import time as _time

        from searchio.providers.registry import Registry

        for p in providers:
            p.health.opened_at = _time.monotonic()
        return Registry(list(providers))

    async def test_empty_search_names_circuit_open_providers(self, settings):
        from searchio.providers.engines import DuckDuckGo

        eng = FakeEngine(settings)
        eng.docs_to_return = []
        eng.router.registry = self._open_registry(DuckDuckGo())
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch(
            "web_search", {"query": "x", "intent": "web"})
        assert not err
        assert "circuit-open" in text and "duckduckgo" in text
        assert "partial report" in text
        assert "Try different keywords" not in text, \
            "keyword advice is the wrong remedy for provider attrition"

    async def test_empty_search_healthy_providers_keeps_keyword_advice(self, settings):
        eng = FakeEngine(settings)
        eng.docs_to_return = []
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch(
            "web_search", {"query": "x", "intent": "web"})
        assert not err and "Try different keywords" in text
        assert "circuit-open" not in text

    async def test_empty_search_reports_domain_filter_drops(self, settings):
        eng = FakeEngine(settings)
        eng.docs_to_return = []
        eng.filtered_to_return = {"bing": 5}
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch(
            "web_search", {"query": "x", "intent": "web"})
        assert not err and "restrictions dropped" in text and "off-domain" in text

    async def test_empty_find_items_names_circuit_open_providers(self, settings):
        from searchio.providers.engines import DuckDuckGo

        eng = FakeEngine(settings)
        eng.items_to_return = []
        eng.router.registry = self._open_registry(DuckDuckGo())
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("find_items", {"query": "x"})
        assert not err and "circuit-open" in text and "duckduckgo" in text

    async def test_web_search_passes_domains_and_freshness_through(self, settings):
        eng = FakeEngine(settings)
        bridge = ToolBridge(eng)
        await bridge.dispatch("web_search", {
            "query": "gpu prices", "intent": "shopping", "k": 5,
            "domains": ["newegg.com"], "exclude_domains": ["ebay.com"],
            "freshness": "week",
        })
        call = eng.search_calls[-1]
        assert call["domains"] == ["newegg.com"]
        assert call["exclude_domains"] == ["ebay.com"]
        assert call["freshness"] == "week"
        assert call["k"] == 5

    async def test_web_search_inline_site_operator_reaches_the_query(self, settings):
        """The bridge must not bypass Engine.search's operator extraction."""
        from searchio.engine import Engine

        captured: dict = {}

        class CaptureRouter:
            async def search(self, q, **kw):
                captured["text"] = q.text
                captured["domains"] = q.domains
                return SimpleNamespace(docs=[], used=[], failed={},
                                       filtered={}, per_provider={},
                                       elapsed_ms=1)

        eng = Engine(settings)
        eng.router = CaptureRouter()
        bridge = ToolBridge(eng)
        await bridge.dispatch(
            "web_search", {"query": "headphones site:bestbuy.com", "intent": "web"})
        assert captured["text"] == "headphones"
        assert captured["domains"] == ["bestbuy.com"]


class TestToolBridgeContracts:
    """Small hard contracts an agent relies on without being able to see them.

    These exist so a refactor cannot quietly widen a fan-out, leak an
    over-long query, or accept a malformed URL. The live twins run in
    bench/span.py's tool leg (tool.k_cap, tool.invalid_url,
    tool.submit_contract).
    """

    async def test_k_is_clamped_into_the_schema_range(self, settings):
        eng = FakeEngine(settings)
        bridge = ToolBridge(eng)
        await bridge.dispatch("web_search",
                              {"query": "x", "intent": "web", "k": 99})
        assert eng.search_calls[-1]["k"] == 15
        await bridge.dispatch("web_search",
                              {"query": "x", "intent": "web", "k": -5})
        assert eng.search_calls[-1]["k"] == 1
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})
        assert eng.search_calls[-1]["k"] == 8

    async def test_query_is_truncated_to_300_chars(self, settings):
        eng = FakeEngine(settings)
        bridge = ToolBridge(eng)
        long_q = "word " * 100  # 500 chars
        await bridge.dispatch("web_search", {"query": long_q, "intent": "web"})
        assert len(eng.search_calls[-1]["text"]) == 300

    async def test_find_items_query_is_truncated_to_300_chars(self, settings):
        eng = FakeEngine(settings)
        eng.items_to_return = []
        bridge = ToolBridge(eng)
        await bridge.dispatch("find_items", {"query": "word " * 100})
        assert len(eng.find_items_calls[-1]["text"]) == 300

    async def test_read_page_rejects_non_http_url(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        for bad in ("notaurl", "ftp://x.com/f", "/relative/path", ""):
            text, err = await bridge.dispatch("read_page", {"url": bad})
            assert err, f"read_page accepted {bad!r}"
            assert "Invalid URL" in text

    async def test_read_page_rendered_rejects_non_http_url(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        text, err = await bridge.dispatch(
            "read_page_rendered", {"url": "file:///etc/passwd"})
        assert err and "Invalid URL" in text

    async def test_submit_report_drops_url_less_citations(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})  # retrieves a.com/1
        await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "C", "confidence": "high",
                          "citations": [{"title": "no url"},
                                        {"url": "https://a.com/1"}]}],
        })
        cites = bridge.report.findings[0].citations
        assert [c.url for c in cites] == ["https://a.com/1"]

    async def test_submit_report_caps_quotes_at_400_chars(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})  # retrieves a.com/1
        await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "C", "confidence": "high",
                          "citations": [{"url": "https://a.com/1",
                                         "quote": "x" * 500}]}],
        })
        assert len(bridge.report.findings[0].citations[0].quote) == 400

    async def test_unknown_tool_is_an_error_not_a_crash(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        text, err = await bridge.dispatch("delete_everything", {})
        assert err and "unknown tool" in text


class TestWorker:
    def _subtask(self):
        return SubTask(id="s1", objective="Find out X", intent="web")

    async def test_completes_on_submit_report(self, settings):
        client = FakeClient([
            response([tool_use("web_search", {"query": "x", "intent": "web"})]),
            response([tool_use("submit_report", {
                "summary": "Answered.",
                "findings": [{"claim": "X is true", "confidence": "high",
                              "citations": [{"url": "https://a.com/1"}]}],
            }, tid="t2")]),
        ])
        report = await Worker(FakeEngine(settings), client, self._subtask(), Budget()).run()
        assert report.summary == "Answered."
        assert len(report.findings) == 1

    async def test_partial_report_when_tool_budget_runs_out(self, settings):
        # A worker that never submits must still hand back what it retrieved.
        never = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")])
                 for i in range(10)]
        client = FakeClient(never)
        budget = Budget(max_tool_calls_per_worker=3)
        report = await Worker(FakeEngine(settings), client, self._subtask(), budget).run()
        assert report.tool_calls == 3
        assert "Did not complete" in report.summary
        assert report.docs, "sources retrieved before running out must be preserved"

    async def test_stops_when_token_budget_exhausted(self, settings):
        client = FakeClient([
            response([tool_use("web_search", {"query": "x", "intent": "web"})],
                     in_tok=10_000, out_tok=10_000),
            response([tool_use("web_search", {"query": "y", "intent": "web"}, tid="t2")]),
        ])
        budget = Budget(max_tokens=15_000)
        report = await Worker(FakeEngine(settings), client, self._subtask(), budget).run()
        assert budget.stopped_reason.startswith("token budget")
        assert report.tool_calls == 1

    async def test_api_error_becomes_a_reported_failure(self, settings):
        class Boom(FakeClient):
            async def _create(self, **kw):
                raise RuntimeError("api down")

        report = await Worker(FakeEngine(settings), Boom([]), self._subtask(), Budget()).run()
        assert "api down" in report.error

    async def test_refusal_is_handled(self, settings):
        client = FakeClient([response([], stop_reason="refusal")])
        report = await Worker(FakeEngine(settings), client, self._subtask(), Budget()).run()
        assert "declined" in report.error

    async def test_system_prompt_carries_the_budget_and_stopping_rule(self, settings):
        # bench/span.py run 080002: agents rephrase-looped to the turn cap
        # under circuit-open providers because the prompt never stated the
        # budget or when to stop. Both create sites must carry it.
        client = FakeClient([
            response([tool_use("submit_report", {
                "summary": "Done.",
                "findings": [{"claim": "x", "confidence": "high",
                              "citations": [{"url": "https://a.com/1"}]}],
            })]),
        ])
        await Worker(FakeEngine(settings), client, self._subtask(), Budget()).run()
        system = client.kwargs[0]["system"]
        assert "at most 12 tool calls" in system  # Budget() default
        assert "circuit-open" in system

    async def test_force_submit_carries_the_budget_prompt_too(self, settings):
        never = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")])
                 for i in range(10)]
        client = FakeClient(never)
        budget = Budget(max_tool_calls_per_worker=3)
        report = await Worker(FakeEngine(settings), client, self._subtask(), budget).run()
        assert report.summary == "" or "Did not complete" in report.summary
        # The last create call is the force-submit attempt: submit_report
        # only, and the same budget-bearing system prompt.
        last = client.kwargs[-1]
        assert "at most 3 tool calls" in last["system"]
        assert "circuit-open" in last["system"]
        assert [t["name"] for t in last["tools"]] == ["submit_report"]
        # And the instruction message must be the force-submit wording.
        last_user = [m for m in last["messages"] if m["role"] == "user"][-1]
        assert "used your tool budget" in str(last_user["content"])


class TestBudget:
    def test_tracks_and_exhausts(self):
        b = Budget(max_tokens=1000)
        b.record(600, 300)
        assert b.soft_check()
        b.record(200, 0)
        assert not b.soft_check() and "token budget" in b.stopped_reason

    def test_cost_estimate(self):
        b = Budget()
        b.record(1_000_000, 1_000_000)
        assert b.estimated_cost == pytest.approx(30.0)  # $5 in + $25 out


def test_tool_definitions_are_well_formed():
    defs = tool_defs()
    names = {d["name"] for d in defs}
    assert names == {
        "web_search", "read_page", "read_page_rendered", "find_items", "submit_report",
    }
    for d in defs:
        schema = d["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        json.dumps(schema)  # must be serialisable


class TestInlineSiteOperators:
    """Agents write site:/-site: inline because that is how engines speak.
    Parsed into the structured fields, the router post-filter can actually
    keep the promise (bench/span.py llm.price_check wrote site:bestbuy.com
    inline and got a useless empty page for it)."""

    def test_parses_site_and_minus_site_out_of_query_text(self):
        from searchio.engine import _extract_operators

        text, dom, exc = _extract_operators(
            "sony headphones site:bestbuy.com -site:pinterest.com", None, None
        )
        assert text == "sony headphones"
        assert dom == ["bestbuy.com"]
        assert exc == ["pinterest.com"]

    def test_merges_with_structured_fields_and_keeps_quotes(self):
        from searchio.engine import _extract_operators

        text, dom, exc = _extract_operators(
            '"exact phrase" SITE:Example.COM', ["epa.gov"], ["junk.com"]
        )
        assert text == '"exact phrase"'
        assert dom == ["epa.gov", "example.com"]
        assert exc == ["junk.com"]

    def test_no_operators_is_a_noop(self):
        from searchio.engine import _extract_operators

        assert _extract_operators("plain query", None, None) == ("plain query", [], [])

    async def test_engine_search_routes_inline_site_into_the_query(self, settings):
        # End to end through Engine.search: the router must receive the
        # structured fields so the post-filter engages.
        from searchio.engine import Engine

        captured = {}

        class CaptureRouter:
            async def search(self, q, **_kw):
                captured["q"] = q
                return SimpleNamespace(docs=[], used=[], failed={})

        eng = Engine(settings)
        eng.router = CaptureRouter()
        await eng.search("sony headphones site:bestbuy.com")
        q = captured["q"]
        assert q.text == "sony headphones"
        assert q.domains == ["bestbuy.com"]
        await eng.ladder.close()


class TestExtractOperators:
    """_extract_operators must reduce a site: value to a bare HOST (bug 50).

    The router filters by host, so a site: value carrying a scheme, a path, a
    port, or trailing punctuation (all of which an LLM writes -- site:
    https://reddit.com/r/python, site:foo.com/bar, site:foo.com,) never
    matched and silently dropped every doc: an honest-but-empty page.
    """

    def test_site_operator_normalized_to_bare_host(self):
        from searchio.engine import _extract_operators

        cases = [
            ("headphones site:bestbuy.com", ["bestbuy.com"], []),
            ("x site:https://reddit.com/r/python", ["reddit.com"], []),
            ("x site:foo.com/bar", ["foo.com"], []),
            ("x site:FOO.com,", ["foo.com"], []),
            ("x -site:pinterest.com.", [], ["pinterest.com"]),
            ("x site:www.example.com", ["example.com"], []),
            ("x site:shop.example.com:8443/deals", ["shop.example.com"], []),
        ]
        for q, dom, exc in cases:
            _, d, e = _extract_operators(q, None, None)
            assert d == dom and e == exc, (q, d, e)

    def test_garbage_operator_value_adds_no_domain(self):
        from searchio.engine import _extract_operators
        # A value that reduces to nothing must not inject an empty domain (an
        # empty allow-list domain would drop everything).
        _, d, e = _extract_operators("x site:///", None, None)
        assert d == [] and e == []


class TestFindLocalCity:
    """find_local_items must not leak one call's city into the next (bug 49).

    ``city`` is per-query geo; the FB provider's *session* state
    (session-gate, breaker health) lives in separate attributes. The
    ``if city:`` guard meant an empty-city call inherited the previous call's
    city -- a query meant for the exit-IP metro silently searched the last
    city instead.
    """

    async def test_city_is_not_inherited_across_calls(self, settings):
        from searchio.engine import Engine

        class _StubFB:
            name = "facebook_marketplace"

            def __init__(self):
                self.city = ""
                self.seen: list[str] = []

            async def find_items(self, q, ctx):
                self.seen.append(self.city)
                return []

        stub = _StubFB()
        eng = Engine(settings)
        eng.registry.get = lambda name: stub if name == "facebook_marketplace" else None
        try:
            await eng.find_local_items("kayak", city="Austin")
            await eng.find_local_items("bike")  # no city -> exit-IP, NOT Austin
            assert stub.seen == ["Austin", ""], stub.seen
        finally:
            await eng.close()


class TestSubmitProvenance:
    """Bug 62 (iteration 41, DeepSeek tools.py review): submit_report accepted
    any citation with a truthy url and never checked seen_docs -- the tool's
    own contract says "a citation URL you actually retrieved". This is the
    exact shape span's LLM leg kept flagging as model weather (llm.price_check:
    DeepSeek cited a target.com/s?searchTerm=... URL it never fetched):
    fabricated provenance walked straight into the WorkerReport. The bridge
    now drops citations whose canonical URL was never retrieved in this task
    and records the drop in the report's gaps, so the report stays honest and
    the model (and the suite) can see it.
    """

    async def test_unretrieved_citation_is_dropped_and_recorded_as_a_gap(self, settings):
        bridge = ToolBridge(FakeEngine(settings), subtask_id="s1")
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})  # a.com/1 seen
        text, err = await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "C", "confidence": "high", "citations": [
                {"url": "https://a.com/1", "title": "seen"},
                {"url": "https://www.target.com/s?searchTerm=sony", "title": "never fetched"},
            ]}],
        })
        assert not err
        cites = bridge.report.findings[0].citations
        assert [c.url for c in cites] == ["https://a.com/1"], [c.url for c in cites]
        assert any("target.com" in g and "never retrieved" in g for g in bridge.report.gaps), bridge.report.gaps
        assert "never retrieved" in text and "target.com" in text

    async def test_url_variants_of_a_retrieved_page_are_tolerated(self, settings):
        # Tracking params, www., scheme: the same page, not a fabrication.
        bridge = ToolBridge(FakeEngine(settings))
        await bridge.dispatch("web_search", {"query": "x", "intent": "web"})  # https://a.com/1
        await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "C", "confidence": "high", "citations": [
                {"url": "http://www.a.com/1?utm_source=chat"}]}],
        })
        assert [c.url for c in bridge.report.findings[0].citations] == ["http://www.a.com/1?utm_source=chat"]
        assert bridge.report.gaps == []

    async def test_a_read_page_url_counts_as_retrieved(self, settings):
        bridge = ToolBridge(FakeEngine(settings))
        await bridge.dispatch("read_page", {"url": "https://read.example/p"})
        await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "C", "confidence": "high", "citations": [
                {"url": "https://read.example/p"}]}],
        })
        assert [c.url for c in bridge.report.findings[0].citations] == ["https://read.example/p"]


class TestToolTextBounds:
    async def test_long_titles_are_capped_in_search_results(self, settings):
        # Rider (iteration 41): d.title went into the tool text uncapped; a
        # hostile <title> times k=15 could flood the model's context.
        from searchio.models import Doc

        eng = FakeEngine(settings)
        eng.docs_to_return = [Doc(url="https://a.com/1", title="T" * 5000,
                                  snippet="s", source="stub")]
        text = await ToolBridge(eng).dispatch("web_search", {"query": "x", "intent": "web"})
        text = text[0]
        assert len(text) < 1200, len(text)

    async def test_find_items_header_says_how_many_are_shown(self, settings):
        # Rider: the header said "N listings:" and then rendered 15 -- the
        # model believed it saw N. Now "showing 15 of N".
        eng = FakeEngine(settings)
        eng.items_to_return = [Item(title=f"I{i}", url=f"https://s.example/{i}",
                                    price=Price(amount=1.0 + i, currency="USD"))
                               for i in range(20)]
        text, _ = await ToolBridge(eng).dispatch("find_items", {"query": "q"})
        assert text.startswith("showing 15 of 20 listings"), text[:60]

    async def test_repeated_find_items_do_not_duplicate_report_items(self, settings):
        eng = FakeEngine(settings)
        eng.items_to_return = [Item(title="I", url="https://s.example/1",
                                    price=Price(amount=1.0, currency="USD"))]
        bridge = ToolBridge(eng)
        await bridge.dispatch("find_items", {"query": "q"})
        await bridge.dispatch("find_items", {"query": "q again"})
        await bridge.dispatch("submit_report", {"summary": "s", "findings": []})
        assert [i.url for i in bridge.report.items] == ["https://s.example/1"]


class TestWorkerBudgetGuards:
    """Bug 63 (iteration 42, DeepSeek worker.py review): the per-turn dispatch
    loop executed EVERY tool_use block in an assistant turn. The cap was
    checked only at the top of the while loop, so a parallel batch of five
    searches at calls == cap-1 ran to cap+4 -- past the budget the system
    prompt PROMISES the model ("at most N tool calls") -- and tools placed
    after submit_report in the same turn still executed after the report was
    built. The loop now answers every tool_use id (the API requires a
    tool_result per id) but executes nothing past the cap (submit_report
    itself excepted, it is the terminal step) or after a report exists.
    """

    def _subtask(self):
        return SubTask(id="s1", objective="Find out X", intent="web")

    async def test_a_parallel_tool_batch_cannot_exceed_the_cap(self, settings):
        eng = FakeEngine(settings)
        batch = [tool_use("web_search", {"query": f"q{i}", "intent": "web"}, tid=f"t{i}")
                 for i in range(5)]
        client = FakeClient([
            response(batch),
            response([tool_use("submit_report", {"summary": "done", "findings": []}, tid="ts")]),
        ])
        budget = Budget(max_tool_calls_per_worker=2)
        report = await Worker(eng, client, self._subtask(), budget).run()
        assert len(eng.search_calls) == 2, "only the calls within the cap execute"
        # Every id in the batch still got a tool_result (three of them errors).
        results_msg = client.kwargs[-1]["messages"][-2]["content"]
        assert [r["tool_use_id"] for r in results_msg] == [f"t{i}" for i in range(5)]
        assert sum(1 for r in results_msg if r["is_error"]) == 3
        assert report is not None and report.tool_calls == 3  # 2 searches + the forced submit

    async def test_tools_after_submit_in_the_same_turn_are_not_executed(self, settings):
        eng = FakeEngine(settings)
        client = FakeClient([response([
            tool_use("submit_report", {"summary": "done", "findings": []}, tid="t1"),
            tool_use("web_search", {"query": "late", "intent": "web"}, tid="t2"),
        ])])
        report = await Worker(eng, client, self._subtask(), Budget()).run()
        assert report is not None and report.summary == "done"
        assert eng.search_calls == [], "nothing runs after the report is submitted"


class TestForceSubmitHonesty:
    """Bug 64 (iteration 42): _force_submit swallowed both messages.create
    attempts' exceptions (``except Exception: continue``) and never looked at
    stop_reason, returned None, and the fallback report then set error="" at
    the cap -- so an API outage or a refusal DURING the forced turn was
    invisible to the orchestrator, while the same failures in the main loop
    are honestly recorded (report.error). The forced turn's last failure now
    lands in the fallback report's error.
    """

    def _subtask(self):
        return SubTask(id="s1", objective="Find out X", intent="web")

    async def test_force_submit_api_failure_is_reported_not_blank(self, settings):
        class Flaky(FakeClient):
            async def _create(self, **kw):
                if self.calls >= 1:
                    self.calls += 1
                    raise RuntimeError("api down")
                return await super()._create(**kw)

        client = Flaky([response([tool_use("web_search", {"query": "x", "intent": "web"})])])
        report = await Worker(FakeEngine(settings), client, self._subtask(),
                              Budget(max_tool_calls_per_worker=1)).run()
        assert "api down" in report.error, report.error
        assert report.docs, "the retrieved sources are still kept"

    async def test_force_submit_refusal_is_reported(self, settings):
        client = FakeClient([
            response([tool_use("web_search", {"query": "x", "intent": "web"})]),
            response([], stop_reason="refusal"),
            response([], stop_reason="refusal"),
        ])
        report = await Worker(FakeEngine(settings), client, self._subtask(),
                              Budget(max_tool_calls_per_worker=1)).run()
        assert "declined" in report.error, report.error


class TestDispatchRobustness:
    async def test_malformed_tool_input_is_an_error_result_not_a_crash(self, settings):
        # Rider (iteration 42): json.loads on a non-dict input sat outside
        # dispatch's try, so a malformed string escaped as JSONDecodeError and
        # would take the worker loop down with it.
        bridge = ToolBridge(FakeEngine(settings))
        text, err = await bridge.dispatch("web_search", "{not json")
        assert err and "JSONDecodeError" in text


class TestSerpGuardsSecondPass:
    """Iteration 55 riders (engines.py second pass, probes): two guards that
    hold for today's provider shapes but not for the drift they exist for."""

    def test_percent_encoded_redirector_path_is_still_a_redirector(self):
        from searchio.providers.engines import _is_self_redirect

        assert _is_self_redirect("https://www.bing.com/ck/a?u=x")
        assert _is_self_redirect("https://www.bing.com/ck%2Fa?u=x")  # was False
        assert _is_self_redirect("https://duckduckgo.com/%6C/?uddg=x")  # was False
        assert not _is_self_redirect("https://html.duckduckgo.com/html/?q=x")

    def test_uddg_is_only_unwrapped_on_duckduckgo_hosts(self):
        from searchio.providers.engines import _unwrap_ddg

        assert _unwrap_ddg("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgood.example%2Fp") == "https://good.example/p"
        assert _unwrap_ddg("//duckduckgo.com/l/?uddg=https%3A%2F%2Fgood.example%2F") == "https://good.example/"
        foreign = "https://evil.example/page?uddg=https%3A%2F%2Fpayload.example%2F"
        assert _unwrap_ddg(foreign) == foreign  # was rewritten to payload.example


class TestShellHintOnCacheReplay:
    async def test_cached_shell_still_gets_the_render_hint(self, settings):
        # Iteration 56 rider (Muse review of tools.py): the JS-shell hint was
        # suppressed when the thin body came from the cache, so the second
        # agent (or the same agent in a new task) that read a cached shell
        # never learned that read_page_rendered was the move.
        from searchio.models import FetchResult

        eng = FakeEngine(settings)

        async def cached_shell(url, **kw):
            return FetchResult(url=url, final_url=url, status=200, body="<html><body><div id=app></div></body></html>",
                               content_type="text/html", tier=0, via="cache", elapsed_ms=1, from_cache=True)

        eng.ladder = SimpleNamespace(fetch=cached_shell)
        bridge = ToolBridge(eng)
        text, err = await bridge.dispatch("read_page", {"url": "https://spa.example/"})
        assert not err and "JavaScript shell" in text, text[:200]


class TestBridgeBookkeeping56:
    """Iteration 56 (Muse 8-sample review of tools.py + probes): three
    bookkeeping defects on the agent path, each red before its fix."""

    async def test_find_items_listings_count_as_retrieved(self, settings):
        # Bug 105: find_items showed listing URLs to the model but never
        # registered them as seen, so citing the listing the agent had just
        # retrieved was dropped as "never retrieved in this task".
        eng = FakeEngine(settings)
        eng.items_to_return = [Item(title="Sony", url="https://shop.example/p/1",
                                    price=Price(amount=10.0, currency="USD"))]
        bridge = ToolBridge(eng)
        await bridge.dispatch("find_items", {"query": "sony"})
        text, err = await bridge.dispatch("submit_report", {
            "summary": "s", "findings": [{"claim": "c", "confidence": "high",
                                          "citations": [{"url": "https://shop.example/p/1", "title": "Sony"}]}],
            "gaps": []})
        assert not err and [c.url for c in bridge.report.findings[0].citations] == ["https://shop.example/p/1"]

    async def test_search_does_not_overwrite_a_read_page(self, settings):
        # Bug 106: web_search stored every result over seen_docs[url], so a
        # page read_page had already fetched (with its text) was replaced by
        # the bare search hit -- the report's sources lost the text.
        eng = FakeEngine(settings)
        bridge = ToolBridge(eng)
        await bridge.dispatch("read_page", {"url": "https://a.com/1"})
        before = len(bridge.seen_docs["https://a.com/1"].text)
        assert before > 0
        await bridge.dispatch("web_search", {"query": "x"})  # FakeEngine returns https://a.com/1
        assert len(bridge.seen_docs["https://a.com/1"].text) == before

    async def test_gaps_given_as_a_string_is_one_gap(self, settings):
        # Rider: a model that passed gaps as a plain string had it iterated
        # character by character ("no data" -> ['n', 'o', ' ', ...]).
        bridge = ToolBridge(FakeEngine(settings))
        await bridge.dispatch("web_search", {"query": "x"})
        await bridge.dispatch("submit_report", {"summary": "s", "findings": [], "gaps": "no more data"})
        assert bridge.report.gaps == ["no more data"]


class TestColonLessSiteOperator:
    def test_colon_less_site_operator_from_a_weaker_model(self):
        # Iteration 56 (Muse agent leg): the model wrote "site nasa.gov ..."
        # without the colon; the extractor saw a keyword, not an operator,
        # and the site-scoped question ran unscoped.
        from searchio.engine import _extract_operators

        text, dom, exc = _extract_operators("site nasa.gov Artemis II crewed mission", None, None)
        assert dom == ["nasa.gov"] and text == "Artemis II crewed mission", (text, dom)
        text, dom, exc = _extract_operators("site: nasa.gov Artemis", None, None)
        assert dom == ["nasa.gov"] and text == "Artemis"
        # A bare "site" that is just a word stays a word.
        text, dom, exc = _extract_operators("building site safety rules", None, None)
        assert dom == [] and text == "building site safety rules"

