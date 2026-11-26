"""Browser-system batch 2 (#196-#203): one class per knob, both arms pinned."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from searchio.config import Settings, set_settings
from tests._ab_baseline import BASELINE
from searchio.errors import TransientError
from searchio.models import Doc, FetchResult
from searchio.net import pdf as pdf_mod
from searchio.net import persona
from searchio.net.embedded import embedded_state_html
from searchio.net.ladder import ACCEPT_MD, Ladder
from searchio.paging import focus_passages, page_links, window
from searchio.swarm import worker as worker_mod
from searchio.swarm.tools import ToolBridge, tool_defs


@pytest.fixture
def settings(tmp_path):
    s = Settings(
        state_dir=tmp_path, cache_enabled=False, robots_policy="off", max_tier=2,
        per_domain_rps=1000.0, per_domain_burst=1000, sidecar_autostart=False,
        **BASELINE,
    )
    yield s
    set_settings(None)


ARTICLE = " ".join(f"The state blob holds sentence number {i} of the article body." for i in range(30))


def next_shell(state: dict) -> str:
    return ('<html><head><title>Shell &amp; Co</title></head><body><div id="__next"></div>'
            '<script id="__NEXT_DATA__" type="application/json">' + json.dumps(state)
            + "</script><script src=\"/app.js\"></script></body></html>")


class TestEmbeddedStateExtraction:
    def test_next_data_prose_becomes_an_article(self):
        html = next_shell({"props": {"pageProps": {"post": {
            "title": "A headline", "body": f"<p>{ARTICLE}</p>", "slug": "a-headline"}}},
            "buildId": "x" * 60})
        out = embedded_state_html(html)
        assert "sentence number 17" in out and "<h2>A headline</h2>" in out
        assert "<title>Shell &amp; Co</title>" in out
        assert "xxxxxxxx" not in out and "a-headline" not in out  # ids are not prose
        assert 'content="embedded-state:__NEXT_DATA__"' in out

    def test_window_assignment_is_read_with_raw_decode(self):
        html = ("<html><body><div id=root></div><script>window.__PRELOADED_STATE__ = "
                + json.dumps({"entities": {"a1": {"text": ARTICLE}}})
                + ";window.other = function(){};</script></body></html>")
        assert "sentence number 3" in embedded_state_html(html)

    def test_no_state_too_little_state_and_non_json_all_yield_nothing(self):
        assert embedded_state_html("<html><body><div id=root></div></body></html>") == ""
        assert embedded_state_html(next_shell({"props": {"pageProps": {"t": "short"}}})) == ""
        assert embedded_state_html(
            "<script>window.__NUXT__ = (function(a){return {x:a}})(1)</script>") == ""
        assert embedded_state_html('<script id="__NEXT_DATA__">{not json</script>') == ""

    def test_translation_catalogs_and_repeats_are_dropped(self):
        html = next_shell({"props": {"pageProps": {
            "messages": {"k": ARTICLE},  # an i18n catalog is not the page
            "a": {"text": ARTICLE}, "b": {"text": ARTICLE}}}})
        out = embedded_state_html(html)
        assert out.count("sentence number 5 ") == 1


class _ShellLadder(Ladder):
    def __init__(self, *a, body="", **kw):
        super().__init__(*a, **kw)
        self._body = body
        self.ran: list[int] = []

    async def _tier0(self, url, *, referer=""):
        self.ran.append(0)
        return 200, {}, self._body, "text/html", url

    async def _tier1(self, url, *, referer=""):
        self.ran.append(1)
        return 200, {}, self._body, "text/html", url


class _Sidecar:
    available = True

    def __init__(self, body):
        self.calls = 0
        self._body = body

    async def fetch(self, url, **kw):
        self.calls += 1
        return {"ok": True, "status": 200, "html": self._body, "url": url}

    async def goto(self, url, **kw):
        return {"ok": True, "status": 200}

    async def cookies(self):
        return []

    async def user_agent(self):
        return ""

    async def close_tab(self, *a, **kw):
        return None

    async def close(self):
        return None


class TestEmbeddedStateRung:
    SHELL = next_shell({"props": {"pageProps": {"post": {"title": "T", "body": ARTICLE}}}})

    async def test_off_the_shell_climbs_to_the_browser(self, settings):
        settings.sidecar_challenge = "engine"
        settings.sidecar_engine = True
        sc = _Sidecar(self.SHELL)
        lad = _ShellLadder(settings, sidecar=sc, body=self.SHELL)
        try:
            try:
                res = await lad.fetch("https://shell.example/a", use_cache=False)
                assert "sentence number 17" not in res.body
            except TransientError:
                pass
            assert sc.calls >= 1
        finally:
            await lad.close()

    async def test_on_tier0_answers_from_the_state(self, settings):
        settings.embedded_state_rung = True
        sc = _Sidecar(self.SHELL)
        lad = _ShellLadder(settings, sidecar=sc, body=self.SHELL)
        try:
            res = await lad.fetch("https://shell.example/a", use_cache=False)
            assert res.tier == 0 and "sentence number 17" in res.body
            assert res.rendered is False and sc.calls == 0 and lad.ran == [0]
            assert res.escalations[-1] == "tier0:embedded_state"
            assert lad.profiles.get("shell.example").min_tier == 0
        finally:
            await lad.close()

    async def test_on_a_shell_without_state_still_climbs(self, settings):
        settings.embedded_state_rung = True
        settings.sidecar_engine = True
        settings.sidecar_challenge = "engine"
        bare = '<html><body><div id="root"></div><script src="/a.js"></script></body></html>'
        sc = _Sidecar(bare)
        lad = _ShellLadder(settings, sidecar=sc, body=bare)
        try:
            with pytest.raises(Exception):
                await lad.fetch("https://shell.example/b", use_cache=False)
            assert sc.calls >= 1
        finally:
            await lad.close()


class TestPersonaKnobs:
    def teardown_method(self):
        persona._REFRESH[0] = False
        persona._HOST_OS[0] = False

    def test_off_the_pool_is_the_four_rows(self):
        assert persona.active_personas() == persona.PERSONAS
        assert persona.by_name("safari-mac").impersonate == "safari17_0"

    def test_refresh_moves_safari_to_a_current_target(self):
        persona._REFRESH[0] = True
        saf = next(p for p in persona.active_personas() if p.name == "safari-mac")
        assert saf.impersonate == "safari260" and "Version/26.0" in saf.ua
        assert saf.sec_ch_ua == ""  # Safari still sends no client hints
        assert len(persona.active_personas()) == 4

    def test_host_os_keeps_only_this_machines_family(self, monkeypatch):
        persona._HOST_OS[0] = True
        monkeypatch.setattr(persona, "_host_platform", lambda: "Windows")
        assert {p.platform for p in persona.active_personas()} == {"Windows"}
        for dom in ("a.example", "b.example", "c.example", "d.example", "e.example"):
            assert persona.for_domain(dom).platform == "Windows"
        monkeypatch.setattr(persona, "_host_platform", lambda: "")  # Linux: no row
        assert persona.active_personas() == persona.PERSONAS

    async def test_the_ladder_sets_the_switches(self, settings):
        settings.persona_refresh = True
        lad = Ladder(settings, sidecar=_Sidecar(""))
        try:
            assert persona._REFRESH[0] is True and persona._HOST_OS[0] is False
        finally:
            await lad.close()


class TestSingleflightVariantKey:
    """Not a knob: a plain fetch must never join an in-flight fetch of
    another variant (it used to receive the markdown-negotiated body)."""

    async def test_plain_and_markdown_fetches_do_not_share(self, settings):
        gate = asyncio.Event()
        seen: list[bool] = []

        class Slow(Ladder):
            async def _tier0(self, url, *, referer=""):
                md = bool(ACCEPT_MD.get())
                seen.append(md)
                await gate.wait()
                body = ("# markdown source\n\n" if md else "<html><body><p>") + "prose words " * 60
                return 200, {}, body, "text/markdown" if md else "text/html", url

        lad = Slow(settings, sidecar=_Sidecar(""))

        async def as_md():
            ACCEPT_MD.set(True)
            return await lad.fetch("https://v.example/p", use_cache=False, max_tier=0)

        try:
            t_md = asyncio.create_task(as_md())
            await asyncio.sleep(0.05)
            t_plain = asyncio.create_task(
                lad.fetch("https://v.example/p", use_cache=False, max_tier=0))
            await asyncio.sleep(0.05)
            gate.set()
            r_md, r_plain = await asyncio.gather(t_md, t_plain)
            assert sorted(seen) == [False, True]
            assert r_md.body.startswith("# markdown") and r_plain.body.startswith("<html>")
        finally:
            await lad.close()


LONG = "\n\n".join(
    f"Paragraph {i} talks about ordinary filler matters at some length here."
    + (" The reactor output was 42 megawatts in March." if i == 300 else "")
    for i in range(400))


class TestPagingHelpers:
    def test_window_continues_exactly_and_cuts_on_a_paragraph(self):
        chunk, total, nxt = window(LONG, 0, 3000)
        assert total == len(LONG) and nxt is not None and chunk.endswith("\n\n")
        chunk2, _, nxt2 = window(LONG, nxt, 3000)
        assert LONG[: nxt2] == chunk + chunk2
        tail, _, end = window(LONG, len(LONG) - 50, 3000)
        assert end is None and LONG.endswith(tail)
        assert window(LONG, 10**9, 100) == ("", len(LONG), None)

    def test_focus_finds_the_needle_with_its_offset(self):
        passages, matched = focus_passages(LONG, "reactor megawatts output", 2000)
        assert matched == 1 and "42 megawatts" in passages
        off = int(passages[2:passages.index("]")])
        assert LONG[off:].startswith("Paragraph 300 ")
        assert focus_passages(LONG, "zebra xylophone", 2000) == ("", 0)
        assert focus_passages(LONG, "the of and", 2000) == ("", 0)  # stop words only

    def test_links_are_absolute_content_first_and_deduplicated(self):
        html = ('<html><body><nav><a href="/home">Home</a></nav><main>'
                '<a href="ref/1#frag">First ref</a><a href="ref/1">dup</a>'
                '<a href="javascript:void(0)">js</a><a href="/logo.png">img</a>'
                '<a href="https://other.example/x">Other</a></main></body></html>')
        links = page_links(html, "https://site.example/docs/page", 10)
        assert links == [
            {"text": "First ref", "url": "https://site.example/docs/ref/1"},
            {"text": "Other", "url": "https://other.example/x"},
            {"text": "Home", "url": "https://site.example/home"},
        ]
        assert page_links(html, "https://site.example/docs/page", 1) == links[:1]


class _Eng:
    """Engine surface for ToolBridge: a ladder fetch and an Engine.read."""

    def __init__(self, s, body):
        self.s = s
        self.router = SimpleNamespace()
        self.calls: list[str] = []
        self._body = body
        self.ladder = SimpleNamespace(fetch=self._fetch)

    async def _fetch(self, url, **kw):
        self.calls.append("ladder")
        return FetchResult(url=url, final_url=url, status=200, body=self._body, tier=0, via="http")

    async def read(self, url):
        self.calls.append("engine")
        return Doc(url=url, text="LEAN TEXT from the engine read path. " * 5,
                   fetched_via="http+lean",
                   meta={"links": [{"text": "Next page", "url": "https://a.example/2"}]})


PAGE = ("<html><body><article>" + "".join(f"<p>{p}</p>" for p in LONG.split("\n\n"))
        + '<p><a href="/cited">The cited source</a></p></article></body></html>')


class TestSwarmReadKnobs:
    async def test_off_reads_through_the_bare_ladder_and_truncates_blind(self, settings):
        eng = _Eng(settings, PAGE)
        text, err = await ToolBridge(eng).dispatch("read_page", {"url": "https://a.example/1"})
        assert not err and eng.calls == ["ladder"]
        assert "truncated to 12000" in text and "offset=" not in text
        assert "42 megawatts" not in text and "Links on this page" not in text

    async def test_via_engine_uses_engine_read(self, settings):
        settings.swarm_read_via_engine = True
        eng = _Eng(settings, PAGE)
        text, err = await ToolBridge(eng).dispatch("read_page", {"url": "https://a.example/1"})
        assert not err and eng.calls == ["engine"]
        assert "LEAN TEXT" in text and "fetched via http+lean" in text

    async def test_via_engine_never_takes_the_rendered_tool(self, settings):
        settings.swarm_read_via_engine = True
        eng = _Eng(settings, PAGE)
        await ToolBridge(eng).dispatch("read_page_rendered", {"url": "https://a.example/1"})
        assert eng.calls == ["ladder"]

    async def test_paging_names_the_continuation_and_focus_finds_the_needle(self, settings):
        settings.read_paging = True
        eng = _Eng(settings, PAGE)
        bridge = ToolBridge(eng)
        first, _ = await bridge.dispatch("read_page", {"url": "https://a.example/1"})
        assert "chars 0-" in first and "42 megawatts" not in first
        nxt = int(first.rsplit("offset=", 1)[1].split(" ")[0])
        second, _ = await bridge.dispatch("read_page", {"url": "https://a.example/1", "offset": nxt})
        assert f"chars {nxt}-" in second
        hit, _ = await bridge.dispatch(
            "read_page", {"url": "https://a.example/1", "focus": "reactor megawatts"})
        assert "42 megawatts" in hit and "[@" in hit and len(hit) < 2000
        miss, _ = await bridge.dispatch(
            "read_page", {"url": "https://a.example/1", "focus": "zebra"})
        assert "no paragraph mentions 'zebra'" in miss and "Paragraph 0 " in miss

    async def test_links_are_listed_when_on(self, settings):
        settings.read_links = True
        eng = _Eng(settings, PAGE)
        text, _ = await ToolBridge(eng).dispatch("read_page", {"url": "https://a.example/1"})
        assert "Links on this page:\n- The cited source -- https://a.example/cited" in text

    def test_the_schema_grows_only_when_paging_is_on(self, settings):
        set_settings(settings)
        props = next(t for t in tool_defs() if t["name"] == "read_page")["input_schema"]["properties"]
        assert set(props) == {"url"}
        settings.read_paging = True
        props = next(t for t in tool_defs() if t["name"] == "read_page")["input_schema"]["properties"]
        assert set(props) == {"url", "offset", "focus"}


class _Bridge:
    def __init__(self):
        self.calls = 0
        self.report = None
        self.order: list[str] = []
        self.live = 0
        self.peak = 0

    async def dispatch(self, name, inp):
        self.calls += 1
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0.03)
        self.live -= 1
        self.order.append(name)
        if name == "submit_report":
            self.report = "REPORT"
        return f"ran {name}", False


def _tu(name):
    return SimpleNamespace(name=name, input={}, id=name)


class TestParallelTools:
    async def test_research_calls_overlap_and_the_report_runs_last(self):
        b = _Bridge()
        uses = [_tu("read_page"), _tu("submit_report"), _tu("web_search")]
        cls = next(v for v in vars(worker_mod).values()
                   if isinstance(v, type) and hasattr(v, "_dispatch_parallel"))
        out = await cls._dispatch_parallel(b, uses, 10)
        assert out[0] == ("ran read_page", False)
        assert out[1] == ("ran submit_report", False)
        assert out[2][1] is True and "not executed" in out[2][0]  # after the report
        assert b.order == ["read_page", "submit_report"]

    async def test_the_budget_is_counted_in_turn_order_before_anything_runs(self):
        b = _Bridge()
        b.calls = 2
        uses = [_tu("a"), _tu("b"), _tu("c"), _tu("d")]
        cls = next(v for v in vars(worker_mod).values()
                   if isinstance(v, type) and hasattr(v, "_dispatch_parallel"))
        out = await cls._dispatch_parallel(b, uses, 4)
        assert [o[1] for o in out] == [False, False, True, True]
        assert "exhausted" in out[2][0] and b.calls == 4 and b.peak == 2


class TestPdfFastExtract:
    def teardown_method(self):
        pdf_mod._FAST[0] = False

    def test_both_extractors_read_the_same_pages(self):
        pytest.importorskip("pypdfium2")
        data = pdf_mod.build_pdf(["alpha page one", "beta page two", "gamma page three"])
        slow = pdf_mod.extract_text(data, max_pages=2)
        pdf_mod._FAST[0] = True
        fast = pdf_mod.extract_text(data, max_pages=2)
        assert slow[1:] == fast[1:] == (2, 3)
        assert "alpha page one" in fast[0] and "beta page two" in fast[0]
        assert "gamma" not in fast[0]

    def test_fast_refuses_rot_the_same_way(self):
        pytest.importorskip("pypdfium2")
        pdf_mod._FAST[0] = True
        with pytest.raises(pdf_mod.PdfUnreadable):
            pdf_mod.extract_text(b"%PDF-1.4 this is not a pdf at all")
