"""The agent path for the non-English and scoped web.

Iteration 56: the Muse-driven agent leg answered a German question from
en.wikipedia three times over, never scoped a nasa.gov question with site:,
and never passed freshness for a this-week question. The schema had the
knobs; the prompt never said when to reach for them, and the language knob
did not exist. Every test here bit RED before its fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.swarm.tools import ToolBridge, tool_defs
from searchio.swarm.worker import WORKER_SYSTEM


class _Eng:
    def __init__(self):
        self.s = Settings(cache_enabled=False)
        self.calls: list[dict] = []
        self.router = SimpleNamespace()
        self.ladder = SimpleNamespace()

    async def search(self, text, **kw):
        from searchio.models import Doc

        self.calls.append({"text": text, **kw})
        return SimpleNamespace(docs=[Doc(url="https://a.example/1", title="T", snippet="s", source="stub")],
                               used=["stub"], failed={}, filtered={}, per_provider={}, elapsed_ms=1)


class TestLanguageReachesTheSearch:
    async def test_web_search_locale_param_reaches_the_engine(self):
        # The web_search tool gains `locale` (e.g. "de-DE"): Bing honors it as
        # mkt/setlang; without it a German question searched the en-US web.
        props = next(t for t in tool_defs() if t["name"] == "web_search")["input_schema"]["properties"]
        assert "locale" in props, list(props)
        eng = _Eng()
        text, err = await ToolBridge(eng).dispatch("web_search", {"query": "wärmepumpe altbau", "locale": "de-DE"})
        assert not err and eng.calls[-1].get("locale") == "de-DE", eng.calls

    def test_engine_search_maps_locale_to_query_locale_and_region(self):
        from searchio.engine import Engine
        from searchio.models import Query

        captured = {}

        class R:
            async def search(self, q: Query, **kw):
                captured["q"] = q
                return SimpleNamespace(docs=[], used=[], failed={}, filtered={}, per_provider={}, elapsed_ms=1)

        eng = Engine.__new__(Engine)
        eng.router = R()
        import asyncio
        asyncio.run(Engine.search(eng, "wärmepumpe", locale="de-DE"))
        assert captured["q"].locale == "de-DE" and captured["q"].region == "de"
        asyncio.run(Engine.search(eng, "heat pump"))
        assert captured["q"].locale == "en-US" and captured["q"].region == "us"


class TestPromptNamesTheKnobs:
    def test_worker_prompt_tells_the_model_when_to_use_them(self):
        p = WORKER_SYSTEM.lower()
        assert "freshness" in p and "site:" in p and "locale" in p and "language" in p


class TestCitationsAreNotRepeated:
    async def test_identical_citations_collapse_to_one(self):
        # Rider: the German answer cited https://en.wikipedia.org/wiki/Heat_pump
        # three times in one finding -- three "sources" that are one page.
        eng = _Eng()
        bridge = ToolBridge(eng)
        await bridge.dispatch("web_search", {"query": "x"})
        u = "https://a.example/1"
        text, err = await bridge.dispatch("submit_report", {
            "summary": "s",
            "findings": [{"claim": "c", "confidence": "high",
                          "citations": [{"url": u, "title": "T"}, {"url": u, "title": "T"},
                                        {"url": u + "?utm_source=x", "title": "T"}]}],
            "gaps": []})
        assert not err and bridge.report is not None
        assert [c.url for c in bridge.report.findings[0].citations] == [u]
