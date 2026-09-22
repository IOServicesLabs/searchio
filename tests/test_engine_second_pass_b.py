"""engine.py second pass, part B (iteration 60, DeepSeek): find_items and
the caller-supplied scope. Every test here bit RED before its fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.errors import ProviderError


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False)


def _engine(settings, providers: dict):
    from searchio.engine import Engine

    eng = Engine.__new__(Engine)
    eng.s = settings
    eng.ladder = SimpleNamespace(session_id="t")
    eng.router = None
    eng.registry = SimpleNamespace(get=lambda name: providers.get(name))
    return eng


class TestFallbackHonestEmptyStaysEmpty:
    async def test_cheap_failed_but_browser_answered_nothing(self, settings):
        # Bug 132 (DeepSeek engine #1, bug 120's second edge): the cheap path
        # refused and the browser fallback ANSWERED with zero listings -- a
        # provider did answer, so the result is an honest empty, not the
        # cheap path's refusal.
        class Broken:
            async def find_items(self, q, ctx):
                raise ProviderError("marketplace", "discovery failed: {'duckduckgo': 'circuit open'}")

        class Empty:
            async def find_items(self, q, ctx):
                return []
        settings.max_tier = 2
        eng = _engine(settings, {"marketplace": Broken(), "sidecar_listings": Empty()})
        assert await eng.find_items("zxqwv widget") == []

    async def test_both_refused_is_still_a_refusal(self, settings):
        class Broken:
            async def find_items(self, q, ctx):
                raise ProviderError("marketplace", "discovery failed")
        settings.max_tier = 2
        eng = _engine(settings, {"marketplace": Broken(), "sidecar_listings": Broken()})
        with pytest.raises(ProviderError):
            await eng.find_items("zxqwv widget")


class TestFindItemsHonoursInlineOperators:
    async def test_inline_site_becomes_the_domain_scope(self, settings):
        # Bug 133 (DeepSeek engine #2): find_items("... site:bestbuy.com")
        # left the operator in the text; discovery searched it as words and
        # the domain scope stayed empty -- bug 12/14's class on the third
        # entry point.
        seen = {}

        class Cap:
            async def find_items(self, q, ctx):
                seen["q"] = q
                return []
        settings.max_tier = 1
        await _engine(settings, {"marketplace": Cap()}).find_items("sony wh-1000xm5 site:bestbuy.com")
        assert seen["q"].text == "sony wh-1000xm5" and seen["q"].domains == ["bestbuy.com"]


class TestCallerDomainsAreNormalized:
    async def test_search_normalizes_urls_and_www(self, settings):
        # Bug 134 (DeepSeek engine #4): domains=["https://www.nasa.gov/"]
        # from an agent was matched literally by the router's host filter,
        # so every result was dropped -- an honest-but-useless empty.
        from searchio.engine import Engine

        captured = {}

        class R:
            async def search(self, q, **kw):
                captured["q"] = q
                return SimpleNamespace(docs=[], used=[], failed={}, filtered={}, per_provider={}, elapsed_ms=1)
        eng = Engine.__new__(Engine)
        eng.router = R()
        await Engine.search(eng, "artemis", domains=["https://www.nasa.gov/", "ESA.int", "*.gov"],
                            exclude_domains=["http://blogs.nasa.gov/x"])
        assert captured["q"].domains == ["nasa.gov", "esa.int", "gov"]
        assert captured["q"].exclude_domains == ["blogs.nasa.gov"]

    async def test_find_items_normalizes_too(self, settings):
        seen = {}

        class Cap:
            async def find_items(self, q, ctx):
                seen["q"] = q
                return []
        settings.max_tier = 1
        await _engine(settings, {"marketplace": Cap()}).find_items("thing", domains=["https://www.bestbuy.com/"])
        assert seen["q"].domains == ["bestbuy.com"]
