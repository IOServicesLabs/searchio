"""Browser batch 5 (#217+). One class per knob; knob off pins the old
behaviour, knob on pins the new one."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.engine import Engine
from searchio.models import FetchResult
from searchio.net import sidecar as sidecar_mod
from searchio.net.ladder import Ladder
from tests._ab_baseline import BASELINE

ARTICLE = "<html><body><article><p>" + "The rendered page lists every result in full. " * 40 \
    + "</p></article></body></html>"
#: A script-built page: megabytes of application data, a footer of markup.
HEAVY_SHELL = ("<html><head><title>Results</title></head><body><div id=\"app\"></div>"
               "<script>var data = \"" + "x" * 300_000 + "\";</script>"
               "<footer><p>About Press Copyright Contact us Creators Advertise Developers Terms "
               "Privacy Policy and Safety How it works Test new features</p></footer></body></html>")
LIGHT_SHELL = HEAVY_SHELL.replace("x" * 300_000, "x" * 500)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path, cache_enabled=False, robots_policy="off", max_tier=2,
        per_domain_rps=1000.0, per_domain_burst=1000, sidecar_autostart=False,
        clearance_enabled=False, sidecar_challenge_auto=False, **BASELINE,
    )


class TestEngineTlsImpersonate:
    """#217: the knob only sets the spawned engine's environment; the
    transport itself is pinned by se-net's impersonate_tests."""

    def test_the_engine_environment_follows_the_knob(self, settings):
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_TLS_IMPERSONATE")
        Ladder(settings.model_copy(update={"engine_tls_impersonate": True}))
        assert sidecar_mod._ENGINE_ENV["SE_TLS_IMPERSONATE"] == "1"
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_TLS_IMPERSONATE")


class _Fetches:
    """Stands in for Ladder.fetch: tier 0 serves ``first``; a rendered ask
    serves ``rendered`` (or raises it)."""

    def __init__(self, first: str, rendered):
        self.first, self.rendered, self.calls = first, rendered, []

    async def __call__(self, url, **kw):
        self.calls.append(kw)
        if kw.get("rendered"):
            if isinstance(self.rendered, Exception):
                raise self.rendered
            return FetchResult(url=url, final_url=url, status=200, body=self.rendered,
                               tier=2, via="browser", rendered=True)
        return FetchResult(url=url, final_url=url, status=200, body=self.first, tier=0, via="http")


class TestThinHeavyRescue:
    URL = "https://shell.example/results?q=x"

    async def _read(self, settings, first, rendered, **knobs):
        eng = Engine(settings.model_copy(update=knobs))
        fetches = _Fetches(first, rendered)
        eng.ladder.fetch = fetches
        try:
            doc = await eng.read(self.URL)
        finally:
            await eng.close()
        return doc, fetches

    async def test_off_serves_the_footer_as_the_page(self, settings):
        doc, f = await self._read(settings, HEAVY_SHELL, ARTICLE)
        assert len(doc.text) < 300
        assert "thin_heavy" not in doc.meta
        assert not any(c.get("rendered") for c in f.calls)

    async def test_heavy_thin_read_gets_one_browser_pass(self, settings):
        doc, f = await self._read(settings, HEAVY_SHELL, ARTICLE, thin_heavy_rescue=True)
        assert "rendered page lists every result" in doc.text
        assert doc.meta["thin_heavy"] == "rescued"
        assert "/" in doc.meta["thin_body"]
        assert doc.fetched_via == "browser"
        assert sum(1 for c in f.calls if c.get("rendered")) == 1
        # The pass names the browser tier: tier 0 judged the page ok, so
        # `rendered` alone would be answered by tier 0 again.
        assert [c.get("force_tier") for c in f.calls if c.get("rendered")] == [2]

    async def test_a_render_that_is_no_better_keeps_the_first_read(self, settings):
        doc, f = await self._read(settings, HEAVY_SHELL, HEAVY_SHELL, thin_heavy_rescue=True)
        assert doc.meta["thin_heavy"] == "still_thin"
        assert doc.fetched_via == "http"
        assert sum(1 for c in f.calls if c.get("rendered")) == 1

    async def test_a_failed_render_is_stamped_not_raised(self, settings):
        doc, _ = await self._read(settings, HEAVY_SHELL, RuntimeError("no browser"),
                                  thin_heavy_rescue=True)
        assert doc.meta["thin_heavy"] == "rescue_failed:RuntimeError"
        assert len(doc.text) < 300

    async def test_a_small_thin_page_is_left_alone(self, settings):
        doc, f = await self._read(settings, LIGHT_SHELL, ARTICLE, thin_heavy_rescue=True)
        assert "thin_heavy" not in doc.meta
        assert not any(c.get("rendered") for c in f.calls)

    async def test_a_readable_heavy_page_is_left_alone(self, settings):
        heavy_article = ARTICLE.replace("</body>", "<script>" + "y" * 300_000 + "</script></body>")
        doc, f = await self._read(settings, heavy_article, ARTICLE, thin_heavy_rescue=True)
        assert "thin_heavy" not in doc.meta
        assert not any(c.get("rendered") for c in f.calls)

    async def test_no_browser_tier_means_no_rescue(self, settings):
        doc, f = await self._read(settings, HEAVY_SHELL, ARTICLE, thin_heavy_rescue=True, max_tier=1)
        assert "thin_heavy" not in doc.meta
        # ... but the thin read is still stamped "<chars>/<bytes>".
        chars, size = doc.meta["thin_body"].split("/")
        assert int(chars) < 300 and int(size) >= 200_000
        assert not any(c.get("rendered") for c in f.calls)


async def _render_params(settings, **knobs):
    """The parameters the engine render rung sends for a shell page."""
    from tests.test_browser_ab_3 import SHELL, NoHttpLadder, _Engine, _with

    eng = _Engine(page=SHELL)
    lad = NoHttpLadder(_with(settings, engine_render_rung=True, engine_render_prelude=True, **knobs),
                       sidecar=eng)
    res = await lad.fetch("https://spa.example/live", use_cache=False)
    await lad.close()
    assert "readable prose" in res.body
    return [p for v, p in eng.calls if v == "fetch"][0]


class TestEngineFetchPool:
    """#219: the render asks for pooled, pump-settled page fetches only when
    the knob is on; the engine side is pinned by se-serve's fetch_pool_tests."""

    async def test_off_the_param_is_not_sent(self, settings):
        assert "fetch_pool" not in await _render_params(settings)

    async def test_on_the_render_asks_for_the_pool(self, settings):
        assert (await _render_params(settings, engine_fetch_pool=True))["fetch_pool"] is True


class TestEngineModulesAndCodeCache:
    """#220 / #221: render options sent only when their knob is on; the engine
    side is pinned by se-serve's modules_tests."""

    async def test_off_neither_param_is_sent(self, settings):
        sent = await _render_params(settings)
        assert "modules" not in sent and "code_cache" not in sent

    async def test_each_knob_sends_its_own_param(self, settings):
        sent = await _render_params(settings, engine_modules=True)
        assert sent["modules"] is True and "code_cache" not in sent
        sent = await _render_params(settings, engine_code_cache=True)
        assert sent["code_cache"] is True and "modules" not in sent


class TestEngineRenderQuiet:
    """#222: the quiet window travels as `quiet_ms`; 0 sends nothing."""

    async def test_off_no_param(self, settings):
        assert "quiet_ms" not in await _render_params(settings)

    async def test_on_sends_the_window(self, settings):
        sent = await _render_params(settings, engine_render_quiet_ms=500)
        assert sent["quiet_ms"] == 500


class TestEngineProcessOptions:
    """#221 / #223: options that reach the engine through its environment."""

    def test_isolates_only_above_one(self, settings):
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_JS_ISOLATES")
        Ladder(settings.model_copy(update={"engine_isolates": 4}))
        assert sidecar_mod._ENGINE_ENV["SE_JS_ISOLATES"] == "4"
        Ladder(settings.model_copy(update={"engine_isolates": 99}))
        assert sidecar_mod._ENGINE_ENV["SE_JS_ISOLATES"] == "8"
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_JS_ISOLATES")

    def test_code_cache_dir_lives_in_the_state_dir(self, settings):
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_CODE_CACHE_DIR")
        on = settings.model_copy(update={"engine_code_cache": True})
        Ladder(on)
        assert sidecar_mod._ENGINE_ENV["SE_CODE_CACHE_DIR"] == str(on.state_dir / "engine-code-cache")
        Ladder(settings)
        assert not sidecar_mod._ENGINE_ENV.get("SE_CODE_CACHE_DIR")
