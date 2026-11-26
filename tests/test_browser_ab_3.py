"""Engine-side knobs as the ladder drives them (#204-#209).

One class per knob. Knob off pins the request the engine received before the
knob existed; knob on pins the new request and what the ladder does with the
answer. The engine itself is covered by its own Rust tests.
"""

from __future__ import annotations

import base64

import pytest

from searchio.config import Settings
from tests._ab_baseline import BASELINE
from searchio.errors import TransientError
from searchio.net import pdf as pdf_mod
from searchio.net.ladder import Ladder

GOOD = "<html><body><article><p>" + ("readable prose sentence here. " * 60) + "</p></article></body></html>"
SHELL = "<html><head><title>App</title></head><body><div id=\"root\"></div><script>boot()</script></body></html>"
PDF = pdf_mod.build_pdf(["Members report page one. " + "steady growth in traffic. " * 20])


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path, cache_enabled=False, robots_policy="off", max_tier=2,
        per_domain_rps=1000.0, per_domain_burst=1000, sidecar_autostart=False,
        clearance_enabled=False, sidecar_challenge_auto=False, **{**BASELINE, "clearance_capture_bg": True},
    )


class _Engine:
    """Engine SidecarClient stand-in: records every request it is sent."""

    available = True
    binary = "se-serve"

    def __init__(self, page=GOOD, rendered=GOOD, envelope=None):
        self.page, self.rendered, self.envelope = page, rendered, envelope or {}
        self.fetches: list[dict] = []
        self.calls: list[tuple[str, dict]] = []

    async def fetch(self, url, **kw):
        self.fetches.append({"url": url, **kw})
        return {"ok": True, "status": 200, "html": self.page, "url": url, **self.envelope}

    async def call(self, verb, params, **kw):
        self.calls.append((verb, dict(params)))
        if verb == "fetch":
            if self.rendered is None:
                return {"ok": False, "error": "render failed"}
            return {"ok": True, "status": 200, "html": self.rendered, "url": params["url"],
                    "rendered": True, "render_ms": 12}
        return {"ok": True}

    async def goto(self, url, **kw):
        return {"ok": True, "status": 200}

    async def cookies(self):
        return []

    async def user_agent(self):
        return "BrowserUA/1.0"

    async def close(self):
        return None


class NoHttpLadder(Ladder):
    async def _tier0(self, url, *, referer=""):
        raise TransientError("no network in unit tests")

    async def _tier1(self, url, *, referer=""):
        raise TransientError("no network in unit tests")


def _with(settings, **kw):
    return settings.model_copy(update=kw)


class TestEngineRequestOptions:
    async def test_off_sends_the_old_request(self, settings):
        eng = _Engine()
        lad = NoHttpLadder(settings, sidecar=eng)
        await lad.fetch("https://opts.example/a", use_cache=False)
        sent = eng.fetches[0]
        assert "extra" not in sent
        assert sent["timeout_ms"] is None  # the engine never honoured it before
        await lad.close()

    async def test_binary_and_scoped_ride_the_fetch(self, settings):
        eng = _Engine()
        lad = NoHttpLadder(_with(settings, engine_binary_body=True, engine_bot_scoped=True),
                           sidecar=eng)
        await lad.fetch("https://opts.example/b", use_cache=False)
        assert eng.fetches[0]["extra"] == {"binary": True, "bot_scoped": True}
        await lad.close()

    async def test_deadline_passes_the_navigation_budget(self, settings):
        eng = _Engine()
        lad = NoHttpLadder(_with(settings, engine_deadline=True, tier2_timeout_s=30.0),
                           sidecar=eng)
        await lad.fetch("https://opts.example/c", use_cache=False)
        assert eng.fetches[0]["timeout_ms"] == 10000
        await lad.close()


class TestEngineBinaryBody:
    async def test_envelope_bytes_feed_the_pdf_extract(self, settings, monkeypatch):
        eng = _Engine(page="%PDF-1.4 lossy text", envelope={
            "content_type": "application/pdf",
            "body_base64": base64.b64encode(PDF).decode()})
        lad = NoHttpLadder(_with(settings, engine_binary_body=True), sidecar=eng)

        async def no_refetch(url):
            raise AssertionError("the PDF was downloaded a second time")

        monkeypatch.setattr(lad, "_fetch_bytes", no_refetch)
        res = await lad.fetch("https://files.example/r.pdf", use_cache=False)
        assert "Members report page one" in res.body
        assert lad.stats()["tiers"].get("engine_binary_body") == 1
        await lad.close()

    async def test_bad_base64_is_not_a_failed_fetch(self, settings):
        eng = _Engine(envelope={"body_base64": "!!not base64!!"})
        lad = NoHttpLadder(_with(settings, engine_binary_body=True), sidecar=eng)
        res = await lad.fetch("https://files.example/page", use_cache=False)
        assert "readable prose" in res.body
        await lad.close()


class TestEngineRenderRung:
    async def test_off_never_asks_for_a_render(self, settings):
        eng = _Engine(page=SHELL)
        lad = NoHttpLadder(settings, sidecar=eng)
        with pytest.raises(Exception):
            await lad.fetch("https://spa.example/a", use_cache=False)
        assert not [c for c in eng.calls if c[0] == "fetch"]
        await lad.close()

    async def test_shell_is_answered_by_one_engine_render(self, settings):
        eng = _Engine(page=SHELL)
        lad = NoHttpLadder(_with(settings, engine_render_rung=True, engine_render_quiesce=3,
                                 engine_page_referer=True, engine_render_prelude=True,
                                 engine_script_pipeline=True),
                           sidecar=eng)
        res = await lad.fetch("https://spa.example/b", use_cache=False)
        assert "readable prose" in res.body and res.tier == 2 and res.rendered is False
        assert "tier2:engine_rendered" in res.escalations
        renders = [p for v, p in eng.calls if v == "fetch"]
        assert len(renders) == 1
        assert renders[0]["render"] is True and renders[0]["quiesce"] == 3
        assert renders[0]["page_referer"] is True and renders[0]["settle_ms"] == 2500
        assert renders[0]["prelude"] is True and renders[0]["script_pipeline"] is True
        # The render tab is closed, and it is not the fetch tab.
        assert ("close_tab", {"tab_id": renders[0]["tab_id"]}) in eng.calls
        assert lad.stats()["tiers"].get("engine_render_ok") == 1
        await lad.close()

    async def test_a_render_that_stays_thin_is_a_miss_tried_once(self, settings):
        eng = _Engine(page=SHELL, rendered=SHELL)
        lad = NoHttpLadder(_with(settings, engine_render_rung=True), sidecar=eng)
        with pytest.raises(Exception) as err:
            await lad.fetch("https://spa.example/c", use_cache=False)
        assert "engine_render_miss" in str(err.value)
        sent = [p for v, p in eng.calls if v == "fetch"][0]
        assert "script_pipeline" not in sent
        assert "prelude" not in sent and "quiesce" not in sent and "page_referer" not in sent
        assert len([c for c in eng.calls if c[0] == "fetch"]) == 1
        await lad.close()

    async def test_a_failed_render_is_a_miss_not_an_error(self, settings):
        eng = _Engine(page=SHELL, rendered=None)
        lad = NoHttpLadder(_with(settings, engine_render_rung=True), sidecar=eng)
        with pytest.raises(TransientError):
            await lad.fetch("https://spa.example/d", use_cache=False)
        assert lad.stats()["tiers"].get("engine_render_miss", 0) >= 1
        await lad.close()

    async def test_a_readable_page_is_never_rendered(self, settings):
        eng = _Engine(page=GOOD)
        lad = NoHttpLadder(_with(settings, engine_render_rung=True), sidecar=eng)
        await lad.fetch("https://spa.example/e", use_cache=False)
        assert not [c for c in eng.calls if c[0] == "fetch"]
        await lad.close()


class TestMetaRefreshStrict:
    COMMENTED = ('<html><head><!-- <meta http-equiv="refresh" content="0; url=/old"> -->'
                 "<title>t</title></head><body>x</body></html>")
    SCRIPTED = ("<html><head><script>if (legacy) document.write('<meta http-equiv=\\\"refresh\\\" "
                "content=\\\"0; url=/old\\\">');</script></head><body>x</body></html>")
    REAL = '<html><head><!-- note --><meta http-equiv="refresh" content="0; url=/new"></head></html>'

    def _target(self, html, strict):
        from searchio.net import ladder as lm
        lm._META_STRICT[0] = strict
        try:
            return lm._meta_refresh_target(200, "text/html", html, "https://m.example/a")
        finally:
            lm._META_STRICT[0] = False

    def test_off_follows_text_that_is_not_a_tag(self):
        assert self._target(self.COMMENTED, False) == "https://m.example/old"

    def test_on_ignores_comments_and_script_bodies(self):
        assert self._target(self.COMMENTED, True) is None
        assert self._target(self.SCRIPTED, True) is None

    def test_on_still_follows_a_real_tag(self):
        assert self._target(self.REAL, True) == "https://m.example/new"

    def test_mask_keeps_offsets(self):
        from searchio.net.ladder import _mask_non_markup
        for doc in (self.COMMENTED, self.SCRIPTED, "<!-- open <meta x>", "<script>never closed"):
            assert len(_mask_non_markup(doc)) == len(doc)


class TestEngineProxy:
    def test_the_engine_gets_the_proxy_only_when_on(self, settings):
        from searchio.net import sidecar as sc
        NoHttpLadder(_with(settings, proxy="http://p.example:8080"), sidecar=_Engine())
        assert sc._ENGINE_PROXY[0] == ""
        NoHttpLadder(_with(settings, proxy="http://p.example:8080", engine_proxy=True),
                     sidecar=_Engine())
        assert sc._ENGINE_PROXY[0] == "http://p.example:8080"
        sc._ENGINE_PROXY[0] = ""
