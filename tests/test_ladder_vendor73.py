"""A block's vendor survives the climb, and a bare block does not buy a
browser (iteration 73, from api.read_blocked_451 taking 80 s to answer
"blocked by unknown" on a Cloudflare-style control page). Red first."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.errors import Blocked, TransientError

from tests.test_ladder_router import HTML, FakeLadder


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                    max_tier=2, per_domain_rps=1000.0, per_domain_burst=1000,
                    sidecar_autostart=False)


def _auto(settings):
    settings.sidecar_engine = True
    settings.sidecar_challenge = "patchright"
    settings.sidecar_challenge_auto = True
    return settings


CF_HEADERS = {"server": "cloudflare", "cf-mitigated": "challenge", "content-type": HTML}
BARE = {"content-type": HTML}
WALL_BODY = "<!DOCTYPE html><title>Attention Required</title>"


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


class TestTheVendorSurvivesTheClimb:
    async def test_a_vendor_named_at_tier_zero_is_the_final_answers_vendor(self, settings):
        # Bug 166: tier 0 and tier 1 read "server: cloudflare, cf-mitigated:
        # challenge" and classify named the vendor; the engine's fetch
        # envelope carries no headers, so tier 2's verdict had none -- and
        # the Blocked the caller got was the LAST tier's: "blocked by
        # unknown" for a wall two tiers had already identified.
        settings.sidecar_challenge_auto = False
        lad = FakeLadder(settings, {
            0: (403, dict(CF_HEADERS), WALL_BODY),
            1: (403, dict(CF_HEADERS), WALL_BODY),
            2: (403, dict(BARE), WALL_BODY),
        })
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://walled.example/x", use_cache=False)
            assert exc.value.vendor == "cloudflare", (exc.value.vendor, str(exc.value))
            assert lad.attempts == [0, 1, 2]
        finally:
            await lad.close()


class TestABareBlockDoesNotBuyTheBrowser:
    async def test_a_403_nobody_can_name_skips_the_rescue(self, settings):
        # Bug 167: every tier-2 block bought the one-shot patchright rescue
        # -- a bare 403 with a 48-byte body and no vendor at ANY tier
        # included. The browser cannot change an origin's mind about a
        # plain refusal, and the rescue cost 80 s per such page live.
        lad = Rescue(_auto(settings), {
            0: (403, dict(BARE), "<html>blocked</html>"),
            1: (403, dict(BARE), "<html>blocked</html>"),
            2: (403, dict(BARE), "<html>blocked</html>"),
        }, (403, dict(BARE), "<html>blocked</html>"))
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://bare.example/x", use_cache=False)
            assert lad.rendered_attempts == 0, str(exc.value)
            assert "tier2:auto_rendered" not in str(exc.value), str(exc.value)
            assert "tier2:http_403" in str(exc.value)
        finally:
            await lad.close()

    async def test_a_vendor_seen_below_still_earns_the_rescue(self, settings):
        # The climb's vendor knowledge feeds the rescue decision too: the
        # engine's headerless tier-2 answer alone would look bare.
        lad = Rescue(_auto(settings), {
            0: (403, dict(CF_HEADERS), WALL_BODY),
            1: (403, dict(CF_HEADERS), WALL_BODY),
            2: (403, dict(BARE), WALL_BODY),
        }, (403, dict(BARE), WALL_BODY))
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://cf.example/x", use_cache=False)
            assert lad.rendered_attempts == 1, str(exc.value)
            assert "tier2:auto_rendered" in str(exc.value)
            assert exc.value.vendor == "cloudflare"
        finally:
            await lad.close()

    async def test_a_challenge_body_without_vendor_headers_still_earns_the_rescue(self, settings):
        body = ('<html><head><title>Just a moment...</title><script src="/cdn-cgi/challenge-platform/x.js">'
                '</script></head><body><div id="challenge-running"></div></body></html>')
        lad = Rescue(_auto(settings), {2: (403, dict(BARE), body)}, (403, dict(BARE), body))
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://js.example/x", force_tier=2, use_cache=False)
            assert lad.rendered_attempts == 1, str(exc.value)
        finally:
            await lad.close()

    async def test_the_engines_headers_name_the_vendor_at_tier_two(self, settings):
        # With the engine reporting the origin's headers (change #59), a
        # forced tier-2 fetch of a Cloudflare 403 knows its vendor and
        # rescues -- the shape the ctl.blocked_* rows drive live.
        lad = Rescue(_auto(settings), {2: (403, dict(CF_HEADERS), WALL_BODY)},
                     (403, dict(BARE), WALL_BODY))
        try:
            with pytest.raises(Blocked) as exc:
                await lad.fetch("https://cf2.example/x", force_tier=2, use_cache=False)
            assert lad.rendered_attempts == 1 and exc.value.vendor == "cloudflare", str(exc.value)
        finally:
            await lad.close()


class TestTheWarmupRefetchStaysOnItsTab:
    async def test_both_fetches_of_one_tier_two_pass_use_the_same_private_tab(self, settings):
        # Rider of bug 159: the warm-up refetch named no tab, so the second
        # navigation of a rescued fetch still went through "default".
        seen: list[str | None] = []

        class Rec:
            available = True
            binary = "engine"
            calls = 0

            async def fetch(self, url, **kw):
                seen.append(kw.get("tab_id"))
                Rec.calls += 1
                if Rec.calls == 1:
                    return {"ok": False, "error": "navigation failed", "status": 0}
                html = "<html><body><p>" + ("real page text " * 60) + "</p></body></html>"
                return {"ok": True, "status": 200, "html": html, "url": url}

            async def goto(self, url, **kw):
                return {"ok": True, "status": 200}

            async def call(self, verb, params=None, **kw):
                return {"ok": True}

            async def cookies(self):
                return []

            async def user_agent(self):
                return ""

            async def close(self):
                return None

        from searchio.net.ladder import Ladder

        class NoHttp(Ladder):
            async def _tier0(self, url, *, referer=""):
                raise TransientError("no network")

            async def _tier1(self, url, *, referer=""):
                raise TransientError("no network")
        s = settings
        s.max_tier = 2
        lad = NoHttp(s, sidecar=Rec())
        try:
            res = await lad.fetch("https://warm.example/deep/page", use_cache=False, force_tier=2)
            assert "real page text" in res.body
        finally:
            await lad.close()
        assert len(seen) == 2 and seen[0] == seen[1] and seen[0] not in (None, "default"), seen
