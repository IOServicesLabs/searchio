"""A 429 teaches the pacing controller, not the rescue policy (iteration 81).

The 2026-09-16 realtor.com pass exposed a wiring break: the host 429'd on all
three tiers, ratelimit.py documents its AIMD contract as "crossing the line
costs one 429" -- and the limiter never heard about any of them. classify()
demoted 429 from `blocked` to an honest refusal in iteration 19 so a throttle
would not boot the challenge browser; correct for the rescue decision, but it
also routed the verdict around ladder.py's `record_block` call, which sat
inside `if verdict.blocked:`. A refusing host kept being probed at an
un-halved rate -- impolite, and the repeated-refusal pattern that gets an
egress IP noticed.

These pin the rewire: a 429 backs the domain's token bucket off (and
surrenders its burst), still buys no browser, and still names no vendor even
when the throttle response carries vendor headers.
"""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.errors import TransientError

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


THROTTLE_BODY = "<html><body>Too Many Requests</body></html>"
AKAMAI_429 = (429, {"content-type": HTML, "server": "AkamaiGHost"}, THROTTLE_BODY)


class CountingRescue(FakeLadder):
    """FakeLadder that counts tier-2 rendered (rescue) attempts."""

    def __init__(self, settings, responses):
        super().__init__(settings, responses)
        self.rendered_attempts = 0

    async def _try_tier(self, tier, url, *, referer="", rendered=False):
        if tier == 2 and rendered:
            self.rendered_attempts += 1
        return await super()._try_tier(tier, url, referer=referer, rendered=rendered)


class TestA429TeachesTheLimiter:
    async def test_throttle_halves_the_domain_rate_per_refusal(self, settings):
        lad = FakeLadder(settings, {0: AKAMAI_429, 1: AKAMAI_429, 2: AKAMAI_429})
        try:
            before = lad.limiter.rate_for("throttled.example")
            with pytest.raises(TransientError):
                await lad.fetch("https://throttled.example/x", use_cache=False)
            after = lad.limiter.rate_for("throttled.example")
            # One refusal per tier tried: 1000 -> 500 -> 250 -> 125. The
            # ratelimit contract is multiplicative on evidence, and three
            # tiers refusing is three pieces of evidence.
            assert after == pytest.approx(before * 0.5 ** 3)
        finally:
            await lad.close()

    async def test_throttle_buys_no_browser(self, settings):
        # The iteration-19 demotion must survive the rewire: a 429 is still
        # an honest refusal and must not fire the challenge rescue.
        lad = CountingRescue(_auto(settings), {0: AKAMAI_429, 1: AKAMAI_429, 2: AKAMAI_429})
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://throttled.example/y", use_cache=False)
            assert lad.rendered_attempts == 0, str(exc.value)
            assert "auto_rendered" not in str(exc.value)
        finally:
            await lad.close()

    async def test_throttle_names_no_vendor_even_with_vendor_headers(self, settings):
        # Akamai marks its throttles, but a rate limit is not a WAF verdict:
        # the exhaustion error must not name a vendor off a 429.
        lad = FakeLadder(settings, {0: AKAMAI_429, 1: AKAMAI_429, 2: AKAMAI_429})
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://throttled.example/z", use_cache=False)
            assert "akamai" not in str(exc.value).lower()
            assert "http_429" in str(exc.value)
        finally:
            await lad.close()

    async def test_success_still_probes_the_rate_up(self, settings):
        # The other AIMD half, pinned so the 429 path can't leak into the
        # success path: a clean answer raises the rate additively.
        lad = FakeLadder(settings, {
            0: (200, {"content-type": HTML},
                "<html><body>" + "real page content a reader wants. " * 20 + "</body></html>"),
        })
        try:
            before = lad.limiter.rate_for("clean.example")
            await lad.fetch("https://clean.example/x", use_cache=False)
            after = lad.limiter.rate_for("clean.example")
            assert after == pytest.approx(min(before + 0.05, 1000.0))
        finally:
            await lad.close()
