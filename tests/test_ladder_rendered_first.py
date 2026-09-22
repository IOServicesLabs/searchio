"""Adaptive-class hosts open on the browser; everything else climbs as before.

The realtor.com finding (scripts/probe_realtor_patchright.py, engine ec52690,
2026-09-17): the host 429'd every cheap tier on first contact, then one
production patchright pass settled on the genuine 801KB SRP -- its gate
serves content only after browser-grade behavior. The ladder could never get
there unaided: iteration 19 makes a 429 an honest refusal that buys no
rescue, so no fetch of such a host ever reached the browser. The
``rendered_first_domains`` knob opens that class directly on the challenge
sidecar, with the normal climb as the fallback when the browser pass itself
fails.

These pin: opener ordering and rendered routing, the fallback hand-off
(blocked / transient / sidecar-unavailable), subdomain matching (and
non-matching), and that the opener stays out of the way of explicit
force_tier/rendered requests, low ceilings, and tenant-scoped fetches.
"""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.errors import Blocked, TransientError

from tests.test_ladder_router import GOOD, HTML, FakeLadder


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,  # keep tests fast; pacing is tested separately
        per_domain_burst=1000,
        sidecar_autostart=False,
        rendered_first_domains=["adaptive.example"],
    )


class RecordingLadder(FakeLadder):
    """FakeLadder that records (tier, rendered) for every attempt."""

    def __init__(self, settings, responses):
        super().__init__(settings, responses)
        self.passes: list[tuple[int, bool]] = []

    async def _try_tier(self, tier, url, *, referer="", rendered=False):
        self.passes.append((tier, rendered))
        return await super()._try_tier(tier, url, referer=referer, rendered=rendered)


class TestRenderedFirst:
    async def test_adaptive_domain_opens_rendered_and_stops(self, settings):
        lad = RecordingLadder(settings, {2: (200, {"content-type": HTML}, GOOD)})
        try:
            res = await lad.fetch("https://www.adaptive.example/x")
            assert lad.passes == [(2, True)], "opens on the challenge sidecar"
            assert res.tier == 2 and res.rendered, "browser pass, honestly stamped"
        finally:
            await lad.close()

    async def test_failed_opener_hands_off_to_the_normal_climb(self, settings):
        blocked = (403, {"set-cookie": "datadome=x"}, "<html>no</html>")
        lad = RecordingLadder(settings, {0: blocked, 1: blocked, 2: blocked})
        try:
            with pytest.raises(Blocked):
                await lad.fetch("https://adaptive.example/x")
            assert lad.passes[0] == (2, True), "opens on the challenge sidecar"
            assert (0, False) in lad.passes and (1, False) in lad.passes, \
                "fallback runs the cheap climb rendered=False"
            assert lad.passes[1][0] == 0, "fallback restarts at the profile's tier"
            # Under SEARCHIO_SIDECAR_ENGINE=1 the fallback's engine tier-2
            # pass draws the same block and the auto-rescue appends one
            # rendered retry -- correct machinery, not an exhaust above the
            # opener (that is why the assertion pins ordering, not exactness).
        finally:
            await lad.close()

    async def test_opener_429_falls_back_and_refuses_honestly(self, settings):
        # The class's signature: even the browser's first document can draw
        # the adaptive 429 (probe ec52690's goto envelope). A 429 verdict is
        # not a block (iteration 19) -- the opener fails into the fallback,
        # the cheap tiers 429 too, and the fetch refuses with no rescue.
        throttle = (429, {"content-type": HTML}, "<html>Too Many Requests</html>")
        lad = RecordingLadder(settings, {0: throttle, 1: throttle, 2: throttle})
        try:
            with pytest.raises(TransientError) as exc:
                await lad.fetch("https://adaptive.example/x")
            assert lad.passes[0] == (2, True)
            assert (0, False) in lad.passes, "fallback climb happened"
            assert "no usable content" in str(exc.value)
        finally:
            await lad.close()

    async def test_opener_sidecar_unavailable_falls_back(self, settings):
        # The browser cannot boot: the climb still gets its honest shot
        # instead of the fetch dying on the opener's infrastructure.
        lad = RecordingLadder(settings, {
            2: TransientError("sidecar down"),
            0: (200, {"content-type": HTML}, GOOD),
        })
        try:
            res = await lad.fetch("https://adaptive.example/x")
            assert lad.passes[0] == (2, True)
            assert (0, False) in lad.passes
            assert res.tier == 0
        finally:
            await lad.close()

    async def test_non_adaptive_domain_climbs_as_before(self, settings):
        lad = RecordingLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        try:
            await lad.fetch("https://plain.example/x")
            assert lad.passes == [(0, False)]
        finally:
            await lad.close()

    async def test_suffix_lookalike_does_not_match(self, settings):
        # "notadaptive.example" contains the entry but is not a subdomain of
        # it -- the public-suffix-shaped bug this matching must not have.
        lad = RecordingLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        try:
            await lad.fetch("https://notadaptive.example/x")
            assert lad.passes == [(0, False)]
        finally:
            await lad.close()

    async def test_force_tier_skips_the_opener(self, settings):
        lad = RecordingLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        try:
            res = await lad.fetch("https://adaptive.example/x", force_tier=0)
            assert lad.passes == [(0, False)] and res.tier == 0
        finally:
            await lad.close()

    async def test_ceiling_below_2_skips_the_opener(self, settings):
        settings.max_tier = 1
        lad = RecordingLadder(settings, {
            0: TransientError("no cheap stack configured"),
            1: (200, {"content-type": HTML}, GOOD),
        })
        try:
            res = await lad.fetch("https://adaptive.example/x")
            assert all(not rendered for _, rendered in lad.passes), \
                "no rendered pass when tier 2 is not permitted"
            assert res.tier == 1
        finally:
            await lad.close()

    async def test_tenant_scoped_fetch_skips_the_opener(self, settings):
        # Bug 136's rule: the browser jar is per-process, so tenant fetches
        # start at the isolated tiers -- an operator routing table does not
        # override tenant isolation.
        lad = RecordingLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        try:
            await lad.fetch("https://adaptive.example/x", session="tenant-a")
            assert lad.passes == [(0, False)]
        finally:
            await lad.close()

    async def test_explicit_rendered_request_is_not_duplicated(self, settings):
        # rendered=True already routes every tier through the challenge
        # sidecar; the opener must not stack a second rendered pass on top.
        lad = RecordingLadder(settings, {0: (200, {"content-type": HTML}, GOOD)})
        try:
            res = await lad.fetch("https://adaptive.example/x", rendered=True)
            assert lad.passes == [(0, True)]
            assert res.rendered is False, "tier 0 answered; no browser pass claimed"
        finally:
            await lad.close()
