"""learn_rendered_first: a host whose read got through only on the challenge
browser's rescue opens its next read there, skipping the tiers and the engine
pass that just failed; a failed opener unlearns it."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from tests.test_ladder_router import GOOD, HTML
from tests.test_ladder_rendered_first import RecordingLadder

WALL = (403, {"content-type": HTML, "server": "AkamaiGHost"}, "<html>Access Denied</html>")


class EngineWalled(RecordingLadder):
    """Every tier walls except a rendered (challenge-browser) tier-2 pass."""

    browser_ok = True

    async def _try_tier(self, tier, url, *, referer="", rendered=False):
        self.passes.append((tier, rendered))
        status, headers, body = (200, {"content-type": HTML}, GOOD) if (
            tier == 2 and rendered and self.browser_ok) else WALL
        return status, headers, body, headers.get("content-type", HTML), url


def _settings(tmp_path, on: bool) -> Settings:
    return Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off", max_tier=2,
                    per_domain_rps=1000.0, per_domain_burst=1000, sidecar_autostart=False,
                    sidecar_engine=True, sidecar_challenge="patchright", sidecar_challenge_auto=True,
                    rendered_first_domains=[], learn_rendered_first=on, skip_dead_tier=False)


@pytest.mark.parametrize("on", [False, True])
async def test_second_read_opens_on_the_browser_only_when_on(tmp_path, on):
    lad = EngineWalled(_settings(tmp_path, on), {})
    try:
        await lad.fetch("https://walled.example/a")
        assert lad.passes[-1] == (2, True), lad.passes
        lad.passes.clear()
        res = await lad.fetch("https://walled.example/b")
        assert res.tier == 2
        assert (lad.passes == [(2, True)]) is on, lad.passes
    finally:
        await lad.close()


async def test_a_failed_opener_unlearns_and_climbs(tmp_path):
    lad = EngineWalled(_settings(tmp_path, True), {})
    try:
        await lad.fetch("https://walled.example/a")
        lad.browser_ok = False
        with pytest.raises(Exception):
            await lad.fetch("https://walled.example/b")
        assert lad.passes[-1] != (2, True) or len(lad.passes) > 3
        assert "walled.example" not in lad._learned_rf
    finally:
        await lad.close()


async def test_other_hosts_are_not_affected(tmp_path):
    lad = EngineWalled(_settings(tmp_path, True), {})
    try:
        await lad.fetch("https://walled.example/a")
        lad.passes.clear()
        await lad.fetch("https://other.example/a")
        assert lad.passes[0][0] == 0, lad.passes
    finally:
        await lad.close()
