"""Sidecar client under concurrent callers (iteration 71, concurrency brief
on sidecar.py). Red first."""

from __future__ import annotations

import asyncio

import pytest

from searchio.config import Settings
from searchio.errors import SidecarUnavailable
from searchio.net.ladder import Ladder
from searchio.net.sidecar import SidecarClient


class TestQueuedBootersHonourAStickyFailure:
    async def test_second_caller_does_not_boot_again_after_the_first_boot_failed(self, tmp_path):
        # Bug 158: after the boot lock was acquired there was no re-check of
        # the sticky unavailable flag, so a caller queued behind a failed
        # boot launched a SECOND boot with the flag already set -- and if
        # that one came up, every later ensure() was refused anyway.
        script = tmp_path / "sidecar.py"
        script.write_text("# placeholder")
        sc = SidecarClient(script=str(script), autostart=True, port=1)
        sc.binary = None
        spawns = 0

        async def failing_spawn(*a, **k):
            nonlocal spawns
            spawns += 1
            await asyncio.sleep(0.05)
            sc._unavailable_reason = "boot failed (test)"
            raise SidecarUnavailable(sc._unavailable_reason)

        async def never_healthy(url):
            return False
        sc._spawn = failing_spawn
        sc._spawn_engine = failing_spawn
        sc._healthy = never_healthy
        results = await asyncio.gather(sc.ensure(), sc.ensure(), sc.ensure(), return_exceptions=True)
        assert all(isinstance(r, SidecarUnavailable) for r in results), results
        assert spawns == 1, f"{spawns} boots for one failure"


class TestTierTwoFetchesGetTheirOwnTab:
    async def test_concurrent_rescues_do_not_share_the_default_tab(self, tmp_path):
        # Bug 159: the ladder's tier-2 fetch named no tab, so every caller
        # navigated "default" -- on the browser rescue, fetch is a goto plus
        # a read on that tab, and two callers at once could hand each other's
        # page back. Each fetch now gets its own tab and closes it.
        seen: list[tuple] = []

        class Rec:
            available = True

            async def fetch(self, url, **kw):
                seen.append(("fetch", url, kw.get("tab_id")))
                await asyncio.sleep(0.01)
                html = "<html><body><p>" + (f"page for {url} " * 40) + "</p></body></html>"
                return {"ok": True, "status": 200, "html": html, "url": url}

            async def call(self, verb, params=None, **kw):
                seen.append((verb, (params or {}).get("tab_id")))
                return {"ok": True}

            async def goto(self, url, **kw):
                return {"ok": True, "status": 200}

            async def cookies(self):
                return []

            async def user_agent(self):
                return ""

            async def close(self):
                return None

        class NoHttp(Ladder):
            async def _tier0(self, url, *, referer=""):
                from searchio.errors import TransientError
                raise TransientError("no network")

            async def _tier1(self, url, *, referer=""):
                from searchio.errors import TransientError
                raise TransientError("no network")
        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=2, robots_policy="off",
                     sidecar_autostart=False)
        lad = NoHttp(s, sidecar=Rec())
        try:
            await asyncio.gather(lad.fetch("https://a.example/1", use_cache=False, force_tier=2),
                                 lad.fetch("https://b.example/2", use_cache=False, force_tier=2))
        finally:
            await lad.close()
        tabs = [rec[2] for rec in seen if rec[0] == "fetch"]
        assert len(tabs) == 2 and tabs[0] != tabs[1] and all(t and t != "default" for t in tabs), seen
        closed = [rec[1] for rec in seen if rec[0] == "close_tab"]
        assert sorted(closed) == sorted(tabs), seen
