"""The tier-2 warm-up refetch stays on the call's own tab (iteration 74,
Muse concurrency review of ladder.py after the bug-159 changes: the warm-up
goto used a FIXED tab_id="warmup", so every concurrent tier-2 rescue
navigated one shared tab). Red first."""

from __future__ import annotations

import asyncio

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                    max_tier=2, per_domain_rps=1000.0, per_domain_burst=1000,
                    sidecar_autostart=False)


class WarmupSidecar:
    """Refuses the first fetch on a tab, serves after that tab has been
    primed by a goto -- so a warm-up on the WRONG tab never unblocks the
    refetch."""

    available = True

    def __init__(self, delay: float = 0.0):
        self.events: list[tuple] = []
        self.primed: set[str] = set()
        self.delay = delay

    async def fetch(self, url, **kw):
        tab = kw.get("tab_id")
        self.events.append(("fetch", url, tab))
        if self.delay:
            await asyncio.sleep(self.delay)
        if tab in self.primed:
            html = "<html><body><p>" + ("real content " * 60) + "</p></body></html>"
            return {"ok": True, "status": 200, "html": html, "url": url}
        return {"ok": True, "status": 403, "html": "denied", "url": url}

    async def goto(self, url, tab_id="default", **kw):
        self.events.append(("goto", url, tab_id))
        self.primed.add(tab_id)
        return {"ok": True, "status": 200}

    async def call(self, verb, params=None, **kw):
        self.events.append((verb, None, (params or {}).get("tab_id")))
        return {"ok": True}

    async def cookies(self):
        return []

    async def user_agent(self):
        return ""

    async def close(self):
        return None


class TestTheWarmupPrimesTheCallsOwnTab:
    async def test_the_warmup_goto_and_the_refetch_share_the_calls_private_tab(self, settings):
        sc = WarmupSidecar()
        lad = Ladder(settings, sidecar=sc)
        try:
            res = await lad._tier2("https://hostile.example/deep/page")
            assert res[0] == 200, res
            assert "real content" in res[2]
        finally:
            await lad.close()
        goto_tabs = [e[2] for e in sc.events if e[0] == "goto"]
        fetch_tabs = [e[2] for e in sc.events if e[0] == "fetch"]
        assert goto_tabs and goto_tabs[0] != "warmup", sc.events
        # The warm-up primed exactly the tab the refetch then used.
        assert goto_tabs[0] == fetch_tabs[-1], sc.events
        assert all(t and t != "default" for t in goto_tabs), sc.events

    async def test_two_concurrent_rescues_do_not_share_a_warmup_tab(self, settings):
        sc = WarmupSidecar(delay=0.02)
        lad = Ladder(settings, sidecar=sc)
        try:
            a, b = await asyncio.gather(
                lad._tier2("https://a.example/deep"),
                lad._tier2("https://b.example/deep"),
            )
            assert a[0] == 200 and b[0] == 200, (a, b)
        finally:
            await lad.close()
        goto_tabs = [e[2] for e in sc.events if e[0] == "goto"]
        assert len(goto_tabs) == 2 and goto_tabs[0] != goto_tabs[1], sc.events
        assert "warmup" not in goto_tabs, sc.events
        # Every tab named in the run is closed exactly once.
        closed = sorted(e[2] for e in sc.events if e[0] == "close_tab")
        opened = sorted(set(e[2] for e in sc.events if e[0] in ("fetch", "goto")))
        assert closed == opened, sc.events
