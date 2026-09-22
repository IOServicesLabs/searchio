"""Marketplace session gate vs concurrent queries (iteration 72, Muse
concurrency brief on facebook.py, candidate #2 confirmed by reading the
engine's session_reset arm: it clears the jar AND drops every tab). Red
first."""

from __future__ import annotations

import asyncio
import json

import pytest

from searchio.models import Query
from searchio.providers.facebook import FacebookMarketplace

from tests.test_facebook import gate_ctx, logged_in_provider, page


class EngineLikeSidecar:
    """Tabs live in a dict the way se-serve keeps them; session_reset drops
    them all (the engine's contract) and flips the jar to anonymous. A read
    on a dropped tab fails the way the client surfaces a verb failure.
    Every read takes a moment and a "slow" navigation takes longer, so two
    queries really are in flight together."""

    available = True

    def __init__(self) -> None:
        self.anonymous = False
        self.tabs: set[str] = set()
        self.calls: list[str] = []
        self.order: list[str] = []
        self.dead_market = False

    async def goto(self, url, *, tab_id="default"):
        self.tabs.add(tab_id)
        self.order.append(f"goto:{tab_id}")
        if "slow" in url:
            await asyncio.sleep(0.15)
        return {"ok": True, "final_url": url}

    async def read_html(self, *, tab_id="default", selector=""):
        await asyncio.sleep(0.05)
        if tab_id not in self.tabs:
            self.order.append(f"read:{tab_id}:unknown-tab")
            raise RuntimeError(f"read_html: unknown tab {tab_id!r}")
        self.order.append(f"read:{tab_id}")
        if self.anonymous and not self.dead_market:
            return page([("555555", ["$30", "Used Scooter", "Chattanooga, TN"])])
        return "<html><body><div role='main'></div></body></html>"

    async def eval_js(self, js, *, tab_id="default", timeout=None):
        if tab_id not in self.tabs:
            raise RuntimeError(f"eval_js: unknown tab {tab_id!r}")
        return 20

    async def call(self, verb, params=None, **kw):
        self.calls.append(verb)
        if verb == "session_reset":
            self.order.append("reset")
            self.anonymous = True
            self.tabs.clear()
        if verb == "close_tab":
            self.tabs.discard((params or {}).get("tab_id", "default"))
        if verb == "storage_state_set":
            self.anonymous = False
            return {"ok": True, "cookies": 3}
        return {"ok": True}


class TestAResetWaitsForInFlightLoads:
    async def test_a_concurrent_query_survives_another_querys_gate(self):
        # Bug 161: the empty-feed gate ran session_reset while another
        # query's tab was mid-load; on the engine the reset drops every tab,
        # so the other query's read failed with "unknown tab" (on patchright
        # its scrolled batches silently went anonymous under a session
        # provenance stamp). Loads in flight now drain before a reset runs.
        sc = EngineLikeSidecar()
        prov = logged_in_provider()
        ctx = gate_ctx(sc)
        fast, slow = await asyncio.gather(
            prov.find_items(Query(text="fast", k=5), ctx),
            prov.find_items(Query(text="slow", k=5), ctx),
        )
        assert [i.title for i in fast] == ["Used Scooter"], (fast, sc.order)
        assert [i.title for i in slow] == ["Used Scooter"], (slow, sc.order)
        assert not any(e.endswith("unknown-tab") for e in sc.order), sc.order
        # The mechanism: the slow query's page was read BEFORE the first reset.
        slow_tab = next(e.split(":")[1] for e in sc.order if e.startswith("goto:") and sc.order.index(e) == 1)
        assert sc.order.index(f"read:{slow_tab}") < sc.order.index("reset"), sc.order


class TestAnEmptyMarketDoesNotLoseTheSession:
    async def test_the_session_is_put_back_after_an_empty_market_check(self, tmp_path):
        # Bug 162: the anonymous check wipes the browser's cookies whatever it
        # concludes, and _ensure_session injects once per process -- so one
        # honest empty market left the browser anonymous for the rest of the
        # process while every later listing was stamped "loaded:2693c" and
        # every later empty result ran a pointless reset + retry.
        state_file = tmp_path / "fb.json"
        state_file.write_text(json.dumps({"cookies": [{"name": "c_user", "value": "1",
                                                        "domain": ".facebook.com", "path": "/"}],
                                          "origins": []}))
        sc = EngineLikeSidecar()
        sc.dead_market = True
        prov = logged_in_provider()
        prov._session_source = str(state_file)
        items = await prov.find_items(Query(text="nothing-here", k=5), gate_ctx(sc))
        assert items == [] and prov.search_gate == "empty_market"
        assert sc.calls.index("storage_state_set") > sc.calls.index("session_reset"), sc.calls
        assert not sc.anonymous, "the browser is back on the session"
        assert prov._session_active()
        assert prov._session_state == "loaded:2693c"

    async def test_a_failed_restore_is_named_and_the_session_counts_as_gone(self, tmp_path):
        sc = EngineLikeSidecar()
        sc.dead_market = True
        orig_call = sc.call

        async def call(verb, params=None, **kw):
            if verb == "storage_state_set":
                self_calls = sc.calls
                self_calls.append(verb)
                return {"ok": False, "error": "no browser"}
            return await orig_call(verb, params, **kw)
        sc.call = call
        prov = logged_in_provider()
        prov._session_source = str(tmp_path / "missing.json")
        await prov.find_items(Query(text="nothing-here", k=5), gate_ctx(sc))
        assert prov.search_gate == "empty_market"
        assert not prov._session_broken, "nothing was proven about the session"
        assert not prov._session_active(), prov._session_state
        assert "restore" in prov._session_state, prov._session_state

    async def test_ensure_session_remembers_where_the_session_came_from(self, tmp_path, monkeypatch):
        state_file = tmp_path / "fb.json"
        state_file.write_text(json.dumps({"cookies": [], "origins": []}))
        sc = EngineLikeSidecar()
        prov = FacebookMarketplace()
        ctx = gate_ctx(sc)
        ctx.settings.facebook_session = str(state_file)
        await prov._ensure_session(sc, ctx)
        assert prov._session_state.startswith("loaded:")
        assert prov._session_source == str(state_file)


class TestTheHandoffLoadIsNotResetFromUnder:
    async def test_a_gate_queued_during_the_handoff_runs_after_its_load(self, monkeypatch):
        # Facebook review #6 (iteration 72): the handoff injected the human's
        # session and loaded outside the session lock, so a concurrent
        # query's gate could wipe the fresh session between the injection
        # and the load -- anonymous listings stamped "handoff".
        from searchio.net import login_handoff
        from searchio.net.login_handoff import HandoffResult

        sc = EngineLikeSidecar()
        orig_call = sc.call

        async def call(verb, params=None, **kw):
            if verb == "storage_state_set":
                sc.order.append("inject:start")
                await asyncio.sleep(0.1)
                r = await orig_call(verb, params, **kw)
                sc.order.append("inject:end")
                return r
            return await orig_call(verb, params, **kw)
        sc.call = call

        async def listings(*, tab_id="default", selector=""):
            await asyncio.sleep(0.02)
            if tab_id not in sc.tabs:
                sc.order.append(f"read:{tab_id}:unknown-tab")
                raise RuntimeError("unknown tab")
            sc.order.append(f"read:{tab_id}")
            return page([("555555", ["$30", "Used Scooter", "Chattanooga, TN"])])
        sc.read_html = listings

        async def fake_login(**kw):
            return HandoffResult(state={"cookies": [{"name": "c_user", "value": "1",
                                                     "domain": ".facebook.com", "path": "/"}],
                                        "origins": []},
                                 login_url=kw["login_url"], elapsed_s=1.0)
        monkeypatch.setattr(login_handoff, "request_human_login", fake_login)
        prov = logged_in_provider()
        prov._session_broken = True
        ctx = gate_ctx(sc)
        url = "https://www.facebook.com/marketplace/search?query=kayak"

        async def gate_later():
            await asyncio.sleep(0.03)
            return await prov._retry_anonymous(sc, url, want=5)
        handed, (_items, gate) = await asyncio.gather(
            prov._handoff_and_retry(sc, url, want=5, ctx=ctx), gate_later())
        assert [i.title for i in handed] == ["Used Scooter"], (handed, sc.order)
        assert not any(e.endswith("unknown-tab") for e in sc.order), sc.order
        reads_before_reset = [e for e in sc.order[:sc.order.index("reset")] if e.startswith("read:")]
        assert reads_before_reset, sc.order  # the handoff's load finished before the reset
        assert sc.order.index("inject:end") < sc.order.index("reset"), sc.order
