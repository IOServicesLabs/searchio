"""Clearance capture, replay, and the constraints that make it narrow."""

from __future__ import annotations

import time

from searchio.net.clearance import (
    CLEARANCE_COOKIES,
    Clearance,
    ClearanceStore,
    relevant_cookies,
)


def jar(*entries):
    return [{"name": n, "value": v, "domain": d} for n, v, d in entries]


class TestRelevantCookies:
    def test_picks_only_retained_cookies(self):
        # Retained = clearance names + the trust-continuity class (_pxvid
        # and friends); trackers and session/login names stay excluded.
        got = relevant_cookies(
            jar(("cf_clearance", "abc", ".g2.com"),
                ("datadome", "xyz", ".g2.com"),
                ("_pxvid", "px", ".g2.com"),
                ("_ga", "tracking", ".g2.com"),
                ("session_id", "s", ".g2.com")),
            "g2.com",
        )
        assert got == {"cf_clearance": "abc", "datadome": "xyz", "_pxvid": "px"}

    def test_parent_domain_cookie_applies_to_host(self):
        got = relevant_cookies(jar(("cf_clearance", "v", ".example.com")), "www.example.com")
        assert got == {"cf_clearance": "v"}

    def test_other_hosts_are_ignored(self):
        got = relevant_cookies(jar(("cf_clearance", "v", ".other.com")), "g2.com")
        assert got == {}

    def test_every_known_cookie_is_a_challenge_cookie(self):
        # Guard against someone adding a session cookie here: replaying a login
        # is a different and much more dangerous thing than replaying a
        # "you already passed the bot check" token.
        assert "session" not in CLEARANCE_COOKIES
        assert "cf_clearance" in CLEARANCE_COOKIES and "datadome" in CLEARANCE_COOKIES


class TestStore:
    def test_round_trip(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3")
        st.put("g2.com", {"cf_clearance": "abc"}, "Mozilla/5.0 Test")
        got = st.get("g2.com")
        assert got and got.cookies == {"cf_clearance": "abc"}
        assert got.user_agent == "Mozilla/5.0 Test"
        assert got.header() == "cf_clearance=abc"
        st.close()

    def test_persists_across_instances(self, tmp_path):
        # The cookie outlives the process and re-earning it costs a browser.
        p = tmp_path / "c.sqlite3"
        a = ClearanceStore(p)
        a.put("x.com", {"datadome": "v"}, "UA")
        a.close()
        b = ClearanceStore(p)
        assert b.get("x.com").cookies == {"datadome": "v"}
        b.close()

    def test_expired_clearance_is_not_returned(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", ttl_s=1)
        st.put("x.com", {"cf_clearance": "v"})
        st._mem["x.com"].captured = time.time() - 10
        assert st.get("x.com") is None
        st.close()

    def test_drop_forgets(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3")
        st.put("x.com", {"cf_clearance": "v"})
        st.drop("x.com")
        assert st.get("x.com") is None
        st.close()

    def test_disabled_store_is_inert(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3", enabled=False)
        st.put("x.com", {"cf_clearance": "v"})
        assert st.get("x.com") is None
        st.close()

    def test_empty_cookies_are_not_stored(self, tmp_path):
        st = ClearanceStore(tmp_path / "c.sqlite3")
        st.put("x.com", {})
        assert st.get("x.com") is None
        st.close()


def test_freshness_window():
    c = Clearance("x", {"cf_clearance": "v"}, captured=time.time())
    assert c.fresh(ttl_s=60)
    assert not Clearance("x", {"cf_clearance": "v"}, captured=time.time() - 99).fresh(ttl_s=60)
