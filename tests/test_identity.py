"""Browser identity layer: persona coherence, clearance replay, store degradation.

Iteration 45 of the span suite (DeepSeek design review of net/clearance.py +
net/persona.py + net/profiles.py as one concatenated target). Every test
here bit RED before its fix, in the shape the review predicted.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from searchio.net import persona as persona_mod
from searchio.net.clearance import ClearanceStore
from searchio.net.ladder import Ladder
from searchio.net.persona import PERSONAS
from searchio.net.profiles import DomainStore


class TestPersonaCoherence:
    """Bug 70: the edge-win persona paired an Edge 131 User-Agent and
    v="131" Client Hints with curl_cffi's `edge101` TLS target -- a Chrome
    101-era JA3/H2 fingerprint under a 131 UA, which is precisely the
    self-contradiction the module's docstring says it exists to prevent.
    Edge IS Chromium: its TLS target is the matching chrome<major>.
    """

    def test_every_persona_tls_target_matches_its_ua_major(self):
        for p in PERSONAS:
            fam, ver = re.match(r"([a-z]+)(\d+)", p.impersonate).groups()
            if "Safari/605" in p.ua and "Chrome/" not in p.ua:
                assert fam == "safari", p.name
                assert f"Version/{ver}" in p.ua, p.name
                continue
            major = re.search(r"Chrome/(\d+)", p.ua).group(1)
            assert fam == "chrome", (p.name, p.impersonate)
            assert ver == major, (p.name, p.impersonate, major)
            assert f'v="{major}"' in p.sec_ch_ua, p.name


class TestHintsFollowThePinnedUA:
    """Bug 73: `_clearance_headers` pinned the User-Agent the browser earned
    the cookie under (Chromium 14x from patchright) but left the persona's
    sec-ch-ua v="131" / platform -- or, under the Safari persona, NO hints
    at all under a Chrome UA. Client Hints are compared against the UA by
    every vendor that binds cf_clearance to it; the pin must carry them.
    """

    def test_hints_derive_from_a_chromium_ua(self):
        from searchio.net.persona import hints_for_ua

        ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")
        h = hints_for_ua(ua)
        assert h["sec-ch-ua"].count('v="143"') == 2 and "Google Chrome" in h["sec-ch-ua"]
        assert h["sec-ch-ua-platform"] == '"Windows"' and h["sec-ch-ua-mobile"] == "?0"
        edge = hints_for_ua(ua + " Edg/143.0.0.0")
        assert "Microsoft Edge" in edge["sec-ch-ua"] and "Google Chrome" not in edge["sec-ch-ua"]
        mac = hints_for_ua("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")
        assert mac["sec-ch-ua-platform"] == '"macOS"'
        assert hints_for_ua("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                            "(KHTML, like Gecko) Version/17.0 Safari/605.1.15") == {}

    def test_pin_rewrites_hints_and_strips_them_for_non_chromium(self):
        from searchio.net.persona import pin_user_agent

        chrome = persona_mod.by_name("chrome-win")
        pinned = pin_user_agent(chrome.headers(), "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")
        assert 'v="143"' in pinned["sec-ch-ua"] and 'v="131"' not in pinned["sec-ch-ua"]
        assert pinned["sec-ch-ua-platform"] == '"Linux"'
        safari_ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                     "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
        pinned = pin_user_agent(chrome.headers(), safari_ua)
        assert pinned["User-Agent"] == safari_ua
        assert not any(k.lower().startswith("sec-ch-ua") for k in pinned)

    def test_clearance_replay_carries_coherent_hints(self, tmp_path: Path):
        store = ClearanceStore(tmp_path / "c.db")
        browser_ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")
        store.put("g2.com", {"cf_clearance": "tok"}, browser_ua)
        lad = SimpleNamespace(clearance=store, _bump=lambda k: None)
        # Safari persona: no hints at all, then a Chrome UA pinned on top.
        safari = persona_mod.by_name("safari-mac")
        base = persona_mod.safari_headers_fix(safari, safari.headers())
        out = Ladder._clearance_headers(lad, "g2.com", base, "https://g2.com/x")
        assert out["User-Agent"] == browser_ua and "cf_clearance=tok" in out["Cookie"]
        assert 'v="143"' in out["sec-ch-ua"] and out["sec-ch-ua-platform"] == '"Windows"'
        store.close()


class TestClearanceNeverLeaksOverPlaintext:
    def test_http_url_gets_no_clearance(self, tmp_path: Path):
        # Bug 72: clearance was keyed by domain and attached to ANY fetch on
        # it, including http:// -- cf_clearance is a Secure cookie, and
        # sending the expensive token in cleartext both leaks it and is a
        # browser-impossible signal.
        store = ClearanceStore(tmp_path / "c.db")
        store.put("g2.com", {"cf_clearance": "tok"}, "UA")
        lad = SimpleNamespace(clearance=store, _bump=lambda k: None)
        out = Ladder._clearance_headers(lad, "g2.com", {"Accept": "*/*"}, "http://g2.com/x")
        assert "Cookie" not in out and out.get("User-Agent") != "UA"
        assert store.get("g2.com").uses == 0
        out = Ladder._clearance_headers(lad, "g2.com", {"Accept": "*/*"}, "https://g2.com/x")
        assert "cf_clearance=tok" in out["Cookie"]
        store.close()


class TestStoresDegradeNeverFail:
    """Bug 71 (the bug-67 class, two more stores): a corrupt clearance or
    profiles file raised sqlite3.DatabaseError out of Ladder construction,
    and a locked database raised out of profiles.record_success /
    clearance.put AFTER a successful fetch -- both called outside any try.
    Both stores have an in-memory layer; a failing database degrades to it.
    """

    def test_corrupt_files_fall_back_to_memory(self, tmp_path: Path):
        for cls in (ClearanceStore, DomainStore):
            p = tmp_path / f"{cls.__name__}.db"
            p.write_bytes(b"not a sqlite file" * 100)
            s = cls(p)  # used to raise sqlite3.DatabaseError
            assert s.errors >= 1 and "not a database" in s.last_error
            s.close()

    def test_locked_database_is_memory_only_not_a_crash(self, tmp_path: Path):
        cp = tmp_path / "c.db"
        cs = ClearanceStore(cp)
        other = sqlite3.connect(str(cp), timeout=0.1)
        other.execute("BEGIN EXCLUSIVE")
        cs._db.execute("PRAGMA busy_timeout=100")
        try:
            cs.put("g2.com", {"cf_clearance": "tok"}, "UA")  # used to raise
            assert cs.get("g2.com").cookies == {"cf_clearance": "tok"}
            cs.record_use("g2.com")
            cs.drop("g2.com")
            assert cs.get("g2.com") is None and cs.errors >= 1
        finally:
            other.rollback()
            other.close()
            cs.close()
        pp = tmp_path / "p.db"
        ds = DomainStore(pp)
        other = sqlite3.connect(str(pp), timeout=0.1)
        other.execute("BEGIN EXCLUSIVE")
        ds._db.execute("PRAGMA busy_timeout=100")
        try:
            ds.record_success("g2.com", 1)  # used to raise
            ds.record_block("g2.com", 1, "cloudflare")
            assert ds.get("g2.com").min_tier == 2 and ds.errors >= 1
            assert ds.start_tier("g2.com", 2) in (0, 2)
        finally:
            other.rollback()
            other.close()
            ds.close()

    def test_non_dict_cookie_row_is_not_a_clearance(self, tmp_path: Path):
        p = tmp_path / "c.db"
        cs = ClearanceStore(p)
        cs.put("g2.com", {"cf_clearance": "tok"}, "UA")
        cs._db.execute("UPDATE clearance SET cookies = ? WHERE domain = ?",
                       (json.dumps(["not", "a", "dict"]), "g2.com"))
        cs._db.commit()
        cs._mem.clear()
        assert cs.get("g2.com") is None  # used to build a Clearance whose header() crashes
        assert cs.all() == []
        cs.close()


# ── iteration 69: Muse second pass on the identity stores ────────────────────


class TestStoreRowsAreCoerced69:
    def test_corrupt_clearance_row_is_none_not_a_crash(self, tmp_path):
        # Bug 153: `_from_row` coerced captured/uses with float()/int() outside
        # the never-raise guard, so a hand-edited or corrupted row made
        # get() raise ValueError into the fetch path.
        import sqlite3

        from searchio.net.clearance import ClearanceStore

        st = ClearanceStore(tmp_path / "c.db")
        st.put("x.com", {"cf_clearance": "v"}, "UA")
        con = sqlite3.connect(tmp_path / "c.db")
        con.execute("UPDATE clearance SET captured='garbage', uses='many'")
        con.commit()
        con.close()
        st2 = ClearanceStore(tmp_path / "c.db")
        assert st2.get("x.com") is None
        assert st2.all() == []

    def test_corrupt_profile_row_is_clamped(self, tmp_path):
        # Same class, profile side: min_tier 'abc' came back as a str and
        # start_tier's min() on it was a TypeError; 99 came back as 99.
        import sqlite3

        from searchio.net.profiles import DomainStore

        ps = DomainStore(tmp_path / "p.db")
        ps.record_block("y.com", 1, "cloudflare")
        con = sqlite3.connect(tmp_path / "p.db")
        con.execute("UPDATE domains SET min_tier='abc', successes='x', rps='fast'")
        con.commit()
        con.close()
        ps2 = DomainStore(tmp_path / "p.db")
        p = ps2.get("y.com")
        assert isinstance(p.min_tier, int) and 0 <= p.min_tier <= 2
        assert isinstance(p.successes, int) and isinstance(p.rps, float)
        assert ps2.start_tier("y.com", 2) in (0, 1, 2)
        con = sqlite3.connect(tmp_path / "p.db")
        con.execute("UPDATE domains SET min_tier=99")
        con.commit()
        con.close()
        assert DomainStore(tmp_path / "p.db").get("y.com").min_tier == 2


class TestStoresAreThreadSafe69:
    def test_concurrent_threads_do_not_drop_writes(self, tmp_path):
        # Bug 154: the three sqlite stores open one connection with
        # check_same_thread=False and no lock; tier 1 builds its clearance
        # headers INSIDE the worker thread while the loop thread writes, and
        # four threads produced "cannot commit - no transaction is active".
        import threading

        from searchio.net.clearance import ClearanceStore

        st = ClearanceStore(tmp_path / "c.db")

        def w(i):
            for k in range(200):
                st.put(f"h{i}-{k}.com", {"cf_clearance": "v"}, "UA")
                st.get(f"h{i}-{k}.com")
                st.record_use(f"h{i}-{k}.com")
        ts = [threading.Thread(target=w, args=(i,)) for i in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert st.errors == 0, st.last_error
        assert len(st.all()) == 800


class TestCookieScopeIsDownwardOnly69:
    def test_subdomain_cookie_is_not_replayed_on_the_parent(self):
        # Bug 155: relevant_cookies() accepted a cookie scoped to
        # sub.example.com for example.com -- a browser never sends a
        # child's cookie to the parent, and the replay put a wrong
        # cookie on the wire. Parent -> child stays valid.
        from searchio.net.clearance import relevant_cookies

        jar = [{"name": "cf_clearance", "value": "A", "domain": ".example.com"},
               {"name": "__cf_bm", "value": "B", "domain": "sub.example.com"},
               {"name": "cf_clearance", "value": "C", "domain": "other.com"},
               {"name": "_cfuvid", "value": "D", "domain": ""}]
        assert relevant_cookies(jar, "example.com") == {"cf_clearance": "A"}
        assert relevant_cookies(jar, "sub.example.com") == {"cf_clearance": "A", "__cf_bm": "B"}
        assert relevant_cookies(jar, "other.com") == {"cf_clearance": "C"}
