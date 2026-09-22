"""Net policy layer: cache degradation, limiter boundedness, robots semantics.

Iteration 44 of the span suite (DeepSeek design review of net/cache.py +
net/robots.py + net/ratelimit.py as one concatenated target). Every test
here bit RED before its fix, in the shape the review predicted.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from searchio.net.cache import Cache
from searchio.net.ratelimit import FLOOR, DomainLimiter
from searchio.net.robots import UA, RobotsCache


class TestCacheDegradesNeverFails:
    """Bug 67: sqlite3 errors escaped the cache into the ladder. A corrupt
    cache file ("file is not a database") raised out of Ladder construction,
    so the whole engine failed to start over a scratch file; a "database is
    locked" from a concurrent CLI run raised out of cache.put AFTER a
    successful fetch -- the fetched result was thrown away and the caller got
    a raw sqlite3.OperationalError, not even a SearchioError. A cache is an
    optimization: every failure is a miss, counted in stats, never a fault.
    """

    def test_corrupt_file_disables_the_cache_instead_of_raising(self, tmp_path: Path):
        p = tmp_path / "c.db"
        p.write_bytes(b"not a sqlite file at all" * 100)
        c = Cache(p, 60, True)  # used to raise sqlite3.DatabaseError
        assert c.get("https://a/1") is None
        c.put("https://a/1", {"x": 1})  # no-op, no raise
        st = c.stats()
        assert st["entries"] == 0
        assert "not a database" in st["error"]
        c.close()

    def test_locked_database_put_is_a_counted_miss(self, tmp_path: Path):
        p = tmp_path / "c.db"
        c = Cache(p, 60, True)
        c.put("https://a/1", {"x": 1})
        other = sqlite3.connect(str(p), timeout=0.1)
        other.execute("BEGIN EXCLUSIVE")
        c._db.execute("PRAGMA busy_timeout=100")
        try:
            t0 = time.monotonic()
            c.put("https://a/2", {"x": 2})  # used to raise OperationalError
            assert time.monotonic() - t0 < 5.0
            assert c.get("https://a/1") == {"x": 1}  # WAL reader still served
            assert c.stats()["errors"] >= 1
        finally:
            other.rollback()
            other.close()
            c.close()

    def test_read_failure_is_a_miss(self, tmp_path: Path):
        c = Cache(tmp_path / "c.db", 60, True)
        c.put("https://a/1", {"x": 1})
        c._db.close()  # every later statement raises sqlite3.ProgrammingError
        assert c.get("https://a/1") is None
        c.put("https://a/1", {"x": 2})
        assert c.stats()["errors"] >= 2
        assert c.purge_expired() == 0
        c.close()


class TestLimiterNeverHangs:
    """Bug 68: a burst of 0 (or a rate of 0 / negative) made acquire() loop
    forever -- tokens could never reach 1.0 -- so every fetch on every domain
    hung silently. Settings does not validate per_domain_burst/per_domain_rps;
    the limiter is the last line and must clamp.
    """

    async def test_zero_burst_still_admits(self):
        lim = DomainLimiter(burst=0)
        await asyncio.wait_for(lim.acquire("a.com"), 1.0)  # used to hang

    async def test_zero_and_negative_rate_are_clamped_to_the_floor(self):
        lim = DomainLimiter(rps=0.0, burst=1)
        await asyncio.wait_for(lim.acquire("b.com"), 1.0)
        assert lim.rate_for("b.com") >= FLOOR
        lim = DomainLimiter(rps=-3.0, burst=-2, max_rps=0.0)
        await asyncio.wait_for(lim.acquire("c.com"), 1.0)
        assert lim.rate_for("c.com") >= FLOOR
        lim.record_success("c.com")
        assert lim.rate_for("c.com") >= FLOOR


def _robots(text: str, *, status: int = 200, policy: str = "enforce") -> RobotsCache:
    async def fetch(url: str):
        return status, text
    return RobotsCache(fetch, policy=policy)


GOOGLE_SHAPED = (
    "User-agent: *\n"
    "Disallow: /search\n"
    "Allow: /search/about\n"
    "Allow: /search/static\n"
    "Disallow: /*.pdf$\n"
    "Disallow: /a/*/print\n"
    "Disallow: /foo\n"
    "Crawl-delay: 2\n"
)


class TestRobotsRfc9309:
    """Bug 69: robots decisions came from urllib.robotparser, which knows
    neither the `*` wildcard nor the `$` anchor and applies FIRST-match
    instead of the RFC 9309 LONGEST-match rule. Google's own robots.txt
    ("Disallow: /search" then "Allow: /search/about") therefore refused
    /search/about under enforce and stamped it disallowed under warn, while
    "Disallow: /*.pdf$" matched nothing at all -- under enforce, a wildcard
    disallow was silently bypassed. Also: an empty Disallow means allow-all,
    and the most specific user-agent group wins over `*`.
    """

    async def test_longest_match_lets_the_allow_exception_win(self):
        rc = _robots(GOOGLE_SHAPED)
        assert (await rc.check("https://g.com/search/about")).allowed is True
        assert (await rc.check("https://g.com/search/static/x.css")).allowed is True
        assert (await rc.check("https://g.com/search?q=1")).allowed is False
        assert (await rc.check("https://g.com/search")).allowed is False

    async def test_wildcard_and_dollar_anchor(self):
        rc = _robots(GOOGLE_SHAPED)
        assert (await rc.check("https://g.com/x/doc.pdf")).allowed is False
        assert (await rc.check("https://g.com/x/doc.pdfx")).allowed is True  # $ anchors
        assert (await rc.check("https://g.com/a/b/print")).allowed is False
        assert (await rc.check("https://g.com/a/b/printer")).allowed is False  # prefix
        assert (await rc.check("https://g.com/foobar")).allowed is False  # RFC prefix
        assert (await rc.check("https://g.com/FOO")).allowed is True  # case-sensitive

    async def test_crawl_delay_and_reason(self):
        rc = _robots(GOOGLE_SHAPED)
        info = await rc.check("https://g.com/foo")
        assert info.crawl_delay == 2.0 and info.checked and info.reason == "disallowed_by_robots"
        assert not rc.permits(info)
        assert _robots(GOOGLE_SHAPED, policy="warn").permits(info)

    async def test_specific_group_beats_star_and_empty_disallow_allows_all(self):
        txt = f"User-agent: *\nDisallow: /\nUser-agent: {UA}\nDisallow: /private\n"
        rc = _robots(txt)
        assert (await rc.check("https://g.com/public")).allowed is True
        assert (await rc.check("https://g.com/private/x")).allowed is False
        rc = _robots("User-agent: *\nDisallow:\n")
        assert (await rc.check("https://g.com/anything")).allowed is True
        rc = _robots("User-agent: *\nDisallow: /\n")
        assert (await rc.check("https://g.com/")).allowed is False

    async def test_unavailable_robots_is_permission(self):
        info = await _robots("", status=404).check("https://g.com/x")
        assert info.allowed and not info.checked and info.reason == "robots_unavailable"
        info = await _robots("boom", status=503).check("https://g.com/x")
        assert info.allowed and not info.checked


class TestRobotsMatcherIsBounded:
    async def test_nested_wildcards_answer_instantly(self):
        # The first iteration-44 fix draft compiled patterns to regexes with
        # `.*` groups; `Disallow: /*/*/*/.../x$` against a long path then
        # backtracked exponentially and the probe ran for minutes (a hostile
        # robots.txt could wedge every fetch to that origin). The glob
        # matcher is O(pattern x path); this pins the bound.
        txt = "User-agent: *\n" + "\n".join(
            "Disallow: /" + "*/" * k + "x$" for k in range(1, 40)
        ) + "\nDisallow: /*a*b*c*d*e*f*g*h$\n"
        rc = _robots(txt)
        t0 = time.monotonic()
        info = await asyncio.wait_for(
            asyncio.to_thread(asyncio.run, rc.check("https://g.com/" + "a/" * 300 + "zzz")), 10.0)
        assert info.allowed is True and time.monotonic() - t0 < 2.0
        info = await rc.check("https://g.com/" + "a/" * 5 + "x")
        assert info.allowed is False
        assert (await rc.check("https://g.com/1a2b3c4d5e6f7g8h")).allowed is False
        assert (await rc.check("https://g.com/" + "abcdefgh" * 100 + "!")).allowed is True


class TestLimiterEdges68:
    """Iteration 68 (Muse second pass on ratelimit.py + probes)."""

    def test_zero_concurrency_does_not_deadlock(self):
        # Rider: DomainLimiter(concurrency=0) built a Semaphore(0) -- every
        # fetch would wait forever. Settings clamps its own field (bug 87);
        # the limiter now clamps too, like it clamps rps/burst (bug 69).
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=1, burst=1, max_rps=2, concurrency=0)
        assert not lim.slot().locked()

    def test_buckets_are_bounded(self):
        # Rider: one bucket per domain forever; a long-lived server that
        # sees a hundred thousand hosts kept a hundred thousand buckets.
        from searchio.net.ratelimit import DomainLimiter

        lim = DomainLimiter(rps=1, burst=1, max_rps=2, concurrency=4)
        for i in range(5000):
            lim._bucket(f"h{i}.example")
        assert len(lim._buckets) <= 4096


class TestDomainKeyNormalization68:
    def test_default_port_and_trailing_dot_do_not_split_a_host(self):
        # Bug 151: "https://example.com:443/" and "https://example.com./"
        # keyed separate limiter buckets, profiles and clearance entries
        # from "https://example.com/" -- one host, two rate budgets, and a
        # clearance banked under one spelling never replayed for the other.
        from searchio.net.ladder import domain_of

        assert domain_of("https://example.com:443/x") == "example.com"
        assert domain_of("http://example.com:80/x") == "example.com"
        assert domain_of("https://example.com./x") == "example.com"
        assert domain_of("https://example.com:8443/x") == "example.com:8443"


class TestRobotsPercentEncoding68:
    # Bug 152 (Muse second pass on robots.py, confirmed by probe): under
    # `enforce`, "/%61dmin" slipped past "Disallow: /admin" and "/café"
    # past "Disallow: /caf%C3%A9". RFC 9309 s2.2.2: percent-encoded
    # unreserved octets are decoded before matching and non-ASCII is
    # compared in its UTF-8 percent-encoded form; both sides are normalized
    # to one spelling now.
    @pytest.mark.parametrize("rule,path,allowed", [
        ("/admin", "/%61dmin", False), ("/admin", "/admin", False), ("/admin", "/Admin", True),
        ("/caf%C3%A9", "/café", False), ("/café", "/caf%C3%A9", False),
        ("/a%20b", "/a b", False), ("/a%20b", "/a%20b", False),
        ("/a%2Fb", "/a/b", True), ("/a/b", "/a%2Fb", True),
    ])
    def test_encoded_forms_match(self, rule, path, allowed):
        from searchio.net.robots import _allowed, _group_for, parse_robots

        g = _group_for(parse_robots(f"User-agent: *\nDisallow: {rule}\n"), "searchio")
        assert _allowed(g, path) is allowed
