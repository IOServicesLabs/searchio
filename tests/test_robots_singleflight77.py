"""Concurrent robots checks for one origin share a single fetch (iteration
77, Muse concurrency review of robots.py): check() did check-then-act with
no single-flight, so N concurrent fetches to one host each fetched
/robots.txt -- the politeness layer hammering the origin's robots.txt N
times per wave. Red first."""

from __future__ import annotations

import asyncio

import pytest

from searchio.net.robots import RobotsCache

ROBOTS = "User-agent: *\nDisallow: /private\nCrawl-delay: 1\n"


def _counting_cache(delay: float = 0.05, policy: str = "enforce"):
    calls: list[str] = []

    async def fetch(url: str):
        calls.append(url)
        await asyncio.sleep(delay)  # a real robots.txt fetch takes time
        return 200, ROBOTS
    return RobotsCache(fetch, policy=policy), calls


class TestConcurrentChecksShareOneFetch:
    async def test_ten_concurrent_same_origin_checks_fetch_robots_once(self):
        rc, calls = _counting_cache()
        infos = await asyncio.gather(*(
            rc.check(f"https://ex.com/page{i}") for i in range(10)
        ))
        assert len(calls) == 1, f"robots.txt fetched {len(calls)} times: {calls}"
        # Every caller still got a correct verdict from the one fetch.
        assert all(i.allowed for i in infos)
        assert (await rc.check("https://ex.com/private")).allowed is False

    async def test_different_origins_still_fetch_independently(self):
        rc, calls = _counting_cache()
        await asyncio.gather(
            rc.check("https://a.com/x"), rc.check("https://b.com/y"),
            rc.check("https://a.com/z"),
        )
        assert sorted(calls) == ["https://a.com/robots.txt", "https://b.com/robots.txt"], calls

    async def test_a_second_wave_uses_the_cache_not_a_refetch(self):
        rc, calls = _counting_cache()
        await asyncio.gather(*(rc.check(f"https://ex.com/p{i}") for i in range(4)))
        await asyncio.gather(*(rc.check(f"https://ex.com/q{i}") for i in range(4)))
        assert len(calls) == 1, calls

    async def test_a_failed_load_shares_one_fetch_and_clears_the_slot(self):
        # A fetch that raises is caught into a robots-unavailable allow (the
        # pre-existing design); the concurrent wave must still fetch once and
        # the in-flight slot must clear so nothing wedges.
        calls: list[str] = []

        async def fetch(url: str):
            calls.append(url)
            await asyncio.sleep(0.02)
            raise RuntimeError("network down")
        rc = RobotsCache(fetch, policy="enforce")
        infos = await asyncio.gather(*(rc.check(f"https://ex.com/p{i}") for i in range(5)),
                                     return_exceptions=True)
        assert all(not isinstance(r, BaseException) for r in infos), infos
        assert all(i.allowed for i in infos), "a robots-unavailable host is allowed"
        assert len(calls) == 1, f"one fetch for the wave, got {calls}"
        assert rc._inflight == {}, "the in-flight slot is cleared"
