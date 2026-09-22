"""Ladder lifecycle under concurrency (iteration 63 probes). Red first."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder


class TestClosedLadderStaysClosed:
    async def test_fetch_after_close_raises(self, tmp_path):
        # Bug 137: a closed Ladder silently recreated its HTTP client on the
        # next fetch and kept serving -- resources nobody will close again
        # (the shape of bug 103 from the other side). A closed ladder must
        # say so.
        lad = Ladder(Settings(state_dir=tmp_path, cache_enabled=False, max_tier=0, robots_policy="off"))
        await lad.close()
        with pytest.raises(RuntimeError, match="closed"):
            await lad.fetch("http://127.0.0.1:9/never", use_cache=False)
        assert lad._http is None

    async def test_close_is_idempotent(self, tmp_path):
        lad = Ladder(Settings(state_dir=tmp_path, cache_enabled=False, max_tier=0, robots_policy="off"))
        await lad.close()
        await lad.close()
