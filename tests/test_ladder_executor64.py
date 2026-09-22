"""Tier 1 must not depend on the loop's default thread pool (iteration 64)."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder

PROSE = "<html><body>" + "<p>Real prose about kayaks and rivers and paddles.</p>" * 30 + "</body></html>"


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        time.sleep(0.8)
        body = PROSE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def port():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class TestTierOneHasItsOwnPool:
    async def test_six_tier1_fetches_overlap_on_a_two_thread_default_executor(self, tmp_path, port):
        # Bug 140 (concurrency drop list): tier 1 ran through
        # asyncio.to_thread, i.e. the loop's DEFAULT executor -- on a 4-core
        # host that is 8 threads for 24 concurrency slots, and PDF extraction
        # and DNS checks queue behind the same threads. The ladder now owns a
        # pool sized to its own concurrency.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=2))
        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=1, robots_policy="off",
                     sidecar_autostart=False, per_domain_rps=50, per_domain_burst=50,
                     per_domain_max_rps=50, global_concurrency=8)
        lad = Ladder(s)
        try:
            t0 = time.monotonic()
            res = await asyncio.gather(*(lad.fetch(f"http://127.0.0.1:{port}/p{i}", use_cache=False, force_tier=1)
                                         for i in range(6)))
            dt = time.monotonic() - t0
            assert all(r.tier == 1 for r in res)
            assert dt < 2.0, f"six 0.8 s tier-1 fetches took {dt:.1f}s (serialized on the default executor)"
        finally:
            await lad.close()
