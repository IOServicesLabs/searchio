"""Head-of-line starvation across domains (iteration 64). Red first."""

from __future__ import annotations

import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder

PROSE = "<html><body>" + "<p>Real prose about kayaks and rivers and paddles.</p>" * 30 + "</body></html>"


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
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


class TestThrottledDomainDoesNotStarveOthers:
    async def test_unrelated_domain_is_not_queued_behind_a_throttled_one(self, tmp_path, port):
        # Bug 139 (DeepSeek concurrency drop list, confirmed by probe): the
        # global concurrency slot was held while waiting for the per-domain
        # permit and during the crawl-delay sleep, so callers queued on one
        # throttled host held every slot and an unrelated host waited 3.7 s
        # for nothing. The permit and the delay come before the slot now.
        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=0, robots_policy="off",
                     sidecar_autostart=False, per_domain_rps=0.5, per_domain_burst=1,
                     per_domain_max_rps=0.5, global_concurrency=4)
        lad = Ladder(s)
        try:
            async def throttled(i):
                await lad.fetch(f"http://127.0.0.1:{port}/a{i}", use_cache=False)

            async def unrelated():
                await asyncio.sleep(0.3)
                t0 = time.monotonic()
                await lad.fetch(f"http://localhost:{port}/b", use_cache=False)
                return time.monotonic() - t0

            res = await asyncio.gather(*(throttled(i) for i in range(6)), unrelated())
            assert res[6] < 1.0, f"unrelated domain waited {res[6]:.2f}s behind a throttled one"
        finally:
            await lad.close()
