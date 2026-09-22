"""A block must not discard a clearance it never replayed (iteration 65).
Red first."""

from __future__ import annotations

import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from searchio.config import Settings
from searchio.errors import Blocked
from searchio.net.clearance import ClearanceStore
from searchio.net.ladder import Ladder


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        time.sleep(0.6)
        body = b"<html><body><h1>Access Denied</h1></body></html>"
        self.send_response(403)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def port():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class TestStoreDropIsScoped:
    def test_drop_older_than_keeps_a_newer_capture(self, tmp_path):
        store = ClearanceStore(tmp_path / "c.db")
        store.put("example.com", {"cf_clearance": "old"}, "UA")
        t_attempt = time.time() + 0.01
        time.sleep(0.02)
        store.put("example.com", {"cf_clearance": "new"}, "UA")
        store.drop("example.com", older_than=t_attempt)
        assert store.get("example.com").cookies["cf_clearance"] == "new"

    def test_drop_older_than_removes_the_stale_one(self, tmp_path):
        store = ClearanceStore(tmp_path / "c.db")
        store.put("example.com", {"cf_clearance": "old"}, "UA")
        time.sleep(0.02)
        store.drop("example.com", older_than=time.time())
        assert store.get("example.com") is None

    def test_plain_drop_still_drops(self, tmp_path):
        store = ClearanceStore(tmp_path / "c.db")
        store.put("example.com", {"cf_clearance": "x"}, "UA")
        store.drop("example.com")
        assert store.get("example.com") is None


class TestLadderBlockDoesNotDiscardAConcurrentCapture:
    async def test_block_keeps_clearance_banked_during_the_attempt(self, tmp_path, port):
        # Bug 143 (concurrency review drop list): caller A's cheap-tier 403
        # dropped the domain's clearance unconditionally -- including one
        # caller B had just earned with a browser while A's request was in
        # flight. The next fetch paid for another browser.
        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=0, robots_policy="off",
                     sidecar_autostart=False)
        lad = Ladder(s)
        dom = f"127.0.0.1:{port}"
        try:
            async def a():
                with pytest.raises(Blocked):
                    await lad.fetch(f"http://{dom}/walled", use_cache=False)

            async def b():
                await asyncio.sleep(0.2)
                lad.clearance.put(dom, {"cf_clearance": "fresh"}, "UA")

            await asyncio.gather(a(), b())
            c = lad.clearance.get(dom)
            assert c is not None and c.cookies["cf_clearance"] == "fresh"
        finally:
            await lad.close()
