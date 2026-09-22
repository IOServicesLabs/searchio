"""A redirect to an opaque non-http scheme is a contractual refusal, not a
raw httpx error (iteration 76, false-green suite review): httpx 0.28 builds
the redirect request eagerly even with follow_redirects=False and raises
InvalidURL on a `javascript:` Location before _open_stream's manual guard
sees it -- so the cheap-tier scheme guard was bypassed and the refusal came
out as httpx.InvalidURL instead of TargetRefused. Red first."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from searchio.config import Settings
from searchio.errors import TargetRefused
from searchio.net.ladder import Ladder


def _server(location: str):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/jump"


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, robots_policy="off",
                    max_tier=0, sidecar_autostart=False)


class TestAnOpaqueSchemeRedirectRefusesContractually:
    @pytest.mark.parametrize("location", [
        "javascript:alert(1)",
        "javascript:/*x*/void(0)",
        "data:text/html,<script>1</script>",
    ], ids=["js", "js-comment", "data"])
    async def test_the_cheap_tier_raises_target_refused_not_invalidurl(self, settings, location):
        httpd, url = _server(location)
        lad = Ladder(settings)
        try:
            with pytest.raises(TargetRefused) as exc:
                await lad.fetch(url, use_cache=False)
            assert not isinstance(exc.value, httpx.InvalidURL)
            assert "target_refused" in str(exc.value)
        finally:
            await lad.close()
            httpd.shutdown()

    async def test_a_normal_http_redirect_still_follows(self, settings):
        # The fix must not turn a good http redirect into a refusal.
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/jump":
                    self.send_response(302)
                    self.send_header("Location", "/dest")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    body = b"<html><body><p>" + b"landed here " * 40 + b"</p></body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, *a):
                pass
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}/jump"
        lad = Ladder(settings)
        try:
            res = await lad.fetch(url, use_cache=False)
            assert res.status == 200 and "landed here" in res.body
        finally:
            await lad.close()
            httpd.shutdown()
