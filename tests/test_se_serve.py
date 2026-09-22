"""Wire-level parity test: searchio's SidecarClient against the Rust se-serve.

Spawns the searchio-engine sidecar binary, serves the real Facebook capture
over a local HTTP server, and drives the whole verb surface searchio's
providers rely on. This is the integration contract from
``docs/sidecar-protocol.md`` exercised from the Python side.

Skipped automatically when the engine binary has not been built yet.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import http.server
import html
import json
import re
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from searchio.net.sidecar import SidecarClient
from searchio.providers.facebook import parse_listings
from searchio.providers.youtube import extract_initial_data, parse_videos

FIXTURE = Path(__file__).parent / "fixtures" / "fb_marketplace.html"
ENGINE_BIN = (
    Path(__file__).parents[2]
    / "searchio-engine"
    / "target"
    / "debug"
    / "se-serve.exe"
)
ITEM_ID = re.compile(r"/marketplace/item/(\d+)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@functools.lru_cache(maxsize=None)
def serve_fixture() -> tuple[str, str]:
    """Serve the fixtures dir; return (base_url, shutdown_token)."""
    base = f"http://127.0.0.1:{free_port()}"
    root = FIXTURE.parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def log_message(self, *a):  # quiet
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(base.rsplit(":", 1)[1])), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return base, srv  # type: ignore[return-value]


@functools.lru_cache(maxsize=None)
def echo_base() -> str:
    """A server that answers with the caller's Cookie header as JSON — the
    wire proof that an injected session actually rides along."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {
                "cookie": self.headers.get("Cookie", ""),
                "user_agent": self.headers.get("User-Agent", ""),
                "accept": self.headers.get("Accept", ""),
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass


# Hand-built 2x3 PNG (signature + IHDR + bare IEND tail) — the engine's
# container sniffer reads BE dims at offsets 16/20, so no pixel data is
# needed. The wire test asserts the bytes round-trip byte-for-byte.
_PNG_2X3 = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00\x00\x00\x13IHDR"
    + (2).to_bytes(4, "big")
    + (3).to_bytes(4, "big")
    + b"\x08\x02\x00\x00\x00"
    + b"\x00\x00\x00\x00IEND"
)
# Minimal GIF89a logical screen descriptor (1x1) for the data: URL image.
_GIF_1X1 = b"GIF89a" + b"\x01\x00\x01\x00" + b"\x00\x00\x00"
_GIF_1X1_B64 = base64.b64encode(_GIF_1X1).decode()
# Cookie headers /pixel.png saw, most recent last — the wire proof the
# session jar rode the image fetch (appended from the handler's thread).
_img_seen: list[str] = []


def _crawl_handler(tag: str, own_base: str, other_base: str):
    """Handler factory for the crawl fixture. Every page reflects the request's
    Cookie header in a data-cookies attribute (the wire proof the jar rode
    along) and sets a host-only cookie named after this server's tag; pages
    link onward with absolute hrefs so eval-based link extraction is the crawl
    primitive under test."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _page(self, cookie_header: str, links: list[str], extra_cookie: bool = False) -> bytes:
            attrs = "".join(f"<a href='{u}'>x</a>" for u in links)
            body = (
                f"<html><body data-cookies=\"{cookie_header}\">{attrs}</body></html>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", f"crawl_{tag}=1; Path=/")
            if extra_cookie:
                self.send_header("Set-Cookie", f"crawl_{tag}_2=2; Path=/")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            cookies = self.headers.get("Cookie", "")
            if self.path == "/start":
                self._page(cookies, [f"{own_base}/same", f"{other_base}/x", f"{own_base}/redir1"])
            elif self.path == "/same":
                # Second cookie proves jar ACCUMULATION across same-origin hops.
                self._page(cookies, [f"{other_base}/x"], extra_cookie=True)
            elif self.path == "/x":
                self._page(cookies, [f"{other_base}/back"])
            elif self.path == "/back":
                self._page(cookies, [f"{own_base}/final"])
            elif self.path == "/redir1":
                self.redirect(f"{own_base}/redir2")
            elif self.path == "/redir2":
                self.redirect(f"{own_base}/final")
            elif self.path == "/redirc1":
                # Cookie-SETTING redirect chain: each 302 hop plants a cookie
                # the next hop's request must carry (per-hop jar feeding).
                self.redirect(f"{own_base}/redirc2", f"crawl_redir_hop1=1; Path=/")
            elif self.path == "/redirc2":
                self.redirect(f"{own_base}/redircfinal", f"crawl_redir_hop2=1; Path=/")
            elif self.path == "/redircfinal":
                self._page(cookies, [])
            elif self.path == "/aged":
                # Session-aging probe: a short-lived cookie (Max-Age=1, gone
                # in a second), a dead-on-arrival one (Max-Age=0), and a
                # long-lived one — the next requests classify all three.
                body = (
                    f"<html><body data-cookies=\"{cookies}\"></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Set-Cookie", "aged_short=1; Max-Age=1; Path=/")
                self.send_header("Set-Cookie", "aged_dead=1; Max-Age=0; Path=/")
                self.send_header("Set-Cookie", "aged_ok=1; Max-Age=3600; Path=/")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/aged2":
                # Overwrite probe: replaces aged_ok's value in the jar. The
                # reflection below is THIS request's cookies (pre-overwrite);
                # the NEXT request proves the replacement.
                body = (
                    f"<html><body data-cookies=\"{cookies}\"></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Set-Cookie", "aged_ok=updated; Max-Age=3600; Path=/")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/final":
                self._page(cookies, [])
            elif self.path == "/pixel.png":
                # Image probe (firing 38): hand-built 2x3 PNG. Records the
                # request's Cookie header so the wire test proves the SESSION
                # jar rode the image fetch; sets a cookie of its own so the
                # test can prove image RESPONSE cookies land in the jar too.
                _img_seen.append(cookies)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Set-Cookie", "crawl_img=1; Path=/")
                self.send_header("Content-Length", str(len(_PNG_2X3)))
                self.end_headers()
                self.wfile.write(_PNG_2X3)
            elif self.path == "/gallery":
                # The agent's-eyes fixture page: visible + hidden elements for
                # read_snapshot, an absolute <img src>, an inline-style
                # background image (relative url()), and a data: URL image
                # for the read_image tiers.
                body = (
                    "<html><body>"
                    "<nav><a href='/same'>Same</a><a href='/x'>X</a></nav>"
                    "<h1>Gallery</h1>"
                    "<p hidden>ghost</p>"
                    f"<img src='{own_base}/pixel.png' alt='dot'>"
                    "<div id='bg' style=\"background-image: url('/pixel.png')\">bg</div>"
                    f"<img src='data:image/gif;base64,{_GIF_1X1_B64}' alt='inline'>"
                    "</body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Set-Cookie", "crawl_gallery=1; Path=/")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/echo":
                # Data-call probe (firing 37): reflects method, body, and
                # selected headers so a wire fetch with method/headers/body
                # is provable attribute-by-attribute. Handles GET and POST.
                length = int(self.headers.get("Content-Length") or 0)
                req_body = self.rfile.read(length).decode("utf-8", "replace") if length else ""
                body = (
                    f"<html><body data-cookies=\"{cookies}\" "
                    f"data-method=\"{self.command}\" "
                    f"data-x-custom=\"{self.headers.get('X-Custom', '')}\" "
                    f"data-content-type=\"{self.headers.get('Content-Type', '')}\" "
                    f"data-accept=\"{self.headers.get('Accept', '')}\" "
                    f"data-body=\"{html.escape(req_body, quote=True)}\"></body></html>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Set-Cookie", "crawl_echo=1; Path=/")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):
            # The echo route answers POSTs; every other path stays GET-only
            # (a 404 for POST is the honest answer there).
            self.do_GET()

        def redirect(self, location: str, cookie: str | None = None):
            self.send_response(302)
            self.send_header("Location", location)
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


@functools.lru_cache(maxsize=None)
def crawl_bases() -> tuple[str, str]:
    """The two crawl origins. Cookies key on HOST, not port, so two servers on
    127.0.0.1 with different ports would NOT be a cookie boundary in any
    conforming store — 127.0.0.1 vs localhost is a genuine host boundary that
    both resolve to loopback."""
    srv_a = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _crawl_handler("a", "", ""))
    threading.Thread(target=srv_a.serve_forever, daemon=True).start()
    srv_b = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _crawl_handler("b", "", ""))
    threading.Thread(target=srv_b.serve_forever, daemon=True).start()
    base_a = f"http://127.0.0.1:{srv_a.server_address[1]}"
    base_b = f"http://localhost:{srv_b.server_address[1]}"
    # Rebind with the real bases so pages can emit absolute cross-origin links.
    srv_a.RequestHandlerClass = _crawl_handler("a", base_a, base_b)
    srv_b.RequestHandlerClass = _crawl_handler("b", base_b, base_a)
    return base_a, base_b


# Handshake headers each WS echo connection saw, most recent last — the
# wire proof the session jar rode the upgrade (appended from handler threads).
_ws_seen: list[dict] = []
_ws_lock = threading.Lock()


def _ws_recvn(conn, n: int) -> bytes:
    """Exactly-n byte read for the raw frame loop; short read = peer gone."""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return data
        data += chunk
    return data


@functools.lru_cache(maxsize=None)
def _ws_echo_base() -> str:
    """Raw-socket RFC 6455 echo fixture (firing 39): an INDEPENDENT second
    implementation of the wire protocol — it answers the opening handshake
    with the RFC 6455 accept and echoes text/binary frames, so the Python
    suite pins the engine's ws_* verbs off the server's side of the wire."""

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            conn = self.request
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                data += chunk
            headers = {}
            for line in data.decode("latin1").split("\r\n")[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            with _ws_lock:
                _ws_seen.append(headers)
            key = headers.get("sec-websocket-key", "")
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                ).digest()
            ).decode()
            conn.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "X-Echo: yes\r\n\r\n"
                ).encode("latin1")
            )
            # Frame loop: unmask, echo text/binary, answer ping, echo close.
            while True:
                head = _ws_recvn(conn, 2)
                if len(head) < 2:
                    return
                op = head[0] & 0x0F
                ln = head[1] & 0x7F
                if ln == 126:
                    ln = struct.unpack(">H", _ws_recvn(conn, 2))[0]
                elif ln == 127:
                    ln = struct.unpack(">Q", _ws_recvn(conn, 8))[0]
                mask = _ws_recvn(conn, 4) if head[1] & 0x80 else b""
                payload = bytearray(_ws_recvn(conn, ln))
                for i in range(ln):
                    payload[i] ^= mask[i % 4]
                if op == 0x8:  # close → echo payload, then done
                    conn.sendall(b"\x88" + bytes([len(payload)]) + bytes(payload))
                    return
                if op == 0x9:  # ping → pong
                    conn.sendall(b"\x8a" + bytes([len(payload)]) + bytes(payload))
                    continue
                # text (1) / binary (2) echo, unmasked
                if ln < 126:
                    conn.sendall(bytes([0x80 | op, len(payload)]) + bytes(payload))
                else:
                    conn.sendall(
                        bytes([0x80 | op, 126])
                        + struct.pack(">H", len(payload))
                        + bytes(payload)
                    )

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    srv = Server(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"ws://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture(scope="module")
def engine_proc():
    if not ENGINE_BIN.exists():
        pytest.skip(f"se-serve binary not built: {ENGINE_BIN}")
    port = free_port()
    proc = subprocess.Popen(
        [str(ENGINE_BIN), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    import urllib.request

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail(f"se-serve did not come up at {base}")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture()
def engine(engine_proc):
    # One client per test: httpx binds to its creating event loop, and
    # pytest-asyncio hands each test a fresh loop.
    return SidecarClient(url=engine_proc, autostart=False)


async def test_health_verb_reports_engine(engine: SidecarClient):
    r = await engine.call("health", {})
    assert r.get("ok") is True
    assert r.get("engine") == "searchio-engine"


async def test_goto_then_read_html_returns_live_dom(engine: SidecarClient):
    base, _srv = serve_fixture()
    res = await engine.goto(f"{base}/fb_marketplace.html", tab_id="fbmp")
    assert res.get("ok") is True
    assert res.get("status") == 200
    assert not res.get("bot_detected")

    html = await engine.read_html(tab_id="fbmp")
    ids = set(ITEM_ID.findall(html))
    assert len(ids) == 11, f"expected 11 distinct listings over the wire, got {len(ids)}"

    # selector-scoped read
    scoped = await engine.read_html(tab_id="fbmp", selector="a[href*='/marketplace/item/']")
    assert len(set(ITEM_ID.findall(scoped))) == 11

    # the provider's own parser sees the same listings it does from the
    # patchright sidecar's DOM
    items = parse_listings(html, source="se-serve-test")
    assert len(items) == 11
    assert all(it.price.amount is not None for it in items)


async def test_read_html_unknown_tab_is_verb_error_not_crash(engine: SidecarClient):
    r = await engine.call("read_html", {"tab_id": "nope"})
    assert r.get("ok") is False


async def test_read_text_verb_extracts_rendered_text_over_the_wire(engine: SidecarClient):
    """The protocol's text verb (firing 30): body text by default, first
    match under `selector`, with the structural innerText floor — blocks
    newline-joined, script/style subtrees excluded, whitespace collapsed."""
    base, _srv = serve_fixture()
    res = await engine.goto(f"{base}/fb_marketplace.html", tab_id="rt")
    assert res.get("ok") is True, res

    r = await engine.call("read_text", {"tab_id": "rt"})
    assert r.get("ok") is True, r
    body = r.get("text") or ""
    assert r.get("url") == f"{base}/fb_marketplace.html"
    # 1390 chars for this fixture — its bulk is script/attribute JSON, which
    # the innerText floor correctly excludes.
    assert len(body) > 1000, f"body text is substantial: {len(body)}"
    assert "\n" in body, "block boundaries became newlines"

    # The fixture's inline envjson script has real textContent; innerText
    # must exclude it (self-calibrating pin — no hardcoded fixture token).
    script_text = await engine.eval_js(
        "var s = document.getElementById('envjson'); s ? s.textContent.length : -1",
        tab_id="rt",
    )
    assert script_text and script_text > 100, f"envjson script has text: {script_text}"
    sample = await engine.eval_js(
        "document.getElementById('envjson').textContent.slice(0, 80)", tab_id="rt"
    )
    assert sample not in body, "script subtree excluded from inner_text"

    # Selector scope: first match's text is a contiguous slice of the body
    # text, and much smaller than it.
    r = await engine.call(
        "read_text", {"tab_id": "rt", "selector": "a[href*='/marketplace/item/']"}
    )
    assert r.get("ok") is True, r
    scoped = r.get("text") or ""
    assert 0 < len(scoped) < len(body), f"scoped text smaller than body: {len(scoped)}"
    assert scoped in body, "selector-scoped text rides inside the body text"

    # No-match selector yields empty text, not an error (read_html's
    # collect-over-empty contract).
    r = await engine.call("read_text", {"tab_id": "rt", "selector": "section.nope"})
    assert r.get("ok") is True, r
    assert r.get("text") == ""

    # Unknown tab is a verb error, like read_html's.
    r = await engine.call("read_text", {"tab_id": "nope"})
    assert r.get("ok") is False


async def test_eval_executes_js_over_the_wire(engine: SidecarClient):
    # about:blank semantics: no tab needed, navigator answers, arithmetic works.
    r = await engine.call("eval", {"js": "1 + 1"})
    assert r.get("ok") is True, r
    assert r.get("result") == 2

    ua = await engine.eval_js("navigator.userAgent")
    assert "Chrome/145" in ua, f"engine presented UA over the wire: {ua!r}"

    # With a page: the real fixture's DOM is live in the engine's JS tier.
    base, _srv = serve_fixture()
    await engine.goto(f"{base}/fb_marketplace.html", tab_id="fbmp")

    n = await engine.eval_js(
        "document.querySelectorAll(\"a[href*='/marketplace/item/']\").length",
        tab_id="fbmp",
    )
    assert n >= 11, f"expected >= 11 listing anchors, got {n}"

    # The provider scroll idiom: an async IIFE whose result is the answer.
    # setTimeout resolves synchronously at this tier (documented bridge shim).
    count = await engine.eval_js(
        "(async () => {"
        "await new Promise(r => setTimeout(r, 50));"
        "var ids = new Set();"
        "document.querySelectorAll(\"a[href*='/marketplace/item/']\").forEach(a => {"
        "var m = a.getAttribute('href').match(/\\/marketplace\\/item\\/(\\d+)/);"
        "if (m) ids.add(m[1]);"
        "});"
        "return ids.size;"
        "})()",
        tab_id="fbmp",
    )
    assert count == 11, f"expected exactly 11 distinct listings, got {count}"

    # Rejection surfaces as a verb error, not a transport failure.
    r = await engine.call("eval", {"js": "Promise.reject(new Error('boom'))"})
    assert r.get("ok") is False
    assert "boom" in r.get("error", "")


async def test_eval_dom_mutation_persists_across_evals_and_read_html(engine: SidecarClient):
    # Cross-eval DOM persistence: the engine's tab owns the document, so a
    # page-JS mutation survives the eval that made it — read_html and later
    # evals observe it, the way patchright's live DOM has always behaved.
    base, _srv = serve_fixture()
    await engine.goto(f"{base}/fb_marketplace.html", tab_id="mut")

    n = await engine.eval_js(
        "document.body.innerHTML += '<div id=\"se-mut\">mutation-persisted</div>';"
        "document.querySelectorAll('#se-mut').length",
        tab_id="mut",
    )
    assert n == 1

    # read_html observes the adopted document.
    html = await engine.read_html(tab_id="mut")
    assert "mutation-persisted" in html

    # A second eval queries the mutated tree.
    n = await engine.eval_js(
        "document.querySelectorAll('#se-mut').length", tab_id="mut"
    )
    assert n == 1

    # Mutate-then-throw: the eval errors, but the pre-throw mutation still
    # lands — a real tab keeps mutations a script made before throwing.
    r = await engine.call(
        "eval",
        {
            "tab_id": "mut",
            "js": "document.querySelector('#se-mut').textContent = 'pre-throw-mark';"
            "throw new Error('mutate-boom');",
        },
    )
    assert r.get("ok") is False
    # Firing 23: a sync throw now carries the exception's message over the
    # wire, just like a promise rejection — rendered by the same
    # thrown_message_text path in the bridge.
    assert "mutate-boom" in r.get("error", "")
    html = await engine.read_html(tab_id="mut")
    assert "pre-throw-mark" in html

    # And the provider's real parser still reads the untouched fixture part:
    # the mutation ADOPTED the document, it did not replace the tab's parse.
    items = parse_listings(html, source="se-serve-mutation-test")
    assert len(items) == 11


async def test_eval_sync_throw_carries_message_over_the_wire(engine: SidecarClient):
    # Firing 23: sync throws and compile errors carry their JS message over
    # the wire — the same rendered text a promise rejection carries — instead
    # of the old "<v8 gave no message>" default. A plain-string throw, a
    # TypeError, and a syntax error each get a distinct, faithful message.
    base, _srv = serve_fixture()
    await engine.goto(f"{base}/fb_marketplace.html", tab_id="throw")

    r = await engine.call(
        "eval", {"tab_id": "throw", "js": "throw new Error('wire-boom');"}
    )
    assert r.get("ok") is False
    assert "wire-boom" in r.get("error", "")

    r = await engine.call("eval", {"tab_id": "throw", "js": "throw 'plain-wire';"})
    assert r.get("ok") is False
    assert "plain-wire" in r.get("error", "")

    r = await engine.call(
        "eval", {"tab_id": "throw", "js": "null.noSuchProperty;"}
    )
    assert r.get("ok") is False
    assert "noSuchProperty" in r.get("error", "")

    # A syntax error surfaces as a compile error carrying the SyntaxError
    # text (extracted by a compile-only probe — never executes).
    r = await engine.call(
        "eval", {"tab_id": "throw", "js": "var 1bad = ;"}
    )
    assert r.get("ok") is False
    assert "<v8 gave no message>" not in r.get("error", "")
    assert "token" in r.get("error", "")

    # A non-throwing eval on the same tab still succeeds — the wrapper
    # preserves completion values.
    n = await engine.eval_js("2 + 3", tab_id="throw")
    assert n == 5


async def test_eval_set_attribute_over_the_wire(engine: SidecarClient):
    # The attribute-mutator trio (firing 22): setAttribute on a resident
    # element persists into read_html and a second eval (via the firing-21
    # writeback adoption), removeAttribute detaches it again, and the
    # createElement idiom grafts a synthetic wrapper's attribute into the
    # live tree — the firing-21 probe idiom that used to throw, now working.
    base, _srv = serve_fixture()
    await engine.goto(f"{base}/fb_marketplace.html", tab_id="attr")

    # Resident element: immediate getAttribute read-back in the same eval.
    v = await engine.eval_js(
        "document.body.setAttribute('data-se-attr', 'resident');"
        "document.body.getAttribute('data-se-attr')",
        tab_id="attr",
    )
    assert v == "resident"

    # read_html observes the adopted attribute...
    html = await engine.read_html(tab_id="attr")
    assert 'data-se-attr="resident"' in html

    # ...and a SECOND eval (fresh State over the adopted document) sees it.
    has = await engine.eval_js(
        "document.body.hasAttribute('data-se-attr')", tab_id="attr"
    )
    assert has is True

    # removeAttribute detaches it again — cross-eval, like the set.
    await engine.eval_js(
        "document.body.removeAttribute('data-se-attr'); 'removed'", tab_id="attr"
    )
    html = await engine.read_html(tab_id="attr")
    assert "data-se-attr" not in html

    # Synthetic wrapper: setAttribute before graft rides appendChild's
    # serialization into the live tree, id selector and all.
    n = await engine.eval_js(
        "var d = document.createElement('div');"
        "d.setAttribute('id', 'se-grafted');"
        "d.setAttribute('data-role', 'card');"
        "d.textContent = ' grafted';"
        "document.body.appendChild(d);"
        "document.querySelectorAll('#se-grafted').length",
        tab_id="attr",
    )
    assert n == 1
    html = await engine.read_html(tab_id="attr")
    assert 'id="se-grafted"' in html
    assert 'data-role="card"' in html

    # The provider's real parser still reads the untouched fixture part.
    items = parse_listings(html, source="se-serve-attr-test")
    assert len(items) == 11


async def test_fetch_verb_is_http_first_and_loads_tab(engine: SidecarClient):
    """The ladder's tier-2 contract: HTTP-first, ok:true envelope, 4xx rides
    inside the envelope rather than becoming a transport error."""
    base, _srv = serve_fixture()
    res = await engine.fetch(f"{base}/fb_marketplace.html", tab_id="fbmp")
    assert res.get("ok") is True, res
    assert res.get("status") == 200
    assert res.get("url") == f"{base}/fb_marketplace.html"
    assert res.get("via") == "http"
    assert not res.get("bot_wall")
    html = res.get("html") or ""
    assert len(set(ITEM_ID.findall(html))) == 11, (
        f"fetch body carries all 11 listings, got {len(set(ITEM_ID.findall(html)))}"
    )
    # The origin's headers ride the page envelope (searchio bug 166): the
    # ladder names the anti-bot vendor from them. Without them a tier-2
    # block was "blocked by unknown" whatever the origin said.
    hdrs = {str(k).lower(): v for k, v in (res.get("headers") or {}).items()}
    assert hdrs.get("content-type", "").startswith("text/html"), res.get("headers")
    assert "server" in hdrs, res.get("headers")

    # fetch leaves a loaded tab behind: eval composes over the fetched page,
    # the same way it does after the patchright sidecar's escalation.
    n = await engine.eval_js(
        "document.querySelectorAll(\"a[href*='/marketplace/item/']\").length",
        tab_id="fbmp",
    )
    assert n >= 11, f"eval over fetched tab saw {n} anchors"

    # 4xx in the ok:true envelope — the ladder's refusal check is
    # status-driven (sidecar._sidecar_refused), so a missing page must not
    # surface as ok:false.
    missing = await engine.fetch(f"{base}/no_such_page.html", tab_id="fbmp")
    assert missing.get("ok") is True, missing
    assert missing.get("status") == 404


async def test_youtube_provider_parses_over_the_wire(engine: SidecarClient):
    """The second provider workload through the whole stack: goto the real
    YT capture, read the DOM back, and run the provider's own parser on it —
    then cross-check with a page-facing script scan over the DOM bridge."""
    base, _srv = serve_fixture()
    res = await engine.goto(f"{base}/youtube_results.html", tab_id="yt")
    assert res.get("ok") is True, res
    assert res.get("status") == 200
    assert not res.get("bot_detected")

    html = await engine.read_html(tab_id="yt")
    data = extract_initial_data(html)
    assert data is not None, "ytInitialData survives the goto/read_html round trip"
    assert "contents" in data

    # the provider's whole-page parse sees the same videos it does from the
    # patchright sidecar's DOM (floor, as in test_youtube.py)
    docs = parse_videos(html)
    assert len(docs) >= 15, f"expected >= 15 videos over the wire, got {len(docs)}"
    first = docs[0]
    assert first.url.startswith("https://www.youtube.com/watch?v=")
    assert "list=" not in first.url and "&pp=" not in first.url  # canonical
    assert first.title
    assert first.meta["channel"]

    # the JS tier reads the same page: a script textContent scan over the
    # DOM bridge finds the renderers in ytInitialData.
    n = await engine.eval_js(
        "(function() {"
        "var s = document.querySelectorAll('script');"
        "for (var i = 0; i < s.length; i++) {"
        "  var t = s[i].textContent;"
        "  if (t.indexOf('ytInitialData') !== -1) {"
        "    var n = 0, p = 0;"
        "    while ((p = t.indexOf('videoRenderer', p)) !== -1) { n++; p += 13; }"
        "    return n;"
        "  }"
        "}"
        "return 0; })()",
        tab_id="yt",
    )
    assert n >= 15, f"bridge script scan found {n} renderers, want >= 15"


async def test_session_storage_state_reset_cycle(engine: SidecarClient):
    """storage_state_set injects the session into the LIVE jar: the next
    fetch carries the cookie on the wire, and cookies_get enumerates the
    real store (including server-set cookies), not a recorded shadow."""
    # the provider's real shape: the parsed JSON object, not a string
    state = {
        "cookies": [
            {
                "name": "c_user",
                "value": "61593717497664",
                "domain": ".facebook.com",
                "path": "/",
                "expires": 1893456000,  # 2030 — must not be expired
                "httpOnly": False,
                "secure": True,
            },
            {
                "name": "echo_sid",
                "value": "sess-abc",
                "domain": "127.0.0.1",  # host-only for the loopback echo server
                "path": "/",
            },
        ],
        "origins": [],
    }
    r = await engine.call("storage_state_set", {"storage_state": state})
    assert r.get("ok") is True, r
    # the provider renders f"loaded:{res['cookies']}c" from this count
    assert r.get("cookies") == 2, r

    cookies = await engine.cookies()
    by_name = {c.get("name"): c for c in cookies}
    assert "c_user" in by_name, f"live jar enumerates injected cookies: {cookies}"
    assert "echo_sid" in by_name
    assert "facebook.com" in str(by_name["c_user"].get("domain"))

    # Live injection: a fetch to the echo server arrives with the session
    # cookie already on the wire.
    res = await engine.fetch(f"{echo_base()}/anything", tab_id="echo")
    seen = json.loads(res.get("html") or "{}")
    assert "echo_sid=sess-abc" in seen.get("cookie", ""), (
        f"injected cookie rode the wire, got {seen}"
    )
    # The network session presents the SAME UA the JS tier answers for
    # navigator.userAgent — a cf_clearance cookie is bound to that exact
    # string, and a reqwest-default UA is a bot tell on its own.
    assert "Chrome/145" in seen.get("user_agent", ""), f"wire UA: {seen}"
    assert "text/html" in seen.get("accept", ""), f"wire Accept: {seen}"

    # The older string-encoded shape still lands.
    r = await engine.call(
        "storage_state_set", {"storage_state": json.dumps({"cookies": [], "origins": []})}
    )
    assert r.get("ok") is True
    assert r.get("cookies") == 0

    r = await engine.call("session_reset", {})
    assert r.get("ok") is True
    assert await engine.cookies() == []


async def test_unknown_verb_maps_to_ok_false(engine: SidecarClient):
    r = await engine.call("screenshot", {})
    assert r.get("ok") is False


async def test_close_tab_frees_the_dom(engine: SidecarClient):
    base, _srv = serve_fixture()
    await engine.goto(f"{base}/fb_marketplace.html", tab_id="temp")
    await engine.call("close_tab", {"tab_id": "temp"})
    r = await engine.call("read_html", {"tab_id": "temp"})
    assert r.get("ok") is False


async def test_crawl_queue_follows_links_across_origins(engine: SidecarClient):
    """The multi-URL crawl workload: one session, one tab, a queue of URLs
    discovered by link extraction. Proves the cross-URL behaviors per-URL
    tests can't reach: jar carry-over same-origin, NO cookie leak across the
    host boundary, jar accumulation over many hops, redirect chains, and
    per-origin Web Storage — including the A→B→A round trip restoring what
    origin A wrote (the storage-archive behavior)."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    # Hop 1 — start on origin A; its Set-Cookie lands in the session jar.
    res = await engine.goto(f"{a}/start", tab_id="crawl")
    assert res.get("ok") is True, res
    assert res.get("status") == 200

    # Link extraction through the DOM bridge — the crawl primitive.
    links = await engine.eval_js(
        "Array.from(document.querySelectorAll('a')).map(x => x.getAttribute('href'))",
        tab_id="crawl",
    )
    assert f"{a}/same" in links, f"same-origin link extracted: {links}"
    assert f"{b}/x" in links, f"cross-origin link extracted: {links}"
    assert f"{a}/redir1" in links, f"redirect-chain link extracted: {links}"

    # Seed per-origin storage on A.
    await engine.eval_js(
        "localStorage.setItem('mark','A'); sessionStorage.setItem('s','a'); 'seeded'",
        tab_id="crawl",
    )

    # Hop 2 — same origin: the jar carries A's cookie, storage carries.
    res = await engine.goto(f"{a}/same", tab_id="crawl")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="crawl")
    assert 'data-cookies="crawl_a=1"' in html, f"jar carried the cookie same-origin: {html}"
    assert await engine.eval_js("localStorage.getItem('mark')", tab_id="crawl") == "A"
    # /same sets a second cookie — the jar must accumulate, not replace.

    # Hop 3 — cross origin to B: A's cookie must NOT ride the host boundary.
    # The reflected Cookie is legitimately empty here: this is B's FIRST
    # visit, so B has no cookie to send yet — the response establishes it.
    res = await engine.goto(f"{b}/x", tab_id="crawl")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="crawl")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert "crawl_a" not in reflected, f"A's cookie leaked cross-origin: {html}"
    assert reflected == "", f"first visit to B sends no cookies: {reflected!r}"
    assert await engine.eval_js("localStorage.getItem('mark')", tab_id="crawl") is None
    await engine.eval_js("localStorage.setItem('mark','B')", tab_id="crawl")

    # Hop 3b — revisit B: the cookie B just set must now ride the wire, and
    # B's seeded localStorage must carry on a same-origin hop.
    res = await engine.goto(f"{b}/x", tab_id="crawl")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="crawl")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert reflected == "crawl_b=1", f"B's cookie rides its own origin now: {reflected!r}"
    assert await engine.eval_js("localStorage.getItem('mark')", tab_id="crawl") == "B"

    # Hop 4 — back to A: the jar carries BOTH A cookies (and not B's), and
    # A's localStorage/sessionStorage restore from the origin archive.
    res = await engine.goto(f"{a}/back", tab_id="crawl")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="crawl")
    jar_line = html.split('data-cookies="')[1].split('"')[0]
    assert "crawl_a=1" in jar_line and "crawl_a_2=2" in jar_line, (
        f"jar accumulated A's cookies over the crawl: {jar_line}"
    )
    assert "crawl_b" not in jar_line, f"B's cookie leaked into A: {jar_line}"
    assert await engine.eval_js("localStorage.getItem('mark')", tab_id="crawl") == "A", (
        "A→B→A restored origin A's localStorage from the archive"
    )
    assert await engine.eval_js("sessionStorage.getItem('s')", tab_id="crawl") == "a"

    # Hop 5 — redirect chain: navigate follows to the final landing URL.
    res = await engine.goto(f"{a}/redir1", tab_id="crawl")
    assert res.get("ok") is True, res
    assert res.get("final_url") == f"{a}/final", f"chain resolved: {res}"
    html = await engine.read_html(tab_id="crawl")
    assert 'data-cookies="' in html, f"chain landing page served: {html}"

    # The session jar enumerates everything the crawl accumulated.
    cookies = await engine.cookies()
    names = {c.get("name") for c in cookies}
    assert {"crawl_a", "crawl_a_2", "crawl_b"} <= names, f"crawl jar: {names}"

    # Reset wipes the jar AND per-origin storage (protocol contract).
    await engine.call("session_reset", {})
    assert await engine.cookies() == []


async def test_crawl_churn_and_cookie_setting_redirect_chain(engine: SidecarClient):
    """Many-URL robustness beyond the single A→B→A round trip: a 10-hop queue
    alternating the two fixture origins, where EVERY revisit must restore that
    origin's localStorage from the archive (not just the first return), the
    jar keeps each origin's cookies on its own side of the host boundary, and
    one tab stays healthy across the churn. Then a redirect chain whose 302
    hops SET cookies: per-hop cookies must land in the jar and ride the next
    hop's request — 'redirect chains on real edges,' the behavior real login
    and consent flows depend on."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})
    res = await engine.goto(f"{a}/start", tab_id="churn")
    assert res.get("ok") is True, res
    await engine.eval_js("localStorage.setItem('mark','A')", tab_id="churn")

    queue = [f"{b}/x", f"{a}/back"] * 5
    visits = {"a": 1, "b": 0}  # A already visited once via /start
    for i, url in enumerate(queue):
        tag = "b" if url.startswith(b) else "a"
        other = "a" if tag == "b" else "b"
        res = await engine.goto(url, tab_id="churn")
        assert res.get("ok") is True, f"hop {i} ({url}): {res}"
        assert res.get("status") == 200, f"hop {i}: {res}"
        html = await engine.read_html(tab_id="churn")
        reflected = html.split('data-cookies="')[1].split('"')[0]
        assert f"crawl_{other}" not in reflected, (
            f"hop {i}: cross-origin leak: {reflected!r}"
        )
        visits[tag] += 1
        if visits[tag] > 1:
            # First visit to an origin legitimately sends no cookies (the
            # response establishes them); from the second visit on, the
            # origin's own cookie must ride.
            assert f"crawl_{tag}=1" in reflected, (
                f"hop {i}: own cookie rides: {reflected!r}"
            )
        got = await engine.eval_js("localStorage.getItem('mark')", tab_id="churn")
        if i == 0:
            # B's first visit starts empty, then seeds its own mark.
            assert got is None, f"hop {i}: first B visit starts empty: {got!r}"
            await engine.eval_js("localStorage.setItem('mark','B')", tab_id="churn")
        else:
            want = "B" if tag == "b" else "A"
            assert got == want, f"hop {i}: archive restored {want!r}: {got!r}"

    # Cookie-setting redirect chain: hop1's cookie must land in the jar AND
    # ride the chain (attached to the next hop's request); hop2's is set on
    # the last response, so it lands in the jar but cannot ride the same
    # request — jar enumeration is the pin for the final hop.
    await engine.call("network_log", {"clear": True})
    res = await engine.goto(f"{a}/redirc1", tab_id="churn")
    assert res.get("ok") is True, res
    assert res.get("status") == 200, res
    assert res.get("final_url") == f"{a}/redircfinal", f"chain resolved: {res}"
    html = await engine.read_html(tab_id="churn")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert "crawl_redir_hop1" in reflected, (
        f"redirect hop1's cookie rode the chain: {reflected!r}"
    )
    names = {c.get("name") for c in await engine.cookies()}
    assert "crawl_redir_hop1" in names, f"hop1 cookie in jar: {names}"
    assert "crawl_redir_hop2" in names, f"hop2 cookie in jar: {names}"

    # The session log recorded exactly ONE document entry for the navigation —
    # the redirect hops inside the HTTP client stay collapsed into their
    # navigation's entry (the documented net_log floor), keyed by the
    # REQUESTED url — and seq never regressed.
    log = await engine.call("network_log", {})
    entries = log.get("entries") or []
    seqs = [e["seq"] for e in entries]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    docs = [e for e in entries if e.get("kind") == "document"]
    assert len(docs) == 1, f"one entry per navigation since the drain: {docs}"
    assert docs[0]["url"].endswith("/redirc1"), docs[0]
    assert docs[0]["status"] == 200, docs[0]


async def test_session_aging_expires_cookies_and_replacement_wins(engine: SidecarClient):
    """Session aging — the time dimension a single-hop crawl can't see: a
    Max-Age=1 cookie rides the immediate next request but is GONE once it
    expires; a Max-Age=0 cookie never rides and never enumerates; a replaced
    value overwrites in the jar (no duplicate name entries)."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    res = await engine.goto(f"{a}/aged", tab_id="aging")
    assert res.get("ok") is True, res
    # First visit: nothing rides yet — the responses above establish the
    # cookies; they attach to the NEXT request.
    html = await engine.read_html(tab_id="aging")
    assert html.split('data-cookies="')[1].split('"')[0] == ""

    # Immediate next request: short + long-lived ride; dead-on-arrival doesn't.
    res = await engine.goto(f"{a}/final", tab_id="aging")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="aging")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert "aged_short=1" in reflected, reflected
    assert "aged_ok=1" in reflected, reflected
    assert "aged_dead" not in reflected, reflected

    # Overwrite: /aged2 replaces aged_ok's value; it rides the NEXT request.
    res = await engine.goto(f"{a}/aged2", tab_id="aging")
    assert res.get("ok") is True, res
    res = await engine.goto(f"{a}/final", tab_id="aging")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="aging")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert "aged_ok=updated" in reflected, reflected

    # Enumeration: the jar reports the long-lived cookie, never the dead one.
    names = [c.get("name") for c in await engine.cookies()]
    assert "aged_ok" in names and "aged_dead" not in names, names

    # Past Max-Age: the short-lived cookie is gone from BOTH the wire and
    # the enumeration; the long-lived (replaced) one still rides.
    await asyncio.sleep(2)
    res = await engine.goto(f"{a}/final", tab_id="aging")
    assert res.get("ok") is True, res
    html = await engine.read_html(tab_id="aging")
    reflected = html.split('data-cookies="')[1].split('"')[0]
    assert "aged_short" not in reflected, reflected
    assert "aged_ok=updated" in reflected, reflected
    names = [c.get("name") for c in await engine.cookies()]
    assert "aged_short" not in names, names


async def test_concurrent_tabs_are_isolated_and_healthy(engine: SidecarClient):
    """The LLM-agent shape: parallel tool calls against one engine. Two tabs
    on two origins driven concurrently — every request must succeed, each tab
    must end up looking at ITS OWN page, the session jar is shared across
    tabs (a cookie set through one tab is visible through the other), and the
    engine must still be healthy afterwards. Concurrency must not cross the
    streams."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    # Round 1: concurrent cold navigations on two tabs, two origins.
    ra, rb = await asyncio.gather(
        engine.goto(f"{a}/start", tab_id="t1"),
        engine.goto(f"{b}/x", tab_id="t2"),
    )
    assert ra.get("ok") is True and ra.get("status") == 200, ra
    assert rb.get("ok") is True and rb.get("status") == 200, rb

    # Round 2: concurrent eval + read back on both tabs.
    (mark_a, mark_b), (html_a, html_b) = await asyncio.gather(
        asyncio.gather(
            engine.eval_js("localStorage.setItem('m','from-t1'); 'ok'", tab_id="t1"),
            engine.eval_js("localStorage.setItem('m','from-t2'); 'ok'", tab_id="t2"),
        ),
        asyncio.gather(
            engine.read_html(tab_id="t1"),
            engine.read_html(tab_id="t2"),
        ),
    )
    assert mark_a == "ok" and mark_b == "ok"
    # Each tab reads its own page: /start carries the /redir1 link, /x does
    # not; /x carries the /back link, /start does not. (The /start page
    # legitimately links ONTO the other origin — the crawl fixture's queue
    # source — so the mere presence of b's URL in a's page is not a mixup.)
    assert f"{a}/redir1" in html_a and f"{a}/redir1" not in html_b, (html_a, html_b)
    assert f"{a}/back" in html_b and f"{a}/back" not in html_a, (html_a, html_b)
    assert await engine.eval_js("localStorage.getItem('m')", tab_id="t1") == "from-t1"
    assert await engine.eval_js("localStorage.getItem('m')", tab_id="t2") == "from-t2"

    # Round 3: concurrent same-origin navigations from both tabs — the shared
    # jar means both requests now carry origin A's cookie.
    await asyncio.gather(
        engine.goto(f"{a}/same", tab_id="t1"),
        engine.goto(f"{a}/final", tab_id="t2"),
    )
    ha, hb = await asyncio.gather(
        engine.read_html(tab_id="t1"),
        engine.read_html(tab_id="t2"),
    )
    assert "crawl_a=1" in ha.split('data-cookies="')[1].split('"')[0], ha
    assert "crawl_a=1" in hb.split('data-cookies="')[1].split('"')[0], hb

    health = await engine.call("health", {})
    assert health.get("ok") is True


async def test_two_tabs_on_one_origin_share_localstorage(engine: SidecarClient):
    """Browser-parity Web Storage: localStorage is a per-ORIGIN pool shared
    by every tab on that origin (a headed browser's per-profile semantics) —
    a write through one tab must be visible through another tab on the same
    origin. sessionStorage keeps its spec lifetime (the top-level browsing
    context = one tab), so it stays per-tab and isolated."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    await engine.goto(f"{a}/start", tab_id="s1")
    await engine.goto(f"{a}/same", tab_id="s2")

    # s1 writes BOTH maps; s2 reads localStorage back (shared pool) and
    # sessionStorage back (its own context — must NOT see s1's value).
    assert await engine.eval_js(
        "localStorage.setItem('shared','yes'); "
        "sessionStorage.setItem('private','s1'); 'ok'",
        tab_id="s1",
    ) == "ok"
    assert await engine.eval_js("localStorage.getItem('shared')", tab_id="s2") == "yes"
    assert await engine.eval_js("sessionStorage.getItem('private')", tab_id="s2") is None

    # And the write survives a navigation on the OTHER tab — the pool entry
    # IS the origin's single map, not a per-tab copy that archives/restores.
    await engine.goto(f"{a}/final", tab_id="s2")
    assert await engine.eval_js("localStorage.getItem('shared')", tab_id="s2") == "yes"

    # Isolation the other way: origin B's pool never sees origin A's entry.
    await engine.goto(f"{b}/x", tab_id="s2")
    assert await engine.eval_js("localStorage.getItem('shared')", tab_id="s2") is None

    # Cross-check the artifact: the whole pool exports with both origins.
    # crawl_bases() yields scheme://host:port origins verbatim.
    exported = await engine.call("storage_state_get", {"tab_id": "s2"})
    assert exported.get("ok") is True, exported
    state = exported["storage_state"]
    origins = {o.get("origin"): o for o in state.get("origins") or []}
    a_entry = origins.get(a)
    assert a_entry is not None, origins
    names = [e.get("name") for e in a_entry.get("localStorage") or []]
    assert "shared" in names, names

    health = await engine.call("health", {})
    assert health.get("ok") is True


async def test_read_snapshot_and_read_image_give_agents_the_page(
    engine: SidecarClient,
):
    """The agent's-eyes tier (firing 38) over the wire. `read_snapshot` is
    the structural dump an LLM agent navigates by — visible-only, flat,
    document order, computed roles + interactivity, hidden subtrees pruned,
    selector scoping. `read_image` resolves an <img> (src or inline-style
    url(), data: URLs decoded in place) and fetches its bytes THROUGH THE
    SESSION with <img>-shaped Fetch Metadata — the fixture records what
    /pixel.png saw, so the jar ride is pinned off the server's side of the
    wire, and the response cookie landing is pinned off the NEXT goto."""
    a, b = crawl_bases()
    _img_seen.clear()
    await engine.call("session_reset", {})

    res = await engine.goto(f"{a}/gallery", tab_id="eyes")
    assert res.get("ok") is True, res

    # --- read_snapshot: whole-document structural dump.
    snap = await engine.call("read_snapshot", {"tab_id": "eyes"})
    assert snap.get("ok") is True, snap
    entries = snap["snapshot"]
    kinds = [e["kind"] for e in entries]
    # The hidden <p>ghost</p> is pruned entirely — not listed, no text.
    assert kinds == ["nav", "a", "a", "h1", "img", "div", "img"], kinds
    assert entries[0]["role"] == "navigation"
    assert entries[0]["depth"] == 0
    assert entries[1]["role"] == "link" and entries[1]["href"] == "/same"
    assert entries[1]["text"] == "Same" and entries[1]["interactive"] is True
    assert entries[3]["role"] == "heading" and entries[3]["text"] == "Gallery"
    assert all(e["text"] != "ghost" for e in entries)
    assert entries[4]["alt"] == "dot"
    assert entries[6]["alt"] == "inline"

    # --- read_snapshot: selector scopes to the first match's subtree.
    scoped = await engine.call(
        "read_snapshot", {"tab_id": "eyes", "selector": "nav"}
    )
    assert scoped.get("ok") is True, scoped
    assert [e["kind"] for e in scoped["snapshot"]] == ["nav", "a", "a"], scoped
    assert scoped["snapshot"][0]["depth"] == 0, scoped

    # --- read_image: absolute <img src> fetched through the session.
    img = await engine.call("read_image", {"tab_id": "eyes"})
    assert img.get("ok") is True, img
    assert img["format"] == "png", img
    assert img["width"] == 2 and img["height"] == 3, img
    assert img["alt"] == "dot"
    assert img["content_type"] == "image/png"
    assert img["byte_length"] == len(_PNG_2X3)
    assert base64.b64decode(img["bytes_base64"]) == _PNG_2X3

    # --- read_image: inline-style background-image url() resolves RELATIVE
    # to the tab URL (the join path), same PNG bytes.
    bg = await engine.call("read_image", {"tab_id": "eyes", "selector": "#bg"})
    assert bg.get("ok") is True, bg
    assert bg["format"] == "png" and bg["width"] == 2 and bg["height"] == 3, bg
    assert bg["src"].endswith("/pixel.png"), bg

    # --- read_image: data: URL decodes in place — NO fetch happens (the
    # fixture saw exactly the two /pixel.png hits above).
    seen_after_pixel_fetches = len(_img_seen)
    inline = await engine.call(
        "read_image", {"tab_id": "eyes", "selector": "img[alt='inline']"}
    )
    assert inline.get("ok") is True, inline
    assert inline["format"] == "gif", inline
    assert inline["width"] == 1 and inline["height"] == 1, inline
    assert inline["byte_length"] == len(_GIF_1X1)
    assert base64.b64decode(inline["bytes_base64"]) == _GIF_1X1
    assert len(_img_seen) == seen_after_pixel_fetches, _img_seen

    # --- Session jar rode BOTH image fetches (the server's side of the wire
    # saw the gallery document's cookie), and the image RESPONSE cookie
    # landed: the NEXT same-origin goto carries crawl_img=1.
    assert len(_img_seen) == 2, _img_seen
    for seen in _img_seen:
        assert "crawl_gallery=1" in seen, _img_seen
    res = await engine.goto(f"{a}/same", tab_id="eyes")
    assert res.get("ok") is True, res
    html_same = await engine.read_html(tab_id="eyes")
    cookies = html_same.split('data-cookies="')[1].split('"')[0]
    assert "crawl_img=1" in cookies, cookies

    # --- net_log: the image fetches are fetch-kind GET entries, distinct
    # from the document navigations.
    log = await engine.call("network_log", {})
    entries = log.get("entries") or []
    fetches = [e for e in entries if e.get("kind") == "fetch"]
    docs = [e for e in entries if e.get("kind") == "document"]
    assert len(fetches) == 2, entries
    assert all(e["method"] == "GET" for e in fetches), fetches
    assert all(e["url"].endswith("/pixel.png") for e in fetches), fetches
    assert len(docs) == 2 and docs[0]["url"].endswith("/gallery"), entries

    # --- Error floors: missing tab, bad selector, no match.
    r = await engine.call("read_snapshot", {"tab_id": "missing"})
    assert r.get("ok") is False, r
    r = await engine.call(
        "read_snapshot", {"tab_id": "eyes", "selector": "nav["}
    )
    assert r.get("ok") is False, r
    r = await engine.call("read_image", {"tab_id": "eyes", "selector": "#nope"})
    assert r.get("ok") is False, r

    health = await engine.call("health", {})
    assert health.get("ok") is True


async def test_ws_verbs_give_agents_a_session_socket(engine: SidecarClient):
    """The agent's WebSocket tier (firing 39) over the wire. The page-side
    WebSocket stub stays shape-only forever — the AGENT owns the socket
    through ws_connect/ws_send/ws_recv/ws_close. The raw echo fixture records
    what the upgrade actually looked like, so the test pins the session jar
    riding the handshake (cookies key on HOST, not port — server A's cookie
    reaches the WS server on its own port), the engine's Chrome UA, caller
    extra headers, text/binary echo round-trips, long-poll timeout as a
    NORMAL outcome, and the hostile battery erroring instead of panicking."""
    a, _b = crawl_bases()
    ws_base = _ws_echo_base()
    _ws_seen.clear()
    await engine.call("session_reset", {})

    # Seed the jar for host 127.0.0.1: server A's /start sets crawl_a=1.
    res = await engine.goto(f"{a}/start", tab_id="ws")
    assert res.get("ok") is True, res

    conn = await engine.call(
        "ws_connect", {"url": f"{ws_base}/feed", "headers": {"X-Auth": "tok"}}
    )
    assert conn.get("ok") is True, conn
    ws_id = conn["ws_id"]
    assert conn["status"] == 101, conn
    assert conn["response_headers"].get("x-echo") == "yes", conn

    # The server's side of the wire: the jar cookie rode the UPGRADE, the
    # engine presented its Chrome UA, and the caller's extra header arrived.
    hs = _ws_seen[-1]
    assert "crawl_a=1" in hs.get("cookie", ""), hs
    assert "Chrome" in hs.get("user-agent", ""), hs
    assert hs.get("x-auth") == "tok", hs

    # Text round-trip.
    sent = await engine.call("ws_send", {"ws_id": ws_id, "data": "hello feed"})
    assert sent.get("ok") is True and sent["sent_bytes"] == len("hello feed"), sent
    msg = await engine.call("ws_recv", {"ws_id": ws_id, "timeout_ms": 5000})
    assert msg["kind"] == "text" and msg["data"] == "hello feed", msg

    # Binary round-trip through data_base64.
    raw = bytes([0, 1, 2, 255])
    sent = await engine.call(
        "ws_send",
        {"ws_id": ws_id, "data_base64": base64.b64encode(raw).decode()},
    )
    assert sent.get("ok") is True, sent
    msg = await engine.call("ws_recv", {"ws_id": ws_id, "timeout_ms": 5000})
    assert msg["kind"] == "binary", msg
    assert base64.b64decode(msg["data_base64"]) == raw

    # Long-poll timeout is a normal outcome for agent loops, not an error.
    msg = await engine.call("ws_recv", {"ws_id": ws_id, "timeout_ms": 100})
    assert msg.get("ok") is True and msg["kind"] == "timeout", msg

    # Hostile battery: misuse errors, never panics the sidecar.
    r = await engine.call("ws_send", {"ws_id": "nope", "data": "x"})
    assert r.get("ok") is False, r
    r = await engine.call("ws_send", {"ws_id": ws_id})
    assert r.get("ok") is False, r
    r = await engine.call(
        "ws_send", {"ws_id": ws_id, "data": "x", "data_base64": "eA=="}
    )
    assert r.get("ok") is False, r
    r = await engine.call("ws_send", {"ws_id": ws_id, "data_base64": "***"})
    assert r.get("ok") is False, r
    r = await engine.call("ws_recv", {"ws_id": "nope"})
    assert r.get("ok") is False, r
    r = await engine.call("ws_connect", {"url": "http://not.ws/"})
    assert r.get("ok") is False, r

    # Clean close removes the conn; using it again is a verb error.
    closed = await engine.call("ws_close", {"ws_id": ws_id})
    assert closed.get("ok") is True and closed["closed"] is True, closed
    r = await engine.call("ws_send", {"ws_id": ws_id, "data": "gone"})
    assert r.get("ok") is False, r

    health = await engine.call("health", {})
    assert health.get("ok") is True


async def test_fetch_data_call_posts_method_headers_body_over_the_wire(
    engine: SidecarClient,
):
    """The agent-facing data tier (firing 37): `fetch` with ANY of
    method/headers/body is a session-bound DATA call, not a document load.
    The fixture's /echo reflects the request attribute-by-attribute, so the
    pin is off the wire: method, JSON body, custom headers, the SUBRESOURCE
    Accept (*/* — never the document Accept), and the session jar riding.
    The response's Set-Cookie lands in the jar; NO tab is left behind."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    # Seed a tab on origin A so the data call has an initiator AND the jar
    # holds origin A's cookie (first visit sets it; it rides the next one).
    res = await engine.goto(f"{a}/start", tab_id="page")
    assert res.get("ok") is True, res

    res = await engine.call(
        "fetch",
        {
            "url": f"{a}/echo",
            "tab_id": "data",
            "method": "POST",
            "headers": {"Content-Type": "application/json", "X-Custom": "abc"},
            "body": '{"k":1}',
        },
    )
    assert res.get("ok") is True, res
    assert res.get("status") == 200, res
    # The DATA envelope: `body`, not `html`; response headers ride along.
    assert "body" in res and "html" not in res, res.keys()
    assert "headers" in res and "content-type" in res["headers"], res

    body = res["body"]
    assert 'data-method="POST"' in body, body
    assert 'data-x-custom="abc"' in body, body
    assert 'data-content-type="application/json"' in body, body
    # Subresource shape: */* Accept, never the document text/html block.
    assert 'data-accept="*/*"' in body, body
    assert 'data-body="{&quot;k&quot;:1}"' in body, body
    # The session jar rode the data call.
    assert "crawl_a=1" in body.split('data-cookies="')[1].split('"')[0], body

    # The echo's Set-Cookie landed in the jar: the NEXT origin-A request
    # carries crawl_echo=1.
    res = await engine.goto(f"{a}/final", tab_id="page")
    assert res.get("ok") is True, res
    html_final = await engine.read_html(tab_id="page")
    assert "crawl_echo=1" in html_final.split('data-cookies="')[1].split('"')[0], html_final

    # NO tab was loaded by the data call: read_html on its tab id is the
    # missing-tab refusal.
    r = await engine.call("read_html", {"tab_id": "data"})
    assert r.get("ok") is False and "no tab" in str(r.get("error", "")), r

    # net_log: ONE fetch-kind entry with the POST method for the data call,
    # and the document navigations are separate entries.
    log = await engine.call("network_log", {})
    entries = log.get("entries") or []
    fetches = [e for e in entries if e.get("kind") == "fetch"]
    assert len(fetches) == 1, entries
    assert fetches[0]["method"] == "POST" and fetches[0]["url"].endswith("/echo"), fetches

    # Wrong-typed params refuse at the verb layer (hostile tier extension).
    for bad in (
        {"url": f"{a}/echo", "method": 7},
        {"url": f"{a}/echo", "headers": "x-custom: abc"},
        {"url": f"{a}/echo", "headers": {"X": 7}},
        {"url": f"{a}/echo", "body": 7},
    ):
        r = await engine.call("fetch", bad)
        assert isinstance(r, dict) and r.get("ok") is False, (bad, r)

    health = await engine.call("health", {})
    assert health.get("ok") is True


async def test_hostile_inputs_return_refusals_and_the_engine_survives(
    engine: SidecarClient,
):
    """A battery of malformed calls: every one must come back as a refusal
    envelope (ok:false or an RPC error) within the client timeout — never a
    hang, never a crash — and the engine must still serve a normal request
    afterwards."""
    base, _srv = serve_fixture()
    cases = [
        ("goto", {}),  # missing url
        ("goto", {"url": "not a url at all"}),  # unparsable
        ("goto", {"url": ""}),  # empty
        ("goto", {"url": "ftp://example.com/x"}),  # unsupported scheme
        ("goto", {"url": 12345}),  # wrong type
        ("fetch", {}),  # missing url
        ("eval", {}),  # missing js
        ("eval", {"js": 999}),  # wrong type
        ("read_html", {"tab_id": 123}),  # wrong type
        ("close_tab", {"tab_id": ["a", "b"]}),  # wrong type
        ("network_log", {"clear": {"nested": "object"}}),  # wrong type: not a crash
        ("totally_unknown_verb", {}),
    ]
    for verb, params in cases:
        try:
            r = await engine.call(verb, params)
            assert isinstance(r, dict), f"{verb} {params}: not a dict: {r!r}"
            # The contract: either an explicit refusal or a graceful answer
            # to a coerced-default call — never a transport-level die.
            if verb in ("goto", "fetch") and "url" in params:
                assert r.get("ok") is False, f"{verb} {params}: {r}"
        except Exception as exc:  # noqa: BLE001 — RPC error envelopes are refusals too
            assert "RPC" in str(exc) or "error" in str(exc).lower(), (
                f"{verb} {params}: unexpected exception: {exc!r}"
            )

    # The engine survived its battery: normal request still works.
    health = await engine.call("health", {})
    assert health.get("ok") is True
    res = await engine.goto(f"{base}/fb_marketplace.html", tab_id="post-hostile")
    assert res.get("ok") is True, res
    assert res.get("status") == 200


async def test_storage_state_get_exports_and_restores_the_session(engine: SidecarClient):
    """The session-save producer: storage_state_get exports the live jar +
    per-origin localStorage (current AND archived origins) as a Playwright
    storage_state artifact — and storage_state_set consumes it back, so a
    saved session restores wholesale into a fresh one."""
    a, b = crawl_bases()
    await engine.call("session_reset", {})

    # Build a session on two origins: cookies from both, localStorage on
    # both, plus sessionStorage that must NOT survive into the artifact.
    res = await engine.goto(f"{a}/start", tab_id="save")
    assert res.get("ok") is True, res
    await engine.eval_js(
        "localStorage.setItem('saved_a','1'); sessionStorage.setItem('transient','x'); 'ok'",
        tab_id="save",
    )
    res = await engine.goto(f"{b}/x", tab_id="save")
    assert res.get("ok") is True, res
    await engine.eval_js("localStorage.setItem('saved_b','2')", tab_id="save")

    exported = await engine.call("storage_state_get", {"tab_id": "save"})
    assert exported.get("ok") is True, exported
    state = exported["storage_state"]

    # Cookies: full Playwright shape, both origins' cookies present. The
    # fixture sends no SameSite attribute, which reports as "Lax" —
    # Chromium Lax-by-default, what Playwright's own export shows.
    cookies = {c["name"]: c for c in state["cookies"]}
    assert {"crawl_a", "crawl_b"} <= set(cookies), f"exported jar: {cookies}"
    assert cookies["crawl_a"]["domain"] == "127.0.0.1", cookies["crawl_a"]
    assert cookies["crawl_b"]["domain"] == "localhost", cookies["crawl_b"]
    for c in state["cookies"]:
        assert c["sameSite"] == "Lax", c

    # Origins: localStorage for BOTH the archived origin and the current
    # one; session storage never appears in the artifact.
    local = {
        o["origin"]: {item["name"]: item["value"] for item in o["localStorage"]}
        for o in state["origins"]
    }
    assert local.get(a) == {"saved_a": "1"}, f"archived origin exported: {local}"
    assert local.get(b) == {"saved_b": "2"}, f"current origin exported: {local}"
    assert "transient" not in json.dumps(state), (
        "sessionStorage excluded from the artifact"
    )

    # Round trip: reset, inject the artifact, and the session is back —
    # cookies in the jar AND the archived localStorage restored on return.
    await engine.call("session_reset", {})
    assert await engine.cookies() == []
    r = await engine.call(
        "storage_state_set", {"storage_state": state, "tab_id": "save"}
    )
    assert r.get("ok") is True and r.get("cookies") == 2, r
    names = {c.get("name") for c in await engine.cookies()}
    assert {"crawl_a", "crawl_b"} <= names, f"restored jar: {names}"
    res = await engine.goto(f"{a}/back", tab_id="save")
    assert res.get("ok") is True, res
    assert await engine.eval_js("localStorage.getItem('saved_a')", tab_id="save") == "1", (
        "archived localStorage restored from the re-injected artifact"
    )


class _RichHandler(http.server.BaseHTTPRequestHandler):
    """A comfortably-readable HTML page, so ladder classify() passes the
    result on content grounds and the test isolates the routing."""

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        body = (
            "<html><body><p>"
            + ("Readable marketplace listing description text. " * 40)
            + "</p></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@functools.lru_cache(maxsize=None)
def rich_base() -> str:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RichHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


async def test_client_binary_mode_spawns_engine_and_full_lifecycle():
    """The production wiring seam: SidecarClient(binary=...) boots se-serve
    itself. Proves the spawn answers (health verb identifies the engine, so
    no stray process on the port can fake it), serves the fetch/read_text/
    cookies/user_agent contract the ladder relies on, and close() reaps it."""
    if not ENGINE_BIN.exists():
        pytest.skip(f"se-serve binary not built: {ENGINE_BIN}")
    client = SidecarClient(
        binary=ENGINE_BIN,
        port=free_port(),
        boot_timeout_s=30.0,
        request_timeout_s=30.0,
    )
    try:
        base = await client.ensure()
        assert base == f"http://127.0.0.1:{client.port}"

        health = await client.call("health", {})
        assert health.get("engine") == "searchio-engine", (
            "the process WE spawned must be the engine, not a stray sidecar"
        )

        fbase, _srv = serve_fixture()
        res = await client.fetch(f"{fbase}/fb_marketplace.html")
        assert res.get("ok") is True and res.get("status") == 200, res
        assert "/marketplace/item/" in (res.get("html") or "")

        rt = await client.call("read_text", {"tab_id": "default"})
        assert rt.get("ok") is True and isinstance(rt.get("text"), str)

        ua = await client.user_agent()
        assert ua.startswith("Mozilla/"), f"engine navigator.userAgent: {ua!r}"

        jar = await client.cookies()
        assert isinstance(jar, list)
    finally:
        await client.close()
    assert client._proc is None, "close() must reap the spawned engine"


async def test_ladder_tier2_routes_through_engine_sidecar(tmp_path):
    """Ladder.fetch(force_tier=2) end-to-end over an engine-backed client:
    the ladder's tier-2 contract (ok/status/html/url keys, refusal envelope)
    is served by se-serve, not patchright."""
    if not ENGINE_BIN.exists():
        pytest.skip(f"se-serve binary not built: {ENGINE_BIN}")

    from searchio.config import Settings
    from searchio.net.ladder import Ladder

    settings = Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,  # keep the test fast; pacing is tested elsewhere
        per_domain_burst=1000,
    )
    client = SidecarClient(
        binary=ENGINE_BIN,
        port=free_port(),
        boot_timeout_s=30.0,
        request_timeout_s=30.0,
    )
    lad = Ladder(settings, sidecar=client)
    try:
        result = await lad.fetch(f"{rich_base()}/page", force_tier=2, use_cache=False)
        assert result.tier == 2
        assert result.via == "browser"
        assert result.status == 200
        assert "Readable marketplace listing description text." in result.body
        assert result.escalations == [], f"engine page refused? {result.escalations}"
    finally:
        await lad.close()
        # The client is the test's (bug 103: an injected client is the
        # caller's to close); the reap contract now lives on the client.
        await client.close()
    assert client._proc is None, "client close() must reap the engine it spawned"


async def test_network_log_records_document_and_subresource_hops(engine):
    """The network_log verb (firing 32): document navigations and
    page-context fetch hops land in one session ring, oldest first, and
    clear:true drains in a single round trip."""
    base, _srv = serve_fixture()
    await engine.call("network_log", {"clear": True})
    res = await engine.goto(f"{base}/fb_marketplace.html", tab_id="nl")
    assert res.get("ok") is True

    # Page-context fetch over the fixture origin (the raw-h1.1 http arm).
    status = await engine.eval_js(
        "fetch('/fb_marketplace.html').then(r => r.status)", tab_id="nl"
    )
    assert status == 200

    log = await engine.call("network_log", {})
    assert log.get("ok") is True
    entries = log.get("entries") or []
    docs = [e for e in entries if e.get("kind") == "document"]
    subs = [e for e in entries if e.get("kind") in ("fetch", "xhr")]
    assert any(
        e.get("url", "").endswith("/fb_marketplace.html") and e.get("status") == 200
        for e in docs
    ), f"document hop missing: {entries}"
    assert any(e.get("status") == 200 for e in subs), f"subresource hop missing: {entries}"
    assert all(
        {"seq", "kind", "method", "url", "status", "elapsed_ms"} <= set(e)
        for e in entries
    )
    seqs = [e["seq"] for e in entries]
    assert seqs == sorted(seqs), "entries arrive oldest-first with monotonic seq"

    # clear:true returns the drained entries, then the ring is empty.
    drained = await engine.call("network_log", {"clear": True})
    assert len(drained.get("entries") or []) == len(entries) > 0
    assert (await engine.call("network_log", {})).get("entries") == []
