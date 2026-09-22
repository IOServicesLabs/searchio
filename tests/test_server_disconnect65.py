"""A client that hangs up on /research must not leave the swarm running
(iteration 65). Red first.

The streaming endpoint already cancels its driver when the generator is
torn down; the plain POST handler never looked at the disconnect message
uvicorn delivers, so a swarm kept burning tokens for minutes after the
caller was gone.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import searchio.server as server


class _Eng:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = False
        self.finished = False
        self.s = None

    async def research(self, question, *, progress=None):
        self.started.set()
        try:
            await asyncio.sleep(5)
            self.finished = True
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        from searchio.models import ResearchResult

        return ResearchResult(query=question, answer="late")


async def _call(app, path: str, body: dict, disconnect_after: float):
    """Drive the ASGI app the way uvicorn does: a body message, then an
    http.disconnect once the client hangs up."""
    payload = json.dumps(body).encode()
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"content-type", b"application/json"),
                                          (b"content-length", str(len(payload)).encode()),
                                          (b"host", b"api")],
             "client": ("127.0.0.1", 1), "server": ("api", 80)}
    sent_body = False
    t0 = asyncio.get_running_loop().time()

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": payload, "more_body": False}
        remaining = disconnect_after - (asyncio.get_running_loop().time() - t0)
        if remaining > 0:
            await asyncio.sleep(remaining)
        return {"type": "http.disconnect"}

    sent = []

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    return sent


class TestResearchStopsWhenTheClientHangsUp:
    async def test_disconnect_cancels_the_swarm(self, monkeypatch):
        eng = _Eng()
        monkeypatch.setattr(server, "_engine", eng)
        monkeypatch.setattr(server, "engine", lambda: eng)
        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(_call(server.app, "/research", {"question": "what is x?"}, 0.3), 4)
        elapsed = asyncio.get_running_loop().time() - t0
        assert eng.started.is_set()
        assert eng.cancelled and not eng.finished, (eng.cancelled, eng.finished)
        assert elapsed < 2.0, f"handler ran on for {elapsed:.1f}s after the client left"

    async def test_connected_client_gets_the_result(self, monkeypatch):
        eng = _Eng()

        async def quick(question, *, progress=None):
            from searchio.models import ResearchResult

            return ResearchResult(query=question, answer="ok")
        eng.research = quick
        monkeypatch.setattr(server, "engine", lambda: eng)
        sent = await _call(server.app, "/research", {"question": "what is x?"}, 60)
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 200
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        assert json.loads(body)["answer"] == "ok"


class TestReadStopsWhenTheClientHangsUp:
    async def test_disconnect_cancels_the_fetch(self, monkeypatch):
        from types import SimpleNamespace

        state = {"cancelled": False, "finished": False}

        async def fetch(url, **kw):
            try:
                await asyncio.sleep(5)
                state["finished"] = True
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
        eng = SimpleNamespace(ladder=SimpleNamespace(fetch=fetch))
        monkeypatch.setattr(server, "engine", lambda: eng)
        scope_path = "/read?url=https%3A%2F%2Fexample.com%2Fslow"
        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(_get(server.app, scope_path, 0.3), 4)
        assert state["cancelled"] and not state["finished"]
        assert asyncio.get_running_loop().time() - t0 < 2.0


async def _get(app, path_qs: str, disconnect_after: float):
    path, _, qs = path_qs.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": qs.encode(),
             "root_path": "", "headers": [(b"host", b"api")], "client": ("127.0.0.1", 1), "server": ("api", 80)}
    sent_body = False
    t0 = asyncio.get_running_loop().time()

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": b"", "more_body": False}
        remaining = disconnect_after - (asyncio.get_running_loop().time() - t0)
        if remaining > 0:
            await asyncio.sleep(remaining)
        return {"type": "http.disconnect"}

    sent = []

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    return sent



class TestStreamKeepsTheConnectionAlive:
    async def test_silent_phase_emits_keepalive_comments(self, monkeypatch):
        # Bug 145: a planning or worker phase can be silent for a minute or
        # more; with nothing on the wire an idle-timeout proxy closes the SSE
        # stream. A comment line every few seconds keeps it open and is
        # invisible to event parsers.
        from types import SimpleNamespace

        async def research(question, *, progress=None):
            await asyncio.sleep(1.2)
            from searchio.models import ResearchResult

            return ResearchResult(query=question, answer="ok")
        eng = SimpleNamespace(research=research)
        monkeypatch.setattr(server, "engine", lambda: eng)
        monkeypatch.setattr(server, "SSE_KEEPALIVE_S", 0.3)
        sent = await _call(server.app, "/research/stream", {"question": "what is x?"}, 60)
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
        assert body.count(": keepalive") >= 2, body[:300]
        assert "event: result" in body
