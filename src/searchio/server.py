"""HTTP API.

One Engine for the process lifetime, not one per request. That is the whole
design note: the domain profiles, circuit breakers, adaptive rate limits and
content cache are all things that get *better* the longer they live, and a
per-request engine throws away every bit of learning between calls -- including
the knowledge that a domain needs tier 2, which is what stops the server
hammering a site it already knows will refuse it.

``/research`` streams Server-Sent Events because a swarm run legitimately takes
tens of seconds and a silent connection for that long is indistinguishable from
a hang.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Request, FastAPI, HTTPException, Query as QueryParam
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from .config import get_settings
from .engine import Engine
from .errors import Blocked, ConfigError, SearchioError, TargetRefused

_engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = Engine(get_settings())
    try:
        yield
    finally:
        await _engine.close()
        _engine = None


app = FastAPI(
    title="searchio",
    version="0.1.0",
    description="LLM-assisted search over a blocking-resistant acquisition ladder.",
    lifespan=lifespan,
)


def _http_error(exc: Exception) -> HTTPException:
    """One stable mapping for every route (bug 75): a refusal is the
    caller's problem (4xx), a block is 451, anything else upstream is 502.
    An agent programs against these codes; a bare 500 tells it nothing."""
    if isinstance(exc, Blocked):
        return HTTPException(451, f"blocked by {exc.vendor or 'unknown'}: {exc.reason}")
    if isinstance(exc, TargetRefused):
        return HTTPException(400, f"target_refused: {exc}")
    if isinstance(exc, ConfigError):
        return HTTPException(400, str(exc))
    if isinstance(exc, SearchioError):
        return HTTPException(502, f"{type(exc).__name__}: {exc}")
    if isinstance(exc, ValidationError):
        return HTTPException(422, str(exc).splitlines()[0][:300])
    return HTTPException(500, f"{type(exc).__name__}: {exc}"[:300])


def engine() -> Engine:
    if _engine is None:  # pragma: no cover - only before startup
        raise HTTPException(503, "engine not ready")
    return _engine


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "service": "searchio"}


@app.get("/search")
async def search(
    q: str = QueryParam(..., min_length=1, max_length=2000, description="Query text."),
    intent: str = QueryParam("web", max_length=32),
    k: int = QueryParam(10, ge=1, le=50),
    domains: str = QueryParam("", max_length=2000, description="Comma-separated allowlist."),
    freshness: str = QueryParam("all", max_length=8,
                                description="all | day | week | month | year: how recent the results must be."),
    locale: str = QueryParam("", max_length=35,
                             description="Language/market of the question, e.g. de-DE; providers search that web."),
    fresh: bool = QueryParam(False, description="Bypass the cache."),
) -> dict:
    # The knobs the engine and the agent tool already had, reachable over
    # HTTP too (bug 144); an unknown value is the caller's fault, not a
    # silent default. Blank text and capitalized intents are what agents
    # send: fold, then refuse only what is really empty.
    q = q.strip()
    if not q:
        raise HTTPException(422, "q is blank")
    intent = intent.strip().lower()
    freshness = freshness.strip().lower()
    if freshness not in ("all", "day", "week", "month", "year"):
        raise HTTPException(422, f"freshness must be all|day|week|month|year, got {freshness!r}")
    eng = engine()
    try:
        res = await eng.search(
            q, intent=intent, k=k,
            domains=[d.strip() for d in domains.split(",") if d.strip()],
            freshness=freshness, locale=locale, fresh=fresh,
        )
    except (SearchioError, ValidationError) as exc:
        raise _http_error(exc) from exc
    skipped = dict(getattr(res, "skipped", None) or {})
    if not res.docs and not res.used and (res.failed or skipped):
        # Nobody answered: not an honest empty (bug 75). 503 says "retry
        # later" and names what failed, which is what an agent needs. A
        # fan-out where everyone sat circuit-open is the same story (bug
        # 164): it used to be 200 + [] with failed={} -- an empty web.
        raise HTTPException(503, {"error": ("all providers failed" if res.failed
                                            else "all providers cooling down"),
                                  "providers_failed": dict(res.failed),
                                  "providers_skipped": skipped})
    return {
        "query": q,
        "intent": intent,
        "elapsed_ms": res.elapsed_ms,
        "providers_used": res.used,
        "providers_failed": res.failed,
        "providers_skipped": skipped,
        "results": [d.model_dump(exclude={"text"}) for d in res.docs],
    }


def _refuse_if_cooling(eng, intent: str) -> None:
    """503 when an empty item result is a provider cooldown, not an absence.

    find_items returns [] both for an honest empty and for every provider
    that serves the intent sitting circuit-open (find_items raises only on a
    total refusal, bug 130). /search already answers 503 + providers_skipped
    on a cooldown (bug 164); /items and /marketplace used to return 200 + []
    and hide it -- the HTTP twin of bug 182. A genuine empty (nobody cooling)
    stays a 200 with an empty list.
    """
    cooling = getattr(eng.router, "_cooling", lambda _i: {})(intent)
    if cooling:
        raise HTTPException(503, {"error": "all providers cooling down",
                                  "providers_cooling": cooling})


@app.get("/items")
async def items(
    q: str = QueryParam(..., min_length=1, max_length=2000),
    k: int = QueryParam(10, ge=1, le=50),
    domains: str = QueryParam("", max_length=2000),
    browser: bool = QueryParam(False, description="Allow the stealth browser tier."),
) -> dict:
    eng = engine()
    try:
        found = await eng.find_items(
            q, k=k, use_browser=browser,
            domains=[d.strip() for d in domains.split(",") if d.strip()] or None,
        )
    except (SearchioError, ValidationError) as exc:
        raise _http_error(exc) from exc
    if not found:
        _refuse_if_cooling(eng, "shopping")  # bug 183
    return {"query": q, "count": len(found), "items": [i.model_dump() for i in found]}


@app.get("/marketplace")
async def marketplace(
    q: str = QueryParam(..., min_length=1, max_length=2000),
    k: int = QueryParam(20, ge=1, le=50),
    city: str = QueryParam("", max_length=100,
                           description="Marketplace city slug, e.g. nyc. "
                                       "Without one, results follow the exit IP."),
) -> dict:
    """Facebook Marketplace classifieds. Browser-backed: expect ~15-30s."""
    eng = engine()
    try:
        found = await eng.find_local_items(q, city=city, k=k)
    except (SearchioError, ValidationError) as exc:
        raise _http_error(exc) from exc
    if not found:
        _refuse_if_cooling(eng, "local")  # bug 183
    return {
        "query": q,
        "city": city or "(exit-ip)",
        "count": len(found),
        "items": [i.model_dump() for i in found],
    }


async def _until_disconnect(request: Request, task: asyncio.Task) -> None:
    """Cancel ``task`` when the client hangs up (bug 142).

    Starlette does not cancel a handler on disconnect; the message sits on
    the receive channel until someone reads it. The streaming endpoint
    learned this through its generator teardown; the plain handler ran a
    swarm to completion for a caller that was long gone.
    """
    while not task.done():
        if await request.is_disconnected():
            task.cancel()
            return
        await asyncio.sleep(0.25)


@app.get("/read")
async def read(url: str = QueryParam(..., max_length=4096,
                                     description="Absolute URL to fetch."),
               session: str = QueryParam("", max_length=64, pattern=r"^[A-Za-z0-9_.:@-]*$",
                                         description="Tenant/session id: cookies banked by "
                                                     "reads with this id never ride on another's."),
               request: Request = None) -> dict:
    eng = engine()
    task = asyncio.ensure_future(eng.ladder.fetch(url, session=session))
    watcher = asyncio.ensure_future(_until_disconnect(request, task)) if request is not None else None
    try:
        res = await task
    except asyncio.CancelledError:
        # A walled page can hold a browser rescue for a minute; nobody is
        # waiting for it any more (bug 142).
        raise HTTPException(status_code=499, detail="client disconnected") from None
    except SearchioError as exc:
        # 451 for a block (refused by an intermediary, not the origin's
        # logic), 400 for a target the policy refuses, 502 otherwise.
        raise _http_error(exc) from exc

    finally:
        if watcher is not None:
            watcher.cancel()

    from .extract import title_of, to_markdown

    return {
        "url": res.final_url or url,
        "status": res.status,
        "tier": res.tier,
        "via": res.via,
        "elapsed_ms": res.elapsed_ms,
        "escalations": res.escalations,
        "title": title_of(res.body),
        "text": to_markdown(res.body, url),
    }


@app.post("/research")
async def research(req: ResearchRequest, request: Request) -> dict:
    eng = engine()
    task = asyncio.ensure_future(eng.research(req.question))
    watcher = asyncio.ensure_future(_until_disconnect(request, task))
    try:
        result = await task
    except asyncio.CancelledError:
        # The caller left; there is nobody to answer. A 499-style close is
        # the honest outcome, and uvicorn discards the response anyway.
        raise HTTPException(status_code=499, detail="client disconnected") from None
    except SearchioError as exc:
        raise _http_error(exc) from exc
    finally:
        watcher.cancel()
    return result.model_dump()


#: Seconds of silence before the SSE stream writes a comment line (bug 145):
#: a planning or worker phase can be quiet for a minute or more, and an
#: idle-timeout proxy closes a stream with nothing on the wire.
SSE_KEEPALIVE_S = 15.0


@app.post("/research/stream")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    """Run the swarm, streaming progress as Server-Sent Events."""
    eng = engine()
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    async def progress(event: str, data: dict) -> None:
        await queue.put((event, data))

    async def drive() -> None:
        try:
            result = await eng.research(req.question, progress=progress)
            await queue.put(("result", json.loads(result.model_dump_json())))
        except Exception as exc:
            await queue.put(("error", {"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            await queue.put(None)

    async def stream():
        task = asyncio.create_task(drive())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), SSE_KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                event, data = item
                yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
        finally:
            # A client that disconnects mid-run must not leave a swarm burning
            # tokens with nobody listening.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/providers")
async def providers() -> dict:
    return {"providers": engine().router.health_report()}


@app.get("/stats")
async def stats() -> dict:
    eng = engine()
    return {
        **eng.stats(),
        "domains": [
            {
                "domain": p.domain,
                "min_tier": p.min_tier,
                "successes": p.successes,
                "blocks": p.blocks,
                "vendor": p.last_block_vendor,
                "block_rate": round(p.block_rate, 3),
            }
            for p in eng.ladder.profiles.all()[:50]
        ],
    }
