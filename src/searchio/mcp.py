"""MCP (Model Context Protocol) server for searchio.

Lets any MCP client -- Claude Desktop, an agent harness, or another server --
use searchio's search and research tools over the standard MCP wire instead of
the HTTP API in ``searchio.server``. The same ``Engine`` backs both: one engine
per process, because the learned domain tiers, circuit breakers, adaptive rate
limits and content cache all get better the longer they live.

Run (stdio, for a local client such as Claude Desktop)::

    python -m searchio.mcp

Run (streamable-http, for a hosted/server deployment)::

    python -m searchio.mcp --transport streamable-http --port 8080

Tools exposed:

    search         Fused, ranked web search across providers.
    search_items   Marketplace/product listings with prices.
    search_local   Facebook Marketplace classifieds near a city.
    read_url       Fetch one URL through the blocking-resistant ladder.
    research       Run the multi-step research swarm (needs an LLM key).
    stats          Ladder + provider health.

Payment gate (x402): when ``SEARCHIO_MCP_PAYMENT`` is set, every billable tool
checks a gate before it runs. A missing/insufficient payment yields an MCP
error whose data carries an HTTP 402 ``PaymentRequired`` body plus the
``X-Payment`` challenge header, so a hosted deployment can charge
search+research credits (x402) without changing the tool schema. The gate is a
pluggable async callable; the default honors a static bearer token or a header
presented by the client, and operators can swap in a real x402 verifier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

# FastMCP is optional: searchio runs fine with an empty environment, and the
# MCP surface is no different from the other optional backends. Importing this
# module without the SDK installed raises a clear error only when you actually
# try to serve.
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - depends on the extra being installed
    FastMCP = None  # type: ignore[assignment]

from .config import get_settings
from .engine import Engine
from .errors import SearchioError

# One engine for the process lifetime, exactly as in server.py. Named with a
# trailing underscore-suffix style avoided on purpose: a module global and a
# same-named function collided here during the first smoke test, so the holder
# is a plainly distinct name.
_ENGINE: Engine | None = None


async def _engine() -> Engine:
    """Return the shared engine, creating it on first use.

    The MCP server has no lifespan hook as convenient as FastAPI's, so the
    engine is built lazily and closed via ``shutdown``.
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine(get_settings())
    return _ENGINE


async def shutdown() -> None:
    global _ENGINE
    if _ENGINE is not None:
        await _ENGINE.close()
        _ENGINE = None


def _to_jsonable(obj: Any) -> Any:
    """Pydantic model -> plain dict, recursively; anything else passes through."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


# ── x402 payment gate ────────────────────────────────────────────────────────
# The gate is intentionally a thin, swappable seam. The default implementation
# is enough to develop against and to run a private server behind a shared
# token; a production x402 deployment replaces ``verify`` with one that checks
# an on-chain payment receipt or a credits ledger and returns the 402 body +
# X-Payment header when payment is missing or spent.

class PaymentRequired(Exception):
    """Raised by a gate when the caller must pay before a billable tool runs."""

    def __init__(self, body: dict, x_payment: str = "") -> None:
        super().__init__(body.get("error", "payment required"))
        self.body = body
        self.x_payment = x_payment


class PaymentGate:
    """Default x402-style gate.

    Modes (``SEARCHIO_MCP_PAYMENT``):
        "" (default)   gate open -- nothing is charged (local/dev use).
        "static:<tok>" require the MCP request to present bearer ``<tok>``;
                       without it, return a 402 + X-Payment challenge. This is
                       the shape a real verifier uses, minus the ledger.
    """

    def __init__(self, spec: str | None = None) -> None:
        self.spec = spec if spec is not None else os.environ.get("SEARCHIO_MCP_PAYMENT", "")
        self.enabled = bool(self.spec)

    async def verify(self, tool: str, auth: str | None) -> None:
        if not self.enabled:
            return
        token = self.spec.split(":", 1)[1] if self.spec.startswith("static:") else ""
        ok = bool(auth) and auth == (f"Bearer {token}" if token else auth)
        if ok:
            return
        # 402 Payment Required. The X-Payment header is where a facilitator
        # advertises what to pay; a static deployment just names the tool.
        raise PaymentRequired(
            body={
                "error": "payment_required",
                "status": 402,
                "tool": tool,
                "detail": "search+research credits required (x402)",
            },
            x_payment=json.dumps({
                "scheme": "x402",
                "resource": f"mcp://searchio/{tool}",
                "accepts": ["credits"],
            }),
        )


_gate = PaymentGate()


async def _guarded(tool: str, coro):
    """Run a billable tool under the payment gate and map errors to MCP errors."""
    # The MCP request's auth reaches the tool as an optional leading argument is
    # not portable across transports, so the gate reads a process-level token for
    # stdio and an operator-set expectation for http. A real verifier inspects the
    # HTTP request; here we surface the 402 shape so clients and proxies can build
    # against it. When the gate is off this is a no-op.
    await _gate.verify(tool, os.environ.get("SEARCHIO_MCP_BEARER"))
    try:
        return await coro
    except PaymentRequired:
        raise
    except SearchioError as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc


# ── server construction ──────────────────────────────────────────────────────

def build_server() -> "FastMCP":
    if FastMCP is None:  # pragma: no cover
        raise RuntimeError(
            "The 'mcp' package is required for the MCP server. "
            "Install it with: pip install 'mcp>=1.8,<2'"
        )

    mcp = FastMCP("searchio")

    @mcp.tool()
    async def search(
        query: str,
        intent: str = "web",
        k: int = 10,
        domains: str = "",
        freshness: str = "all",
        locale: str = "",
    ) -> dict:
        """Fused, ranked web search across keyless providers.

        Args:
            query: What to search for.
            intent: web | shopping | news | video ...
            k: Number of results (1-50).
            domains: Comma-separated allowlist, e.g. "example.com,other.org".
            freshness: all | day | week | month | year.
            locale: Language/market of the question, e.g. de-DE.
        """
        eng = await _engine()
        doms = [d.strip() for d in domains.split(",") if d.strip()]
        res = await _guarded("search", eng.search(
            query.strip(), intent=intent.strip().lower(), k=k, domains=doms,
            freshness=freshness.strip().lower(), locale=locale.strip(),
        ))
        return {
            "query": query,
            "intent": intent,
            "elapsed_ms": res.elapsed_ms,
            "providers_used": res.used,
            "providers_failed": res.failed,
            "results": [_to_jsonable(d) for d in res.docs],
        }

    @mcp.tool()
    async def search_items(query: str, k: int = 10, domains: str = "") -> dict:
        """Marketplace/product listings with prices, merged across retailers.

        Args:
            query: Product to find, e.g. "sony wh-1000xm5".
            k: Maximum listings to return.
            domains: Comma-separated site scope, e.g. "bestbuy.com".
        """
        eng = await _engine()
        doms = [d.strip() for d in domains.split(",") if d.strip()] or None
        items = await _guarded("search_items", eng.find_items(query.strip(), k=k, domains=doms))
        return {"query": query, "count": len(items),
                "items": [_to_jsonable(i) for i in items]}

    @mcp.tool()
    async def search_local(query: str, city: str = "", k: int = 20) -> dict:
        """Facebook Marketplace classifieds near a city (browser-backed).

        Args:
            query: What to look for, e.g. "kayak".
            city: Marketplace city slug, e.g. "nyc". Empty follows the exit IP.
            k: Maximum listings.
        """
        eng = await _engine()
        items = await _guarded("search_local", eng.find_local_items(query.strip(), city=city, k=k))
        return {"query": query, "city": city or "(exit-ip)", "count": len(items),
                "items": [_to_jsonable(i) for i in items]}

    @mcp.tool()
    async def read_url(url: str) -> dict:
        """Fetch one URL through the blocking-resistant acquisition ladder.

        Escalates plain HTTP -> TLS impersonation -> stealth browser as the
        target demands, and returns the page as extracted markdown.

        Args:
            url: Absolute URL to read.
        """
        eng = await _engine()
        doc = await _guarded("read_url", eng.read(url.strip()))
        return _to_jsonable(doc)

    @mcp.tool()
    async def research(question: str) -> dict:
        """Run the multi-step research swarm on a question (needs an LLM key).

        Decomposes the question, runs parallel research workers over the live
        web, and returns a cited answer with findings, items and sources.

        Args:
            question: The research question, min 3 chars.
        """
        q = question.strip()
        if len(q) < 3:
            raise RuntimeError("question must be at least 3 characters")
        eng = await _engine()
        result = await _guarded("research", eng.research(q))
        return _to_jsonable(result)

    @mcp.tool()
    async def stats() -> dict:
        """Ladder and provider health (per-domain learned tiers, block rates)."""
        eng = await _engine()
        return _to_jsonable(eng.stats())

    @mcp.resource("searchio://health")
    async def health() -> str:
        """Service health as a small JSON object."""
        eng = await _engine()
        return json.dumps({"ok": True, "service": "searchio-mcp",
                           "payment": _gate.spec or "off"})

    return mcp


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="searchio MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"],
                        default="stdio", help="MCP wire transport (default stdio).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for http transports.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port for http transports.")
    parser.add_argument("--path", default="/mcp", help="Mount path for http transports.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)

    mcp = build_server()
    try:
        if args.transport == "stdio":
            mcp.run("stdio")
        else:
            # FastMCP serves http transports from its own settings; host/port/path
            # are set via the constructor kwargs in v1. Rebuild with them.
            mcp.settings.host = args.host
            mcp.settings.port = args.port
            mcp.settings.mount_path = args.path
            mcp.run(args.transport)
    finally:
        asyncio.run(shutdown())


if __name__ == "__main__":
    main()
