"""The searchio MCP server: tool surface, payment gate, arg routing.

Offline by construction: tool *listing* and the payment gate need no network;
the two tool-*execution* tests stub the engine so nothing leaves the machine.
The wire protocol itself (stdio framing) is exercised separately by the live
smoke in the runbook -- these pin the parts that can regress silently.
"""

from __future__ import annotations

import pytest

import searchio.mcp as mcp_mod
from searchio.mcp import PaymentGate, PaymentRequired, build_server


@pytest.fixture
def server():
    return build_server()


class TestToolSurface:
    async def test_six_tools_exposed(self, server):
        tools = {t.name: t for t in await server.list_tools()}
        assert set(tools) == {
            "search", "search_items", "search_local",
            "read_url", "research", "stats",
        }

    async def test_health_resource_present(self, server):
        res = await server.list_resources()
        assert any(str(r.uri) == "searchio://health" for r in res)

    async def test_tools_have_docstrings_for_the_client(self, server):
        # MCP clients render the docstring as the tool description; a tool with
        # none shows the model an empty contract.
        for t in await server.list_tools():
            assert t.description and t.description.strip(), t.name


class TestPaymentGate:
    def test_default_gate_is_open(self):
        assert PaymentGate("").enabled is False
        # Does not raise.
        import asyncio
        asyncio.run(PaymentGate("").verify("research", None))

    def test_static_mode_rejects_without_bearer(self):
        gate = PaymentGate("static:secret-token")
        assert gate.enabled
        with pytest.raises(PaymentRequired) as exc:
            import asyncio
            asyncio.run(gate.verify("research", None))
        assert exc.value.body["status"] == 402
        assert exc.value.body["tool"] == "research"
        assert "x402" in exc.value.x_payment

    def test_static_mode_accepts_matching_bearer(self):
        gate = PaymentGate("static:secret-token")
        import asyncio
        asyncio.run(gate.verify("research", "Bearer secret-token"))  # no raise

    def test_env_var_enables_gate(self, monkeypatch):
        monkeypatch.setenv("SEARCHIO_MCP_PAYMENT", "static:env-token")
        assert PaymentGate().enabled is True


class TestEngineHolder:
    async def test_engine_created_lazily_and_closed(self, monkeypatch, tmp_path):
        from searchio.config import Settings
        calls = {"made": 0, "closed": 0}

        class _FakeEngine:
            def __init__(self, settings):
                calls["made"] += 1

            async def close(self):
                calls["closed"] += 1

        monkeypatch.setattr(mcp_mod, "Engine", _FakeEngine)
        monkeypatch.setattr(mcp_mod, "get_settings",
                            lambda: Settings(state_dir=tmp_path, sidecar_autostart=False))
        await mcp_mod.shutdown()  # clean any prior
        eng = await mcp_mod._engine()
        assert eng is await mcp_mod._engine()  # same instance, one engine
        assert calls["made"] == 1
        await mcp_mod.shutdown()
        assert calls["closed"] == 1
        assert mcp_mod._ENGINE is None


class TestArgRouting:
    def test_default_is_stdio(self):
        args = mcp_mod._parse([])
        assert args.transport == "stdio"

    def test_http_transports_take_host_port_path(self):
        args = mcp_mod._parse(["--transport", "streamable-http",
                               "--host", "0.0.0.0", "--port", "9000", "--path", "/mcp"])
        assert args.transport == "streamable-http"
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.path == "/mcp"
