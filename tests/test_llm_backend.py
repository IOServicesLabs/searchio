"""Translation between the Anthropic message shape and OpenAI's.

The swarm speaks Anthropic content blocks internally. Everything here is about
the lossy corners of mapping that onto a flat chat format -- mostly tool
results, which Anthropic batches into one user message and OpenAI wants split
into one `role: tool` message per call.
"""

from __future__ import annotations

import json

import pytest

from searchio.config import Settings
from searchio.errors import ConfigError
from searchio.swarm.llm import (
    TextBlock,
    ToolUseBlock,
    _anthropic_tools_to_openai,
    _to_openai_messages,
    make_client,
)
from searchio.swarm.tools import tool_defs


class TestToolTranslation:
    def test_wraps_schema_in_function_envelope(self):
        oa = _anthropic_tools_to_openai(tool_defs())
        assert {t["function"]["name"] for t in oa} == {
            "web_search", "read_page", "read_page_rendered", "find_items", "submit_report"
        }
        search = next(t for t in oa if t["function"]["name"] == "web_search")
        assert search["type"] == "function"
        assert search["function"]["parameters"]["type"] == "object"

    def test_none_stays_none(self):
        assert _anthropic_tools_to_openai(None) is None


class TestMessageTranslation:
    def test_system_becomes_first_message(self):
        out = _to_openai_messages("be helpful", [{"role": "user", "content": "hi"}])
        assert out[0] == {"role": "system", "content": "be helpful"}
        assert out[1] == {"role": "user", "content": "hi"}

    def test_assistant_tool_use_becomes_tool_calls(self):
        msgs = [
            {"role": "user", "content": "find x"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Looking."),
                    ToolUseBlock(id="tu_1", name="web_search", input={"query": "x"}),
                ],
            },
        ]
        out = _to_openai_messages("", msgs)
        asst = out[-1]
        assert asst["role"] == "assistant"
        assert asst["content"] == "Looking."
        call = asst["tool_calls"][0]
        assert call["id"] == "tu_1"
        assert call["function"]["name"] == "web_search"
        assert json.loads(call["function"]["arguments"]) == {"query": "x"}

    def test_batched_tool_results_fan_out(self):
        # Anthropic returns every result for one turn in a single user message;
        # OpenAI needs one per call, keyed by id.
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": "first"},
                    {"type": "tool_result", "tool_use_id": "b", "content": "second"},
                ],
            }
        ]
        out = _to_openai_messages("", msgs)
        assert len(out) == 2
        assert [m["role"] for m in out] == ["tool", "tool"]
        assert [m["tool_call_id"] for m in out] == ["a", "b"]
        assert out[0]["content"] == "first"

    def test_assistant_with_no_text_has_null_content(self):
        # OpenAI rejects an empty string here; null is the correct encoding of
        # "this turn was only a tool call".
        msgs = [{"role": "assistant", "content": [ToolUseBlock(id="t", name="f", input={})]}]
        out = _to_openai_messages("", msgs)
        assert out[0]["content"] is None
        assert out[0]["tool_calls"][0]["id"] == "t"

    def test_round_trip_of_a_full_worker_turn(self):
        msgs = [
            {"role": "user", "content": "objective"},
            {"role": "assistant", "content": [ToolUseBlock(id="t1", name="web_search",
                                                           input={"query": "q", "intent": "web"})]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "results", "is_error": False}
            ]},
        ]
        out = _to_openai_messages("sys", msgs)
        assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]


class TestFactory:
    def test_deepseek_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = Settings(llm_provider="deepseek", deepseek_api_key="")
        with pytest.raises(ConfigError, match="deepseek"):
            make_client(s)

    def test_deepseek_returns_backend_and_model(self):
        s = Settings(
            llm_provider="deepseek",
            deepseek_api_key="test-key",
            deepseek_model="deepseek-v4-flash",
        )
        client, model = make_client(s)
        assert model == "deepseek-v4-flash"
        assert hasattr(client.messages, "create")

    def test_anthropic_without_a_key_explains_the_alternative(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = Settings(llm_provider="anthropic", anthropic_api_key="")
        with pytest.raises(ConfigError, match="deepseek"):
            make_client(s)


class TestPricing:
    def test_anthropic_is_priced_by_default(self):
        assert Settings(llm_provider="anthropic").prices() == (5.0, 25.0)

    def test_other_backends_are_unpriced_rather_than_guessed(self):
        # Reporting Claude's list price for a DeepSeek run printed a confident,
        # wrong dollar figure. Unknown is the honest answer.
        assert Settings(llm_provider="deepseek").prices() == (0.0, 0.0)

    def test_explicit_prices_win(self):
        s = Settings(llm_provider="deepseek", price_in_per_mtok=0.28, price_out_per_mtok=0.42)
        assert s.prices() == (0.28, 0.42)


import asyncio as _asyncio  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402

from searchio.errors import ProviderError  # noqa: E402
from searchio.swarm.llm import OpenAICompatBackend, _to_openai_messages  # noqa: E402


def _backend(responses):
    """A backend whose HTTP layer is scripted: each entry is either a dict
    (a 200 body) or an int (an HTTP status with a JSON error body)."""
    be = OpenAICompatBackend(api_key="sk-test", base_url="https://x.example", model="m")
    queue = list(responses)
    be.calls = 0

    async def fake_post_http(url, headers=None, json=None):
        be.calls += 1
        entry = queue.pop(0)
        if isinstance(entry, int):
            return _NS(status_code=entry, text='{"error":{"message":"rate limited"}}',
                       json=lambda: {"error": {"message": "rate limited"}})
        if isinstance(entry, Exception):
            raise entry
        return _NS(status_code=200, text="", json=lambda: entry)

    be._client = _NS(post=fake_post_http)
    be._sleep = lambda s: _asyncio.sleep(0)  # no real backoff in tests
    return be


def _create(be, **kw):
    kw.setdefault("model", "m")
    kw.setdefault("messages", [{"role": "user", "content": "hi"}])
    return _asyncio.run(be.messages.create(**kw))


def _choice(finish, message):
    return {"choices": [{"finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


class TestCompatBackendHonesty:
    """Bugs 80-82 (iteration 47, DeepSeek backends review + probes): the
    OpenAI-compatible adapter turned HTTP failures into bare RuntimeErrors
    with no retry (a DeepSeek 429 ended a worker turn, and reached /research
    as a bare 500), answered an EMPTY `choices` list with a silent end_turn,
    swallowed a tool call's malformed JSON arguments into {}, and gave every
    id-less tool call the same id "call_0" (the next turn's tool_results
    then collide). The Anthropic SDK retries 429/5xx twice and raises typed
    errors; the adapter must be at least as honest.
    """

    def test_http_failures_are_typed_and_retried_with_a_cap(self):
        be = _backend([429, 503, _choice("stop", {"role": "assistant", "content": "ok"})])
        r = _create(be)
        assert r.stop_reason == "end_turn" and be.calls == 3
        be = _backend([429, 429, 429, 429])
        with pytest.raises(ProviderError, match="429"):
            _create(be)
        assert be.calls == 3  # two retries, then the truth
        be = _backend([401])
        with pytest.raises(ProviderError, match="401"):
            _create(be)
        assert be.calls == 1  # auth failures are not retried

    def test_empty_choices_is_an_error_not_a_silent_end_turn(self):
        with pytest.raises(ProviderError, match="no choices"):
            _create(_backend([{"choices": [], "usage": {}}]))

    def test_malformed_tool_arguments_become_a_visible_tool_error(self):
        be = _backend([_choice("tool_calls", {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{not json"}}]})])
        r = _create(be)
        blk = r.content[0]
        assert blk.type == "tool_use" and blk.name == "web_search"
        assert "not valid JSON" in blk.input.get("_error", ""), blk.input
        # ...and the bridge refuses it as an error result instead of running
        # web_search with no query.
        from searchio.swarm.tools import ToolBridge
        from searchio.config import Settings

        bridge = ToolBridge(_NS(s=Settings(cache_enabled=False), router=_NS(), ladder=_NS()))
        text, is_err = _asyncio.run(bridge.dispatch("web_search", blk.input))
        assert is_err and "not valid JSON" in text

    def test_idless_tool_calls_get_unique_ids(self):
        be = _backend([_choice("tool_calls", {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            {"type": "function", "function": {"name": "read_page", "arguments": "{}"}}]}),
            _choice("tool_calls", {"role": "assistant", "content": None, "tool_calls": [
            {"type": "function", "function": {"name": "web_search", "arguments": "{}"}}]})])
        ids = [b.id for b in _create(be).content] + [b.id for b in _create(be).content]
        assert len(set(ids)) == 3, ids

    def test_tool_result_errors_are_marked_for_the_model(self):
        # Rider: OpenAI's tool message has no is_error flag; the marker is
        # the only way the model can tell a refusal from a result.
        out = _to_openai_messages("", [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": "Tool budget exhausted", "is_error": True}]},
        ])
        assert out[-1]["role"] == "tool" and out[-1]["content"].startswith("[tool error] ")



class TestCompatBackendArgumentsShapes:
    def test_already_parsed_arguments_object_is_accepted(self):
        # Bug 148 (Muse second pass): some OpenAI-compatible gateways return
        # `arguments` as a parsed object, not a JSON string; json.loads on a
        # dict raised TypeError and the whole turn crashed.
        be = _backend([_choice("tool_calls", {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": {"query": "x"}}}]})])
        r = _create(be)
        blk = r.content[0]
        assert blk.type == "tool_use" and blk.input == {"query": "x"}

    def test_null_arguments_is_an_empty_call(self):
        be = _backend([_choice("tool_calls", {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": None}}]})])
        assert _create(be).content[0].input == {}
