"""Model backends for the swarm.

Anthropic is the default and the reference implementation. This module exists so
the swarm can also run against an OpenAI-compatible endpoint (DeepSeek, or a
self-hosted vLLM) without the orchestrator and worker knowing which one they
are talking to.

The Anthropic message shape is the internal lingua franca -- content blocks,
``tool_use``/``tool_result``, ``stop_reason`` -- and the OpenAI-compatible
backend translates in both directions. That direction of translation is the
right one: the block shape carries strictly more structure than OpenAI's
flat ``content`` string plus sidecar ``tool_calls``, so going the other way
would lose information the swarm relies on.

Both backends expose ``client.messages.create(...)``, so :mod:`worker` and
:mod:`orchestrator` are backend-agnostic and untouched.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import ConfigError, ProviderError


# ── normalized response shapes (Anthropic-flavoured) ─────────────────────────


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    content: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)


# ── OpenAI-compatible backend ────────────────────────────────────────────────


def _anthropic_tools_to_openai(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Flatten Anthropic-shaped messages into the OpenAI chat format.

    The awkward part is tool results. Anthropic puts every result for one
    assistant turn into a single user message; OpenAI wants one ``role: tool``
    message per call. Fanning them out here keeps the worker loop written the
    Anthropic way, which is the shape that actually matches how the model was
    asked to behave.
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for b in content:
                btype = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
                if btype == "text":
                    text_parts.append(getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else ""))
                elif btype == "tool_use":
                    bid = getattr(b, "id", None) or b.get("id")
                    name = getattr(b, "name", None) or b.get("name")
                    inp = getattr(b, "input", None) if not isinstance(b, dict) else b.get("input")
                    tool_calls.append(
                        {
                            "id": bid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(inp or {})},
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        # user turn: either plain text blocks or a batch of tool results
        results = [
            b for b in content
            if (b.get("type") if isinstance(b, dict) else getattr(b, "type", None)) == "tool_result"
        ]
        if results:
            for r in results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": r["tool_use_id"] if isinstance(r, dict) else r.tool_use_id,
                        "content": (("[tool error] " if (r.get("is_error") if isinstance(r, dict)
                                                         else getattr(r, "is_error", False)) else "")
                                    + str(r["content"] if isinstance(r, dict) else r.content)),
                    }
                )
        else:
            texts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif getattr(b, "type", None) == "text":
                    texts.append(getattr(b, "text", ""))
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


_STOP_MAP = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
}


class _OpenAICompatMessages:
    def __init__(self, backend: "OpenAICompatBackend") -> None:
        self._b = backend

    async def create(
        self,
        *,
        model: str,
        max_tokens: int = 4096,
        system: str = "",
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        thinking: dict | None = None,
        output_config: dict | None = None,
        **_: Any,
    ) -> Response:
        # `thinking` and `output_config.effort` are Anthropic-only knobs. Drop
        # them rather than failing: the swarm asks for adaptive thinking on
        # every call, and refusing to run on a backend that has no equivalent
        # would make this adapter useless.
        payload: dict[str, Any] = {
            "model": self._b.model or model,
            "messages": _to_openai_messages(system, messages),
            "max_tokens": max_tokens,
        }
        oa_tools = _anthropic_tools_to_openai(tools)
        if oa_tools:
            payload["tools"] = oa_tools
        if tool_choice:
            # Anthropic's {type: tool, name: X} becomes OpenAI's nested form.
            if tool_choice.get("type") == "tool":
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            elif tool_choice.get("type") in ("any", "required"):
                payload["tool_choice"] = "required"

        fmt = (output_config or {}).get("format")
        if fmt and fmt.get("type") == "json_schema":
            # This endpoint accepts json_object but not json_schema, so the
            # schema has to be carried in the prompt instead. Stated as a hard
            # requirement rather than a hint, because a planner that returns a
            # differently-shaped object fails downstream at json.loads.
            payload["response_format"] = {"type": "json_object"}
            schema_note = (
                "\n\nRespond with a single JSON object and nothing else. It must "
                "validate against this JSON Schema exactly:\n"
                + json.dumps(fmt["schema"], indent=2)
            )
            for m in payload["messages"]:
                if m["role"] == "system":
                    m["content"] = (m["content"] or "") + schema_note
                    break
            else:
                payload["messages"].insert(0, {"role": "system", "content": schema_note.strip()})

        data = await self._b.post(payload)
        choices = data.get("choices") or []
        if not choices:
            # A completion with no choices is the endpoint failing, not the
            # model answering nothing (bug 81): it used to come back as a
            # silent end_turn with empty content -- a "final answer" of "".
            raise ProviderError("llm", f"{self._b.base_url} returned no choices: "
                                       f"{json.dumps(data)[:200]}")
        choice = choices[0]
        msg = choice.get("message") or {}

        content: list = []
        text = msg.get("content")
        if text:
            content.append(TextBlock(text=text))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if raw_args in (None, ""):
                raw_args = "{}"
            try:
                # Some gateways hand back the arguments already parsed (bug
                # 148): json.loads on a dict was a TypeError that killed the
                # turn. An object is taken as-is; anything else must be JSON.
                args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("arguments are not an object", str(raw_args), 0)
            except (json.JSONDecodeError, TypeError) as exc:
                # Not {} (bug 82): the bridge then ran the tool with no
                # arguments and the model learned "query is required"
                # instead of "your JSON was broken". The reserved key is
                # what ToolBridge.dispatch refuses with an is_error result.
                args = {"_error": f"tool arguments were not valid JSON ({getattr(exc, 'msg', exc)}): "
                                  f"{str(raw_args)[:200]}"}
            tid = tc.get("id")
            if not tid:
                # Every id-less call used to be "call_0": the next turn's
                # tool_results then collided (bug 82).
                self._b._call_seq += 1
                tid = f"call_{self._b._call_seq}"
            content.append(ToolUseBlock(id=tid, name=fn.get("name") or "", input=args))

        usage = data.get("usage") or {}
        return Response(
            content=content,
            stop_reason=_STOP_MAP.get(choice.get("finish_reason") or "stop", "end_turn"),
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )


class OpenAICompatBackend:
    """Any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, *, api_key: str, base_url: str, model: str, timeout: float = 180.0) -> None:
        if not api_key:
            raise ConfigError("no API key for the OpenAI-compatible backend")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        self.messages = _OpenAICompatMessages(self)
        self._call_seq = 0
        self._sleep = asyncio.sleep

    #: Retried like the Anthropic SDK does: twice, on the statuses that mean
    #: "try again" and on transport failures. 4xx other than 429 is the
    #: caller's problem and is raised at once.
    RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
    MAX_RETRIES = 2

    async def post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = ""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                r = await self._client.post(url, headers=headers, json=payload)
            except (httpx.HTTPError, OSError) as exc:
                # Transport failure: typed (bug 80 -- it used to escape as
                # the raw httpx exception / a RuntimeError, reaching the
                # orchestrator and /research as a bare 500) and retried.
                last = f"{type(exc).__name__}: {exc}"[:200]
                if attempt < self.MAX_RETRIES:
                    await self._sleep(0.5 * (2 ** attempt))
                    continue
                raise ProviderError("llm", f"{self.base_url} unreachable after "
                                           f"{attempt + 1} attempts: {last}") from exc
            if r.status_code == 200:
                data = r.json()
                if "error" in data and data["error"]:
                    raise ProviderError("llm", f"{self.base_url}: {str(data['error'])[:300]}")
                return data
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in self.RETRY_STATUSES and attempt < self.MAX_RETRIES:
                await self._sleep(0.5 * (2 ** attempt))
                continue
            raise ProviderError("llm", f"{self.base_url} {last}"
                                       + (f" (after {attempt + 1} attempts)" if attempt else ""))
        raise ProviderError("llm", f"{self.base_url} {last}")  # pragma: no cover

    async def close(self) -> None:
        await self._client.aclose()


# ── factory ──────────────────────────────────────────────────────────────────


def make_client(settings):
    """Build the model client the settings ask for.

    Returns ``(client, model_id)``. The model id is returned separately because
    the orchestrator passes ``model=`` on every call and the right value differs
    per backend.
    """
    provider = (getattr(settings, "llm_provider", "") or "anthropic").lower()

    if provider in ("deepseek", "openai_compat", "openai-compatible"):
        key = settings.deepseek_api_key or ""
        if not key:
            import os

            key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ConfigError(
                "llm_provider is deepseek but no key found -- set "
                "SEARCHIO_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY"
            )
        return (
            OpenAICompatBackend(
                api_key=key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
            ),
            settings.deepseek_model,
        )

    from anthropic import AsyncAnthropic

    key = settings.anthropic_key()
    if not key:
        raise ConfigError(
            "research requires an Anthropic API key -- set ANTHROPIC_API_KEY "
            "(or SEARCHIO_ANTHROPIC_API_KEY), or set SEARCHIO_LLM_PROVIDER=deepseek. "
            "`searchio search` needs no key."
        )
    return AsyncAnthropic(api_key=key), settings.model
