"""Text out of a JS shell's embedded state (the ``embedded_state_rung`` knob).

A client-rendered page that strips to nothing often still carries its whole
content in the document: Next.js serialises the page props into
``<script id="__NEXT_DATA__">``, Nuxt into ``__NUXT_DATA__`` / ``window.__NUXT__``,
Redux and Apollo apps into ``window.__PRELOADED_STATE__`` / ``__APOLLO_STATE__``.
A browser pass exists to run the code that turns that JSON into DOM; reading
the JSON directly gets the same words for the cost of a parse.

This module is pure: HTML in, a small synthetic HTML document out (or ""
when the page has no usable state). The ladder decides when to ask.
"""

from __future__ import annotations

import html as html_mod
import json
import re
from typing import Any, Iterator

#: ``<script id="...">`` blocks known to hold serialised page state.
_ID_SCRIPT = re.compile(
    r"<script\b[^>]*\bid\s*=\s*[\"'](__NEXT_DATA__|__NUXT_DATA__|__FRONTITY_STATE__|"
    r"__APOLLO_STATE__|__REDUX_STATE__|__PRELOADED_STATE__|__INITIAL_STATE__|"
    r"__UNIVERSAL_DATA_FOR_REHYDRATION__|__MODERN_ROUTER_DATA__)[\"'][^>]*>(.*?)</script\s*>",
    re.I | re.S,
)
#: Any other inline JSON script; only large ones are worth a parse.
_JSON_SCRIPT = re.compile(
    r"<script\b[^>]*\btype\s*=\s*[\"']application/json[\"'][^>]*>(.*?)</script\s*>",
    re.I | re.S,
)
#: ``window.__X__ = {...}`` assignments (decoded with raw_decode from the brace).
_ASSIGN = re.compile(
    r"(?:window\.|self\.|globalThis\.)?(__INITIAL_STATE__|__PRELOADED_STATE__|"
    r"__APOLLO_STATE__|__NUXT__|__INITIAL_DATA__|__INITIAL_PROPS__|__REDUX_STATE__|"
    r"__remixContext|__STATE__|__DATA__)\s*=\s*(?=[\[{])"
)
_TITLE = re.compile(r"<title\b[^>]*>(.*?)</title\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]{1,400}>")
_WS = re.compile(r"\s+")
_HEAD_KEYS = frozenset({"title", "headline", "heading", "name", "subtitle"})
#: Keys whose values are never page text.
_SKIP_KEYS = frozenset({
    "buildId", "assetPrefix", "runtimeConfig", "publicRuntimeConfig", "locales",
    "__N_SSG", "__N_SSP", "scriptLoader", "dynamicIds", "gssp", "gsp", "customServer",
    "query", "translations", "messages", "i18n", "intl", "_nextI18Next", "featureFlags",
})
_MAX_SCAN = 8 * 1024 * 1024
_MAX_OUT = 200_000
_MIN_GENERIC_JSON = 2000


def _sources(html: str) -> Iterator[tuple[str, Any]]:
    seen: set[int] = set()
    for m in _ID_SCRIPT.finditer(html):
        seen.add(m.start(2))
        try:
            yield m.group(1), json.loads(html_mod.unescape(m.group(2)) if "&quot;" in m.group(2)
                                         else m.group(2))
        except (ValueError, RecursionError):
            continue
    for m in _JSON_SCRIPT.finditer(html):
        if m.start(1) in seen or len(m.group(1)) < _MIN_GENERIC_JSON:
            continue
        try:
            yield "json", json.loads(m.group(1))
        except (ValueError, RecursionError):
            continue
    dec = json.JSONDecoder()
    for m in _ASSIGN.finditer(html):
        try:
            obj, _end = dec.raw_decode(html, m.end())
        except (ValueError, RecursionError):
            continue  # a JS object literal, not JSON: nothing to read safely
        yield m.group(1), obj


def _prose(s: str) -> str:
    """The string as page text, or "" when it is not prose."""
    if len(s) < 40:
        return ""
    if "<" in s:
        s = _TAG.sub(" ", s)
    s = _WS.sub(" ", html_mod.unescape(s)).strip()
    if len(s) < 40 or s.count(" ") < 6:
        return ""
    head = s[:12].lower()
    if head.startswith(("http://", "https://", "data:", "{", "[", "function", "/*")):
        return ""
    # Code, class lists, tracking payloads: too few letters to be sentences.
    letters = sum(1 for ch in s[:400] if ch.isalpha() or ch == " ")
    if letters < 0.8 * min(len(s), 400):
        return ""
    return s


def _walk(node: Any, key: str, out: list[tuple[str, str]], depth: int = 0) -> None:
    if depth > 60:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            ks = str(k)
            if ks in _SKIP_KEYS:
                continue
            _walk(v, ks, out, depth + 1)
    elif isinstance(node, list):
        for v in node:
            _walk(v, key, out, depth + 1)
    elif isinstance(node, str):
        if key in _HEAD_KEYS and 3 <= len(node) <= 200 and "\n" not in node:
            out.append(("h", _WS.sub(" ", html_mod.unescape(_TAG.sub(" ", node))).strip()))
            return
        p = _prose(node)
        if p:
            out.append(("p", p))


def embedded_state_html(html: str, min_chars: int = 600) -> str:
    """A synthetic article built from the page's embedded state, or "".

    Only prose survives: strings of sentences, plus short strings under
    title-like keys as headings (kept only next to prose, so a state blob of
    nothing but names yields nothing). Exact repeats are dropped -- state
    stores carry the same entity under several keys.
    """
    if not html or len(html) > _MAX_SCAN:
        return ""
    if "__" not in html and "application/json" not in html:
        return ""
    parts: list[tuple[str, str]] = []
    names: list[str] = []
    for name, obj in _sources(html):
        before = len(parts)
        _walk(obj, "", parts)
        if len(parts) > before:
            names.append(name)
    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    total = 0
    for kind, text in parts:
        if not text or text in seen:
            continue
        seen.add(text)
        kept.append((kind, text))
        if kind == "p":
            total += len(text)
        if total >= _MAX_OUT:
            break
    if total < max(1, min_chars):
        return ""
    # A heading with no prose after it before the next heading is a label.
    body: list[str] = []
    pending: str | None = None
    for kind, text in kept:
        if kind == "h":
            pending = text
            continue
        if pending:
            body.append(f"<h2>{html_mod.escape(pending)}</h2>")
            pending = None
        body.append(f"<p>{html_mod.escape(text)}</p>")
    m = _TITLE.search(html[:200_000])
    title = _WS.sub(" ", html_mod.unescape(m.group(1))).strip() if m else ""
    src = html_mod.escape(",".join(dict.fromkeys(names)))
    return (
        "<!doctype html><html><head>"
        f"<title>{html_mod.escape(title)}</title>"
        f'<meta name="searchio-source" content="embedded-state:{src}">'
        "</head><body><article>"
        + (f"<h1>{html_mod.escape(title)}</h1>" if title else "")
        + "".join(body)
        + "</article></body></html>"
    )
