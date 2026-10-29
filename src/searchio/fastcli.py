"""The ``searchio`` console entry: a daemon fast path in front of the CLI.

A cold ``searchio search`` pays for the interpreter, ~0.4 s of imports (typer,
rich, pydantic, the engine), a fresh ladder with no warm connections, and
empty in-memory caches -- before its first byte of network. With
``SEARCHIO_DAEMON_URL`` naming a running ``searchio serve``, ``search --json``
is handed to that server with the standard library alone and the server's
warm engine answers. The JSON printed is the CLI's own shape.

Anything else -- another command, a flag the fast path does not know, no
daemon, a daemon that does not answer -- runs the full CLI, unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

#: search flags the fast path forwards (anything else: the full CLI).
_FLAGS = {"-i": "intent", "--intent": "intent", "-k": "k", "--k": "k",
          "-d": "domains", "--domains": "domains"}


def _parse_search(argv: list[str]) -> dict | None:
    params: dict[str, str] = {}
    query = None
    json_out = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--json":
            json_out = True
        elif a in _FLAGS and i + 1 < len(argv):
            params[_FLAGS[a]] = argv[i + 1]
            i += 1
        elif a.startswith("-"):
            return None  # --no-cache, --max-tier, --verbose, --help ...
        elif query is None:
            query = a
        else:
            return None
        i += 1
    if not json_out or not query or not query.strip():
        return None  # the table output is rich's: leave it to the CLI
    params["q"] = query
    return params


def _forward(base: str, params: dict) -> int | None:
    """Exit code after printing the answer, or None to run the full CLI."""
    url = base.rstrip("/") + "/search?" + urllib.parse.urlencode(params)
    timeout = float(os.environ.get("SEARCHIO_DAEMON_TIMEOUT_S", "60"))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code != 503:
            return None  # a 422 the CLI words better, a server bug: run locally
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail") or {}
        except (ValueError, AttributeError):
            return None
        reasons = {**(detail.get("providers_failed") or {}),
                   **(detail.get("providers_skipped") or {})}
        sys.stderr.write(f"error: {detail.get('error', 'all providers failed')}: "
                         + "; ".join(f"{n}: {e}" for n, e in reasons.items()) + "\n")
        return 1
    except (OSError, ValueError):
        return None  # no daemon listening (or it died mid-answer)
    sys.stdout.write(json.dumps(
        {"query": body.get("query", params["q"]), "used": body.get("providers_used", []),
         "failed": body.get("providers_failed", {}),
         "skipped": body.get("providers_skipped", {}),
         "elapsed_ms": body.get("elapsed_ms"), "results": body.get("results", [])},
        default=str) + "\n")
    sys.stdout.flush()
    return 0


def main() -> None:
    argv = sys.argv[1:]
    base = os.environ.get("SEARCHIO_DAEMON_URL", "").strip()
    if base and argv[:1] == ["search"]:
        params = _parse_search(argv[1:])
        if params is not None:
            code = _forward(base, params)
            if code is not None:
                raise SystemExit(code)
    from .cli import app

    app()


if __name__ == "__main__":
    main()
