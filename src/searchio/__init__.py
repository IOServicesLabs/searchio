import os as _os
import sys as _sys

#: lean_imports (SEARCHIO_LEAN_IMPORTS): httpx's __init__ imports its own
#: command-line client (httpx._main -> click, pygments, rich), 36-77 ms of a
#: CLI search's startup for a module searchio never calls. A None entry in
#: sys.modules makes that import fail, and httpx falls back to its stub
#: ``main`` -- so only the `httpx` console script, a separate process, could
#: notice. Read from the environment because it must act before Settings
#: (which imports httpx's dependency chain) exists.
#: On: A/B boot5, cold -107 ms [-205, -4] (n=48).
_LEAN_IMPORTS_DEFAULT = True
if (_os.environ.get("SEARCHIO_LEAN_IMPORTS", str(_LEAN_IMPORTS_DEFAULT)).strip().lower()
        in ("1", "true", "yes", "on") and "httpx" not in _sys.modules):
    _sys.modules.setdefault("httpx._main", None)  # type: ignore[arg-type]

#: dns_prefetch (SEARCHIO_DNS_PREFETCH): resolve the keyless providers' hosts
#: on a daemon thread from the first line of startup, so the lookups overlap
#: the ~300 ms of imports that follow instead of opening the first fan-out
#: (the OS resolver cache answers the real dial). Read from the environment
#: for the same reason as lean_imports: Settings does not exist yet.
_DNS_PREFETCH_HOSTS = (
    "html.duckduckgo.com", "lite.duckduckgo.com", "www.bing.com",
    "en.wikipedia.org", "hn.algolia.com", "api.github.com",
    "api.stackexchange.com", "api.openalex.org", "api.crossref.org",
    "export.arxiv.org",
)
if _os.environ.get("SEARCHIO_DNS_PREFETCH", "").strip().lower() in ("1", "true", "yes", "on"):
    import threading as _threading

    def _prefetch_dns() -> None:
        import socket
        from concurrent.futures import ThreadPoolExecutor

        def one(host: str) -> None:
            try:
                socket.getaddrinfo(host, 443, 0, socket.SOCK_STREAM)
            except OSError:
                pass

        with ThreadPoolExecutor(max_workers=len(_DNS_PREFETCH_HOSTS)) as pool:
            list(pool.map(one, _DNS_PREFETCH_HOSTS))

    _threading.Thread(target=_prefetch_dns, name="searchio-dns-prefetch", daemon=True).start()
