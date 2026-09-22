"""The acquisition ladder: climb only as far as the page forces you to.

Three tiers, cheapest first:

  0. **plain** -- httpx over HTTP/2. Milliseconds. Answers most of the web,
     including every well-behaved JSON API.
  1. **impersonate** -- curl_cffi with a real browser TLS/JA3 and HTTP/2
     fingerprint. Still no browser, still fast. This is the tier that matters
     most in practice: modern WAFs inspect the TLS handshake before a single
     header is read, so a stock Python client is refused before its beautifully
     crafted User-Agent is ever looked at. Presenting a genuine Chrome
     handshake clears a large share of "bot" blocks at roughly the cost of a
     plain GET.
  2. **browser** -- a sidecar speaking the JSON-RPC verb table: SwarmIO's
     patchright script by default, or the Rust searchio-engine ``se-serve``
     binary with ``SEARCHIO_SIDECAR_ENGINE=1``. Seconds and (for patchright)
     hundreds of megabytes, but it executes JavaScript, solves managed
     challenges, and carries a persistent profile. Reserved for pages that
     genuinely need it.

Tier 2 can hold **both** backends at once. The primary sidecar (chosen by
``sidecar_engine``) takes every ordinary tier-2 climb; a second *challenge*
sidecar (``sidecar_challenge``, default patchright) is built lazily and only
touches a URL in two cases: an explicit ``rendered=True`` fetch, which routes
tier 2 straight to it, and the one-shot auto-rescue -- when the engine comes
back from tier 2 with a JS shell (or a managed-challenge block), the same tier
is retried once through the fidelity browser before the ladder gives up. The
rescue is the reason the composition stays cheap: patchright is never booted
for a page the engine can already read, and ``rendered`` results carry
``FetchResult.rendered=True`` so callers can see exactly which backend served
them.

The escalation rule is the whole design: climb on *refusal or unreadability*,
never on a network flake. Escalating on a timeout converts one slow request
into one slow request plus a browser launch, and it is the single easiest way
to make a scraper both expensive and conspicuous.

Two things make this cheaper over time. :mod:`searchio.net.profiles` remembers
the tier that actually worked per domain so we stop paying rediscovery costs,
and :mod:`searchio.net.cache` means a repeated URL costs nothing at all.
"""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, deque
import ipaddress
import os
import re
import socket
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..config import Settings, get_settings
from ..errors import (
    Blocked,
    NavErrorPage,
    SearchioError,
    SidecarUnavailable,
    TargetRefused,
    TransientError,
)
from ..models import FetchResult
from . import pdf as pdf_mod
from . import persona as persona_mod
from .blocks import _looks_like_challenge, classify
from .cache import Cache
from .clearance import (
    CLEARANCE_COOKIES,
    RETAINED_COOKIES,
    TRUST_COOKIES,
    ClearanceStore,
    relevant_cookies,
)
from .profiles import DomainStore
from .ratelimit import DomainLimiter
from .robots import RobotsCache
from .sidecar import SidecarClient, default_engine_path

TIER_NAMES = {0: "http", 1: "impersonate", 2: "browser"}

#: Reasons whose tier-2 engine failure is always worth one patchright retry:
#: the page honestly answered, but with nothing a parse-only backend can
#: read -- exactly the class of failure a real browser fixes. ``too_thin``
#: only joins them when the body carries render-could-help evidence (a script
#: or a mount): a genuinely tiny page (nowsecure.nl serves 64 characters) is
#: thin for a browser too, and rescuing it burns a 10-second render to
#: re-discover the same 64 characters.
_RESCUE_REASONS = ("js_required", "empty_mount")

#: classify verdicts that mean "these bytes are a PDF". Either the origin
#: labeled it honestly (not_html) or the body magic outranked a text-ish lie
#: (binary:). Both trigger one bounded extraction attempt per fetch.
_PDF_REASONS = ("not_html:application/pdf", "binary:application/pdf")


def _header_charset(content_type: str) -> str | None:
    """The charset= parameter of a content-type value, lowercased."""
    for part in (content_type or "").split(";")[1:]:
        k, _, v = part.partition("=")
        if k.strip().lower() == "charset":
            return v.strip().strip('"').lower() or None
    return None


#: Statuses that carry a Location the ladder will chase. 300 Multiple
#: Choices is excluded on purpose: it is a page, not a hop.
_REDIRECT_STATUSES = frozenset((301, 302, 303, 307, 308))


def _refused_target(url: str) -> str | None:
    """Policy refusal for a fetch or redirect target, ``None`` if allowed.

    Two classes never get fetched, at any tier, on any hop:

    * non-http(s) schemes -- file: is local-file inclusion, javascript: is
      script-in-page-context, ftp: is a protocol the ladder doesn't speak.
      (The clients used to refuse these natively; the guard makes the
      refusal explicit now that redirects are followed by hand.)
    * link-local and unspecified IP literals -- 169.254.169.254 is the
      cloud metadata endpoint and the canonical SSRF target; 0.0.0.0/::
      are not routable destinations at all. IPv4-mapped IPv6 literals
      (``::ffff:a9fe:a9a9``) are unwrapped before the check, and IPv6
      link-local (fe80::/10) counts too.

    Loopback and RFC1918 literals stay allowed: the control suite runs on
    127.0.0.1 and intranet targets are a legitimate user-driven fetch.
    Hostnames pass the literal check -- their ADDRESSES are vetted at
    connect time by ``_resolve_checked`` (bug 27), which shares the
    ``_refused_ip`` predicate.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return f"scheme:{parts.scheme or 'none'}"
    host = parts.hostname or ""
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(host)
    except ValueError:
        return None  # a hostname, not an IP literal
    return _refused_ip(ip)


def _refused_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """The address classes no fetch may dial, ``None`` if allowed.

    Shared by the URL-literal guard (``_refused_target``) and the
    connect-time resolution guard (``_resolve_checked``): an IPv4-mapped
    IPv6 address is unwrapped first, link-local (169.254/16, fe80::/10) is
    the cloud-metadata SSRF class, unspecified (0.0.0.0, ::) is not a
    routable destination.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_link_local:
        return f"link_local:{ip}"
    if ip.is_unspecified:
        return f"unspecified:{ip}"
    return None


#: Client-side navigation (iteration 25): a <meta http-equiv="refresh"
#: content="0; url=..."> in a 200 HTML head is a redirect the server never
#: sent -- the squeeze/parked/affiliate-interstitial class routes through
#: it, and a stack that ignores the tag serves the interstitial as content
#: or boots a browser for a page one GET away. The contract below is
#: mirrored EXACTLY by the engine (se-net's navigate); span control rows
#: 100-118 pin both stacks over the wire.
_META_REFRESH_MAX_HOPS = 10    # mirrors the HTTP redirect budget
_META_REFRESH_MAX_DELAY_S = 5.0  # above this the interstitial IS the document
_META_HEAD_BYTES = 65536       # the refresh contract lives in the head window
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.I)
_META_ATTR_RE = r"\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))"


def _meta_attr(tag: str, name: str) -> str | None:
    """One attribute's value from a meta tag (any quote style, or bare)."""
    m = re.search(r"(?:^|\s)" + re.escape(name) + _META_ATTR_RE, tag, re.I)
    if not m:
        return None
    return next(g for g in m.groups() if g is not None)


def _meta_refresh_target(
    status: int, content_type: str, text: str, final_url: str
) -> str | None:
    """The navigation a document's first meta-refresh tag asks for.

    Returns the absolute target URL, or ``None`` when the document is
    served as-is. The contract (mirrored by se-net's navigate):

    * only a 200 with an html content-type navigates -- a JSON body whose
      text carries a refresh-looking string is data, not a page;
    * the tag must live in the head window (first 64 KiB);
    * the FIRST refresh tag decides, even a broken one (Chrome's rule --
      scanning for the first *followable* tag hops where a browser would
      not);
    * delay <= 5s is followed IMMEDIATELY -- a search engine never sleeps
      for a redirect (Google treats an instant refresh as a 301-class
      hop); above the bound the interstitial is the document;
    * no url= is a self-reload (the polling pattern): a no-op;
    * the target resolves against the document's final URL.
    """
    if status != 200 or "html" not in content_type.lower():
        return None
    for m in _META_TAG_RE.finditer(text[:_META_HEAD_BYTES]):
        tag = m.group(0)
        equiv = _meta_attr(tag, "http-equiv")
        if equiv is None or equiv.strip().lower() != "refresh":
            continue
        content = _meta_attr(tag, "content")
        if content is None:
            return None
        # WHATWG-lenient: leading delay, an optional ; or , separator, an
        # optional case-insensitive url= prefix, then the URL (any quoting).
        pm = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:[;,]\s*(.*))?$",
                      content, re.S)
        if not pm:
            return None
        delay = float(pm.group(1))
        rest = pm.group(2)
        if rest is None:
            return None  # bare delay = reload this page
        um = re.match(r"url\s*=\s*(.*)$", rest, re.I | re.S)
        raw = (um.group(1) if um else rest).strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            raw = raw[1:-1].strip()
        if not raw or delay > _META_REFRESH_MAX_DELAY_S:
            return None
        return urljoin(final_url, raw)
    return None


def _resolve_checked(host: str, port: int) -> list:
    """getaddrinfo answers for ``host:port``, refusing a poisoned answer set.

    Bug 27: the URL-literal guard never sees a hostname's ADDRESS, so a
    name that resolves to a link-local/unspecified address (the DNS-
    rebinding shape) dialed the refused target at every tier. Resolution
    happens here, in Python, BEFORE any dial: every answer passes
    ``_refused_ip`` and ANY refused answer poisons the whole set (a legit
    name never answers link-local). The checked answers are what callers
    then dial -- tier0's guarded backend dials them directly, tier1 pins
    them via CURLOPT_RESOLVE -- so a flipped second resolution (TOCTOU)
    never reaches the wire. gaierror (NXDOMAIN and friends) propagates as
    the transport failure it is.
    """
    infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        except ValueError:
            continue  # unparseable answer -- the dial errors honestly anyway
        why = _refused_ip(ip)
        if why:
            raise TargetRefused(
                f"target_refused: dns:{host} -> {why}")
    return infos


def _resolve_pin(host: str, port: int) -> dict | None:
    """CURLOPT_RESOLVE option pinning ``host:port`` to checked answers.

    tier1's libcurl resolves in C, out of the guard's reach -- so the hop
    loop resolves in Python first (``_resolve_checked``, which raises
    TargetRefused on a poisoned set) and hands libcurl exactly those
    addresses. Returns None for an IP-literal host: no resolution happens
    and the literal guard already vetted it.
    """
    try:
        ipaddress.ip_address(host)
        return None
    except ValueError:
        pass
    from curl_cffi import CurlOpt  # local like the tier1 import (optional dep)
    entries = []
    for info in _resolve_checked(host, port):
        ip = info[4][0].split("%", 1)[0]
        entries.append(
            f"{host}:{port}:[{ip}]" if ":" in ip else f"{host}:{port}:{ip}")
    return {CurlOpt.RESOLVE: entries} if entries else None

# Headers that authenticate the CALLER to one origin. On a redirect hop to
# a different host (or port -- a different service on the same box), they
# must not ride: the credential belongs to the origin it was built for.
# The set mirrors reqwest's remove_sensitive_headers (vendored
# redirect.rs:239) exactly, so every tier strips the same list.
_SENSITIVE_HOP_HEADERS = frozenset(
    ("cookie", "authorization", "proxy-authorization", "www-authenticate"))


def _origin_key(url: str) -> tuple[str, int]:
    """(host, effective port) -- the redirect-strip boundary.

    RFC 6265 scopes cookies by host alone, but a caller-supplied credential
    header is only known to belong to the origin it was sent to; reqwest
    and httpx both treat a port change as crossing (a different service on
    the same host is a different trust domain), and so does this ladder.
    """
    p = urlsplit(url)
    return (p.hostname or "").lower(), p.port or (443 if p.scheme == "https" else 80)


def _hop_headers(headers: dict[str, str], entry_url: str, current: str) -> dict[str, str]:
    """Caller-supplied headers as one redirect hop may send them.

    Same origin as the entry URL: unchanged. Crossed host/port -- or an
    https->http DOWNGRADE (cleartext must never receive a credential that
    was built for an encrypted origin; httpx's follow has the same nuance,
    keeping Authorization only on the upgrade direction): the sensitive set
    is stripped. This is what the clients' own auto-follow did before
    redirects were followed by hand (bug 25) -- httpx pops Cookie on EVERY
    hop and rebuilds it jar-scoped, reqwest strips the four-name set on
    host|port change; the manual loops (bug 26) forwarded the whole dict
    verbatim, leaking clearance cookies and Authorization across hosts.
    """
    p_entry, p_cur = urlsplit(entry_url), urlsplit(current)
    downgrade = p_entry.scheme == "https" and p_cur.scheme == "http"
    if not downgrade and _origin_key(current) == _origin_key(entry_url):
        return headers
    return {k: v for k, v in headers.items()
            if k.lower() not in _SENSITIVE_HOP_HEADERS}


#: The tenant of the fetch in progress (bug 135). One server process serves
#: every agent that talks to it; the cheap tier's ONE long-lived client
#: deposited every response cookie into ONE jar, so a session cookie banked
#: by tenant A's read rode along on tenant B's read of the same host. A
#: task-local variable (not an instance attribute: concurrent fetches on one
#: ladder must not see each other's tenant) names the jar; nested internal
#: refetches inherit it.
_FETCH_SESSION: contextvars.ContextVar[str] = contextvars.ContextVar("searchio_fetch_session", default="")
_JARS_MAX = 256


class _HopCookies:
    """Minimal RFC 6265 jar for tier 1's manual redirect loop.

    The flat name=value dict it replaces (bug 26) had two holes: it
    ACCEPTED a cookie whose Domain attribute the setting host had no right
    to (127.0.0.1 setting Domain=localhost -- s5.3 says reject outright),
    and it ignored Secure (an https-only cookie would ride an http hop).
    Host scoping itself was already handled by libcurl's cookie engine,
    which attributes each dict entry to the request URL's host and refuses
    cross-host sends -- verified by span ctl.cookie_cross_not_leaked_tier1
    and a direct probe. This jar keeps the same forwarding contract
    (``for_host`` returns a plain dict for the next one-shot get) while
    validating Domain and honoring Secure at collection time. Path and
    expiry stay unscoped by design: a fetch chain is seconds long, and
    path-scoped cookies on one host are a nuance, not a boundary.
    """

    def __init__(self) -> None:
        # (name, scope-domain) -> (value, host_only, secure)
        self._cookies: dict[tuple[str, str], tuple[str, bool, bool]] = {}

    #: Public suffixes a cookie may never be scoped to (RFC 6265 s5.3 rule 5,
    #: the public-suffix rule; bug 91): a.com setting Domain=com used to
    #: ride to evil.com on the next redirect hop. A full PSL is a dependency
    #: this jar does not need -- it lives for one fetch's redirect chain --
    #: so: no dot means a TLD, and the common two-label registries are
    #: listed.
    _PUBLIC_SUFFIXES = frozenset({
        "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "ltd.uk", "plc.uk",
        "com.au", "net.au", "org.au", "edu.au", "gov.au", "co.nz", "org.nz",
        "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp", "co.kr", "or.kr",
        "com.br", "net.br", "org.br", "com.cn", "net.cn", "org.cn", "com.tw",
        "co.in", "net.in", "org.in", "com.mx", "com.tr", "com.ar", "com.sg",
        "co.za", "com.hk", "com.my", "co.id", "com.ph", "com.vn", "com.eg",
        "github.io", "herokuapp.com", "azurewebsites.net", "cloudfront.net",
        "amazonaws.com", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    })

    @staticmethod
    def _domain_matches(host: str, domain: str) -> bool:
        return host == domain or host.endswith("." + domain)

    @classmethod
    def _is_public_suffix(cls, domain: str) -> bool:
        return "." not in domain or domain in cls._PUBLIC_SUFFIXES

    def set_cookie(self, header: str, setting_host: str) -> None:
        """Collect one Set-Cookie value, validating it RFC 6265-style."""
        try:
            segs = header.split(";")
            name, eq, value = segs[0].partition("=")
            name, value = name.strip(), value.strip()
            if not name or not eq:
                # s5.2: no "=" means the NAME is empty (the string is the
                # value); s5.3 requires a non-empty name. Both drop.
                return
            domain_attr, secure = "", False
            for attr in segs[1:]:
                k, _, v = attr.partition("=")
                if k.strip().lower() == "domain":
                    domain_attr = v.strip().lstrip(".").lower()
                elif k.strip().lower() == "secure":
                    secure = True
            host = setting_host.lower()
            if domain_attr:
                # The setter may only name its own host or a parent domain;
                # anything else is rejected outright (s5.3 rule 5) -- and a
                # public suffix is nobody's parent domain (bug 91).
                if not self._domain_matches(host, domain_attr) or self._is_public_suffix(domain_attr):
                    return
                scope, host_only = domain_attr, False
            else:
                scope, host_only = host, True
            self._cookies[(name, scope)] = (value, host_only, secure)
        except Exception:  # noqa: BLE001 -- a cookie is never worth a fetch
            return

    def for_host(self, host: str, scheme: str) -> dict[str, str]:
        """The cookies a request to (host, scheme) may carry."""
        host = host.lower()
        out: dict[str, str] = {}
        for (name, scope), (value, host_only, secure) in self._cookies.items():
            if secure and scheme != "https":
                continue
            if host_only and host != scope:
                continue
            if not host_only and not self._domain_matches(host, scope):
                continue
            out[name] = value
        return out


def _guarded_async_backend():
    """httpcore's anyio backend with the refused-answer guard on the dial.

    Bug 27: a hostname's address exists only after resolution, so the
    URL-literal guard never saw a rebound name. ``connect_tcp`` resolves
    via ``_resolve_checked`` (every answer vetted, any refused answer
    poisons the set -- TargetRefused escapes BEFORE any dial, unmapped, so
    the tier loop's terminal-refusal semantics apply) and then dials the
    CHECKED addresses directly: the dial cannot observe a flipped second
    resolution (no TOCTOU window), and TLS still validates against the
    origin hostname because httpcore takes server_hostname from the
    origin, not from the dialed address. The class is built here so the
    private httpcore import fails at guarded-client construction (loud in
    every test), never at ladder import.
    """
    from httpcore import ConnectError, ConnectTimeout
    from httpcore._backends.anyio import AnyIOBackend, AnyIOStream

    class _GuardedAsyncBackend(AnyIOBackend):
        async def connect_tcp(self, host, port, timeout=None,
                              local_address=None, socket_options=None):
            import anyio
            infos = await anyio.to_thread.run_sync(
                lambda: _resolve_checked(host, port))
            last_exc: OSError | None = None
            for info in infos:
                try:
                    with anyio.fail_after(timeout):
                        stream = await anyio.connect_tcp(
                            remote_host=info[4][0].split("%", 1)[0],
                            remote_port=port,
                            local_host=local_address,
                        )
                    for option in (socket_options or []):
                        stream._raw_socket.setsockopt(*option)
                    return AnyIOStream(stream)
                except (OSError, anyio.BrokenResourceError,
                        TimeoutError) as exc:
                    last_exc = exc
            if last_exc is None:
                raise ConnectError(f"no usable addresses for {host}:{port}")
            if isinstance(last_exc, TimeoutError):
                raise ConnectTimeout(str(last_exc))
            raise ConnectError(str(last_exc))

    return _GuardedAsyncBackend()


class _GuardedTransport(httpx.AsyncHTTPTransport):
    """AsyncHTTPTransport whose pool dials through the guarded backend.

    httpx's constructor does not expose httpcore's ``network_backend``
    param, so the pool's backend is swapped after construction (the pool
    hands its backend to each connection as it is created, so a
    pre-first-request swap is complete). Proxy pools are left as-is: with
    a proxy the dial targets the PROXY and the target name resolves
    proxy-side (the URL-literal guard still applies to the target
    string). The private reach is pinned by TestGuardedBackend -- an
    httpx/httpcore upgrade that reshapes the pool fails there loudly.
    """

    def __init__(self, *args, **kwargs):
        import httpcore
        super().__init__(*args, **kwargs)
        pool = getattr(self, "_pool", None)
        if isinstance(pool, httpcore.AsyncConnectionPool):
            pool._network_backend = _guarded_async_backend()

_RENDER_COULD_HELP = (
    "<script",
    'id="root"', "id='root'", 'id="app"', "id='app'",
    'id="__next"', 'id="__nuxt"', "data-reactroot",
)


def _default_engine_binary() -> Path:
    """Resolution order for the se-serve binary, shared by the primary and
    challenge engine clients: explicit env var, then a sibling checkout."""
    found = default_engine_path()
    if found:
        return found
    return (
        Path.home()
        / "Documents"
        / "searchio-engine"
        / "target"
        / "debug"
        / ("se-serve.exe" if os.name == "nt" else "se-serve")
    )

#: Floor on the raw body before a thin page may be accepted as a result.
#: Small enough to admit a genuinely minimal page, large enough to reject
#: an empty response dressed up as one.
MIN_THIN_BODY = 512


#: Certificate VERIFICATION failures as the three stacks word them (httpx /
#: OpenSSL, curl, reqwest / rustls). A protocol-version or cipher alert is
#: not in the list on purpose: tier 1's different TLS stack can genuinely
#: succeed where tier 0's handshake failed, and that climb is fair.
_CERT_FAILURE_MARKERS = (
    "certificate_verify_failed", "certificate verify failed", "certificate problem",
    "self signed certificate", "self-signed certificate", "invalid peer certificate",
    "unable to get local issuer certificate", "certificate subject name",
    "hostname mismatch", "unknownissuer", "certificate has expired", "certificate is not yet valid",
)


def _is_cert_failure(msg: str) -> bool:
    """Whether a transport error is a certificate that will never verify."""
    low = (msg or "").lower()
    return any(m in low for m in _CERT_FAILURE_MARKERS)


def domain_of(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    # The scheme's default port is no port (bug 151): "example.com:443"
    # keyed a second limiter bucket, profile and clearance entry beside
    # "example.com" -- one host, two rate budgets, and a clearance banked
    # under one spelling never replayed for the other. A trailing dot is
    # the same host too. Userinfo never keys anything.
    try:
        port = parts.port
    except ValueError:
        port = None
    if port and port != (443 if parts.scheme == "https" else 80):
        host = f"{host}:{port}"
    # Strip a "www." label only when a registrable name remains: www.com is
    # a site, and keying it under "com" merged its profile, limiter bucket
    # and clearance with every other TLD-only miss (iteration 51 rider).
    if host.startswith("www.") and host.count(".") >= 2:
        host = host[4:]
    return host


def _nav_error_landing(res: dict) -> str:
    """The nav_error_page refusal, recognized from either sidecar generation.

    A page-initiated navigation (page script or meta refresh) is followed
    NATIVELY by a real browser, and when the followed target is unreachable
    the tab commits Chromium's network-error page. Fixed sidecars refuse with
    the contractual ``nav_error_page`` token in the error text; an older
    patchright script still answers ok with the tab sitting on the error
    page, where the landed URL betrays it (``chrome-error://…``) — the belt.
    Either way the caller gets one honest refusal and the error-page DOM
    never ships as content (bug 32).
    """
    err = str(res.get("error") or "")
    if "nav_error_page" in err:
        return err
    landed = str(res.get("final_url") or res.get("url") or "")
    if landed.startswith("chrome-error://"):
        return f"nav_error_page: the tab committed the browser error page ({landed})"
    return ""


class Ladder:
    """Fetches URLs, escalating through transports until something readable comes back."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sidecar: SidecarClient | None = None,
        challenge_sidecar: SidecarClient | None = None,
        session_id: str = "",
    ) -> None:
        self.s = settings or get_settings()
        self.s.ensure_state_dir()
        self.session_id = session_id or str(int(time.time()))
        #: Per-tenant cheap-tier cookie jars, LRU-bounded (bug 135).
        self._jars: OrderedDict[str, httpx.Cookies] = OrderedDict()
        self._closed = False
        #: Threads for the blocking tiers (bug 140): tier 1 and PDF
        #: extraction ran on the loop's DEFAULT executor -- 8 threads on a
        #: 4-core host for 24 concurrency slots, shared with DNS checks -- so
        #: fetches serialized behind each other. Sized to this ladder's own
        #: concurrency; closed with it.
        self._pool = ThreadPoolExecutor(
            max_workers=max(4, int(self.s.global_concurrency)), thread_name_prefix="searchio-blocking")

        self.cache = Cache(self.s.cache_path(), self.s.cache_ttl_s, self.s.cache_enabled)
        self.profiles = DomainStore(self.s.profile_path())
        self.clearance = ClearanceStore(
            self.s.clearance_path(), self.s.clearance_enabled, self.s.clearance_ttl_s
        )
        self.limiter = DomainLimiter(
            rps=self.s.per_domain_rps,
            burst=self.s.per_domain_burst,
            max_rps=self.s.per_domain_max_rps,
            concurrency=self.s.global_concurrency,
        )
        self.robots = RobotsCache(self._raw_get, policy=self.s.robots_policy)
        if self.s.sidecar_engine:
            # Engine tier 2. A missing binary must surface as a loud,
            # actionable "engine binary not found" at escalation time, never
            # as a silent fallback to patchright -- so when resolution fails
            # we still hand the client the path we WOULD have used, letting
            # ensure() raise its not-found reason naming it.
            # A client the caller handed us is the caller's to close (bug
            # 103): closing it here killed the span suite's shared engine
            # after every control row, and autostart quietly respawned it.
            self._owns_sidecar = sidecar is None
            self.sidecar = sidecar or SidecarClient(
                url=self.s.sidecar_url,
                binary=_default_engine_binary(),
                port=self.s.sidecar_port,
                autostart=self.s.sidecar_autostart,
                boot_timeout_s=self.s.sidecar_boot_timeout_s,
                request_timeout_s=self.s.tier2_timeout_s,
            )
        else:
            # A client the caller handed us is the caller's to close (bug
            # 103): closing it here killed the span suite's shared engine
            # after every control row, and autostart quietly respawned it.
            self._owns_sidecar = sidecar is None
            self.sidecar = sidecar or SidecarClient(
                url=self.s.sidecar_url,
                script=self.s.sidecar_script,
                port=self.s.sidecar_port,
                token=self.s.sidecar_token,
                python=self.s.sidecar_python,
                proxy=self.s.browser_proxy or self.s.proxy,
                autostart=self.s.sidecar_autostart,
                boot_timeout_s=self.s.sidecar_boot_timeout_s,
                request_timeout_s=self.s.tier2_timeout_s,
            )

        # Challenge/fidelity sidecar (the other backend). Built lazily on the
        # first rendered fetch or auto-rescue -- a plain run must never pay
        # for a browser (or a second engine) it never needed.
        self._challenge: SidecarClient | None = challenge_sidecar

        self._http: httpx.AsyncClient | None = None
        self._stats: dict[str, int] = {}
        #: Bounded ring of the last few clearance-capture outcomes, names and
        #: domains only (never values -- cookies are credentials). The 2026-09-18
        #: realtor opener pass banked nothing while fourteen other domains bank
        #: fine; the old silent return left no way to tell a timing miss from a
        #: jar-snapshot gap, so every non-banking capture now leaves a note.
        self._capture_notes: deque[str] = deque(maxlen=8)

    # ── public API ───────────────────────────────────────────────────────────

    def _cache_variant(self, rendered: bool) -> str:
        """The cache key's variant: rendered/plain, and the TENANT for a
        tenant-scoped fetch (bug 138: keyed by URL alone, a body fetched under
        tenant A's cookies -- A's private page -- was served from cache to
        tenant B and to the default tenant). The default tenant's cache stays
        shared, as before."""
        base = "rendered" if rendered else ""
        sid = _FETCH_SESSION.get()
        if sid and sid != self.session_id:
            return f"{base}|s:{sid}" if base else f"s:{sid}"
        return base

    def _jar(self) -> httpx.Cookies:
        """The cookie jar of the tenant whose fetch is in progress."""
        sid = _FETCH_SESSION.get() or self.session_id
        jar = self._jars.get(sid)
        if jar is None:
            jar = self._jars[sid] = httpx.Cookies()
        self._jars.move_to_end(sid)
        while len(self._jars) > _JARS_MAX:
            self._jars.popitem(last=False)
        return jar

    async def fetch(
        self,
        url: str,
        *,
        max_tier: int | None = None,
        force_tier: int | None = None,
        use_cache: bool = True,
        referer: str = "",
        rendered: bool = False,
        session: str = "",
    ) -> FetchResult:
        """Retrieve ``url`` for one tenant; see :meth:`_fetch`.

        ``session`` names the tenant whose cheap-tier cookie jar this fetch
        reads and deposits into (bug 135). Empty inherits the enclosing
        fetch's tenant (internal refetches) or the ladder's own session.
        """
        if self._closed:
            # A closed ladder used to recreate its client lazily and keep
            # serving -- resources nobody closes again (bug 137, the shape
            # of bug 103 from the other side).
            raise RuntimeError("Ladder is closed")
        tok = _FETCH_SESSION.set(session or _FETCH_SESSION.get() or self.session_id)
        try:
            return await self._fetch(url, max_tier=max_tier, force_tier=force_tier,
                                     use_cache=use_cache, referer=referer, rendered=rendered)
        finally:
            _FETCH_SESSION.reset(tok)

    async def _fetch(
        self,
        url: str,
        *,
        max_tier: int | None = None,
        force_tier: int | None = None,
        use_cache: bool = True,
        referer: str = "",
        rendered: bool = False,
    ) -> FetchResult:
        """Retrieve ``url``, climbing tiers until the content is readable.

        Raises :class:`Blocked` if every permitted tier was refused, or
        :class:`TransientError` if the network never cooperated.

        ``rendered=True`` routes tier 2 through the challenge/fidelity
        sidecar (a real browser when the primary is the engine) instead of
        the primary sidecar -- the explicit, expensive "just render it" knob.
        With the default configuration (engine primary, patchright
        challenge) an unreadable engine tier-2 answer is also retried once
        through the challenge sidecar automatically; disable with
        ``sidecar_challenge_auto=False``.

        Domains in ``Settings.rendered_first_domains`` (the adaptive class:
        hosts that 429 the initial document and serve content only after
        browser-grade behavior) open directly on the challenge sidecar
        instead of the cheap tiers, with the normal climb as the fallback
        when the browser pass fails; see :meth:`_rendered_first`.
        """
        started = time.monotonic()
        # Policy gate on the request itself, before any tier, cache, or
        # robots work: link-local/unspecified IP literals and non-http(s)
        # schemes are never fetched (bug 25 -- a 302 to the cloud metadata
        # endpoint rode every stack's auto-follow before this guard).
        why = _refused_target(url)
        if why:
            raise TargetRefused(f"target_refused: {why} ({url[:120]})")
        dom = domain_of(url)
        ceiling = self.s.max_tier if max_tier is None else min(max_tier, self.s.max_tier)

        if use_cache:
            # Rendered requests cache under their own variant: a cheap-tier
            # body (e.g. an Amazon shell read_page returned) must never answer
            # read_page_rendered -- the caller paid for browser fidelity and
            # getting the shell back from cache both lies about the stamp and
            # invites the model to retry the same doomed call.
            hit = self.cache.get(url, variant=self._cache_variant(rendered))
            if hit:
                return FetchResult(**{**hit, "from_cache": True, "via": "cache"})

        rinfo = await self.robots.check(url)
        if not self.robots.permits(rinfo):
            raise Blocked("disallowed_by_robots", vendor="robots")

        async def robots_landing_gate(final: str, esc: list[str]) -> None:
            """robots.txt applies to the URL actually served (bug 95): a
            redirect to a disallowed path used to be fetched and served
            under enforce, and went unannotated under warn."""
            if not final or urlsplit(final)[:3] == urlsplit(url)[:3]:
                return
            info = await self.robots.check(final)
            if info.allowed:
                return
            esc.append("robots:disallowed_redirect_target")
            if not self.robots.permits(info):
                raise Blocked("disallowed_by_robots", vendor="robots")

        start = force_tier if force_tier is not None else self.profiles.start_tier(dom, ceiling)
        if force_tier is None and _FETCH_SESSION.get() not in ("", self.session_id) and start > 0:
            # A tenant-scoped fetch starts at the ISOLATED tiers (bug 136):
            # the domain profile learns to begin at the browser after a
            # wall, and the browser's cookie store is one per process --
            # so every tenant's later read of that host, plain pages
            # included, rode through shared cookies. Tier 0/1 keep per-
            # tenant jars; the browser is reached only when actually refused.
            start = 0
        #: Adaptive-host opener (probe ec52690, config comment on
        #: ``rendered_first_domains``): hosts in that class 429 the initial
        #: document and serve content only after browser-grade behavior, so
        #: the cheap tiers can never pass them -- and iteration 19's honest
        #: 429 means no unaided fetch ever reaches the browser. Open these
        #: directly on the challenge sidecar; when that pass fails, hand off
        #: to the normal climb (advance()) rather than exhausting above it.
        #: Skipped for explicit force_tier/rendered requests, a ceiling below
        #: 2, a profile that already opens at tier 2, and tenant-scoped
        #: fetches (the browser jar is per-process; bug 136).
        opener = (
            force_tier is None
            and not rendered
            and 2 <= ceiling
            and start < 2
            and _FETCH_SESSION.get() in ("", self.session_id)
            and self._rendered_first(dom)
        )

        def advance() -> None:
            """Move to the next tier -- or, after a failed opener, hand off
            to the normal climb at the profile's starting tier."""
            nonlocal tier, rendered_pass, fallback_pending
            if fallback_pending:
                fallback_pending = False
                tier, rendered_pass = start, bool(rendered)
            else:
                tier += 1

        escalations: list[str] = []
        last_block: Blocked | None = None
        #: The tier that drew the most recent block: a thin page from a
        #: HIGHER tier got past that wall and outranks it (iteration 31).
        last_block_tier = -1
        last_transient: Exception | None = None
        #: The block verdict a rescue pass was launched over (bug 74): if the
        #: rescue's own infrastructure then fails, the SITE's refusal is
        #: still the answer, not "try again".
        rescued_block: Blocked | None = None
        #: The anti-bot vendor any tier identified (bug 166): the cheap
        #: tiers see the origin's headers, a browser-backed tier often does
        #: not, and the Blocked the caller gets is the LAST tier's -- so a
        #: wall two tiers named came out as "blocked by unknown". The first
        #: identification stands for the whole climb.
        seen_vendor: str | None = None
        #: Best 2xx body that failed only a content heuristic, kept as a
        #: fallback so a thin page beats no page.
        best: tuple | None = None
        #: PDF extraction fires at most once per fetch -- every tier downloads
        #: the same bytes, so a failed extract is terminal, not climb-worthy.
        pdf_tried = False
        #: Whether the current tier-2 pass (if tier 2 is where we are) goes
        #: through the challenge sidecar instead of the primary one.
        rendered_pass = bool(rendered) or opener
        fallback_pending = opener

        tier = 2 if opener else start
        while tier <= ceiling:
            # The per-domain permit and the crawl delay come BEFORE the global
            # slot (bug 139): held inside it, callers queued on one throttled
            # host occupied every slot while they slept, and an unrelated
            # host waited seconds for nothing. The slot bounds in-flight
            # network work only.
            await self.limiter.acquire(dom)
            if rinfo.crawl_delay:
                # A host that states a crawl-delay has told us its price.
                await asyncio.sleep(min(rinfo.crawl_delay, 10.0))
            attempt_started = time.time()
            async with self.limiter.slot():
                try:
                    res = await self._try_tier(tier, url, referer=referer, rendered=rendered_pass)
                except TargetRefused:
                    # A policy refusal is PERMANENT: no tier can fetch a
                    # refused target, so retrying and climbing are both
                    # pure waste (and the climb drowned the refusal text
                    # under the last tier's unrelated error -- bug 25's
                    # suite shape). Out of the loop, at once.
                    raise
                except NavErrorPage:
                    # The browser followed a page-initiated navigation into
                    # its network-error page. Like a policy refusal this
                    # fetch is decided: the shell fetch proved the network,
                    # and re-fetching the shell re-follows the same script
                    # into the same dead end -- no same-tier retry, and tier
                    # 2 is the ceiling so there is nothing to climb to
                    # (bug 32).
                    raise
                except TransientError as exc:
                    if _is_cert_failure(str(exc)):
                        # A certificate that does not verify is a DECIDED
                        # fetch (bug 157): no retry can change the chain, no
                        # tier can make it valid, and the climb used to
                        # replace this reason with the engine's "request
                        # failed" -- an agent could not tell a bad cert from
                        # a dead host. Out of the loop with the reason kept.
                        escalations.append(f"tier{tier}:tls_invalid_cert")
                        raise TransientError(f"tls_invalid_cert: {exc}") from exc
                    # Network trouble is not refusal. Retry this same tier once,
                    # then give up on it. Climbing to the next CHEAP tier is
                    # fair -- tier 1's different TLS stack genuinely dodges
                    # selective connection resets -- but a flake must never be
                    # what boots a real browser: when tier 2 IS a browser
                    # (patchright primary), stop here. Engine tier 2 is cheap,
                    # so climbing to it stays allowed. The comment above used
                    # to say "do NOT escalate" while the code climbed into the
                    # browser anyway -- bench/span.py ctl.timeout_not_escalated
                    # caught the contradiction. Escape hatch for a host that
                    # only answers the browser: force_tier/rendered=True.
                    last_transient = exc
                    try:
                        res = await self._try_tier(tier, url, referer=referer, rendered=rendered_pass)
                    except TransientError as exc2:
                        last_transient = exc2
                        escalations.append(f"tier{tier}:transient")
                        if tier + 1 == 2 and self._primary_kind() == "patchright":
                            break
                        advance()
                        continue
                    except SidecarUnavailable as exc2:
                        # The retry found the sidecar gone (a flake, then
                        # sticky-unavailable): the same bookkeeping the
                        # first-attempt branch below does -- not an escape
                        # past the loop with no stamp and no exhaustion
                        # message (iteration 31).
                        escalations.append(f"tier{tier}:sidecar_unavailable")
                        last_transient = exc2
                        advance()
                        continue
                except SidecarUnavailable as exc:
                    escalations.append(f"tier{tier}:sidecar_unavailable")
                    last_transient = exc
                    advance()
                    continue

            status, headers, body, ctype, final_url = res
            verdict = classify(
                status, headers, body, content_type=ctype, min_text=self.s.min_text_chars
            )
            seen_vendor = seen_vendor or verdict.vendor

            # PDF interception: classify just told us the bytes are a PDF
            # (envelope content-type or body magic). The tier contract hands
            # us a decoded str, which PDF bytes do not survive, so the text
            # comes from a dedicated bounded byte re-fetch (see _pdf_text).
            # Once per fetch: every tier downloads the same file, so a failed
            # extract is terminal for this fetch, not a reason to climb.
            if (
                not verdict.ok
                and verdict.reason in _PDF_REASONS
                and not pdf_tried
                and self.s.pdf_extraction
            ):
                pdf_tried = True
                if tier == 2:
                    # The browser that reached the PDF may have cleared a
                    # wall to get there; the extraction refetch rides the
                    # tier-0 client with the BANKED clearance, so bank it
                    # now -- the ok path only banks after classify, and a
                    # gated PDF's refetch drew the wall again
                    # (pdf_text:refetch_failed; iteration 31).
                    await self._capture_clearance(dom, self._client_for(rendered_pass))
                text, stamp = await self._pdf_text(final_url or url)
                if text is not None:
                    self.limiter.record_success(dom)
                    self.profiles.record_success(dom, tier)
                    self._bump(f"tier{tier}_ok")
                    self._bump("pdf_text_ok")
                    out = FetchResult(
                        url=url,
                        final_url=final_url or url,
                        status=status,
                        body=text,
                        content_type=ctype,
                        tier=tier,
                        via=TIER_NAMES.get(tier, str(tier)),
                        elapsed_ms=int((time.monotonic() - started) * 1000),
                        escalations=escalations + [f"tier{tier}:{verdict.reason}", stamp],
                        rendered=False,
                    )
                    if use_cache:
                        self.cache.put(url, out.model_dump(exclude={"from_cache"}),
                                       variant=self._cache_variant(rendered), tier=tier)
                    return out
                # Extraction failed (scanned, rotten, oversized, refetch drew
                # a wall): record the honest reason and climb. PDF reasons
                # are not in _RESCUE_REASONS, so no browser pass fires.
                escalations.append(f"tier{tier}:{verdict.reason}/{stamp}")
                advance()
                continue

            if verdict.ok:
                self.limiter.record_success(dom)
                self.profiles.record_success(dom, tier)
                self._bump(f"tier{tier}_ok")
                # rendered_pass is a *request* knob; the cheap tiers can
                # still answer it. Only a tier-2 pass actually went through
                # the challenge sidecar, so only those results may carry
                # the fidelity stamp (and the stat) -- and only when the
                # challenge backend is a real browser: an engine challenge
                # answers rendered=True with parse-only output (the ladder
                # never sends render:true -- iteration 26), so stamping it
                # rendered would claim a browser pass that never happened
                # (bug 36). A rendered=True request served from tier 0 must
                # not claim a browser pass either -- the mislabel teaches
                # the caller that rendering was spent and invites wasted
                # escalation.
                served_rendered = (
                    rendered_pass and tier == 2 and self._challenge_is_browser()
                )
                if served_rendered:
                    self._bump("tier2_rendered")
                if tier == 2:
                    # The browser just paid for a challenge. Bank the cookie
                    # so the next page on this host costs a tier-1 request.
                    await self._capture_clearance(dom, self._client_for(rendered_pass))
                await robots_landing_gate(final_url, escalations)
                out = FetchResult(
                    url=url,
                    final_url=final_url or url,
                    status=status,
                    body=body,
                    content_type=ctype,
                    tier=tier,
                    via=TIER_NAMES.get(tier, str(tier)),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    escalations=escalations,
                    rendered=served_rendered,
                )
                if use_cache:
                    self.cache.put(url, out.model_dump(exclude={"from_cache"}),
                                   variant=self._cache_variant(rendered), tier=tier)
                return out

            escalations.append(f"tier{tier}:{verdict.reason}")

            # One-shot fidelity rescue: the engine answered tier 2 but with a
            # shell (or drew a managed-challenge block) -- classes of failure
            # a real browser fixes. Retry the SAME tier once through the
            # challenge sidecar before any blocked/thin bookkeeping runs.
            # rendered_pass flips, so this can fire at most once per fetch,
            # and only engine->patchright (see _auto_render_active).
            rescue_worthy = verdict.reason.startswith(_RESCUE_REASONS)
            if verdict.blocked:
                # A block buys the browser only on evidence a browser can act
                # on (bug 167): a vendor any tier identified, a challenge
                # signature, or a page with something to render. A bare 403
                # with a 48-byte body and no name at any tier is the origin
                # refusing this client; the browser cannot change its mind,
                # and the rescue cost 80 s per such page live.
                low_headers = {str(k).lower(): (v or "") for k, v in headers.items()}
                blo = body.lower()
                rescue_worthy = bool(
                    verdict.vendor or seen_vendor
                    or _looks_like_challenge(body, low_headers)
                    or any(m in blo for m in _RENDER_COULD_HELP)
                )
            if not rescue_worthy and verdict.reason.startswith("too_thin"):
                blo = body.lower()
                rescue_worthy = any(m in blo for m in _RENDER_COULD_HELP)
            if (
                tier == 2
                and not rendered_pass
                and self._auto_render_active()
                and rescue_worthy
            ):
                rendered_pass = True
                self._bump("tier2_auto_render")
                escalations.append("tier2:auto_rendered")
                if verdict.blocked:
                    rescued_block = Blocked(verdict.reason, verdict.vendor or seen_vendor, status)
                continue

            if verdict.blocked:
                # A replayed cookie that draws a 403 is worse than none:
                # it wastes the tier and looks like a stolen session.
                if tier < 2:
                    # Only the clearance this attempt could have replayed
                    # (bug 143): one banked meanwhile by another caller's
                    # rescue is not the one that drew the 403.
                    self.clearance.drop(dom, older_than=attempt_started)
                self.limiter.record_block(dom)
                self.profiles.record_block(dom, tier, verdict.vendor or "")
                self._bump(f"tier{tier}_blocked")
                last_block = Blocked(verdict.reason, verdict.vendor or seen_vendor, status)
                last_block_tier = tier
            else:
                self._bump(f"tier{tier}_unusable")
                if verdict.reason.startswith("http_429"):
                    # ratelimit.py's contract is "crossing the line costs one
                    # 429": the AIMD controller backs off on a refusal it
                    # exists to absorb. classify() demoted 429 out of
                    # `blocked` to keep the rescue honest (no challenge boot
                    # at a throttle -- iteration 19), but that demotion also
                    # routed every 429 around record_block: since then a host
                    # that started refusing kept being probed at an un-halved
                    # rate, which is both a politeness failure and the exact
                    # traffic pattern that gets an IP noticed. Found live:
                    # realtor.com 429'd all three tiers and the limiter never
                    # learned (2026-09-16). Back off the limiter; leave
                    # profiles alone (a 429 names no vendor and teaches no
                    # tier lesson) and leave the no-rescue demotion as is.
                    self.limiter.record_block(dom)
                # Keep the best thin-but-real answer. Some pages genuinely have
                # almost no text -- nowsecure.nl serves 64 characters -- and the
                # min_text floor cannot tell those from an empty SPA shell.
                # Rather than let a heuristic discard a page we successfully
                # fetched, remember it and hand it back if nothing better turns
                # up and no tier at or above it refused us.
                if (
                    200 <= status < 300
                    # Only a THIN page is thin (bug 92): the exclusion list
                    # let a login wall (auth_required_*), a bot-marker page
                    # or a mojibake body through as "thin-but-real".
                    and verdict.reason.startswith("too_thin")
                    # There has to be *something* on the page. A body with zero
                    # visible text is not a thin page, it is no page -- and
                    # accepting one is exactly the silent false-ok this module
                    # exists to prevent. Observed: a sidecar reply came back in
                    # 31 ms with an empty body and was scored as a retrieval.
                    and verdict.text_len > 0
                    and len(body) >= MIN_THIN_BODY
                    and (best is None or len(body) > len(best[2]))
                ):
                    best = (status, headers, body, ctype, final_url, tier, verdict.reason,
                            rendered_pass and tier == 2)

            advance()

        # A block only outranks the thin page when it came from the thin
        # page's own tier or above. A HIGHER tier's 2xx got PAST the wall the
        # cheap tiers drew -- and refusing to serve it because "somebody was
        # blocked" reported Blocked with the browser's success in hand: the
        # nowsecure.nl shape the comment above cites, defeated by its own
        # condition (iteration 31). A block at or above the thin tier is the
        # later word and still refuses.
        if (
            best is not None
            and (last_block is None or best[5] > last_block_tier)
            # A rescue that never answered leaves the site's refusal standing
            # (bug 96, bug 74's sibling): a thin page from a LOWER tier is
            # not the later word over a tier-2 block whose rescue died.
            and not (rescued_block is not None and last_transient is not None)
        ):
            status, headers, body, ctype, final_url, tier, reason, best_challenge = best
            await robots_landing_gate(final_url, escalations)
            self.profiles.record_success(dom, tier)
            self._bump(f"tier{tier}_thin_ok")
            # best_challenge records that the FALLBACK body came through the
            # challenge sidecar -- which is what picks the clearance client.
            # The fidelity stamp is narrower still: only a real browser may
            # claim it (bug 36 -- an engine challenge serves parse-only).
            best_rendered = best_challenge and self._challenge_is_browser()
            if tier == 2:
                # A browser that got through still earned its cookie, even if
                # the page it returned was too thin for the content heuristic.
                await self._capture_clearance(dom, self._client_for(best_challenge))
            return FetchResult(
                url=url,
                final_url=final_url or url,
                status=status,
                body=body,
                content_type=ctype,
                tier=tier,
                via=TIER_NAMES.get(tier, str(tier)),
                elapsed_ms=int((time.monotonic() - started) * 1000),
                escalations=escalations + [f"accepted_thin:{reason}"],
                rendered=best_rendered,
            )

        if last_block:
            last_block.args = (f"{last_block.args[0]} after {escalations}",)
            raise last_block
        if rescued_block is not None and last_transient is not None:
            # The rescue never got an answer (its sidecar died, its network
            # flaked): the 403 it was launched over stands (bug 74). A
            # rescue that DID answer supersedes it above -- content wins,
            # a second block is last_block, thin-but-real is served.
            rescued_block.args = (
                f"{rescued_block.args[0]} after {escalations} "
                f"(rescue failed: {last_transient})",)
            raise rescued_block
        if last_transient:
            raise TransientError(f"{url}: {last_transient} (tried {escalations})")
        raise TransientError(f"{url}: no usable content (tried {escalations})")

    async def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False)
        if self._http:
            await self._http.aclose()
            self._http = None
        # The challenge sidecar is only ever ours when it is a distinct
        # client; when it de-duplicates to the primary, one close suffices.
        if self._challenge is not None and self._challenge is not self.sidecar:
            await self._challenge.close()
            self._challenge = None
        if getattr(self, "_owns_sidecar", True):
            await self.sidecar.close()
        self.cache.close()
        self.profiles.close()
        self.clearance.close()

    def stats(self) -> dict[str, Any]:
        return {
            "tiers": dict(self._stats),
            "cache": self.cache.stats(),
            "sidecar_available": self.sidecar.available,
            "challenge_configured": self.s.sidecar_challenge,
            "challenge_built": self._challenge is not None,
            "capture_notes": list(self._capture_notes),
        }

    # ── tiers ────────────────────────────────────────────────────────────────

    async def _try_tier(
        self, tier: int, url: str, *, referer: str = "", rendered: bool = False
    ) -> tuple[int, dict[str, str], str, str, str]:
        if tier == 0:
            return await self._tier0(url, referer=referer)
        if tier == 1:
            return await self._tier1(url, referer=referer)
        return await self._tier2(url, rendered=rendered)

    def _client_for(self, rendered: bool) -> SidecarClient:
        """Which sidecar serves a tier-2 pass: the primary, or the lazily
        built challenge sidecar for an explicit/rendered or rescued pass."""
        if not rendered:
            return self.sidecar
        ch = self._challenge
        if ch is None:
            ch = self._build_challenge()
            self._challenge = ch
        return ch

    def fidelity_sidecar(self) -> SidecarClient:
        """The sidecar for the patchright script's composite verbs
        (``search_engine_results`` and friends).

        A patchright primary serves them directly. With an engine primary
        they belong to the challenge sidecar -- that fidelity pairing is
        exactly why the second backend exists. With engine on both knobs
        there is no patchright to serve them: the primary comes back,
        ``backend_kind()`` says ``engine``, and the caller fails honestly
        before sending a verb the wire would reject with "Unknown method".
        """
        if self._primary_kind() == "patchright":
            return self.sidecar
        return self._client_for(True)

    def _primary_kind(self) -> str:
        """Which backend the primary sidecar is: the configured kind, with
        the binary attribute as the honest tell when a client was injected
        (tests, benches) with settings that don't describe it."""
        if getattr(self.sidecar, "binary", None) is not None:
            return "engine"
        return "engine" if self.s.sidecar_engine else "patchright"

    def _challenge_is_browser(self) -> bool:
        """Whether a rendered tier-2 pass goes through a REAL browser.

        The rendered fidelity stamp may be True only then: an engine
        challenge sidecar answers rendered=True with parse-only output (the
        ladder never sends render:true -- iteration 26), so stamping it
        rendered would claim a browser pass that never happened (bug 36).
        The kind logic mirrors _build_challenge's de-duplication exactly:
        same-kind-and-no-URL means the challenge IS the primary.
        """
        kind = (self.s.sidecar_challenge or "patchright").strip().lower()
        if kind == self._primary_kind() and not self.s.sidecar_challenge_url:
            return self._primary_kind() == "patchright"
        return kind == "patchright"

    def _auto_render_active(self) -> bool:
        """Whether an unreadable engine tier-2 answer earns one patchright
        retry. Directional on purpose: only engine -> patchright. A
        patchright primary is already the fidelity backend, and downgrading
        a real browser's failure to the engine would be escalation in name
        only."""
        return (
            self.s.sidecar_challenge_auto
            and self._primary_kind() == "engine"
            and (self.s.sidecar_challenge or "patchright").strip().lower() == "patchright"
        )

    def _rendered_first(self, dom: str) -> bool:
        """Whether this domain opens on the challenge sidecar (Settings'
        ``rendered_first_domains``; the adaptive class, see the config
        comment). Subdomain match: an entry "realtor.com" covers
        www.realtor.com; "notrealtor.com" never matches it."""
        for entry in self.s.rendered_first_domains:
            e = entry.strip().lstrip(".").lower()
            if e and (dom == e or dom.endswith("." + e)):
                return True
        return False

    def _build_challenge(self) -> SidecarClient:
        """Construct (never spawn) the challenge sidecar from settings.

        De-duplicates to the primary client when both knobs name the same
        backend and no separate URL was configured -- two clients would
        otherwise boot two copies of the same browser for zero capability.
        """
        kind = (self.s.sidecar_challenge or "patchright").strip().lower()
        url = self.s.sidecar_challenge_url
        if kind == self._primary_kind() and not url:
            return self.sidecar
        if kind == "engine":
            return SidecarClient(
                url=url,
                binary=_default_engine_binary(),
                port=self.s.sidecar_challenge_port,
                autostart=self.s.sidecar_autostart,
                boot_timeout_s=self.s.sidecar_boot_timeout_s,
                request_timeout_s=self.s.tier2_timeout_s,
            )
        return SidecarClient(
            url=url,
            script=self.s.sidecar_script,
            port=self.s.sidecar_challenge_port,
            token=self.s.sidecar_token,
            python=self.s.sidecar_python,
            proxy=self.s.browser_proxy or self.s.proxy,
            autostart=self.s.sidecar_autostart,
            boot_timeout_s=self.s.sidecar_boot_timeout_s,
            request_timeout_s=self.s.tier2_timeout_s,
        )

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            # The guarded transport owns the dial (bug 27): every hop of
            # every tier-0 request resolves through _resolve_checked and
            # dials only vetted answers. http2/proxy move to the transport
            # -- httpx ignores (or conflicts on) them when a transport is
            # supplied; timeout/redirect policy stay client-level.
            self._http = httpx.AsyncClient(
                transport=_GuardedTransport(
                    http2=True, proxy=self.s.proxy or None),
                follow_redirects=True,
                max_redirects=self.s.max_redirects,
                timeout=self.s.tier0_timeout_s,
            )
        return self._http

    async def _pdf_text(self, url: str) -> tuple[str | None, str]:
        """Extract text from the PDF at ``url``, or explain why not.

        Returns ``(text, stamp)`` on success -- stamp records the page usage
        (``pdf_text:used/total[:capped]``) for the escalations trail. On any
        failure returns ``(None, reason)`` where reason is the honest stamp
        the ladder appends before climbing: the fetch already happened, the
        caller deserves to know exactly which step refused.
        """
        try:
            data = await self._fetch_bytes(url)
        except pdf_mod.PdfTooLarge:
            return None, "pdf_text:too_large"
        except TargetRefused:
            # A policy refusal is permanent (bug 94): stamping it
            # refetch_failed sent the ladder up two more tiers to redraw
            # the same refusal and end in "no usable content".
            raise
        except Exception:
            return None, "pdf_text:refetch_failed"
        if not pdf_mod.looks_like_pdf(data):
            # The re-fetch drew a login wall or an HTML error page instead of
            # the file -- feeding it to pypdf would only confuse the error.
            return None, "pdf_text:not_pdf_bytes"
        try:
            text, used, total = await asyncio.get_running_loop().run_in_executor(
                self._pool, pdf_mod.extract_text, data, self.s.pdf_max_pages)
        except pdf_mod.PdfUnreadable:
            return None, "pdf_text:unreadable"
        if not text.strip():
            # Scanned/images-only PDF: parseable but textless, and no tier
            # in the ladder can OCR. The honest answer is a refusal.
            return None, "pdf_text:no_text"
        stamp = f"pdf_text:{used}/{total}" + (":capped" if total > used else "")
        return text, stamp

    async def _fetch_bytes(self, url: str) -> bytes:
        """Bounded raw-byte GET for the PDF re-fetch.

        The tier contract hands decoded ``str`` (PDF bytes do not survive
        it) and the engine's body bytes never cross the se-serve JSON
        envelope, so extraction re-downloads through the tier-0 client with
        the same persona/clearance headers. The byte cap is enforced twice:
        on the Content-Length header when present (refuse before reading)
        and on the streamed total (a lying header does not get us OOM'd).
        """
        dom = domain_of(url)
        p = persona_mod.for_domain(dom, session=self.session_id)
        headers = persona_mod.safari_headers_fix(
            p, p.headers(contact=self.s.user_agent_contact)
        )
        headers = self._clearance_headers(dom, headers, url)
        cap = self.s.pdf_max_bytes
        r = await self._open_stream(url, headers, self.s.tier0_timeout_s)
        try:
            if r.status_code < 200 or r.status_code >= 300:
                raise TransientError(f"pdf refetch http_{r.status_code}")
            ce = r.headers.get("content-encoding", "").lower()
            cl = r.headers.get("content-length", "")
            # Only an unencoded body's wire length equals its decoded length.
            if ce in ("", "identity") and cl.isdigit() and int(cl) > cap:
                raise pdf_mod.PdfTooLarge(f"content-length {cl} > {cap}")
            chunks: list[bytes] = []
            total = 0
            async for chunk in r.aiter_bytes(65536):
                total += len(chunk)
                if total > cap:
                    raise pdf_mod.PdfTooLarge(f"streamed {total} > {cap}")
                chunks.append(chunk)
        finally:
            await r.aclose()
        return b"".join(chunks)

    async def _read_capped(self, r: httpx.Response, cap: int) -> bytes:
        """Read at most ``cap`` DECODED bytes of a streaming response.

        The cap binds the decoded stream, never the wire: a gzip bomb is
        kilobytes on the wire and gigabytes in memory -- precisely the shape
        that must never materialize. Over the cap is an honest too_large
        refusal, not a silent truncation served as a page.
        """
        ce = r.headers.get("content-encoding", "").lower()
        cl = r.headers.get("content-length", "")
        if ce in ("", "identity") and cl.isdigit() and int(cl) > cap:
            raise TransientError(
                f"too_large: content-length {cl} exceeds cap {cap}")
        chunks: list[bytes] = []
        total = 0
        async for chunk in r.aiter_bytes(65536):
            total += len(chunk)
            if total > cap:
                raise TransientError(
                    f"too_large: decoded body exceeds cap {cap}")
            chunks.append(chunk)
        return b"".join(chunks)

    async def _open_stream(self, url: str, headers: dict, timeout: float) -> httpx.Response:
        """GET ``url`` following redirects hop by hop, policy-checking each.

        Auto-follow was replaced by this manual loop because no client
        exposes a per-hop hook: a 302 can point anywhere, and the guard
        (``_refused_target``) only works if it sees every Location before
        the client commits a connection to it. Returns the final streaming
        response; the caller consumes and closes it. Hop budget matches the
        old client-level follow (initial + ``max_redirects`` follows).
        """
        c = await self._client()
        current = url
        for _hop in range(self.s.max_redirects + 1):
            why = _refused_target(current)
            if why:
                raise TargetRefused(f"target_refused: {why} ({current[:120]})")
            # Caller headers (clearance Cookie, Authorization) belong to the
            # entry origin; a host/port change strips them (bug 26). Cookies
            # DEPOSITED mid-chain ride through the client jar, which scopes
            # by host on its own.
            jar = self._jar()
            try:
                r = await c.send(
                    c.build_request("GET", current,
                                    headers=_hop_headers(headers, url, current),
                                    timeout=timeout, cookies=jar),
                    stream=True,
                    follow_redirects=False,
                )
            except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
                # httpx (0.28) builds the redirect request eagerly even with
                # follow_redirects=False, and chokes on an opaque-scheme
                # Location -- javascript:, data: -- BEFORE returning the 3xx,
                # so the _refused_target guard on the next hop never sees it.
                # A redirect the client cannot even construct is a non-http(s)
                # target: refuse it with the contract's exception, not a raw
                # httpx InvalidURL (bug 169 -- the iteration-76 false-green
                # suite review found the cheap-tier javascript: redirect
                # passing only because the oracle accepted ANY exception;
                # file:/ftp: reached the guard because httpx CAN build them).
                raise TargetRefused(
                    "target_refused: scheme:non_http_redirect "
                    f"(unbuildable redirect from {current[:120]})"
                ) from exc
            # Deposit into the TENANT's jar, never the shared client's (bug
            # 135); the client jar is kept empty so nothing rides across
            # tenants.
            jar.extract_cookies(r)
            c.cookies.clear()
            if r.status_code in _REDIRECT_STATUSES and r.headers.get("location"):
                target = urljoin(current, r.headers["location"])
                await r.aclose()
                current = target
                continue
            return r
        raise TransientError(
            f"redirect_loop: exceeded {self.s.max_redirects} hops ({url[:100]})")

    async def _tier0(self, url: str, *, referer: str = "") -> tuple[int, dict, str, str, str]:
        """Meta-refresh loop over one-document fetches (iteration 25).

        A followed refresh is a client-side redirect: the target passes the
        same _refused_target guard an HTTP Location would (with the hop's
        provenance in the token), the referer chains off the previous hop's
        final URL exactly like a 302, and exceeding the HTTP redirect budget
        is the same honest redirect_loop. The persona/clearance rebuild per
        hop happens inside _tier0_document -- a cross-domain hop arrives with
        the NEW domain's headers, never the old entry's credential.
        """
        current = url
        ref = referer
        for _meta in range(_META_REFRESH_MAX_HOPS + 1):
            out = await self._tier0_document(current, referer=ref)
            target = _meta_refresh_target(out[0], out[3], out[2], out[4])
            if not target:
                return out
            why = _refused_target(target)
            if why:
                raise TargetRefused(
                    f"target_refused: meta refresh: {why} ({target[:120]})")
            ref = out[4]
            current = target
        raise TransientError(
            f"redirect_loop: meta-refresh chain exceeded "
            f"{_META_REFRESH_MAX_HOPS} hops ({url[:100]})")

    async def _tier0_document(self, url: str, *, referer: str = "") -> tuple[int, dict, str, str, str]:
        """Plain HTTP/2. Honest headers, no impersonation."""
        dom = domain_of(url)
        p = persona_mod.for_domain(dom, session=self.session_id)
        headers = persona_mod.safari_headers_fix(
            p, p.headers(referer=referer, contact=self.s.user_agent_contact)
        )
        headers = self._clearance_headers(dom, headers, url)
        try:
            r = await self._open_stream(url, headers, self.s.tier0_timeout_s)
            try:
                raw = await self._read_capped(r, self.s.max_body_bytes)
                # Match r.text's behavior on the cheap tiers: header charset
                # wins, utf-8-lossy otherwise (a meta-only charset mojibakes
                # here -- classify's guard refuses it and the climb reaches
                # the engine tier whose meta prescan decodes; bug-23 shape).
                text = raw.decode(r.charset_encoding or "utf-8", errors="replace")
                # Bank any trust/clearance names this response set, so the
                # next fetch on this host opens warm (trust-cookie jar). The
                # request UA -- persona or clearance-pinned -- is what the
                # origin bound the cookies to.
                self._harvest_trust(
                    str(r.url),
                    self._jar_from_set_cookie(
                        r.headers.get_list("set-cookie"),
                        urlsplit(str(r.url)).hostname or ""),
                    headers.get("User-Agent", ""),
                )
                return (
                    r.status_code,
                    dict(r.headers),
                    text,
                    r.headers.get("content-type", ""),
                    str(r.url),
                )
            finally:
                await r.aclose()
        except TransientError:
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as exc:
            raise TransientError(f"tier0 {type(exc).__name__}: {exc}") from exc

    async def _tier1(self, url: str, *, referer: str = "") -> tuple[int, dict, str, str, str]:
        """curl_cffi with a genuine browser TLS + HTTP/2 fingerprint.

        The import is local so curl_cffi stays an optional dependency: without
        it the ladder simply skips this tier rather than failing to import.
        """
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError as exc:
            raise TransientError(f"tier1 unavailable: {exc}") from exc

        def _blocking() -> tuple[int, dict, str, str, str]:
            cap = self.s.max_body_bytes
            # Redirects followed by hand (allow_redirects=False) so every
            # Location passes _refused_target before a connection is
            # committed to it -- module-level curl_cffi has no per-hop hook.
            # Two bug-26 rules ride along: caller credential headers are
            # stripped on any host/port change (_hop_headers, mirroring
            # reqwest's strip set), and collected Set-Cookies go through
            # _HopCookies so a Domain the setter had no right to is rejected
            # and Secure stays on https. (Host scoping of the FORWARDED
            # cookies is libcurl's own cookie engine -- it attributes each
            # dict entry to the request host and will not send cross-host;
            # probed directly, pinned by ctl.cookie_cross_not_leaked_tier1.)
            # The jar lives ABOVE the meta-refresh loop (iteration 25): a
            # refresh hop rides the squeeze's cookies exactly like a 302 hop.
            jar = _HopCookies()

            def fetch_doc(doc_url: str, doc_ref: str) -> tuple[int, dict, str, str, str]:
                """One document: the manual 302 loop, then the capped read.

                Persona, impersonation target and clearance are rebuilt for
                THIS document's domain -- a cross-domain meta hop must arrive
                with the new domain's headers; the old entry's credential
                never enters this dict at all.
                """
                dom = domain_of(doc_url)
                p = persona_mod.for_domain(dom, session=self.session_id)
                hdrs = persona_mod.safari_headers_fix(p, p.headers(referer=doc_ref))
                # The persona UA the impersonation target will speak -- kept
                # for the trust-cookie harvest below: the response's cookies
                # are bound to the UA on the wire, which is either this one
                # or the clearance pin's if _clearance_headers replaced it.
                persona_ua = hdrs.get("User-Agent", "")
                # curl_cffi sets its own UA/hints to match the impersonation
                # target; ours would only risk contradicting it.
                hdrs.pop("User-Agent", None)
                hdrs.pop("sec-ch-ua", None)
                # Clearance goes on last, and its UA pin wins: the cookie is
                # bound to the browser's UA, so that pairing has to survive
                # intact even though everything else here defers to the
                # impersonation target.
                hdrs = self._clearance_headers(dom, hdrs, doc_url)
                current = doc_url
                r = None
                for _hop in range(self.s.max_redirects + 1):
                    why = _refused_target(current)
                    if why:
                        raise TargetRefused(
                            f"target_refused: {why} ({current[:120]})")
                    cur_parts = urlsplit(current)
                    hop_cookies = jar.for_host(cur_parts.hostname or "",
                                               cur_parts.scheme)
                    # Bug 27: curl_cffi resolves in C (never sees a Python
                    # getaddrinfo patch, and its answer would be unvetted), so
                    # the hop's addresses are checked+pinned up front --
                    # _resolve_pin raises TargetRefused on a refused answer and
                    # otherwise hands libcurl a CurlOpt.RESOLVE entry, making
                    # the dial reuse the SAME vetted answers (no TOCTOU).
                    hop_hdrs = _hop_headers(hdrs, doc_url, current)
                    r = cffi_requests.get(
                        current,
                        headers=hop_hdrs,
                        # The fingerprint follows the UA on the wire (bug 141):
                        # a clearance replay pins the browser's UA, and the
                        # persona's own UA maps back to its own target.
                        impersonate=persona_mod.impersonate_for_ua(
                            hop_hdrs.get("User-Agent", ""), p.impersonate),
                        timeout=self.s.tier1_timeout_s,
                        allow_redirects=False,
                        cookies=hop_cookies or None,
                        curl_options=_resolve_pin(
                            cur_parts.hostname or "",
                            cur_parts.port
                            or (443 if cur_parts.scheme == "https" else 80)),
                        proxies={"http": self.s.proxy, "https": self.s.proxy} if self.s.proxy else None,
                        stream=True,
                    )
                    if r.status_code in _REDIRECT_STATUSES and r.headers.get("location"):
                        try:
                            for sc_hdr in r.headers.get_list("set-cookie"):
                                jar.set_cookie(sc_hdr, cur_parts.hostname or "")
                        except Exception:  # noqa: BLE001 -- a cookie is never worth a fetch
                            pass
                        target = urljoin(current, r.headers["location"])
                        r.close()
                        current = target
                        continue
                    break
                else:
                    raise TransientError(
                        f"redirect_loop: exceeded {self.s.max_redirects} hops ({doc_url[:100]})")
                try:
                    # A FINAL response can set cookies too -- bank them so a
                    # meta-refresh hop below rides them like a 302 hop's.
                    try:
                        for sc_hdr in r.headers.get_list("set-cookie"):
                            jar.set_cookie(sc_hdr, cur_parts.hostname or "")
                    except Exception:  # noqa: BLE001 -- same rule
                        pass
                    # The chain-validated jar now holds every hop's names;
                    # contribute the retained ones (trust + clearance) to the
                    # per-domain store so the NEXT fetch on this host opens
                    # warm. UA: the clearance pin's if one rode, else the
                    # impersonation target's persona UA captured above.
                    self._harvest_trust(
                        str(r.url), jar,
                        hdrs.get("User-Agent") or persona_ua)
                    ct = r.headers.get("content-type", "")
                    # Same cap contract as tier 0, on the decoded stream: the
                    # wire-length pre-check is only valid unencoded (compressed
                    # bodies DECODE larger, so the streamed total owns those).
                    ce = r.headers.get("content-encoding", "").lower()
                    cl = r.headers.get("content-length", "")
                    if ce in ("", "identity") and cl.isdigit() and int(cl) > cap:
                        raise TransientError(
                            f"too_large: content-length {cl} exceeds cap {cap}")
                    chunks: list[bytes] = []
                    total = 0
                    for chunk in r.iter_content():
                        total += len(chunk)
                        if total > cap:
                            raise TransientError(
                                f"too_large: decoded body exceeds cap {cap}")
                        chunks.append(chunk)
                    text = b"".join(chunks).decode(
                        _header_charset(ct) or "utf-8", errors="replace")
                    return (
                        r.status_code,
                        dict(r.headers),
                        text,
                        ct,
                        str(r.url),
                    )
                finally:
                    r.close()

            # The meta-refresh loop mirrors _tier0's: same guard, same
            # referer chaining, same budget as the HTTP redirects.
            current = url
            ref = referer
            for _meta in range(_META_REFRESH_MAX_HOPS + 1):
                out = fetch_doc(current, ref)
                target = _meta_refresh_target(out[0], out[3], out[2], out[4])
                if not target:
                    return out
                why = _refused_target(target)
                if why:
                    raise TargetRefused(
                        f"target_refused: meta refresh: {why} ({target[:120]})")
                ref = out[4]
                current = target
            raise TransientError(
                f"redirect_loop: meta-refresh chain exceeded "
                f"{_META_REFRESH_MAX_HOPS} hops ({url[:100]})")

        try:
            return await asyncio.get_running_loop().run_in_executor(self._pool, _blocking)
        except TransientError:
            raise
        except Exception as exc:
            name = type(exc).__name__
            # Transport weather keeps the retryable class BY ORIGIN: curl's
            # own RequestsError is an OSError on curl_cffi's MRO, and so are
            # the socket/DNS layer's failures. Anything else escaping
            # _blocking -- a ValueError/TypeError/KeyError from the hop
            # machinery -- is a permanent LOGIC failure: calling it
            # transient retries a deterministic re-fail and can climb the
            # fetch into a tier-2 spend that "answers" it, hiding the dead
            # tier forever (the expensive mistake the taxonomy exists to
            # prevent). It escapes as a plain SearchioError instead:
            # in-family for callers, outside the tier loop's retry/climb
            # net. (The keyword-matching if/else here used to raise the
            # identical TransientError from BOTH branches -- bug 35.)
            if isinstance(exc, OSError):
                raise TransientError(f"tier1 {name}: {exc}") from exc
            raise SearchioError(f"tier1 {name}: {exc}") from exc

    async def _tier2(self, url: str, *, rendered: bool = False) -> tuple[int, dict, str, str, str]:
        """A sidecar's browser, with an origin warm-up on refusal.

        ``rendered=True`` sends the pass through the challenge sidecar (the
        fidelity backend) instead of the primary one.

        The sidecar's own ``fetch`` verb is HTTP-first internally and escalates
        to a real page only when the content needs it, so this is the heaviest
        thing we have but not always the slowest.

        The warm-up is the part that matters. A first browser request to a deep
        URL on a hostile host arrives with no cookies, no session, and no
        referer -- which is not what any real visitor looks like, because real
        visitors land on the homepage and click through. Measured on
        upwork.com/freelance-jobs and crunchbase.com/organization: both refused
        the browser on first contact and both succeeded once the profile had
        seen the origin. So on a refusal, navigate the homepage, then ask again.
        """
        return await self._tier2_via(self._client_for(rendered), url)

    async def _tier2_via(self, client: SidecarClient, url: str) -> tuple[int, dict, str, str, str]:
        # One budget, caller-owned: a third of the tier budget per navigation
        # attempt. At the 90 s default this is 30000 ms -- patchright's own
        # built-in default, so production behavior is unchanged -- while a
        # tight test/ctl budget makes stalls deterministic instead of
        # parking the call for two 30 s attempts the agent never agreed to.
        # (The verb may spend two attempts -- the timeout-soften degrade --
        # plus DOM reads inside the call window; a third leaves room.)
        # A tab of its own (bug 159): every tier-2 fetch used to navigate
        # "default", and on the browser rescue -- a goto plus a read on that
        # tab -- two callers at once could hand each other's page back. The
        # tab is closed afterwards so neither backend accumulates pages.
        tab_id = f"fetch-{uuid.uuid4().hex[:10]}"
        try:
            return await self._tier2_on_tab(client, url, tab_id)
        finally:
            try:
                await client.call("close_tab", {"tab_id": tab_id})
            except Exception:  # noqa: BLE001 -- cleanup never fails a fetch
                pass

    async def _tier2_on_tab(
        self, client: SidecarClient, url: str, tab_id: str
    ) -> tuple[int, dict, str, str, str]:
        res = await client.fetch(
            url, tab_id=tab_id, timeout=self.s.tier2_timeout_s,
            timeout_ms=int(self.s.tier2_timeout_s * 1000 / 3),
        )

        nav_err = _nav_error_landing(res)
        if nav_err:
            # The tab committed the browser's network-error page after a
            # page-initiated navigation: permanent for this fetch and never
            # curable by an origin warmup, so raise before the warmup dance
            # would spend a second browser navigation on it (bug 32). The
            # exact type matters: the tier loop treats NavErrorPage as
            # terminal, so its generic same-tier retry doesn't spend a
            # second navigation reaching the identical dead end either.
            raise NavErrorPage(f"tier2: {nav_err}")

        if self._sidecar_refused(res):
            origin = self._origin_of(url)
            if origin and origin != url:
                try:
                    # On the call's OWN tab (rider of bug 159, found by the
                    # iteration-74 review): a fixed "warmup" tab was shared
                    # by every concurrent tier-2 rescue -- and priming a
                    # different tab than the refetch then used did not even
                    # help it. The warm-up now primes the exact tab the
                    # refetch navigates.
                    await client.goto(origin, tab_id=tab_id)
                    self._bump("tier2_warmup")
                except Exception:
                    pass  # a failed warm-up is not worse than no warm-up
                # On the same private tab as the first attempt (rider of
                # bug 159): the refetch used to navigate "default" again.
                res = await client.fetch(
                    url, tab_id=tab_id, timeout=self.s.tier2_timeout_s,
                    timeout_ms=int(self.s.tier2_timeout_s * 1000 / 3),
                )
                nav_err = _nav_error_landing(res)
                if nav_err:
                    # The refetch can land on the error page just like the
                    # first fetch (bug 93): the check above only guarded the
                    # first, so the dino page shipped through this side door.
                    raise NavErrorPage(f"tier2: {nav_err}")

        # The origin's headers, when the backend reports them: the vendor
        # tell for classify (a 503 carrying cf-mitigated is a challenge, a
        # bare 503 is an outage) and the content-type for the ok path.
        env_headers = res.get("headers") if isinstance(res.get("headers"), dict) else {}
        env_low = {str(k).lower(): str(v) for k, v in env_headers.items()}
        if self._sidecar_refused(res):
            reason = str(res.get("error") or res.get("bot_reason") or "browser_failed")
            if res.get("bot_wall") or res.get("blocked"):
                # Report as a real response so classify() sees the block.
                return 403, {"x-searchio-sidecar": reason}, reason, "text/html", url
            status = int(res.get("status") or 0)
            if status in (401, 403, 429) or status >= 500:
                # The verb ran; the site said no. Surface the real status so
                # classify() and the blocked bookkeeping see a gate rather
                # than a transient flake -- otherwise a 403 page forced
                # through tier 2 reports as a retryable browser_failed and
                # the caller can never tell "blocked" from "flaky".
                # (Caught live by bench/span.py ctl.blocked_classification.)
                # 5xx joined the list in iteration 31: a 500 was reported as
                # "browser_failed" -- the origin's outage mislabeled as OUR
                # browser breaking -- and retried once for it, and a 503 WAF
                # challenge never reached classify's http_503_challenge
                # verdict, the one that drives the blocked bookkeeping and
                # the challenge rescue. The envelope headers ride along so
                # classify can name the vendor.
                body = str(res.get("html") or res.get("text") or reason)
                return status, {**env_low, "content-type": "text/html"}, body, "text/html", url
            if "target_refused" in reason:
                # The engine's policy guard, riding the error text out of
                # se-serve: same permanent class as the Python-side refusal
                # -- out immediately, no same-tier retry.
                raise TargetRefused(f"tier2: {reason}")
            raise TransientError(f"tier2: {reason}")

        body = res.get("html") or res.get("text") or res.get("content") or ""
        # The decoded-size cap every cheap tier enforces on the stream
        # (iteration 18) and the engine enforces at its source -- but a
        # patchright body arrived here uncapped (iteration 31), so the
        # contract holds at the ladder for whichever backend served it.
        # Refusal, not truncation: a capped prefix served as a page is the
        # silent-garbage defect in a size costume. (The char count is a
        # cheap lower bound on the byte count -- encode only when it is
        # under the cap.)
        cap = self.s.max_body_bytes
        if len(body) > cap or len(body.encode("utf-8", "surrogatepass")) > cap:
            raise TransientError(f"too_large: tier2 body exceeds cap {cap}")
        # The origin's content-type, when the backend reports it. Synthesizing
        # "text/html" unconditionally was bug 20: a PDF (or any binary) fetched
        # through tier 2 arrived at classify() labeled as a page, and a PDF's
        # ASCII operator stream clears the body heuristics -- control bytes
        # shipped to the agent as an ok page (bench/span.py bulk seed 113).
        # When the envelope stays silent the classify body-magic sniff is the
        # backstop, so the fallback here only names the common case.
        ctype = str(
            res.get("content_type") or res.get("contentType")
            or env_headers.get("content-type") or env_headers.get("Content-Type")
            or ""
        ) or "text/html"
        return (
            int(res.get("status") or 200),
            {"content-type": ctype, "x-searchio-via": str(res.get("via") or "browser")},
            body,
            ctype,
            # A sidecar that followed a page-initiated navigation reports the
            # LANDING url under final_url (patchright iteration 27; the
            # engine's fetch arm already overwrites url itself).
            str(res.get("final_url") or res.get("url") or url),
        )

    @staticmethod
    def _sidecar_refused(res: dict) -> bool:
        """Whether a sidecar reply is a refusal rather than a page.

        A 4xx carried in an ``ok: true`` envelope still counts: the verb ran
        fine, the *site* said no, and only the status distinguishes them.
        """
        if not res.get("ok") or res.get("bot_wall") or res.get("blocked"):
            return True
        status = int(res.get("status") or 200)
        return status in (401, 403, 429) or status >= 500

    @staticmethod
    def _origin_of(url: str) -> str:
        p = urlsplit(url)
        if not p.scheme or not p.netloc:
            return ""
        return f"{p.scheme}://{p.netloc}/"

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _capture_clearance(self, domain: str, client: SidecarClient) -> str:
        """Bank the retained cookies the browser just earned for this host.

        ``client`` is whichever sidecar served the pass: a cf_clearance
        earned by patchright lives in patchright's jar, not the engine's.
        The jar carries the full RETAINED_COOKIES set -- challenge tokens
        and trust-continuity names alike -- so a browser pass warms the
        same per-domain store the cheap-tier harvest feeds.

        Returns the outcome (``banked`` / ``empty`` / ``error``) and always
        leaves evidence: the 2026-09-18 realtor opener pass banked nothing
        while fourteen other domains bank fine through this same call, and
        the old silent return made the miss untraceable. A non-banking
        capture now bumps a counter and records a note (jar names + domains
        only, never values -- cookies are credentials). The empty branch
        settles 6s and re-snapshots once: Akamai's sensor POST often lands
        just after load-event, so a single load-time snapshot false-empties
        on slow-sensor domains. The note then discriminates what remains:
        ``first_party=0`` = wrong-jar snapshot; retained names in
        ``retained_anywhere`` but not banked = domain-shape bug (check
        ``retained_sites`` name@domain pairs for the scoping); neither =
        the sensor never set them at all.
        """
        try:
            jar = await client.cookies()
            cookies = relevant_cookies(jar, domain)
            settled = False
            if not cookies:
                # Settle retry: cheap (one jar read after 6s), runs only on
                # the empty path, and turns a timing race into a banked row
                # instead of a mystery.
                self._bump("clearance_capture_settle_retry")
                await asyncio.sleep(6.0)
                jar = await client.cookies()
                cookies = relevant_cookies(jar, domain)
                settled = True
            if cookies:
                ua = await client.user_agent()
                self.clearance.put(domain, cookies, ua)
                self._bump("clearance_captured")
                if settled:
                    self._bump("clearance_capture_banked_after_settle")
                    self._capture_notes.append(
                        f"{domain}: banked after 6s settle (timing race)"
                    )
                return "banked"
            want = domain.removeprefix("www.")
            dom_match = [
                c for c in jar or []
                if (cdom := str(c.get("domain") or "").lstrip(".").removeprefix("www."))
                and (want == cdom or want.endswith("." + cdom))
            ]
            retained_hit = sorted(
                {str(c.get("name") or "") for c in jar or []} & RETAINED_COOKIES
            )
            retained_sites = sorted({
                f"{str(c.get('name') or '')}@{str(c.get('domain') or '')}"
                for c in jar or []
                if str(c.get("name") or "") in RETAINED_COOKIES
            })[:8]
            self._bump("clearance_capture_empty")
            self._capture_notes.append(
                f"{domain}: empty capture; jar={len(jar or [])} "
                f"first_party={len(dom_match)} retained_anywhere={retained_hit} "
                f"retained_sites={retained_sites} "
                f"domains={sorted({str(c.get('domain') or '') for c in jar or []})[:4]}"
            )
            return "empty"
        except Exception as exc:
            # Never let bookkeeping fail a fetch that already succeeded.
            self._bump("clearance_capture_error")
            self._capture_notes.append(
                f"{domain}: capture error {type(exc).__name__}: {exc}"
            )
            return "error"

    def _clearance_headers(self, domain: str, headers: dict[str, str],
                           url: str = "") -> dict[str, str]:
        """Attach banked clearance, pinning the UA it was issued against.

        The UA pin is not optional. Cloudflare binds ``cf_clearance`` to the
        User-Agent that solved the challenge, so sending the cookie under this
        session's rotating persona is a *contradiction* -- the exact failure
        mode :mod:`searchio.net.persona` exists to prevent -- and reads worse
        than sending no cookie at all.
        """
        if url and not url.lower().startswith("https://"):
            # Clearance cookies are Secure: replaying the expensive token
            # over plaintext both leaks it and is a browser-impossible
            # signal (bug 72). The https fetch after the usual redirect
            # gets it.
            return headers
        c = self.clearance.get(domain)
        if not c:
            return headers
        headers = dict(headers)
        existing = headers.get("Cookie")
        headers["Cookie"] = f"{existing}; {c.header()}" if existing else c.header()
        if c.user_agent:
            # The hints must follow the pinned UA (bug 73): Chrome 143 with
            # the persona's v="131" hints -- or no hints under the Safari
            # persona -- is the contradiction the pin exists to avoid.
            headers = persona_mod.pin_user_agent(headers, c.user_agent)
        self.clearance.record_use(domain)
        self._bump("clearance_used")
        return headers

    def _harvest_trust(self, url: str, jar: _HopCookies, ua: str) -> None:
        """Bank trust/clearance cookies a cheap-tier response deposited.

        The cookie-retention lever, wired into the ladder: every response on
        the way through contributes the retained names (CLEARANCE_COOKIES +
        TRUST_COOKIES) to the per-domain jar -- 200s included, and challenge
        responses too, which is where vendors set their visitor ids. Tier 0/1
        then arrive at the NEXT fetch looking like a browser that has been
        here before, not a first contact. ``jar`` is a :class:`_HopCookies`
        already fed the response's (or the redirect chain's) raw Set-Cookie
        headers, so Domain validation, the public-suffix rule, and Secure
        all applied at collection; ``for_host`` applies the send-scope here.

        Merge, not replace: an existing fresh row keeps cookies the new
        harvest lacks (a browser-earned cf_clearance must survive a later
        cheap-tier harvest that only carried bm_sz). The row's UA updates
        only when the harvest carries a clearance-class name -- the UA-bound
        tokens (_abck, cf_clearance, bm_sz) were earned by the request whose
        UA we hold -- while a pure trust-name harvest (_pxvid/_pxhd/bm_sv
        are not UA-bound) leaves the bug-72/73 pin untouched. When a merged
        UA does contradict an older cookie, the origin's 403 routes through
        the existing clearance drop() and self-heals; never harvesting at
        all is the old status quo, so the merge can only help.
        """
        try:
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
            if not host:
                return
            found = jar.for_host(host, parts.scheme or "https")
            new = {k: v for k, v in found.items() if k in RETAINED_COOKIES}
            if not new:
                return
            dom = domain_of(url)
            existing = self.clearance.get(dom)
            if existing:
                merged = {**existing.cookies, **new}
                # A clearance-class name in the harvest means a UA-bound
                # token just changed hands -- the earning request's UA owns
                # the row. A trust-only harvest keeps the existing pin.
                row_ua = ua if (set(new) & CLEARANCE_COOKIES) else (
                    existing.user_agent or ua)
            else:
                merged = new
                row_ua = ua
            self.clearance.put(dom, merged, row_ua)
            self._bump("trust_cookies_captured")
        except Exception:  # noqa: BLE001 -- bookkeeping never fails a fetch
            pass

    @staticmethod
    def _jar_from_set_cookie(set_cookie_headers: list, host: str) -> _HopCookies:
        """A validated one-response jar, for tiers whose response headers are
        only reachable at their own call site (tier 0's live response)."""
        jar = _HopCookies()
        try:
            for sc in set_cookie_headers or []:
                jar.set_cookie(str(sc), host)
        except Exception:  # noqa: BLE001 -- a cookie is never worth a fetch
            pass
        return jar

    async def _raw_get(self, url: str) -> tuple[int, str]:
        """Unpoliced GET used only for robots.txt, which must not recurse.

        Redirects still follow (RFC 9309 allows at least 5 hops), but
        through the same policy-checked loop as the tiers: a hostile origin
        answering /robots.txt with a 302 to the metadata endpoint is the
        same SSRF shape as any page redirect.
        """
        try:
            r = await self._open_stream(url, {}, 8.0)
            try:
                # RFC 9309: only the first 500 KiB of a robots.txt is
                # parsed anyway -- truncate there rather than refuse.
                cap = 500 * 1024
                chunks: list[bytes] = []
                total = 0
                async for chunk in r.aiter_bytes(65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > cap:
                        break
                raw = b"".join(chunks)[:cap]
                return r.status_code, raw.decode(
                    r.charset_encoding or "utf-8", errors="replace")
            finally:
                await r.aclose()
        except Exception:
            return 0, ""

    def _bump(self, key: str) -> None:
        self._stats[key] = self._stats.get(key, 0) + 1
