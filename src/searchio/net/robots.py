"""robots.txt, with a policy switch rather than a hardcoded stance.

Three settings, because the honest answer differs by what you are fetching:

* ``enforce`` -- a disallowed path is not fetched. The right default for broad
  crawling of sites you have no relationship with.
* ``warn`` (default) -- fetch, but record the disallow on the result so it
  surfaces in logs and in the API response. Chosen as the default because
  searchio's normal mode is retrieving a handful of documents a user explicitly
  asked for, which is much closer to a browser following a link than to a
  crawler sweeping a site, and many robots.txt files disallow paths that serve
  the exact page a person just requested.
* ``off`` -- do not fetch robots.txt at all.

``crawl-delay`` is honored when present regardless of policy, since it is a
direct statement of what rate a host is willing to serve, which is information
worth having even when the disallow rules are advisory here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

UA = "searchio"

#: Rules per group are capped so a 500 KiB hostile robots.txt cannot turn
#: every check into a scan of hundreds of thousands of regexes.
MAX_RULES = 2000


@dataclass
class _Rule:
    allow: bool
    pattern: str
    glob: str  # pattern minus a trailing `$`, plus a trailing `*` when unanchored


@dataclass
class _Group:
    rules: list[_Rule] = field(default_factory=list)
    crawl_delay: float | None = None


_UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")


def _canon(path: str) -> str:
    """One spelling for a path or a rule pattern (bug 152, RFC 9309 s2.2.2).

    Percent-encoded UNRESERVED octets are decoded ("%61" is "a"); every
    other escape keeps its bytes with upper-case hex ("%2f" stays a
    reserved "/", so "/a%2Fb" is not "/a/b"); non-ASCII and the characters a
    URL cannot carry bare are re-encoded as UTF-8 percent-escapes. Without
    this, under ``enforce`` a request for "/%61dmin" slipped past
    "Disallow: /admin" and "/caf\u00e9" past "Disallow: /caf%C3%A9". The glob
    metacharacters '*' and '$' are left alone so patterns still compile.
    """
    out: list[str] = []
    i = 0
    n = len(path)
    while i < n:
        ch = path[i]
        if ch == "%" and i + 2 < n and _is_hex(path[i + 1:i + 3]):
            code = int(path[i + 1:i + 3], 16)
            if chr(code) in _UNRESERVED:
                out.append(chr(code))
            else:
                out.append("%" + path[i + 1:i + 3].upper())
            i += 3
            continue
        if ch in _UNRESERVED or ch in "/*$?=&+,;:@!'()":
            out.append(ch)
        else:
            out.append("".join(f"%{b:02X}" for b in ch.encode("utf-8")))
        i += 1
    return "".join(out)


def _is_hex(s: str) -> bool:
    return len(s) == 2 and all(c in "0123456789abcdefABCDEF" for c in s)


def _compile(pattern: str) -> str:
    """RFC 9309 path-pattern -> glob: `*` matches any run, a trailing `$`
    anchors the end, everything else is a literal prefix (so an unanchored
    pattern gets a trailing `*`)."""
    pattern = _canon(pattern)
    if pattern.endswith("$"):
        return pattern[:-1]
    return pattern + "*"


def _glob(pat: str, s: str) -> bool:
    """Full-string glob match with `*` only, in O(len(pat) * len(s)).

    NOT a regex: a hostile robots.txt line like `Disallow: /*/*/*/*/*/x$`
    compiled to `.*` groups backtracks exponentially against a long path
    and wedged the fetch path for minutes (caught by the iteration-44
    adversarial probe of the first fix draft). The classic greedy
    backtrack-to-last-star scan has no such blowup.
    """
    i = j = 0
    star = -1
    mark = 0
    n, m = len(s), len(pat)
    while i < n:
        if j < m and pat[j] == "*":
            star, mark = j, i
            j += 1
        elif j < m and pat[j] == s[i]:
            i += 1
            j += 1
        elif star != -1:
            j = star + 1
            mark += 1
            i = mark
        else:
            return False
    while j < m and pat[j] == "*":
        j += 1
    return j == m


def parse_robots(text: str) -> dict[str, _Group]:
    """Groups keyed by lower-cased user-agent token (`*` for the default).

    RFC 9309 semantics replaced urllib.robotparser (bug 69): that parser
    knows neither `*` nor `$` and applies FIRST match instead of LONGEST
    match, so Google's own "Disallow: /search / Allow: /search/about"
    refused /search/about, and "Disallow: /*.pdf$" matched nothing at all
    -- under enforce, a wildcard disallow was silently bypassed.
    """
    groups: dict[str, _Group] = {}
    current: list[_Group] = []
    last_was_agent = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        fld, _, val = line.partition(":")
        fld = fld.strip().lower()
        val = val.strip()
        if fld == "user-agent":
            token = val.lower()
            if not last_was_agent:
                current = []
            g = groups.get(token)
            if g is None:
                g = groups[token] = _Group()
            current.append(g)
            last_was_agent = True
            continue
        last_was_agent = False
        if fld in ("allow", "disallow"):
            if not val:
                continue  # "Disallow:" (empty) means no restriction
            if not val.startswith(("/", "*")):
                val = "/" + val
            for g in current:
                if len(g.rules) < MAX_RULES:
                    g.rules.append(_Rule(fld == "allow", val, _compile(val)))
        elif fld == "crawl-delay":
            try:
                delay = float(val)
            except ValueError:
                continue
            for g in current:
                g.crawl_delay = delay if delay > 0 else None
    return groups


def _group_for(groups: dict[str, _Group], agent: str) -> _Group | None:
    """The most specific group whose token occurs in our product token;
    `*` only when no specific group matches (RFC 9309 §2.2.1)."""
    agent = agent.lower()
    best: tuple[int, _Group] | None = None
    for token, g in groups.items():
        if token != "*" and token in agent and (best is None or len(token) > best[0]):
            best = (len(token), g)
    if best is not None:
        return best[1]
    return groups.get("*")


def _allowed(group: _Group | None, path: str) -> bool:
    """Longest matching rule wins; a tie goes to allow; no match allows."""
    if group is None:
        return True
    best_len = -1
    best_allow = True
    path = _canon(path)
    for r in group.rules:
        if _glob(r.glob, path):
            n = len(r.pattern)
            if n > best_len or (n == best_len and r.allow):
                best_len, best_allow = n, r.allow
    return best_allow


@dataclass
class RobotsInfo:
    allowed: bool = True
    crawl_delay: float | None = None
    checked: bool = False
    reason: str = ""


@dataclass
class _Entry:
    group: _Group | None  # the rules that apply to us; None = unreadable
    fetched: float = field(default_factory=time.time)
    failed: bool = False
    present: bool = False  # a 200 robots.txt was read (a 404 is absence, not failure)


class RobotsCache:
    """Fetches and caches robots.txt per origin.

    A failure to fetch robots.txt is treated as permission, not refusal. The
    alternative -- refusing to read a site because its robots.txt 500'd -- turns
    an unrelated server error into a total outage for that host, and no crawler
    convention asks for that.
    """

    def __init__(self, fetcher, policy: str = "warn", ttl_s: int = 86400) -> None:
        self._fetch = fetcher  # async (url) -> (status, text)
        self.policy = policy
        self.ttl_s = ttl_s
        self._cache: dict[str, _Entry] = {}
        #: Single-flight for the robots.txt load (bug 170): a wave of
        #: concurrent same-host fetches all missed the cache and each
        #: fetched /robots.txt -- the politeness layer hammering the
        #: origin's robots.txt N times. Concurrent loads for one origin
        #: now share the first's future.
        self._inflight: dict[str, asyncio.Future[_Entry]] = {}

    async def check(self, url: str) -> RobotsInfo:
        if self.policy == "off":
            return RobotsInfo(allowed=True, checked=False, reason="policy_off")

        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        entry = self._cache.get(origin)
        if entry is None or time.time() - entry.fetched > self.ttl_s:
            entry = await self._load_once(origin)

        if not entry.present:
            return RobotsInfo(allowed=True, checked=False, reason="robots_unavailable")

        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        allowed = _allowed(entry.group, path)
        delay = entry.group.crawl_delay if entry.group is not None else None
        info = RobotsInfo(
            allowed=allowed,
            crawl_delay=float(delay) if delay else None,
            checked=True,
            reason="" if allowed else "disallowed_by_robots",
        )
        return info

    def permits(self, info: RobotsInfo) -> bool:
        """Whether the ladder should proceed, given the policy."""
        if info.allowed or self.policy != "enforce":
            return True
        return False

    async def _load_once(self, origin: str) -> _Entry:
        """Load ``origin``'s robots.txt, collapsing concurrent loads into one.

        A caller that arrives while a load for the same origin is in flight
        awaits that load's result instead of starting its own fetch (bug
        170). The in-flight slot is always cleared, so a failed load never
        wedges later waves.
        """
        fut = self._inflight.get(origin)
        if fut is not None:
            return await fut
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._inflight[origin] = fut
        try:
            entry = await self._load(origin)
            self._cache[origin] = entry
            if not fut.done():
                fut.set_result(entry)
            return entry
        except BaseException as exc:  # noqa: BLE001 -- propagate to every waiter
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(origin, None)

    async def _load(self, origin: str) -> _Entry:
        try:
            status, text = await self._fetch(f"{origin}/robots.txt")
        except Exception:
            return _Entry(group=None, failed=True)
        if status != 200 or not text:
            # 404 means no restrictions; anything else we could not read.
            return _Entry(group=None, failed=status != 404)
        return _Entry(group=_group_for(parse_robots(text), UA), present=True)
