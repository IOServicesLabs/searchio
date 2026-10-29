"""General web search from engines that publish plain HTML.

DuckDuckGo's HTML and Lite endpoints exist for browsers without JavaScript, and
they answer a plain GET with a complete, parseable result list. That makes them
the highest-value keyless source available: real web coverage, no API key, no
quota, and no JavaScript.

They are also, obviously, the most likely thing here to be rate-limited, which
is why these providers go through the ladder like everything else -- so a
DuckDuckGo 403 backs the domain off and escalates rather than hammering.

The last resort is the SwarmIO sidecar's own ``search_engine_results`` verb,
which drives a real browser through the same engines and clears challenges. It
is slower by an order of magnitude, so it only runs when the HTML paths have
all failed.
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import urllib.parse
from collections import deque
from html import unescape

from ..errors import Blocked, ProviderError, SearchioError
from ..models import Capability, Doc, Query
from .base import Provider, ProviderContext

_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _WS.sub(" ", unescape(_TAGS.sub(" ", s or ""))).strip()


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results in /l/?uddg=<encoded>. Unwrap to the real URL."""
    if "uddg=" in href:
        try:
            parts = urllib.parse.urlsplit(href if "://" in href else "https:" + href)
            host = (parts.hostname or "").lower()
            if not (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")):
                # Only DuckDuckGo's own redirector carries the target in
                # uddg (iteration 55 rider): a foreign page with a uddg
                # query parameter used to be REPLACED by whatever that
                # parameter said, and the agent cited the wrong page.
                return "https:" + href if href.startswith("//") else href
            qs = urllib.parse.parse_qs(parts.query)
            if qs.get("uddg"):
                # parse_qs already percent-decoded the value once, and DDG
                # single-encodes uddg -- a second unquote() decoded the
                # TARGET's own encoding too (a literal %20 in its path, a
                # %26/%2B inside a query value) and emitted a different URL
                # (bug 51). One decode is exactly right.
                return qs["uddg"][0]
        except Exception:
            pass
    if href.startswith("//"):
        return "https:" + href
    return href


# DuckDuckGo HTML result block: an anchor with class result__a, then a snippet.
_DDG_HTML = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?P<rest>.*?)(?=<a[^>]+class="[^"]*result__a|</body>)',
    re.I | re.S,
)
_DDG_SNIP = re.compile(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.I | re.S)

# Lite is a bare table: result links carry class result-link.
_DDG_LITE = re.compile(
    r'<a[^>]+class="[^"]*result-link[^"]*"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.I | re.S,
)

# Bing wraps every organic href in a /ck/a redirect whose `u` parameter holds
# the destination, base64url-encoded behind a two-character marker. The <h2>
# carries attributes, so it cannot be matched as a bare tag.
_BING = re.compile(r'<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>(?P<block>.*?)</li>', re.I | re.S)
_BING_LINK = re.compile(
    r'<h2[^>]*>\s*<a[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.I | re.S
)
_BING_CAPTION = re.compile(r'class="[^"]*b_caption[^"]*"[^>]*>(?P<t>.*?)</div>', re.I | re.S)


def _unwrap_bing(href: str) -> str:
    """Decode a Bing /ck/a redirect back to the destination URL."""
    if "bing.com/ck/a" not in href:
        return href
    try:
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(unescape(href)).query)
        raw = (qs.get("u") or [""])[0]
        if not raw:
            return href
        if raw.startswith("a1"):  # Bing-internal marker, not part of the payload
            raw = raw[2:]
        pad = "=" * (-len(raw) % 4)
        return base64.urlsafe_b64decode(raw + pad).decode("utf-8", "replace")
    except Exception:
        return href


# A parsed result that still points at the provider's own redirect or ad
# endpoint is not a result. DDG's ad blocks ride the same result__a anchors as
# organic results, with /y.js hrefs, and an /l/ link whose uddg param is
# missing is an unresolvable click tracker; a /ck/a link that did not decode
# is the same for Bing. Agents cite and fetch whatever the SERP parser
# returns, so a leaked redirector costs a real fetch against an ad network
# (bench/span.py run 080002, forum.reddit_reach: the TOP result was
# duckduckgo.com/y.js?ad_domain=... and the scenario fetched the ad hop).
_SELF_REDIRECTS = {
    "duckduckgo.com": ("/y.js", "/l/"),
    "bing.com": ("/ck/a",),
}


def _is_self_redirect(url: str) -> bool:
    """True when a parsed result URL is the provider's own redirect/ad hop."""
    if "://" not in url:
        return False
    host_path = url.split("://", 1)[1]
    host, _, path = host_path.partition("/")
    # Case-fold the path too and drop an explicit port: DuckDuckGo.com/Y.js
    # and duckduckgo.com:443/y.js slipped past a guard that only lower-cased
    # the host (iteration 37 hardening -- providers do not emit these today,
    # and bug 17's class is exactly provider shape drift).
    host = host.lower().split(":", 1)[0]
    # Percent-decode before matching (iteration 55 rider): /ck%2Fa is the
    # same hop as /ck/a to the server, and slipped past the marker check.
    path = "/" + urllib.parse.unquote(path).lower()
    for own, markers in _SELF_REDIRECTS.items():
        if host == own or host.endswith("." + own):
            return any(m in path for m in markers)
    return False


def _tree(html: str):
    """Parse to a DOM, or None if selectolax is unavailable.

    Regex over SERP markup works right up until the engine ships a layout
    variant, and then it silently returns the wrong links rather than none --
    the failure mode that is hardest to notice. A real parser makes the
    selectors say what they mean, so the regex paths below survive only as a
    fallback for when the optional dependency is missing.
    """
    try:
        from selectolax.parser import HTMLParser

        return HTMLParser(html)
    except Exception:
        return None


def parse_ddg_html(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    tree = _tree(html)
    if tree is None:
        return _parse_ddg_html_re(html, limit)
    out: list[tuple[str, str, str, str]] = []
    for node in tree.css("a.result__a"):
        url = _unwrap_ddg(node.attributes.get("href") or "")
        if not url.startswith("http") or _is_self_redirect(url):
            continue
        snippet = ""
        # The snippet lives in a sibling subtree; walk up to the result body.
        holder = node.parent
        for _ in range(4):
            if holder is None:
                break
            found = holder.css_first("a.result__snippet, .result__snippet")
            if found is not None:
                snippet = _clean(found.text())
                break
            holder = holder.parent
        out.append((url, _clean(node.text()), snippet, ""))
        if len(out) >= limit:
            break
    return out


def parse_ddg_lite(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    tree = _tree(html)
    if tree is None:
        return _parse_ddg_lite_re(html, limit)
    out: list[tuple[str, str, str, str]] = []
    for node in tree.css("a.result-link"):
        url = _unwrap_ddg(node.attributes.get("href") or "")
        if url.startswith("http") and not _is_self_redirect(url):
            out.append((url, _clean(node.text()), "", ""))
        if len(out) >= limit:
            break
    return out


def parse_bing(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    """Organic Bing results only.

    Bing mixes ads, brand panels, videos and "people also ask" into the same
    result column. Anchoring on ``li.b_algo > h2 > a`` keeps us to the organic
    list; taking every ``h2`` on the page is how you end up returning a
    company's homepage instead of the reviews that were asked for.
    """
    tree = _tree(html)
    if tree is None:
        return _parse_bing_re(html, limit)
    out: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for li in tree.css("li.b_algo"):
        a = li.css_first("h2 > a") or li.css_first("h2 a")
        if a is None:
            continue
        url = _unwrap_bing(a.attributes.get("href") or "")
        if not url.startswith("http") or url in seen or _is_self_redirect(url):
            continue
        seen.add(url)
        cap = li.css_first(".b_caption p") or li.css_first(".b_caption") or li.css_first("p")
        pub = ""
        out.append((url, _clean(a.text()), _clean(cap.text())[:300] if cap is not None else "", pub))
        if len(out) >= limit:
            break
    return out


# ── regex fallbacks (used only when selectolax is unavailable) ───────────────


def _parse_ddg_html_re(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    for m in _DDG_HTML.finditer(html):
        url = _unwrap_ddg(unescape(m.group("url")))
        if not url.startswith("http") or _is_self_redirect(url):
            continue
        snip_m = _DDG_SNIP.search(m.group("rest") or "")
        out.append((url, _clean(m.group("title")), _clean(snip_m.group(1)) if snip_m else "", ""))
        if len(out) >= limit:
            break
    return out


def _parse_ddg_lite_re(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    for m in _DDG_LITE.finditer(html):
        url = _unwrap_ddg(unescape(m.group("url")))
        if url.startswith("http") and not _is_self_redirect(url):
            out.append((url, _clean(m.group("title")), "", ""))
        if len(out) >= limit:
            break
    return out


def _parse_bing_re(html: str, limit: int) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    for m in _BING.finditer(html):
        link = _BING_LINK.search(m.group("block"))
        if not link:
            continue
        url = _unwrap_bing(unescape(link.group("url")))
        if not url.startswith("http") or url in seen or _is_self_redirect(url):
            continue
        seen.add(url)
        cap = _BING_CAPTION.search(m.group("block"))
        out.append((url, _clean(link.group("title")), _clean(cap.group("t"))[:300] if cap else "", ""))
        if len(out) >= limit:
            break
    return out


# ── soft-degradation detection ───────────────────────────────────────────────
#
# The failure mode nobody codes for. A search engine that has decided it does
# not like you does not always answer 403 -- it answers 200 with a *plausible*
# result set that is quietly useless. Measured against Bing from a datacenter
# IP: "postgres index bloat vacuum" returned the PostgreSQL homepage, "rust
# async runtime comparison" returned rust-lang.org. Ten results, valid markup,
# correct schema, no error anywhere; the engine had simply substituted the
# dominant brand entity for the actual query.
#
# Status codes cannot see this and neither can markup heuristics. The only
# signal is relevance: on a real SERP most results echo the distinctive terms
# of the query somewhere in title, snippet, or URL. When almost none do, we are
# being handed a brush-off, and it should be treated like the block it is --
# back the domain off, and let another provider answer.

_STOPWORDS = {
    "the", "and", "for", "with", "how", "what", "why", "best", "vs", "versus",
    "review", "reviews", "price", "buy", "top", "guide", "tutorial", "is", "are",
    "of", "to", "in", "on", "a", "an", "does", "do", "can", "should",
}


def query_terms(text: str) -> list[str]:
    """Distinctive tokens from a query, in the sense that matters for relevance."""
    toks = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
    return [t for t in toks if t not in _STOPWORDS]


def specific_terms(terms: list[str]) -> list[str]:
    """The tokens that actually pin a query down.

    Matching on *any* query term is useless for detecting degradation, because
    the brand term is exactly what a degraded SERP returns: every one of
    "Sony Electronics", "PlayStation", "Sony Group Portal" contains "sony".
    What none of them contain is "wh-1000xm5".

    Only digit-bearing tokens (model codes, version numbers, part numbers)
    qualify. Long words used to count too, but vocabulary words like
    "documentation" are long without being distinctive -- real results for
    "nextjs app router documentation" say "docs" and "App Router", so the
    rule condemned two perfectly good SERPs in one run (bench/span.py
    web.js_docs). Queries without a model code fall through to the coverage
    rule, which is where brush-offs like the PostgreSQL homepage are caught.
    """
    from ..config import get_settings

    if getattr(get_settings(), "degraded_codes_strict", False):
        # "6-speed", "3-way", "2-in-1" carry a digit but are vocabulary:
        # results say "manual" and the rule condemned a good SERP. A part
        # must mix letters and digits (1000xm5) or be 3+ digits (2023).
        return [t for t in terms if any(
            (any(c.isdigit() for c in p) and any(c.isalpha() for c in p))
            or (p.isdigit() and len(p) >= 3) for p in t.split("-"))]
    return [t for t in terms if any(c.isdigit() for c in t)]


def looks_degraded(query: str, rows: list[tuple], *, floor: float = 0.34) -> bool:
    """True when a result set does not appear to answer the query at all.

    Two rules, in order of confidence:

    1. If the query contains distinctive tokens, a real SERP will echo them.
       When almost no result does, the engine has substituted something else.
    2. Otherwise fall back to mean term coverage -- a degraded set matches the
       one dominant word and nothing else.

    Deliberately forgiving in both branches. A false positive here discards a
    perfectly good result set, so the thresholds are set to fire only on the
    blatant cases actually observed in the wild.
    """
    terms = query_terms(query)
    if not terms or len(rows) < 3:
        return False
    hays = [f"{t} {s} {u}".lower() for u, t, s, *_ in rows]

    specific = specific_terms(terms)
    if specific:
        hits = sum(1 for h in hays if any(t in h for t in specific))
        return (hits / len(hays)) < floor

    coverage = sum(sum(1 for t in terms if t in h) / len(terms) for h in hays) / len(hays)
    return coverage < 0.30


def _apply_operators(q: Query) -> str:
    """Fold allow/deny domain lists into engine query operators.

    Freshness is deliberately NOT an operator: DuckDuckGo takes it as the
    ``df`` URL parameter and Bing as ``filters`` -- both are appended by the
    providers below, where the URL is built. (It used to be dropped on the
    floor here; bench/span.py news.fresh_week caught that.)
    """
    text = q.text
    if q.domains:
        text += " (" + " OR ".join(f"site:{d}" for d in q.domains) + ")"
    for d in q.exclude_domains:
        text += f" -site:{d}"
    return text


#: Query.freshness -> DuckDuckGo's df parameter.
_DDG_FRESH = {"day": "d", "week": "w", "month": "m", "year": "y"}


def _bing_freshness(freshness: str) -> str:
    """Query.freshness -> Bing's ``filters`` recency parameter.

    Both the RSS and HTML endpoints honor it. A year has no ez code, so it
    uses the custom range form with OLE date serials (days since 1899-12-30),
    the same format bing.com's own UI emits.
    """
    if freshness == "day":
        return 'ex1:"ez1"'
    if freshness == "week":
        return 'ex1:"ez2"'
    if freshness == "month":
        return 'ex1:"ez3"'
    if freshness == "year":
        from datetime import date, timedelta

        epoch = date(1899, 12, 30)
        today = date.today()
        return f'ex1:"ez5_{(today - timedelta(days=365) - epoch).days}_{(today - epoch).days}"'
    return ""


async def run_hedged(attempts, hedge_ms: int, stop=None) -> tuple[list[Doc] | None, list[str]]:
    """Run endpoint attempts in fallback order, optionally hedged.

    Each attempt is a zero-arg coroutine factory returning ``(docs, err)``:
    docs on success, ``(None, reason)`` on failure. With ``hedge_ms <= 0``
    this is the plain sequential fallback. Otherwise attempt i+1 starts as
    soon as attempt i fails OR has been running ``hedge_ms`` without an
    answer, and the first success in hand wins; the losers are cancelled. A
    slow primary then costs the fallback's latency, not the sum of both.
    Errors come back in attempt order, so the provider's refusal text reads
    the same either way.

    ``stop(err)`` returning True on a failed attempt ends the fallback: no
    later attempt starts and running ones are cancelled. A provider uses it
    when one endpoint's failure already answers for its siblings.
    """
    if hedge_ms <= 0 or len(attempts) < 2:
        errors: list[str] = []
        for make in attempts:
            docs, err = await make()
            if docs is not None:
                return docs, errors
            errors.append(err)
            if stop is not None and stop(err):
                break
        return None, errors

    hedge_s = hedge_ms / 1000.0
    tasks: list[asyncio.Task] = []
    errs: dict[int, str] = {}

    def launch() -> None:
        tasks.append(asyncio.ensure_future(attempts[len(tasks)]()))

    launch()
    try:
        while True:
            running = [t for t in tasks if not t.done()]
            more = len(tasks) < len(attempts)
            if not running:
                if not more:
                    break
                launch()  # everything so far failed: fall back now
                continue
            done, _ = await asyncio.wait(
                running, timeout=hedge_s if more else None,
                return_when=asyncio.FIRST_COMPLETED)
            # Harvest every finished attempt, in attempt order: a success
            # from the primary beats a same-tick success from the hedge.
            for i, t in enumerate(tasks):
                if t.done() and i not in errs:
                    docs, err = t.result()
                    if docs is not None:
                        return docs, [errs[j] for j in sorted(errs)]
                    errs[i] = err
                    if stop is not None and stop(err):
                        return None, [errs[j] for j in sorted(errs)]
            if not done and more:
                launch()  # the primary is slow: hedge
        return None, [errs[j] for j in sorted(errs)]
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


class DuckDuckGo(Provider):
    """DuckDuckGo via its no-JavaScript HTML endpoints."""

    name = "duckduckgo"
    caps = frozenset({Capability.WEB, Capability.NEWS, Capability.SHOPPING, Capability.FORUM})
    primary_domain = "duckduckgo.com"

    ENDPOINTS = (
        ("https://html.duckduckgo.com/html/?q={q}", parse_ddg_html),
        ("https://lite.duckduckgo.com/lite/?q={q}", parse_ddg_lite),
    )

    def __init__(self) -> None:
        super().__init__()
        self._order = _EndpointOrder()

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        query = urllib.parse.quote_plus(_apply_operators(q))
        fresh = _DDG_FRESH.get(q.freshness, "")

        def attempt(tmpl, parser):
            async def run() -> tuple[list[Doc] | None, str]:
                url = tmpl.format(q=query)
                if fresh:
                    url += f"&df={fresh}"
                try:
                    res = await ctx.ladder.fetch(url, max_tier=1)
                except Blocked as exc:
                    return None, f"{url.split('/')[2]}: blocked({exc.vendor})"
                except Exception as exc:
                    return None, f"{type(exc).__name__}"
                rows = await _parse(ctx, parser, res.body, q.k)
                if rows and looks_degraded(q.text, rows):
                    return None, "degraded_results"
                if rows:
                    return [
                        Doc(url=u, title=t, snippet=s, source=self.name, rank=i,
                            published=pub or None)
                        for i, (u, t, s, pub) in enumerate(rows)
                    ], ""
                return None, "no_results"
            return run

        s = getattr(ctx, "settings", None)
        # html and lite are one backend behind one IP reputation: a wall on
        # the first is the answer for both (measured live: lite drew the
        # same anomaly modal every time, ~3 s later, after the limiter's
        # post-block wait).
        stop = _is_wall if getattr(s, "block_skip_siblings", False) else None
        makers = [attempt(t, p) for t, p in self.ENDPOINTS]
        idx = self._order.order(len(makers), getattr(s, "endpoint_learning", False))
        docs, errors = await run_hedged(
            self._order.wrap(idx, makers), getattr(s, "hedge_ms", 0), stop)
        if docs is not None:
            return docs
        raise ProviderError(self.name, "; ".join(errors) or "no results",
                            walled=bool(errors) and all(_is_wall(e) for e in errors))


#: parse_offloop: bodies at least this large parse on a worker thread.
_OFFLOOP_MIN = 20_000


async def _parse(ctx, parser, body: str, k: int):
    """``parser(body, k)``, on a worker thread for a large body when
    ``parse_offloop`` is on: a ~124 KB Bing HTML SERP parsed on the loop
    holds every other provider's response handling behind it."""
    if getattr(getattr(ctx, "settings", None), "parse_offloop", False) and len(body) >= _OFFLOOP_MIN:
        return await asyncio.get_running_loop().run_in_executor(None, parser, body, k)
    return parser(body, k)


class _EndpointOrder:
    """endpoint_learning: lead with the endpoint that answered last time.

    When a provider's primary failed and a fallback answered, the next
    search starts on that fallback instead of re-paying the primary's
    failure first. Self-correcting: if the lead fails, the remaining order
    runs as usual and whichever answers becomes the lead (the primary
    winning resets it).
    """

    def __init__(self) -> None:
        self.lead: int | None = None

    def order(self, n: int, on: bool) -> list[int]:
        idx = list(range(n))
        if on and self.lead is not None and 0 < self.lead < n:
            idx.remove(self.lead)
            idx.insert(0, self.lead)
        return idx

    def wrap(self, idx: list[int], makers: list) -> list:
        """Attempts in ``idx`` order, each noting itself as the winner."""
        def tag(i, make):
            async def run():
                docs, err = await make()
                if docs is not None:
                    self.lead = i if i != 0 else None
                return docs, err
            return run
        return [tag(i, makers[i]) for i in idx]


def _is_wall(err: str) -> bool:
    """An attempt's error text says an anti-bot wall refused it."""
    return "blocked(" in err


_RSS_ITEM = re.compile(r"<item>(.*?)</item>", re.I | re.S)
_RSS_FIELD = {
    f: re.compile(rf"<{f}>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</{f}>", re.I | re.S)
    for f in ("title", "link", "description", "pubDate")
}


def _rss_date(raw: str) -> str:
    """RSS pubDate -> strict ISO date, or "" when it will not parse.

    ``published`` is a machine field: the router's stale post-filter reads it
    with ``date.fromisoformat``. Returning the raw pubDate on a parse failure
    (bug 171) shipped a non-ISO string that escaped every freshness bound as
    "unprovable" -- bug 19's exact hole -- and handed the agent a fake date.
    An un-normalizable date is no date: emit nothing.
    """
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(raw.strip()).date().isoformat()
    except Exception:
        return ""


def parse_bing_rss(xml: str, limit: int) -> list[tuple[str, str, str, str]]:
    """Parse Bing's RSS SERP.

    Preferred over the HTML page: ~5 KB instead of ~124 KB, a schema that does
    not churn with front-end redesigns, and destination URLs given directly
    rather than wrapped in a base64 redirect. Unlike the HTML SERP, the feed
    carries ``<pubDate>`` per item -- the only Bing path that knows when a
    result was published.
    """
    out: list[tuple[str, str, str, str]] = []
    for block in _RSS_ITEM.findall(xml):
        link = _RSS_FIELD["link"].search(block)
        title = _RSS_FIELD["title"].search(block)
        desc = _RSS_FIELD["description"].search(block)
        pub = _RSS_FIELD["pubDate"].search(block)
        url = unescape((link.group(1) if link else "").strip())
        if not url.startswith("http"):
            continue
        out.append(
            (
                url,
                _clean(title.group(1) if title else ""),
                _clean(desc.group(1) if desc else "")[:300],
                _rss_date(pub.group(1)) if pub else "",
            )
        )
        if len(out) >= limit:
            break
    return out


class Bing(Provider):
    """Bing, via its RSS output with the HTML SERP as fallback.

    Second opinion when DuckDuckGo is thin or blocked. Note that Bing is the
    provider most prone to the soft degradation described above, so its results
    go through :func:`looks_degraded` before they are trusted.
    """

    name = "bing"
    caps = frozenset({Capability.WEB, Capability.NEWS, Capability.SHOPPING})
    primary_domain = "bing.com"
    #: degraded_skip re-asks the HTML SERP one time in this many.
    PROBE_ODDS = 8

    def __init__(self) -> None:
        super().__init__()
        #: Did the HTML fallback answer, the last few times RSS was degraded?
        self._html_after_degraded: deque[bool] = deque(maxlen=3)
        self._order = _EndpointOrder()

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        text = urllib.parse.quote_plus(_apply_operators(q))
        # Pin the market and language explicitly. Bing geolocates from the
        # egress IP and will happily answer an English query with Chinese
        # results from a container in the wrong region, regardless of
        # Accept-Language.
        loc = f"&mkt={q.locale}&setlang={q.locale.split('-')[0]}"
        fresh = _bing_freshness(q.freshness)
        if fresh:
            loc += "&filters=" + urllib.parse.quote(fresh)
        s = getattr(ctx, "settings", None)
        # bing_html_tier1: on hard queries Bing brushes off its RSS endpoint
        # (any client) and its HTML SERP fetched with httpx's fingerprint, but
        # answers the HTML SERP honestly for a Chrome-impersonating client
        # (2026-10-28 probe: "Chilton repair manual" vs real used-car hits).
        html_tier = 1 if getattr(s, "bing_html_tier1", False) else None
        html_first = getattr(s, "bing_html_first", False)
        rss = (f"https://www.bing.com/search?q={text}&format=rss"
               f"&count={max(q.k, 10)}{loc}", parse_bing_rss, None)
        html = (f"https://www.bing.com/search?q={text}{loc}", parse_bing, html_tier)
        attempts = (html, rss) if html_first else (rss, html)

        def attempt(url, parser, tier=None):
            async def run() -> tuple[list[Doc] | None, str]:
                try:
                    res = await ctx.ladder.fetch(url, max_tier=1, force_tier=tier)
                except Blocked as exc:
                    return None, f"blocked({exc.vendor})"
                except SearchioError as exc:
                    # Any fetch failure -- a transient reset/timeout, a
                    # refusal -- is this ATTEMPT failing, not Bing failing
                    # (bug 172): the RSS->HTML ladder exists so a flaky RSS
                    # fetch falls back to the HTML SERP. Only Blocked was
                    # caught before, so a common transient RSS error skipped
                    # the fallback and failed the provider outright. The
                    # parse call stays OUTSIDE the try, so a parser
                    # regression still surfaces.
                    return None, f"{type(exc).__name__}: {str(exc)[:60]}"
                rows = await _parse(ctx, parser, res.body, q.k)
                if not rows:
                    return None, "no_results"
                if looks_degraded(q.text, rows):
                    # Not an error we can retry around -- the engine is
                    # brushing this client off. Report it so the router opens
                    # the breaker and the fan-out leans on providers that are
                    # still answering honestly.
                    return None, "degraded_results"
                return [
                    Doc(url=u, title=t, snippet=s, source=self.name, rank=i,
                        published=pub or None)
                    for i, (u, t, s, pub) in enumerate(rows)
                ], ""
            return run

        # The HTML-after-degraded bookkeeping assumes RSS goes first.
        learn = getattr(s, "degraded_skip", False) and not html_first

        def stop(err: str) -> bool:
            if getattr(s, "block_skip_siblings", False) and _is_wall(err):
                return True
            # A degraded RSS is the engine brushing this egress off; when the
            # HTML SERP has said the same the last few times, asking it again
            # costs a round trip for an answer we already know. One in
            # PROBE_ODDS still asks, so a recovered HTML path is noticed.
            return (learn and err == "degraded_results"
                    and len(self._html_after_degraded) == self._html_after_degraded.maxlen
                    and not any(self._html_after_degraded)
                    and random.randrange(self.PROBE_ODDS) != 0)

        makers = [attempt(u, p, t) for u, p, t in attempts]
        idx = self._order.order(len(makers), getattr(s, "endpoint_learning", False))
        docs, errors = await run_hedged(
            self._order.wrap(idx, makers), getattr(s, "hedge_ms", 0), stop)
        if (not html_first and idx[0] == 0 and errors[:1] == ["degraded_results"]
                and (docs is not None or len(errors) > 1)):
            # The HTML fallback ran after a degraded RSS: remember how it did.
            self._html_after_degraded.append(docs is not None)
        if docs is not None:
            return docs
        raise ProviderError(self.name, "; ".join(errors) or "no results",
                            walled=bool(errors) and all(_is_wall(e) for e in errors))


class SidecarSearch(Provider):
    """Search through SwarmIO's browser, for when plain HTTP is walled off.

    Deliberately expensive in the router's eyes (see ``cost_per_1k``) so it is
    chosen last: it launches a real browser, drives the engine's UI, and clears
    challenges, which is exactly what you want when nothing else works and
    exactly what you do not want to pay for routine queries.
    """

    name = "sidecar_search"
    caps = frozenset({Capability.WEB, Capability.NEWS, Capability.SHOPPING, Capability.LOCAL})
    cost_per_1k = 5.0  # not dollars; a latency/resource penalty in the same units

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        ladder = getattr(ctx, "ladder", None)
        sc = None
        if ladder is not None:
            # The verb is patchright-only, so ask the ladder for whichever
            # backend can serve it: the primary when patchright, else the
            # challenge sidecar (the fidelity pairing is why it exists).
            # Test doubles only carry .sidecar -- fall back to that.
            fid = getattr(ladder, "fidelity_sidecar", None)
            sc = fid() if callable(fid) else getattr(ladder, "sidecar", None)
        if sc is None or not sc.available:
            raise ProviderError(self.name, "sidecar unavailable")
        if await sc.backend_kind() == "engine":
            # Documented divergence (searchio-engine docs/sidecar-protocol.md):
            # search_engine_results is one of the patchright script's
            # composite conveniences, composed in Python over Playwright --
            # se-serve deliberately does not implement it, and with engine on
            # both sidecar knobs there is no patchright to ask. Fail fast
            # with the reason instead of relaying a bare wire
            # "Unknown method".
            raise ProviderError(
                self.name,
                "search_engine_results is a patchright-only verb and no "
                "patchright sidecar is configured",
            )
        res = await sc.search(_apply_operators(q), engine="auto", k=q.k)
        if not res.get("ok"):
            raise ProviderError(self.name, str(res.get("error"))[:200])
        docs: list[Doc] = []
        for r in res.get("results") or []:
            # The composite verb hands back raw SERP anchors: a protocol-
            # relative uddg wrapper (which "startswith http" silently DROPPED
            # as a non-result), a Bing /ck/a wrapper, and DDG's /y.js ad hops
            # -- which the sidecar unwraps but never filters. Every other
            # parse site unwraps then applies the bug-17 redirector guard;
            # this one emitted the ad hop for an agent to cite and fetch
            # (bug 52). Rank follows the EMITTED order.
            url = _unwrap_bing(_unwrap_ddg(r.get("url") or ""))
            if not url.startswith("http") or _is_self_redirect(url):
                continue
            docs.append(
                Doc(
                    url=url,
                    title=_clean(r.get("title") or ""),
                    snippet=_clean(r.get("snippet") or r.get("text") or ""),
                    source=self.name,
                    rank=len(docs),
                    meta={"engine": res.get("engine", "")},
                )
            )
        return docs
