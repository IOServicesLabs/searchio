"""The public face of searchio.

Three entry points at three very different price points, and choosing between
them is most of using this library well:

* :meth:`Engine.search` -- fan out to providers, fuse, return ranked documents.
  No model in the loop. Hundreds of milliseconds, free.
* :meth:`Engine.find_items` -- the same, plus fetching candidate product pages
  and normalising them into comparable listings. A few seconds, free.
* :meth:`Engine.research` -- the swarm. A lead agent plans, workers run in
  parallel, findings are synthesised with citations. Tens of seconds, and it
  costs real money.

The distinction matters because "look up an item quickly" and "research this
properly" are different questions wearing similar clothes, and answering the
first with the third is how a search tool becomes too slow and expensive to
use. The swarm earns its cost on open-ended questions; it is pure overhead on
"what does this cost".
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit

from .config import Settings, get_settings
from .errors import ProviderError
from .models import Doc, Item, Query, ResearchResult
from .net.ladder import Ladder
from .providers.base import ProviderContext
from .providers.facebook import FacebookMarketplace
from .providers.marketplace import MarketplaceItems, SidecarListings
from .providers.registry import Registry
from .router import RouteResult, Router

#: Inline ``site:`` / ``-site:`` operators, the way search engines speak them.
#: ``site:host`` and ``-site:host``, plus the colon-less ``site host`` (and
#: ``site: host``) a weaker model writes (iteration 56) -- only when the next
#: token is domain-shaped, so "building site safety" stays words.
_SITE_OP_RE = re.compile(
    r"(?<![^\s(\[\"'])(-?)site(?::\s*([^\s]+)|\s+((?=[A-Za-z0-9.-]*\.[A-Za-z]{2,})[^\s]+))",
    re.IGNORECASE)
#: Bare boolean glue left behind once the operators around it are pulled
#: ("site:a.gov OR site:b.int moon" -> "moon", not "OR moon").
_DANGLING_BOOL_RE = re.compile(r"(?<!\S)(?:OR|AND|\|)(?!\S)")
_LOCALE_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
#: lean_miss_memo: a lean fetch that ended on a 404/410 -- the lean form does
#: not exist for this URL (anything else may be transient).
_LEAN_GONE = re.compile(r"http_(?:404|410)'\]")


def _relevance_gate(q: Query, res: RouteResult, floor: float) -> RouteResult:
    """relevance_gate: drop docs whose title, snippet and url carry under
    ``floor`` of the query's distinctive terms. When the only engines left
    answering are brushing the client off or answering a different question
    (Wikipedia's "List of Ford factories" for a used-car query), an honest
    empty beats a confident page of junk. Queries with fewer than 3
    distinctive terms are left alone: coverage is too coarse there."""
    import dataclasses

    from .providers.engines import query_terms

    terms = query_terms(q.text)
    if len(terms) < 3:
        return res
    keep: list[Doc] = []
    dropped: dict[str, int] = {}
    for d in res.docs:
        hay = f"{d.title} {d.snippet} {d.url}".lower()
        if sum(t in hay for t in terms) / len(terms) >= floor:
            keep.append(d)
        else:
            dropped[d.source] = dropped.get(d.source, 0) + 1
    if not dropped:
        return res
    return dataclasses.replace(res, docs=keep, off_topic=dropped)


def _locale_fields(locale: str) -> dict:
    """Query locale/region from a BCP-47 tag; {} keeps the defaults.

    Subtags, not a split-on-first-dash (bug 131): "zh-Hant-TW" used to map
    to locale "zh-HANT-TW" with no region -- an invalid market tag Bing
    ignores (en-US results for a Traditional-Chinese question) and no gl=
    for YouTube. The region is the 2-letter (or 3-digit) subtag wherever it
    sits; a script subtag is title-cased; garbage is ignored.
    """
    loc = (locale or "").strip().replace("_", "-")
    if not loc or len(loc) > 35 or not _LOCALE_RE.match(loc):
        return {}
    parts = loc.split("-")
    canon = [parts[0].lower()]
    region = ""
    for p in parts[1:]:
        if len(p) == 4 and p.isalpha():
            canon.append(p.title())
        elif (len(p) == 2 and p.isalpha()) or (len(p) == 3 and p.isdigit()):
            canon.append(p.upper())
            if p.isalpha() and not region:
                region = p.lower()
        else:
            canon.append(p.lower())
    out = {"locale": "-".join(canon)}
    if region:
        out["region"] = region
    return out


def _operator_host(raw: str) -> str:
    """Reduce a ``site:`` operator value to a bare, lower-cased host.

    The router filters by HOST, so anything but a bare host silently dropped
    every doc (bug 50). An LLM writes ``site:example.com`` but also
    ``site:https://example.com/path``, ``site:example.com/r/x``,
    ``site:example.com:8443/deals``, or ``site:example.com,`` -- strip a
    scheme, keep the authority before the first ``/?#``, drop userinfo and a
    port, strip ``www.`` and surrounding punctuation. Returns ``""`` for a
    value that reduces to nothing (so no empty domain is injected).
    """
    raw = raw.strip().strip(".,;:!?()[]{}<>\"'")
    if "//" in raw:
        raw = raw.split("//", 1)[1]
    raw = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    raw = raw.split("@")[-1].split(":", 1)[0]
    # site:*.gov means "any host under .gov" (bug 130): the literal "*.gov"
    # matched nothing in the router's dot-boundary filter. The bare suffix
    # is exactly that filter's semantics.
    return raw.strip().strip(".").lower().removeprefix("www.").removeprefix("*.").lstrip(".")


def _extract_operators(
    text: str,
    domains: list[str] | None,
    exclude_domains: list[str] | None,
) -> tuple[str, list[str], list[str]]:
    """Pull inline ``site:``/``-site:`` out of query text into the structured
    fields, returning ``(cleaned_text, domains, exclude_domains)``.

    Agents write operators inline because that is how search engines speak
    them; left as raw text, honoring them depends on each engine's own
    parser (Bing's RSS honors the site: OR-group only partially) and the
    router's post-filter never engages. Parsed here -- the one entry point
    every caller shares -- the guarantee holds no matter how the query was
    phrased, and the engines still receive the operators in canonical form
    via the structured fields. bench/span.py llm.price_check wrote
    ``site:bestbuy.com`` inline and got an honest-but-useless empty page.
    """
    # Caller-supplied scope is normalized the same way an inline operator is
    # (bug 134): an agent passes domains=["https://www.nasa.gov/"] and the
    # router's host filter matched it literally -- every doc dropped.
    dom: list[str] = []
    exc: list[str] = []
    for raw, target in ((domains or [], dom), (exclude_domains or [], exc)):
        for entry in raw:
            host = _operator_host(str(entry or ""))
            if host and host not in target:
                target.append(host)

    def _pull(m: re.Match) -> str:
        # "site:a.gov,b.int" is two hosts; a host already listed is not
        # listed twice (riders, iteration 60).
        for piece in (m.group(2) or m.group(3) or "").split(","):
            host = _operator_host(piece)
            target = exc if m.group(1) else dom
            if host and host not in target:
                target.append(host)
        return " "

    cleaned = _SITE_OP_RE.sub(_pull, text)
    cleaned = _DANGLING_BOOL_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, dom, exc


def _name_failure(exc: BaseException) -> bool:
    """A failed DNS lookup (any getaddrinfo error), not a refused dial."""
    import socket

    seen = 0
    while exc is not None and seen < 5:
        if isinstance(exc, socket.gaierror):
            return True
        exc = exc.__cause__ or exc.__context__
        seen += 1
    return False


def _markdown_reply(res, plain_ok: bool = False) -> bool:
    """accept_markdown: whether a reply is the markdown the read asked for --
    text/markdown, or (accept_markdown_plain) a text/plain body shaped like
    markdown (a heading or front matter first), as some docs hosts send."""
    ctype = (getattr(res, "content_type", "") or "").lower()
    if "markdown" in ctype:
        return True
    if not plain_ok or "text/plain" not in ctype:
        return False
    head = (getattr(res, "body", "") or "").lstrip()[:4]
    return head.startswith("# ") or head.startswith("---")


def _body_key(body: str) -> tuple:
    return (len(body), hash(body))


def _settle(t: "asyncio.Future") -> None:
    """Done-callback for a background prefetch: a failed speculative fetch
    is the read's problem when (if) it is asked for, never an unretrieved
    exception."""
    if not t.cancelled():
        t.exception()


class Engine:
    """Owns the ladder, the router, and the provider registry for a process.

    Long-lived by design. The domain profiles, circuit breakers, adaptive rate
    limits, and content cache all accumulate value over a session, and building
    a fresh Engine per request throws every bit of that away.
    """

    def __init__(self, settings: Settings | None = None, *, registry: Registry | None = None):
        self.s = settings or get_settings()
        self.ladder = Ladder(self.s)
        self.registry = registry or Registry()
        self.router = Router(self.ladder, settings=self.s, registry=self.registry)
        if self.s.prefetch_early and self.s.prefetch_top_k > 0:
            self.router.on_first_docs = self._early_prefetch

    # ── fast paths ───────────────────────────────────────────────────────────

    async def search(
        self,
        text: str,
        *,
        intent: str = "web",
        k: int = 10,
        domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        freshness: str = "all",
        allow_expensive: bool = False,
        fresh: bool = False,
        locale: str = "",
    ) -> RouteResult:
        """Ranked documents from a fused multi-provider search.

        ``locale`` ("de-DE") is the language/market of the QUESTION: Bing
        honors it as mkt/setlang, so a German question is searched on the
        German web instead of en-US (iteration 56 -- the agent leg answered
        a German heat-pump question from en.wikipedia).

        ``fresh`` bypasses the ladder's cache for every provider fetch."""
        text, domains, exclude_domains = _extract_operators(
            text, domains, exclude_domains
        )
        q = Query(
            text=text,
            intent=intent,  # type: ignore[arg-type]
            k=k,
            domains=domains or [],
            exclude_domains=exclude_domains or [],
            freshness=freshness,  # type: ignore[arg-type]
            **_locale_fields(locale),
        )
        if getattr(getattr(self, "s", None), "prefetch_yield", False):
            # The new search outranks the last one's speculative page
            # downloads: on a slow link they would share its bandwidth.
            for t in list(getattr(self, "_pf_tasks", ())):
                t.cancel()
        res = await self.router.search(q, allow_expensive=allow_expensive, fresh=fresh)
        gate = float(getattr(getattr(self, "s", None), "relevance_gate", 0) or 0)
        if gate > 0:
            res = _relevance_gate(q, res, gate)
        self._prefetch(res)
        self._preconnect(res)
        return res

    def _preconnect(self, res: RouteResult) -> None:
        """preconnect_top_k: open a connection to the hosts of the top results
        the prefetch does not download, by GETting their robots.txt on the
        shared pool (~1 KB; loads the robots entry when it is missing). A read
        that follows skips the TCP+TLS setup -- most of a read's cost on a
        slow, high-latency link -- and this costs too few bytes to skip there."""
        n = int(getattr(getattr(self, "s", None), "preconnect_top_k", 0) or 0)
        if n <= 0:
            return
        from urllib.parse import urlsplit

        from .leanread import lean_target

        skip = int(getattr(self.s, "prefetch_top_k", 0) or 0)
        seen: set[str] = set()
        for i, doc in enumerate(res.docs[:n]):
            url = getattr(doc, "url", "")
            lean = lean_target(url, self.s) if url else None
            if lean is not None:
                url = lean[0][0] if isinstance(lean[0], tuple) else lean[0]
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                continue
            origin = f"{parts.scheme}://{parts.netloc}"
            if origin in seen:
                continue
            seen.add(origin)
            if i < skip:
                continue  # the prefetch already connects there
            t = asyncio.ensure_future(self._warm_origin(origin, url))
            self.router._background.add(t)
            t.add_done_callback(self.router._background.discard)
            t.add_done_callback(_settle)

    def _early_prefetch(self, provider, docs) -> None:
        """prefetch_early: the first provider to answer is usually the one
        whose top hit leads the fused ranking (the generalist); start that
        page now, while the fan-out waits out its deadline. Only a
        generalist's hit: a specialist's top result rarely leads."""
        if provider.name not in self.router.GENERALISTS or not docs:
            return
        from .router import RouteResult

        self._prefetch(RouteResult(docs=list(docs[:1])))

    def _prefetch(self, res: RouteResult) -> None:
        """prefetch_top_k: start fetching the top results' pages in the
        background, so the read an agent issues next for one of them joins
        that fetch (ladder singleflight) or hits the content cache instead
        of starting a cold climb after the search returns."""
        k = int(getattr(getattr(self, "s", None), "prefetch_top_k", 0) or 0)
        if k <= 0:
            return
        floor = float(getattr(self.s, "prefetch_min_kbps", 0) or 0)
        kbps = getattr(self.ladder, "link_kbps", None)
        if floor > 0 and kbps is not None and kbps < floor:
            # A slow link: a speculative page download (often 1 MB) would
            # hold the pipe the agent's next search needs.
            return
        for doc in res.docs[:k]:
            url = getattr(doc, "url", "")
            if not url:
                continue
            from .leanread import lean_target

            lean = lean_target(url, self.s)
            if lean is not None:
                # Warm what the read will actually fetch.
                urls = lean[0] if isinstance(lean[0], tuple) else (lean[0],)
                coro = asyncio.gather(*(self.ladder.fetch(u) for u in urls),
                                      return_exceptions=True)
            elif getattr(self.s, "prefetch_extract", False):
                coro = self._prefetch_doc(url)
            else:
                coro = self.ladder.fetch(url)
            t = asyncio.ensure_future(coro)
            pf = self.__dict__.setdefault("_pf_tasks", set())
            pf.add(t)
            t.add_done_callback(pf.discard)
            if getattr(self.s, "prefetch_extract", False):
                pending = self.__dict__.setdefault("_prefetching", {})
                pending[url] = t
                t.add_done_callback(lambda _t, u=url: pending.pop(u, None))
            self.router._background.add(t)
            t.add_done_callback(self.router._background.discard)
            t.add_done_callback(_settle)

    async def _warm_origin(self, origin: str, url: str) -> None:
        info = await self.ladder.robots.check(url)
        if info.reason != "robots_pending":
            # Rules already known (memory or the robots store): the load that
            # would have connected does not run, so connect with a tiny GET.
            await self.ladder._raw_get(origin + "/robots.txt")

    async def search_many(
        self, texts: list[str], *, intent: str = "web", k: int = 10, **kw: Any,
    ) -> list[RouteResult | Exception]:
        """Several searches at once: each its own fan-out, at most
        ``search_many_concurrency`` in flight (a whole wave at once draws a
        SERP wall). An agent that can only call one tool at a time otherwise
        pays N fan-outs back to back. Results keep the input order; a failed
        search is its exception, never the whole batch's."""
        gate = asyncio.Semaphore(max(1, int(getattr(self.s, "search_many_concurrency", 3) or 1)))

        async def one(text: str):
            async with gate:
                return await self.search(text, intent=intent, k=k, **kw)

        return list(await asyncio.gather(*(one(t) for t in texts), return_exceptions=True))

    async def search_stream(
        self,
        text: str,
        *,
        intent: str = "web",
        k: int = 10,
        domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        freshness: str = "all",
        allow_expensive: bool = False,
        locale: str = "",
    ):
        """:meth:`search`, yielding a fused ranking as each provider lands."""
        text, domains, exclude_domains = _extract_operators(
            text, domains, exclude_domains
        )
        q = Query(
            text=text,
            intent=intent,  # type: ignore[arg-type]
            k=k,
            domains=domains or [],
            exclude_domains=exclude_domains or [],
            freshness=freshness,  # type: ignore[arg-type]
            **_locale_fields(locale),
        )
        async for res in self.router.search_stream(q, allow_expensive=allow_expensive):
            yield res

    async def find_items(
        self, text: str, *, domains: list[str] | None = None, k: int = 10,
        use_browser: bool = False,
    ) -> list[Item]:
        """Normalized, deduplicated marketplace listings for a product query.

        Falls back to the browser-backed extractor only when the cheap path
        finds nothing, so the common case never pays for Chromium.
        """
        from .fuse import merge_items

        # Inline site:/-site: are honored on this entry point too (bug 133):
        # "sony wh-1000xm5 site:bestbuy.com" used to search the operator as
        # words and leave the domain scope empty.
        text, domains, _exc = _extract_operators(text, domains, None)
        q = Query(text=text, intent="shopping", k=k, domains=domains or [])
        ctx = ProviderContext(
            ladder=self.ladder, settings=self.s,
            session_id=self.ladder.session_id, router=self.router,
        )

        items: list[Item] = []
        failures: list[str] = []
        answered = False
        provider = self.registry.get("marketplace") or MarketplaceItems()
        try:
            items = await provider.find_items(q, ctx)  # type: ignore[attr-defined]
            answered = True
        except Exception as exc:  # noqa: BLE001 -- the refusal is kept, see below
            failures.append(f"{type(exc).__name__}: {str(exc)[:200]}")

        if not items and (use_browser or self.s.max_tier >= 2):
            fallback = self.registry.get("sidecar_listings") or SidecarListings()
            try:
                items = await fallback.find_items(q, ctx)  # type: ignore[attr-defined]
                answered = True
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {str(exc)[:200]}")

        # A path that ANSWERED with nothing is an honest empty even when the
        # other path refused (bug 132, bug 120's second edge).
        if not items and failures and not answered:
            # Every path that ran refused and nothing answered: that is the
            # provider's refusal, not "no listings" (bug 120 -- bug 61's
            # "all product pages failed" was swallowed here into [] and the
            # tool, the API and the CLI all reported an honest empty).
            raise ProviderError("marketplace", "; ".join(failures)[:400])
        return merge_items(items)[:k]

    async def find_local_items(
        self, text: str, *, city: str = "", k: int = 20,
    ) -> list[Item]:
        """Classified listings from Facebook Marketplace.

        Kept separate from :meth:`find_items` rather than folded into it,
        because the two answer different questions. That one asks "what does
        this product cost", and merges offers for one SKU across retailers into
        a single row. This one asks "what is for sale near me", where every
        result is a distinct second-hand object that must never merge with
        another -- two "Kayak" listings in one city are two kayaks.

        Browser-only and geolocated, so ``city`` matters: without one Facebook
        answers against the exit IP, which for a hosted container is some
        datacentre's metro rather than the user's.
        """
        provider = self.registry.get("facebook_marketplace") or FacebookMarketplace()
        # Same instance, not a fresh one: the provider keeps per-process state
        # the next call benefits from -- the observed session gate, the
        # session-search-broken flag, and circuit-breaker health. A throwaway
        # here would re-prove the gate on every query. But ``city`` is NOT that
        # state -- it is this query's geo -- so set it UNCONDITIONALLY: the old
        # ``if city:`` guard let an empty-city call inherit the previous call's
        # city and silently search the wrong metro (bug 49). An empty city is
        # the honest "answer against the exit IP" request.
        provider.city = city
        ctx = ProviderContext(
            ladder=self.ladder, settings=self.s,
            session_id=self.ladder.session_id, router=self.router,
        )
        return await provider.find_items(  # type: ignore[attr-defined]
            Query(text=text, intent="shopping", k=k), ctx
        )

    async def fetch(self, url: str, **kw: Any):
        """Retrieve one URL through the acquisition ladder."""
        return await self.ladder.fetch(url, **kw)

    async def read(self, url: str) -> Doc:
        """Fetch a URL and return it as a document with extracted markdown."""
        thin = float(getattr(getattr(self, "s", None), "thin_read_memo_s", 0) or 0)
        if thin <= 0:
            return await self._read_walled(url)
        # thin_read_memo_s: a near-empty read (a JS shell no reachable tier
        # renders) is never page-cached, so each repeat fetched the same
        # shell again; within the TTL a repeat gets a copy of that Doc.
        memo = self.__dict__.setdefault("_thin", {})
        hit = memo.get(url)
        if hit is not None and hit[0] > time.monotonic():
            return hit[1].model_copy()
        doc = await self._read_walled(url)
        if len((doc.text or "").strip()) < 200:
            if len(memo) > 4096:
                memo.clear()
            memo[url] = (time.monotonic() + thin, doc.model_copy())
        return doc

    async def _read_walled(self, url: str) -> Doc:
        ttl = float(getattr(getattr(self, "s", None), "blocked_read_memo_s", 0) or 0)
        if ttl <= 0:
            return await self._read_swr(url)
        # blocked_read_memo_s: a read walled at every tier it could climb
        # fails the same way for a while -- a retry within the TTL gets that
        # answer at once instead of climbing into the same wall again.
        from .errors import Blocked

        memo = self.__dict__.setdefault("_walled", {})
        hit = memo.get(url)
        if hit is not None and hit[0] > time.monotonic():
            raise hit[1]
        try:
            return await self._read_swr(url)
        except Blocked as exc:
            if len(memo) > 4096:
                memo.clear()
            memo[url] = (time.monotonic() + ttl, exc)
            raise

    async def _read_swr(self, url: str) -> Doc:
        swr = float(getattr(getattr(self, "s", None), "page_swr_s", 0) or 0)
        if swr <= 0:
            return await self._read_or_doi(url)
        from .net.ladder import SERP_SWR

        # page_swr_s: an expired cached page answers now (via "cache-stale")
        # and is refetched in the background, as SERPs are under serp_swr_s.
        tok = SERP_SWR.set(swr)
        try:
            return await self._read_or_doi(url)
        finally:
            SERP_SWR.reset(tok)

    async def _read_or_doi(self, url: str) -> Doc:
        """doi_fallback: a read that fails (a publisher's 403 at every tier,
        a JS shell) or comes back near-empty, of a URL whose path names a
        DOI, answers with the paper's Crossref/OpenAlex record (metadata +
        abstract) instead. Pages that read fine never pay for it."""
        if not getattr(getattr(self, "s", None), "doi_fallback", False):
            return await self._read(url)
        from .leanread import doi_in_url

        doi = doi_in_url(url)
        if doi is None and getattr(self.s, "doi_ssrn", False):
            m = re.search(r"^https?://(?:papers|ssrn)\.ssrn\.com/.*[?&](?:abstract_?id)=([0-9]{4,9})", url, re.I)
            doi = f"10.2139/ssrn.{m.group(1)}" if m else None
        if doi is None:
            return await self._read(url)
        # doi_direct_memo: a URL whose page failed (or read near-empty) and
        # whose record answered goes straight to the record for cache_ttl_s
        # -- the wall is not climbed again on every repeat read.
        memo = self.__dict__.setdefault("_doi_direct", {}) if getattr(
            self.s, "doi_direct_memo", False) else None
        if memo is not None and memo.get(url, 0.0) > time.monotonic():
            rec = await self._doi_record(url, doi)
            if rec is not None:
                return rec
            memo.pop(url, None)

        def remember():
            if memo is not None:
                if len(memo) > 4096:
                    memo.clear()
                memo[url] = time.monotonic() + float(getattr(self.s, "cache_ttl_s", 3600) or 0)

        try:
            doc = await self._read(url)
        except Exception as exc:  # noqa: BLE001 -- re-raised when no record
            rec = await self._doi_record(url, doi)
            if rec is None:
                raise exc
            remember()
            return rec
        if len((doc.text or "").strip()) >= 200:
            return doc
        rec = await self._doi_record(url, doi)
        if rec is not None:
            remember()
        return rec if rec is not None else doc

    async def _doi_record(self, url: str, doi: str) -> Doc | None:
        from .leanread import doi_candidates

        for lean_url, build in doi_candidates(doi):
            urls = lean_url if isinstance(lean_url, tuple) else (lean_url,)
            try:
                parts = await asyncio.gather(*(self.ladder.fetch(u) for u in urls))
            except Exception:  # noqa: BLE001 -- the next candidate
                continue
            if any(getattr(r, "status", 200) != 200 for r in parts):
                continue
            built = build(*(r.body or "" for r in parts))
            if built is not None:
                return Doc(url=url, title=built[0], text=built[1], fetched_via=f"{parts[0].via}+doi")
        return None

    async def _read(self, url: str) -> Doc:
        lean = await self._lean_read(url)
        if lean is not None:
            return lean
        inflight = getattr(self, "_prefetching", {}).get(url)
        if inflight is not None:
            # Its extraction is already running: wait for it, not a second one.
            await asyncio.wait([inflight])
        md = bool(getattr(self.s, "accept_markdown", False))
        redir = bool(getattr(self.s, "redirect_memo", False))
        target = self._redirect_target(url) if redir else None
        if target is None and url.startswith("http://") and getattr(self.s, "https_upgrade", False):
            # https_upgrade: nearly every http:// origin answers with a
            # redirect to https -- ask https first; a miss reads as asked.
            target = "https://" + url[len("http://"):]
        res = None
        if target is not None:
            # redirect_memo: this origin moved the path before -- go straight
            # to where it went. Any miss reads the URL as asked, and forgets.
            try:
                res = await self._page_fetch(target, md)
                if getattr(res, "status", 200) != 200:
                    res = None
            except Exception as exc:  # noqa: BLE001 -- the original URL decides
                if (getattr(self.s, "upgrade_dns_final", False)
                        and _name_failure(exc)
                        and urlsplit(target).hostname == urlsplit(url).hostname):
                    # upgrade_dns_final: the name did not resolve -- the URL
                    # as asked has the same name, and its lookup can only
                    # fail the same way (a dead zone cost 2.8 s + 1.5 s).
                    raise
                res = None
            if res is None:
                self._redirect_forget(url)
                target = None
        fetched = target or url
        if res is None:
            res = await self._page_fetch(url, md)
        final = res.final_url or fetched
        if redir:
            self._redirect_learn(fetched, final)
        if md and _markdown_reply(res, getattr(self.s, "accept_markdown_plain", False)):
            from .leanread import markdown_build

            built = markdown_build(final)(res.body or "")
            if built is not None:
                return Doc(url=final, title=built[0], text=built[1],
                           fetched_via=f"{res.via}+md")
            # A stub ("Continue to <link>", a near-empty page): the HTML
            # read decides, never an extraction of the stub.
            res = await self._read_fetch(fetched)
            final = res.final_url or fetched
            if redir:
                self._redirect_learn(fetched, final)
        if getattr(self.s, "lean_md_alternate", False) or getattr(self.s, "lean_edit_link", False):
            self._learn_alternate(final, res.body, alias=url)
        memo = getattr(self, "_read_memo", None)
        if memo:
            hit = memo.get(final)
            if hit is not None and hit[0] == _body_key(res.body):
                doc = hit[1].model_copy()
                doc.fetched_via = res.via
                return doc
        doc = await self._extract(final, res.body)
        if getattr(self.s, "youtube_watch_details", False) and len(doc.text or "") < 1000:
            from .leanread import youtube_watch_details

            yt = youtube_watch_details(final, res.body or "")
            if yt is not None:
                doc.title, doc.text = yt
        if getattr(self.s, "read_memo", False):
            self._memo_put(final, res.body, doc)
        doc.fetched_via = res.via
        return doc

    async def _page_fetch(self, url: str, md: bool):
        budget = int(getattr(self.s, "pdf_read_budget", 0) or 0)
        if budget > 0:
            from .net.ladder import PDF_CHAR_BUDGET

            btok = PDF_CHAR_BUDGET.set(budget)
            try:
                return await self._page_fetch_md(url, md)
            finally:
                PDF_CHAR_BUDGET.reset(btok)
        return await self._page_fetch_md(url, md)

    async def _page_fetch_md(self, url: str, md: bool):
        if not md:
            return await self._read_fetch(url)
        from .net.ladder import ACCEPT_MD

        tok = ACCEPT_MD.set(True)
        try:
            return await self._read_fetch(url)
        finally:
            ACCEPT_MD.reset(tok)

    # ── redirect_memo ────────────────────────────────────────────────────────
    #: rules kept per origin (newest first)
    REDIRECT_RULES_MAX = 8

    def _redirects(self) -> dict:
        store = getattr(self, "_redir_map", None)
        if store is None:
            path = Path(self.s.state_dir) / "redirects.json"
            try:
                store = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(store, dict):
                    store = {}
            except (OSError, ValueError):
                store = {}
            self._redir_map = store
        return store

    def _redirects_save(self) -> None:
        path = Path(self.s.state_dir) / "redirects.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._redirects()), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    def _redirect_target(self, url: str) -> str | None:
        """Where this URL went last time its origin redirected this path
        prefix (chains followed, at most three hops)."""
        store = self._redirects()
        if not store:
            return None
        once = bool(getattr(self.s, "redirect_host_once", False))
        cur = url
        for _ in range(3):
            p = urlsplit(cur)
            origin = f"{p.scheme}://{p.netloc}".lower()
            rules = store.get(origin) or []
            for old, new, seen in sorted(rules, key=lambda r: -len(r[0])):
                if not p.path.startswith(old):
                    continue
                if seen < 2 and not (once and old == "/" and urlsplit(new).path == "/"):
                    continue  # one read is an anecdote (a single page's rename);
                    # a whole-host move (apex -> www, golang.org -> go.dev) is not
                if new.lower().startswith(origin + "/") and p.path.startswith(new[len(origin):]):
                    continue  # an inserted prefix (/en-US/) the path already has
                cur = new + p.path[len(old):] + (f"?{p.query}" if p.query else "")
                break
            else:
                break
        return cur if cur != url else None

    def _redirect_learn(self, url: str, final: str) -> None:
        """A redirect that kept the path's tail (and the query) is a prefix
        move: /dev/peps/pep-8/ -> peps.python.org/pep-8/. Anything else (a
        login wall, a homepage bounce) teaches nothing."""
        if not final or final == url:
            return
        a, b = urlsplit(url), urlsplit(final)
        if a.query != b.query or b.scheme not in ("http", "https") or not b.netloc:
            return
        if a.path.endswith("/") != b.path.endswith("/"):
            return  # a slash normalisation, not a move
        sa, sb = a.path.rstrip("/").split("/"), b.path.rstrip("/").split("/")
        n = 0
        while n < min(len(sa), len(sb)) and sa[-1 - n] == sb[-1 - n] and sa[-1 - n]:
            n += 1
        if n == 0:
            return
        old = "/".join(sa[:-n]) + "/"
        new = f"{b.scheme}://{b.netloc}" + "/".join(sb[:-n]) + "/"
        key = f"{a.scheme}://{a.netloc}".lower()
        if f"{key}{old}" == new.lower():
            return
        rules = self._redirects().get(key, [])
        same = [r for r in rules if r[0] == old and r[1] == new]
        seen = (same[0][2] + 1) if same else 1  # a different target restarts the count
        rules = [r for r in rules if r[0] != old]
        self._redirects()[key] = ([[old, new, seen]] + rules)[:self.REDIRECT_RULES_MAX]
        self._redirects_save()

    def _redirect_forget(self, url: str) -> None:
        p = urlsplit(url)
        key = f"{p.scheme}://{p.netloc}".lower()
        rules = self._redirects().get(key) or []
        keep = [r for r in rules if not p.path.startswith(r[0])]
        if len(keep) != len(rules):
            self._redirects()[key] = keep
            self._redirects_save()

    def _memo_put(self, url: str, body: str, doc: Doc) -> None:
        memo = getattr(self, "_read_memo", None)
        if memo is None:
            memo = self._read_memo = OrderedDict()
        memo[url] = (_body_key(body), doc.model_copy())
        memo.move_to_end(url)
        while len(memo) > self.READ_MEMO_MAX:
            memo.popitem(last=False)

    async def read_many(self, urls: list[str]) -> list[Doc | Exception]:
        """Several reads at once, at most ``read_many_concurrency`` in flight.
        An agent that can only call one tool at a time otherwise pays N page
        downloads back to back. Input order kept; a failed read is its
        exception, never the whole batch's."""
        gate = asyncio.Semaphore(max(1, int(getattr(self.s, "read_many_concurrency", 4) or 1)))

        async def one(url: str):
            async with gate:
                return await self.read(url)

        return list(await asyncio.gather(*(one(u) for u in urls), return_exceptions=True))

    async def _lean_read(self, url: str) -> Doc | None:
        """lean_read_*: read a page through its site's lean representation
        (searchio.leanread). Any miss -- no rewrite, a failed fetch, a body
        the rewrite does not recognise -- is None: the normal read runs."""
        from .leanread import lean_targets

        targets = lean_targets(url, getattr(self, "s", None))
        s = getattr(self, "s", None)
        if not targets and (getattr(s, "lean_md_alternate", False)
                            or getattr(s, "lean_edit_link", False)):
            targets = self._alt_targets(url)
        if not targets:
            return None
        # lean_miss_memo: a URL whose every candidate answered definitively
        # without a document (a 404 README.md, an arXiv paper with no HTML)
        # skips the lean attempt for cache_ttl_s -- otherwise each repeat
        # read pays that dead round trip before its cached page.
        memo = self.__dict__.setdefault("_lean_miss", {}) if getattr(
            s, "lean_miss_memo", False) else None
        if memo is not None and memo.get(url, 0.0) > time.monotonic():
            return None
        definitive = True
        for lean_url, build in targets:
            # A tuple is several parts of one document, fetched together.
            urls = lean_url if isinstance(lean_url, tuple) else (lean_url,)
            try:
                parts = await asyncio.gather(*(self.ladder.fetch(u) for u in urls))
            except Exception as exc:  # noqa: BLE001 -- the next candidate, then the page
                if not _LEAN_GONE.search(str(exc)):
                    definitive = False  # transient: try the lean form next time
                continue
            if any(getattr(r, "status", 200) != 200 for r in parts):
                continue
            res = parts[0]
            built = build(*(r.body or "" for r in parts))
            if built is None:
                continue
            title, text = built
            return Doc(url=url, title=title, text=text, fetched_via=f"{res.via}+lean")
        if memo is not None and definitive:
            if len(memo) > 4096:
                memo.clear()
            memo[url] = time.monotonic() + float(getattr(s, "cache_ttl_s", 3600) or 0)
        return None

    def _alt_store(self):
        store = self.__dict__.get("_alts")
        if store is None:
            from .leanread import AltStore

            store = self._alts = AltStore(self.s.state_dir / "md_alternates.json")
        return store

    def _alt_targets(self, url: str) -> list:
        from urllib.parse import urlsplit

        from .leanread import apply_edit_rule, apply_template, markdown_build, _source_build

        host = urlsplit(url).netloc.lower()
        rule = self._alt_store().get(host)
        name = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
        if isinstance(rule, str) and getattr(self.s, "lean_md_alternate", False):
            lean = apply_template(rule, url)
            return [(lean, markdown_build(name))] if lean else []
        if isinstance(rule, dict) and getattr(self.s, "lean_edit_link", False):
            build = _source_build(name)
            return [(u, build) for u in apply_edit_rule(rule, url)]
        return []

    def _learn_alternate(self, url: str, body: str, alias: str = "") -> None:
        """lean_md_alternate / lean_edit_link: remember the host's lean rule
        -- under the requested host too when a redirect kept the path
        (vitejs.dev -> vite.dev), or later reads there would never use it."""
        from urllib.parse import urlsplit

        hosts = [urlsplit(url).netloc.lower()]
        a, f = urlsplit(alias or url), urlsplit(url)
        if a.netloc.lower() != hosts[0] and a.path.rstrip("/") == f.path.rstrip("/"):
            hosts.append(a.netloc.lower())
        for host in hosts:
            self._learn_for(host, url, body)

    def _learn_for(self, host: str, url: str, body: str) -> None:
        from .leanread import alternate_template, markdown_alternate

        if getattr(self.s, "lean_md_alternate", False):
            alt = markdown_alternate(url, body or "")
            template = alternate_template(url, alt) if alt else None
            if template:
                self._alt_store().put(host, template)
                return
        if getattr(self.s, "lean_edit_link", False) and not isinstance(
                self._alt_store().get(host), str):
            from .leanread import edit_rule

            rule = edit_rule(url, body or "")
            if rule:
                self._alt_store().put(host, rule)

    async def _read_fetch(self, url: str):
        """read_slow_kbps: on a measured-slow link, keep the first
        read_soft_cap_kb of the page (never cached) instead of all of it."""
        slow = float(getattr(self.s, "read_slow_kbps", 0) or 0)
        kbps = getattr(self.ladder, "link_kbps", None)
        if slow <= 0 or kbps is None or kbps >= slow:
            return await self.ladder.fetch(url)
        from .net.ladder import BODY_SOFT_CAP

        tok = BODY_SOFT_CAP.set((int(self.s.read_soft_cap_kb) * 1024, []))
        try:
            return await self.ladder.fetch(url)
        finally:
            BODY_SOFT_CAP.reset(tok)

    async def _extract(self, url: str, body: str) -> Doc:
        """doc_from_fetch, on a worker thread for a page of at least
        read_offloop_bytes: extraction is pure CPU (up to ~30 ms for a 1 MB
        page) and on the loop it stalls every concurrent fan-out."""
        from .extract import doc_from_fetch

        cut = int(getattr(self.s, "read_offloop_bytes", 0) or 0)
        if cut > 0 and len(body) >= cut:
            return await asyncio.get_running_loop().run_in_executor(
                None, doc_from_fetch, url, body)
        return doc_from_fetch(url, body)

    #: prefetch_extract: Docs kept (newest last).
    READ_MEMO_MAX = 64

    async def _prefetch_doc(self, url: str) -> None:
        """prefetch_extract: fetch, then extract off the loop and keep the
        Doc keyed by the body it came from, so :meth:`read` skips the work
        and a changed page (a cache refresh) is never answered stale."""
        from .extract import doc_from_fetch

        if getattr(self.s, "accept_markdown", False):
            # Warm the variant the read will ask for; a markdown reply needs
            # no extraction (the read builds it), so stop at the fetch.
            from .net.ladder import ACCEPT_MD

            tok = ACCEPT_MD.set(True)
            try:
                res = await self.ladder.fetch(url)
            finally:
                ACCEPT_MD.reset(tok)
            if _markdown_reply(res, getattr(self.s, "accept_markdown_plain", False)):
                return
        else:
            res = await self.ladder.fetch(url)
        final = res.final_url or url
        doc = await asyncio.get_running_loop().run_in_executor(
            None, doc_from_fetch, final, res.body)
        self._memo_put(final, res.body, doc)

    async def transcript(self, target: str, *, lang: str = "en") -> Transcript:
        """Caption transcript of one YouTube video.

        ``target`` is a watch / youtu.be / shorts / embed / live URL or a bare
        11-char video id; ``lang`` is the preferred caption language. This
        rides youtube-transcript-api, not the ladder: captions are a
        PO-token-gated handshake the library maintains (see
        :mod:`searchio.transcripts`), and no fetch tier can substitute for
        it. A missing track is a typed TranscriptError, never an honest
        empty.
        """
        from .transcripts import fetch_transcript

        wanted = lang.strip() or "en"
        return await fetch_transcript(target, languages=[wanted])

    # ── the swarm ────────────────────────────────────────────────────────────

    async def research(self, question: str, *, progress=None) -> ResearchResult:
        """Run the research swarm. Requires an Anthropic API key."""
        from .swarm.llm import make_client
        from .swarm.orchestrator import Orchestrator

        client, model = make_client(self.s)
        try:
            return await Orchestrator(self, client, model=model, progress=progress).run(question)
        finally:
            close = getattr(client, "close", None)
            if close:
                await close()

    # ── lifecycle ────────────────────────────────────────────────────────────

    #: Hosts the keyless search providers query, warmed by :meth:`warmup`.
    WARM_ORIGINS = (
        "https://html.duckduckgo.com", "https://lite.duckduckgo.com",
        "https://www.bing.com", "https://en.wikipedia.org",
        "https://hn.algolia.com", "https://api.github.com",
        "https://api.stackexchange.com", "https://api.openalex.org",
        "https://api.crossref.org", "https://export.arxiv.org",
    )

    async def warmup(self) -> None:
        """Warm the ladder for the providers' hosts (see Ladder.warm).

        For long-lived processes: the MCP and HTTP servers call it at startup
        (``warmup_on_start``), so the first search runs as fast as the
        hundredth. A CLI run gains nothing from awaiting it inline.
        """
        await self.ladder.warm(list(self.WARM_ORIGINS))

    def stats(self) -> dict:
        return {
            "ladder": self.ladder.stats(),
            "providers": self.router.health_report(),
        }

    async def close(self) -> None:
        # Detached stragglers (fanout_grace_ms) would otherwise run on into a
        # closed ladder and record a failure that is only shutdown.
        for t in list(self.router._background):
            t.cancel()
        if self.router._health is not None:
            self.router._health.close()
        if getattr(self.router, "_holds", None) is not None:
            self.router._holds.close()
        if getattr(self.router, "_qstore", None) is not None:
            self.router._qstore.close()
        await self.ladder.close()

    async def __aenter__(self) -> "Engine":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
