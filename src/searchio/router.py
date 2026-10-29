"""Choosing which providers answer a query, and merging what they say.

Three jobs:

1. **Select.** Match intent to capability, then rank candidates by health,
   cost, and a per-intent affinity -- OpenAlex should win an academic query
   outright, and lose a shopping one outright.
2. **Fan out.** Query the chosen providers concurrently, with a per-provider
   timeout, and never let one slow or broken provider hold up the answer.
3. **Fuse.** Hand the per-provider lists to :mod:`searchio.fuse` for
   deduplication and reciprocal rank fusion.

The important property is that a provider failing is *normal*, not
exceptional. Engines get blocked, APIs rate-limit, the sidecar is not always
running. So failures are recorded against a circuit breaker and the fan-out
returns whatever came back, rather than propagating one adapter's bad day into
a failed search.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

from .config import Settings, get_settings
from .errors import RateLimited, Throttled
from .fuse import fuse, strip_trackers
from .models import Capability, Doc, Query
from .providers.base import Provider, ProviderContext
from .providers.registry import Registry


def _host_of(url: str) -> str:
    # urlsplit(url).hostname raises ValueError on a malformed URL (e.g.
    # "http://[::1" -> "Invalid IPv6 URL"). A provider can return such a URL,
    # and the domain post-filter runs this over every doc: an unguarded raise
    # crashed the WHOLE fused query on one bad URL (bug 42). An unparseable
    # host is simply "" -- not in any allow-list (so a garbage-URL doc drops
    # out of a site: query) and not in any deny-list (so -site: keeps it).
    try:
        return urlsplit(url).hostname or ""
    except ValueError:
        return ""


def _host_allowed(url: str, domains: list[str]) -> bool:
    """Dot-boundary suffix match: ``www.epa.gov`` is inside ``epa.gov``.

    Case-folded on both sides: urlsplit lowercases the host, but a domain
    passed structurally (the tool's ``domains=`` param) is not (only inline
    ``site:`` is, in engine._extract_operators), so a mixed-case
    ``domains=["EPA.gov"]`` silently matched nothing -- a site-scoped query
    returned empty (bug 43).
    """
    host = _host_of(url)
    return any(host == (dl := d.lower()) or host.endswith("." + dl) for d in domains)


#: Freshness bounds in days, each +1 of slack for timezone edge dates.
_FRESH_DAYS = {"day": 2, "week": 8, "month": 32, "year": 367}


def _drop_stale(docs: list[Doc], freshness: str) -> tuple[list[Doc], int]:
    """Remove provably-stale docs for a freshness-bound query.

    Only dated docs outside the bound are dropped -- an undated doc cannot be
    proven stale, and dropping it would gut providers that never publish dates
    (Wikipedia). DuckDuckGo's endpoints silently ignore their own date filter
    (verified live: byte-identical SERP with and without df=w) and HN's
    freshness handling is loose, so the bound is enforced HERE, after the
    fan-out, same as the domain operators. bench/span.py tool.freshness_week
    caught a 2025-03 doc in a week-bound query without it.
    """
    import datetime as dt

    cutoff = dt.date.today() - dt.timedelta(days=_FRESH_DAYS[freshness])
    kept: list[Doc] = []
    dropped = 0
    for d in docs:
        try:
            day = dt.date.fromisoformat(str(d.published)[:10])
        except (ValueError, TypeError):
            kept.append(d)
            continue
        if day < cutoff:
            dropped += 1
        else:
            kept.append(d)
    return kept, dropped

#: How much a provider is trusted for a given intent, multiplying its RRF
#: contribution and its selection score. Absent means 1.0.
AFFINITY: dict[str, dict[str, float]] = {
    "academic": {"openalex": 2.0, "arxiv": 1.8, "crossref": 1.6, "wikipedia": 1.2,
                 "duckduckgo": 0.8, "bing": 0.6},
    "code": {"github": 2.0, "stackexchange": 1.8, "hackernews": 1.2, "duckduckgo": 1.0},
    "reference": {"wikipedia": 2.2, "duckduckgo": 1.0, "bing": 0.8},
    "shopping": {"marketplace": 2.0, "sidecar_listings": 1.4, "duckduckgo": 1.0, "bing": 0.7},
    "forum": {"hackernews": 1.8, "stackexchange": 1.6, "duckduckgo": 1.0},
    "news": {"duckduckgo": 1.2, "bing": 1.0, "hackernews": 1.0},
    "video": {"youtube": 2.2, "duckduckgo": 1.0, "bing": 0.8},
    "local": {"sidecar_listings": 1.6, "duckduckgo": 1.0},
    "web": {"duckduckgo": 1.4, "bing": 1.0, "wikipedia": 0.9},
}

#: Extra trust for providers that can actually honor a freshness bound, when
#: the query carries one. DuckDuckGo's no-JS endpoints ignore the date filter
#: (verified live: byte-identical SERP with and without df=w), so treating
#: every provider alike on a freshness query silently returns stale results.
FRESH_AFFINITY = {"bing": 1.4, "hackernews": 1.6, "youtube": 1.4, "duckduckgo": 0.7}

#: commerce_skip_reference: a query that is shopping for something. Strict on
#: purpose -- "used" or "best" alone are as often about software as goods.
_COMMERCE = re.compile(
    r"(?i)\b(?:for sale|to buy|buy(?:ing)? (?:a|an|used|new)\b|cheapest|deals on|deals for|discount|coupon"
    r"|near me|in stock|msrp|best price|prices? (?:of|for|on)\b|under \$?\d[\d,]*k?)\b|\$\d")


class _NoCacheLadder:
    """The ladder with ``use_cache`` forced off for every fetch (fresh=True)."""

    def __init__(self, ladder):
        self._ladder = ladder

    async def fetch(self, url, **kw):
        kw["use_cache"] = False
        return await self._ladder.fetch(url, **kw)

    def __getattr__(self, name):
        return getattr(self._ladder, name)


class _Called(tuple):
    """A ``(provider, docs, err)`` result that also carries the call's wall
    time, so every ``for p, docs, err in results`` keeps working."""

    ms: int

    def __new__(cls, result: tuple, elapsed_s: float) -> "_Called":
        obj = super().__new__(cls, result)
        obj.ms = int(elapsed_s * 1000)
        return obj


@dataclass
class RouteResult:
    """One fan-out: the fused ranking plus everything that went wrong."""

    docs: list[Doc]
    used: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    per_provider: dict[str, int] = field(default_factory=dict)
    #: Docs dropped by the domain post-filter, per provider. Without this a
    #: provider that answered entirely off-domain looks exactly like one that
    #: errored or had nothing -- three very different stories for the caller
    #: (and for the span suite's weather-vs-defect bookkeeping).
    filtered: dict[str, int] = field(default_factory=dict)
    #: Docs dropped by the freshness post-filter (provably stale: dated older
    #: than the bound). Same story as ``filtered`` -- an engine that ignored
    #: df=w must not read as an outage.
    filtered_stale: dict[str, int] = field(default_factory=dict)
    #: Providers that serve this intent but sat circuit-open and were not
    #: asked (bug 164). Without this an all-cooling fan-out answered "no
    #: provider supports intent web" -- they do -- and a partly cooling
    #: one answered [] with failed={} : an HTTP caller could not tell a
    #: cooldown from an empty web. Not a failure (nothing was tried), so
    #: it lives beside ``failed``, not in it.
    skipped: dict[str, str] = field(default_factory=dict)
    #: Providers still running when the fan-out returned early
    #: (``fanout_grace_ms``). Not failures: they finish detached, and their
    #: health and cache entries still land for the next query.
    late: list[str] = field(default_factory=list)
    #: Wall time of each provider call that finished inside this fan-out
    #: (ms): which source was slow, not just how slow the search was.
    provider_ms: dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    #: relevance_gate: docs dropped as off-topic, per provider -- so a gated
    #: empty answer reads as "nothing on topic", not as an outage.
    off_topic: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.docs)


class Router:
    """Selects, fans out to, and fuses providers."""

    def __init__(
        self,
        ladder,
        *,
        settings: Settings | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.s = settings or get_settings()
        self.ladder = ladder
        self.registry = registry or Registry()
        #: Detached straggler calls (fanout_grace_ms). Held so the loop does
        #: not garbage-collect a running task; each removes itself on exit.
        self._background: set[asyncio.Task] = set()
        #: singleflight: one in-flight fan-out per identical query.
        self._inflight: dict[tuple, asyncio.Future] = {}
        #: query_cache_ttl_s: key -> (expires_at, RouteResult).
        self._qcache: OrderedDict[tuple, tuple[float, RouteResult]] = OrderedDict()
        #: persist_health: breakers and EMAs carried across processes.
        self._health = None
        if self.s.persist_health:
            from .net.healthstore import HealthStore

            self._health = HealthStore(self.s.state_dir / "health.sqlite")
            self._health.load(self.registry.all())
        #: query_cache_swr_s: keys with a background refresh in flight.
        self._refreshing: set[tuple] = set()
        self._late_tasks: list = []
        #: prefetch_early: called with (provider, docs) on a fan-out's first answer.
        self.on_first_docs = None
        #: qcache_k_superset: the k each query-cache entry was fused for.
        self._qk: dict[tuple, int] = {}
        #: link_down_guard: recent network-level failures, last success time.
        self._link_errs: list[tuple[float, str]] = []
        self._last_ok = 0.0
        #: persist_query_cache: fused results shared across processes.
        self._qstore = None
        if self.s.persist_query_cache and self.s.query_cache_ttl_s > 0:
            from .net.qcachestore import QueryCacheStore

            self._qstore = QueryCacheStore(self.s.state_dir / "qcache.sqlite")
        #: persist_holds: stated holds (a quota's reset) shared across processes.
        self._holds = None
        if self.s.persist_holds:
            from .net.healthstore import HoldStore

            self._holds = HoldStore(self.s.state_dir / "holds.sqlite")
            self._holds.load(self.registry.all())
        #: breaker_backoff_s rides on each provider's Health (it has no Settings).
        for p in self.registry.all():
            p.health.backoff_base_s = float(getattr(self.s, "breaker_backoff_s", 0.0) or 0.0)

    # ── selection ────────────────────────────────────────────────────────────

    #: General web providers, kept out of the specialist-only trap below.
    GENERALISTS = ("duckduckgo", "bing", "marginalia")

    def _ranked(self, q: Query, allow_expensive: bool) -> list[Provider]:
        """Every available provider for ``q``, best first."""
        candidates = self.registry.available(self.s, q.intent)
        if self.s.halfopen_single:
            candidates = [p for p in candidates if not p.health.probing]
        if not allow_expensive:
            candidates = [p for p in candidates if p.cost_per_1k <= 0]
        if (self.s.commerce_skip_reference and q.intent == "web"
                and _COMMERCE.search(q.text)):
            # An encyclopedia does not list things for sale: on a buying
            # query its slot only ever returned the maker's history page.
            candidates = [p for p in candidates if Capability.REFERENCE not in p.caps]
        aff = AFFINITY.get(q.intent, {})

        def score(p: Provider) -> float:
            # Health dominates: a provider that is failing is worth nothing
            # regardless of how well suited it would otherwise be.
            h = p.health
            w = self.s.latency_weight
            if w == 0.2 or h.open:
                health = h.score  # the stock 0.8 success / 0.2 latency blend
            else:
                health = (1 - w) * h.ema_success + w / (1.0 + h.ema_latency_ms / 2000.0)
            base = health * aff.get(p.name, 1.0)
            if q.freshness != "all":
                base *= FRESH_AFFINITY.get(p.name, 1.0)
            # Cost is a mild penalty, enough to keep the browser-backed
            # providers out of routine queries without banning them.
            return base / (1.0 + p.cost_per_1k / 5.0)

        return sorted(candidates, key=score, reverse=True)

    def select(
        self, q: Query, *, fanout: int | None = None, allow_expensive: bool = False
    ) -> list[Provider]:
        """Rank and take the top providers for this query.

        ``allow_expensive`` gates the browser-backed providers. They are off by
        default because a routine lookup should never cost a Chromium launch;
        the swarm turns them on when the cheap providers have already failed.
        """
        n = fanout or self.s.fanout
        chosen = self._ranked(q, allow_expensive)[:n]

        # Always keep at least one general web provider in the mix. A purely
        # specialist fan-out answers the query it recognised rather than the one
        # that was asked: OpenAlex and Crossref index papers, so an academic
        # query about a brand-new technique finds nothing there while a plain
        # web search finds the blog post that introduced it. The generalists are
        # pulled from the whole registry, not from the intent-filtered list,
        # precisely because they do not declare the specialist capability.
        if q.intent != "shopping" and not any(p.name in self.GENERALISTS for p in chosen):
            for name in self.GENERALISTS:
                p = self.registry.get(name)
                if p is not None and p.configured(self.s) and not p.health.open:
                    # max(n-1, 0), not max(n-1, 1): at fanout=1 the latter kept
                    # one specialist AND appended the generalist, returning two
                    # providers for a one-provider budget (bug 44). Zero floor
                    # means fanout=1 yields just the generalist.
                    chosen = chosen[: max(n - 1, 0)] + [p]
                    break
            else:
                probe = self._starved_probe()
                if probe is not None:
                    chosen = chosen[: max(n - 1, 0)] + [probe]
        return chosen

    def _starved_probe(self) -> Provider | None:
        """starved_probe_s: every generalist is cooling, so this fan-out has
        no general web engine at all -- a product query then comes back
        empty from Wikipedia alone. Let the generalist that has cooled
        longest through early as a half-open probe, at most once per
        ``starved_probe_s`` per provider. Its own failure re-trips it."""
        every = self.s.starved_probe_s
        if every <= 0:
            return None
        now = time.monotonic()
        best = None
        for name in self.GENERALISTS:
            p = self.registry.get(name)
            if p is None or not p.configured(self.s):
                continue
            h = p.health
            if now - h.opened_at < every or now - h.last_probe < every:
                continue
            if best is None or h.opened_at < best.health.opened_at:
                best = p
        if best is not None:
            best.health.last_probe = now
        return best

    # ── fan-out ──────────────────────────────────────────────────────────────

    #: Bound on the in-memory query cache (query_cache_ttl_s).
    QCACHE_MAX = 256

    def _query_key(self, q: Query, fanout: int | None, allow_expensive: bool) -> tuple:
        from .net.ladder import _FETCH_SESSION

        # The tenant is part of the key: a tenant-scoped fetch may read
        # private pages (bug 138's shape), so its fused list never answers
        # another tenant.
        if self.s.normalize_query_keys:
            # The engines fold case and whitespace themselves, so these are
            # the same question; domain lists are sets, not sequences.
            q = q.model_copy(update={
                "text": " ".join(q.text.lower().split()),
                "domains": sorted(d.lower() for d in q.domains),
                "exclude_domains": sorted(d.lower() for d in q.exclude_domains),
            })
        if self.s.qcache_k_superset:
            # k is checked on the hit instead (_superset_ok): a k=5 ask is
            # answered by a cached k=10 fan-out's first five.
            q = q.model_copy(update={"k": 0})
        return (q.model_dump_json(), fanout, allow_expensive, _FETCH_SESSION.get())

    @staticmethod
    def _copy(res: RouteResult) -> RouteResult:
        """A result a second caller can mutate without touching the first's."""
        return replace(
            res, docs=[d.model_copy(deep=True) for d in res.docs], used=list(res.used),
            failed=dict(res.failed), per_provider=dict(res.per_provider),
            filtered=dict(res.filtered), filtered_stale=dict(res.filtered_stale),
            skipped=dict(res.skipped), late=list(res.late),
            provider_ms=dict(res.provider_ms))

    async def search(
        self, q: Query, *, fanout: int | None = None, allow_expensive: bool = False,
        fresh: bool = False,
    ) -> RouteResult:
        """Fan out, fuse, return -- through the query cache and singleflight
        when they are on. ``fresh`` bypasses both: the caller asked for a
        new answer, not a shared or remembered one."""
        ttl = self.s.query_cache_ttl_s
        if fresh or not (ttl > 0 or self.s.singleflight):
            return await self._search(q, fanout=fanout, allow_expensive=allow_expensive,
                                      fresh=fresh)
        key = self._query_key(q, fanout, allow_expensive)
        if ttl > 0:
            hit = self._qcache.get(key)
            if hit is None and self._qstore is not None:
                hit = self._qstore.get(key)
                if hit is not None:
                    self._qcache[key] = hit
            now = time.monotonic()
            if hit is not None and not self._superset_ok(key, q):
                hit = None
            if hit is not None and hit[0] > now:
                self._qcache.move_to_end(key)
                out = self._copy(hit[1])
                out.docs = out.docs[:q.k]
                out.elapsed_ms = 0
                return out
            swr = self.s.query_cache_swr_s
            if hit is not None and swr > 0 and hit[0] + swr > now:
                # Expired but inside the SWR window: answer now, refresh once
                # in the background (a later caller finds the new entry).
                self._qcache.move_to_end(key)
                if key not in self._refreshing:
                    self._refreshing.add(key)
                    t = asyncio.ensure_future(self._refresh(key, q, fanout, allow_expensive))
                    self._background.add(t)
                    t.add_done_callback(self._background.discard)
                out = self._copy(hit[1])
                out.docs = out.docs[:q.k]
                out.elapsed_ms = 0
                return out
        if self.s.singleflight:
            while (fut := self._inflight.get(key)) is not None:
                # wait(), not shield(): a cancelled LEADER must not cancel
                # its waiters -- one of them runs the fan-out instead.
                await asyncio.wait({fut})
                if not fut.cancelled():
                    return self._copy(fut.result())
            fut = asyncio.get_running_loop().create_future()
            self._inflight[key] = fut
        try:
            res = await self._search(q, fanout=fanout, allow_expensive=allow_expensive)
        except asyncio.CancelledError:
            if self.s.singleflight:
                fut.cancel()
            raise
        except BaseException as exc:
            if self.s.singleflight and not fut.done():
                fut.set_exception(exc)
                fut.exception()  # mark retrieved: a waiter-less failure is not "never retrieved"
            raise
        finally:
            if self.s.singleflight:
                self._inflight.pop(key, None)
        if not res.docs and ttl > 0 and self.s.offline_stale_s > 0:
            stale = self._stale_answer(key, res)
            if stale is not None:
                res = stale
        if self.s.singleflight and not fut.done():
            fut.set_result(res)
        # Only an answer is remembered: an empty or all-failed fan-out is
        # weather, and caching it would pin the outage for the TTL.
        if ttl > 0 and res.docs and not getattr(res, "_stale", False):
            self._remember(key, res, q.k)
            late = getattr(res, "_late", None)
            if late is not None:
                self._merge_when_late(key, q, *late)
        return res

    def _stale_answer(self, key: tuple, res: RouteResult) -> RouteResult | None:
        """offline_stale_s: the fan-out came back empty (the link dropped,
        every provider failed): an expired answer to the same query, up to
        this old, beats nothing. ``failed`` keeps the real failures and says
        the answer is stale, so a caller can tell."""
        hit = self._qcache.get(key)
        if hit is None and self._qstore is not None:
            hit = self._qstore.get(key)
        if hit is None:
            return None
        age = time.monotonic() - (hit[0] - self.s.query_cache_ttl_s)
        if age > self.s.offline_stale_s + self.s.query_cache_ttl_s:
            return None
        out = self._copy(hit[1])
        out.failed = {**res.failed,
                      "router": f"no provider answered; served the cached answer from {age:.0f}s ago"}
        out.elapsed_ms = res.elapsed_ms
        out._stale = True
        return out

    def _superset_ok(self, key: tuple, q: Query) -> bool:
        """qcache_k_superset: the entry was fused for at least q.k results."""
        return not self.s.qcache_k_superset or self._qk.get(key, 0) >= q.k

    def _remember(self, key: tuple, res: RouteResult, k: int = 0) -> None:
        if self.s.qcache_k_superset:
            if k < self._qk.get(key, 0) and key in self._qcache:
                return  # never replace a wider answer with a narrower one
            self._qk[key] = k
        entry = (time.monotonic() + self.s.query_cache_ttl_s, self._copy(res))
        self._qcache[key] = entry
        self._qcache.move_to_end(key)
        while len(self._qcache) > self.QCACHE_MAX:
            self._qcache.popitem(last=False)
        if self._qstore is not None:
            self._qstore.put(key, entry)

    async def _refresh(self, key: tuple, q: Query, fanout, allow_expensive: bool) -> None:
        """query_cache_swr_s: re-run a query served stale. An empty refresh
        keeps the old entry: weather is not an answer."""
        try:
            res = await self._search(q, fanout=fanout, allow_expensive=allow_expensive)
            if res.docs:
                self._remember(key, res, q.k)
        finally:
            self._refreshing.discard(key)

    def _merge_when_late(self, key: tuple, q: Query, late_tasks, results, skipped,
                         started) -> None:
        """late_merge: as each straggler the deadline left behind lands,
        re-fuse it into the cached entry, so a repeat of the query gets the
        whole fan-out instead of the first 1.5 s of it."""
        results = list(results)
        still = {p.name for p, _t in late_tasks}

        def landed(p: Provider, t: asyncio.Future) -> None:
            still.discard(p.name)
            if t.cancelled() or t.exception() is not None:
                return
            r = t.result()
            results.append(r)
            if not r[1] or r[2] or key not in self._qcache:
                return  # nothing to add, or the entry is gone meanwhile
            merged = self._assemble(q, results, sorted(still), skipped, started)
            merged.elapsed_ms = 0
            self._remember(key, merged, q.k)

        for p, t in late_tasks:
            t.add_done_callback(lambda t, p=p: landed(p, t))

    async def _gather(self, providers: list[Provider], q: Query, ctx: ProviderContext,
                      spares: list[Provider] = ()):
        """Run every provider call; return ``(results, late)``.

        With ``fanout_grace_ms`` off this is a plain gather. On, the fan-out
        returns once ``fanout_quorum`` providers have answered with docs and
        the grace window after that has elapsed (or everyone is done). The
        stragglers are not cancelled -- they finish detached, so their health
        is recorded and their SERP lands in the ladder cache for next time.
        """
        providers = list(providers)
        spares = list(spares)
        tasks = [asyncio.ensure_future(self._call(p, q, ctx)) for p in providers]
        grace = self.s.fanout_grace_ms / 1000.0
        backup = self.s.backup_provider_ms / 1000.0
        early = grace > 0 and len(tasks) >= 2
        soft = self.s.fanout_deadline_ms / 1000.0 if len(tasks) >= 2 else 0.0
        if not early and soft <= 0 and not (backup > 0 and spares):
            return list(await asyncio.gather(*tasks)), []
        quorum = max(1, min(self.s.fanout_quorum, len(tasks)))
        loop = asyncio.get_running_loop()
        pending = set(tasks)
        answered = 0
        #: Raw docs in hand (deadline_min_docs counts these, pre-dedupe).
        docs_in = 0
        min_docs = self.s.deadline_min_docs
        deadline: float | None = None
        #: backup_provider_ms: when this passes with fewer than ``quorum``
        #: answers in, the next-ranked provider outside the fan-out starts.
        backup_at = loop.time() + backup if backup > 0 and spares else None
        #: fanout_deadline_ms: past this, any answer in hand is the answer.
        soft_at = loop.time() + soft if soft > 0 else None
        #: deadline_min_docs: past soft_at, a thin answer keeps waiting to here.
        hard_at = (soft_at + max(0.0, self.s.fanout_deadline_max_ms / 1000.0 - soft)
                   if soft_at is not None and min_docs > 0 else None)
        while pending:
            now = loop.time()
            if soft_at is not None and answered and now >= soft_at:
                if not (hard_at is not None and docs_in < min_docs and now < hard_at):
                    break
            # A mark already passed is no reason to wake: left in, it made
            # the wait time out at 0 and spin the loop until an answer came.
            marks = [m for m in (deadline, backup_at, soft_at) if m is not None and m > now]
            if hard_at is not None and soft_at is not None and soft_at <= now < hard_at:
                marks.append(hard_at)
            timeout = max(0.0, min(marks) - now) if marks else None
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                _p, docs, err = t.result()
                if docs and not err:
                    answered += 1
                    docs_in += len(docs)
                    if answered == 1 and self.on_first_docs is not None:
                        # prefetch_early: the engine starts the likely read
                        # now, not after the fan-out's deadline.
                        try:
                            self.on_first_docs(_p, docs)
                        except Exception:  # noqa: BLE001 -- a hint, never a failure
                            pass
            if early and deadline is None and answered >= quorum:
                deadline = loop.time() + grace
            if backup_at is not None and loop.time() >= backup_at:
                backup_at = None
                if answered < quorum:
                    p = spares.pop(0)
                    providers.append(p)
                    t = asyncio.ensure_future(self._call(p, q, ctx))
                    tasks.append(t)
                    pending.add(t)
            if deadline is not None and loop.time() >= deadline:
                break
            if (self.s.predictive_cut and soft_at is not None and answered
                    and docs_in >= q.k and self._all_due_late(providers, tasks, pending,
                                                              soft_at - soft, soft_at)):
                break
        late: list[str] = []
        results = []
        late_tasks = []
        for p, t in zip(providers, tasks):
            if t.done():
                results.append(t.result())
            else:
                late.append(p.name)
                if self.s.cancel_stragglers and not self.s.late_merge:
                    t.cancel()
                    continue
                late_tasks.append((p, t))
                self._background.add(t)
                t.add_done_callback(self._background.discard)
        #: Read by _search right after this returns (no await in between).
        self._late_tasks = late_tasks
        return results, late

    @staticmethod
    def _all_due_late(providers, tasks, pending, began: float, soft_at: float) -> bool:
        """predictive_cut: every provider still out has a latency history
        that lands it after the soft deadline. Waiting would only reach the
        deadline and return what is already in hand -- so return it now."""
        for p, t in zip(providers, tasks):
            if t not in pending:
                continue
            h = p.health
            if h.calls < 3 or began + h.ema_latency_ms / 1000.0 < soft_at:
                return False
        return True

    async def drain(self) -> None:
        """Wait for detached straggler calls (tests, orderly shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def _search(
        self, q: Query, *, fanout: int | None = None, allow_expensive: bool = False,
        fresh: bool = False,
    ) -> RouteResult:
        started = time.monotonic()
        providers = self.select(q, fanout=fanout, allow_expensive=allow_expensive)
        skipped = self._cooling(q.intent)
        if not providers:
            # Say what actually happened (bug 165): "no provider supports
            # intent local" went out while three providers support it --
            # they are browser-backed and this call did not allow them, and
            # the general-web backstop that serves such intents was cooling.
            gated = [] if allow_expensive else [
                p.name for p in self.registry.all()
                if p.configured(self.s) and p.supports(q.intent)
                and p.cost_per_1k > 0 and not p.health.open]
            failed: dict[str, str] = {}
            if gated:
                failed["router"] = (
                    f"intent {q.intent!r} is served by browser-backed providers "
                    f"({', '.join(gated)}) this call did not allow (allow_expensive)"
                    + (f"; the general-web backstop is cooling down: {', '.join(skipped)}"
                       if skipped else ""))
            elif not skipped:
                failed["router"] = f"no provider supports intent {q.intent}"
            return RouteResult(docs=[], failed=failed, skipped=skipped,
                               elapsed_ms=int((time.monotonic() - started) * 1000))

        ctx = ProviderContext(
            # fresh=True (bug 76): every provider fetch bypasses the ladder's
            # SERP cache; the API promised "bypass the cache" and nothing
            # honored it -- an hour-old SERP came back with a straight face.
            ladder=_NoCacheLadder(self.ladder) if fresh else self.ladder,
            settings=self.s,
            session_id=getattr(self.ladder, "session_id", ""), router=self,
        )
        spares = ([p for p in self._ranked(q, allow_expensive) if p not in providers]
                  if self.s.backup_provider_ms > 0 else [])
        results, late = await self._gather(providers, q, ctx, spares)
        res = self._assemble(q, results, late, skipped, started)
        if self.s.search_wall_rescue and not allow_expensive and self._web_walled(res):
            # Every general web engine in the fan-out came back empty-handed
            # (walled, degraded, timed out): what is left is reference filler.
            # One browser-backed provider gets the query before giving up.
            rescue = [p for p in self._ranked(q, True)
                      if p.cost_per_1k > 0 and p not in providers][:1]
            if rescue:
                more, _late = await self._gather(rescue, q, ctx)
                results = list(results) + list(more)
                res = self._assemble(q, results, late, skipped, started)
        if late and self.s.late_merge:
            res._late = (list(self._late_tasks), list(results), skipped, started)
        return res

    async def search_stream(
        self, q: Query, *, fanout: int | None = None, allow_expensive: bool = False,
    ):
        """Yield a fused RouteResult each time another provider answers.

        Progressive results (#18): the first yield arrives with the FASTEST
        provider's answer instead of the slowest's, so a caller that can act
        on partial results (an agent reading the top hit, a UI) starts early.
        Every yield is a complete fused ranking of what is in so far; the
        last one equals what :meth:`search` would have returned. ``late``
        names the providers still running.
        """
        started = time.monotonic()
        providers = self.select(q, fanout=fanout, allow_expensive=allow_expensive)
        if not providers:
            yield await self._search(q, fanout=fanout, allow_expensive=allow_expensive)
            return
        skipped = self._cooling(q.intent)
        ctx = ProviderContext(ladder=self.ladder, settings=self.s,
                              session_id=getattr(self.ladder, "session_id", ""), router=self)
        tasks = {asyncio.ensure_future(self._call(p, q, ctx)): p for p in providers}
        results: list = []
        try:
            for fut in asyncio.as_completed(list(tasks)):
                r = await fut
                results.append(r)
                last = len(results) == len(tasks)
                if r[1] or last:
                    late = [p.name for t, p in tasks.items() if not t.done()]
                    yield self._assemble(q, results, late, skipped, started)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    def _assemble(self, q: Query, results, late: list[str], skipped: dict[str, str],
                  started: float) -> RouteResult:
        """Post-filter and fuse provider results into one RouteResult."""
        lists: dict[str, list[Doc]] = {}
        failed: dict[str, str] = {}
        counts: dict[str, int] = {}
        filtered: dict[str, int] = {}
        filtered_stale: dict[str, int] = {}
        for provider, docs, err in results:
            if err:
                failed[provider.name] = err
                continue
            if docs:
                # The domain operators are best-effort hints at the engines
                # (Bing's RSS feed only partially honors the site: OR-group,
                # and the API providers -- Wikipedia et al. -- never see them),
                # so the allow/deny lists are enforced HERE, where the
                # guarantee actually holds. bench/span.py web.gov_domain
                # caught samsclub.com and sciencedirect.com in an
                # epa.gov-only query without it.
                before = len(docs)
                if q.domains:
                    docs = [d for d in docs if _host_allowed(d.url, q.domains)]
                if q.exclude_domains:
                    docs = [d for d in docs
                            if not _host_allowed(d.url, q.exclude_domains)]
                if len(docs) < before:
                    filtered[provider.name] = before - len(docs)
            if docs and q.freshness != "all":
                docs, n_stale = _drop_stale(docs, q.freshness)
                if n_stale:
                    filtered_stale[provider.name] = n_stale
            if docs:
                lists[provider.name] = docs
                counts[provider.name] = len(docs)

        weights = {}
        for name in lists:
            w = AFFINITY.get(q.intent, {}).get(name, 1.0)
            if q.freshness != "all":
                w *= FRESH_AFFINITY.get(name, 1.0)
            weights[name] = w
        docs = fuse(lists, weights=weights, limit=q.k)
        if self.s.strip_result_trackers:
            for d in docs:
                d.url = strip_trackers(d.url)
        return RouteResult(
            docs=docs,
            used=list(lists),
            failed=failed,
            per_provider=counts,
            filtered=filtered,
            filtered_stale=filtered_stale,
            skipped=skipped,
            late=late,
            provider_ms={r[0].name: r.ms for r in results if hasattr(r, "ms")},
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def _web_walled(self, res: RouteResult) -> bool:
        """search_wall_rescue: a general web engine was asked and none answered."""
        asked = [n for n in self.GENERALISTS
                 if n in res.failed or n in res.late or n in res.skipped]
        return bool(asked) and not any(n in res.used for n in self.GENERALISTS)

    def _cooling(self, intent: str) -> dict[str, str]:
        """The configured providers for ``intent`` whose breaker is open.

        The generalists count for every intent but shopping: ``select``
        keeps one in every such fan-out as the backstop, so a cooling
        generalist is a provider this query lost.
        """
        out: dict[str, str] = {}
        for p in self.registry.all():
            h = p.health
            serves = p.supports(intent) or (intent != "shopping" and p.name in self.GENERALISTS)
            if not (p.configured(self.s) and serves and h.open):
                continue
            left = max(0, int(h.cooldown_s - (time.monotonic() - h.opened_at)))
            out[p.name] = (f"circuit open: {h.consecutive_failures} consecutive failures, "
                           f"last {h.last_error!r}, cooling for {left}s")
        return out

    #: adaptive_timeout: history needed before a provider's own latency
    #: sets its timeout, and the bounds the derived value is held to.
    ADAPTIVE_MIN_CALLS = 3
    ADAPTIVE_FLOOR_S = 2.0

    def _timeout_for(self, p: Provider) -> float:
        cap = self.s.provider_timeout_s
        if not self.s.adaptive_timeout or p.health.calls < self.ADAPTIVE_MIN_CALLS:
            return cap
        derived = 3.0 * p.health.ema_latency_ms / 1000.0 + 1.0
        return min(cap, max(self.ADAPTIVE_FLOOR_S, derived))

    async def _call(
        self, p: Provider, q: Query, ctx: ProviderContext
    ) -> tuple[Provider, list[Doc], str]:
        """Run one provider under a timeout, recording health either way."""
        t0 = time.monotonic()
        probe = self.s.halfopen_single and p.health.half_open
        if probe:
            p.health.probing = True
        try:
            return _Called(await self._call_once(p, q, ctx), time.monotonic() - t0)
        finally:
            if probe:
                p.health.probing = False
            if self._health is not None:
                self._health.save(p)

    async def _call_once(
        self, p: Provider, q: Query, ctx: ProviderContext
    ) -> tuple[Provider, list[Doc], str]:
        t0 = time.monotonic()
        timeout = self._timeout_for(p)
        from .net.ladder import MAX_LIMITER_WAIT, SERP_SWR

        # Set in this task only: wait_for runs p.search in a child task that
        # copies the context, and nothing outside the provider call sees it.
        tok = MAX_LIMITER_WAIT.set(self.s.serp_max_wait_s) if self.s.serp_max_wait_s > 0 else None
        # A freshness-bound query asked for recent results: never a stale SERP.
        swr_tok = (SERP_SWR.set(self.s.serp_swr_s)
                   if self.s.serp_swr_s > 0 and q.freshness == "all" else None)
        trips = p.health.trips
        try:
            docs = await asyncio.wait_for(p.search(self._provider_query(p, q), ctx),
                                          timeout=timeout)
        except asyncio.TimeoutError:
            if self._link_down(p, "timeout"):
                return p, [], "timeout (link down)"
            p.health.record(False, timeout * 1000, "timeout")
            self._transient_trip(p, trips)
            return p, [], "timeout"
        except Exception as exc:
            ms = (time.monotonic() - t0) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            if self._link_down(p, msg):
                return p, [], (msg + " (link down)")[:200]
            p.health.record(False, ms, msg)
            if isinstance(exc, Throttled):
                self._transient_trip(p, trips)
            if self.s.ratelimit_aware and isinstance(exc, RateLimited):
                # The host said when: hold exactly that long, no guessing,
                # no probes into a quota that has not reset.
                p.health.hold(exc.retry_after)
                if self._holds is not None and exc.retry_after >= 60:
                    self._holds.put(p.name, p.health.hold_s)
            elif self.s.degraded_hold_s > 0 and "degraded_results" in msg:
                p.health.hold(self.s.degraded_hold_s)
            elif self.s.fast_trip_on_block and getattr(exc, "walled", False):
                # Every endpoint drew a wall: the egress is refused, and the
                # next two searches would each spend seconds (the limiter's
                # post-block wait) earning the same answer before the
                # breaker's third-strike rule caught up.
                p.health.trip()
            return p, [], msg[:200]
        finally:
            if tok is not None:
                MAX_LIMITER_WAIT.reset(tok)
            if swr_tok is not None:
                SERP_SWR.reset(swr_tok)
        ms = (time.monotonic() - t0) * 1000
        # An empty list is not a failure -- the provider answered, the answer
        # was "nothing". Counting it as a failure would open breakers on
        # specialists that simply have no coverage for a given topic.
        if self._holds is not None and p.health.opened_at:
            self._holds.clear(p.name)  # a half-open probe answered: the hold is over
        self._last_ok = time.monotonic()
        p.health.record(True, ms)
        return p, docs, ""

    #: link_down_guard: network-level failure text (connect, DNS, timeouts).
    _LINK_ERR = re.compile(
        r"Connect(Error|Timeout)|gaierror|getaddrinfo|unreachable|NetworkError|"
        r"ReadTimeout|PoolTimeout|RemoteProtocolError|^timeout", re.I)
    #: Failures this close together are one event.
    LINK_WINDOW_S = 5.0

    def _link_down(self, p: Provider, msg: str) -> bool:
        """link_down_guard: is this failure the LINK rather than ``p``?

        Yes when it is network-level and another provider failed the same way
        within LINK_WINDOW_S with no provider answering since. Then nothing is
        recorded: on a dropped connection every provider fails at once, and
        counting it opened every breaker -- the fan-out came back from the
        outage with its whole provider set cooling for minutes."""
        if not self.s.link_down_guard or not self._LINK_ERR.search(msg):
            return False
        now = time.monotonic()
        recent = [(t, name) for t, name in self._link_errs if now - t < self.LINK_WINDOW_S]
        recent.append((now, p.name))
        self._link_errs = recent[-32:]
        others = {name for t, name in recent if name != p.name and t > self._last_ok}
        if not others:
            return False
        p.health.last_error = f"link down: {msg}"[:200]
        return True

    def _provider_query(self, p: Provider, q: Query) -> Query:
        """specialist_k_ratio: a specialist is asked for fewer results than k.
        Fusion keeps k across every provider, and a specialist's tail rarely
        survives it -- on a slow link those rows are pure transfer time
        (arXiv ships ~2.3 KB of abstract per result, uncompressed)."""
        ratio = self.s.specialist_k_ratio
        if ratio <= 0 or ratio >= 1 or p.name in self.GENERALISTS:
            return q
        return q.model_copy(update={"k": max(3, math.ceil(q.k * ratio))})

    def _transient_trip(self, p: Provider, trips_before: int) -> None:
        """transient_hold_s: a breaker opened by timeouts or by our own
        limiter's Throttled (nothing was even sent) says the provider was
        slow or we were busy -- not that it refuses us. Hold it briefly
        instead of the wall's full cooldown."""
        if self.s.transient_hold_s > 0 and p.health.trips > trips_before:
            p.health.hold_s = self.s.transient_hold_s

    # ── introspection ────────────────────────────────────────────────────────

    def health_report(self) -> list[dict]:
        rows = []
        for p in self.registry.all():
            h = p.health
            rows.append(
                {
                    "provider": p.name,
                    "configured": p.configured(self.s),
                    "capabilities": sorted(c.value for c in p.caps),
                    "calls": h.calls,
                    "failures": h.failures,
                    "success_rate": round(h.ema_success, 3),
                    "latency_ms": round(h.ema_latency_ms),
                    "breaker": "open" if h.open else "closed",
                    "last_error": h.last_error,
                }
            )
        return rows
