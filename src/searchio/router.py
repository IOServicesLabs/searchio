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
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .config import Settings, get_settings
from .fuse import fuse
from .models import Doc, Query
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


class _NoCacheLadder:
    """The ladder with ``use_cache`` forced off for every fetch (fresh=True)."""

    def __init__(self, ladder):
        self._ladder = ladder

    async def fetch(self, url, **kw):
        kw["use_cache"] = False
        return await self._ladder.fetch(url, **kw)

    def __getattr__(self, name):
        return getattr(self._ladder, name)


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
    elapsed_ms: int = 0

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

    # ── selection ────────────────────────────────────────────────────────────

    #: General web providers, kept out of the specialist-only trap below.
    GENERALISTS = ("duckduckgo", "bing")

    def select(
        self, q: Query, *, fanout: int | None = None, allow_expensive: bool = False
    ) -> list[Provider]:
        """Rank and take the top providers for this query.

        ``allow_expensive`` gates the browser-backed providers. They are off by
        default because a routine lookup should never cost a Chromium launch;
        the swarm turns them on when the cheap providers have already failed.
        """
        n = fanout or self.s.fanout
        candidates = self.registry.available(self.s, q.intent)
        if not allow_expensive:
            candidates = [p for p in candidates if p.cost_per_1k <= 0]
        aff = AFFINITY.get(q.intent, {})

        def score(p: Provider) -> float:
            # Health dominates: a provider that is failing is worth nothing
            # regardless of how well suited it would otherwise be.
            base = p.health.score * aff.get(p.name, 1.0)
            if q.freshness != "all":
                base *= FRESH_AFFINITY.get(p.name, 1.0)
            # Cost is a mild penalty, enough to keep the browser-backed
            # providers out of routine queries without banning them.
            return base / (1.0 + p.cost_per_1k / 5.0)

        ranked = sorted(candidates, key=score, reverse=True)
        chosen = ranked[:n]

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
        return chosen

    # ── fan-out ──────────────────────────────────────────────────────────────

    async def search(
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
        results = await asyncio.gather(
            *(self._call(p, q, ctx) for p in providers), return_exceptions=False
        )

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
        return RouteResult(
            docs=docs,
            used=list(lists),
            failed=failed,
            per_provider=counts,
            filtered=filtered,
            filtered_stale=filtered_stale,
            skipped=skipped,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

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
            left = max(0, int(h.COOLDOWN_S - (time.monotonic() - h.opened_at)))
            out[p.name] = (f"circuit open: {h.consecutive_failures} consecutive failures, "
                           f"last {h.last_error!r}, cooling for {left}s")
        return out

    async def _call(
        self, p: Provider, q: Query, ctx: ProviderContext
    ) -> tuple[Provider, list[Doc], str]:
        """Run one provider under a timeout, recording health either way."""
        t0 = time.monotonic()
        try:
            docs = await asyncio.wait_for(p.search(q, ctx), timeout=self.s.provider_timeout_s)
        except asyncio.TimeoutError:
            p.health.record(False, self.s.provider_timeout_s * 1000, "timeout")
            return p, [], "timeout"
        except Exception as exc:
            ms = (time.monotonic() - t0) * 1000
            msg = f"{type(exc).__name__}: {exc}"
            p.health.record(False, ms, msg)
            return p, [], msg[:200]
        ms = (time.monotonic() - t0) * 1000
        # An empty list is not a failure -- the provider answered, the answer
        # was "nothing". Counting it as a failure would open breakers on
        # specialists that simply have no coverage for a given topic.
        p.health.record(True, ms)
        return p, docs, ""

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
