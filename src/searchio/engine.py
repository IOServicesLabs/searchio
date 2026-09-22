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

import re
from typing import Any

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
        return await self.router.search(q, allow_expensive=allow_expensive, fresh=fresh)

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
        from .extract import doc_from_fetch

        res = await self.ladder.fetch(url)
        doc = doc_from_fetch(res.final_url or url, res.body)
        doc.fetched_via = res.via
        return doc

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

    def stats(self) -> dict:
        return {
            "ladder": self.ladder.stats(),
            "providers": self.router.health_report(),
        }

    async def close(self) -> None:
        await self.ladder.close()

    async def __aenter__(self) -> "Engine":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
