"""Finding the same product across marketplaces.

The approach is deliberately indirect. Querying each marketplace's own search
UI is the obvious design and the wrong one: every one of them has a bespoke
result layout, a bespoke pagination scheme, and the most aggressive bot
protection on the site, and all three change without notice. Maintaining eight
such adapters is most of the work of a shopping engine and none of the value.

So instead: use a general web search to *find* candidate product pages, then
fetch each one and read its schema.org JSON-LD. Product detail pages carry that
markup because Google Shopping requires it, sites keep it accurate for the same
reason, and it is a stable published schema rather than a layout to reverse
engineer. One extraction path serves every marketplace.

Where that is not enough -- a page that renders its price only after
JavaScript, or a site that walls off plain HTTP -- the SwarmIO sidecar's
``extract_listings`` handles it, using generic DOM-repetition analysis with no
per-site selectors.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import urlsplit

from ..errors import Blocked, ProviderError
from ..extract import identity_for, items_from_html, parse_price, price_from_meta
from ..models import Capability, Doc, Item, Query
from .base import Provider, ProviderContext

#: Marketplaces worth scoping a shopping query to. Ordered roughly by breadth
#: of catalogue, which is also the order in which they tend to have the item.
MARKETPLACES: tuple[str, ...] = (
    "amazon.com",
    "walmart.com",
    "bestbuy.com",
    "target.com",
    "ebay.com",
    "newegg.com",
    "homedepot.com",
    "lowes.com",
    "bhphotovideo.com",
    "costco.com",
    # Manufacturer direct-sales sites: often the authoritative price and
    # always a real offer, unlike an encyclopedia article about the brand.
    "logitech.com",
    "sony.com",
    "apple.com",
    "samsung.com",
    "dell.com",
)

#: Paths that are catalogue navigation rather than a specific product.
_NON_PRODUCT_HINTS = (
    "/s?", "/search", "/b/", "/c/", "/browse", "/deals", "/gp/bestsellers",
    "/category", "/categories", "/collections", "/shop/all", "/brands",
    # Editorial and account paths (bug 119): a 5-digit run in /blog/12345-...
    # or /news/2024/12345 made an article a "product page" to fetch.
    "/blog", "/news", "/help", "/support", "/article", "/wiki", "/forum",
    "/community", "/question", "/review", "/compare", "/guide", "/about",
    "/press", "/career", "/policy", "/terms", "/privacy", "/cart", "/account",
    "/login", "/checkout", "/wishlist", "/track",
)

#: Markers that mean "one product" on their own.
_STRONG_MARKERS = (
    "/dp/", "/gp/product/", "/clp/",   # Amazon
    "/ip/",                            # Walmart
    "/itm/",                           # eBay
    "/-/a-",                           # Target
    "/product/", "/products/",         # generic / Shopify
)

#: Markers that a retailer uses for products *and* departments. Best Buy serves
#: both /site/sony-wh1000xm5/6505727.p and /site/logitech/computer-accessories,
#: so these only count alongside a product id.
_WEAK_MARKERS = ("/p/", "/site/", "/pd/", "/buy/", "/shop/")

#: A product id in the path: SKUs and item numbers are long numeric runs. This
#: is what separates a Best Buy product from a Best Buy department.
_PATH_ID = re.compile(r"\d{5,}")


def _domain_allowed(domain: str, allowed) -> bool:
    """Dot-boundary, case-folded host match -- the router's _host_allowed rule.

    The provider used to test ``a in d.domain`` (a SUBSTRING), so
    domains=["amazon.com"] admitted amazon.com.mx / .br / .au -- regional
    stores in other currencies -- and merge_items then picked the cheapest
    offer ACROSS currencies (bug 60). ``www.amazon.com`` is inside
    ``amazon.com``; ``amazon.com.au`` and ``notamazon.com`` are not.
    """
    host = (domain or "").lower().split(":", 1)[0]
    for a in allowed or ():
        a = (a or "").lower().strip().removeprefix("www.")
        if a and (host == a or host.endswith("." + a)):
            return True
    return False


def is_product_url(url: str) -> bool:
    """Whether a URL is plausibly one product's detail page.

    Length alone is not enough. "Products - Logitech" and "Logitech: Computer
    Accessories - Best Buy" are long-pathed category pages that carry no offer,
    and admitting them means fetching a page that cannot answer, then reporting
    a priceless "listing" that is really a department.
    """
    low = url.lower()
    if any(h in low for h in _NON_PRODUCT_HINTS):
        return False
    path = urlsplit(low).path
    if len(path) <= 8:
        return False
    if any(m in path for m in _STRONG_MARKERS):
        return True
    if any(m in path for m in _WEAK_MARKERS):
        return bool(_PATH_ID.search(path))
    return bool(_PATH_ID.search(path))


#: items_vehicle_listings: where vehicles are sold. A car is a classified
#: listing on a results page, not a product page with one SKU, and none of
#: MARKETPLACES carries one -- "2022 ford f-150 fx4" answered with two eBay
#: decals after 68 s.
VEHICLE_SITES: tuple[str, ...] = (
    "autotrader.com", "cars.com", "cargurus.com", "carmax.com", "carvana.com",
    "edmunds.com", "truecar.com", "kbb.com", "carfax.com",
)
_MAKES = frozenset((
    "acura", "alfa", "audi", "bmw", "buick", "cadillac", "chevrolet", "chevy", "chrysler", "dodge",
    "fiat", "ford", "gmc", "honda", "hyundai", "infiniti", "jaguar", "jeep", "kia",
    "lexus", "lincoln", "maserati", "mazda", "mercedes", "mercedes-benz",
    "mitsubishi", "nissan", "porsche", "rivian", "subaru", "tesla", "toyota",
    "volkswagen", "vw", "volvo",
))
_VEHICLE_YEAR = re.compile(r"\b(?:19[89]\d|20[0-4]\d)\b")
#: A part or accessory for a vehicle is a retail product, not a vehicle.
_NOT_VEHICLE = re.compile(
    r"\b(?:decals?|stickers?|emblems?|badges?|parts?|accessor\w+|covers?|mats?|liners?|kits?|wheels?|rims?|"
    r"tires?|grilles?|bumpers?|lights?|lamps?|filters?|manuals?|keys?|fobs?|toys?|diecast|"
    r"die-cast|models? kit|lego|shirts?|hats?|posters?|brakes?|pads?|rotors?|battery|"
    r"batteries|wipers?|mirrors?|hitch|tonneau|exhaust|intake|shocks?|struts?)\b", re.I)
_FOR_SALE = re.compile(r"\b(?:for sale|used|pre-?owned|certified|dealer|near|miles?)\b", re.I)
_QUERY_STOP = frozenset(("for", "sale", "used", "new", "pre-owned", "preowned", "certified", "near", "me",
                         "the", "a", "an", "with", "and", "or", "in", "best", "deal", "deals", "price",
                         "cheap", "cheapest", "package", "pkg", "truck", "car", "suv", "within", "of"))


def is_vehicle_query(text: str) -> bool:
    """A query for a vehicle itself: a make, with a model year or for-sale
    wording, and no part / accessory word."""
    low = text.lower()
    words = set(re.findall(r"[a-z][a-z-]+", low))
    if not words & _MAKES or _NOT_VEHICLE.search(low):
        return False
    return bool(_VEHICLE_YEAR.search(low) or _FOR_SALE.search(low))


def vehicle_urls(text: str) -> list[str]:
    """Results pages for a vehicle query on sites whose URL shape is its
    make / model / year (each shape seen answering live, 2026-11-25). Used
    when web search finds no vehicle page: discovery was the single point
    of failure of the shopping path whenever the search engines refused."""
    low = text.lower()
    words = re.findall(r"[a-z0-9][a-z0-9.-]*", low)
    make = next((w for w in words if w in _MAKES), "")
    if not make:
        return []
    make = {"chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz"}.get(make, make)
    rest = [w for w in words[words.index(next(w for w in words if w in _MAKES)) + 1:]
            if w not in _QUERY_STOP and not _VEHICLE_YEAR.fullmatch(w)]
    if not rest:
        return []
    model, extra = rest[0].strip(".-"), " ".join(rest[1:])
    flat = re.sub(r"[^a-z0-9]", "", model)
    ym = _VEHICLE_YEAR.search(low)
    year = ym.group(0) if ym else ""
    kw = ("?keywordPhrases=" + extra.replace(" ", "+")) if extra else ""
    if year:
        return [
            f"https://www.autotrader.com/cars-for-sale/{year}/{make}/{flat}{kw}",
            f"https://www.kbb.com/cars-for-sale/used/{year}/{make}/{flat}{kw}",
            f"https://www.carmax.com/cars/{make}/{model}?year={year}-{year}",
            f"https://www.carvana.com/cars/{make}-{flat}",
        ]
    return [
        f"https://www.autotrader.com/cars-for-sale/{make}/{flat}{kw}",
        f"https://www.carmax.com/cars/{make}/{model}",
        f"https://www.carvana.com/cars/{make}-{flat}",
    ]


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def vehicle_items(html: str, page: str, query: str = "") -> list[Item]:
    """Vehicle listings on a results page as Items, from the same rows a
    read prints (extract.with_listings). With ``query``, only cars whose row
    states every query term other than the year and filler words (a package
    like "fx4" included) are kept; cars of the asked model year win, and
    only when the page has none are other years returned (marked in meta)."""
    from ..extract import with_listings

    text = with_listings("", html, page, packages=True)
    at = text.find("## Listings")
    if at < 0:
        return []
    low = query.lower()
    years = set(_VEHICLE_YEAR.findall(low))
    terms = [_squash(w) for w in re.findall(r"[a-z0-9][a-z0-9.-]*", low)
             if w not in _QUERY_STOP and not _VEHICLE_YEAR.fullmatch(w)]
    terms = [t for t in terms if len(t) >= 2]
    domain = urlsplit(page).netloc.lower().removeprefix("www.")
    out: list[Item] = []
    other_year: list[Item] = []
    for line in text[at:].splitlines():
        if not line.startswith("- "):
            continue
        parts = line[2:].split(" | ")
        name = parts[0]
        attrs: dict[str, str] = {}
        price, seller, link = "", "", ""
        for p in parts[1:]:
            if p.startswith("$") and not price:
                price = p
            elif p.endswith(" mi") and p[:1].isdigit():
                attrs["mileage"] = p
            elif p.startswith("VIN "):
                attrs["vin"] = p[4:]
            elif p.startswith("distance "):
                attrs["distance_mi"] = p[9:]
            elif p.startswith("pkgs: "):
                attrs["packages"] = p[6:]
            elif p.startswith("history: "):
                attrs["history"] = p[9:]
            elif p.startswith("http"):
                link = p
            elif "(" in p and p.endswith(")"):
                seller, _, where = p[:-1].partition("(")
                seller = seller.strip()
                attrs["location"] = where.strip()
            else:
                attrs.setdefault("details", "")
                attrs["details"] = (attrs["details"] + "; " + p).strip("; ")
        if not attrs.get("vin") or not price:
            # No VIN: not one car. No price: nothing to compare.
            continue
        flat = _squash(line)
        if any(t not in flat for t in terms):
            continue
        year_ok = not years or any(y in name for y in years)
        (out if year_ok else other_year).append(Item(
            title=name, url=link or f"{page}#{attrs['vin']}", price=parse_price(price),
            seller=seller or domain, condition="used" if name.lower().startswith("used") else "",
            identity=f"vin:{attrs['vin']}", attributes=attrs, source=domain,
            meta={"extraction": "listing_row", "page": page, **({} if year_ok else {"year_match": False})},
        ))
    # The asked model year first; a page with none of that year still answers
    # with the cars that match everything else (each row states its year).
    return out or other_year


class MarketplaceItems(Provider):
    """Shopping results assembled from web search plus JSON-LD extraction."""

    name = "marketplace"
    caps = frozenset({Capability.SHOPPING})

    def __init__(self, domains: tuple[str, ...] = MARKETPLACES, *, max_pages: int = 8) -> None:
        super().__init__()
        self.domains = domains
        self.max_pages = max_pages

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        """Provider interface: return the candidate product pages as documents.

        :meth:`find_items` is the interesting entry point; this exists so the
        router can treat shopping like any other intent.
        """
        items = await self.find_items(q, ctx)
        return [
            Doc(
                url=it.url,
                title=it.title,
                snippet=f"{it.price} · {it.seller or it.domain}",
                source=self.name,
                rank=i,
                meta={"item": it.model_dump()},
            )
            for i, it in enumerate(items)
        ]

    async def find_items(self, q: Query, ctx: ProviderContext) -> list[Item]:
        """Search, fetch candidate product pages in parallel, extract items."""
        if (getattr(ctx.settings, "items_vehicle_listings", False) and not q.domains
                and self.domains is MARKETPLACES and is_vehicle_query(q.text)):
            # A miss (no page, no matching car, every page refused) falls
            # through to the retail path: the query may not be a vehicle.
            try:
                found = await self._vehicles(q, ctx)
            except Exception:  # noqa: BLE001
                found = []
            if found:
                return found
        docs = await self._discover(q, ctx)
        # Engines honour `site:` loosely at best, and a generalist fan-out will
        # happily return the Wikipedia article about the manufacturer. Only
        # pages on an actual retailer can carry an offer, so anything else is
        # a wasted fetch that would surface as a priceless "listing".
        allowed = tuple(q.domains or self.domains)
        candidates = [
            d for d in docs if is_product_url(d.url) and _domain_allowed(d.domain, allowed)
        ][: self.max_pages]
        if not candidates:
            return []

        results = await asyncio.gather(
            *(self._one(d, ctx, allowed=allowed) for d in candidates), return_exceptions=True
        )
        # A page that failed -- fetch refused, or the extractor crashed on it
        # -- used to vanish silently (bug 61): _one swallowed fetch errors to
        # [] and this loop kept only lists. SOME pages failing is weather and
        # the pages that worked still serve; EVERY page failing is the
        # provider being down, and that is a refusal the router, health and
        # breaker must see -- not "no listings".
        items: list[Item] = []
        failures: list[str] = []
        for d, r in zip(candidates, results):
            if isinstance(r, BaseException):
                failures.append(f"{d.domain}: {type(r).__name__}: {str(r)[:80]}")
            else:
                items.extend(r)
        if failures and len(failures) == len(candidates):
            raise ProviderError(
                self.name,
                f"all {len(candidates)} product pages failed: " + "; ".join(failures)[:400],
            )
        return items

    #: items_vehicle_listings: results pages read per query, and how long one
    #: may take (a page behind a browser-only wall must not hold the answer).
    VEHICLE_PAGES = 5
    VEHICLE_BUDGET_S = 30.0

    async def _vehicles(self, q: Query, ctx: ProviderContext) -> list[Item]:
        """A vehicle query: results pages on the vehicle marketplaces, one
        Item per car (VIN-keyed, so two trucks never merge), cheapest first."""
        text = q.text if _FOR_SALE.search(q.text) else f"{q.text} for sale"
        docs: list[Doc] = []
        router = getattr(ctx, "router", None)
        if router is not None:
            # Unscoped first: eight ``site:`` terms make a 29-word query that
            # one engine rejected as prose and another blocked (live), while
            # the plain query's first page is vehicle marketplaces anyway.
            try:
                res = await router.search(Query(text=text, intent="web", k=max(q.k * 2, 12)),
                                          allow_expensive=False)
                docs = [d for d in res.docs if _domain_allowed(d.domain, VEHICLE_SITES)]
            except Exception:  # noqa: BLE001 -- the scoped search decides
                docs = []
        if not docs:
            try:
                docs = await self._discover(
                    Query(text=text, intent="shopping", k=q.k, domains=list(VEHICLE_SITES)), ctx)
            except Exception:  # noqa: BLE001 -- the direct pages below still answer
                docs = []
        if not any(_domain_allowed(d.domain, VEHICLE_SITES) for d in docs):
            # No engine found a vehicle page (or every engine refused): ask
            # the marketplaces directly.
            docs = [Doc(url=u, title=q.text, source=self.name) for u in vehicle_urls(q.text)]
        seen: set[str] = set()
        pages: list[Doc] = []
        for d in docs:
            # One page per site first: each site's results repeat across its URLs.
            if _domain_allowed(d.domain, VEHICLE_SITES) and d.domain not in seen:
                seen.add(d.domain)
                pages.append(d)
        pages = pages[: self.VEHICLE_PAGES]
        if not pages:
            return []

        async def one(d: Doc) -> list[Item]:
            res = await ctx.ladder.fetch(d.url)
            page = res.final_url or d.url
            loop = asyncio.get_running_loop()
            items = await loop.run_in_executor(None, vehicle_items, res.body or "", page, q.text)
            for it in items:
                it.meta["tier"] = res.tier
            return items

        # One deadline for all pages, and no waiting on a cancelled fetch to
        # unwind (wait_for does; a browser pass took 90 s to let go live): the
        # pages that answered in time are the answer.
        tasks = [asyncio.ensure_future(one(d)) for d in pages]
        done, pending = await asyncio.wait(tasks, timeout=self.VEHICLE_BUDGET_S)
        for t in pending:
            t.add_done_callback(lambda f: f.cancelled() or f.exception())
            t.cancel()
        results = [t.exception() or t.result() if t in done and not t.cancelled()
                   else asyncio.TimeoutError(f"over {self.VEHICLE_BUDGET_S:.0f} s") for t in tasks]
        items: list[Item] = []
        failures: list[str] = []
        for d, r in zip(pages, results):
            if isinstance(r, BaseException):
                failures.append(f"{d.domain}: {type(r).__name__}: {str(r)[:80]}")
            else:
                items.extend(r)
        if failures and len(failures) == len(pages):
            raise ProviderError(
                self.name, f"all {len(pages)} vehicle pages failed: " + "; ".join(failures)[:400])
        items.sort(key=lambda it: it.price.amount if it.price.amount is not None else float("inf"))
        return items

    async def _discover(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        """Find candidate product pages, with failover.

        Discovery goes through the router when there is one. Calling a single
        engine directly makes that engine a single point of failure for the
        whole shopping path -- when DuckDuckGo started refusing this client
        mid-development, item lookup returned nothing even though Bing and the
        browser tier were both available and willing.

        Intent is ``web``, not ``shopping``: routing a shopping intent here
        would select this very provider and recurse.
        """
        scoped = Query(
            text=q.text,
            intent="web",
            k=max(q.k * 2, 12),
            domains=list(q.domains or self.domains),
        )
        router = getattr(ctx, "router", None)
        if router is not None:
            res = await router.search(scoped, allow_expensive=False)
            if not res.docs:
                # Every cheap generalist refused. A browser is worth it here,
                # because the alternative is answering "no listings" to a
                # question that does have an answer.
                res = await router.search(scoped, allow_expensive=True)
            if res.docs:
                return res.docs
            if not res.failed:
                # Every provider answered and none had a candidate: that is
                # an honest empty, not a failure (bug 129 -- "discovery
                # failed: {}" for a nonsense query became a provider refusal
                # on the tool, the API and the CLI once find_items stopped
                # swallowing refusals in bug 120).
                return []
            raise ProviderError(self.name, f"discovery failed: {res.failed}")

        from .engines import DuckDuckGo

        try:
            return await DuckDuckGo().search(scoped, ctx)
        except ProviderError as exc:
            raise ProviderError(self.name, f"discovery failed: {exc}") from exc

    async def _one(self, doc: Doc, ctx: ProviderContext, *, allowed=()) -> list[Item]:
        """Fetch one product page and extract whatever it declares about itself."""
        # Let a fetch failure propagate: find_items counts it per page and
        # refuses only when every page failed (bug 61). Swallowing it here
        # made an outage indistinguishable from a page with no offer.
        res = await ctx.ladder.fetch(doc.url)
        page = res.final_url or doc.url
        # The page that ANSWERED must be in scope, not just the one asked for
        # (bug 117): a scoped amazon.com candidate that geo-redirected to
        # amazon.com.mx was extracted as amazon.com -- MXN offers in a USD
        # merge, bug 60 through the back door. Counted as a failed page.
        landed = urlsplit(page).netloc
        if allowed and landed and not _domain_allowed(landed, allowed):
            raise ProviderError(self.name, f"{doc.domain}: redirected off-scope to {landed}")

        items = items_from_html(res.body, page, source=doc.domain)
        if items:
            # A product page's OWN product, not its carousel (bug 118): every
            # Product node used to go out as a listing for the query, so the
            # "customers also bought" strip became three cheaper offers for
            # other things. When some node names this page, only those count.
            from ..fuse import canonical_url

            own = [it for it in items if canonical_url(it.url) == canonical_url(page)]
            items = own or items
            for it in items:
                it.meta["tier"] = res.tier
                it.meta["extraction"] = "jsonld"
                # The detail page was opened and declared this offer in its own
                # structured data -- exactly Item.verified's meaning ("detail
                # page opened and confirmed live"). The snippet fallback below
                # stays unverified: its price was never confirmed on the page.
                it.verified = True
            return items

        # No JSON-LD Product on the page. Try the page's own meta tags before
        # falling back to the search snippet -- og:price is present on plenty of
        # listings whose structured data is incomplete, and it is the retailer's
        # own number rather than whatever the SERP happened to quote.
        price = price_from_meta(res.body)
        if price.amount is None:
            price = parse_price(doc.snippet)
        if not doc.title:
            return []
        return [
            Item(
                title=doc.title,
                url=doc.url,
                price=price,
                seller=doc.domain,
                source=doc.domain,
                identity=identity_for(title=doc.title, url=doc.url),
                meta={"tier": res.tier, "extraction": "snippet_fallback"},
            )
        ]


class SidecarListings(Provider):
    """Listing extraction through SwarmIO's browser.

    For pages whose prices only exist after JavaScript runs, or whose HTML is
    behind a challenge. Expensive, so the router reaches for it last.
    """

    name = "sidecar_listings"
    caps = frozenset({Capability.SHOPPING, Capability.LOCAL})
    cost_per_1k = 5.0

    def __init__(self, *, max_pages: int = 8) -> None:
        super().__init__()
        self.max_pages = max_pages

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        items = await self.find_items(q, ctx)
        return [
            Doc(url=i.url, title=i.title, snippet=str(i.price), source=self.name, rank=n,
                meta={"item": i.model_dump()})
            for n, i in enumerate(items)
        ]

    async def find_items(self, q: Query, ctx: ProviderContext) -> list[Item]:
        sc = getattr(ctx.ladder, "sidecar", None)
        if sc is None or not sc.available:
            raise ProviderError(self.name, "sidecar unavailable")

        if await sc.backend_kind() == "engine":
            # The Rust engine speaks the protocol verbs but not the sidecar
            # script's composed research_listings pipeline — SERP scraping has
            # no engine equivalent by design (the engine owns verbs, not web
            # search). The same shopping intent is served with the engine's
            # own primitives instead: router discovery, the ladder's browser
            # tier for JS-rendered prices, JSON-LD/meta extraction. One
            # backend, one adapter less, and the router's fallback provider
            # answers instead of erroring on an unknown verb.
            return await self._engine_items(q, ctx)

        res = await sc.research_listings(
            q.text,
            sources=list(q.domains or self.default_sources()),
            extract_fields=["price", "brand", "condition", "availability"],
        )
        if not res.get("ok"):
            raise ProviderError(self.name, str(res.get("error"))[:200])

        # The sidecar was ASKED for these sources; rows from anywhere else are
        # not the answer to the question (bug 60: an ad or comparison-site hop
        # in its SERP scrape went out as a listing). Same predicate as the
        # cheap path. The cap is a cost bound -- engine.find_items trims the
        # merged result to k, so unbounded rows only buy merge work.
        allowed = tuple(q.domains or self.default_sources())
        cap = max(q.k * 4, 40)
        out: list[Item] = []
        for raw in res.get("listings") or res.get("items") or []:
            url = raw.get("url") or ""
            if not url.startswith("http"):
                continue
            if allowed and not _domain_allowed(urlsplit(url).netloc, allowed):
                continue
            if len(out) >= cap:
                break
            title = str(raw.get("title") or "")
            attrs = raw.get("attributes") or {}
            out.append(
                Item(
                    title=title,
                    url=url,
                    price=parse_price(str(raw.get("price") or "")),
                    seller=str(raw.get("seller") or urlsplit(url).netloc),
                    brand=str(attrs.get("brand") or ""),
                    condition=str(attrs.get("condition") or ""),
                    availability=str(attrs.get("availability") or ""),
                    attributes={k: str(v) for k, v in attrs.items()},
                    source=self.name,
                    verified=bool(raw.get("verified")),
                    identity=identity_for(brand=str(attrs.get("brand") or ""), title=title, url=url),
                )
            )
        return out

    async def _engine_items(self, q: Query, ctx: ProviderContext) -> list[Item]:
        """Engine-backend realization of this provider's intent.

        Delegates to the marketplace flow scoped to this provider's sources:
        same discovery discipline (router first, one engine directly as the
        failover), same rendered-page extraction. Items are tagged so an
        agent can see which backend served them.
        """
        prov = MarketplaceItems(
            domains=tuple(q.domains or self.default_sources()),
            max_pages=self.max_pages,
        )
        items = await prov.find_items(q, ctx)
        for it in items:
            it.meta["backend"] = "engine"
        return items

    @staticmethod
    def default_sources() -> list[str]:
        return list(MARKETPLACES[:6])
