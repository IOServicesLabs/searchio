"""Facebook Marketplace: the large marketplace with no structured data at all.

Every other adapter in this package leans on the same trick -- fetch the page,
read its schema.org JSON-LD, done -- because retailers maintain that markup for
Google Shopping and therefore keep it accurate. Facebook publishes none of it.
A rendered Marketplace search page carries zero JSON-LD Products, and running
it through the article extractor returns 375 characters of filter sidebar
("Sort by / Price / Delivery method / Condition"), because the listings are not
prose and trafilatura is right to discard them.

So this provider reads the page twice, in two different languages:

* **The streamed Relay payload.** Facebook ships the search feed inside the
  document itself, as a ``RelayPrefetchedStreamCache`` segment keyed
  ``"marketplace_search"`` -> ``feed_units.edges[].node.listing``. Each edge
  is a fully structured record -- FB-normalized title ("2011 Ford F-150 · XL
  Pickup 2D 6 1/2 ft"), amount + formatted price, reverse-geocoded city/state,
  mileage subtitles, was-price, live/sold/pending flags, seller name. This is
  present in the *raw response document*, which makes the listing feed
  readable over plain HTTP: no browser, no scripts, no scrolling. Both
  sidecar backends can serve it, and the from-scratch engine does it in
  seconds. A gated session (see below) is visible here as
  ``"marketplace_search": {"feed_units": null}`` -- an explicit empty answer,
  which the parser reports as emptiness rather than an error.
* **The rendered DOM.** ``parse_listings`` reads listing cards out of the
  live document by URL shape and span order. This is what sees listings that
  arrive only after load -- a trusted session lazy-loads hundreds of cards
  past the initial batch, and scrolling grows the DOM but never the payload.
  The provider merges both parses (payload wins per listing id), so the
  scroll loop still earns its keep for sessioned queries and the payload
  carries anonymous queries even after React unmounts the feed on scroll
  (an anonymous quirk measured September 2026: three scrolls in, the login
  wall unmounts every card; the streamed script that carried the data is
  still in the document).

Three facts make the DOM half affordable:

* The listing anchor is keyed by URL shape (``/marketplace/item/<id>``), not by
  a class name. Facebook regenerates its CSS constantly and has never moved
  that path.
* Inside each anchor the fields arrive as an ordered run of ``<span>`` elements
  -- ``[badge] price [was-price] title [location]`` -- which is stable because
  it is reading order, not styling.
* Everything is optional except the price and the title, so the parse reads
  positionally from the price outward rather than by index.

Both halves of that shape earn their complexity from real listings::

    ['Just listed', '$130', 'Bicycle', 'New York, NY']
    ['$65', '$80', 'Medium Sized Foldable Bike', 'New York, NY']
    ['Free', '$50', 'Old Bikes', 'Morristown, NJ']
    ['$500', '1 Bed 1 Bath Apartment']

The second row is why the *first* price token wins and the rest are recorded as
``was_price``: the second number is the struck-through original, and taking the
last one would report $80 for a $65 bike -- the same class of error as reading
a related-items carousel for a product's price. The fourth row is why location
is detected rather than assumed to be last: a positional rule would silently
eat "1 Bed 1 Bath Apartment" and file it as a place.

Browser-only, and honestly so. Tiers 0 and 1 both retrieve the page and both
get eight characters of text out of it, because Marketplace is an empty React
mount until scripts run.

How many results to expect
--------------------------

Logged out, Facebook caps a search at roughly **14 listings** and a category
page at roughly 24, and no amount of scrolling moves that. Measured, not
assumed: with the results pane scrolled to its own bottom and given 2.2s to
settle, fourteen consecutive scrolls left the count at 14 while the page footer
offered "See more on Facebook" and "Create new account". The ceiling is a login
gate, not a lazy loader that needs more patience. Re-measured headed with
human-smooth scrolling in September 2026: still capped, and the same is true of
the category feed at 24 -- the limits are server-side, not interaction-side.

The scroll loop is still here and still worth its ~3s, because it is exactly
what turns 14 into hundreds the moment a session exists **that Facebook's risk
engine trusts**. Point ``SEARCHIO_FACEBOOK_SESSION`` at a Playwright
``storage_state`` JSON and this provider will inject it before navigating. The
way to produce that file without handing an automated script a password is
SwarmIO's own ``interactive_login`` verb -- it opens a headed browser, waits for
a human to log in by hand, and keeps the resulting cookies in the persistent
profile, capturing no credentials. That is also the safer route in practice:
scripted credential submission from a fresh automation profile is the single
most reliable way to earn a checkpoint, and a checkpointed account returns
fewer listings than no account at all.

What "trusts" means, measured on a scripted sock-puppet account (September
2026): the account passed every visible gate -- Marketplace status page reads
"You have full access to Marketplace", Account Quality reads "No account or
asset issues", browse/category/home feeds all render -- yet every search, in
every city, returned a page whose streamed Relay payload carries
``"marketplace_search": {"feed_units": null}`` with ``complete: true``. Not an
error, not a checkpoint, an explicit empty answer, invisible in the UI (the
results pane just renders blank grey). The identical browser, same query,
cookies wiped: 11-37 listings. So a session from a young or automation-flagged
account can be *worse* than no session for search specifically, and the only
surface that tells you is this meta field.

That is why a configured session that yields zero listings is not reported as
plain emptiness: the provider retries once without the session and tags the
results ``session="anonymous(fallback)"`` with ``session_gate="empty_feed"``,
so a silently-gated login is visible instead of masquerading as a dead market.
For volume, two levers matter: a real, aged account's session (the reliable
lift), and ``SEARCHIO_FACEBOOK_HEADED=1`` to run the browser with a visible
window -- measured 37 listing hrefs headed vs 11 for the identical anonymous
query headless, which matches the user's own suspicion that the forced
headless window reads as bot traffic.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

from selectolax.parser import HTMLParser

from ..errors import ProviderError
from ..extract import parse_price
from ..models import Capability, Doc, Item, Price, Query
from .base import Provider, ProviderContext

HOST = "www.facebook.com"
DOMAIN = "facebook.com"

#: The listing link. Keyed on the path because it is the one selector Facebook
#: has never churned; every class name on the page is generated per build.
_ITEM_HREF = re.compile(r"/marketplace/item/(\d+)")

#: A span that is a price rather than a word. Anchored so a title that merely
#: mentions a number ("2010 Ford f250") cannot be mistaken for one.
_PRICE_TOKEN = re.compile(r"^(?:US\$|CA\$|A\$|[$€£¥₹])\s?\d[\d,]*(?:\.\d{2})?\b")

#: "City, REGION". The tail must be capitalised, at most three words, and free
#: of digits, which is what separates a real location from a title that happens
#: to contain a comma: "3-Bed, 2-Bath House for Rent" fails on the leading
#: digit, "Bike, red" fails on the lowercase tail.
_LOCATION = re.compile(r"^[^,]{1,60},\s*(?P<tail>[A-Z][A-Za-z .'\-]{0,24})$")

#: Only ``[a-z0-9-]`` survives into the URL path. A city slug arrives from
#: configuration or an API caller, and it is interpolated into a path segment.
_CITY_UNSAFE = re.compile(r"[^a-z0-9-]")

#: Trailing spans that describe delivery rather than the item. They sit where
#: a location would and must not be glued onto the title: "Intex Explorer K2
#: Inflatable Kayak W/ Paddles Ships to you" is not the name of anything.
_DELIVERY = frozenset({"ships to you", "free shipping", "local pickup",
                       "pickup only", "delivery available"})

#: Facebook's own filter values; it offers no 90/365-day option, so anything
#: longer than a month is expressed by not filtering.
_FRESHNESS_DAYS = {"day": 1, "week": 7, "month": 30}

#: Scroll the results pane and report how many listings exist afterwards.
#:
#: Two things here were learned the hard way against the live page. Facebook
#: scrolls an inner container, not the window, so ``window.scrollBy`` moved
#: nothing at all -- ``scrollY`` stayed 0 and ``body.scrollHeight`` stayed 800
#: through twelve scrolls while the count sat at 14. And the pane cannot be
#: found by class, because every class on the page is build-generated
#: (``x9f619 x1ja2u2z ...``); the durable way to it is to walk up from a
#: listing link to the first scrollable ancestor.
#:
#: Returning the count is what lets the caller stop as soon as the page stops
#: growing rather than spending a fixed budget of scrolls.
_SCROLL_JS = """
(async () => {
  const sel = 'a[href*="/marketplace/item/"]';
  const count = () => document.querySelectorAll(sel).length;
  const anchor = document.querySelector(sel);
  let pane = null;
  for (let el = anchor; el && el !== document.documentElement; el = el.parentElement) {
    const st = getComputedStyle(el);
    if ((st.overflowY === 'auto' || st.overflowY === 'scroll') &&
        el.scrollHeight > el.clientHeight + 50) { pane = el; break; }
  }
  if (pane) {
    pane.scrollTop = Math.min(pane.scrollTop + pane.clientHeight * 2, pane.scrollHeight);
  } else {
    window.scrollBy(0, window.innerHeight * 3);
  }
  await new Promise(r => setTimeout(r, __SETTLE_MS__));
  return count();
})()
"""


def _is_free(text: str) -> bool:
    return text.strip().lower() in ("free", "gratis")


def _dom_price(text: str) -> "Price":
    if _is_free(text):
        return Price(amount=0.0, raw="Free")
    return parse_price(text)


def _looks_like_location(text: str) -> bool:
    m = _LOCATION.match(text)
    if not m:
        return False
    tail = m.group("tail")
    return len(tail.split()) <= 3 and not any(c.isdigit() for c in tail)


def _fields_from_spans(spans: list[str]) -> tuple[list[str], str, str, str, str, str]:
    """Split a card's spans into (badges, price, was, title, location, delivery).

    Reads outward from the first price token, which is the only span whose role
    can be identified by its own content. Everything before it is a badge,
    everything after it is title text, minus a trailing location if one is
    there and minus any further price tokens.
    """
    # "Free" IS a price token (bug 86): a free listing renders "Free" then
    # its struck-through old price, and without this the old price became
    # THE price and "Free" a badge -> condition "Free", price $50.
    price_at = [i for i, s in enumerate(spans) if _PRICE_TOKEN.match(s) or _is_free(s)]
    if price_at:
        first = price_at[0]
        badges = spans[:first]
        price = spans[first]
        was = spans[price_at[1]] if len(price_at) > 1 else ""
        skip = set(price_at)
        rest = [s for i, s in enumerate(spans) if i > first and i not in skip]
    else:
        # A card with no price at all -- "Free Stuff" listings do this. Still a
        # real listing, so keep it and let the price stay unknown.
        badges, price, was, rest = [], "", "", list(spans)

    # Peel trailing metadata off the title. Both a location and a delivery
    # marker can be present, in either order, so this loops rather than
    # checking one fixed position.
    location = delivery = ""
    while rest:
        tail = rest[-1]
        if not delivery and tail.strip().lower() in _DELIVERY:
            delivery, rest = tail, rest[:-1]
        elif not location and _looks_like_location(tail):
            location, rest = tail, rest[:-1]
        else:
            break
    return badges, price, was, " ".join(rest).strip(), location, delivery


def parse_listings(html: str, *, source: str = "facebook_marketplace") -> list[Item]:
    """Every distinct listing on a rendered Marketplace page, in page order."""
    tree = HTMLParser(html)
    out: list[Item] = []
    seen: set[str] = set()

    for a in tree.css('a[href*="/marketplace/item/"]'):
        m = _ITEM_HREF.search(a.attributes.get("href") or "")
        if not m:
            continue
        item_id = m.group(1)
        if item_id in seen:
            continue

        spans = [" ".join(s.text(deep=False, separator=" ").split()) for s in a.css("span")]
        badges, price_txt, was_txt, title, location, delivery = _fields_from_spans(
            [s for s in spans if s]
        )
        if not title:
            # An image-only card, or a shape this parse does not understand.
            # Emitting it with an empty title would put a blank row in front of
            # a user; dropping it loses nothing they could have used.
            #
            # Deliberately *not* marked seen. Facebook wraps most cards in two
            # anchors to the same item -- one around the photo, one around the
            # text -- and claiming the id on the photo would discard the only
            # copy that carries the fields.
            continue

        seen.add(item_id)

        attrs = {"item_id": item_id}
        if location:
            attrs["location"] = location
        if delivery:
            attrs["delivery"] = delivery
        if was_txt:
            attrs["was_price"] = was_txt
        if badges:
            attrs["badges"] = ", ".join(badges)
        # A Sold/Pending badge is the listing's availability (bug 149 -- the
        # payload path learned this as bug 84; the DOM path filed it under
        # `condition` and offered the card as live).
        status = {b.strip().lower() for b in badges}
        if "sold" in status:
            availability = "SoldOut"
        elif "pending" in status:
            availability = "Pending"
        else:
            availability = "InStock"
        condition_badges = [b for b in badges if b.strip().lower() not in ("sold", "pending")]

        out.append(
            Item(
                title=title[:400],
                # Canonical and query-free: the href carries tracking and
                # referrer parameters that differ per render, so two fetches of
                # the same listing would otherwise dedupe as two items.
                url=f"https://{HOST}/marketplace/item/{item_id}/",
                price=_dom_price(price_txt),
                availability=availability,
                condition=", ".join(condition_badges)[:120],
                source=source,
                # NOT the usual brand+model join key. Every other provider
                # describes catalogue products, where collapsing two sellers of
                # one SKU is the whole point. These are individual used goods:
                # two "Bicycle" listings in one city are two different bicycles
                # and must never merge, so identity is the listing itself.
                identity=f"fbmp:{item_id}",
                attributes=attrs,
            )
        )
    return out


#: The streamed Relay result object as it appears in the document: the feed
#: sits at ``result.data.marketplace_search.feed_units.edges`` inside a
#: ``RelayPrefetchedStreamCache`` script segment. Braced form is matched so a
#: mention of the key in prose or tracking data never starts a decode.
_PAYLOAD_OPEN = '{"marketplace_search"'


def _item_from_relay_listing(listing: dict, *, source: str) -> Item | None:
    """One ``GroupCommerceProductItem`` node -> Item, richer than the DOM parse.

    The payload carries fields the card spans never surface: FB-normalized
    title, exact amount, mileage subtitles, reverse-geocoded city/state,
    was-price, live/sold/pending flags, seller name, creation time.
    """
    lid = str(listing.get("id") or "")
    if not lid:
        return None
    title = str(
        listing.get("marketplace_listing_title")
        or listing.get("custom_title")
        or ""
    ).strip()
    if not title:
        return None

    price_obj = listing.get("listing_price") or {}
    formatted = str(price_obj.get("formatted_amount") or "")
    price = _dom_price(formatted)
    if price.amount is None and price_obj.get("amount") not in (None, ""):
        price = parse_price(str(price_obj["amount"]))
        if price.amount is None and str(price_obj["amount"]).strip() in ("0", "0.0", "0.00"):
            price = Price(amount=0.0, raw=formatted or "0")

    was = (listing.get("strikethrough_price") or {}).get("formatted_amount") or ""

    geo = (listing.get("location") or {}).get("reverse_geocode") or {}
    location = ", ".join(p for p in (geo.get("city") or "", geo.get("state") or "") if p)

    subtitles = [
        str(s.get("subtitle"))
        for s in (listing.get("custom_sub_titles_with_rendering_flags") or [])
        if isinstance(s, dict) and s.get("subtitle")
    ]
    seller = (listing.get("marketplace_listing_seller") or {}).get("name") or ""
    delivery = ", ".join(str(d) for d in (listing.get("delivery_types") or []))
    flags = ",".join(
        name for name in ("is_live", "is_sold", "is_pending")
        if listing.get(name) is True
    )

    attrs: dict[str, str] = {"item_id": lid, "extraction": "relay_payload"}
    if location:
        attrs["location"] = location
    if subtitles:
        attrs["mileage"] = " · ".join(subtitles)
    if was:
        attrs["was_price"] = was
    if seller:
        attrs["seller"] = seller
    if delivery:
        attrs["delivery"] = delivery
    if flags:
        attrs["flags"] = flags
    created = listing.get("creation_time")
    if isinstance(created, (int, float)):
        attrs["listed"] = time.strftime("%Y-%m-%d", time.gmtime(created))
    custom = str(listing.get("custom_title") or "").strip()
    if custom and custom != title:
        attrs["custom_title"] = custom

    # The flags are the listing's availability (bug 84): a sold or pending
    # listing used to carry availability "" -- unknown -- and was offered to
    # the agent exactly like a live one. schema.org vocabulary, as the other
    # extractors emit it.
    if listing.get("is_sold") is True:
        availability = "SoldOut"
    elif listing.get("is_pending") is True:
        availability = "Pending"
    elif listing.get("is_live") is True:
        availability = "InStock"
    else:
        availability = ""

    return Item(
        title=title[:400],
        url=f"https://{HOST}/marketplace/item/{lid}/",
        price=price,
        seller=seller,
        availability=availability,
        source=source,
        identity=f"fbmp:{lid}",
        attributes=attrs,
    )


def parse_listings_payload(
    html: str, *, source: str = "facebook_marketplace"
) -> list[Item]:
    """Every listing in the streamed Relay payload, across all embedded batches.

    Anonymous feeds ship in the initial document; every occurrence of the
    result object is decoded and merged, deduped by listing id. A session the
    risk engine has gated answers ``feed_units: null`` -- that is emptiness,
    not an error, and surfaces here as no items, same as a genuinely empty
    market. Malformed segments are skipped, not fatal: the DOM parse still
    has a chance.
    """
    items: list[Item] = []
    seen: set[str] = set()
    dec = json.JSONDecoder()
    start = 0
    while True:
        i = html.find(_PAYLOAD_OPEN, start)
        if i < 0:
            break
        start = i + len(_PAYLOAD_OPEN)
        try:
            obj, _ = dec.raw_decode(html[i:])
        except json.JSONDecodeError:
            continue
        search = obj.get("marketplace_search")
        if not isinstance(search, dict):
            continue  # gated: null; also tolerate {"marketplace_search": ...}
        edges = (search.get("feed_units") or {}).get("edges") or []
        for edge in edges:
            node = (edge or {}).get("node") or {}
            listing = node.get("listing")
            if not isinstance(listing, dict):
                continue  # ad units and story placeholders carry no listing
            it = _item_from_relay_listing(listing, source=source)
            if it is not None and it.identity not in seen:
                seen.add(it.identity)
                items.append(it)
    return items


def merge_listings(
    primary: list[Item], fallback: list[Item]
) -> list[Item]:
    """Two parses of one page, payload first, DOM cards filling any gaps.

    Same listing in both: the payload copy wins (richer fields). Only ids the
    payload missed -- cards lazy-loaded after the initial batch -- are taken
    from the DOM parse.
    """
    seen = {it.identity for it in primary}
    return primary + [it for it in fallback if it.identity and it.identity not in seen]


class FacebookMarketplace(Provider):
    """Local classified listings from the streamed feed and the rendered DOM.

    Anonymous queries are carried by the Relay payload alone -- readable over
    plain HTTP through either sidecar backend, and immune to the scroll-time
    login wall that unmounts the card DOM. Sessioned queries lean on the DOM
    parse, which is what sees the hundreds of cards lazy-loaded past the
    initial batch.
    """

    name = "facebook_marketplace"
    caps = frozenset({Capability.SHOPPING, Capability.LOCAL})
    #: The browser tier costs ~15-30s per query when it is involved, so the
    #: router should reach for it deliberately rather than in a cheap fan-out.
    cost_per_1k = 5.0
    primary_domain = DOMAIN

    def __init__(
        self,
        *,
        city: str = "",
        max_scrolls: int = 8,
        settle_ms: int = 1400,
        timeout_s: float = 90.0,
    ) -> None:
        super().__init__()
        self.city = city
        self.max_scrolls = max_scrolls
        self.settle_ms = settle_ms
        self.timeout_s = timeout_s
        self._session_state = "none"
        self._session_tried = False
        #: Set when a loaded session returned an empty search feed and the
        #: anonymous retry produced listings. Sticky for the process: once the
        #: gate has been observed, later queries skip the session rather than
        #: re-proving it per query.
        self._session_broken = False
        #: What the last query concluded about the session/market, for any
        #: caller that wants to explain a small result set. One of
        #: "", "empty_feed" (session answered null, anonymous worked),
        #: "empty_market" (anonymous also found nothing), "reset_failed".
        self.search_gate = ""
        self._gates: dict[str, str] = {}
        self._session_lock = None  # created lazily: no loop at construction
        #: Where the injected session can be re-read from (the session file,
        #: or the handoff's exported state): the anonymous check wipes the
        #: browser whatever it concludes, and an empty market must give the
        #: session back (bug 162).
        self._session_source: str | dict | None = None
        #: Loads in flight vs the session reset (bug 161): the reset waits
        #: for every load to finish and no load starts while it runs.
        self._loads_active = 0
        self._loads_idle = None  # asyncio.Event, set while nothing is loading
        self._reset_open = None  # asyncio.Event, cleared while a reset runs

    # ── URL ──────────────────────────────────────────────────────────────────

    @staticmethod
    def url_for(q: Query, city: str = "") -> str:
        """The search URL, with the city pinned into the path when we have one.

        The city matters more than it looks. Facebook resolves an unpinned
        Marketplace URL against the request's exit IP, so the same query
        answered from a datacentre returns whatever metro that address lives in
        -- a bare ``/marketplace/category/vehicles`` came back full of
        Chattanooga, Tennessee. Results that are confidently about the wrong
        city are worse than no results, so callers are given somewhere to say
        which one they meant.
        """
        slug = _CITY_UNSAFE.sub("", (city or "").strip().lower())
        base = f"https://{HOST}/marketplace/{slug}/search" if slug else f"https://{HOST}/marketplace/search"
        params = {"query": q.text}
        days = _FRESHNESS_DAYS.get(q.freshness)
        if days:
            params["daysSinceListed"] = str(days)
        return f"{base}?{urlencode(params)}"

    def _city(self, ctx: ProviderContext) -> str:
        return self.city or getattr(ctx.settings, "facebook_city", "") or ""

    # ── provider interface ───────────────────────────────────────────────────

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        items = await self.find_items(q, ctx)
        return [
            Doc(
                url=it.url,
                title=it.title,
                snippet=" · ".join(
                    p for p in (str(it.price), it.attributes.get("location", "")) if p
                ),
                source=self.name,
                rank=i,
                meta={"item": it.model_dump()},
            )
            for i, it in enumerate(items)
        ]

    async def find_items(self, q: Query, ctx: ProviderContext) -> list[Item]:
        sc = getattr(ctx.ladder, "sidecar", None)
        if sc is None or not getattr(sc, "available", False):
            raise ProviderError(self.name, "sidecar unavailable: Marketplace is browser-only")

        self._prefer_headed(ctx)
        city = self._city(ctx)
        url = self.url_for(q, city)
        await self._gate(ctx, url)
        await self._ensure_session(sc, ctx)

        # The gate is a PER-CALL verdict (bug 85): it used to live only on the
        # instance (reset at the top of every call), so two concurrent
        # searches on the one registry instance raced -- B's reset erased A's
        # proven gate before A read it, and A's handoff decision was taken on
        # B's state. `search_gate` stays as the last verdict, for callers
        # that read it after a single search; `last_gate(text)` is per query.
        gate = ""
        # A dedicated tab per call. Marketplace mutates the DOM as it scrolls,
        # so sharing the default tab would let two concurrent queries read each
        # other's listings.
        tab = f"fbmp-{uuid.uuid4().hex[:8]}"
        try:
            items, res = await self._load(sc, tab, url, want=q.k)
            if not items and self._session_active():
                # A loaded session that returns zero listings is indistinguishable
                # from a dead market -- unless the identical anonymous query
                # finds something, which proves the session is the problem.
                items, gate = await self._retry_anonymous(sc, url, want=q.k)
                if gate == "empty_feed" and getattr(
                    ctx.settings, "facebook_auto_handoff", False
                ):
                    # The session was proven to be the problem and the operator
                    # opted into human assistance: a headed window a human logs
                    # into by hand, its exported session injected for one retry.
                    # Any handoff failure leaves the anonymous fallback standing.
                    handed = await self._handoff_and_retry(sc, url, want=q.k, ctx=ctx)
                    if handed:
                        # Recovered, and the record says so: the gate WAS
                        # proven (review candidate 6 -- erasing it hid why a
                        # human had to log in).
                        items, gate = handed, "empty_feed(recovered by handoff)"
        finally:
            try:
                await sc.call("close_tab", {"tab_id": tab})
            except Exception:
                # Losing a tab leaks a little browser memory; failing the query
                # over it would trade a real answer for tidiness.
                pass
        self.search_gate = gate
        self._gates[q.text] = gate

        if not items and res.get("bot_detected"):
            # Only now is the bot verdict worth acting on. Taken at face value
            # it is wrong on this host: the sidecar's keyword check fires on
            # "checkpoint"/"captcha"/"challenge", and Facebook ships all three
            # inside its own JavaScript bundle on every Marketplace page. A run
            # reporting `keyword:challenge` was carrying seventeen real Seattle
            # kayak listings. Retrieval is the stronger evidence, so the verdict
            # only decides the empty case.
            raise ProviderError(self.name, f"blocked: {res.get('bot_reason') or 'bot_detected'}")
        for it in items:
            it.meta["geo"] = f"city:{city}" if city else "ip-derived"
            it.meta.setdefault("session", self._session_state)
        return items[: q.k]

    def _sync(self) -> None:
        import asyncio

        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        if self._loads_idle is None:
            self._loads_idle = asyncio.Event()
            self._loads_idle.set()
            self._reset_open = asyncio.Event()
            self._reset_open.set()

    async def _load(self, sc, tab: str, url: str, *, want: int) -> tuple[list[Item], dict]:
        """Navigate one tab and read whatever listings it ends up holding.

        Counted as in flight for the session reset (bug 161): session_reset
        drops every tab on the engine and every cookie on both backends, so a
        gate hit by one query used to fail -- or silently de-session -- the
        query loading next to it. A reset now drains the loads before it and
        holds new ones until its own reload has run.
        """
        self._sync()
        await self._reset_open.wait()
        self._loads_active += 1
        self._loads_idle.clear()
        try:
            return await self._load_now(sc, tab, url, want=want)
        finally:
            self._loads_active -= 1
            if self._loads_active == 0:
                self._loads_idle.set()

    async def _load_now(self, sc, tab: str, url: str, *, want: int) -> tuple[list[Item], dict]:
        res = await sc.goto(url, tab_id=tab)
        await self._scroll(sc, tab, want=want)
        html = await sc.read_html(tab_id=tab)
        # The payload parse works on the raw streamed document (anonymous
        # feeds ship complete in it) and survives the scroll-time unmount;
        # the DOM parse sees cards lazy-loaded after the initial batch.
        # Same listing in both: payload wins for field richness.
        payload = parse_listings_payload(html, source=self.name)
        return merge_listings(payload, parse_listings(html, source=self.name)), res

    def _session_active(self) -> bool:
        return self._session_state.startswith(("loaded", "handoff")) and not self._session_broken

    async def _handoff_and_retry(self, sc, url: str, *, want: int, ctx) -> list[Item]:
        """One human-assisted recovery once the empty-feed gate is proven.

        The handoff window and the poll live in net/login_handoff.py; this
        method only wires it to the provider's lifecycle: fresh state in via
        storage_state_set (persisted where _ensure_session reads when a
        session file is configured), session marked usable again, one load.
        LoginHandoffError covers timeout/cancel/missing-backend — the caller
        keeps the anonymous fallback on any of them.
        """
        from ..errors import LoginHandoffError
        from ..net.login_handoff import apply_to_engine, request_human_login

        try:
            result = await request_human_login(
                login_url=getattr(ctx.settings, "facebook_login_url", ""),
                timeout_s=float(getattr(ctx.settings, "handoff_timeout_s", 300.0)),
            )
        except LoginHandoffError:
            return []
        # The injection and the load under the session lock (iteration 72,
        # review #6): the human's wait stays outside it, but a gate queued
        # by another query used to run its reset between the two and wipe
        # the fresh session -- anonymous listings stamped "handoff".
        self._sync()
        async with self._session_lock:
            try:
                await apply_to_engine(
                    sc, result.state, save_path=getattr(ctx.settings, "facebook_session", "")
                )
            except LoginHandoffError:
                return []
            self._session_broken = False
            self._session_state = f"handoff:{result.cookies}c/{result.origins}o"
            self._session_source = result.state
            tab = f"fbmp-{uuid.uuid4().hex[:8]}"
            try:
                items, _res = await self._load(sc, tab, url, want=want)
            finally:
                try:
                    await sc.call("close_tab", {"tab_id": tab})
                except Exception:
                    pass
        if items:
            self.search_gate = ""
            for it in items:
                it.meta["session"] = "handoff"
                it.meta.pop("session_gate", None)
        return items

    def last_gate(self, text: str) -> str:
        """The gate verdict of the last search for exactly this query text."""
        return self._gates.get(text, "")

    async def _retry_anonymous(self, sc, url: str, *, want: int) -> tuple[list[Item], str]:
        """One logged-out retry for the empty-feed session gate.

        Wipes the browser's Facebook cookies (the injected session included)
        and answers the same query anonymously. If that finds listings, the
        session was the problem, not the market, and the items say so.
        Returns ``(items, gate)``; the gate is one of ``empty_feed`` (the
        session was the problem), ``empty_market`` (anonymous found nothing
        either) or ``reset_failed`` (the browser could not drop the session,
        so nothing was proven -- not an empty market).
        """
        self._sync()
        # Under the session lock (rider, iteration 67): two searches hitting
        # the gate at once ran reset + reload interleaved, and one search's
        # reset wiped the session under the other's in-flight load.
        async with self._session_lock:
            # Exclusive against loads (bug 161): drain what is in flight,
            # admit nothing new until the verdict -- and the session's
            # restore (bug 162) -- are in.
            self._reset_open.clear()
            try:
                await self._loads_idle.wait()
                return await self._retry_anonymous_locked(sc, url, want=want)
            finally:
                self._reset_open.set()

    async def _retry_anonymous_locked(self, sc, url: str, *, want: int) -> tuple[list[Item], str]:
        try:
            await sc.call("session_reset", {})
        except Exception:
            # Nothing was proven about the session (bug 150): it stays usable.
            self.search_gate = "reset_failed"
            return [], "reset_failed"
        tab = f"fbmp-{uuid.uuid4().hex[:8]}"
        try:
            items, _res = await self._load_now(sc, tab, url, want=want)
        finally:
            try:
                await sc.call("close_tab", {"tab_id": tab})
            except Exception:
                pass
        if items:
            # PROVEN: anonymous found what the session could not. Only now is
            # the session broken (bug 150 -- the flag used to be set before
            # the retry, so one honest empty market retired the session for
            # the rest of the process).
            gate = "empty_feed"
            self._session_broken = True
            self._session_state = "anonymous(after empty-feed gate)"
            for it in items:
                it.meta["session"] = "anonymous(fallback)"
                it.meta["session_gate"] = "empty_feed"
        else:
            # Anonymous also found nothing: the market really is empty for this
            # query/location. Not a gate; say so -- and put the session back
            # (bug 162): the reset wiped it whatever it proved, and with the
            # injection once per process the browser stayed anonymous while
            # every later listing was stamped with the session and every
            # later empty result ran another pointless reset + retry.
            gate = "empty_market"
            await self._restore_session(sc)
        self.search_gate = gate
        return items, gate

    async def _restore_session(self, sc) -> None:
        src = self._session_source
        if src is None:
            self._session_state = "anonymous(session wiped by an empty-market check; nothing to restore from)"
            return
        try:
            state = json.loads(Path(src).read_text(encoding="utf-8")) if isinstance(src, str) else src
            res = await sc.call("storage_state_set", {"storage_state": state})
            ok = bool(res.get("ok"))
        except Exception:
            ok = False
        if not ok:
            # Not broken -- nothing was proven -- but not there either: an
            # honest state keeps later empties from gating a session the
            # browser no longer holds.
            self._session_state = "anonymous(session wiped by an empty-market check; restore failed)"

    def _prefer_headed(self, ctx: ProviderContext) -> None:
        """Opt into a visible browser window when configured to.

        Measured September 2026 on the identical anonymous query: 37 listing
        hrefs headful vs 11 headless. A forced-invisible window is one of the
        stronger bot signals, and Marketplace is the most block-prone target in
        the system, so the lift is worth a window on a desktop. An operator-set
        ``SWARM_BROWSER_WINDOW_MODE`` or ``SWARM_BROWSER_HEADLESS`` always
        wins; this only fills in the default. Must run before the sidecar's
        first spawn, which is why it lives at the top of ``find_items``.
        """
        if not getattr(ctx.settings, "facebook_headed", False):
            return
        if os.environ.get("SWARM_BROWSER_WINDOW_MODE") or os.environ.get("SWARM_BROWSER_HEADLESS"):
            return
        os.environ["SWARM_BROWSER_WINDOW_MODE"] = "headful"

    async def _ensure_session(self, sc, ctx) -> None:
        """Inject a saved browser session, once per process, if one is configured.

        The outcome is recorded on every item rather than swallowed. A
        configured session that silently failed to load looks identical from
        the outside to no session at all -- fourteen results -- and the user
        would have no way to tell a login problem from Facebook's own ceiling.
        """
        self._sync()
        # Always through the lock: a fast path on the flag let a concurrent
        # first search skip past an injection still in flight.
        async with self._session_lock:
            # Under the lock (iteration 48 rider): the flag used to flip
            # before the first await, so a concurrent first search skipped
            # past an injection still in flight and ran without the session.
            if self._session_tried:
                return
            self._session_tried = True  # one attempt per process, not one per query
            path = getattr(ctx.settings, "facebook_session", "") or ""
            if not path:
                return
            p = Path(path)
            if not p.exists():
                self._session_state = "missing"
                return
            try:
                state = json.loads(p.read_text(encoding="utf-8"))
                res = await sc.call("storage_state_set", {"storage_state": state})
            except Exception as exc:
                self._session_state = f"error:{type(exc).__name__}"
                return
            self._session_state = (
                f"loaded:{res.get('cookies', 0)}c" if res.get("ok") else "rejected"
            )
            if res.get("ok"):
                self._session_source = str(p)

    async def _gate(self, ctx: ProviderContext, url: str) -> None:
        """Borrow the ladder's robots policy and per-domain pacing.

        This provider drives the browser directly -- ``fetch`` returns the
        pre-scroll shell, so it cannot be used -- which would otherwise put the
        single most block-prone request in the system outside the machinery
        built to keep requests unblocked. Note that Facebook's robots.txt
        disallows /marketplace/, so under ``robots_policy=enforce`` this
        provider correctly declines to run at all.
        """
        robots = getattr(ctx.ladder, "robots", None)
        if robots is not None:
            info = await robots.check(url)
            if not robots.permits(info):
                raise ProviderError(self.name, "disallowed_by_robots")
        limiter = getattr(ctx.ladder, "limiter", None)
        if limiter is not None:
            await limiter.acquire(DOMAIN)

    async def _scroll(self, sc, tab: str, want: int) -> int:
        """Scroll until the page stops producing listings, or we have enough.

        Two stalls rather than one before giving up: the first scroll after a
        cold load routinely reports no growth simply because the lazy loader
        has not finished, and treating that as the end of the results caps
        every query at the first screenful.
        """
        js = _SCROLL_JS.replace("__SETTLE_MS__", str(int(self.settle_ms)))
        best = stalls = 0
        for _ in range(self.max_scrolls):
            try:
                raw = await sc.eval_js(js, tab_id=tab, timeout=self.timeout_s)
            except Exception:
                break
            n = int(raw) if isinstance(raw, (int, float)) else 0
            if n >= want:
                return n
            if n <= best:
                stalls += 1
                if stalls >= 2:
                    break
            else:
                stalls = 0
                best = n
        return best
