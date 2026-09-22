"""Facebook Marketplace card parsing.

Every fixture below is a span sequence copied from a real rendered Marketplace
page, not one invented to suit the parser. The awkward ones -- two prices, a
badge before the price, a listing with no location -- are the reason the parse
reads outward from the price instead of by index.
"""

from __future__ import annotations

import pytest

from searchio.models import Query
from searchio.providers.facebook import (
    FacebookMarketplace,
    _fields_from_spans,
    _looks_like_location,
    parse_listings,
)


def card(item_id: str, spans: list[str], *, nest: bool = False) -> str:
    """One listing anchor in the shape Facebook actually renders."""
    inner = "".join(
        f"<div><span><span>{s}</span></span></div>" if nest else f"<div><span>{s}</span></div>"
        for s in spans
    )
    return f'<a href="/marketplace/item/{item_id}/?ref=search&referral_code=x">{inner}</a>'


REAL_CARDS = [
    ("4443505275967227", ["Just listed", "$130", "Bicycle", "New York, NY"]),
    ("1596449692142370", ["$35", "Bike", "Fresh Meadows, NY"]),
    ("964857849937122", ["Free", "$50", "Old Bikes", "Morristown, NJ"]),
    ("1034352085705792", ["$65", "$80", "Medium Sized Foldable Bike", "New York, NY"]),
    ("2365657317258857", ["$60", "Two Person Riding Bicycle (READ BIO)", "Elizabeth, NJ"]),
    ("6000000000000001", ["$6,000", "2010 Ford f250 super duty Pickup 4D", "Trion, GA"]),
    ("5000000000000002", ["$500", "1 Bed 1 Bath Apartment"]),
]


def page(cards=REAL_CARDS, **kw) -> str:
    return "<html><body>" + "".join(card(i, s, **kw) for i, s in cards) + "</body></html>"


# ── field splitting ──────────────────────────────────────────────────────────


def test_badge_before_price_is_not_the_title():
    badges, price, was, title, loc, _ = _fields_from_spans(
        ["Just listed", "$130", "Bicycle", "New York, NY"]
    )
    assert badges == ["Just listed"]
    assert (price, was, title, loc) == ("$130", "", "Bicycle", "New York, NY")


def test_first_price_wins_second_is_recorded_as_was():
    """The struck-through original must never become the asking price.

    Reporting $80 for a $65 bike is the same failure as reading a related-items
    carousel for a product's price -- a confident number that is someone
    else's.
    """
    _, price, was, title, _, _ = _fields_from_spans(
        ["$65", "$80", "Medium Sized Foldable Bike", "New York, NY"]
    )
    assert price == "$65"
    assert was == "$80"
    assert title == "Medium Sized Foldable Bike"


def test_listing_with_no_location_keeps_its_whole_title():
    """A positional 'last span is the location' rule ate this title."""
    _, price, _, title, loc, _ = _fields_from_spans(["$500", "1 Bed 1 Bath Apartment"])
    assert title == "1 Bed 1 Bath Apartment"
    assert loc == ""
    assert price == "$500"


def test_card_with_no_price_still_parses():
    _, price, _, title, loc, _ = _fields_from_spans(["Free stuff", "Chattanooga, TN"])
    assert price == ""
    assert title == "Free stuff"
    assert loc == "Chattanooga, TN"


# ── location heuristic ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["New York, NY", "West New York, NJ", "Signal Mountain, TN", "London, England"],
)
def test_locations_recognised(text):
    assert _looks_like_location(text)


@pytest.mark.parametrize(
    "text",
    [
        "3-Bed, 2-Bath House for Rent",  # digits in the tail
        "Bike, red",  # lowercase tail: a title, not a place
        "1 Bed 1 Bath Apartment",  # no comma at all
        "Blue Schwinn Cruiser Bicycle (Classic Old School)",
    ],
)
def test_titles_not_mistaken_for_locations(text):
    assert not _looks_like_location(text)


# ── whole-page parse ─────────────────────────────────────────────────────────


def test_parses_every_real_card():
    items = parse_listings(page())
    assert len(items) == len(REAL_CARDS)
    by_id = {i.attributes["item_id"]: i for i in items}

    bike = by_id["4443505275967227"]
    assert bike.title == "Bicycle"
    assert bike.price.amount == 130.0
    assert bike.price.currency == "USD"
    assert bike.attributes["location"] == "New York, NY"
    assert bike.condition == "Just listed"

    truck = by_id["6000000000000001"]
    assert truck.price.amount == 6000.0  # thousands separator survives
    assert truck.title.startswith("2010 Ford f250")

    discounted = by_id["1034352085705792"]
    assert discounted.price.amount == 65.0
    assert discounted.attributes["was_price"] == "$80"


def test_url_is_canonical_not_the_tracked_href():
    """Two renders hand out different ref/referral params for one listing."""
    items = parse_listings(page())
    assert items[0].url == "https://www.facebook.com/marketplace/item/4443505275967227/"


def test_identity_is_per_listing_not_per_product():
    """Two 'Bicycle' listings are two bicycles.

    The brand+model join key that collapses one SKU across retailers is exactly
    wrong here: these are individual used goods and merging them would hide a
    listing behind an unrelated one that happens to share a title.
    """
    items = parse_listings(page([("111111", ["$40", "Bike", "Newark, NJ"]),
                                 ("222222", ["$40", "Bike", "Newark, NJ"])]))
    assert len(items) == 2
    assert items[0].identity != items[1].identity
    assert items[0].identity == "fbmp:111111"


def test_photo_anchor_does_not_shadow_the_text_anchor():
    """Facebook wraps each card in two anchors to the same item.

    The first has only an image. Claiming the id there and skipping the second
    would drop the only copy carrying the price and title.
    """
    html = (
        '<a href="/marketplace/item/777777/"><div><img src="x.jpg"></div></a>'
        + card("777777", ["$99", "Real Listing", "Bronx, NY"])
    )
    items = parse_listings(html)
    assert len(items) == 1
    assert items[0].title == "Real Listing"
    assert items[0].price.amount == 99.0


def test_repeated_text_anchor_dedupes():
    html = page([("333333", ["$10", "Thing", "Bronx, NY"])]) * 2
    assert len(parse_listings(html)) == 1


def test_nested_spans_do_not_duplicate_text():
    """Direct-text reads keep a wrapper span from repeating its child."""
    items = parse_listings(page(nest=True))
    assert [i.title for i in items][:2] == ["Bicycle", "Bike"]


# ── URL construction ─────────────────────────────────────────────────────────


def test_city_is_pinned_into_the_path():
    url = FacebookMarketplace.url_for(Query(text="bicycle"), "nyc")
    assert url.startswith("https://www.facebook.com/marketplace/nyc/search?")
    assert "query=bicycle" in url


def test_city_slug_cannot_escape_its_path_segment():
    """Unsafe characters are dropped, not truncated at.

    The slug is interpolated straight into a path segment, so the property that
    matters is that nothing able to leave that segment survives -- no dot, no
    slash, and nothing that could start a query string.
    """
    url = FacebookMarketplace.url_for(Query(text="x"), "../../evil?a=b")
    segment = url.split("/marketplace/", 1)[1].split("/search", 1)[0]
    assert segment == "evilab"
    assert not any(c in segment for c in "./?&=")


def test_freshness_maps_to_facebooks_own_filter():
    assert "daysSinceListed=7" in FacebookMarketplace.url_for(
        Query(text="x", freshness="week"), "nyc"
    )
    # Facebook offers no 90/365-day option; "all" means do not filter.
    assert "daysSinceListed" not in FacebookMarketplace.url_for(
        Query(text="x", freshness="all"), "nyc"
    )


def test_query_text_is_encoded():
    url = FacebookMarketplace.url_for(Query(text="road bike & helmet"), "la")
    assert " " not in url
    assert "%26" in url or "&amp;" not in url


# ── scroll loop ──────────────────────────────────────────────────────────────


class FakeSidecar:
    """Counts scrolls and reports a listing count per call."""

    available = True

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.calls = 0

    async def eval_js(self, js, *, tab_id="default", timeout=None):
        i = min(self.calls, len(self.counts) - 1)
        self.calls += 1
        return self.counts[i]


async def test_scroll_stops_once_it_has_enough():
    sc = FakeSidecar([8, 16, 40])
    n = await FacebookMarketplace()._scroll(sc, "t", want=15)
    assert n == 16
    assert sc.calls == 2  # did not keep scrolling past the target


async def test_scroll_tolerates_one_stall_then_gives_up():
    """A cold page reports no growth on the first scroll while it lazy-loads.

    Stopping on a single stall caps every query at the first screenful, so two
    consecutive stalls are required before the loop concludes it is done.
    """
    sc = FakeSidecar([0, 12, 12, 12])
    n = await FacebookMarketplace(max_scrolls=8)._scroll(sc, "t", want=100)
    assert n == 12
    assert sc.calls == 4


async def test_scroll_survives_a_dead_tab():
    class Boom(FakeSidecar):
        async def eval_js(self, js, *, tab_id="default", timeout=None):
            raise RuntimeError("tab closed")

    assert await FacebookMarketplace()._scroll(Boom([]), "t", want=10) == 0


# ── delivery markers ─────────────────────────────────────────────────────────


def test_delivery_marker_is_not_glued_onto_the_title():
    """"Ships to you" sits where a location would and is not part of the name."""
    _, price, _, title, loc, delivery = _fields_from_spans(
        ["$75", "Intex Explorer 2-Person K2 Inflatable Kayak W/ Paddles", "Ships to you"]
    )
    assert title == "Intex Explorer 2-Person K2 Inflatable Kayak W/ Paddles"
    assert delivery == "Ships to you"
    assert loc == ""
    assert price == "$75"


def test_location_and_delivery_can_both_be_present():
    _, _, _, title, loc, delivery = _fields_from_spans(
        ["$140", "Kayak", "Seattle, WA", "Ships to you"]
    )
    assert (title, loc, delivery) == ("Kayak", "Seattle, WA", "Ships to you")


def test_delivery_lands_in_attributes():
    items = parse_listings(page([("444444", ["$75", "Kayak", "Ships to you"])]))
    assert items[0].attributes["delivery"] == "Ships to you"
    assert "location" not in items[0].attributes


# ── session gate: anonymous fallback ─────────────────────────────────────────

from types import SimpleNamespace  # noqa: E402

from searchio.config import Settings  # noqa: E402
from searchio.providers.base import ProviderContext  # noqa: E402


class GateFakeSidecar:
    """Blank page until session_reset, listing page afterwards."""

    available = True

    def __init__(self) -> None:
        self.anonymous = False
        self.calls: list[str] = []

    async def goto(self, url, *, tab_id="default"):
        return {"ok": True, "final_url": url}

    async def read_html(self, *, tab_id="default", selector=""):
        if self.anonymous:
            return page([("555555", ["$30", "Used Scooter", "Chattanooga, TN"])])
        return "<html><body><div role='main'></div></body></html>"

    async def eval_js(self, js, *, tab_id="default", timeout=None):
        return 20  # one screenful after any scroll

    async def call(self, verb, params=None, **kw):
        self.calls.append(verb)
        if verb == "session_reset":
            self.anonymous = True
        return {"ok": True}


def gate_ctx(sc) -> ProviderContext:
    return ProviderContext(ladder=SimpleNamespace(sidecar=sc), settings=Settings())


def logged_in_provider() -> FacebookMarketplace:
    prov = FacebookMarketplace()
    prov._session_tried = True
    prov._session_state = "loaded:2693c"
    return prov


async def test_empty_feed_from_session_falls_back_to_anonymous():
    sc = GateFakeSidecar()
    prov = logged_in_provider()
    items = await prov.find_items(Query(text="scooter", k=5), gate_ctx(sc))
    assert [i.title for i in items] == ["Used Scooter"]
    assert "session_reset" in sc.calls  # the session was wiped and bypassed
    assert items[0].meta["session"] == "anonymous(fallback)"
    assert items[0].meta["session_gate"] == "empty_feed"
    assert prov.search_gate == "empty_feed"
    assert prov._session_broken  # later queries skip the session entirely


async def test_working_session_never_triggers_fallback():
    sc = GateFakeSidecar()
    sc.anonymous = True  # first (sessioned) load already returns cards
    prov = logged_in_provider()
    items = await prov.find_items(Query(text="scooter", k=5), gate_ctx(sc))
    assert items
    assert "session_reset" not in sc.calls
    assert all(i.meta["session"] == "loaded:2693c" for i in items)
    assert prov.search_gate == ""


async def test_anonymous_emptiness_is_a_dead_market_not_a_gate():
    sc = GateFakeSidecar()
    sc.anonymous = True
    sc.blank_always = True  # both attempts see nothing
    orig_read = sc.read_html

    async def blank_read(*, tab_id="default", selector=""):
        return "<html><body></body></html>"

    sc.read_html = blank_read
    prov = logged_in_provider()
    items = await prov.find_items(Query(text="nothing-here", k=5), gate_ctx(sc))
    assert items == []
    assert prov.search_gate == "empty_market"


async def test_no_session_never_resets():
    sc = GateFakeSidecar()
    prov = FacebookMarketplace()  # _session_state stays "none"
    items = await prov.find_items(Query(text="scooter", k=5), gate_ctx(sc))
    assert items == []
    assert "session_reset" not in sc.calls


# ── headed preference ────────────────────────────────────────────────────────


async def test_headed_mode_sets_window_mode_once(monkeypatch):
    for var in ("SWARM_BROWSER_WINDOW_MODE", "SWARM_BROWSER_HEADLESS"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(facebook_headed=True)
    prov = FacebookMarketplace()
    prov._prefer_headed(ProviderContext(ladder=SimpleNamespace(), settings=s))
    import os

    assert os.environ["SWARM_BROWSER_WINDOW_MODE"] == "headful"


async def test_explicit_window_mode_beats_the_flag(monkeypatch):
    monkeypatch.setenv("SWARM_BROWSER_WINDOW_MODE", "offscreen")
    s = Settings(facebook_headed=True)
    prov = FacebookMarketplace()
    prov._prefer_headed(ProviderContext(ladder=SimpleNamespace(), settings=s))
    import os

    assert os.environ["SWARM_BROWSER_WINDOW_MODE"] == "offscreen"


# ── streamed Relay payload ───────────────────────────────────────────────────
#
# The fixture is a real anonymous-search document segment: three feed_units
# edges captured from a live F-150 search, trimmed but not invented. Facebook
# ships the feed inside the document as a RelayPrefetchedStreamCache segment,
# which is what makes the listing feed readable without executing anything.

from pathlib import Path  # noqa: E402

from searchio.providers.facebook import (  # noqa: E402
    merge_listings,
    parse_listings_payload,
)

RELAY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "fb-relay-search.html"
).read_text(encoding="utf-8")


def test_relay_payload_yields_structured_listings():
    items = parse_listings_payload(RELAY_FIXTURE)
    assert len(items) == 3
    first = items[0]
    assert first.identity == "fbmp:4277319972580877"
    assert first.url == "https://www.facebook.com/marketplace/item/4277319972580877/"
    assert first.title == "2011 Ford F-150 · XL Pickup 2D 6 1/2 ft"
    assert first.price.amount == 11500
    assert first.attributes["location"] == "San Francisco, CA"
    assert first.attributes["mileage"] == "125K miles"
    assert first.attributes["extraction"] == "relay_payload"
    assert first.attributes["flags"] == "is_live"


def test_relay_payload_gated_feed_is_emptiness_not_error():
    """The session-gate signature: feed_units present, edges null."""
    gated = RELAY_FIXTURE.replace('"edges": [', '"edges": null, "edges_orphaned": [')
    assert parse_listings_payload(gated) == []


def test_relay_payload_dedupes_repeated_batches():
    """Prefetch + main delivery can both carry the feed; ids merge."""
    assert parse_listings_payload(RELAY_FIXTURE + RELAY_FIXTURE) == parse_listings_payload(
        RELAY_FIXTURE
    )


def test_relay_payload_ignores_pages_without_the_feed():
    assert parse_listings_payload("<html><body>nothing here</body></html>") == []


def test_merge_prefers_payload_and_fills_only_gaps_from_dom():
    payload = parse_listings_payload(RELAY_FIXTURE)
    dup_id = payload[0].identity.removeprefix("fbmp:")
    dom_html = page(
        [
            # Same listing as the payload's first edge: DOM title must lose.
            (dup_id, ["$11,500", "Old DOM title", "San Francisco, CA"]),
            # A card the payload never carried: survives the merge.
            ("7000000000000007", ["$9,000", "2012 Ford F-150 XLT", "Oakland, CA"]),
        ]
    )
    dom = parse_listings(dom_html)
    merged = merge_listings(payload, dom)
    assert len(merged) == 4
    by_id = {it.identity: it for it in merged}
    assert by_id[f"fbmp:{dup_id}"].title == "2011 Ford F-150 · XL Pickup 2D 6 1/2 ft"
    assert by_id["fbmp:7000000000000007"].title == "2012 Ford F-150 XLT"


# ── iteration 48 (DeepSeek facebook.py review + probes) ──────────────────────

import asyncio as _asyncio  # noqa: E402
import json as _json  # noqa: E402


def _relay(*listings):
    edges = [{"node": {"listing": l}} for l in listings]
    return "<html><script>" + _json.dumps(
        {"marketplace_search": {"feed_units": {"edges": edges}}}) + "</script></html>"


def _listing(lid, title, amount="1,234", **flags):
    d = {"id": lid, "marketplace_listing_title": title,
         "listing_price": {"formatted_amount": f"${amount}", "amount": amount.replace(",", "")}}
    d.update(flags)
    return d


class TestFlagsBecomeAvailability:
    def test_sold_and_pending_are_not_offered_as_available(self):
        # Bug 84: the payload's is_sold / is_pending flags landed only in
        # attributes["flags"]; Item.availability stayed "" -- unknown -- so
        # an agent (and the find_items tool text) offered a sold listing
        # exactly like a live one.
        items = parse_listings_payload(_relay(
            _listing("1", "Live one", is_live=True),
            _listing("2", "Sold one", is_sold=True),
            _listing("3", "Pending one", is_pending=True, is_live=True)))
        by = {it.attributes["item_id"]: it for it in items}
        assert by["1"].availability == "InStock"
        assert by["2"].availability == "SoldOut"
        assert by["3"].availability == "Pending"


class InterleavingSidecar(GateFakeSidecar):
    """Yields to the loop on every verb so two concurrent searches interleave."""

    async def goto(self, url, *, tab_id="default"):
        await _asyncio.sleep(0)
        return await super().goto(url, tab_id=tab_id)

    async def read_html(self, *, tab_id="default", selector=""):
        await _asyncio.sleep(0)
        return await super().read_html(tab_id=tab_id, selector=selector)

    async def call(self, verb, params=None, **kw):
        await _asyncio.sleep(0)
        return await super().call(verb, params, **kw)


async def test_concurrent_searches_do_not_share_the_gate_decision():
    # Bug 85: the empty-feed gate lived on the provider INSTANCE
    # (self.search_gate, reset at the top of every find_items), so two
    # concurrent searches on the one registry instance raced: B's reset
    # erased A's proven gate before A read it, and A's handoff decision
    # (and its items' provenance) was taken on B's state.
    sc = InterleavingSidecar()
    prov = logged_in_provider()
    ctx = gate_ctx(sc)
    a, b = await _asyncio.gather(
        prov.find_items(Query(text="scooter", k=5), ctx),
        prov.find_items(Query(text="kayak", k=5), ctx),
    )
    # The gate was proven at least once (the first session load was blank,
    # the anonymous retry found the scooter), and every item that came out
    # of an anonymous retry says so -- regardless of what the other call did
    # to the instance in between.
    fell_back = [it for it in a + b if it.meta.get("session") == "anonymous(fallback)"]
    assert fell_back, [it.meta for it in a + b]
    assert all(it.meta.get("session_gate") == "empty_feed" for it in fell_back)
    assert prov.last_gate("scooter") in ("empty_feed", "") and prov.search_gate in ("empty_feed", "")


async def test_session_injection_happens_once_even_under_concurrency():
    # Rider: _ensure_session flipped _session_tried BEFORE its first await,
    # so a concurrent first search skipped past an injection still in flight
    # and ran (and labelled its items) without the session.
    from pathlib import Path as _P

    class SlowSidecar(GateFakeSidecar):
        async def call(self, verb, params=None, **kw):
            if verb == "storage_state_set":
                await _asyncio.sleep(0.05)
                self.calls.append(verb)
                return {"ok": True, "cookies": 3}
            return await super().call(verb, params, **kw)

    sc = SlowSidecar()
    sc.anonymous = True
    prov = FacebookMarketplace()
    sess = _P(__file__).parent / "_fb_session_probe.json"
    sess.write_text(_json.dumps(_FACEBOOK_STATE_MIN), encoding="utf-8")
    try:
        ctx = ProviderContext(ladder=SimpleNamespace(sidecar=sc),
                              settings=Settings(facebook_session=str(sess)))
        a, b = await _asyncio.gather(
            prov.find_items(Query(text="scooter", k=5), ctx),
            prov.find_items(Query(text="kayak", k=5), ctx),
        )
    finally:
        sess.unlink(missing_ok=True)
    assert sc.calls.count("storage_state_set") == 1
    assert all(it.meta["session"] == "loaded:3c" for it in a + b), [it.meta for it in a + b]


_FACEBOOK_STATE_MIN = {"cookies": [{"name": "c_user", "value": "1", "domain": ".facebook.com",
                                    "path": "/", "secure": True, "expires": 1893456000}],
                       "origins": []}


async def test_failed_session_reset_is_not_an_empty_market():
    # Rider: session_reset failing returned [] with search_gate left "", so a
    # gated session whose reset verb died read as "the market is empty".
    class NoReset(GateFakeSidecar):
        async def call(self, verb, params=None, **kw):
            if verb == "session_reset":
                raise RuntimeError("verb unsupported")
            return await super().call(verb, params, **kw)

    sc = NoReset()
    prov = logged_in_provider()
    items = await prov.find_items(Query(text="scooter", k=5), gate_ctx(sc))
    assert items == []
    assert prov.search_gate == "reset_failed", prov.search_gate


def test_free_listing_is_zero_priced_and_the_struck_price_is_the_was():
    # Bug 86 (iteration 48, review candidate 3 + the REAL_CARDS fixture): a
    # free listing renders "Free" then its struck-through "$50". "Free" is
    # not a price token, so the splitter took "$50" as THE price and filed
    # "Free" as a badge -> condition "Free", price $50 -- a free item offered
    # at its old price, on every free listing.
    badges, price, was, title, loc, _ = _fields_from_spans(
        ["Free", "$50", "Old Bikes", "Morristown, NJ"])
    assert (badges, price, was, title, loc) == ([], "Free", "$50", "Old Bikes", "Morristown, NJ")
    items = parse_listings(page([("964857849937122", ["Free", "$50", "Old Bikes", "Morristown, NJ"])]))
    assert items[0].price.amount == 0.0 and items[0].price.raw == "Free"
    assert items[0].attributes.get("was_price") == "$50"
    assert items[0].condition == ""
    # The payload path: formatted "Free" with amount 0 is a zero price too.
    it = parse_listings_payload(_relay({"id": "7", "marketplace_listing_title": "Freebie",
                                        "listing_price": {"formatted_amount": "Free", "amount": "0"}}))[0]
    assert it.price.amount == 0.0
    # A listing TITLED "Free stuff" (no price span) is still a title, not a price.
    _, price, _, title, _, _ = _fields_from_spans(["Free stuff", "Chattanooga, TN"])
    assert price == "" and title == "Free stuff"



# ── iteration 67: Muse second pass ─────────────────────────────────────────


def test_dom_sold_and_pending_badges_set_availability():
    # Bug 149: the payload path learned availability from the flags (bug 84);
    # the DOM path put "Sold"/"Pending" into `condition` and left
    # availability "" -- a sold card parsed from the DOM offered as live.
    items = parse_listings(page([("1", ["Sold", "$450", "Kayak 12ft", "Seattle, WA"]),
                                 ("2", ["Pending", "$1,200", "Ford F-150", "Tacoma, WA"]),
                                 ("3", ["$30", "Used Scooter", "Chattanooga, TN"])]))
    by = {it.url.rsplit("/", 2)[-2]: it for it in items}
    assert by["1"].availability == "SoldOut" and "Sold" not in by["1"].condition
    assert by["2"].availability == "Pending"
    assert by["3"].availability == "InStock"


async def test_dead_market_does_not_retire_the_session(tmp_path):
    # Bug 150: `_session_broken` was set at the TOP of the anonymous retry,
    # before it proved anything, so one query with no listings (an honest
    # empty market) -- or a reset that failed -- retired the logged-in
    # session for the rest of the process.
    sc = GateFakeSidecar()
    sc.anonymous = True

    async def blank_read(*, tab_id="default", selector=""):
        return "<html><body></body></html>"
    sc.read_html = blank_read
    prov = logged_in_provider()
    # The check wipes the browser; the session comes back from its file
    # (bug 162), which is what keeps it active afterwards.
    state_file = tmp_path / "fb.json"
    state_file.write_text('{"cookies": [], "origins": []}')
    prov._session_source = str(state_file)
    await prov.find_items(Query(text="nothing-here", k=5), gate_ctx(sc))
    assert prov.search_gate == "empty_market"
    assert not prov._session_broken, "an empty market is not a broken session"
    assert "storage_state_set" in sc.calls, "the wiped session was put back"
    assert prov._session_active()


async def test_handoff_session_counts_as_active():
    # Rider: after a successful handoff the state was "handoff:..." and
    # `_session_active()` only knew "loaded", so the next empty result was
    # never gate-checked.
    prov = logged_in_provider()
    prov._session_state = "handoff:12c/3o"
    prov._session_broken = False
    assert prov._session_active()


async def test_concurrent_gate_retries_do_not_interleave_the_reset():
    # Rider (concurrency): two searches hitting the gate at once ran
    # session_reset + reload without a lock, so one search's reset wiped
    # the session under the other's in-flight load.
    import asyncio

    sc = GateFakeSidecar()
    order: list[str] = []
    orig_call = sc.call

    async def call(verb, params=None, **kw):
        order.append(f"{verb}:start")
        if verb == "session_reset":
            await asyncio.sleep(0.05)
        r = await orig_call(verb, params, **kw)
        order.append(f"{verb}:end")
        return r
    sc.call = call
    prov = logged_in_provider()
    await asyncio.gather(prov._retry_anonymous(sc, "https://www.facebook.com/marketplace/x", want=5),
                         prov._retry_anonymous(sc, "https://www.facebook.com/marketplace/y", want=5))
    resets = [i for i, e in enumerate(order) if e == "session_reset:start"]
    closes = [i for i, e in enumerate(order) if e == "close_tab:end"]
    assert len(resets) == 2 and closes and resets[1] > closes[0], order
