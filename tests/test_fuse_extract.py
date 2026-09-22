"""URL canonicalization, fusion, identity resolution, and price parsing."""

from __future__ import annotations

from searchio.extract import (identity_for, items_from_html, parse_price,
                              price_from_meta, price_from_offer)
from searchio.fuse import canonical_url, dedupe_docs, fuse, jaccard, merge_items
from searchio.models import Doc, Item, Price


class TestCanonicalUrl:
    def test_strips_tracking_params(self):
        a = canonical_url("https://x.com/p?utm_source=t&utm_campaign=c&id=5")
        assert a == canonical_url("https://x.com/p?id=5")

    def test_keeps_meaningful_params(self):
        # ?p=2 selects different content; merging it away loses a page.
        assert canonical_url("https://x.com/a?p=2") != canonical_url("https://x.com/a?p=3")

    def test_normalizes_host_and_scheme(self):
        assert canonical_url("http://WWW.X.com/a/") == canonical_url("https://x.com/a")

    def test_amazon_collapses_to_asin(self):
        a = canonical_url("https://www.amazon.com/Sony-Wireless-Headphones/dp/B09XS7JWHH/ref=sr_1_1")
        b = canonical_url("https://amazon.com/dp/B09XS7JWHH")
        assert a == b == "https://amazon.com/dp/B09XS7JWHH"

    def test_amazon_international_collapses_to_asin(self):
        # Bug 46 (iteration 34, DeepSeek fuse.py review): the ASIN branch
        # matched only ``endswith("amazon.com") or ".amazon." in host``, so
        # amazon.co.uk / amazon.de / amazon.co.jp never canonicalized to
        # their ASIN -- the same product under different slug paths kept
        # different keys and duplicate listings survived dedup/merge.
        a = canonical_url("https://www.amazon.co.uk/Sony-Wireless/dp/B09XS7JWHH/ref=sr_1_1")
        b = canonical_url("https://amazon.co.uk/dp/B09XS7JWHH")
        assert a == b == "https://amazon.co.uk/dp/B09XS7JWHH"

    def test_non_amazon_lookalike_is_not_treated_as_amazon(self):
        # The fix must not false-positive on a host that merely contains the
        # substring "amazon": notamazon.com keeps its slug path.
        na = canonical_url("https://notamazon.com/Some-Slug/dp/B09XS7JWHH/ref=x")
        assert na != "https://notamazon.com/dp/B09XS7JWHH"
        assert "some-slug" in na.lower()


class TestDedupe:
    def test_merges_same_url(self):
        docs = [
            Doc(url="https://x.com/a?utm_source=q", title="A", source="p1", rank=0),
            Doc(url="https://x.com/a", title="A", snippet="body", source="p2", rank=1),
        ]
        out = dedupe_docs(docs)
        assert len(out) == 1 and out[0].snippet == "body"

    def test_detects_syndication_across_domains(self):
        text = "Regulators approved the merger after a lengthy review process today"
        docs = [
            Doc(url="https://a.com/1", title=text, snippet="x", source="p", rank=0),
            Doc(url="https://b.com/2", title=text, snippet="x", source="p", rank=1),
        ]
        assert len(dedupe_docs(docs)) == 1

    def test_keeps_distinct_pages_on_one_domain(self):
        docs = [
            Doc(url="https://a.com/1", title="Guide to postgres vacuum", source="p", rank=0),
            Doc(url="https://a.com/2", title="Guide to postgres vacuum", source="p", rank=1),
        ]
        assert len(dedupe_docs(docs)) == 2


    def test_urlless_docs_do_not_all_collapse_to_one(self):
        # Bug 179 (fuse.py third pass): dedupe keyed on canonical_url(""), so
        # every doc with no URL got the same key "" and distinct urlless docs
        # merged into one franken-record (title from one, snippet from
        # another). merge_items already keeps urlless rows separate by id();
        # dedupe now matches -- an empty URL is not a shared identity.
        docs = [
            Doc(url="", title="First thing", snippet="a", source="p", rank=0),
            Doc(url="", title="Second thing", snippet="b", source="p", rank=1),
        ]
        out = dedupe_docs(docs)
        assert len(out) == 2, [d.title for d in out]
        assert {d.title for d in out} == {"First thing", "Second thing"}

class TestFuse:
    def test_agreement_beats_single_provider_top_rank(self):
        lists = {
            "p1": [Doc(url="https://solo.com/x", title="Solo", source="p1", rank=0),
                   Doc(url="https://both.com/y", title="Both", source="p1", rank=1)],
            "p2": [Doc(url="https://other.com/z", title="Other", source="p2", rank=0),
                   Doc(url="https://both.com/y", title="Both", source="p2", rank=1)],
        }
        out = fuse(lists)
        assert out[0].url == "https://both.com/y"
        assert out[0].meta["agreement"] == 2

    def test_weights_shift_ranking(self):
        lists = {
            "a": [Doc(url="https://a.com/1", title="A", source="a", rank=0)],
            "b": [Doc(url="https://b.com/1", title="B", source="b", rank=0)],
        }
        out = fuse(lists, weights={"b": 5.0})
        assert out[0].url == "https://b.com/1"

    def test_winner_inherits_the_date_a_dated_dupe_carried(self):
        # Cross-provider dupes keep the best-ranked copy; the date the other
        # copy carried must survive the pick, or dated news loses its date
        # whenever an undated engine ranks the same URL higher.
        lists = {
            "duckduckgo": [Doc(url="https://n.com/story", title="S", source="duckduckgo", rank=0)],
            "bing": [Doc(url="https://n.com/story", title="S", source="bing", rank=2,
                          published="2026-09-08")],
        }
        out = fuse(lists)
        assert out[0].published == "2026-09-08"

    def test_dedupe_merges_published(self):
        docs = [
            Doc(url="https://x.com/a", title="A", source="p1", rank=0),
            Doc(url="https://x.com/a", title="A", source="p2", rank=1,
                published="2026-09-07"),
        ]
        out = dedupe_docs(docs)
        assert len(out) == 1 and out[0].published == "2026-09-07"

    def test_cross_provider_syndication_is_collapsed(self):
        # Bug 45 (iteration 34, DeepSeek fuse.py review): dedupe_docs catches
        # syndication WITHIN one provider's list, but fuse() ran it
        # per-provider only and then keyed rrf by canonical_url -- so the same
        # wire story arriving from TWO providers under different URLs (the
        # exact case the module docstring promises to handle) survived as two
        # rows. The near-dup collapse now also runs on the fused ranking.
        text = "Regulators approved the merger after a lengthy review process today"
        lists = {
            "duckduckgo": [Doc(url="https://nytimes.com/a", title=text,
                               snippet="x", source="duckduckgo", rank=0)],
            "bing": [Doc(url="https://cnn.com/b", title=text,
                         snippet="x", source="bing", rank=0)],
        }
        out = fuse(lists)
        assert len(out) == 1, [d.url for d in out]

    def test_distinct_topics_not_collapsed_cross_provider(self):
        # The cross-provider collapse must not OVER-merge: two genuinely
        # different stories keep their own rows (guards the 0.82 threshold).
        lists = {
            "duckduckgo": [Doc(url="https://a.com/1",
                               title="Postgres 17 released with a rewritten vacuum engine",
                               snippet="database", source="duckduckgo", rank=0)],
            "bing": [Doc(url="https://b.com/2",
                         title="Apple unveils the iPhone 18 lineup at its fall event",
                         snippet="phone", source="bing", rank=0)],
        }
        out = fuse(lists)
        assert len(out) == 2, [d.url for d in out]


class TestIdentity:
    def test_gtin_wins(self):
        assert identity_for(gtin="0027242919655", title="whatever") == "gtin:0027242919655"

    def test_asin_from_url(self):
        assert identity_for(url="https://amazon.com/dp/B09XS7JWHH", title="x") == "asin:B09XS7JWHH"

    def test_model_code_collapses_titles_across_sites(self):
        # The case that broke merging in practice: same headphones, three sites,
        # three very different titles.
        titles = [
            "Amazon.com: Sony WH-1000XM5 Premium Noise Cancelling Wireless Headphones",
            "Sony WH-1000XM5 The Best Wireless Noise Canceling Headphones, Black",
            "Sony WH-1000XM5 Bluetooth Wireless Noise-Canceling Headphones - Silver",
        ]
        ids = {identity_for(title=t) for t in titles}
        assert ids == {"model:1000xm5"}

    def test_different_models_stay_distinct(self):
        assert identity_for(title="Sony WH-1000XM4") != identity_for(title="Sony WH-1000XM5")


class TestPrice:
    def test_symbol_prefix(self):
        p = parse_price("Only $1,299.99 today")
        assert p.amount == 1299.99 and p.currency == "USD"

    def test_currency_suffix(self):
        assert parse_price("349.00 EUR").currency == "EUR"

    def test_no_price_returns_empty_not_garbage(self):
        # Regression: this used to stuff the whole snippet into `raw`, so a
        # listing rendered its description in the price column.
        p = parse_price("Experience superb noise cancellation with these headphones")
        assert p.amount is None and p.raw == ""

    def test_a_zero_offer_price_is_free_not_the_list_price(self):
        # Bug 186 (DeepSeek extract review, iteration 91): the price `or` chain
        # treated a numeric 0 as absent and fell through to highPrice, so a
        # free product (`price: 0`) was quoted at its $250 list price -- and a
        # string "0"/"0.00" was kept, so the chain disagreed with itself by
        # JSON type. A present 0 is $0.
        assert price_from_offer({"price": 0, "highPrice": 250}).amount == 0.0
        assert price_from_offer({"price": 0.0, "lowPrice": 199}).amount == 0.0
        assert price_from_offer({"price": "0.00"}).amount == 0.0
        # The fallback chain still fills in from lowPrice/highPrice when there
        # is genuinely no primary price.
        assert price_from_offer({"lowPrice": 199, "highPrice": 250}).amount == 199.0
        assert price_from_offer({"highPrice": 250}).amount == 250.0
        assert price_from_offer({}).amount is None

    def test_meta_price(self):
        html = '<meta property="og:price:amount" content="249.99"><meta property="og:price:currency" content="USD">'
        assert price_from_meta(html).amount == 249.99


class TestJsonLd:
    def test_product_with_offer(self):
        html = """
        <script type="application/ld+json">
        {"@type":"Product","name":"Test Widget","brand":{"name":"Acme"},
         "gtin13":"0123456789012",
         "aggregateRating":{"ratingValue":4.5,"reviewCount":120},
         "offers":{"@type":"Offer","price":"99.95","priceCurrency":"USD",
                   "availability":"https://schema.org/InStock"}}
        </script>"""
        items = items_from_html(html, "https://shop.example/p/1")
        assert len(items) == 1
        it = items[0]
        assert it.title == "Test Widget"
        assert it.price.amount == 99.95
        assert it.rating == 4.5 and it.reviews == 120
        assert it.availability == "InStock"
        assert it.identity == "gtin:0123456789012"

    def test_price_specification_nesting(self):
        html = """
        <script type="application/ld+json">
        {"@type":"Product","name":"Nested",
         "offers":{"priceSpecification":{"price":"42.00","priceCurrency":"GBP"}}}
        </script>"""
        items = items_from_html(html, "https://x.com/p")
        assert items[0].price.amount == 42.0 and items[0].price.currency == "GBP"


class TestMergeItems:
    def test_collapses_by_identity_and_keeps_offers(self):
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=210.0),
                 seller="a.com", identity="model:1000xm5"),
            Item(title="X", url="https://b.com/2", price=Price(amount=196.0),
                 seller="b.com", identity="model:1000xm5", rating=4.3, reviews=1415),
            Item(title="X", url="https://c.com/3", price=Price(amount=198.0),
                 seller="c.com", identity="model:1000xm5"),
        ]
        out = merge_items(raw)
        assert len(out) == 1
        assert out[0].price.amount == 196.0  # cheapest wins
        assert out[0].meta["offer_count"] == 3
        assert len(out[0].meta["offers"]) == 2

    def test_fills_missing_fields_from_siblings(self):
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=10.0),
                 identity="i", brand=""),
            Item(title="X", url="https://b.com/2", price=Price(amount=20.0),
                 identity="i", brand="Acme", rating=4.0),
        ]
        out = merge_items(raw)
        assert out[0].brand == "Acme" and out[0].rating == 4.0

    def test_same_listing_twice_never_becomes_its_own_offer(self):
        # Sponsored + organic slots of one listing share a URL; without a
        # per-URL collapse inside the identity group, the loser lands in
        # meta['offers'] and the row offers ITSELF as another seller.
        raw = [
            Item(title="X", url="https://a.com/1?ref=ads", price=Price(amount=210.0),
                 seller="a.com", identity="model:1000xm5"),
            Item(title="X", url="https://a.com/1", price=Price(amount=199.0),
                 seller="a.com", identity="model:1000xm5", rating=4.3),
            Item(title="X", url="https://b.com/2", price=Price(amount=205.0),
                 seller="b.com", identity="model:1000xm5"),
        ]
        out = merge_items(raw)
        assert len(out) == 1
        assert out[0].meta["offer_count"] == 2  # two DISTINCT listings
        offer_urls = [o["url"] for o in out[0].meta["offers"]]
        assert out[0].url not in offer_urls
        # The richer duplicate (rating) was kept as the group representative.
        assert out[0].rating == 4.3

    def test_final_sort_does_not_compare_amounts_across_currencies(self):
        # Bug 178 (iteration 81, fuse.py third pass): the winner pick is
        # currency-aware ("cheapest only means something within one currency"),
        # but the FINAL sort ordered every row by raw price.amount regardless
        # of currency -- the same defect the winner pick was fixed for, one
        # level up. A EUR 5 row interleaved between two USD rows by raw number.
        raw = [
            Item(title="A", url="https://a.com/1", price=Price(amount=3.0, currency="USD"), identity="a"),
            Item(title="B", url="https://b.com/2", price=Price(amount=10.0, currency="USD"), identity="b"),
            Item(title="C", url="https://c.com/3", price=Price(amount=5.0, currency="EUR"), identity="c"),
        ]
        out = merge_items(raw)
        titles = [i.title for i in out]
        # The two USD rows must stay a contiguous cheapest-first block; the
        # single EUR row must not split them by its raw amount (5 between 3/10).
        assert titles.index("B") == titles.index("A") + 1, titles
        # The dominant currency (USD, 2 rows) leads; EUR (1 row) follows.
        assert titles == ["A", "B", "C"], titles

    def test_single_currency_sort_is_unchanged(self):
        raw = [
            Item(title="hi", url="https://a.com/1", price=Price(amount=210.0, currency="USD"), identity="a"),
            Item(title="lo", url="https://b.com/2", price=Price(amount=196.0, currency="USD"), identity="b"),
            Item(title="none", url="https://c.com/3", identity="c"),  # unpriced last
        ]
        out = merge_items(raw)
        assert [i.title for i in out] == ["lo", "hi", "none"]

    def test_loners_never_merge(self):
        raw = [
            Item(title="kayak red", url="https://a.com/1", identity=""),
            Item(title="kayak red", url="https://b.com/2", identity=""),
        ]
        assert len(merge_items(raw)) == 2

    def test_unpriced_group_does_not_crash(self):
        raw = [
            Item(title="X", url="https://a.com/1", identity="i"),
            Item(title="X", url="https://b.com/2", identity="i"),
        ]
        out = merge_items(raw)
        assert len(out) == 1 and out[0].meta["offer_count"] == 2

    def test_cheapest_stale_offer_does_not_mask_live_ones(self):
        # The F-150 re-test merge, generalized: three offers collapsed by a
        # weak identity, cheapest sold -- the row used to read "$10,000,
        # OutOfStock" and the live pricier offers hid in meta['offers'].
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=10000.0),
                 identity="model:fx4", availability="OutOfStock"),
            Item(title="X", url="https://b.com/2", price=Price(amount=35000.0),
                 identity="model:fx4", availability="InStock"),
            Item(title="X", url="https://c.com/3", price=Price(amount=34500.0),
                 identity="model:fx4", availability="InStock"),
        ]
        out = merge_items(raw)
        assert len(out) == 1
        assert out[0].price.amount == 34500.0  # cheapest LIVE offer wins
        assert out[0].availability == "InStock"
        stale = [o for o in out[0].meta["offers"] if o["price"] == 10000.0]
        assert stale and stale[0]["availability"] == "OutOfStock"

    def test_all_stale_group_still_quotes_cheapest(self):
        # No live offer: the cheapest stale quote still represents the group,
        # honestly marked -- dropping the row would hide the product entirely.
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=10.0),
                 identity="i", availability="OutOfStock"),
            Item(title="X", url="https://b.com/2", price=Price(amount=12.0),
                 identity="i", availability="SoldOut"),
        ]
        out = merge_items(raw)
        assert len(out) == 1 and out[0].price.amount == 10.0

    def test_unknown_availability_counts_as_live(self):
        # An empty availability is "unknown", not stale: demoting it would
        # bury the majority of listings whose retailer emits no availability
        # at all.
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=10.0),
                 identity="i", availability=""),
            Item(title="X", url="https://b.com/2", price=Price(amount=35.0),
                 identity="i", availability="InStock"),
        ]
        out = merge_items(raw)
        assert out[0].price.amount == 10.0

    def test_offers_carry_availability_and_condition(self):
        # The alternative offers keep their own stock/condition state, so an
        # agent can pick a live offer out of the group (the F-150 re-test's
        # "$35k, in stock" hid in this list with no way to tell).
        raw = [
            Item(title="X", url="https://a.com/1", price=Price(amount=10.0),
                 identity="i", availability="InStock"),
            Item(title="X", url="https://b.com/2", price=Price(amount=20.0),
                 identity="i", availability="OutOfStock", condition="UsedCondition"),
        ]
        out = merge_items(raw)
        offer = out[0].meta["offers"][0]
        assert offer["availability"] == "OutOfStock"
        assert offer["condition"] == "UsedCondition"


def test_jaccard_similarity():
    assert jaccard("the quick brown fox jumps", "the quick brown fox jumps") == 1.0
    assert jaccard("totally different words here", "nothing alike whatsoever friend") < 0.2


class TestShortModelCodes:
    """Model codes are often only two characters, and titles disagree on which
    token is longest. Both were real merge failures."""

    def test_two_character_model_code_merges(self):
        titles = [
            "Logitech MX Master 3S, Wireless Performance Mouse, Ergo, 8K DPI, Quiet",
            "Logitech MX Master 3S - Performance Wireless Mouse with Ultra-fast Scrolling",
            "Logitech Master Series MX Master 3S Performance Wireless Mouse, USB-a",
        ]
        assert {identity_for(title=t) for t in titles} == {"model:3s"}

    def test_first_model_token_wins_not_longest(self):
        # "8K DPI" is a spec, not the model. Taking the longest token would make
        # these two titles disagree.
        a = identity_for(title="Logitech MX Master 3S, Mouse, 8K DPI")
        b = identity_for(title="Logitech MX Master 3S, Mouse")
        assert a == b == "model:3s"


class TestPricePrecision:
    """A wrong price is worse than no price.

    An unanchored `"price": N` scan over a marketplace page matched a
    related-items carousel and reported $23.00 for a pair of $400 headphones.
    Only declared price fields count now.
    """

    def test_unrelated_inline_json_price_is_ignored(self):
        html = (
            '<html><head><title>Sony WH-1000XM5</title></head><body>'
            '<script>window.__RELATED__={"items":[{"name":"Carrying Case","price":23.00}]}</script>'
            "</body></html>"
        )
        assert price_from_meta(html).amount is None

    def test_declared_meta_price_still_works(self):
        html = '<meta property="og:price:amount" content="399.99">'
        assert price_from_meta(html).amount == 399.99

    def test_itemprop_price_still_works(self):
        html = '<span itemprop="price" content="349.00">$349</span>'
        assert price_from_meta(html).amount == 349.00

    def test_the_products_own_json_ld_still_wins(self):
        # JSON-LD is a declared statement about *this* product, so it is
        # unaffected by the meta tightening above.
        html = """
        <script type="application/ld+json">
        {"@type":"Product","name":"Sony WH-1000XM5",
         "offers":{"price":"398.00","priceCurrency":"USD"}}
        </script>
        <script>var related={"price":23.00}</script>"""
        items = items_from_html(html, "https://shop.example/p")
        assert items[0].price.amount == 398.00


class TestPriceLocales:
    """Bug 55 (iteration 39, DeepSeek extract.py review): _PRICE_RE knew only
    comma-thousands / dot-decimal, so a European price was read WRONG, not
    skipped -- "EUR 1.299,00" became 1.29 (the ".29" matched as a decimal),
    "1 299,00 EUR" became 29900.0, and "1.299,00 EUR"/"JPY 1,200" became
    None. merge_items picks the CHEAPEST offer as the representative row, so
    a EUR1,299 item mis-parsed to EUR1.29 would WIN the merge and be quoted
    -- while satisfying the "winner never pricier than its offers"
    invariant. The decimal separator is the LAST of . or , when both appear;
    a lone comma followed by exactly two digits at the end is a decimal
    comma; spaces and thin spaces are thousands separators.
    """

    def test_european_formats_parse_to_the_right_amount(self):
        assert parse_price("\u20ac1.299,00").amount == 1299.0
        assert parse_price("1.299,00 \u20ac").amount == 1299.0
        assert parse_price("1 299,00 EUR").amount == 1299.0
        assert parse_price("EUR 1.299,00").amount == 1299.0
        assert parse_price("12,50 \u20ac").amount == 12.5

    def test_us_formats_still_parse(self):
        assert parse_price("$1,299.99").amount == 1299.99
        assert parse_price("$1299").amount == 1299.0
        assert parse_price("1,299.99 USD").amount == 1299.99
        assert parse_price("\u00a312.50").amount == 12.5

    def test_currency_codes_beyond_the_big_four(self):
        p = parse_price("JPY 1,200")
        assert p.amount == 1200.0 and p.currency == "JPY"
        p = parse_price("CHF 49.90")
        assert p.amount == 49.9 and p.currency == "CHF"


class TestIdentityStrongKeys:
    """Bug 56 (iteration 39): items_from_html fed ``mpn or sku`` into
    identity_for's gtin= slot, and anything >= 8 alphanumerics was accepted
    as a GTIN -- the STRONGEST cross-site join key. A retailer-internal SKU
    ("ITEM-0001") is not globally unique, so two different products from two
    sellers merged into one row. A GTIN is 8-14 DIGITS; an MPN is a weaker,
    manufacturer-scoped key; a SKU is no cross-site key at all.
    """

    def test_non_numeric_sku_is_not_a_gtin(self):
        a = identity_for(gtin="ABC-123-XYZ", title="Widget alpha model")
        b = identity_for(gtin="ABC-123-XYZ", title="Gadget beta thing")
        assert not a.startswith("gtin:") and not b.startswith("gtin:")
        assert a != b, "two different products must never share a key"

    def test_numeric_gtin_still_strongest(self):
        assert identity_for(gtin="0027242919655", title="whatever") == "gtin:0027242919655"
        assert identity_for(gtin="4006381333931", title="x") == identity_for(gtin="4006381333931", title="y")

    def test_mpn_is_its_own_scoped_key(self):
        a = identity_for(mpn="WH1000XM5/B", title="Sony headphones black")
        b = identity_for(mpn="WH1000XM5/B", title="Sony WH-1000XM5 the best")
        assert a == b and a.startswith("mpn:")


class TestIdentityDifferentProductsStaySplit:
    """Bug 57 (iteration 39): identity_for merged DIFFERENT products two ways.
    (a) A spec token with letters+digits ("128gb") satisfied _MODEL_RE and,
    being the first such token in title order, became the model key -- so
    "iPhone 15 Pro 128GB" and "iPhone 15 Pro MAX 128GB" both hashed to
    model:128gb and merge_items quoted the cheaper Pro's price for the Pro
    Max. (b) The slug fallback dropped tokens shorter than 3 chars, so the
    generation number "15" vanished and iPhone 14 Pro / 15 Pro collided.
    Unit/spec tokens are now excluded from model candidates, and numeric
    tokens survive into the slug.
    """

    def test_spec_token_is_not_the_model_and_variants_stay_apart(self):
        pro = identity_for(brand="Apple", title="iPhone 15 Pro 128GB Blue")
        pro_max = identity_for(brand="Apple", title="iPhone 15 Pro Max 128GB Blue")
        assert pro != pro_max, (pro, pro_max)
        assert "128gb" not in pro.split(":", 1)[0] and not pro.startswith("model:128gb")

    def test_generation_number_survives_into_the_slug(self):
        i14 = identity_for(brand="Apple", title="iPhone 14 Pro 128GB")
        i15 = identity_for(brand="Apple", title="iPhone 15 Pro 128GB")
        assert i14 != i15, (i14, i15)

    def test_real_model_codes_still_collapse(self):
        # The docstring's own cases: the model token wins over spec noise.
        a = identity_for(title="Logitech MX Master 3S, Wireless Mouse, 8K DPI")
        b = identity_for(title="Logitech MX Master 3S - Performance Wireless Mouse")
        assert a == b == "model:3s"
        assert identity_for(title="Sony WH-1000XM5 Black") == identity_for(title="Sony WH-1000XM5 Silver")


class TestItemsFromHtmlRobustness:
    def test_list_shaped_rating_does_not_abort_the_page(self):
        # Bug 58 (iteration 39): schema.org allows arrays anywhere; a listing
        # whose aggregateRating was a LIST crashed items_from_html with
        # AttributeError and the WHOLE page yielded zero items.
        html = ('<script type="application/ld+json">{"@type":"Product","name":"Widget",'
                '"offers":{"price":"9.99","priceCurrency":"USD"},'
                '"aggregateRating":[{"ratingValue":"4.5","reviewCount":"12"}]}</script>')
        items = items_from_html(html, "https://shop.example/w")
        assert [(i.title, i.price.amount) for i in items] == [("Widget", 9.99)]
        assert items[0].rating == 4.5 and items[0].reviews == 12

    def test_page_price_fallback_only_for_a_single_product_page(self):
        # Bug 59 (iteration 39): the page-global og:price fallback applied to
        # EVERY priceless Product node, so a category/ItemList page stamped
        # its one meta price on all of its products. The fallback is the
        # single-product-page case only.
        meta = ('<meta property="og:price:amount" content="9.99">'
                '<meta property="og:price:currency" content="USD">')
        one = meta + ('<script type="application/ld+json">{"@type":"Product","name":"Solo",'
                      '"url":"https://s.example/solo"}</script>')
        assert [i.price.amount for i in items_from_html(one, "https://s.example/solo")] == [9.99]
        many = meta + "".join(
            '<script type="application/ld+json">{"@type":"Product","name":"P%d",'
            '"url":"https://s.example/p%d"}</script>' % (i, i) for i in range(3))
        assert [i.price.amount for i in items_from_html(many, "https://s.example/list")] == [None, None, None]


def test_cheapest_offer_is_chosen_within_the_majority_currency():
    # Iteration 53 rider (probe): merge_items took min(price.amount) across
    # currencies, so a 1 EUR offer (a deposit, a parse slip) beat 250 USD and
    # became the representative row's price. Compare within the currency most
    # offers share; the others stay listed under meta["offers"].
    from searchio.fuse import merge_items
    from searchio.models import Item, Price

    items = [Item(title="S", url="https://a.com/1", price=Price(amount=299.0, currency="USD"), identity="s"),
             Item(title="S", url="https://b.com/2", price=Price(amount=1.0, currency="EUR"), identity="s"),
             Item(title="S", url="https://c.com/3", price=Price(amount=250.0, currency="USD"), identity="s")]
    m = merge_items(items)[0]
    assert m.url == "https://c.com/3" and (m.price.amount, m.price.currency) == (250.0, "USD")
    assert {o["url"] for o in m.meta["offers"]} == {"https://a.com/1", "https://b.com/2"}

