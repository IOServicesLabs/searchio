"""extract.py second pass (iteration 57): Muse review + parser probes.

Every test here bit RED before its fix. The theme is the same as bug 55's:
a price or an identity that is WRONG is worse than one that is missing,
because merge_items ranks by it and the agent quotes it.
"""

from __future__ import annotations

import pytest

from searchio.extract import (
    _amount,
    identity_for,
    items_from_html,
    parse_price,
    price_from_meta,
    price_from_offer,
    title_of,
)


class TestLoneSeparatorIsADecimalUnlessItGroupsThousands:
    # Bug 107: a lone "." or "," was a decimal ONLY when exactly two digits
    # followed it. "$1.2" read 12, "0.5" read 5, "1,5 EUR" read 15, and the
    # Magento-style "1299.0000" read 12,990,000.
    @pytest.mark.parametrize("raw,want", [
        ("1.2", 1.2), ("0.5", 0.5), ("1,5", 1.5), ("4.5", 4.5),
        ("1299.0000", 1299.0), ("1.23456", 1.23456), ("1234.567", 1234.567),
        # the bug-55 lean survives: one separator, three digits, short head
        ("1.299", 1299.0), ("1,299", 1299.0), ("1.234.567", 1234567.0),
        ("1.299,00", 1299.0), ("1,299.99", 1299.99), ("12,50", 12.5),
    ])
    def test_amounts(self, raw, want):
        assert _amount(raw) == want

    @pytest.mark.parametrize("raw", ["12.3.4.5", "1,,,2", "1.299.00", "1,2.3,4"])
    def test_version_strings_and_garbage_are_not_amounts(self, raw):
        assert _amount(raw) is None

    @pytest.mark.parametrize("raw", ["1 1 1 1", "1 5", "12 3456", "1 22 333"])
    def test_space_groups_must_be_threes(self, raw):
        assert _amount(raw) is None

    @pytest.mark.parametrize("raw,want", [("1 299,00", 1299.0), ("12 345", 12345.0), ("1 234 567.89", 1234567.89)])
    def test_space_groups_in_threes(self, raw, want):
        assert _amount(raw) == want

    def test_price_search_is_bounded_on_long_digit_runs(self):
        # Rider: the unbounded number class made the regex quadratic; a
        # 20 KB run of "1," took a minute.
        import time
        t0 = time.perf_counter()
        parse_price("1," * 20000 + "5")
        parse_price("1" * 20000)
        assert time.perf_counter() - t0 < 2.0

    def test_parse_price_short_decimal(self):
        assert parse_price("only $1.5 each").amount == 1.5


class TestMetaPriceHonoursTheLocale:
    # Bug 108: price_from_meta stripped commas and called float(): the
    # og:price:amount "1.299,00" became 1.299 EUR, "1299,00" became 129900.
    @pytest.mark.parametrize("content,want", [
        ("1.299,00", 1299.0), ("1299,00", 1299.0), ("12,50", 12.5), ("1,299.99", 1299.99),
        ("$12.50", 12.5), ("1299.0000", 1299.0),
    ])
    def test_meta_amounts(self, content, want):
        html = (f'<meta property="og:price:amount" content="{content}">'
                '<meta property="og:price:currency" content="EUR">')
        p = price_from_meta(html)
        assert (p.amount, p.currency) == (want, "EUR")


class TestPrefixedDollarsAreNotUSD:
    # Bug 109: the symbol class was [$ EUR GBP YEN], so "R$ 1.234,56"
    # (Brazilian real), "C$ 19.99" (Canadian) and "A$", "HK$", "NZ$", "S$"
    # all shipped as USD -- a wrong currency, stated confidently.
    @pytest.mark.parametrize("text,amount,cur", [
        ("R$ 1.234,56", 1234.56, "BRL"), ("C$ 19.99", 19.99, "CAD"), ("CA$19.99", 19.99, "CAD"),
        ("A$ 5", 5.0, "AUD"), ("AU$5", 5.0, "AUD"), ("HK$100", 100.0, "HKD"), ("NZ$5", 5.0, "NZD"),
        ("S$5", 5.0, "SGD"), ("US$ 5", 5.0, "USD"), ("MX$ 5", 5.0, "MXN"), ("just $5", 5.0, "USD"),
        ("ABC$5", 5.0, "USD"),
    ])
    def test_dollar_prefixes(self, text, amount, cur):
        p = parse_price(text)
        assert (p.amount, p.currency) == (amount, cur), p

    @pytest.mark.parametrize("text,amount,cur", [
        ("1 299,00 zł", 1299.0, "PLN"), ("1.234 Kč", 1234.0, "CZK"),
        ("₹1,299", 1299.0, "INR"), ("Rs. 1,299", 1299.0, "INR"), ("₩12,000", 12000.0, "KRW"),
        ("12800円", 12800.0, "JPY"), ("₺100", 100.0, "TRY"), ("₽100", 100.0, "RUB"),
        ("฿100", 100.0, "THB"), ("CHF 1'234.50", 1234.5, "CHF"),
    ])
    def test_more_of_the_worlds_currencies(self, text, amount, cur):
        p = parse_price(text)
        assert (p.amount, p.currency) == (amount, cur), p


class TestOfferCurrencyIsAValidCode:
    # Bug 110: priceCurrency as a list became "['U", "US DOLLARS" became "US ".
    @pytest.mark.parametrize("cur,want", [
        (["USD"], "USD"), ("US DOLLARS", "USD"), ("eur", "EUR"), ("", "USD"), (None, "USD"), (["GBP", "USD"], "GBP"),
    ])
    def test_currency(self, cur, want):
        assert price_from_offer({"price": "10", "priceCurrency": cur}).currency == want

    def test_price_as_a_list_takes_the_first(self):
        assert price_from_offer({"price": ["10", "20"], "priceCurrency": "USD"}).amount == 10.0


class TestTitleOfIsThePageTitle:
    # Bug 111: title_of never unescaped ("A &amp; B" reached the agent
    # verbatim), and the FIRST <title> in the document won even when it was
    # an inline SVG icon's or a string inside a <script>.
    def test_entities_unescaped(self):
        assert title_of("<title>A &amp; B &#39;c&#39;</title>") == "A & B 'c'"

    def test_svg_title_does_not_win(self):
        assert title_of("<svg><title>icon</title></svg><title>Real</title>") == "Real"

    def test_script_string_does_not_win(self):
        assert title_of("<script>var t='<title>fake</title>'</script><title>Real</title>") == "Real"

    def test_still_the_first_real_title(self):
        assert title_of("<head><title>Real</title></head><body><title>later</title>") == "Real"


class TestGtinIsCheckDigitValid:
    # Bug 112: any 8-14 digit string was a GTIN identity -- all-zero
    # placeholders and truncated SKU numbers merged unrelated products.
    @pytest.mark.parametrize("g", ["4006381333931", "012345678905", "96385074", "10614141000415"])
    def test_real_gtins_accepted(self, g):
        assert identity_for(gtin=g) == f"gtin:{g}"

    @pytest.mark.parametrize("g", ["4006381333932", "012345678901", "96385075", "00000000000000", "1111111111111"])
    def test_invalid_check_digit_is_not_an_identity(self, g):
        assert not identity_for(gtin=g).startswith("gtin:")


class TestVariantTokensSplitTheModelKey:
    # Bug 113 (Muse #5): "Galaxy S23" and "Galaxy S23 Ultra" both keyed
    # model:s23, so merge_items quoted the base phone's price for the Ultra.
    def test_base_and_ultra_differ(self):
        base = identity_for(brand="Samsung", title="Samsung Galaxy S23 256GB")
        ultra = identity_for(brand="Samsung", title="Samsung Galaxy S23 Ultra 256GB")
        assert base != ultra and base == "model:s23"

    def test_same_variant_agrees_across_sites(self):
        a = identity_for(title="Samsung Galaxy S23 Ultra 5G Phantom Black")
        b = identity_for(title="Galaxy S23 Ultra - 256GB, Green (Samsung)")
        assert a == b

    def test_plain_model_codes_unchanged(self):
        assert identity_for(title="Sony WH-1000XM5 Headphones Black") == "model:1000xm5"


class TestItemUrlsAreAbsoluteAndOwn:
    def _html(self, nodes: str) -> str:
        return f'<script type="application/ld+json">{nodes}</script>'

    def test_relative_offer_url_is_resolved(self):
        # Bug 114 (Muse #6): a relative offers.url shipped as-is; the agent
        # cited "/p/123".
        h = self._html('{"@type":"Product","name":"X","offers":{"url":"/p/123","price":"5","priceCurrency":"USD"}}')
        assert items_from_html(h, "https://shop.example/cat/")[0].url == "https://shop.example/p/123"

    def test_non_http_offer_url_falls_back_to_the_page(self):
        h = self._html('{"@type":"Product","name":"X","offers":{"url":"javascript:alert(1)"}}')
        assert items_from_html(h, "https://shop.example/x")[0].url == "https://shop.example/x"

    def test_list_url_takes_the_first(self):
        h = self._html('{"@type":"Product","name":"X","url":["https://shop.example/a","https://shop.example/b"]}')
        assert items_from_html(h, "https://shop.example/x")[0].url == "https://shop.example/a"

    def test_page_asin_is_not_inherited_by_every_node(self):
        # Bug 115 (Muse #1): identity_for(url=PAGE) for every node on a
        # /dp/ page gave all GTIN-less products the page's ASIN.
        h = self._html('[{"@type":"Product","name":"Main thing","url":"https://m.example/dp/B0AAAAAAAA"},'
                       '{"@type":"Product","name":"Other thing","url":"https://m.example/dp/B0BBBBBBBB"},'
                       '{"@type":"Product","name":"Third thing"}]')
        ids = [i.identity for i in items_from_html(h, "https://m.example/dp/B0AAAAAAAA")]
        assert ids[0] == "asin:B0AAAAAAAA" and ids[1] == "asin:B0BBBBBBBB" and ids[2] != ids[0]

    def test_single_node_still_uses_the_page_asin(self):
        h = self._html('{"@type":"Product","name":"Main thing"}')
        assert items_from_html(h, "https://m.example/dp/B0AAAAAAAA")[0].identity == "asin:B0AAAAAAAA"

    def test_items_keep_document_order(self):
        # Rider: _iter_ld popped a LIFO stack, so an ItemList of products
        # came out last-first.
        h = self._html('[{"@type":"Product","name":"first"},{"@type":"Product","name":"second"},'
                       '{"@type":"Product","name":"third"}]')
        assert [i.title for i in items_from_html(h, "https://x.example/")] == ["first", "second", "third"]

