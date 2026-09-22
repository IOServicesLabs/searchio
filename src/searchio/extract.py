"""Turning fetched bytes into something a model can read, and items it can compare.

Two jobs that look similar and are not:

* :func:`to_markdown` -- article extraction. Strip chrome, keep prose. This is
  what goes in a worker's context, so the win is measured in tokens: a 3 MB
  Amazon page is ~750k tokens of mostly navigation, and the part worth reading
  is maybe 2k.
* :func:`items_from_html` -- structured listing extraction, via schema.org
  JSON-LD. Prose extraction actively destroys this; a price is not prose.

JSON-LD first because it is the site telling us what the page means, in a
format it maintains for Google's sake and therefore keeps accurate. Guessing
from the DOM is the fallback, and for anything harder than that the SwarmIO
sidecar's ``extract_listings`` does generic DOM-repetition analysis far better
than a regex pass here would.
"""

from __future__ import annotations

import json
import html as html_mod
import re
import urllib.parse
from typing import Any

from .models import Doc, Item, Price

_WS = re.compile(r"\n{3,}")


def to_markdown(html: str, url: str = "") -> str:
    """Extract the readable article body as markdown.

    trafilatura when available -- it is the best-in-class boilerplate remover --
    with a crude tag strip as a floor so this never returns nothing just because
    an optional dependency is missing.
    """
    try:
        import trafilatura

        out = trafilatura.extract(
            html,
            url=url or None,
            output_format="markdown",
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_recall=True,
        )
        if out and out.strip():
            return _WS.sub("\n\n", out.strip())
    except Exception:
        pass

    from .net.blocks import visible_text

    return visible_text(html)[:20000]


#: <title> elements that are NOT the page title: an inline SVG icon's
#: accessible name, or the string "<title>" inside a script/style body.
_TITLE_NOISE = re.compile(r"<(script|style|svg)\b.*?</\1\s*>", re.I | re.S)


def title_of(html: str) -> str:
    """The document title, entity-decoded (bug 111).

    The first <title> in the byte stream is not always the page's: pages put
    inline SVG logos (with their own <title>) and scripts (whose strings can
    spell "<title>") ahead of the real one, and the old regex took whichever
    came first -- and shipped "A &amp; B" to the agent undecoded.
    """
    m = re.search(r"<title[^>]*>(.*?)</title>", _TITLE_NOISE.sub("", html), re.I | re.S)
    if not m:
        return ""
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return html_mod.unescape(text)[:300]


# ── structured data ──────────────────────────────────────────────────────────

_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S
)

#: A number with any mix of thousands/decimal separators (".", ",", space,
#: thin space) -- the LOCALE is decided afterwards by _amount(), not here.
#: Bounded (rider to bug 107): a price is never more than ~20 characters of
#: digits and separators, and the unbounded ``[...]*`` made the search
#: quadratic -- a 20 KB run of "1,1,1," took a minute per parse_price call.
_NUM = r"\d(?:[\d.,\u00a0\u202f '\u2019]{0,22}\d)?"
_CODES = (r"USD|EUR|GBP|CAD|AUD|JPY|CHF|SEK|NOK|DKK|PLN|CZK|INR|CNY|KRW|MXN|BRL"
          r"|HKD|NZD|SGD|TRY|RUB|THB|ZAR|HUF|RON|ILS|AED|TWD|PHP|IDR|MYR|VND")
#: Symbols that go BEFORE the number. Longest first: "US$" must win over
#: "S$" (Singapore), which must win over "$" (bug 109 -- every prefixed
#: dollar, R$ C$ A$ HK$ NZ$ S$ MX$, was read as USD). The lookbehind keeps
#: "ABC$5" from being C$5.
_PRE_SYMS = (r"US\$|CA\$|C\$|AU\$|A\$|HK\$|NZ\$|S\$|MX\$|R\$|\$|\u20ac|\u00a3|\u00a5"
             r"|\u20b9|Rs\.?|\u20a9|\u20ba|\u20bd|\u0e3f")
#: Symbols that go AFTER the number ("1.299,00 \u20ac", "12 z\u0142", "12800\u5186").
_POST_SYMS = r"\$|\u20ac|\u00a3|\u00a5|z\u0142|K\u010d|\u5186|\u20b9|\u20a9|\u20ba|\u20bd|\u0e3f"
_PRICE_RE = re.compile(
    r"(?:(?P<sym>(?<![A-Za-z])(?:" + _PRE_SYMS + r")|[$€£¥₹₩₺₽฿])\s?(?P<a>" + _NUM + r"))"
    r"|(?:(?P<b>" + _NUM + r")\s?(?P<cur>" + _CODES + r")\b)"
    r"|(?:(?<![A-Za-z])(?P<cur2>" + _CODES + r")\s?(?P<c>" + _NUM + r"))"
    r"|(?:(?P<d>" + _NUM + r")\s?(?P<sym2>" + _POST_SYMS + r"))",  # symbol AFTER
    re.I,
)

_CUR_BY_SYM = {
    "$": "USD", "US$": "USD", "CA$": "CAD", "C$": "CAD", "AU$": "AUD", "A$": "AUD",
    "HK$": "HKD", "NZ$": "NZD", "S$": "SGD", "MX$": "MXN", "R$": "BRL",
    "\u20ac": "EUR", "\u00a3": "GBP", "\u00a5": "JPY", "\u5186": "JPY",
    "\u20b9": "INR", "RS": "INR", "RS.": "INR", "\u20a9": "KRW", "\u20ba": "TRY",
    "\u20bd": "RUB", "\u0e3f": "THB", "Z\u0141": "PLN", "K\u010c": "CZK",
}


def _sym_currency(sym: str) -> str:
    return _CUR_BY_SYM.get(sym.upper(), "USD")


def _amount(num: str) -> float | None:
    """Decide the decimal separator, then parse (bug 55).

    The old regex knew only comma-thousands / dot-decimal, so a European
    price was read WRONG rather than skipped: "1.299,00" matched as 1.29 (the
    trailing ".29" taken for a decimal), "1 299,00" as 29900. Rules: when
    both "." and "," appear the LAST one is the decimal separator (and must
    appear once); a lone "," or "." is a THOUSANDS separator only when it
    groups digits in threes behind a head of at most three digits ("1.299",
    "1,234,567") -- otherwise it is the decimal point ("1.2", "0.5", "12,50",
    "1299.0000"; bug 107: the old rule needed exactly two decimals, so $1.5
    read 15 and Magento's "1299.0000" read 12,990,000). Anything else --
    "12.3.4.5" (a version), "1,,,2" -- is not an amount at all.
    """
    parts = re.split(r"[\u00a0\u202f '\u2019]+", num.strip())
    if len(parts) > 1:
        # Space/apostrophe grouping is a THOUSANDS separator or nothing: the
        # groups must be threes behind a short head ("1 299,00", "12 345",
        # "1'234.50"); "1 1 1 1" is not an amount (it read 1111).
        head, *mid, last = parts
        m = re.fullmatch(r"(\d{3})(?:[.,](\d{1,4}))?", last)
        if (not m or not head.isdigit() or not 0 < len(head) <= 3
                or not all(len(g) == 3 and g.isdigit() for g in mid)):
            return None
        return float(head + "".join(mid) + m.group(1) + ("." + m.group(2) if m.group(2) else ""))
    t = parts[0] if parts else ""
    if "." in t and "," in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        thou = "." if dec == "," else ","
        head, _, tail = t.rpartition(dec)
        if dec in head or not all(len(p) == 3 for p in head.split(thou)[1:]):
            return None
        t = head.replace(thou, "") + "." + tail
    elif "," in t or "." in t:
        sep = "," if "," in t else "."
        head, *groups = t.split(sep)
        if len(groups) == 1 and (len(groups[0]) != 3 or len(head) > 3):
            t = head + "." + groups[0]
        elif all(len(g) == 3 for g in groups) and 0 < len(head) <= 3:
            t = head + "".join(groups)
        else:
            return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_price(text: str) -> Price:
    """Pull the first plausible price out of free text.

    Returns an empty Price when there is no price to find. The tempting
    alternative -- keeping the source text in ``raw`` as a hint -- means a
    listing's price field renders as "Experience superb noise cancellati", which
    is worse than admitting the price is unknown.
    """
    if not text:
        return Price()
    m = _PRICE_RE.search(text)
    if not m:
        return Price()
    if m.group("a"):
        amt, cur = m.group("a"), _sym_currency(m.group("sym") or "")
    elif m.group("b"):
        amt, cur = m.group("b"), (m.group("cur") or "USD").upper()
    elif m.group("c"):
        amt, cur = m.group("c"), (m.group("cur2") or "USD").upper()
    else:
        amt, cur = m.group("d"), _sym_currency(m.group("sym2") or "")
    value = _amount(amt)
    if value is None:
        return Price(raw=m.group(0))
    return Price(amount=value, currency=cur, raw=m.group(0))


def _iter_ld(html: str):
    """Yield every JSON-LD object on the page, flattening @graph and arrays."""
    for blob in _LD_RE.findall(html):
        try:
            data = json.loads(blob.strip())
        except Exception:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                # LIFO stack: push reversed so nodes come out in DOCUMENT
                # order (an ItemList's products were yielded last-first).
                stack.extend(reversed(node))
            elif isinstance(node, dict):
                if "@graph" in node:
                    stack.append(node["@graph"])
                yield node


def _types(node: dict) -> set[str]:
    t = node.get("@type") or node.get("type") or ""
    if isinstance(t, list):
        return {str(x).lower() for x in t}
    return {str(t).lower()}


def _first(v: Any) -> Any:
    return v[0] if isinstance(v, list) and v else v


# Only patterns anchored to a real meta tag. A bare `"price": 23` regex over the
# whole document was tempting and wrong: on a marketplace page the first such
# field is usually a related-items carousel, not the product, which produced a
# confident $23.00 for a pair of $400 headphones. A listing with no price is
# honest; a listing with someone else's price is not.
_META_PRICE = (
    re.compile(
        r'<meta[^>]+(?:itemprop|property|name)="(?:product:)?price:?a?m?o?u?n?t?"'
        r'[^>]+content="([^"]+)"',
        re.I,
    ),
    re.compile(
        r'<meta[^>]+content="([^"]+)"[^>]+(?:itemprop|property|name)="(?:product:)?price"', re.I
    ),
    re.compile(r'<meta[^>]+property="og:price:amount"[^>]+content="([^"]+)"', re.I),
    # itemprop on a visible element, still anchored to a declared price field.
    re.compile(r'itemprop="price"[^>]*content="([^"]+)"', re.I),
)
_META_CURRENCY = re.compile(
    r'<meta[^>]+(?:property|itemprop)="(?:og:|product:)?price:?c?u?r?r?e?n?c?y?"[^>]+content="([A-Z]{3})"', re.I
)


def price_from_offer(offer: dict) -> Price:
    """Read a price out of a schema.org offer, however it chose to nest it.

    Retailers disagree about where the number goes: directly on the offer, in a
    ``priceSpecification``, or as ``lowPrice`` on an AggregateOffer. Checking
    one path finds a price on maybe half of real product pages.
    """
    if not isinstance(offer, dict):
        return Price()
    spec = offer.get("priceSpecification")
    if isinstance(spec, list):
        spec = spec[0] if spec else {}
    if not isinstance(spec, dict):
        spec = {}
    # First PRESENT field, not first truthy (bug 186): a numeric 0 -- a
    # genuinely free item -- is falsy, so the old `or` chain skipped
    # ``price: 0`` and fell through to highPrice, quoting a free product at
    # its list price (extract's own rule: someone else's price is not
    # honest). A string "0"/"0.00" was already kept, so the chain even
    # disagreed with itself by JSON type. A field counts as present when it
    # is not None and not an empty string; 0 counts.
    raw = ""
    for field in (offer.get("price"), spec.get("price"), offer.get("lowPrice"),
                  spec.get("minPrice"), offer.get("highPrice")):
        v = _first(field)
        if v is not None and v != "":
            raw = v
            break
    # A currency is a 3-letter ISO code or it is nothing (bug 110: a list
    # became "['U", "US DOLLARS" became "US ").
    cur = str(_first(offer.get("priceCurrency")) or _first(spec.get("priceCurrency")) or "").strip().upper()
    currency = cur if re.fullmatch(r"[A-Z]{3}", cur) else "USD"
    if raw in ("", None):
        return Price(currency=currency)
    if isinstance(raw, (int, float)):
        return Price(amount=float(raw), currency=currency, raw=str(raw))
    txt = str(raw).replace("$", "").strip()
    value = _amount(txt) if re.fullmatch(r"[\d.,\u00a0\u202f '\u2019]+", txt) else None
    if value is not None:
        return Price(amount=value, currency=currency, raw=str(raw))
    return parse_price(str(raw))


def price_from_meta(html: str) -> Price:
    """Last-resort price scrape from meta tags and inline JSON.

    Plenty of product pages ship a Product JSON-LD block with no price in it --
    the price arrives later from an API call -- but still expose it in
    ``og:price:amount`` or a nearby JSON blob for their own analytics.
    """
    cur_m = _META_CURRENCY.search(html)
    currency = cur_m.group(1).upper() if cur_m else "USD"
    for rx in _META_PRICE:
        m = rx.search(html)
        if not m:
            continue
        # Through the locale-aware parser (bug 108): float(s.replace(",", ""))
        # read "1.299,00" as 1.299 and "1299,00" as 129900.
        digits = re.sub(r"[^\d.,\u00a0\u202f '\u2019]", "", m.group(1))
        amount = _amount(digits) if digits else None
        if amount is None:
            continue
        if 0 < amount < 1_000_000:
            return Price(amount=amount, currency=currency, raw=m.group(1))
    return Price(currency=currency)


def items_from_html(html: str, url: str, source: str = "") -> list[Item]:
    """Extract product/offer records from schema.org JSON-LD.

    Covers the shapes that actually appear in the wild: Product with an offer,
    ItemList of products, and the AggregateOffer variant that marketplaces use
    when several sellers list the same thing.
    """
    items: list[Item] = []
    nodes = [n for n in _iter_ld(html)
             if _types(n) & {"product", "productgroup", "vehicle", "individualproduct"}]
    # The page-global og:price fallback is the SINGLE-PRODUCT-PAGE case: a
    # Product block with no price whose page states one in a meta tag. On a
    # category/ItemList page it stamped that one meta price on EVERY
    # priceless product (bug 59) -- and merge_items would then rank them by
    # a price none of them has.
    meta_price = price_from_meta(html) if len(nodes) == 1 else None
    for node in nodes:
        offer = _first(node.get("offers")) or {}
        if not isinstance(offer, dict):
            offer = {}
        price = price_from_offer(offer)
        if price.amount is None and meta_price is not None:
            price = meta_price

        brand = node.get("brand")
        if isinstance(brand, dict):
            brand = brand.get("name", "")
        # schema.org allows arrays anywhere; a LIST here crashed the whole
        # page's extraction on one malformed listing (bug 58).
        rating_node = _first(node.get("aggregateRating")) or {}
        if not isinstance(rating_node, dict):
            rating_node = {}
        seller = offer.get("seller")
        if isinstance(seller, dict):
            seller = seller.get("name", "")

        image = _first(node.get("image")) or ""
        if isinstance(image, dict):
            image = image.get("url", "")

        # The item's OWN url, absolute (bug 114: a relative offers.url shipped
        # as "/p/123" and a list as "['https://...']"); the page url only when
        # the node has none. Identity sees the page url only on a single-
        # product page (bug 115: every node on a /dp/ page inherited the
        # page's ASIN and all GTIN-less products merged into one).
        own = _item_url(_first(offer.get("url")) or _first(node.get("url")) or "", url)
        it = Item(
            title=str(node.get("name") or "")[:400],
            url=own or url,
            price=price,
            brand=str(brand or "")[:120],
            seller=str(seller or "")[:120],
            availability=str(offer.get("availability") or "").rsplit("/", 1)[-1],
            condition=str(offer.get("itemCondition") or "").rsplit("/", 1)[-1],
            image=str(image)[:500],
            source=source,
            identity=identity_for(
                gtin=str(node.get("gtin13") or node.get("gtin14") or node.get("gtin")
                         or node.get("gtin12") or node.get("gtin8") or ""),
                # mpn is manufacturer-scoped (a weaker key); a retailer sku is
                # NOT a cross-site key at all and no longer feeds identity
                # (bug 56 -- two sellers' unrelated "ITEM-0001" rows merged).
                mpn=str(node.get("mpn") or ""),
                brand=str(brand or ""),
                title=str(node.get("name") or ""),
                url=own or (url if len(nodes) == 1 else ""),
            ),
        )
        try:
            if rating_node.get("ratingValue"):
                it.rating = float(rating_node["ratingValue"])
            if rating_node.get("reviewCount") or rating_node.get("ratingCount"):
                it.reviews = int(rating_node.get("reviewCount") or rating_node["ratingCount"])
        except (TypeError, ValueError):
            pass
        if it.title:
            items.append(it)
    return items


def _item_url(raw: Any, page: str) -> str:
    """``raw`` resolved against the page, or "" when it is not an http(s) url."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    u = urllib.parse.urljoin(page, raw.strip())
    return u if u.lower().startswith(("http://", "https://")) else ""


# ── identity resolution ──────────────────────────────────────────────────────

_ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})", re.I)
_STOP = {
    "the", "and", "for", "with", "new", "best", "buy", "official", "genuine",
    "free", "shipping", "sale", "deal", "in", "of", "a", "to",
}


#: A model code: letters and digits mixed. "wh-1000xm5" tokenises to "wh" +
#: "1000xm5"; the second half is the part that is the same on every site that
#: sells it. Two characters is the floor because plenty of real model codes are
#: that short -- the Logitech MX Master *3S*, the Sony WH-1000X*M4*.
_MODEL_RE = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{2,}$")

#: Retailer and boilerplate words that leak in from page titles and would
#: otherwise make the same product hash differently per site.
_SITE_WORDS = {
    "amazon", "walmart", "target", "ebay", "bestbuy", "newegg", "costco",
    "com", "www", "shop", "store", "online", "usa", "us",
}


#: Unit/spec tokens that look like model codes (letters+digits) but name a
#: capacity, size, frequency or count -- "128gb", "8k", "1080p", "5ghz". They
#: are exactly what differs between variants of ONE product and what is
#: shared between DIFFERENT products, so they must never be the join key
#: (bug 57: "iPhone 15 Pro 128GB" and "iPhone 15 Pro Max 128GB" both hashed
#: to model:128gb). "3s"/"1000xm5" carry no unit and stay model codes.
_UNIT_TOKEN_RE = re.compile(
    r"^\d+(?:gb|tb|mb|kb|mah|wh|kwh|hz|khz|mhz|ghz|mm|cm|km|in|inch|ft|oz|lb|lbs"
    r"|kg|mg|ml|pcs|pk|ct|mp|mpx|dpi|fps|k|p|x)$"
)


#: Words that, directly after a model code, name a DIFFERENT product:
#: "S23" vs "S23 Ultra", "A54" vs "A54 Plus".
_VARIANT_WORDS = {"pro", "max", "ultra", "plus", "mini", "lite", "fe", "neo", "air", "xl", "edge"}


def _gtin_check_ok(g: str) -> bool:
    """GS1 mod-10 check digit (bug 112): an all-zero placeholder or a
    truncated SKU number is not a globally unique key."""
    if len(set(g)) == 1:
        return False
    body = [int(c) for c in g[:-1]]
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == int(g[-1])


def identity_for(*, gtin: str = "", mpn: str = "", brand: str = "", title: str = "", url: str = "") -> str:
    """A cross-site join key for one physical product.

    Strongest evidence first: a real GTIN/MPN, then a marketplace's own product
    id, then the model code, then a normalized word slug.

    The model-code rule does the real work. The same headphones are titled six
    different ways across six sites -- "Sony WH-1000XM5 Premium Noise
    Cancelling Wireless...", "Sony WH-1000XM5 The Best Wireless... Black",
    "Sony WH-1000XM5 Bluetooth Wireless Noise-Canceling..." -- and a slug built
    from those words hashes differently every time. The token "1000xm5" is in
    all of them and in nothing else, so keying on it collapses them correctly
    where a word slug cannot.
    """
    # A GTIN is 8-14 DIGITS (GTIN-8/12/13/14); anything else in this slot is
    # not the globally unique key it was being treated as (bug 56).
    g = re.sub(r"[^0-9]", "", gtin or "")
    if (g.isdigit() and 8 <= len(g) <= 14 and g == re.sub(r"[^0-9A-Za-z]", "", gtin or "")
            and _gtin_check_ok(g)):
        return f"gtin:{g}"
    m = _ASIN_RE.search(url or "")
    if m:
        return f"asin:{m.group(1).upper()}"
    p = re.sub(r"[^0-9A-Za-z]", "", mpn or "").lower()
    if len(p) >= 4 and any(ch.isdigit() for ch in p):
        return f"mpn:{p}"

    words = [w for w in re.findall(r"[a-z0-9]+", f"{brand} {title}".lower())
             if w not in _STOP and w not in _SITE_WORDS]

    models = [i for i, w in enumerate(words) if _MODEL_RE.match(w) and not _UNIT_TOKEN_RE.match(w)]
    if models:
        # The FIRST such token, in title order -- not the longest. Titles put
        # the model code right after the brand and push spec tokens to the tail,
        # so "Logitech MX Master 3S, Wireless Mouse, 8K DPI" and "Logitech MX
        # Master 3S - Performance Wireless Mouse" agree on "3s" but disagree on
        # which token is longest ("8k" appears in only one of them).
        i = models[0]
        # A variant word right after the model code is part of the key
        # (bug 113): "Galaxy S23" and "Galaxy S23 Ultra" are two products
        # with two prices, and both were model:s23.
        variants = sorted(w for w in words[i + 1:i + 3] if w in _VARIANT_WORDS)
        return "model:" + "+".join([words[i], *variants])

    # Numeric tokens survive whatever their length: a bare "15" is the
    # generation that separates iPhone 14 Pro from 15 Pro (bug 57b).
    keep = [w[:40] for w in words if len(w) >= 3 or w.isdigit()][:6]
    return "slug:" + "-".join(sorted(keep)) if keep else ""


def doc_from_fetch(url: str, html: str, *, source: str = "", snippet: str = "") -> Doc:
    return Doc(
        url=url,
        title=title_of(html),
        snippet=snippet,
        text=to_markdown(html, url),
        source=source,
    )
