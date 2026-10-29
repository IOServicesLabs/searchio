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

import bisect
import functools
import json
import html as html_mod
import re
import urllib.parse
from typing import Any

from .models import Doc, Item, Price

_WS = re.compile(r"\n{3,}")


#: memo_extract_import: False once the import has failed in this process.
_TRAF: Any = None
#: extract_strip_scripts: blocks trafilatura drops without reading.
_NOISE_BLOCKS = re.compile(r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1\s*>", re.S | re.I)


def _trafilatura():
    """The trafilatura module, or None. With memo_extract_import a failed
    import is remembered: Python does not cache a failure, and a broken
    install (lxml without lxml_html_clean) re-ran the import on every read."""
    global _TRAF
    if _TRAF is not None:
        return _TRAF or None
    try:
        import trafilatura
    except Exception:
        from .config import get_settings

        if get_settings().memo_extract_import:
            _TRAF = False
        return None
    _TRAF = trafilatura
    return trafilatura


def to_markdown(html: str, url: str = "") -> str:
    """Extract the readable article body as markdown.

    trafilatura when available -- it is the best-in-class boilerplate remover --
    with a crude tag strip as a floor so this never returns nothing just because
    an optional dependency is missing.
    """
    try:
        trafilatura = _trafilatura()
        if trafilatura is None:
            raise ImportError("trafilatura unavailable")
        from .config import get_settings

        s = get_settings()
        src = html
        if getattr(s, "extract_strip_scripts", False):
            # Inline scripts/styles are often most of a heavy page's bytes;
            # trafilatura parses and then discards them anyway.
            src = _NOISE_BLOCKS.sub("", html)
        out = trafilatura.extract(
            src,
            url=url or None,
            output_format="markdown",
            include_links=False,
            include_images=False,
            include_tables=True,
            favor_recall=True,
            # extract_no_comments: reader comments are not the page, and
            # finding them is a separate pass over the tree.
            include_comments=not getattr(s, "extract_no_comments", False),
        )
        if out and out.strip():
            return _WS.sub("\n\n", out.strip())
    except Exception:
        pass

    import html as _html

    from .net.blocks import visible_text

    # The tag strip leaves character references as written (``AWS&#x27;s``,
    # ``&quot;x509``): decode them, or every fallback read shows raw entities.
    return _html.unescape(visible_text(html))[:20000]


#: <title> elements that are NOT the page title: an inline SVG icon's
#: accessible name, or the string "<title>" inside a script/style body.
_TITLE_NOISE = re.compile(r"<(script|style|svg)\b.*?</\1\s*>", re.I | re.S)


_HEAD_END = re.compile(r"</head\s*>|<body[\s>]", re.I)


def _fast_title() -> bool:
    from .config import get_settings

    return get_settings().fast_title


def title_of(html: str) -> str:
    """The document title, entity-decoded (bug 111).

    The first <title> in the byte stream is not always the page's: pages put
    inline SVG logos (with their own <title>) and scripts (whose strings can
    spell "<title>") ahead of the real one, and the old regex took whichever
    came first -- and shipped "A &amp; B" to the agent undecoded.
    """
    m = None
    if _fast_title():
        # The title lives in <head>: clean and search only that much, not a
        # whole 1 MB body (~3 ms), falling back to the body when it is not there.
        end = _HEAD_END.search(html)
        if end is not None:
            m = re.search(r"<title[^>]*>(.*?)</title>",
                          _TITLE_NOISE.sub("", html[:end.start()]), re.I | re.S)
    if m is None:
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


#: structured_listings: cheap pre-check before any JSON is parsed -- two thirds
#: of pages carry JSON-LD, almost none of it a product.
_LISTING_HINT = re.compile(r'"(?:Product|Car|Vehicle|IndividualProduct|Motorcycle)"')
_LISTING_TYPES = {"product", "car", "vehicle", "individualproduct", "motorcycle"}
_UNITS = {"SMI": "mi", "KMT": "km"}
_MAX_ROWS = 60


def _walk_ld(html: str):
    """Every JSON-LD dict at any depth: listing pages nest their products
    under ``Offer.itemOffered`` or ``ItemList.itemListElement[].item``.
    Walks each blob once (JSON has no cycles); deduping by ``id()`` across
    blobs dropped nodes whose ids a freed earlier blob had used."""
    for blob in _LD_RE.findall(html):
        try:
            stack = [json.loads(blob.strip())]
        except Exception:
            continue
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(reversed(node))
            elif isinstance(node, dict):
                yield node
                stack.extend(reversed([v for v in node.values() if isinstance(v, (dict, list))]))


def _money(p: Price) -> str:
    if p.amount is None:
        return ""
    amt = f"{p.amount:,.0f}" if p.amount == int(p.amount) else f"{p.amount:,.2f}"
    return f"${amt}" if p.currency == "USD" else f"{amt} {p.currency}"


def _ld_row(node: dict, page: str = "") -> tuple[str, str]:
    """(row, key) for one product/vehicle node; key is what proves the
    extracted prose already carries it (its VIN, else its name)."""
    offer = _first(node.get("offers")) or {}
    if not isinstance(offer, dict):
        offer = {}
    miles = node.get("mileageFromOdometer")
    if isinstance(miles, dict):
        v = miles.get("value")
        if str(v).replace(",", "").isdigit():
            v = f"{int(str(v).replace(',', '')):,}"
        miles = f"{v} {_UNITS.get(str(miles.get('unitCode', '')).upper(), '')}".strip() if v else ""
    seller = offer.get("seller") or node.get("seller") or {}
    where = ""
    if isinstance(seller, dict):
        addr = seller.get("address") or {}
        if isinstance(addr, dict):
            where = ", ".join(str(addr.get(k)) for k in ("addressLocality", "addressRegion") if addr.get(k))
        seller = seller.get("name", "")
    miles = str(miles or "")
    if miles.replace(",", "").isdigit():
        miles = f"{int(miles.replace(',', '')):,} mi"
    vin = str(node.get("vehicleIdentificationNumber") or "")
    name = str(node.get("name") or "").strip()
    engine = _pick(node, "vehicleEngine")
    parts = [name, _money(price_from_offer(offer)), miles,
             str(node.get("vehicleTransmission") or ""),
             engine if _ENGINE.fullmatch(engine) and engine.lower() not in name.lower() else "",
             f"VIN {vin}" if vin else "",
             " ".join(x for x in (str(seller or ""), f"({where})" if where else "") if x),
             _item_url(_first(offer.get("url")) or _first(node.get("url")) or "", page)]
    return " | ".join(p for p in parts if p), vin or name


_VIN_KEY = re.compile(r'"vin"\s*:\s*"([A-HJ-NPR-Z0-9]{17})"', re.I)
_DECODER = json.JSONDecoder()


def _pick(d: dict, *keys: str) -> str:
    """The first present value among ``keys``, flattened to text (sites wrap
    a make as ``[{"code": "FORD", "name": "Ford"}]`` as often as ``"Ford"``)."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, list):
            v = v[0] if v else None
        if isinstance(v, dict):
            v = v.get("name") or v.get("value")
        if v not in (None, "", []):
            return str(v)
    return ""


#: listing_generic: a listing record is recognised by its shape -- a JSON
#: object with a name and a price (or a VIN) -- wherever it sits: a ``data-*``
#: attribute or inline script state. No site's markup or key names are known.
# Anchor keys are found by literal search on their tails (``salePrice"``,
# ``VIN"``): any case-insensitive regex over a 2 MB page costs 4-15x more.
_ANCHOR_TAILS = ('rice"', 'RICE"', 'vin"', 'Vin"', 'VIN"', 'ionNumber"')
# A price OBJECT under a GraphQL-style key (``"pricing({\"storeId\":1})":``)
# has no fixed tail; its head is enough to find the item around it.
_ANCHOR_HEADS = ('"pricing', '"Pricing')
_COLON = re.compile(r'"\s*:')
_DATA_ATTR = re.compile(r'data-[\w-]+=(?:"(\{[^"]{20,}\})"|\'(\{[^\']{20,}\})\')')
_MAX_ANCHORS = 600


@functools.lru_cache(maxsize=4096)
def _norm(k: str) -> str:
    return k.lower().replace("_", "").replace("-", "")


#: Stems _field matches as key suffixes (``salePrice`` -> price); exact
#: matches need no list.
_SUFFIX_STEMS = frozenset(("price", "year", "make", "model", "trim", "mileage", "odometer", "miles",
                           "city", "locality", "state", "region", "dealername", "storename",
                           "sellername", "seller", "dealer", "transmission", "exteriorcolor", "color",
                           "engine", "providername"))
_SUFFIX_LENS = sorted({len(x) for x in _SUFFIX_STEMS})


def _key_index(d: dict) -> dict[str, list[tuple[str, bool]]]:
    """stem -> [(key, exact)] for every key of ``d``, built in one pass so
    each _field lookup is a dict hit rather than a scan of the keys."""
    idx: dict[str, list[tuple[str, bool]]] = {}
    for k in d:
        n = _norm(k)
        idx.setdefault(n, []).append((k, True))
        if n.endswith("name") and len(n) > 4:
            idx.setdefault(n[:-4], []).append((k, True))
        for ln in _SUFFIX_LENS:
            if len(n) > ln and n[-ln:] in _SUFFIX_STEMS:
                idx.setdefault(n[-ln:], []).append((k, False))
    return idx


def _field(d: dict, *stems: str, exact: bool = False, idx: dict | None = None,
           valid: re.Pattern | None = None, skip: re.Pattern | None = None) -> str:
    """The scalar field whose normalised key is a stem, ``<stem>name`` or
    (unless ``exact``) ends in the stem: ``makeName``, ``car_year`` and
    ``salePrice`` all match without knowing whose markup they come from.
    Among matches for the first stem that has any, the longest wins, so a
    record with both a code (``"MUST"``) and a label (``"Mustang"``) reads
    as the label."""
    idx = idx if idx is not None else _key_index(d)
    for stem in stems:
        best = ""
        for k, is_exact in idx.get(stem, ()):
            if is_exact or not exact:
                if isinstance(d[k], bool) or (skip is not None and skip.search(_norm(k))):
                    continue
                v = _pick(d, k)
                if valid is not None and not valid.fullmatch(v.strip()):
                    continue
                # Ties go to the mixed-case label ("Ford" over "FORD").
                if (len(v), not v.isupper()) > (len(best), not best.isupper()) \
                        and not v.startswith(("{", "[")):
                    best = v
        if best:
            return best
    return ""


#: Item names live under these keys; a looser match reads ``countryName``.
_NAME_KEYS = ("name", "title", "productname", "producttitle", "itemname", "itemtitle",
              "displayname", "listingtitle", "headline", "productlabel", "itemlabel")
#: Keys that identify one sellable item.
_ID_KEYS = ("sku", "productid", "itemid", "listingid", "offerid", "gtin", "gtin13", "gtin12",
            "upc", "mpn", "asin", "stocknumber", "stockid", "id")
_ASSET = re.compile(r"(?:image|img|photo|thumb|hero|logo|icon|video|asset"
                    r"|\.(?:jpe?g|png|webp|gif|svg|avif)\b)", re.I)


def _lift(obj: dict) -> dict:
    """``obj`` with its child objects' scalars lifted onto it where it lacks
    them. Normalised stores (GraphQL caches) split one item into children --
    ``identifiers: {productLabel, itemId}``, ``pricing(...): {value: 299}`` --
    and a price object's amount becomes the item's ``price``."""
    out = dict(obj)
    for k, v in obj.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and _norm(k).startswith("offer"):
            v = v[0]
        if not isinstance(v, dict) or _ASSET.search(k):
            continue
        # schema.org ``offers`` and GraphQL ``pricing(...)`` hold the amount.
        if _norm(k).startswith(("pric", "offer")):
            if "price" not in _key_index(out):
                amt = _price_of(v)
                if amt:
                    out["price"] = amt
            continue
        # Only a child that describes the item itself: a ``make``, ``dealer``
        # or ``tracking`` child's name/url is its own, not the item's
        # ("Ford", "ALM CDJR Perry", a click beacon all named items once).
        if not _ITEM_PART.search(k):
            continue
        for ck, cv in v.items():
            if ck not in out and not isinstance(cv, (dict, list)) and not ck.startswith("__"):
                out[ck] = cv
    return out


_ITEM_PART = re.compile(r"(?:identif|product|item|detail|info|summary|listing|descri)", re.I)
_NUMBER = re.compile(r"\$?\s*\d[\d,]*(?:\.\d+)?")
#: Price-suffixed keys that are not the asking price: a market estimate
#: (``imvPrice``, ``expectedPrice``), a struck-through original, a payment.
_PRICE_NOT = re.compile(r"(?:imv|expected|estimat|market|average|avg|typical|fair|original|previous|"
                        r"monthly|payment|downpay|saving|discount|differ|lowest|highest|minimum|maximum)")
#: Distance keys that are a search setting, not how far the item is.
_DIST_NOT = re.compile(r"(?:search|radius|max|min|nationwide|filter|default|km)")
def _distance(d: dict, depth: int = 3) -> str:
    """A ``*distance*`` number in ``d`` or its descriptive children
    (``vehicle.details.dealerDistanceMiles``), never a search setting."""
    for k, v in d.items():
        n = _norm(k)
        if "distance" in n and not _DIST_NOT.search(n) and isinstance(v, (int, float, str)) \
                and not isinstance(v, bool) and _NUMBER.fullmatch(str(v)):
            return str(v)
    if depth > 1:
        for k, v in d.items():
            # A transfer/delivery child's distance is to the buyer's store.
            if isinstance(v, dict) and not _ASSET.search(k) and not _TRACKING.search(k) \
                    and not re.search(r"transfer|deliver|ship|destination", k, re.I) \
                    and (got := _distance(v, depth - 1)):
                return got
    return ""


#: Row parts a second copy of the same item may add to the kept one.
_MERGE_PART = re.compile(r"distance [\d,]+|(?:\d+-Speed )?(?:Manual|Automatic|CVT)|history: .*", re.I)
#: A wrapper child's own suffix (``sellerDto``, ``dealerInfo``) hides what it is.
_WRAP_SUFFIX = re.compile(r"(?:dto|info|data|details?|obj(?:ect)?)$")


def _price_of(v: dict) -> str:
    """The amount in a price/offer object: its plain amount before any
    ``*Price`` sibling, which may be a fee-inclusive total or a base price."""
    return (_field(v, "price", "current", "amount", "value", "sale", "final", exact=True, valid=_NUMBER)
            or _field(v, "price", valid=_NUMBER, skip=_PRICE_NOT))
#: An engine description ("5L V-8 Gas", "315 hp 2.3L I4", "8-Cylinder"), not
#: a fuel word alone ("Gas") or an id: a shopper filtering on V8 vs I4 needs it.
_ENGINE = re.compile(r"(?=.{2,60}$).*(?:\d(?:\.\d)?\s*L\b|cyl|\b[VIW]-?\d{1,2}\b|inline|flat-?\d|boxer"
                     r"|electric|hybrid|rotary|turbo|diesel|\bhp\b).*", re.I)


def _engine(obj: dict, ix: dict) -> str:
    """The record's engine, or one assembled from a size and a cylinder
    count when the record splits them (``engineSize: "5.0L", cylinders: 8``)."""
    eng = _field(obj, "engine", idx=ix, valid=_ENGINE)
    if eng:
        return eng
    size = _field(obj, "enginesize", "displacement", "enginedisplacement", exact=True, idx=ix)
    cyl = _field(obj, "cylinders", "enginecylinders", "cylindercount", exact=True, idx=ix)
    return " ".join(x for x in (size if re.fullmatch(r"\d+(?:\.\d+)?\s*L", size, re.I) else "",
                                f"{cyl}-cyl" if cyl.isdigit() else "") if x)
_TRACKING = re.compile(r"(?:beacon|track|event|analytic|pixel|click|impression|adserv)", re.I)


#: listing_history: vehicle-history fields, by normalised key. Counts must be
#: integers and flags booleans, so a dealer's ``ownerName``, a product's
#: ``cleanTitle`` string or a facet's ``accidentFreeCount`` never read as one.
_HIST_COUNTS = {"accidentcount": "accidents", "accidents": "accidents", "numberofaccidents": "accidents",
                "numaccidents": "accidents", "ownercount": "owners", "owners": "owners",
                "numberofowners": "owners", "numowners": "owners", "previousowners": "owners"}
_HIST_FLAGS = {"iscleantitle": "clean title", "cleantitle": "clean title", "hascleantitle": "clean title",
               "isoneowner": "one owner", "oneowner": "one owner", "singleowner": "one owner",
               "isaccidentfree": "no accidents", "accidentfree": "no accidents", "noaccidents": "no accidents",
               "personaluse": "personal use", "ispersonaluse": "personal use"}
#: Problem flags: true prints the problem, false prints its absence.
_HIST_PROBLEMS = {"hasaccidents": "accidents", "hasaccident": "accidents", "accidentreported": "accidents",
                  "salvage": "salvage title", "issalvage": "salvage title", "salvagetitle": "salvage title",
                  "lemon": "lemon record", "islemon": "lemon record", "framedamaged": "frame damage",
                  "framedamage": "frame damage", "hasframedamage": "frame damage", "damaged": "damage",
                  "isdamaged": "damage", "theft": "theft record", "thefttitle": "theft title",
                  "totalloss": "total loss", "istotalloss": "total loss", "flooddamage": "flood damage",
                  "isfleet": "fleet use", "fleet": "fleet use", "fleetuse": "fleet use",
                  "rentalfleet": "rental use", "isrental": "rental use", "rentaluse": "rental use",
                  "taxifleet": "taxi use", "odometerrollback": "odometer rollback"}
#: A history summary as a list of codes (``["NO_SALVAGE_TITLE", "ACCIDENTS_REPORTED"]``).
_HIST_CODE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_HIST_WORDS = re.compile(r"ACCIDENT|OWNER|TITLE|DAMAGE|SALVAGE|FLOOD|FRAME|LEMON|FLEET|RENTAL|ODOMETER")


def _history(obj: dict) -> str:
    """The record's vehicle-history summary ("0 accidents, 2 owners, clean
    title"), read from it and its direct child objects, or ""."""
    out: list[str] = []
    seen: set[str] = set()
    clear: list[str] = []  # problems the record says are absent

    def add(part: str, label: str) -> None:
        if label not in seen:
            seen.add(label)
            out.append(part)

    for src in [obj] + [v for k, v in obj.items() if isinstance(v, dict) and not _ASSET.search(k)]:
        for k, v in src.items():
            n = _norm(k)
            if n in _HIST_COUNTS and isinstance(v, int) and not isinstance(v, bool) and 0 <= v < 100:
                what = _HIST_COUNTS[n]
                add(f"{v} {what[:-1] if v == 1 else what}", what)
            elif n in _HIST_FLAGS and isinstance(v, bool):
                what = _HIST_FLAGS[n]
                add(what if v else f"{what}: no", what)
            elif n in _HIST_PROBLEMS and isinstance(v, bool):
                what = _HIST_PROBLEMS[n]
                if v:
                    add(what, what)
                elif what not in seen and what not in clear:
                    clear.append(what)
            elif isinstance(v, list) and v and len(v) <= 20 and all(
                    isinstance(c, str) and _HIST_CODE.fullmatch(c) for c in v):
                codes = [c for c in v if _HIST_WORDS.search(c)]
                if codes:
                    for c in codes:
                        add(c.replace("_", " ").lower(), c)
    clear = [c for c in clear if c not in seen]
    if clear:
        out.append("no " + "/".join(clear))
    return ", ".join(out)


#: Literal pieces that start a history field's search (C-speed finds); each
#: hit is then judged by its whole key and value.
_HIST_HINTS = ("ccident", "wner", "itle", "ALVAGE", "AMAGE", "leet", "ental", "ersonalUse")
_HIST_KEY = re.compile(r'"([A-Za-z_]{3,40})"\s*:\s*')
_MAX_HIST_HITS = 400
#: Objects decoded walking out from one history field (a vehicle record can
#: hold hundreds of nested objects before its field).
_HIST_BUDGET = 400
_VIN_ONLY = re.compile(r"[A-HJ-NPR-Z0-9]{17}")
#: A VIN under any of its keys, bare or wrapped (``{"value": "1FA..."}``).
_VIN_ANY = re.compile(r'"(?:vin|vehicleIdentifier|vehicleIdentificationNumber)"\s*:\s*'
                      r'(?:\{\s*"value"\s*:\s*)?"([A-HJ-NPR-Z0-9]{17})"', re.I)


def _histories(src: str) -> dict[str, str]:
    """VIN -> history summary. From each history field, objects are decoded
    outward: the innermost one holding the field is summarised, and the
    first enclosing one whose text names a VIN gives it its owner -- only
    when it names exactly one (a results page's wrapper names them all)."""
    out: dict[str, str] = {}
    memo: dict[int, tuple] = {}
    hits: list[int] = []
    for hint in _HIST_HINTS:
        at = src.find(hint)
        while at >= 0 and len(hits) < _MAX_HIST_HITS:
            hits.append(at)
            at = src.find(hint, at + 1)
    covered = -1
    for at in sorted(hits):
        if at < covered:
            continue
        # The hit is in a key ("accidentCount") or a code ("NO_SALVAGE_TITLE").
        q = src.rfind('"', max(0, at - 40), at)
        m = _HIST_KEY.match(src, q) if q >= 0 else None
        if m is not None:
            # Only a key of the tables is worth decoding around ("listingTitle"
            # matches a hint but is no history field).
            n = _norm(m.group(1))
            if n not in _HIST_COUNTS and n not in _HIST_FLAGS and n not in _HIST_PROBLEMS:
                continue
        else:
            # A code inside a list: find the list's key.
            lb = src.rfind("[", max(0, at - 400), at)
            q = src.rfind('"', max(0, lb - 50), lb - 1) if lb > 0 else -1
            q = src.rfind('"', max(0, q - 50), q) if q > 0 else -1
            m = _HIST_KEY.match(src, q) if q >= 0 else None
            if m is None or m.end() != lb:
                continue
        pos, summary, budget = m.start(), "", _HIST_BUDGET
        while budget > 0:
            budget -= 1
            pos = src.rfind("{", 0, pos)
            if pos < 0 or at - pos > 400000:
                break
            if pos not in memo:
                if src[pos + 1:pos + 3].lstrip()[:1] not in ('"', "}"):
                    memo[pos] = (None, -1)
                    continue
                try:
                    memo[pos] = _DECODER.raw_decode(src, pos)
                except ValueError:
                    memo[pos] = (None, -1)
            obj, end = memo[pos]
            if not isinstance(obj, dict) or end <= at:
                continue
            if not summary:
                summary = _history(obj)
                if not summary:
                    break
            vins = set(v.upper() for v in _VIN_ANY.findall(src, pos, end))
            if vins:
                if len(vins) == 1:
                    vin = vins.pop()
                    if vin not in out:
                        out[vin] = summary
                    covered = end
                break
    return out


def _rec_row(obj: dict, page: str) -> tuple[str, str, bool, tuple]:
    """(row, key, named, ids) when ``obj`` looks like one listing, else ("",
    "", False, ()); ``named`` is False when the name was only assembled from
    parts, and ``ids`` are the item's identifiers when it carries no link."""
    ix = _key_index(obj)
    if any(isinstance(v, (dict, list)) for v in obj.values()) and not (
            any(nk in ix for nk in _NAME_KEYS) and ("price" in ix or "vin" in ix)):
        obj = _lift(obj)
        ix = _key_index(obj)
    if not ("price" in ix or "vin" in ix or "vehicleidentificationnumber" in ix or "vehicleidentifier" in ix):
        return "", "", False, ()
    vin = _field(obj, "vin", "vehicleidentificationnumber", "vehicleidentifier", exact=True, idx=ix).upper()
    if not vin:
        # A VIN wrapped with its checks: ``"vin": {"value": "1FA...", "invalidVin": false}``.
        box = next((v for k, v in obj.items() if _norm(k) == "vin" and isinstance(v, dict)), None)
        if box:
            vin = str(box.get("value") or box.get("vin") or "").upper()
    if vin and not _VIN_KEY.fullmatch(f'"vin":"{vin}"'):
        vin = ""
    # A price is a number: "control" under ``variantPrice`` is an
    # experiment, and a ``"Price": "None"`` beside ``FinalPrice`` is not it.
    # A market estimate beside it is not the price; the asking price may sit
    # in a price child the record also has (``imvPrice`` + ``priceDto``).
    price = _field(obj, "price", idx=ix, valid=_NUMBER, skip=_PRICE_NOT) or next(
        (amt for k, v in obj.items() if isinstance(v, dict) and _norm(k).startswith(("pric", "offer"))
         and (amt := _price_of(v))), "")
    # ... and zero is a placeholder, not an offer.
    if price and not price.strip("$ 0.,"):
        price = ""
    # The longest is the descriptive title, not a model number.
    name = max((v for nk in _NAME_KEYS for k, ex in ix.get(nk, ()) if ex and not isinstance(obj[k], bool)
                and not (v := _pick(obj, k)).startswith(("{", "["))), key=len, default="")
    named = bool(name)
    if not name:
        parts = [x for x in (_field(obj, stem, idx=ix) for stem in ("year", "make", "model", "trim")) if x]
        # A combined field matches a stem too (``yearMakeModel`` as the model,
        # ``yearMakeModelTrim`` as the trim): a part inside another is dropped.
        parts = [x for i, x in enumerate(parts)
                 if not any(i != j and x.lower() in y.lower() and (len(y) > len(x) or j < i)
                            for j, y in enumerate(parts))]
        # A lone word ("ford") is a tracking stub's category; a lone model
        # code ("CT1000P310SSD5-01") still names one item.
        name = " ".join(parts) if len(parts) >= 2 or (parts and re.search(r"\d", parts[0])) else ""
    brand = _field(obj, "brand", exact=True, idx=ix)
    if name and brand and len(brand) < 40 and brand.lower() not in name.lower():
        name = f"{brand} {name}"
    cond = _field(obj, "itemcondition", "condition", "stocktype", exact=True, idx=ix)
    cond = cond.rsplit("/", 1)[-1]
    if cond.endswith("Condition") and len(cond) > 9:
        cond = cond[:-9]
    if cond and len(cond) < 20 and cond.lower() not in name.lower():
        name = f"{cond} {name}"
    if not name or len(name) > 200 or not (price or vin):
        return "", "", False, ()
    # A bare "2024 Ford Mustang" hides the trim (GT vs EcoBoost) the record
    # carries beside it.
    trim = (_field(obj, "trim", idx=ix) or next(
        (t for k, v in obj.items() if isinstance(v, dict) and not _ASSET.search(k) and (t := _field(v, "trim"))),
        "")) if named else ""
    if trim and len(trim) < 40 and not trim.isdigit() \
            and trim.split()[0].lower() not in name.lower().split():
        name = f"{name} {trim}"
    link = next((str(v) for k, v in obj.items() if _norm(k).endswith(("url", "href", "link"))
                 and isinstance(v, str) and v.startswith(("http", "/"))
                 and not _ASSET.search(k) and not _ASSET.search(v)
                 and not _TRACKING.search(k) and not _TRACKING.search(v)), "")
    # A priced name alone is often a shipping option or a currency table;
    # an item also carries an identifier, a link or a picture.
    if not vin and not (link or _field(obj, *_ID_KEYS, exact=True, idx=ix)
                        or any(_ASSET.search(k) for k in obj)):
        return "", "", False, ()
    # Seller/dealer details are commonly one nested object down.
    sub = next((v for k, v in obj.items() if isinstance(v, dict)
                and _WRAP_SUFFIX.sub("", _norm(k)).endswith(("seller", "dealer", "store", "location"))), {})
    try:
        price = f"${float(price.replace(',', '').lstrip('$')):,.0f}" if price else ""
    except ValueError:
        pass
    miles = _field(obj, "mileage", "odometer", "miles", idx=ix)
    if miles.replace(",", "").isdigit():
        miles = f"{int(miles.replace(',', '')):,} mi"
    where = ", ".join(x for x in (_field(obj, "city", "locality", idx=ix) or _field(sub, "city", "locality"),
                                   _field(obj, "state", "region", idx=ix) or _field(sub, "state", "region"),
                                   _field(sub, "zip", "zipcode", "postalcode", exact=True)) if x)
    seller = (_field(obj, "dealername", "storename", "sellername", "seller", "dealer", idx=ix)
              or _field(sub, "dealername", "storename", "sellername", "providername", "name"))
    # How far away it is, when the record says: a "within 100 miles" search
    # still ships far stores' cars (a store transfer from 1,863 away).
    dist = _field(obj, "distance", exact=True, idx=ix, valid=_NUMBER) or _distance(obj)
    try:
        dist = f"distance {float(dist.replace(',', '')):,.0f}" if dist else ""
    except ValueError:
        dist = ""
    engine = _engine(obj, ix)
    parts = [name, price, miles, _field(obj, "transmission", idx=ix),
             engine if engine.lower() not in name.lower() else "",
             _field(obj, "exteriorcolor", "color", idx=ix), f"VIN {vin}" if vin else "",
             " ".join(x for x in (seller, f"({where})" if where else "") if x), dist,
             _item_url(link, page) if link else ""]
    rid = () if link else tuple(v for k in _ID_KEYS if (v := _field(obj, k, exact=True, idx=ix)))
    return " | ".join(p for p in parts if p), vin or name, named, rid


def _anchors(src: str) -> list[tuple[int, int]]:
    """(start, end) of each price/VIN-like key, in page order."""
    out = []
    for tail in _ANCHOR_TAILS:
        i = src.find(tail)
        while i >= 0:
            m = _COLON.match(src, i + len(tail) - 1)
            if m:
                out.append((i, m.end()))
            i = src.find(tail, i + 1)
    for head in _ANCHOR_HEADS:
        i = src.find(head)
        while i >= 0:
            out.append((i, i + len(head)))
            i = src.find(head, i + 1)
    out.sort()
    return out


def _records_in(src: str, page: str, rows: list, done: set, ids: dict | None = None) -> None:
    """Rows for the innermost listing-shaped object around each anchor key,
    found by decoding outward from the key a bounded number of times. A
    record whose name was only assembled from parts ("CT1000P310SSD5-01")
    may be a child of the real item, so a few more levels are tried for an
    enclosing record that carries a name of its own."""
    covered, worked = -1, 0
    # Part-named rows (start, end, row index): a named record found inside
    # one is the real item and takes its row.
    wrappers: list[tuple[int, int, int]] = []
    # Neighbouring anchors share enclosing objects: decode (and judge) each
    # start position once. A miss is (None, end) or (None, -1) when invalid.
    memo: dict[int, tuple] = {}
    for a0, a1 in _anchors(src):
        if worked >= _MAX_ANCHORS or len(rows) >= _MAX_ROWS:
            return
        if a0 < covered:
            continue
        worked += 1
        pos, take, fallback, budget = a0, None, None, 12
        while budget > 0:
            budget -= 1
            pos = src.rfind("{", 0, pos)
            if pos < 0 or a0 - pos > 20000:
                break
            if pos not in memo:
                # Only a real JSON object start is worth a decode: a failed
                # one costs O(pos) (the error counts lines up to it), and
                # escaped JSON inside strings (``{\"``) fails every time.
                nxt = src[pos + 1:pos + 3].lstrip()[:1]
                if nxt not in ('"', "}"):
                    memo[pos] = (None, -1)
                    continue
                try:
                    cand, end = _DECODER.raw_decode(src, pos)
                except ValueError:
                    cand, end = None, -1
                memo[pos] = (_rec_row(cand, page) if isinstance(cand, dict) else None, end)
            hit, end = memo[pos]
            if end <= a1 or hit is None or not hit[0]:
                continue
            if hit[2]:
                take = (hit[0], hit[1], pos, end, hit[3])
                break
            if fallback is None:
                fallback = (hit[0], hit[1], pos, end, hit[3])
                budget = min(budget, 4)
        if take:
            row, key, start, end, rid = take
            if fallback and fallback[1] == key:
                # The inner, part-named object around the anchor may carry
                # what the enclosing record does not (``details.dealerDistanceMiles``).
                row = _merge_row(row, fallback[0])
            covered = end
            if key in done:
                # Another copy of the item (an app-state twin of a JSON-LD
                # node) may know what the first lacks: its distance.
                at = next((n for n, (_r, k) in enumerate(rows) if k == key), None)
                if at is not None:
                    rows[at] = (_merge_row(rows[at][0], row), key)
                continue
            done.add(key)
            if ids is not None and rid:
                ids[key] = rid
            # A part-named record does not cover its span: it may be a
            # wrapper whose child is the real item, found by a later anchor.
            w = next((n for n, (ws, we, _) in enumerate(wrappers) if ws <= start and end <= we), None)
            if w is not None:
                rows[wrappers.pop(w)[2]] = (row, key)
            else:
                rows.append((row, key))
        elif fallback:
            row, key, start, end, rid = fallback
            if key in done:
                at = next((n for n, (_r, k) in enumerate(rows) if k == key), None)
                if at is not None:
                    rows[at] = (_merge_row(rows[at][0], row), key)
            else:
                done.add(key)
                if ids is not None and rid:
                    ids[key] = rid
                wrappers.append((start, end, len(rows)))
                rows.append((row, key))


#: listing_card_href: the first link-valued attribute in a card's markup.
_CARD_HREF = re.compile(r'\s(?:href|data-[\w-]*(?:href|url))="(/[^"#\s]*|https?://[^"\s]+)"', re.I)
#: How far past an item's data attribute its card's link is looked for.
_CARD_SPAN = 3000
#: A distance a card prints as text: "(97 mi)", "97 miles away" -- not a
#: mileage ("19,612 mi."), which has neither the brackets nor "away".
_CARD_DIST_SPAN = 20000
_CARD_DIST = re.compile(r">[^<]{0,80}?\((\d{1,4}) mi\.?\)\s*<|>[^<]{0,80}?\b(\d{1,4}) mi(?:les?|\.)? away\b", re.I)


def _card_link(html: str, start: int, stop: int, page: str) -> str:
    """The card link right after a data attribute: same site, not an asset
    or a beacon, before the next card's data begins."""
    host = urllib.parse.urlsplit(page).netloc.lower() if page else ""
    for m in _CARD_HREF.finditer(html, start, min(stop, start + _CARD_SPAN)):
        u = html_mod.unescape(m.group(1))
        if _ASSET.search(u) or _TRACKING.search(u) or len(u) < 3:
            continue
        if u.startswith("http") and host and urllib.parse.urlsplit(u).netloc.lower() != host:
            continue
        return _item_url(u, page)
    return ""


#: listing_id_href: what an identifier is split from in a link's path.
_ATTR_NAME = re.compile(r'\s(?:href|data-[\w-]*(?:href|url))\Z', re.I)
_ID_SPLIT = re.compile(r"[/\-_.~,;]+")


def _id_links(html: str, page: str, ids: dict[str, tuple]) -> dict[str, str]:
    """key -> link for each item one of whose identifiers is a whole path segment
    (or a ``-``/``_``-separated part of one) of exactly one same-site page.
    Pages that keep item data in a script often render each card's link
    apart from it, keyed by the same id (``/details/458801235``)."""
    want: dict[str, str] = {}
    for key, rids in ids.items():
        for rid in rids:
            if len(rid) >= 5 and any(c.isdigit() for c in rid):
                # An id two items carry (a dealer's, a page's) names neither.
                want[rid] = "" if want.get(rid, key) != key else key
    want = {rid: key for rid, key in want.items() if key}
    if not want:
        return {}
    host = urllib.parse.urlsplit(page).netloc.lower() if page else ""
    found: dict[str, dict[str, str]] = {}  # id -> path -> first url
    # Every link-valued attribute, found by its value's start (two C-speed
    # scans; a regex over a 2 MB page costs tens of ms).
    for opener in ('="/', '="http'):
        at = html.find(opener)
        while at >= 0:
            if _ATTR_NAME.search(html[max(0, at - 60):at]):
                u = html_mod.unescape(html[at + 2:html.find('"', at + 2)])
                parts = urllib.parse.urlsplit(u)
                off_site = parts.netloc and host and parts.netloc.lower() != host
                if not (off_site or _ASSET.search(u) or _TRACKING.search(u)):
                    for tok in _ID_SPLIT.split(parts.path):
                        if tok in want:
                            found.setdefault(tok, {}).setdefault(parts.path, u)
            at = html.find(opener, at + 3)
    out: dict[str, str] = {}
    for rid, paths in found.items():
        key = want[rid]
        if len(paths) == 1 and key not in out:  # an id in two pages' paths names neither
            link = _item_url(next(iter(paths.values())), page)
            if link:
                out[key] = link
    # A link two items claim is the page's, not either item's.
    seen: dict[str, int] = {}
    for link in out.values():
        seen[link] = seen.get(link, 0) + 1
    return {k: v for k, v in out.items() if seen[v] == 1}


def _generic_rows(html: str, page: str = "") -> list[tuple[str, str]]:
    from .config import get_settings

    rows: list[tuple[str, str]] = []
    done: set[str] = set()
    s = get_settings()
    card = getattr(s, "listing_card_href", False)
    ids: dict[str, tuple] | None = {} if getattr(s, "listing_id_href", False) else None
    attrs = list(_DATA_ATTR.finditer(html))
    linked: dict[str, list[int]] = {}
    raws = [html_mod.unescape(m.group(1) or m.group(2)) for m in attrs]
    # Where each item's data starts: a card runs to the next one.
    item_at = [m.start() for m, raw in zip(attrs, raws) if any(t in raw for t in _ANCHOR_TAILS)]
    for i, m in enumerate(attrs):
        raw = raws[i]
        if not any(t in raw for t in _ANCHOR_TAILS):
            continue
        n = len(rows)
        _records_in(raw, page, rows, done, ids)
        stop = attrs[i + 1].start() if i + 1 < len(attrs) else len(html)
        # One item from this attribute and no distance in its data: its card
        # may print one ("Duluth, GA (97 mi)").
        if card and len(rows) == n + 1 and " | distance " not in rows[n][0]:
            # The card runs to the next item's data, past badge/tooltip attributes.
            nxt = next((item_at[j] for j in range(bisect.bisect_right(item_at, m.start()), len(item_at))),
                       len(html))
            d = _CARD_DIST.search(html, m.end(), min(nxt, m.end() + _CARD_DIST_SPAN))
            if d:
                rows[n] = (_merge_row(rows[n][0], f"x | distance {d.group(1) or d.group(2)}"), rows[n][1])
        # One item from this attribute and no link in its data: its card's
        # own link sits in the markup right after it.
        if card and len(rows) == n + 1 and not rows[n][0].rsplit(" | ", 1)[-1].startswith("http"):
            link = _card_link(html, m.end(), stop, page)
            if link:
                linked.setdefault(link, []).append(n)
    for link, at in linked.items():
        if len(at) == 1:  # a link two cards share is the page's, not either item's
            row, key = rows[at[0]]
            rows[at[0]] = (f"{row} | {link}", key)
    if any(t in html for t in _ANCHOR_TAILS):
        _records_in(html, page, rows, done, ids)
    if ids:
        bare = {key for row, key in rows if not row.rsplit(" | ", 1)[-1].startswith("http")}
        links = _id_links(html, page, {k: v for k, v in ids.items() if k in bare})
        for n, (row, key) in enumerate(rows):
            link = links.get(key)
            if link and not row.rsplit(" | ", 1)[-1].startswith("http"):
                rows[n] = (f"{row} | {link}", key)
    # Only vehicles have a history: a page with no VIN-keyed row skips the scan.
    if getattr(s, "listing_history", False) and any(_VIN_ONLY.fullmatch(k) for _r, k in rows):
        hist = _histories(html)
        for n, (row, key) in enumerate(rows):
            if key in hist and " | history: " not in row:
                head, sep, last = row.rpartition(" | ")
                if sep and last.startswith("http"):
                    rows[n] = (f"{head} | history: {hist[key]} | {last}", key)
                else:
                    rows[n] = (f"{row} | history: {hist[key]}", key)
    return rows


#: listing_result_count: key tails of a search's total (``totalCount``,
#: ``result_count``, ``ResultCount``...); the leading letter is left off so
#: either case of it matches, and the value must follow the key at once.
_COUNT_TAILS = ("otalCount", "otal_count", "otalResults", "otalListings", "otal_listings",
                "esultCount", "esult_count", "esultsCount")
_COUNT_VAL = re.compile(r'["\']?\s*[:=]\s*"?(\d{1,7})(?![\d.])')


def _result_count(html: str) -> int | None:
    """The search's total from the page's own data, or None. Pages carry
    facet and widget totals too, so a value counts only when at least two
    count fields give it and it outnumbers every other value they give."""
    seen: dict[int, int] = {}
    for tail in _COUNT_TAILS:
        at = html.find(tail)
        while at >= 0:
            # A whole key: the character before the tail's first letter is a
            # quote, a space or a dot, never part of a longer word.
            lead = html[at - 2:at - 1] if at >= 2 else ""
            m = _COUNT_VAL.match(html, at + len(tail))
            if m and lead in ('"', "'", " ", ".", "{", ",", "") and html[at - 1:at].lower() in ("t", "r"):
                v = int(m.group(1))
                seen[v] = seen.get(v, 0) + 1
            at = html.find(tail, at + len(tail))
    if not seen:
        return None
    best, n = max(seen.items(), key=lambda kv: kv[1])
    return best if n >= 2 and n > sum(seen.values()) - n else None


def _merge_row(kept: str, other: str) -> str:
    """``kept`` plus what a second copy of the same item knows and it lacks:
    a JSON-LD row has no distance, the page's app state has it."""
    parts = kept.split(" | ")

    def kind(p: str) -> str:
        m = _MERGE_PART.fullmatch(p)
        if m:
            return p.split()[0].lower() if p.lower().startswith(("distance", "history")) else "trans"
        # A digit: "Turbo Motors" (a dealer) is no engine.
        return "engine" if _ENGINE.fullmatch(p) and re.search(r"\d", p) and not p.startswith("http") else ""

    have = {kind(p) for p in parts[1:]}
    add = [p for p in other.split(" | ")[1:] if (k := kind(p)) and k not in have and not have.add(k)]
    if not add:
        return kept
    tail = [parts.pop()] if parts[-1].startswith("http") else []
    return " | ".join(parts + add + tail)


def with_listings(text: str, html: str, url: str = "") -> str:
    """structured_listings: ``text`` plus the page's listing cards, when the
    prose extraction did not keep them. Rows already present (by VIN, else
    name) are not repeated; a page with no listing data is returned as is."""
    from .config import get_settings

    s = get_settings()
    rows: list[tuple[str, str]] = []
    if "ld+json" in html and _LISTING_HINT.search(html):
        rows += [_ld_row(n, url) for n in _walk_ld(html) if _types(n) & _LISTING_TYPES]
    if getattr(s, "listing_generic", False):
        rows += _generic_rows(html, url)
    out, keys, at_key = [], set(), {}
    generic, sigs = getattr(s, "listing_generic", False), {}
    keep_links = getattr(s, "listing_keep_links", False)
    # Items the prose already names: key -> [name, vin, history, link], the
    # parts of each a reader could not rebuild from the prose.
    compact: dict[str, list[str]] = {}
    for row, key in rows:
        if not row or not key:
            continue
        if key in text:
            # listing_keep_links: the prose kept the item but not where it
            # lives; its detail link is the one thing a reader cannot rebuild.
            link = row.rsplit(" | ", 1)[-1] if keep_links else ""
            link = link if link.startswith("http") and link not in text else ""
            # listing_history: nor, usually, its history, which the prose
            # around a results card does not print.
            hist = next((p for p in row.split(" | ") if p.startswith("history: ")), "")
            hist = hist if hist and f"{key} | {hist}" not in text else ""
            if (link or hist) and (key in compact or len(compact) < _MAX_ROWS):
                e = compact.setdefault(key, [row.split(" | ", 1)[0], "", "", ""])
                e[1] = e[1] or (key if hist and f"VIN {key}" in row else "")
                e[2], e[3] = e[2] or hist, e[3] or link
            continue
        # The same item reached through JSON-LD and an attribute may differ
        # only in entity escaping (``&amp;#174;`` vs ``&#174;``).
        key = html_mod.unescape(html_mod.unescape(key)) if "&" in key else key
        if key in keys:
            if key in at_key:
                out[at_key[key]] = "- " + _merge_row(out[at_key[key]][2:], row)
            continue
        if generic:
            # Pages often hold each item twice (a card and a tracking
            # object), one copy without its VIN: at one price, a name
            # containing the other ("Used 2023 Ford ..." / "2023 Ford ...")
            # is one item unless both carry (different) VINs.
            # The fuller copy wins ("CT1000P310SSD5-01" vs its full title).
            name, _, rest = row.lower().partition(" | ")
            price, has_vin = rest.split(" | ")[0], " | vin " in row.lower()
            seen = sigs.setdefault(price, [])
            dup = next((j for j, (n, v, _at) in enumerate(seen)
                        if (name in n or n in name) and not (has_vin and v)), None)
            if dup is not None:
                n, v, at = seen[dup]
                # A VIN outranks length: it is the one field a reader keys on.
                if (has_vin, len(row)) > (v, len(out[at]) - 2):
                    out[at] = f"- {row}"
                    seen[dup] = (name, has_vin or v, at)
                keys.add(key)
                continue
            seen.append((name, has_vin, len(out)))
        keys.add(key)
        at_key[key] = len(out)
        out.append(f"- {row}")
        if len(out) >= _MAX_ROWS:
            break
    if compact:
        # One car may arrive twice: a JSON-LD copy keyed by name with its
        # link, a data copy keyed by VIN with its history. The VIN in the
        # link joins them.
        for key, e in list(compact.items()):
            if e[1] and not e[3]:
                other = next((k for k, o in compact.items() if k != key and o[3] and e[1] in o[3]), None)
                if other:
                    e[3] = compact.pop(other)[3]
        full = "\n".join(out)
        for name, vin, hist, link in compact.values():
            if (link and link not in full) or (hist and f"VIN {vin} | {hist}" not in full):
                parts = [name, f"VIN {vin}" if vin else "", hist, link]
                out.append("- " + " | ".join(x for x in parts if x))
    total = None
    if getattr(s, "listing_result_count", False) and (out or len(text) < 500):
        total = _result_count(html)
    if total == 0 and not out:
        # A search that matched nothing reads as just that, not as a page
        # whose content went missing.
        return f"{text}\n\n## Listings\n\nNo matches: the page's own data reports 0 results."
    if not out:
        return text
    head = f"The page's own data reports {total:,} results in all.\n\n" if total else ""
    return f"{text}\n\n## Listings\n\n{head}" + "\n".join(out)


def doc_from_fetch(url: str, html: str, *, source: str = "", snippet: str = "") -> Doc:
    from .config import get_settings

    text = to_markdown(html, url)
    if get_settings().structured_listings:
        text = with_listings(text, html, url)
    return Doc(
        url=url,
        title=title_of(html),
        snippet=snippet,
        text=text,
        source=source,
    )
