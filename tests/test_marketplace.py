"""Product-URL classification.

Every rejection case here was something that actually reached the item list and
rendered as a priceless "listing" that was really a department.
"""

from __future__ import annotations

import pytest

from searchio.providers.marketplace import is_product_url

DETAIL_PAGES = [
    "https://www.amazon.com/clp/B09XS7JWHH",
    "https://www.amazon.com/Sony-Headphones/dp/B09XS7JWHH/ref=sr_1_1",
    "https://www.walmart.com/ip/Sony-WH-1000XM5/1234567",
    "https://www.target.com/p/sony-wh-1000xm5/-/A-85978234",
    "https://www.bestbuy.com/site/sony-wh1000xm5/6505727.p",
    "https://www.ebay.com/itm/276123456789",
    "https://www.logitech.com/en-us/shop/p/mx-master-3s-mouse.910-006556",
    "https://shop.example.com/products/blue-widget",
]

CATEGORY_PAGES = [
    "https://www.logitech.com/en-us/products",
    # Best Buy serves products and departments under the same /site/ prefix;
    # only the product carries a SKU.
    "https://www.bestbuy.com/site/logitech/computer-accessories",
    "https://www.bestbuy.com/site/shop/mice-trackballs",
    "https://www.amazon.com/s?k=sony+headphones",
    "https://www.walmart.com/browse/electronics",
    "https://www.target.com/c/headphones",
    "https://en.wikipedia.org/wiki/List_of_Logitech_products",
]


@pytest.mark.parametrize("url", DETAIL_PAGES)
def test_detail_pages_accepted(url):
    assert is_product_url(url), url


@pytest.mark.parametrize("url", CATEGORY_PAGES)
def test_category_pages_rejected(url):
    assert not is_product_url(url), url
