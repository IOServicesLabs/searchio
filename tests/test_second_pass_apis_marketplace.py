"""apis.py + marketplace.py second pass (iteration 58, DeepSeek + probes).

Every test here bit RED before its fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from searchio.errors import ProviderError
from searchio.models import Doc, Query
from searchio.providers import apis
from searchio.providers.apis import ArxivProvider, GitHubRepos, _envelope_error
from searchio.providers.marketplace import MarketplaceItems, is_product_url


@pytest.fixture
def settings(tmp_path):
    from searchio.config import Settings

    return Settings(state_dir=tmp_path, cache_enabled=False)


class _Res:
    def __init__(self, body: str, final_url: str = "", tier: int = 0):
        self.body = body
        self.final_url = final_url
        self.tier = tier
        self.status = 200


def _ctx(body: str, final_url: str = ""):
    async def fetch(url, **kw):
        return _Res(body, final_url or url)
    return SimpleNamespace(ladder=SimpleNamespace(fetch=fetch, session_id="t"), router=None,
                           settings=None, session_id="t")


class TestArxivRefusesNonAtomBodies:
    # Bug 116: a body that is not an Atom feed (a maintenance HTML page, a
    # proxy error) had no <entry> and shipped as an honest "no results".
    async def test_html_body_is_a_refusal(self):
        with pytest.raises(ProviderError):
            await ArxivProvider().search(Query(text="x", intent="academic"), _ctx("<html><body>503 Service Unavailable</body></html>"))

    async def test_empty_atom_feed_is_an_honest_empty(self):
        feed = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
                '<opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0'
                '</opensearch:totalResults></feed>')
        assert await ArxivProvider().search(Query(text="x", intent="academic"), _ctx(feed)) == []


class TestEnvelopeRiders:
    def test_nested_message_is_named(self):
        # Rider: {"error": {"message": "x"}} was reported as the dict repr.
        with pytest.raises(ProviderError, match="quota exceeded"):
            _envelope_error("p", {"error": {"message": "quota exceeded"}}, "items", "error")

    async def test_github_non_object_payload_is_a_refusal(self, monkeypatch):
        async def fake_json(ctx, url):
            return ["not", "an", "object"]
        monkeypatch.setattr(apis, "_json", fake_json)
        with pytest.raises(ProviderError):
            await GitHubRepos().search(Query(text="x", intent="code"), _ctx(""))


class TestProductUrlsAreNotEditorial:
    # Bug 119 (DeepSeek #6 + probe): any 5-digit run in the path made a blog
    # post, a news article or a help page a "product page" to fetch.
    @pytest.mark.parametrize("url", [
        "https://x.com/blog/12345-best-headphones", "https://x.com/news/2024/12345",
        "https://x.com/help/12345", "https://x.com/support/articles/12345",
        "https://x.com/community/questions/12345", "https://x.com/dp/B09XS7JWHH/reviews",
    ])
    def test_editorial_paths_rejected(self, url):
        assert not is_product_url(url), url

    @pytest.mark.parametrize("url", [
        "https://www.amazon.com/dp/B09XS7JWHH", "https://www.bestbuy.com/site/sony-wh1000xm5/6505727.p",
        "https://www.walmart.com/ip/12345", "https://www.target.com/p/x/-/A-12345",
    ])
    def test_product_paths_still_accepted(self, url):
        assert is_product_url(url), url


_LD = '<script type="application/ld+json">%s</script>'


class TestProductPageScope:
    def _doc(self, url="https://www.amazon.com/dp/B0MAIN00001"):
        return Doc(url=url, title="Main", snippet="$10", source="ddg")

    async def test_off_domain_redirect_is_not_a_listing(self):
        # Bug 117 (DeepSeek #4): a scoped amazon.com candidate that redirected
        # to amazon.com.mx was extracted anyway -- MXN prices tagged amazon.com,
        # the bug-60 currency mix through the back door.
        body = _LD % '{"@type":"Product","name":"Thing","offers":{"price":"199","priceCurrency":"MXN"}}'
        prov = MarketplaceItems(domains=("amazon.com",))
        with pytest.raises(ProviderError, match="off-scope|redirect"):
            await prov._one(self._doc(), _ctx(body, final_url="https://www.amazon.com.mx/dp/B0MAIN00001"),
                            allowed=("amazon.com",))

    async def test_all_pages_off_scope_is_a_refusal(self, monkeypatch):
        body = _LD % '{"@type":"Product","name":"Thing","offers":{"price":"199","priceCurrency":"MXN"}}'
        prov = MarketplaceItems(domains=("amazon.com",))

        async def discover(q, ctx):
            return [self._doc()]
        monkeypatch.setattr(prov, "_discover", discover)
        with pytest.raises(ProviderError):
            await prov.find_items(Query(text="thing", intent="shopping"),
                                  _ctx(body, final_url="https://www.amazon.com.mx/dp/B0MAIN00001"))

    async def test_related_products_on_the_page_are_not_listings(self):
        # Bug 118 (DeepSeek #5): every Product node on a product page went out
        # as a listing for the query -- the "customers also bought" carousel
        # became three cheaper "offers" for a different product.
        page = "https://www.amazon.com/dp/B0MAIN00001"
        body = _LD % ('[{"@type":"Product","name":"Main","url":"%s","offers":{"price":"300","priceCurrency":"USD"}},'
                      '{"@type":"Product","name":"Other A","url":"https://www.amazon.com/dp/B0OTHER0001","offers":{"price":"20","priceCurrency":"USD"}},'
                      '{"@type":"Product","name":"Other B","url":"https://www.amazon.com/dp/B0OTHER0002","offers":{"price":"30","priceCurrency":"USD"}}]' % page)
        items = await MarketplaceItems()._one(self._doc(page), _ctx(body), allowed=("amazon.com",))
        assert [it.title for it in items] == ["Main"]

    async def test_single_product_with_canonical_url_still_returned(self):
        page = "https://www.amazon.com/dp/B0MAIN00001?tag=aff"
        body = _LD % '{"@type":"Product","name":"Main","url":"https://www.amazon.com/Main-Thing/dp/B0MAIN00001","offers":{"price":"300","priceCurrency":"USD"}}'
        items = await MarketplaceItems()._one(self._doc(page), _ctx(body), allowed=("amazon.com",))
        assert [it.title for it in items] == ["Main"]


class TestDetailPageVerification:
    # Item.verified means "detail page opened and confirmed live" (models.py).
    # The structured path left it False, so an offer confirmed in the page's
    # own JSON-LD went out unverified -- and the snippet fallback, whose price
    # never came off the page, looked identical to it downstream.

    def _doc(self, url="https://www.amazon.com/dp/B0MAIN00001"):
        return Doc(url=url, title="Main", snippet="$10", source="ddg")

    async def test_structured_offer_is_verified(self):
        body = _LD % ('{"@type":"Product","name":"Main",'
                      '"offers":{"price":"300","priceCurrency":"USD",'
                      '"availability":"https://schema.org/InStock"}}')
        items = await MarketplaceItems()._one(self._doc(), _ctx(body), allowed=("amazon.com",))
        assert len(items) == 1
        assert items[0].verified is True
        assert items[0].meta["extraction"] == "jsonld"

    async def test_snippet_fallback_stays_unverified(self):
        # No JSON-LD on the page at all: the price falls back to the SERP
        # snippet, which the opened page itself never confirmed.
        items = await MarketplaceItems()._one(
            self._doc(), _ctx("<html><body><h1>Main</h1></body></html>"),
            allowed=("amazon.com",))
        assert len(items) == 1
        assert items[0].verified is False
        assert items[0].meta["extraction"] == "snippet_fallback"
        assert items[0].price.amount == 10.0  # from the "$10" snippet


class TestEngineFindItemsDoesNotSwallowRefusals:
    # Bug 120: Engine.find_items caught every provider exception into [] --
    # bug 61's "all product pages failed" refusal never reached the tool,
    # the API or the CLI; an outage read as "No listings found".
    def _engine(self, settings, provider):
        from searchio.engine import Engine

        eng = Engine.__new__(Engine)
        eng.s = settings
        eng.ladder = SimpleNamespace(session_id="t")
        eng.router = None
        eng.registry = SimpleNamespace(get=lambda name: provider if name == "marketplace" else None)
        return eng

    async def test_provider_refusal_propagates_when_nothing_answered(self, settings):
        class Broken:
            async def find_items(self, q, ctx):
                raise ProviderError("marketplace", "all 3 product pages failed: amazon.com: TransientError")
        settings.max_tier = 1
        eng = self._engine(settings, Broken())
        with pytest.raises(ProviderError, match="product pages failed"):
            await eng.find_items("sony wh-1000xm5")

    async def test_honest_empty_stays_empty(self, settings):
        class Empty:
            async def find_items(self, q, ctx):
                return []
        settings.max_tier = 1
        assert await self._engine(settings, Empty()).find_items("sony wh-1000xm5") == []

    async def test_tool_reports_the_failure_instead_of_no_listings(self, settings):
        from searchio.swarm.tools import ToolBridge

        class Eng:
            s = settings
            router = SimpleNamespace(registry=SimpleNamespace(all=lambda: []))
            ladder = SimpleNamespace()

            async def find_items(self, text, domains=None):
                raise ProviderError("marketplace", "all 3 product pages failed: amazon.com: TransientError")
        text, err = await ToolBridge(Eng()).dispatch("find_items", {"query": "x"})
        assert "product pages failed" in text and "No listings found" not in text


class TestDiscoveryEmptyIsNotAFailure:
    # Bug 129 (blast radius of bug 120, caught live by shop.items_empty): a
    # nonsense query for which EVERY provider answered with zero docs raised
    # "discovery failed: {}" -- and once find_items stopped swallowing
    # refusals, an honest empty became a provider failure on the tool, the
    # API and the CLI.
    async def _run(self, docs, failed):
        prov = MarketplaceItems(domains=("amazon.com",))

        class R:
            async def search(self, q, **kw):
                return SimpleNamespace(docs=docs, used=[], failed=failed, filtered={}, per_provider={}, elapsed_ms=1)
        ctx = SimpleNamespace(router=R(), ladder=SimpleNamespace(session_id="t"), settings=None, session_id="t")
        return await prov.find_items(Query(text="zxqwv nonexistent widget", intent="shopping"), ctx)

    async def test_all_providers_answered_nothing_is_an_empty(self):
        assert await self._run([], {}) == []

    async def test_all_providers_failed_is_a_refusal(self):
        with pytest.raises(ProviderError, match="discovery failed"):
            await self._run([], {"duckduckgo": "circuit open", "bing": "circuit open"})


class TestApiTextIsDecoded:
    # Bug 121 (DeepSeek drop list, bug 111's sibling): arXiv titles and
    # summaries kept their XML entities, Wikipedia snippets their HTML
    # entities, Crossref abstracts their JATS tags -- "A &amp; B" and
    # "<jats:p>..." reached the agent verbatim.
    async def test_arxiv_entities_decoded(self):
        feed = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1234.5678v1</id>'
                '<title>Graphs &amp; Networks: a &quot;survey&quot;</title><summary>x &lt; y</summary>'
                '<published>2024-01-02T00:00:00Z</published></entry></feed>')
        docs = await ArxivProvider().search(Query(text="x", intent="academic"), _ctx(feed))
        assert docs[0].title == 'Graphs & Networks: a "survey"' and docs[0].snippet == "x < y"

    async def test_wikipedia_snippet_entities_decoded(self, monkeypatch):
        from searchio.providers.apis import Wikipedia

        async def fake_json(ctx, url):
            return {"query": {"search": [{"title": "Rock & Roll", "snippet": 'the <span class="searchmatch">term</span> &quot;rock&quot; &amp; roll',
                                          "timestamp": "2024-01-01T00:00:00Z"}]}}
        monkeypatch.setattr(apis, "_json", fake_json)
        docs = await Wikipedia().search(Query(text="x", intent="reference"), _ctx(""))
        assert docs[0].snippet == 'the term "rock" & roll'

    async def test_crossref_jats_stripped(self, monkeypatch):
        from searchio.providers.apis import Crossref

        async def fake_json(ctx, url):
            return {"status": "ok", "message": {"items": [{"URL": "https://doi.org/10.1/x", "title": ["T"],
                    "abstract": "<jats:p>We study <jats:italic>graphs</jats:italic> &amp; networks.</jats:p>"}]}}
        monkeypatch.setattr(apis, "_json", fake_json)
        docs = await Crossref().search(Query(text="x", intent="academic"), _ctx(""))
        assert docs[0].snippet == "We study graphs & networks."
