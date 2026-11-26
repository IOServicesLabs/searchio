"""Pages that load with their data in them but read as almost nothing
(#230-#233), pinned on bodies captured from the live sites and trimmed
(tests/fixtures/youtube_*, x_*, cars_*).

One class per knob. Knob off pins what a read returned before; knob on pins
the rows the page actually held.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from searchio.config import Settings, set_settings
from searchio.extract import _fuller_name, _packages, _url_keyword, doc_from_fetch, with_listings
from searchio.models import Doc, FetchResult, Query
from searchio.providers.marketplace import MarketplaceItems, is_vehicle_query, vehicle_items, vehicle_urls
from searchio.sitedata import X_SCRIPT_JUNK, is_youtube_list, site_text, x_posts_text, youtube_list_text
from tests._ab_baseline import BASELINE

FIX = Path(__file__).parent / "fixtures"
OFF = dict(extract_youtube_lists=False, extract_x_posts=False, listing_packages=False,
           items_vehicle_listings=False)

YT_CHANNEL = "https://www.youtube.com/@NASA/videos"
YT_RESULTS = "https://www.youtube.com/results?search_query=2022+f150+fx4+review"
X_PROFILE = "https://x.com/NASA"
X_STATUS = "https://x.com/jack/status/20"
CARMAX = "https://www.carmax.com/cars/ford/f-150?year=2022-2025"
AUTOTRADER = ("https://www.autotrader.com/cars-for-sale/used-cars/ford/f-150/ooltewah-tn"
              "?zip=37363&searchRadius=300&startYear=2022&keywordPhrases=fx4")
CARVANA = "https://www.carvana.com/cars/ford-f150"


def body(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.fixture
def use(tmp_path):
    def _use(**over):
        set_settings(Settings(state_dir=tmp_path, **{**BASELINE, **OFF, **over}))
    yield _use
    set_settings(None)


def rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


class TestYoutubeLists:
    def test_off_a_channel_tab_reads_as_its_footer(self, use):
        use()
        text = doc_from_fetch(YT_CHANNEL, body("youtube_channel_videos.html")).text
        assert len(text) < 300 and "watch?v=" not in text

    def test_on_a_channel_tab_lists_its_videos(self, use):
        use(extract_youtube_lists=True)
        text = doc_from_fetch(YT_CHANNEL, body("youtube_channel_videos.html")).text
        got = rows(text)
        assert len(got) == 30
        assert text.startswith("YouTube channel: NASA")
        first = got[0]
        assert "2026 Thanksgiving Message from the International Space Station" in first
        assert "| NASA |" in first and "55 thousand views" in first and "2 days ago" in first
        assert "| 2:46 |" in first and first.endswith("https://www.youtube.com/watch?v=TQUqnOtUC_M")

    def test_off_a_results_page_reads_as_its_footer(self, use):
        use()
        text = doc_from_fetch(YT_RESULTS, body("youtube_results_lockups.html")).text
        assert len(text) < 300 and "watch?v=" not in text

    def test_on_a_results_page_lists_videos_then_shorts(self, use):
        use(extract_youtube_lists=True)
        text = doc_from_fetch(YT_RESULTS, body("youtube_results_lockups.html")).text
        assert text.startswith('YouTube results for "2022 f150 fx4 review"')
        videos, _, shorts = text.partition("## Shorts")
        assert len(rows(videos)) == 20 and len(rows(shorts)) == 20
        first = rows(videos)[0]
        assert "I Got A 2022 F-150 XLT Fx4" in first and "Carter Eldridge" in first
        assert "1,889 views" in first and "| 15:39 |" in first
        assert first.endswith("https://www.youtube.com/watch?v=mWIAf-pcWe8")
        assert "/shorts/" in rows(shorts)[0]

    def test_an_ad_lockup_is_not_a_video(self, use):
        # The results page holds five lockupViewModel cards, all of them ads
        # (feedAdMetadataViewModel): none may be listed.
        text = youtube_list_text(body("youtube_results_lockups.html"), YT_RESULTS)
        assert "Super Duty" not in text and "x3RhM_lp8Zc" not in text

    def test_a_watch_page_is_left_to_its_own_reader(self):
        assert not is_youtube_list("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_list(YT_CHANNEL) and is_youtube_list(YT_RESULTS)
        assert not is_youtube_list("https://example.com/results?search_query=x")

    def test_a_page_without_data_keeps_its_text(self):
        s = Settings(extract_youtube_lists=True)
        assert site_text(YT_CHANNEL, "<html><body>nothing here</body></html>", "prose", s) == "prose"
        assert youtube_list_text("<script>var ytInitialData = {};</script>") == ""


class TestXPosts:
    def test_off_a_profile_reads_as_one_post_and_script_text(self, use):
        use()
        text = doc_from_fetch(X_PROFILE, body("x_profile.html")).text
        assert len(text) < 600
        assert "followers" not in text and "x.com/NASA/status/" not in text

    def test_on_a_profile_lists_the_account_and_its_posts(self, use):
        use(extract_x_posts=True)
        text = doc_from_fetch(X_PROFILE, body("x_profile.html")).text
        assert text.startswith("NASA (@NASA)")
        assert "followers" in text and "74,587 posts" in text
        got = rows(text)
        assert len(got) == 5
        assert got[0].startswith("- NASA (@NASA) | 2026-11-25 17:24 UTC | 135 replies, 217 reposts, 1,086 likes")
        assert "Applications for our next class of astronauts close in one week!" in text
        # t.co wrappers are replaced by where they go; media wrappers are dropped.
        assert "https://go.nasa.gov/astro2026" in text and "t.co/" not in text
        assert "https://x.com/NASA/status/2125641758432284978" in text
        assert "__xClientTextFormatters" not in text
        assert "signed-out visitor" in text

    def test_on_a_status_page_gives_the_post_then_its_replies(self, use):
        use(extract_x_posts=True)
        text = doc_from_fetch(X_STATUS, body("x_status.html")).text
        post, _, replies = text.partition("## Replies")
        assert post.startswith("## Post")
        assert "jack (@jack) | 2006-03-21 20:50 UTC" in post and "just setting up my twttr" in post
        assert "324,626 likes" in post
        assert len(rows(replies)) == 3 and "Hello from the future" in replies
        # The hero copy and the thread copy of the post are one post.
        assert text.count("just setting up my twttr") == 1

    def test_a_page_with_no_posts_only_loses_the_script_text(self):
        s = Settings(extract_x_posts=True)
        leaked = ('4hwindow.__xClientTextFormatters?.run("_R_2v_", "formatRelativePastTimestamp", '
                  '{"timestamp":1795627462000,"lang":"en"}); Sign in to X')
        out = site_text("https://x.com/search?q=f150", "<html><body>login</body></html>", leaked, s)
        assert "__xClientTextFormatters" not in out and "Sign in to X" in out
        assert x_posts_text("<html><script>var a = 1;</script></html>") == ""
        assert X_SCRIPT_JUNK.sub("", "plain prose") == "plain prose"

    def test_other_hosts_are_untouched(self):
        s = Settings(extract_x_posts=True, extract_youtube_lists=True)
        assert site_text("https://example.com/NASA", body("x_profile.html"), "kept", s) == "kept"


class TestListingPackages:
    def test_off_no_row_names_a_package_field(self, use):
        use()
        text = doc_from_fetch(CARMAX, body("cars_carmax_results.html")).text
        assert "pkgs:" not in text and "FX4" not in text

    def test_on_rows_carry_the_records_packages(self, use):
        use(listing_packages=True)
        got = rows(doc_from_fetch(CARMAX, body("cars_carmax_results.html")).text)
        fx4 = [r for r in got if "FX4" in r]
        assert len(fx4) == 7
        row = next(r for r in got if "1FTFW1EDXPFA88735" in r)
        assert row.startswith("- 2023 Ford F150 Platinum | $44,998 | 68,231 mi")
        assert "pkgs: FX4 Package; Utility Package" in row and "distance 0" in row

    def test_off_an_escaped_payload_gives_title_only_rows(self, use):
        use()
        text = doc_from_fetch(AUTOTRADER, body("cars_autotrader_results.html")).text
        listing = [r for r in rows(text) if "VIN " in r]
        assert listing and not any("pkgs:" in r or " | distance " in r for r in listing)

    def test_on_an_escaped_payload_adds_packages_and_distance(self, use):
        use(listing_packages=True)
        text = doc_from_fetch(AUTOTRADER, body("cars_autotrader_results.html")).text
        row = next(r for r in rows(text) if "1FTFW1EDXNFB55685" in r)
        assert "Used 2022 Ford F150 Platinum" in row and "$45,094" in row
        assert "pkgs: Trailer Tow Package; FX4 Off-Road Package" in row and "distance 7" in row
        assert row.endswith("https://www.autotrader.com/cars-for-sale/vehicle/790482486")
        assert 'The page URL filters these results by keyword "fx4".' in text
        # A streamed payload's reference to a value is not a package name.
        assert "$c:" not in text

    def test_on_the_payloads_fuller_name_replaces_the_bare_one(self, use):
        use()
        before = doc_from_fetch(CARVANA, body("cars_carvana_results.html")).text
        assert "2026 Ford F150 SuperCrew Cab | $53,590" in before
        use(listing_packages=True)
        after = doc_from_fetch(CARVANA, body("cars_carvana_results.html")).text
        row = next(r for r in rows(after) if "1FTEW2L56TFB41809" in r)
        assert "2026 Ford F150 SuperCrew Cab STX Pickup 4D 5 1/2 ft | $53,590" in row
        # Named packages come before loose options.
        assert "pkgs: F-150 Lobo Package;" in row
        assert len(rows(after)) == len(rows(before))

    def test_package_names_by_shape(self):
        assert _packages({"packages": ["FX4 Package", "Utility Package"]}) == "pkgs: FX4 Package; Utility Package"
        assert _packages({"packages": [{"name": "FX4 Off-Road Package"}, {"name": "$c:props:x"}]}) \
            == "pkgs: FX4 Off-Road Package"
        # A long equipment list gives up only its packages.
        assert _packages({"options": ["Heated Seats", "Off Road Package", "CarPlay"]}) == "pkgs: Off Road Package"
        assert _packages({"highlightedFeatures": "FX4 Package,Bed Cover,4WD/AWD"}) == "pkgs: FX4 Package"
        assert _packages({"factoryUpgrades": {"packagesAndOptions": [{"displayName": "Lobo Package"}]}}) \
            == "pkgs: Lobo Package"
        # One the name already states is not repeated; none is "".
        assert _packages({"packages": ["Sport Package"]}, "2023 Ford F150 w/ Sport Package") == ""
        assert _packages({"options": ["Heated Seats"], "packages": []}) == ""

    def test_url_keyword_and_fuller_name(self):
        assert _url_keyword("https://www.cars.com/shopping/results/?zip=37363&keyword=fx4") == "fx4"
        assert _url_keyword("https://a.example/s?keywordPhrases=fx4%20tow") == "fx4 tow"
        assert _url_keyword("https://a.example/s?q=fx4&zip=1") == ""
        assert _fuller_name("Used 2026 Ford F150", "2026 Ford F150 STX") == "Used 2026 Ford F150 STX"
        assert _fuller_name("2026 Ford F150 STX", "2026 Ford F150") == "2026 Ford F150 STX"
        assert _fuller_name("2026 Ford F150", "2024 Toyota Tundra SR5") == "2026 Ford F150"

    def test_a_page_with_no_listing_is_unchanged(self, use):
        use(listing_packages=True)
        html = "<html><body><p>" + "plain prose. " * 80 + "</p></body></html>"
        assert with_listings("prose", html, "https://example.com/?keyword=fx4") == "prose"


EBAY = "https://www.ebay.com/itm/234559138525"
DECAL = ('<html><head><script type="application/ld+json">{"@context":"https://schema.org","@type":"Product",'
         '"name":"2022 Ford F150 bedside decal FX4 Off Road","url":"' + EBAY + '","offers":{"@type":"Offer",'
         '"price":"80.86","priceCurrency":"USD"}}</script></head><body>decal</body></html>')


class _Ladder:
    session_id = "t"

    def __init__(self, pages):
        self.pages, self.fetched = pages, []

    async def fetch(self, url, **_kw):
        self.fetched.append(url)
        if url not in self.pages:
            raise RuntimeError("refused")
        return FetchResult(url=url, final_url=url, status=200, body=self.pages[url], tier=0, via="http")


class _Router:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    async def search(self, q, **_kw):
        self.queries.append(q)

        class R:
            failed: dict = {}
        r = R()
        r.docs = [d for d in self.docs if not q.domains or any(d.domain.endswith(x) for x in q.domains)]
        return r


class _Ctx:
    def __init__(self, settings, ladder, router):
        self.settings, self.ladder, self.router, self.session_id = settings, ladder, router, "t"


class TestVehicleItems:
    def test_vehicle_query_shape(self):
        assert is_vehicle_query("2022 ford f-150 fx4")
        assert is_vehicle_query("used toyota tacoma trd for sale")
        assert not is_vehicle_query("ford f-150 fx4 decals 2022")
        assert not is_vehicle_query("2022 f150 floor mats")
        assert not is_vehicle_query("sony wh-1000xm5 2022")
        assert not is_vehicle_query("ford f-150")  # no year, no for-sale wording

    def test_rows_become_vin_keyed_items_matching_every_term(self):
        items = vehicle_items(body("cars_autotrader_results.html"), AUTOTRADER, "2022 ford f-150 fx4")
        assert [it.attributes["vin"] for it in items] == ["1FTFW1EDXNFB55685"]
        it = items[0]
        assert it.price.amount == 45094 and it.identity == "vin:1FTFW1EDXNFB55685"
        assert "FX4 Off-Road Package" in it.attributes["packages"] and it.attributes["distance_mi"] == "7"
        assert it.attributes["mileage"] == "48,190 mi" and it.attributes["location"] == "Chattanooga, TN"
        assert it.url == "https://www.autotrader.com/cars-for-sale/vehicle/790482486"

    def test_no_car_of_the_asked_year_falls_back_to_other_years(self):
        items = vehicle_items(body("cars_carmax_results.html"), CARMAX, "2022 ford f-150 fx4")
        assert len(items) == 5
        assert all("FX4" in it.attributes["packages"] and "F150" in it.title for it in items)
        assert all(it.meta.get("year_match") is False for it in items)
        # A model the page does not sell matches nothing.
        assert vehicle_items(body("cars_carmax_results.html"), CARMAX, "2022 toyota tundra") == []

    def _run(self, tmp_path, knob, query="2022 ford f-150 fx4"):
        s = Settings(state_dir=tmp_path, **{**BASELINE, **OFF, "items_vehicle_listings": knob})
        docs = [Doc(url=AUTOTRADER, title="F-150 for sale"), Doc(url=CARMAX, title="Used F-150"),
                Doc(url=EBAY, title="FX4 decal", snippet="$80.86")]
        ladder = _Ladder({AUTOTRADER: body("cars_autotrader_results.html"),
                          CARMAX: body("cars_carmax_results.html"), EBAY: DECAL})
        router = _Router(docs)
        set_settings(s)
        try:
            items = asyncio.run(MarketplaceItems().find_items(
                Query(text=query, intent="shopping", k=10), _Ctx(s, ladder, router)))
        finally:
            set_settings(None)
        return items, ladder, router

    def test_off_a_vehicle_query_goes_to_the_retail_sites(self, tmp_path):
        items, ladder, router = self._run(tmp_path, False)
        assert len(router.queries) == 1
        assert "ebay.com" in router.queries[0].domains and "autotrader.com" not in router.queries[0].domains
        assert ladder.fetched == [EBAY]
        assert [it.title for it in items] == ["2022 Ford F150 bedside decal FX4 Off Road"]

    def test_on_a_vehicle_query_reads_the_vehicle_marketplaces(self, tmp_path):
        items, ladder, router = self._run(tmp_path, True)
        # One plain search finds the vehicle pages; no site-scoped query is needed.
        assert [q.domains for q in router.queries] == [[]]
        assert router.queries[0].text == "2022 ford f-150 fx4 for sale"
        assert set(ladder.fetched) == {AUTOTRADER, CARMAX}
        prices = [it.price.amount for it in items]
        assert prices == sorted(prices) and len(items) == 6
        assert len({it.identity for it in items}) == 6
        assert all("fx4" in (it.title + it.attributes.get("packages", "")).lower() for it in items)

    def test_on_a_part_query_still_goes_to_the_retail_sites(self, tmp_path):
        items, ladder, router = self._run(tmp_path, True, query="2022 ford f-150 fx4 decals")
        assert "ebay.com" in router.queries[0].domains and ladder.fetched == [EBAY]
        assert len(items) == 1 and items[0].price.amount == 80.86

    def test_on_a_slow_page_does_not_hold_the_answer(self, tmp_path, monkeypatch):
        import time

        class Slow(_Ladder):
            async def fetch(self, url, **kw):
                if url == CARMAX:
                    await asyncio.sleep(30)
                return await super().fetch(url, **kw)

        monkeypatch.setattr(MarketplaceItems, "VEHICLE_BUDGET_S", 1.5)
        s = Settings(state_dir=tmp_path, **{**BASELINE, **OFF, "items_vehicle_listings": True})
        ladder = Slow({AUTOTRADER: body("cars_autotrader_results.html"), CARMAX: body("cars_carmax_results.html")})
        router = _Router([Doc(url=AUTOTRADER, title="a"), Doc(url=CARMAX, title="b")])
        set_settings(s)
        t0 = time.perf_counter()
        try:
            items = asyncio.run(MarketplaceItems().find_items(
                Query(text="2022 ford f-150 fx4", intent="shopping", k=10), _Ctx(s, ladder, router)))
        finally:
            set_settings(None)
        assert time.perf_counter() - t0 < 10
        assert [it.attributes["vin"] for it in items] == ["1FTFW1EDXNFB55685"]

    def test_the_tool_prints_what_tells_one_car_from_the_next(self):
        from searchio.swarm.tools import ToolBridge

        found = vehicle_items(body("cars_autotrader_results.html"), AUTOTRADER, "2022 ford f-150 fx4")

        class Eng:
            async def find_items(self, *_a, **_kw):
                return found

        text, err = asyncio.run(ToolBridge(Eng()).dispatch("find_items", {"query": "2022 ford f-150 fx4"}))
        assert not err
        assert "48,190 mi · Chattanooga, TN · 7 mi away · packages: Trailer Tow Package; FX4 Off-Road Package" in text
        assert "VIN 1FTFW1EDXNFB55685" in text and "45,094.00 USD" in text

    def test_on_no_vehicle_page_in_the_plain_search_tries_the_scoped_one(self, tmp_path):
        s = Settings(state_dir=tmp_path, **{**BASELINE, **OFF, "items_vehicle_listings": True})

        class Scoped(_Router):
            async def search(self, q, **kw):
                r = await super().search(q, **kw)
                if not q.domains:
                    r.docs = [Doc(url="https://en.wikipedia.org/wiki/Ford_F-Series", title="F-Series")]
                return r

        ladder = _Ladder({AUTOTRADER: body("cars_autotrader_results.html")})
        router = Scoped([Doc(url=AUTOTRADER, title="a")])
        set_settings(s)
        try:
            items = asyncio.run(MarketplaceItems().find_items(
                Query(text="2022 ford f-150 fx4", intent="shopping", k=10), _Ctx(s, ladder, router)))
        finally:
            set_settings(None)
        assert [bool(q.domains) for q in router.queries] == [False, True]
        assert "autotrader.com" in router.queries[1].domains
        assert [it.attributes["vin"] for it in items] == ["1FTFW1EDXNFB55685"]

    def test_direct_pages_for_a_vehicle_query(self):
        assert vehicle_urls("2022 ford f-150 fx4") == [
            "https://www.autotrader.com/cars-for-sale/2022/ford/f150?keywordPhrases=fx4",
            "https://www.kbb.com/cars-for-sale/used/2022/ford/f150?keywordPhrases=fx4",
            "https://www.carmax.com/cars/ford/f-150?year=2022-2022",
            "https://www.carvana.com/cars/ford-f150",
        ]
        assert vehicle_urls("used chevy silverado for sale")[0]             == "https://www.autotrader.com/cars-for-sale/chevrolet/silverado"
        assert vehicle_urls("2022 ford") == [] and vehicle_urls("sony headphones") == []

    def test_on_every_search_refusing_reads_the_marketplaces_directly(self, tmp_path):
        s = Settings(state_dir=tmp_path, **{**BASELINE, **OFF, "items_vehicle_listings": True})

        class Down(_Router):
            async def search(self, q, **kw):
                self.queries.append(q)
                raise RuntimeError("every engine refused")

        direct = "https://www.autotrader.com/cars-for-sale/2022/ford/f150?keywordPhrases=fx4"
        ladder = _Ladder({direct: body("cars_autotrader_results.html")})
        router = Down([])
        set_settings(s)
        try:
            items = asyncio.run(MarketplaceItems().find_items(
                Query(text="2022 ford f-150 fx4", intent="shopping", k=10), _Ctx(s, ladder, router)))
        finally:
            set_settings(None)
        assert direct in ladder.fetched and len(ladder.fetched) == 4
        assert [it.attributes["vin"] for it in items] == ["1FTFW1EDXNFB55685"]
