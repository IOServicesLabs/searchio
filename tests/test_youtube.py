"""YouTube search: ytInitialData extraction, renderer parsing, freshness.

The page fixture is a real ``/results?search_query=rust+programming+language``
response captured with a plain HTTP client -- which is the point of the
provider. The synthetic fixtures below exist only to exercise shapes one
capture does not happen to contain (a Shorts reel, a live card, malformed
payloads).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from searchio.config import Settings
from searchio.models import Query
from searchio.providers.base import ProviderContext
from searchio.providers.registry import default_providers
from searchio.providers.youtube import (
    YouTube,
    _age_days,
    _duration_s,
    _views_int,
    extract_initial_data,
    filter_freshness,
    parse_videos,
)

FIXTURE = Path(__file__).parent / "fixtures" / "youtube_results.html"


@pytest.fixture(scope="module")
def html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# ── payload extraction ───────────────────────────────────────────────────────


def test_extract_initial_data_from_real_page(html):
    data = extract_initial_data(html)
    assert data is not None
    assert "contents" in data


def test_balanced_scan_survives_braces_inside_strings():
    """A title containing "}" must not end the object early."""
    payload = {"contents": {"title": "why } breaks naive scanners", "n": 1}}
    doc = "<script>var ytInitialData = " + json.dumps(payload) + ";</script>"
    assert extract_initial_data(doc) == payload


def test_missing_marker_and_malformed_payload_return_none():
    assert extract_initial_data("<html><body>nothing here</body></html>") is None
    assert extract_initial_data('<script>var ytInitialData = {"a": </script>') is None


# ── field helpers ────────────────────────────────────────────────────────────


def test_age_parsing():
    assert _age_days("3 hours ago") == pytest.approx(0.125)
    assert _age_days("5 days ago") == 5.0
    assert _age_days("2 weeks ago") == 14.0
    assert _age_days("3 months ago") == 90.0
    assert _age_days("2 years ago") == 730.0
    assert _age_days("Premieres in 2 hours") is None


def test_view_count_parsing():
    assert _views_int("607,025 views") == 607025
    assert _views_int("No views") == 0  # was None until iteration 43: zero is a count
    assert _views_int("1,234 watching") is None


def test_duration_parsing():
    assert _duration_s("3:05:04") == 3 * 3600 + 5 * 60 + 4
    assert _duration_s("0:42") == 42
    assert _duration_s("LIVE") is None


# ── whole-page parse of the real fixture ─────────────────────────────────────


def test_parses_real_results_page(html):
    docs = parse_videos(html)
    # 17 renderers on the captured page; the count is asserted as a floor so a
    # ranking shuffle on YouTube's side does not break the suite.
    assert len(docs) >= 15

    first = docs[0]
    assert first.url.startswith("https://www.youtube.com/watch?v=")
    assert "list=" not in first.url and "&pp=" not in first.url  # canonical
    assert first.title
    assert first.meta["channel"]
    assert first.meta["views"] and first.meta["views"] > 1000
    assert first.meta["age_days"] is not None
    assert first.rank == 0
    assert docs[1].rank == 1


def test_ranks_follow_page_order(html):
    docs = parse_videos(html)
    assert [d.rank for d in docs] == list(range(len(docs)))


def test_duplicate_renderers_dedupe():
    payload = {
        "contents": {
            "a": {"videoRenderer": {"videoId": "abc123", "title": {"runs": [{"text": "One"}]}}},
            "b": {"videoRenderer": {"videoId": "abc123", "title": {"runs": [{"text": "One"}]}}},
        }
    }
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    docs = parse_videos(html)
    assert len(docs) == 1


def test_renderer_without_title_is_skipped():
    """Ad cards carry a bare videoId; there is no video to name."""
    payload = {"contents": {"videoRenderer": {"videoId": "abc123"}}}
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    assert parse_videos(html) == []


def test_reels_parse_as_videos():
    payload = {
        "contents": {
            "reelItemRenderer": {
                "videoId": "reel9",
                "headline": {"runs": [{"text": "A very short take"}]},
                "ownerText": {"runs": [{"text": "SomeChannel"}]},
                "viewCountText": {"simpleText": "1.2M views"},
            }
        }
    }
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    docs = parse_videos(html)
    assert len(docs) == 1
    assert docs[0].meta["shorts"] is True
    assert docs[0].meta["age_days"] is None  # no age on a reel card
    assert docs[0].url == "https://www.youtube.com/watch?v=reel9"


def test_live_card_counts_as_happening_now():
    payload = {
        "contents": {
            "videoRenderer": {
                "videoId": "live1",
                "title": {"runs": [{"text": "LIVE: launch"}]},
                "viewCountText": {"simpleText": "12,345 watching"},
                "badges": [{"metadataBadgeRenderer": {"label": "LIVE"}}],
            }
        }
    }
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    docs = parse_videos(html)
    assert docs[0].meta["live"] is True
    assert docs[0].meta["age_days"] == 0.0  # live is now, so any freshness passes
    assert docs[0].published == dt.date.today().isoformat()  # live IS today


def test_published_is_an_absolute_date_not_relative_text():
    """`published` is a machine field consumed by the router's stale
    post-filter, which parses it as an ISO date. Shipping YouTube's relative
    phrasing ("3 days ago") made video results unprovable and they escaped
    every downstream freshness bound -- caught by bench/span.py's bulk
    date-sanity detector (iteration 11). The relative text stays in
    meta["age_text"] for display."""
    payload = {
        "contents": {
            "videoRenderer": {
                "videoId": "abc123",
                "title": {"runs": [{"text": "A talk"}]},
                "publishedTimeText": {"simpleText": "3 days ago"},
            }
        }
    }
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    (doc,) = parse_videos(html)
    want = (dt.date.today() - dt.timedelta(days=3)).isoformat()
    assert doc.published == want
    assert doc.meta["age_text"] == "3 days ago"
    # And the router's parser can actually read it.
    assert dt.date.fromisoformat(doc.published) == dt.date.today() - dt.timedelta(days=3)


def test_absolute_date_lets_the_router_stale_filter_bind():
    """The bug-19 chain end to end: with an absolute `published`, a card aged
    past the bound is provably stale to the router's post-filter; with the
    old relative text it was kept as unprovable."""
    from searchio.router import _drop_stale

    payload = {
        "contents": {
            "videoRenderer": {
                "videoId": "old1",
                "title": {"runs": [{"text": "Old talk"}]},
                "publishedTimeText": {"simpleText": "2 months ago"},
            }
        }
    }
    html = f"<script>var ytInitialData = {json.dumps(payload)};</script>"
    (doc,) = parse_videos(html)
    kept, dropped = _drop_stale([doc], "week")
    assert kept == [] and dropped == 1
    # And an unparseable card (no publishedTimeText) is still kept -- unknown
    # age is not proof of staleness.
    payload["contents"]["videoRenderer"].pop("publishedTimeText")
    (undated,) = parse_videos(
        f"<script>var ytInitialData = {json.dumps(payload)};</script>")
    kept, dropped = _drop_stale([undated], "week")
    assert dropped == 0 and kept == [undated]


# ── freshness filter ─────────────────────────────────────────────────────────


def test_freshness_filter_drops_old_keeps_unknown(html):
    docs = parse_videos(html)
    fresh = filter_freshness(docs, "week")
    assert all(
        d.meta["age_days"] is None or d.meta["age_days"] <= 7 for d in fresh
    )
    assert 0 < len(fresh) < len(docs)


def test_freshness_all_keeps_everything(html):
    docs = parse_videos(html)
    assert len(filter_freshness(docs, "all")) == len(docs)


# ── provider interface ───────────────────────────────────────────────────────


class FakeLadder:
    """Serves one canned body through the fetch seam the provider uses."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.fetched: list[str] = []

    async def fetch(self, url: str):
        self.fetched.append(url)

        class Res:
            body = self.body
            via = "http"
            tier = 0
            final_url = url

        return Res()


async def test_provider_search_end_to_end(html):
    ladder = FakeLadder(html)
    ctx = ProviderContext(ladder=ladder, settings=Settings())
    docs = await YouTube().search(Query(text="rust programming", intent="video", k=5), ctx)
    assert len(docs) == 5
    assert ladder.fetched == [YouTube.url_for(Query(text="rust programming"))]
    assert all(d.fetched_via == "http" for d in docs)


async def test_provider_refuses_a_degraded_page():
    """Consent walls carry no ytInitialData. Until iteration 43 this test
    pinned that as "emptiness, not failure" -- which is exactly bug 65: the
    router recorded "used, 0 docs" and health never saw the wall. A page with
    no ytInitialData is now a ProviderError (see TestNoInitialDataIsARefusal);
    a results page whose payload holds no cards is still an honest [].
    """
    from searchio.errors import ProviderError

    ladder = FakeLadder("<html><body>Consent</body></html>")
    ctx = ProviderContext(ladder=ladder, settings=Settings())
    with pytest.raises(ProviderError, match="no ytInitialData"):
        await YouTube().search(Query(text="x", intent="video"), ctx)


def test_url_for_encoding_and_locale():
    url = YouTube.url_for(Query(text="lo-fi beats & study"))
    assert "search_query=lo-fi+beats+%26+study" in url
    assert " " not in url
    assert YouTube.url_for(Query(text="x", locale="de-DE", region="de")).endswith(
        "&hl=de&gl=DE"
    )
    # The default locale adds nothing; most queries stay en-US.
    assert "hl=" not in YouTube.url_for(Query(text="x"))


def test_registry_includes_youtube_for_video_intent():
    providers = default_providers()
    yt = next(p for p in providers if p.name == "youtube")
    assert yt.supports("video")
    assert not yt.supports("web")
    assert yt.configured(Settings())


import pytest as _pytest  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402


def _ctx_for(body: str):
    """A ladder whose fetch serves one canned body; _gate tolerates no robots/limiter."""
    async def fetch(url, **kw):
        return _NS(body=body, via="http", tier=0, final_url=url, status=200)
    return ProviderContext(ladder=_NS(fetch=fetch), settings=Settings())


class TestNoInitialDataIsARefusal:
    """Bug 65 (iteration 43, DeepSeek youtube.py review): a 200 page with NO
    ytInitialData -- a consent interstitial, a "sorry for the interruption"
    bot check, a shape change -- is not a results page, yet parse_videos
    returned [] and YouTube.search shipped it as an honest empty: the router
    recorded "used, 0 docs", health never saw a failure (the bug-53 class).
    No ytInitialData at all is a ProviderError; ytInitialData present with
    zero video renderers is a real empty and stays [].
    """

    async def test_consent_wall_shape_is_a_provider_error(self):
        from searchio.errors import ProviderError
        from searchio.providers.youtube import YouTube

        wall = "<html><body><h1>Before you continue to YouTube</h1><p>" + "x" * 2000 + "</p></body></html>"
        with _pytest.raises(ProviderError, match="ytInitialData"):
            await YouTube().search(Query(text="x", intent="video", k=5), _ctx_for(wall))

    async def test_results_page_with_no_cards_is_an_honest_empty(self):
        from searchio.providers.youtube import YouTube

        page = '<script>var ytInitialData = {"contents": {"twoColumnSearchResultsRenderer": {}}};</script>'
        docs = await YouTube().search(Query(text="x", intent="video", k=5), _ctx_for(page))
        assert docs == []


class TestSubHourAges:
    def test_minute_and_just_now_ages_date_to_today(self):
        # Bug 66 (iteration 43): _AGE knew hour/day/week/month/year only, so
        # "45 minutes ago" and "just now" -- the FRESHEST cards -- parsed to
        # None and shipped UNDATED (the bug-19 class in miniature: a relative
        # phrasing not mapped to a machine date).
        from searchio.providers.youtube import _age_days

        assert _age_days("45 minutes ago") == _pytest.approx(45 / 1440)
        assert _age_days("30 seconds ago") == _pytest.approx(30 / 86400)
        assert _age_days("just now") == 0.0
        assert _age_days("Streamed 5 minutes ago") == _pytest.approx(5 / 1440)

    def test_a_minutes_old_card_is_published_today(self):
        from searchio.providers.youtube import parse_videos

        page = ('<script>var ytInitialData = ' + json.dumps({"contents": [{"videoRenderer": {
            "videoId": "abcdefghijk", "title": {"runs": [{"text": "Fresh"}]},
            "publishedTimeText": {"simpleText": "12 minutes ago"},
            "viewCountText": {"simpleText": "1.2M views"}}}]}) + ';</script>')
        docs = parse_videos(page)
        assert docs and docs[0].published == dt.date.today().isoformat()
        assert docs[0].meta["views"] == 1_200_000  # rider: compact counts parse


class TestCompactViewCounts:
    def test_k_m_b_and_no_views(self):
        # Rider (iteration 43): "1.2M views" -- the shape most popular videos
        # carry -- parsed to None; K/M/B suffixes and "No views" now parse.
        from searchio.providers.youtube import _views_int

        assert _views_int("1.2M views") == 1_200_000
        assert _views_int("1.2K views") == 1_200
        assert _views_int("1.5B views") == 1_500_000_000
        assert _views_int("No views") == 0
        assert _views_int("1,234 views") == 1234
        assert _views_int("1 view") == 1

