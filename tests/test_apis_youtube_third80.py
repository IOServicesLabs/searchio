"""Third pass on the API + YouTube parsers (iteration 80, ds_brief_apis /
ds_brief_youtube). Every test bit RED before its fix."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from searchio.config import Settings
from searchio.models import Query
from searchio.providers.apis import Crossref, GitHubRepos, HackerNews, StackExchange
from searchio.providers.base import ProviderContext
from searchio.providers.youtube import _text


class FakeLadder:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload)

    async def fetch(self, url: str, **kw):
        class Res:
            pass
        r = Res()
        r.body = self.body
        r.via = "http"
        r.tier = 0
        return r


def _ctx(payload):
    return ProviderContext(ladder=FakeLadder(payload), settings=Settings())


def _q(intent="code"):
    return Query(text="x", intent=intent, k=5)


class TestYoutubeTextNullRuns:
    def test_null_runs_value_does_not_crash(self):
        # Bug 173: node.get("runs", []) returns None when the key is present
        # with a null value ({"runs": null}) -- a shape YouTube emits -- so
        # `for r in None` raised, and _text runs everywhere in the page parse,
        # so one such card killed the whole results page (bugs 65/133, the
        # null variant).
        assert _text({"runs": None}) == ""
        assert _text({"simpleText": None, "runs": None}) == ""
        assert _text({"runs": [{"text": "a"}, None, {"text": "b"}]}) == "ab"


class TestGithubNullItems:
    async def test_a_null_items_value_is_a_refusal_not_a_crash(self):
        # Bug 174: _envelope_error returned as soon as the "items" KEY was
        # present, even with a null value, and GitHubRepos then did
        # data["items"] directly -> enumerate(None) TypeError. A null results
        # value is not an answer.
        from searchio.errors import ProviderError
        with pytest.raises(ProviderError):
            await GitHubRepos().search(_q(), _ctx({"items": None, "total_count": 0}))

    async def test_a_normal_items_list_still_parses(self):
        docs = await GitHubRepos().search(_q(), _ctx({"items": [
            {"html_url": "https://github.com/a/b", "full_name": "a/b",
             "description": "d", "pushed_at": "2026-09-07T00:00:00Z"}]}))
        assert docs and docs[0].url == "https://github.com/a/b"


class TestSnippetsAreDetagged:
    async def test_hackernews_snippet_strips_html(self):
        # Bug 175: HN story_text/comment_text is HTML (comments carry <p>,
        # <a>, <i>); the snippet shipped raw tags, unlike every other API
        # snippet that goes through _plain (bug 121's class).
        payload = {"hits": [{"objectID": "1", "title": "T", "url": "https://a.com/1",
                             "story_text": "<p>First line.</p><p>See <a href='x'>this</a> &amp; that.</p>",
                             "created_at": "2026-09-07T00:00:00Z"}]}
        docs = await HackerNews().search(_q("forum"), _ctx(payload))
        assert "<p>" not in docs[0].snippet and "<a" not in docs[0].snippet
        assert "First line." in docs[0].snippet and "& that" in docs[0].snippet

    async def test_crossref_title_strips_jats_markup(self):
        # Bug 176: the abstract went through _plain but the title did not, so
        # a JATS/MathML title ("<i>E. coli</i>") shipped literal tags.
        payload = {"status": "ok", "message": {"items": [
            {"URL": "https://doi.org/10.1/x",
             "title": ["Genome of <i>E. coli</i> K-12 &amp; relatives"],
             "issued": {"date-parts": [[2026, 9]]}}]}}
        docs = await Crossref().search(_q("academic"), _ctx(payload))
        assert "<i>" not in docs[0].title and "E. coli" in docs[0].title
        assert "& relatives" in docs[0].title


class TestStackExchangeIsDated:
    async def test_creation_date_epoch_becomes_iso_so_freshness_applies(self):
        # Bug 177: StackExchange dropped creation_date entirely, so every SE
        # result was undated and escaped every freshness bound. The API's
        # creation_date is a Unix epoch; it must become an ISO date (a raw
        # epoch int would be the date-honesty trap -- unparseable, undated).
        epoch = int(dt.datetime(2026, 9, 7, tzinfo=dt.timezone.utc).timestamp())
        payload = {"items": [{"link": "https://stackoverflow.com/q/1", "title": "Q",
                              "creation_date": epoch, "score": 5}]}
        docs = await StackExchange().search(_q("code"), _ctx(payload))
        assert docs and docs[0].published == "2026-09-07", docs[0].published
        # A missing/garbage creation_date stays undated, never a fake date.
        docs2 = await StackExchange().search(_q("code"), _ctx(
            {"items": [{"link": "https://stackoverflow.com/q/2", "title": "Q2"}]}))
        assert docs2[0].published is None
