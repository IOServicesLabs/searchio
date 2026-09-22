"""API-backed providers: published-date normalization.

`published` is a machine field consumed by the router's stale post-filter via
`date.fromisoformat` -- strict on Python 3.10 (padded YYYY-MM-DD only). Any
other shape escapes the filter as "unprovable": bug 19 was YouTube's relative
text, and its siblings were Crossref's unpadded date-parts join ("2026-9-5")
and OpenAlex's bare year ("2021"). These tests pin the normalization at
emission and the parse -> _drop_stale chain.
"""

from __future__ import annotations

import datetime as dt
import json

from searchio.config import Settings
from searchio.models import Query
from searchio.providers.apis import Crossref, OpenAlex, _iso_from_parts
from searchio.providers.base import ProviderContext


class TestIsoFromParts:
    def test_full_date_is_padded(self):
        assert _iso_from_parts([2026, 9, 5]) == "2026-09-05"

    def test_year_month_completes_to_month_end(self):
        assert _iso_from_parts([2026, 9]) == "2026-09-30"
        # February, leap and not.
        assert _iso_from_parts([2024, 2]) == "2024-02-29"
        assert _iso_from_parts([2026, 2]) == "2026-02-28"

    def test_bare_year_completes_to_year_end(self):
        # The latest day a bare year can mean: the stale filter may only drop
        # a doc when even its most generous reading is too old.
        assert _iso_from_parts([2021]) == "2021-12-31"

    def test_strings_parse(self):
        # Crossref date-parts are ints, but JSON could hand strings.
        assert _iso_from_parts(["2026", "9", "5"]) == "2026-09-05"

    def test_garbage_is_undated_not_garbage(self):
        assert _iso_from_parts([]) == ""
        assert _iso_from_parts([None]) == ""
        assert _iso_from_parts(["nope"]) == ""
        assert _iso_from_parts([2026, 13]) == ""  # no such month
        assert _iso_from_parts([2026, 2, 30]) == ""  # no such day

    def test_every_output_is_strictly_parseable(self):
        for parts in ([2026, 9, 5], [2026, 9], [2021], [2000, 1, 1]):
            out = _iso_from_parts(parts)
            assert dt.date.fromisoformat(out), parts


class FakeLadder:
    """Serves one canned JSON body through the fetch seam the providers use."""

    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload)

    async def fetch(self, url: str, **kw):
        body = self.body

        class Res:
            pass

        r = Res()
        r.body = body
        return r


class TestCrossref:
    async def test_unpadded_date_parts_become_strict_iso(self):
        payload = {
            "message": {
                "items": [
                    {"URL": "https://doi.org/10.1/a", "title": ["Paper A"],
                     "issued": {"date-parts": [[2026, 9, 5]]}},
                    {"URL": "https://doi.org/10.1/b", "title": ["Paper B"],
                     "issued": {"date-parts": [[2021]]}},
                    {"URL": "https://doi.org/10.1/c", "title": ["Paper C"],
                     "issued": {"date-parts": [[]]}},
                ]
            }
        }
        ctx = ProviderContext(ladder=FakeLadder(payload), settings=Settings())
        docs = await Crossref().search(Query(text="x", intent="academic", k=5), ctx)
        assert [d.published for d in docs] == ["2026-09-05", "2021-12-31", None]


class TestOpenAlex:
    async def test_full_date_preferred_over_year(self):
        payload = {
            "results": [
                {"id": "https://openalex.org/W1", "title": "Work A",
                 "publication_date": "2021-03-15", "publication_year": 2021},
                {"id": "https://openalex.org/W2", "title": "Work B",
                 "publication_year": 2021},
            ]
        }
        ctx = ProviderContext(ladder=FakeLadder(payload), settings=Settings())
        docs = await OpenAlex().search(Query(text="x", intent="academic", k=5), ctx)
        assert docs[0].published == "2021-03-15"
        assert docs[1].published == "2021-12-31"

    async def test_stale_bound_now_binds_academic_docs(self):
        """The bug-19-sibling chain end to end: a work whose most generous
        reading predates the bound is dropped; one inside it survives."""
        from searchio.router import _drop_stale

        payload = {
            "results": [
                {"id": "https://openalex.org/W1", "title": "Ancient",
                 "publication_year": 2020},
                {"id": "https://openalex.org/W2", "title": "Fresh",
                 "publication_date": dt.date.today().isoformat()},
            ]
        }
        ctx = ProviderContext(ladder=FakeLadder(payload), settings=Settings())
        docs = await OpenAlex().search(Query(text="x", intent="academic", k=5), ctx)
        kept, dropped = _drop_stale(docs, "year")
        assert dropped == 1
        assert [d.title for d in kept] == ["Fresh"]


import pytest  # noqa: E402

from searchio.errors import ProviderError  # noqa: E402
from searchio.providers.apis import (  # noqa: E402
    ArxivProvider, HackerNews, StackExchange, Wikipedia,
)


class RawLadder:
    """Serves one canned RAW body (Atom XML) through the providers' fetch seam."""

    def __init__(self, body: str) -> None:
        self.body = body

    async def fetch(self, url: str, **kw):
        class Res:
            pass

        r = Res()
        r.body = self.body
        return r


def _q(intent: str = "academic") -> Query:
    return Query(text="x", intent=intent, k=5)


class TestApiErrorEnvelopes:
    """Bug 53 (iteration 38, DeepSeek apis.py review): _json only refuses a
    body that is not JSON. An API's OWN error envelope -- 200 + valid error
    JSON -- flowed into the parser and came out as [] ("no results"), so a
    StackExchange throttle_violation (it throttles at 300 req/day
    unauthenticated), a Crossref status:"failed", an OpenAlex/Wikipedia error
    object never reached the router as a failure: health recorded a success,
    the breaker never tripped, and span's attrition bookkeeping read a
    refused provider as "answered, 0 docs". GitHub was the only provider that
    already raised on a missing results key; the rest now match it. A results
    key that is PRESENT but empty is a real answer and stays [].
    """

    def _ctx(self, payload):
        return ProviderContext(ladder=FakeLadder(payload), settings=Settings())

    async def test_stackexchange_throttle_is_a_refusal_not_empty(self):
        payload = {"error_id": 502, "error_name": "throttle_violation",
                   "error_message": "too many requests from this IP"}
        with pytest.raises(ProviderError, match="throttle|too many"):
            await StackExchange().search(_q("code"), self._ctx(payload))

    async def test_crossref_failed_status_is_a_refusal(self):
        # Pre-fix this CRASHED (message is a list on failure -> .get on a
        # list), which the router counts as a parser crash, not a refusal.
        payload = {"status": "failed", "message-type": "validation-failure",
                   "message": [{"type": "parameter-not-allowed",
                                "message": "This route does not support offset"}]}
        with pytest.raises(ProviderError, match="offset|failed"):
            await Crossref().search(_q(), self._ctx(payload))

    async def test_openalex_error_is_a_refusal(self):
        payload = {"error": "Invalid query parameters error.",
                   "message": "Something went wrong."}
        with pytest.raises(ProviderError, match="Invalid"):
            await OpenAlex().search(_q(), self._ctx(payload))

    async def test_wikipedia_error_is_a_refusal(self):
        payload = {"error": {"code": "internal_api_error_DBQueryError",
                             "info": "[abc123] Exception caught: A database query error"}}
        with pytest.raises(ProviderError, match="database query"):
            await Wikipedia().search(_q("reference"), self._ctx(payload))

    async def test_hackernews_missing_hits_is_a_refusal(self):
        payload = {"message": "Invalid Application-ID or API key", "status": 403}
        with pytest.raises(ProviderError, match="Application"):
            await HackerNews().search(_q("forum"), self._ctx(payload))

    async def test_legit_empty_results_stay_empty(self):
        # The results key is present -> the API really answered "nothing".
        assert await OpenAlex().search(_q(), self._ctx({"results": []})) == []
        assert await Wikipedia().search(_q("reference"), self._ctx({"query": {"search": []}})) == []
        assert await StackExchange().search(_q("code"), self._ctx({"items": []})) == []
        assert await Crossref().search(_q(), self._ctx({"status": "ok", "message": {"items": []}})) == []
        assert await HackerNews().search(_q("forum"), self._ctx({"hits": []})) == []


class TestArxivErrorFeed:
    """Bug 54 (iteration 38): arXiv reports an error as an Atom feed with ONE
    <entry> whose <id> is http://arxiv.org/api/errors#<reason> and whose
    <title> is "Error". The regex parser turned that into a FAKE Doc -- a
    citable "paper" titled Error at an arxiv.org/api/errors URL -- worse than
    an empty. It is a refusal carrying the feed's <summary>.
    """

    ERR = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
           '<id>http://arxiv.org/api/errors#malformed_id</id>'
           '<title>Error</title><summary>malformed id</summary>'
           '<updated>2026-09-13T00:00:00Z</updated></entry></feed>')
    OK = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
          '<id>http://arxiv.org/abs/2401.00001v1</id><title>A real paper</title>'
          '<summary>abs</summary><published>2024-01-01T00:00:00Z</published>'
          '<author><name>A</name></author></entry></feed>')

    async def test_error_feed_is_a_refusal_not_a_fake_paper(self):
        ctx = ProviderContext(ladder=RawLadder(self.ERR), settings=Settings())
        with pytest.raises(ProviderError, match="malformed"):
            await ArxivProvider().search(_q(), ctx)

    async def test_real_entry_still_parses(self):
        ctx = ProviderContext(ladder=RawLadder(self.OK), settings=Settings())
        docs = await ArxivProvider().search(_q(), ctx)
        assert [d.url for d in docs] == ["http://arxiv.org/abs/2401.00001v1"]
        assert docs[0].title == "A real paper" and docs[0].published == "2024-01-01T00:00:00Z"

