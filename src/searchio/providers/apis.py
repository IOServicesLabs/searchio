"""Providers backed by public JSON APIs.

These are the quiet workhorses. An official API is better than scraping on
every axis that matters here: it is faster, it returns structured fields
instead of guessed ones, it will not change shape next Tuesday because someone
shipped a redesign, and -- the point of this whole package -- it cannot be
blocked by an anti-bot system, because there is no bot detection in front of a
documented endpoint you are using as intended.

So the router prefers them wherever they can answer. The scraped engines in
:mod:`searchio.providers.engines` exist for the queries these cannot serve, not
the other way round.

Every adapter here is keyless. GitHub optionally takes a token, which only
raises its rate limit.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
import urllib.parse
from typing import Any

from ..errors import ProviderError
from ..models import Capability, Doc, Query
from .base import Provider, ProviderContext


async def _json(ctx: ProviderContext, url: str, *, max_tier: int = 1) -> Any:
    """Fetch and parse JSON through the ladder, so pacing and caching apply."""
    res = await ctx.ladder.fetch(url, max_tier=max_tier)
    try:
        return json.loads(res.body)
    except json.JSONDecodeError as exc:
        raise ProviderError("json", f"{url}: {exc}") from exc


def _envelope_error(name: str, data: Any, results_key: str, *msg_keys: str) -> None:
    """Refuse an API's OWN error envelope instead of parsing it into [].

    ``_json`` only refuses a body that is not JSON. A 200 carrying valid error
    JSON (StackExchange's throttle_violation, an OpenAlex/Wikipedia error
    object, an Algolia message) used to flow into the parser and come out as
    "no results" -- so a throttled provider never reached the router as a
    failure: health recorded a success, the breaker never tripped, and the
    span suite's attrition bookkeeping read a refusal as "answered, 0 docs"
    (bug 53). A body WITH the results key is a real answer even when empty; a
    body WITHOUT it is a refusal, named by the API's own message when it has
    one. GitHub already did this; every provider now does.
    """
    if not isinstance(data, dict):
        raise ProviderError(name, f"unexpected payload: {type(data).__name__}")
    # A results key that is PRESENT is a real answer -- but only if its value
    # is not null (bug 174): {"items": null} passed the key check and then
    # GitHub's data["items"] did enumerate(None) -> TypeError. A null results
    # value is a refusal, named by the API's message when it has one.
    if results_key in data and data[results_key] is not None:
        return
    for k in msg_keys:
        v = data.get(k)
        if v:
            if isinstance(v, dict):  # Wikipedia: {"error": {"code", "info"}}
                v = v.get("info") or v.get("message") or v.get("code") or v
            raise ProviderError(name, str(v)[:150])
    raise ProviderError(name, f"no '{results_key}' in response")


def _epoch_to_iso(epoch) -> str | None:
    """A Unix epoch (seconds) as a strict ISO date, or None when it is not a
    usable number. ``published`` must be ISO-or-absent, never a raw epoch."""
    try:
        return _dt.datetime.fromtimestamp(
            float(epoch), tz=_dt.timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _iso_from_parts(parts) -> str:
    """A strict ISO date from a partial ``[year, month?, day?]`` sequence.

    ``published`` is a machine field: the router's stale post-filter parses it
    with ``date.fromisoformat``, which on Python 3.10 is strict -- padded
    YYYY-MM-DD only. A bare year ("2021") or Crossref's unpadded join
    ("2026-9-5") fails the parse and escapes every freshness bound as
    "unprovable" -- the same hole as bug 19's relative YouTube text.

    Partial precision is completed with the LAST day consistent with it
    (year Y -> Y-12-31), which keeps the post-filter's provably-stale-only
    semantics: a doc is dropped only when even its most generous reading is
    too old.
    """
    if not parts:
        return ""
    try:
        y = int(parts[0])
        if len(parts) > 1 and parts[1] is not None:
            m = int(parts[1])
            if not 1 <= m <= 12:
                return ""
        else:
            m = 12
        if len(parts) > 2 and parts[2] is not None:
            d = int(parts[2])
        else:
            # Day 0 of the next month is the last day of this one.
            d = (_dt.date(y + (m == 12), m % 12 + 1, 1) - _dt.timedelta(days=1)).day
        return _dt.date(y, m, d).isoformat()
    except (TypeError, ValueError):
        return ""


def _q(text: str) -> str:
    return urllib.parse.quote_plus(text)


_TAG_RE = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    """Tags out, entities decoded, whitespace folded (bug 121): arXiv titles
    kept their XML entities, Wikipedia snippets their HTML entities and
    Crossref abstracts their JATS tags -- "A &amp; B" and "<jats:p>" reached
    the agent verbatim."""
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


class Wikipedia(Provider):
    """Wikipedia full-text search.

    Cheap, instant, and unusually good at the thing search engines are worst
    at: giving a model a correct, neutral definition of an entity before it
    starts reasoning about it.
    """

    name = "wikipedia"
    caps = frozenset({Capability.REFERENCE, Capability.WEB})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
            f"&srsearch={_q(q.text)}&srlimit={min(q.k, 20)}&srprop=snippet|timestamp"
        )
        data = await _json(ctx, url)
        _envelope_error(self.name, data, "query", "error")
        hits = (data.get("query") or {}).get("search") or []
        out = []
        for i, h in enumerate(hits):
            title = h.get("title", "")
            snippet = _plain(h.get("snippet", ""))
            out.append(
                Doc(
                    url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    title=title,
                    snippet=snippet,
                    published=h.get("timestamp"),
                    source=self.name,
                    rank=i,
                )
            )
        return out


class HackerNews(Provider):
    """Hacker News via the Algolia index.

    Worth its slot for a reason that is easy to miss: HN comment threads are
    often the only place where a product or library's real failure modes are
    discussed candidly. Marketing pages and review sites systematically omit
    exactly what a research task most needs.
    """

    name = "hackernews"
    caps = frozenset({Capability.FORUM, Capability.CODE, Capability.NEWS})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        tags = "story"
        endpoint = "search" if q.freshness == "all" else "search_by_date"
        url = (
            f"https://hn.algolia.com/api/v1/{endpoint}?query={_q(q.text)}"
            f"&tags={tags}&hitsPerPage={min(q.k, 30)}"
        )
        data = await _json(ctx, url)
        _envelope_error(self.name, data, "hits", "message", "error")
        out = []
        for i, h in enumerate(data.get("hits") or []):
            hn_url = f"https://news.ycombinator.com/item?id={h.get('objectID')}"
            out.append(
                Doc(
                    url=h.get("url") or hn_url,
                    title=h.get("title") or h.get("story_title") or "",
                    # HN story_text/comment_text is HTML (bug 175): de-tag it
                    # like every other API snippet (bug 121's class).
                    snippet=_plain(h.get("story_text") or h.get("comment_text") or "")[:300],
                    published=h.get("created_at"),
                    source=self.name,
                    rank=i,
                    meta={
                        "points": h.get("points"),
                        "comments": h.get("num_comments"),
                        "discussion": hn_url,
                    },
                )
            )
        return out


class ArxivProvider(Provider):
    """arXiv preprints. Atom feed, so parsed as XML rather than JSON."""

    name = "arxiv"
    caps = frozenset({Capability.ACADEMIC})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = (
            "http://export.arxiv.org/api/query?search_query=all:"
            f"{_q(q.text)}&start=0&max_results={min(q.k, 20)}"
            "&sortBy=relevance&sortOrder=descending"
        )
        res = await ctx.ladder.fetch(url, max_tier=1)
        # A body that is not an Atom feed at all -- a maintenance HTML page,
        # a proxy error served with 200 -- has no <entry> and used to ship as
        # an honest "no results" (bug 116). Zero results is a <feed> with
        # totalResults 0; anything without a <feed> root is a refusal.
        if "<feed" not in res.body[:4096]:
            raise ProviderError(self.name, "not an Atom feed: " + re.sub(r"\s+", " ", res.body[:120]))
        entries = re.findall(r"<entry>(.*?)</entry>", res.body, re.S)
        # arXiv reports an error as an Atom feed with a single entry whose id
        # is http://arxiv.org/api/errors#<reason> and title "Error". Parsed
        # naively that became a FAKE Doc -- a citable "paper" titled Error at
        # an api/errors URL (bug 54). It is a refusal carrying the summary.
        if len(entries) == 1 and "arxiv.org/api/errors" in entries[0]:
            m = re.search(r"<summary[^>]*>(.*?)</summary>", entries[0], re.S)
            raise ProviderError(self.name, (m.group(1).strip() if m else "api error")[:150])
        out = []
        for i, e in enumerate(entries):
            def pick(tag: str) -> str:
                m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", e, re.S)
                return _plain(m.group(1)) if m else ""

            link = re.search(r'<id>(.*?)</id>', e, re.S)
            out.append(
                Doc(
                    url=(link.group(1).strip() if link else ""),
                    title=pick("title"),
                    snippet=pick("summary")[:400],
                    published=pick("published"),
                    source=self.name,
                    rank=i,
                    meta={"authors": re.findall(r"<name>(.*?)</name>", e)},
                )
            )
        return [d for d in out if d.url]


class OpenAlex(Provider):
    """OpenAlex: open bibliographic metadata across all of scholarly publishing.

    Preferred over scraping Google Scholar, which is both hostile to automation
    and legally murkier. OpenAlex is explicitly built to be queried.
    """

    name = "openalex"
    caps = frozenset({Capability.ACADEMIC})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = (
            "https://api.openalex.org/works?search="
            f"{_q(q.text)}&per-page={min(q.k, 25)}&sort=relevance_score:desc"
        )
        data = await _json(ctx, url)
        _envelope_error(self.name, data, "results", "error", "message")
        out = []
        for i, w in enumerate(data.get("results") or []):
            loc = (w.get("primary_location") or {}) or {}
            landing = loc.get("landing_page_url") or w.get("doi") or w.get("id") or ""
            # OpenAlex stores abstracts as an inverted index to dodge copyright
            # on the full text; rebuilding word order is the documented use.
            inv = w.get("abstract_inverted_index") or {}
            abstract = ""
            if inv:
                positions: list[tuple[int, str]] = []
                for word, idxs in inv.items():
                    positions.extend((idx, word) for idx in idxs)
                abstract = " ".join(w2 for _, w2 in sorted(positions))[:400]
            out.append(
                Doc(
                    url=landing,
                    title=w.get("title") or w.get("display_name") or "",
                    snippet=abstract,
                    # publication_date is full ISO; the bare year alone fails
                    # the router's strict parse, so complete it at the latest
                    # day it could mean.
                    published=w.get("publication_date")
                    or _iso_from_parts([w.get("publication_year")])
                    or None,
                    source=self.name,
                    rank=i,
                    meta={
                        "citations": w.get("cited_by_count"),
                        "doi": w.get("doi"),
                        "open_access": (w.get("open_access") or {}).get("is_oa"),
                    },
                )
            )
        return [d for d in out if d.url]


class Crossref(Provider):
    """Crossref DOI metadata. Authoritative for publication records."""

    name = "crossref"
    caps = frozenset({Capability.ACADEMIC})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = f"https://api.crossref.org/works?query={_q(q.text)}&rows={min(q.k, 20)}"
        data = await _json(ctx, url)
        msg = data.get("message") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("status", "ok") != "ok" or not isinstance(msg, dict):
            # Refuse the FAILURE shape, not the absence of a success marker:
            # a validation failure ships status:"failed" with message as a
            # LIST of errors, and the old ``.get("items")`` on that list
            # crashed the fan-out with an AttributeError -- a parser crash
            # where an honest refusal was owed (bug 53). A dict message with
            # items is the answer, whether or not status accompanies it.
            if isinstance(msg, list) and msg:
                msg = (msg[0] or {}).get("message") or msg[0]
            status = data.get("status", "no status") if isinstance(data, dict) else "not an object"
            raise ProviderError(self.name, f"{status}: {str(msg)[:120]}")
        out = []
        for i, w in enumerate(((data.get("message") or {}).get("items")) or []):
            # The abstract went through _plain but the title did not (bug
            # 176): a JATS/MathML title ("<i>E. coli</i>") shipped raw tags.
            title = _plain(" ".join(w.get("title") or [])) or ""
            out.append(
                Doc(
                    url=w.get("URL") or "",
                    title=title,
                    snippet=_plain(w.get("abstract") or "")[:400],
                    published=_iso_from_parts(
                        ((w.get("issued") or {}).get("date-parts") or [[]])[0]
                    ) or None,
                    source=self.name,
                    rank=i,
                    meta={
                        "doi": w.get("DOI"),
                        "citations": w.get("is-referenced-by-count"),
                        "publisher": w.get("publisher"),
                    },
                )
            )
        return [d for d in out if d.url and d.title]


class GitHubRepos(Provider):
    """GitHub repository search.

    A token is optional and only buys rate limit: 60 requests/hour
    unauthenticated, 5000 with one.
    """

    name = "github"
    caps = frozenset({Capability.CODE})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = (
            "https://api.github.com/search/repositories?q="
            f"{_q(q.text)}&sort=stars&order=desc&per_page={min(q.k, 20)}"
        )
        data = await _json(ctx, url)
        _envelope_error(self.name, data, "items", "message")
        out = []
        for i, r in enumerate(data["items"]):
            out.append(
                Doc(
                    url=r.get("html_url") or "",
                    title=r.get("full_name") or "",
                    snippet=(r.get("description") or "")[:300],
                    published=r.get("pushed_at"),
                    source=self.name,
                    rank=i,
                    meta={
                        "stars": r.get("stargazers_count"),
                        "language": r.get("language"),
                        "forks": r.get("forks_count"),
                    },
                )
            )
        return [d for d in out if d.url]


class StackExchange(Provider):
    """Stack Overflow and friends. Excellent for concrete error messages."""

    name = "stackexchange"
    caps = frozenset({Capability.CODE, Capability.FORUM})

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        url = (
            "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
            f"&q={_q(q.text)}&site=stackoverflow&pagesize={min(q.k, 20)}&filter=default"
        )
        data = await _json(ctx, url)
        _envelope_error(self.name, data, "items", "error_message", "error_name")
        out = []
        for i, it in enumerate(data.get("items") or []):
            out.append(
                Doc(
                    url=it.get("link") or "",
                    title=it.get("title") or "",
                    snippet="",
                    # creation_date is a Unix epoch (bug 177): SE used to drop
                    # it, so every result was undated and escaped every
                    # freshness bound. Emit an ISO date -- a raw epoch int
                    # would be the date-honesty trap (unparseable -> undated).
                    published=_epoch_to_iso(it.get("creation_date")),
                    source=self.name,
                    rank=i,
                    meta={
                        "score": it.get("score"),
                        "answered": it.get("is_answered"),
                        "answers": it.get("answer_count"),
                        "tags": it.get("tags"),
                    },
                )
            )
        return [d for d in out if d.url]
