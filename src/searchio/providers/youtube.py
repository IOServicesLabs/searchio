"""YouTube search, read out of the page's own ``ytInitialData``.

Three reasons this provider exists when the router can already scope a web
search to ``youtube.com/watch``:

* Ranking. A ``site:`` query ranks pages by web-page signals; YouTube ranks by
  watch-time and engagement. "Best rust tutorial" wants the second list, and
  only YouTube's own search computes it.
* Richness. The results page ships its data as JSON -- every card's title,
  channel, view count, age, duration, and live status -- which makes the
  parse a tree walk instead of the HTML archaeology the engines endure.
* Freshness. YouTube's first page mixes ages; the card's own
  ``publishedTimeText`` ("3 days ago") lets the ``freshness`` filter apply
  client-side without reverse-engineering the ``sp`` filter parameter
  encoding.

The JSON arrives as a plain ``var ytInitialData = {...}`` in the initial
HTML, which a bare HTTP GET receives in full. No browser, no key, no quota.
When a datacentre IP gets YouTube's consent wall instead, the ladder
escalates to the browser tier exactly like every other provider, and the same
parser reads whatever DOM resulted -- consent pages simply carry no
``ytInitialData`` and parse to nothing, which is ordinary emptiness rather
than a failure.

Robots note, mirroring the Facebook provider: YouTube's robots.txt disallows
``/results``, so under ``robots_policy=enforce`` this provider declines to
run, and the ladder's per-domain pacing applies at every policy setting.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from urllib.parse import urlencode

from ..models import Capability, Doc, Query
from .base import Provider, ProviderContext

HOST = "www.youtube.com"
DOMAIN = "youtube.com"

#: The assignment that hands the search payload to the page. Two spellings
#: because YouTube has shipped both; try each.
_MARKERS = ('var ytInitialData = ', 'window["ytInitialData"] = ')

#: Relative upload age, e.g. "3 hours ago", "2 weeks ago". Premiere and
#: upcoming videos phrase it as "Premieres in 2 hours" -- those are not out
#: yet and carry no publishedTimeText at all.
_AGE = re.compile(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s+ago", re.I)
_AGE_DAYS = {"second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24,
             "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}
#: "just now" / "moments ago": the freshest phrasing of all, and the one the
#: old regex could not read -- it shipped those cards UNDATED (bug 66).
_JUST_NOW = re.compile(r"\b(just now|moments? ago)\b", re.I)

#: Query.freshness -> maximum age in days. Client-side filter; YouTube's own
#: upload-date filters use an opaque ``sp`` token that changes encoding.
_FRESHNESS_MAX_DAYS = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}


def _balanced(s: str, start: int) -> str | None:
    """The JSON object opening at ``start``, with string-awareness.

    Brace-counting alone breaks the first time a title contains "}" -- a
    StackOverflow answer that ships inside a ``descriptionSnippet`` does --
    so the scan tracks whether it is inside a string and honours backslash
    escapes.
    """
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : j + 1]
    return None


def extract_initial_data(html: str) -> dict | None:
    """The parsed ``ytInitialData`` object, or None when the page has none."""
    for marker in _MARKERS:
        i = html.find(marker)
        if i == -1:
            continue
        i += len(marker)
        while i < len(html) and html[i] in " \t":
            i += 1
        if i >= len(html) or html[i] != "{":
            return None
        raw = _balanced(html, i)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None
    return None


def _walk(node, key: str):
    """Every dict named ``key`` in the tree, in document order."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, dict):
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)


def _text(node) -> str:
    """YouTube's two string shapes: {simpleText: ...} and {runs: [{text}...]}."""
    if not isinstance(node, dict):
        return ""
    if node.get("simpleText"):
        return str(node["simpleText"])
    # `node.get("runs", [])` returns None when the key is PRESENT with a null
    # value ({"runs": null}, a shape YouTube emits) -- `or []` covers it, so a
    # single malformed card no longer kills the whole page parse (bug 173,
    # bugs 65/133's null variant).
    return "".join(str(r.get("text", "")) for r in (node.get("runs") or []) if isinstance(r, dict))


_VIEWS = re.compile(r"([\d,]+(?:\.\d+)?)\s*([KMB])?\s+views?", re.I)
_VIEW_MULT = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def _views_int(text: str) -> int | None:
    t = text or ""
    if t.strip().lower().startswith("no view"):
        return 0
    m = _VIEWS.match(t)
    if not m:
        return None
    # "1.2M views" is the shape most popular videos carry (rider, iteration
    # 43): the old \d-only pattern returned None for all of them.
    return int(float(m.group(1).replace(",", "")) * _VIEW_MULT[(m.group(2) or "").upper()])


def _age_days(text: str) -> float | None:
    m = _AGE.search(text or "")
    if not m:
        return 0.0 if _JUST_NOW.search(text or "") else None
    return int(m.group(1)) * _AGE_DAYS[m.group(2).lower()]


def _duration_s(text: str) -> int | None:
    """"3:05:04" -> seconds. Live and premiere cards have no length."""
    if not text or ":" not in text:
        return None
    try:
        parts = [int(p) for p in text.split(":")]
    except ValueError:
        return None
    total = 0
    for p in parts:
        total = total * 60 + p
    return total


def _is_live(renderer: dict, views_text: str) -> bool:
    if (views_text or "").endswith("watching"):
        return True
    for badge in renderer.get("badges", []) or []:
        label = _text(badge.get("metadataBadgeRenderer", {}).get("label"))
        if label == "LIVE":
            return True
    return False


def parse_videos(html: str, *, source: str = "youtube") -> list[Doc]:
    """Every distinct video on a results page: regular videos in page order,
    then Shorts (the two renderer walks are separate, so Shorts land after
    every regular card). A page with no ytInitialData yields [] here;
    YouTube.search refuses that page instead (bug 65)."""
    data = extract_initial_data(html)
    if not data:
        return []
    return _videos_from_data(data, source=source)


def _videos_from_data(data: dict, *, source: str = "youtube") -> list[Doc]:
    out: list[Doc] = []
    seen: set[str] = set()

    def add(video_id: str, title: str, renderer: dict, *, shorts: bool) -> None:
        if not video_id or video_id in seen:
            return
        title = " ".join(title.split()).strip()
        if not title:
            # Ad renderers carry a bare videoId and nothing else; there is no
            # video to name, so nothing worth returning.
            return
        seen.add(video_id)

        channel = _text(renderer.get("ownerText")) or _text(renderer.get("shortBylineText"))
        age_text = _text(renderer.get("publishedTimeText"))
        views_raw = _text(renderer.get("viewCountText"))
        views = _views_int(views_raw)
        length_text = _text(renderer.get("lengthText"))
        live = _is_live(renderer, views_raw)
        # A live video is happening now; "premieres in 2 hours" has no
        # publishedTimeText and stays None (unknown age, never filtered).
        age = 0.0 if live else _age_days(age_text)
        # `published` is a machine field. Relative text ("3 days ago") is
        # display phrasing: the router's stale post-filter cannot parse it,
        # which let video results escape every downstream freshness bound
        # (bench/span.py bulk date-sanity detector, iteration 11). Emit the
        # absolute day; the relative phrasing stays available in
        # meta["age_text"].
        published = (
            (_dt.date.today() - _dt.timedelta(days=int(age))).isoformat()
            if age is not None else None
        )

        # The card's own description snippet (DeepSeek youtube #5): the
        # snippet used to be channel/views/age chrome only, and the one
        # thing an agent needs to judge a video was dropped.
        snips = renderer.get("detailedMetadataSnippets") or []
        description = ""
        if isinstance(snips, list) and snips and isinstance(snips[0], dict):
            description = " ".join(_text(snips[0].get("snippetText")).split())[:300]
        parts = [p for p in (channel, views_raw, age_text) if p]
        if description:
            parts.append(description)
        meta: dict = {
            "video_id": video_id,
            "channel": channel,
            "views": views,
            "views_raw": views_raw,
            "age_text": age_text,
            "age_days": age,
            "live": live,
            "shorts": shorts,
        }
        if description:
            meta["description"] = description
        if length_text:
            meta["duration"] = length_text
            dur = _duration_s(length_text)
            if dur is not None:
                meta["duration_s"] = dur
            parts.append(length_text)
        # One odd card must not kill the page (DeepSeek youtube #2): an empty
        # ownerText.runs indexed [0] raised and the whole results page parsed
        # to nothing.
        runs = (renderer.get("ownerText") or {}).get("runs")
        run0 = runs[0] if isinstance(runs, list) and runs and isinstance(runs[0], dict) else {}
        channel_id = str(
            ((run0.get("navigationEndpoint") or {}).get("browseEndpoint") or {}).get("browseId") or ""
        )
        if channel_id:
            meta["channel_id"] = channel_id

        out.append(
            Doc(
                # Canonical watch URL: the renderer's own navigationEndpoint
                # href carries tracking params (pp, feature) that differ per
                # render, so two fetches of one video would dedupe as two.
                url=f"https://{HOST}/watch?v={video_id}",
                title=title[:400],
                snippet=" · ".join(parts)[:500],
                published=published,
                source=source,
                rank=len(out),
                meta=meta,
            )
        )

    for renderer in _walk(data, "videoRenderer"):
        add(str(renderer.get("videoId") or ""), _text(renderer.get("title")), renderer, shorts=False)
    for renderer in _walk(data, "reelItemRenderer"):
        # Shorts: vertical player, no duration or upload age on the card, but
        # the same watch URL shape. The reel card's title is "headline".
        add(str(renderer.get("videoId") or ""), _text(renderer.get("headline")), renderer, shorts=True)
    return out


def filter_freshness(docs: list[Doc], freshness: str) -> list[Doc]:
    """Apply the query's freshness bound to parsed cards.

    Cards whose age cannot be parsed are kept: freshness filtering is a
    client-side nicety here, and silently dropping an unparsable card would
    report "no recent videos" for what was really "could not tell".
    """
    max_days = _FRESHNESS_MAX_DAYS.get(freshness)
    if max_days is None:
        return docs
    return [
        d for d in docs
        if d.meta.get("age_days") is None or float(d.meta["age_days"]) <= max_days
    ]


class YouTube(Provider):
    """Native YouTube search, keyless, read from ``ytInitialData``."""

    name = "youtube"
    caps = frozenset({Capability.VIDEO})
    primary_domain = DOMAIN

    async def search(self, q: Query, ctx: ProviderContext) -> list[Doc]:
        await self._gate(ctx)
        res = await ctx.ladder.fetch(self.url_for(q))
        data = extract_initial_data(res.body)
        if data is None:
            # A 200 with no ytInitialData is not a results page: a consent
            # interstitial ("Before you continue"), a bot check, or a shape
            # change. It used to parse to [] and ship as an honest empty, so
            # the router recorded "used, 0 docs" and health never saw a
            # failure (bug 65 -- the 200-error-envelope class of bug 53).
            from ..errors import ProviderError

            head = " ".join((res.body or "")[:4000].split())[:160]
            raise ProviderError(self.name, f"no ytInitialData on the results page (consent wall / bot check / shape change?): {head!r}")
        docs = filter_freshness(_videos_from_data(data, source=self.name), q.freshness)
        for d in docs:
            d.fetched_via = res.via
            d.meta["tier"] = res.tier
        return docs[: q.k]

    @staticmethod
    def url_for(q: Query) -> str:
        params = {"search_query": q.text}
        if q.locale and q.locale.lower() not in ("en-us", "en_us"):
            lang = q.locale.replace("_", "-").split("-")[0].lower()
            if lang.isalpha() and len(lang) == 2:
                params["hl"] = lang
        if q.region and len(q.region) == 2:
            params["gl"] = q.region.upper()
        return f"https://{HOST}/results?{urlencode(params)}"

    async def _gate(self, ctx: ProviderContext) -> None:
        """Borrow the ladder's robots policy and per-domain pacing.

        Same reasoning as the Facebook provider: this adapter drives a
        host whose results path is robots-disallowed, and under
        ``robots_policy=enforce`` it declines to run. At the default
        ``warn`` it runs with the ladder pacing requests.
        """
        url = f"https://{HOST}/results"
        robots = getattr(ctx.ladder, "robots", None)
        if robots is not None:
            from ..errors import ProviderError

            info = await robots.check(url)
            if not robots.permits(info):
                raise ProviderError(self.name, "disallowed_by_robots")
        limiter = getattr(ctx.ladder, "limiter", None)
        if limiter is not None:
            await limiter.acquire(DOMAIN)
