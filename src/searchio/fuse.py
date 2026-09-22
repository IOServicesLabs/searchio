"""Merging results from providers that do not agree on anything.

Every provider returns a ranked list with its own scoring scheme, and those
scores are not comparable: DuckDuckGo relevance, an OpenAlex citation count,
and a GitHub star count live in different universes. Normalizing them against
each other requires calibration data nobody has.

Reciprocal Rank Fusion sidesteps that entirely by throwing the scores away and
keeping only the ranks::

    score(d) = sum over providers of  1 / (k + rank(d))

It is the standard answer for exactly this situation and it has two properties
worth the tradeoff. A document several providers rank highly beats one that a
single provider loves, which is the agreement signal we want. And ``k`` (60 by
convention) flattens the top of each list so the difference between rank 1 and
rank 2 does not dominate the difference between "found" and "not found".

Deduplication runs first, because fusing before deduping counts the same
document twice and lets a URL with three tracking-parameter variants outrank
everything.
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Doc, Item

RRF_K = 60

# Parameters that never change what a page says.
_JUNK_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "fbclid", "gclid", "gbraid", "wbraid", "msclkid", "mc_cid", "mc_eid",
    "ref", "ref_", "referrer", "source", "spm", "_ga", "igshid", "yclid",
    "ck_subscriber_id", "sc_channel", "sc_campaign", "trk", "trkCampaign",
}

_AMAZON_DP = re.compile(r"^(?P<pre>/(?:[^/]+/)?)?(?:dp|gp/product)/(?P<asin>[A-Z0-9]{10})", re.I)


def canonical_url(url: str) -> str:
    """A stable key for "the same page".

    Conservative on purpose. Stripping a parameter that *does* select content
    (``?p=2``, ``?id=99``) silently merges two different pages into one, which
    is a worse failure than keeping a duplicate.
    """
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return url
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    host = p.netloc.lower().split("@")[-1]
    host = host.removeprefix("www.").removesuffix(":443").removesuffix(":80")
    path = p.path or "/"

    # Amazon serves one product under many slug-decorated paths; the ASIN is
    # the only part that identifies it. Match every Amazon TLD (amazon.com,
    # amazon.co.uk, amazon.de, ...) and subdomains (smile.amazon.com) -- the
    # old ``endswith("amazon.com")`` both MISSED the international domains
    # (bug 46) and FALSE-MATCHED "notamazon.com" (which endswith amazon.com).
    # A leading-label or dot-delimited "amazon" is the registrable tell.
    if host.startswith("amazon.") or ".amazon." in host:
        m = _AMAZON_DP.search(path)
        if m:
            path = f"/dp/{m.group('asin').upper()}"
            return urlunsplit((scheme, host, path, "", ""))

    if len(path) > 1:
        path = path.rstrip("/")
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
               if k.lower() not in _JUNK_PARAMS)
    )
    return urlunsplit((scheme, host, path, query, ""))


# ── near-duplicate detection ─────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _shingles(text: str, n: int = 3) -> set[str]:
    words = _TOKEN.findall(text.lower())
    if len(words) < n:
        return set(words)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedupe_docs(docs: list[Doc], *, near_threshold: float = 0.82) -> list[Doc]:
    """Collapse duplicate documents, keeping the best-ranked copy.

    Exact canonical-URL match first (cheap, certain), then a shingle-overlap
    pass over titles+snippets to catch syndicated articles republished under
    different URLs -- the same wire story on four news sites.
    """
    by_url: dict[str, Doc] = {}
    for d in docs:
        # An empty URL is not a shared identity (bug 179): canonical_url("")
        # normalises to "https:///" (its path defaults to "/"), so every
        # urlless doc keyed to that one slot and distinct docs merged into a
        # franken-record. Give each its own key, as merge_items already does
        # for urlless rows.
        key = canonical_url(d.url) if d.url.strip() else f"_{id(d)}"
        prev = by_url.get(key)
        if prev is None:
            by_url[key] = d
            continue
        # Merge: keep the richer record, remember every provider that found it.
        keep, drop = (prev, d) if prev.rank <= d.rank else (d, prev)
        keep.text = keep.text or drop.text
        keep.snippet = keep.snippet or drop.snippet
        keep.title = keep.title or drop.title
        keep.published = keep.published or drop.published
        srcs = set(keep.meta.get("sources", [keep.source])) | {drop.source}
        keep.meta["sources"] = sorted(s for s in srcs if s)
        by_url[key] = keep

    unique = list(by_url.values())
    out: list[Doc] = []
    for d in unique:
        probe = f"{d.title} {d.snippet}"
        dup_of = None
        for kept in out:
            if kept.domain == d.domain:
                continue  # same site, different page: not a syndication dupe
            if jaccard(probe, f"{kept.title} {kept.snippet}") >= near_threshold:
                dup_of = kept
                break
        if dup_of is None:
            out.append(d)
        else:
            dup_of.meta.setdefault("near_duplicates", []).append(d.url)
    return out


# ── fusion ───────────────────────────────────────────────────────────────────


def rrf(lists: dict[str, list[Doc]], *, k: int = RRF_K, weights: dict[str, float] | None = None) -> list[Doc]:
    """Fuse per-provider ranked lists into one ranking.

    ``weights`` lets the router express that a provider is more trustworthy for
    the current intent (OpenAlex for academic, say) without having to make its
    raw scores comparable to anyone else's.
    """
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, Doc] = {}
    found_by: dict[str, set[str]] = defaultdict(set)
    #: A dated copy loses the per-key best-doc pick to any better-ranked
    #: undated copy; remember the date so the winner can inherit it.
    dated: dict[str, str] = {}

    for provider, docs in lists.items():
        w = weights.get(provider, 1.0)
        for rank, d in enumerate(docs):
            key = canonical_url(d.url)
            scores[key] += w / (k + rank + 1)
            found_by[key].add(provider)
            if d.published:
                dated.setdefault(key, d.published)
            if key not in best or d.rank < best[key].rank:
                best[key] = d

    out: list[Doc] = []
    for key, score in scores.items():
        d = best[key]
        d.published = d.published or dated.get(key)
        d.score = score
        d.meta["sources"] = sorted(found_by[key])
        d.meta["agreement"] = len(found_by[key])
        out.append(d)
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _collapse_syndication(docs: list[Doc], *, near_threshold: float = 0.82) -> list[Doc]:
    """The near-duplicate (syndication) pass over an already-URL-unique list.

    dedupe_docs runs this per-provider, but fuse keys rrf by canonical_url --
    so the SAME wire story arriving from two providers under different URLs
    (nytimes via ddg, cnn via bing) never got compared and both survived into
    the final ranking (bug 45), the exact case the module docstring promises
    to handle. Run over the rrf-sorted list, the higher-scored copy wins and
    records the rest under meta['near_duplicates']; the same-domain guard
    keeps distinct pages of one site apart, as in dedupe_docs.
    """
    out: list[Doc] = []
    for d in docs:
        probe = f"{d.title} {d.snippet}"
        dup_of = None
        for kept in out:
            if kept.domain == d.domain:
                continue
            if jaccard(probe, f"{kept.title} {kept.snippet}") >= near_threshold:
                dup_of = kept
                break
        if dup_of is None:
            out.append(d)
        else:
            dup_of.meta.setdefault("near_duplicates", []).append(d.url)
    return out


def fuse(lists: dict[str, list[Doc]], *, k: int = RRF_K,
         weights: dict[str, float] | None = None, limit: int = 20) -> list[Doc]:
    """Dedupe, then fuse, then collapse cross-provider syndication, then trim.

    The order matters (see the module docstring): exact/URL dedup runs
    per-provider BEFORE rrf so a tracking-param variant cannot outrank
    everything, and the cross-provider near-duplicate collapse runs AFTER rrf,
    on the fused ranking, where syndicated copies from different providers
    finally meet (bug 45).
    """
    cleaned = {name: dedupe_docs(docs) for name, docs in lists.items() if docs}
    fused = rrf(cleaned, k=k, weights=weights)
    return _collapse_syndication(fused)[:limit]


# ── items ────────────────────────────────────────────────────────────────────

#: Availability suffixes (schema.org, lowercased) that mean a buyer cannot
#: take the offer now. An EMPTY availability is unknown, not stale: most
#: retailers emit none at all, and demoting unknowns would bury live listings
#: whose pages simply stay quiet about stock.
_STALE_AVAILABILITY = frozenset({"outofstock", "soldout", "discontinued"})


def _is_stale(availability: str) -> bool:
    return availability.strip().lower() in _STALE_AVAILABILITY


def merge_items(items: list[Item]) -> list[Item]:
    """Collapse listings of the same product into one row per identity.

    Offers for the same product from different sellers are kept on the winner
    under ``meta['offers']`` rather than thrown away, because "who else sells
    this and for how much" is usually the actual question. The representative
    row is the cheapest LIVE offer with a real price -- a sold cheapest offer
    used to headline the row ("$10,000, OutOfStock") and mask a live pricier
    one in ``meta['offers']``. When nothing is live the cheapest stale quote
    still represents the group, honestly marked.
    """
    groups: dict[str, list[Item]] = defaultdict(list)
    loners: list[Item] = []
    for it in items:
        if it.identity:
            groups[it.identity].append(it)
        else:
            loners.append(it)

    out: list[Item] = []
    for identity, group in groups.items():
        # The same listing can arrive twice in one identity group (sponsored
        # and organic slots on the same SERP, or two workers fetching the same
        # page). Without this collapse the loser lands in meta['offers'] and
        # the row offers ITSELF as an alternative seller -- a nonsense row an
        # agent will happily quote. Keep the richer copy; first wins ties so
        # the result is stable.
        def _richness(g: Item) -> tuple:
            return (g.price.amount is not None, g.rating is not None,
                    g.reviews is not None, bool(g.brand), bool(g.image))

        by_url: dict[str, Item] = {}
        for g in group:
            key = canonical_url(g.url) if g.url else f"_{id(g)}"
            if key not in by_url or _richness(g) > _richness(by_url[key]):
                by_url[key] = g
        group = list(by_url.values())

        priced = [g for g in group if g.price.amount is not None]
        if priced:
            # "Cheapest" only means something within one currency (iteration
            # 53 rider): min over raw amounts let a 1 EUR offer beat 250 USD.
            # Compare inside the currency most offers share; ties keep the
            # first seen so the pick is stable.
            counts: dict[str, int] = defaultdict(int)
            for g in priced:
                counts[g.price.currency or ""] += 1
            majority = max(counts, key=lambda c: (counts[c], c == priced[0].price.currency))
            pool = [g for g in priced if (g.price.currency or "") == majority]
            # Live offers rank before stale ones, cheapest within each band;
            # only an all-stale group quotes a sold price (documented above).
            winner = min(pool, key=lambda g: (_is_stale(g.availability), g.price.amount))
        else:
            winner = group[0]
        others = [g for g in group if g is not winner]
        if others:
            winner.meta["offers"] = [
                {
                    "url": o.url,
                    "seller": o.seller or o.domain,
                    "price": o.price.amount,
                    "currency": o.price.currency,
                    "availability": o.availability,
                    "condition": o.condition,
                }
                for o in others
            ]
            winner.meta["offer_count"] = len(group)
        # Fill gaps from siblings: one site lists the brand, another the rating.
        for o in others:
            winner.brand = winner.brand or o.brand
            winner.image = winner.image or o.image
            winner.rating = winner.rating if winner.rating is not None else o.rating
            winner.reviews = winner.reviews if winner.reviews is not None else o.reviews
        out.append(winner)

    out.extend(loners)
    # Currency-aware final order (bug 178): the winner pick already refuses to
    # compare amounts across currencies ("cheapest only means something within
    # one currency"), but the final sort used raw amounts regardless -- a EUR 5
    # row would slot between a $3 and a $10 row by the number 5. Order by the
    # most-common currency first (what a single-region shopper is looking at),
    # cheapest within each currency; a currency name breaks ties so it stays
    # deterministic. Single-currency results (the common case) are unchanged:
    # every row shares one currency, so this collapses to (unpriced-last,
    # amount).
    cur_counts: dict[str, int] = defaultdict(int)
    for i in out:
        if i.price.amount is not None:
            cur_counts[i.price.currency or ""] += 1

    def _sort_key(i: Item) -> tuple:
        unpriced = i.price.amount is None
        cur = i.price.currency or ""
        # Higher count first (negated), then currency name, then amount.
        return (unpriced, -cur_counts.get(cur, 0), cur, i.price.amount or 0.0)

    out.sort(key=_sort_key)
    return out
