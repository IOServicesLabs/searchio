"""Windows, focused passages and links over an extracted page.

A page read hands a model one slice of the text. Before this module the
slice was always the first N characters and the rest was unreachable: no
offset to continue from, no way to ask for the part that mentions the thing
being researched, and no links to follow. Three pure helpers:

``window``          a slice by offset, cut at a paragraph boundary
``focus_passages``  the paragraphs that match a query, in document order
``page_links``      the page's content links, absolute and de-duplicated
"""

from __future__ import annotations

import re
from urllib.parse import urldefrag, urljoin, urlsplit

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']+")
_STOP = frozenset(
    "the a an and or of to in on for with by is are was were be been it this that "
    "as at from how what which who when where why does do did can could should "
    "would about into over than then there their its not no yes".split()
)


def window(text: str, offset: int = 0, max_chars: int = 12_000) -> tuple[str, int, int | None]:
    """``(chunk, total, next_offset)``; ``next_offset`` is None at the end.

    The cut backs up to the last paragraph break in the final fifth of the
    window (else the last line break, else the last space), so a continued
    read never starts mid-sentence. Offsets are plain character offsets
    into the full text: ``window(text, next_offset)`` continues exactly.
    """
    total = len(text)
    offset = max(0, min(int(offset or 0), total))
    max_chars = int(max_chars or 0)
    if max_chars <= 0 or offset + max_chars >= total:
        return text[offset:], total, None
    end = offset + max_chars
    floor = offset + int(max_chars * 0.8)
    for sep in ("\n\n", "\n", " "):
        cut = text.rfind(sep, floor, end)
        if cut > offset:
            end = cut + len(sep)
            break
    return text[offset:end], total, end


def _terms(query: str) -> list[str]:
    seen: dict[str, None] = {}
    for w in _WORD.findall(query.lower()):
        if len(w) >= 2 and w not in _STOP:
            seen.setdefault(w, None)
    return list(seen)


def focus_passages(text: str, query: str, max_chars: int = 12_000) -> tuple[str, int]:
    """The paragraphs of ``text`` that best match ``query``.

    Returns ``(passages, matched)``. Each kept paragraph is prefixed with
    ``[@<offset>]`` -- its character offset in the full text -- so the reader
    can ask for the surrounding window. Paragraphs are scored by how many
    distinct query terms they contain (then by total hits), the best are
    taken until the budget is spent, and the result is in document order.
    ``matched`` is 0 when no paragraph contains any term: the caller should
    fall back to a plain window rather than return nothing.
    """
    terms = _terms(query)
    if not terms or not text:
        return "", 0
    paras: list[tuple[int, str]] = []
    pos = 0
    for block in text.split("\n\n"):
        if block.strip():
            paras.append((pos, block))
        pos += len(block) + 2
    scored: list[tuple[int, int, int]] = []  # (distinct, hits, index)
    for i, (_off, block) in enumerate(paras):
        low = block.lower()
        hits = [low.count(t) for t in terms]
        distinct = sum(1 for h in hits if h)
        if distinct:
            scored.append((distinct, sum(hits), i))
    if not scored:
        return "", 0
    scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
    keep: set[int] = set()
    used = 0
    for _d, _h, i in scored:
        size = len(paras[i][1]) + 16
        if used and used + size > max_chars:
            continue
        keep.add(i)
        used += size
        if used >= max_chars:
            break
    out = []
    for i in sorted(keep):
        off, block = paras[i]
        out.append(f"[@{off}] {block[:max_chars]}")
    return "\n\n".join(out), len(scored)


_SKIP_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "#", "blob:")
_ASSET = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|ico|css|js|woff2?|ttf|mp4|mp3|zip)(?:$|\?)", re.I)


def page_links(html: str, base_url: str, limit: int = 40) -> list[dict[str, str]]:
    """Up to ``limit`` content links of a page as ``{"text", "url"}``.

    Links inside ``<article>`` / ``<main>`` come first (they are the page's
    own citations and pagination); the rest of the document fills what is
    left. Fragment-only, script and asset links are dropped, URLs are made
    absolute against the page's URL, and each target appears once.
    """
    if not html or limit <= 0 or "<a" not in html.lower():
        return []
    try:
        from selectolax.parser import HTMLParser

        tree = HTMLParser(html)
    except Exception:  # noqa: BLE001 -- links are an extra, never a failure
        return []
    base = base_url
    try:
        b = tree.css_first("base[href]")
        if b is not None:
            base = urljoin(base_url, b.attributes.get("href") or "")
    except Exception:  # noqa: BLE001
        pass
    page = urldefrag(base_url)[0]
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def take(nodes) -> None:
        for a in nodes:
            if len(out) >= limit:
                return
            href = (a.attributes.get("href") or "").strip()
            if not href or href.lower().startswith(_SKIP_SCHEMES):
                continue
            url = urldefrag(urljoin(base, href))[0]
            if not url.startswith(("http://", "https://")) or url == page or url in seen:
                continue
            if _ASSET.search(urlsplit(url).path):
                continue
            text = " ".join((a.text(deep=True, separator=" ") or "").split())[:120]
            if not text:
                continue
            seen.add(url)
            out.append({"text": text, "url": url})

    try:
        for sel in ("article a[href]", "main a[href]", "[role=main] a[href]"):
            take(tree.css(sel))
        take(tree.css("a[href]"))
    except Exception:  # noqa: BLE001
        return out
    return out
