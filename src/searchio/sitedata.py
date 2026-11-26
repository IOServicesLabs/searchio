"""Pages that load with their data in them but extract to almost nothing.

Two cases measured on the walled-site baseline (bench/walled.py):

* YouTube results / channel pages: every video is in ``ytInitialData``, the
  visible markup is a footer (164-179 chars from 1.4-3.6 MB).
* X / Twitter profile, status and timeline pages: the posts are in a Relay
  payload streamed as an escaped JSON string inside a script; the prose
  extractor returned inline ``window.__xClientTextFormatters...`` calls and
  one to three posts.

Each reader is pure (html in, text out) and returns "" when the page does
not hold what it expects, so a caller keeps whatever it had.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from urllib.parse import parse_qs, urlsplit

_MAX_VIDEOS = 60
_MAX_POSTS = 40

# ── YouTube ────────────────────────────────────────────────────────────────

_YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com")


def is_youtube_list(url: str) -> bool:
    """A YouTube page that lists videos (results, channel tabs, playlists),
    never a watch page: that one has its own reader."""
    u = urlsplit(url)
    if (u.hostname or "").lower() not in _YT_HOSTS:
        return False
    path = u.path or "/"
    return not path.startswith(("/watch", "/embed", "/shorts/", "/live/"))


def _vm_text(node) -> str:
    """``{"content": "..."}`` of a view-model text node."""
    return str(node.get("content") or "").strip() if isinstance(node, dict) else ""


def _lockup_rows(data: dict, seen: set[str], default_channel: str) -> list[str]:
    """Rows for ``lockupViewModel`` cards: the layout channel tabs (and some
    result shelves) use, which carry no ``videoRenderer`` at all. Ad lockups
    have no ``lockupMetadataViewModel`` and are skipped."""
    from .providers.youtube import HOST, _walk

    out: list[str] = []
    for card in _walk(data, "lockupViewModel"):
        vid = str(card.get("contentId") or "")
        ctype = str(card.get("contentType") or "")
        meta = (card.get("metadata") or {}).get("lockupMetadataViewModel")
        if not vid or vid in seen or not isinstance(meta, dict) or "VIDEO" not in ctype:
            continue
        title = " ".join(_vm_text(meta.get("title")).split())
        if not title:
            continue
        seen.add(vid)
        parts: list[str] = []
        rows = ((meta.get("metadata") or {}).get("contentMetadataViewModel") or {}).get("metadataRows") or []
        for row in rows if isinstance(rows, list) else []:
            for part in (row.get("metadataParts") or []) if isinstance(row, dict) else []:
                if not isinstance(part, dict):
                    continue
                # The spoken label is the full figure ("55 thousand views").
                txt = str(part.get("accessibilityLabel") or "") or _vm_text(part.get("text"))
                if txt:
                    parts.append(txt)
        length = ""
        for badge in _walk(card.get("contentImage") or {}, "thumbnailBadgeViewModel"):
            length = str(badge.get("text") or "")
            if length:
                break
        if default_channel and not any(default_channel.lower() in p.lower() for p in parts):
            parts.insert(0, default_channel)
        if length:
            parts.append(length)
        out.append(" | ".join([title, *parts, f"https://{HOST}/watch?v={vid}"]))
    return out


def _shorts_rows(data: dict, seen: set[str]) -> list[str]:
    from .providers.youtube import HOST, _walk

    out: list[str] = []
    for card in _walk(data, "shortsLockupViewModel"):
        ent = str(card.get("entityId") or "")
        vid = ent.rsplit("shorts-shelf-item-", 1)[-1] if "shorts-shelf-item-" in ent else ""
        over = card.get("overlayMetadata") or {}
        title = " ".join(_vm_text(over.get("primaryText")).split())
        if not vid or vid in seen or not title:
            continue
        seen.add(vid)
        views = _vm_text(over.get("secondaryText"))
        out.append(" | ".join(x for x in (title, "Short", views, f"https://{HOST}/shorts/{vid}") if x))
    return out


def youtube_list_text(html: str, url: str = "") -> str:
    """Every video a results / channel / playlist page holds, one row each:
    ``title | channel | views | age | length | watch URL``. "" when the page
    has no ``ytInitialData`` or no video in it."""
    from .providers.youtube import _text, _videos_from_data, _walk, extract_initial_data

    data = extract_initial_data(html)
    if not data:
        return ""
    rows: list[str] = []
    seen: set[str] = set()
    for doc in _videos_from_data(data):
        m = doc.meta
        if m.get("shorts"):
            continue
        seen.add(str(m.get("video_id")))
        parts = [doc.title, m.get("channel") or "", m.get("views_raw") or "", m.get("age_text") or "",
                 "LIVE" if m.get("live") else (m.get("duration") or ""), doc.url]
        row = " | ".join(p for p in parts if p)
        if m.get("description"):
            row += f"\n  {m['description']}"
        rows.append(row)
    channel, about = "", ""
    for meta in _walk(data, "channelMetadataRenderer"):
        channel = str(meta.get("title") or "")
        about = " ".join(str(meta.get("description") or "").split())[:500]
        break
    rows += _lockup_rows(data, seen, channel)
    # Grid cards of the older channel layout.
    for r in _walk(data, "gridVideoRenderer"):
        vid = str(r.get("videoId") or "")
        title = " ".join(_text(r.get("title")).split())
        if not vid or vid in seen or not title:
            continue
        seen.add(vid)
        parts = [title, channel, _text(r.get("viewCountText")), _text(r.get("publishedTimeText")),
                 f"https://www.youtube.com/watch?v={vid}"]
        rows.append(" | ".join(p for p in parts if p))
    shorts = _shorts_rows(data, seen)
    if not rows and not shorts:
        return ""
    q = (parse_qs(urlsplit(url).query).get("search_query") or [""])[0]
    head = []
    if q:
        head.append(f'YouTube results for "{q}"')
    elif channel:
        head.append(f"YouTube channel: {channel}")
        if about:
            head.append(about)
    out = "\n\n".join(head)
    if rows:
        out += ("\n\n" if out else "") + "## Videos\n\n" + "\n".join(f"- {r}" for r in rows[:_MAX_VIDEOS])
    if shorts:
        out += ("\n\n" if out else "") + "## Shorts\n\n" + "\n".join(f"- {r}" for r in shorts[:20])
    return out


# ── X / Twitter ────────────────────────────────────────────────────────────

_X_HOSTS = ("x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com", "mobile.x.com")
_SCRIPT = re.compile(r"<script\b[^>]*>(.*?)</script>", re.S | re.I)
_JS_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)
#: What the prose extractor leaks from X's streamed markup.
X_SCRIPT_JUNK = re.compile(
    r"window\.__xClientTextFormatters\?\.run\([^;]*\);?"
    r"|\[(?:\"[\w-]+\",?)+\]\.forEach\(function\(name\)\{.*?document\.currentScript\.remove\(\)", re.S)


def is_x(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in _X_HOSTS


def _x_payloads(html: str):
    """Each JSON document X streamed as a JS string literal (one per Relay
    query), decoded. A literal that is not JSON, or has no post or user in
    it, is skipped unparsed."""
    if "full_text" not in html and "screen_name" not in html:
        return
    for sm in _SCRIPT.finditer(html):
        body = sm.group(1)
        if 'full_text\\"' not in body and 'screen_name\\"' not in body:
            continue
        for lit in _JS_STRING.finditer(body):
            raw = lit.group(0)
            if len(raw) < 200 or not raw.startswith('"{\\"'):
                continue
            try:
                data = json.loads(json.loads(raw))
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict):
                yield data


def _x_walk(node, depth: int = 0):
    """Every post object (one with ``details.full_text``) under ``node``."""
    if depth > 60:
        return
    if isinstance(node, dict):
        det = node.get("details")
        if isinstance(det, dict) and isinstance(det.get("full_text"), str):
            yield node
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _x_walk(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                yield from _x_walk(v, depth + 1)


def _x_users(node, depth: int = 0):
    if depth > 60:
        return
    if isinstance(node, dict):
        if node.get("__typename") == "User" and isinstance(node.get("core"), dict) \
                and node["core"].get("screen_name"):
            yield node
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _x_users(v, depth + 1)
    elif isinstance(node, list):
        for v in node:
            if isinstance(v, (dict, list)):
                yield from _x_users(v, depth + 1)


def _n(v) -> str:
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return ""


def _x_post(tw: dict) -> tuple[str, str, int] | None:
    """(post id, text block, created ms) for one post object."""
    det = tw.get("details") or {}
    text = str(det.get("full_text") or "")
    note = tw.get("note_tweet")
    if isinstance(note, dict):
        # A long post: ``full_text`` is its first 280 chars.
        for cand in _x_strings(note, "text"):
            if len(cand) > len(text):
                text = cand
    # t.co wrappers -> where they go; media wrappers say nothing.
    for ent in tw.get("url_entities") or []:
        if isinstance(ent, dict) and ent.get("url") and ent.get("expanded_url"):
            text = text.replace(str(ent["url"]), str(ent["expanded_url"]))
    media = [m for m in (tw.get("media_entities2") or []) if isinstance(m, dict)]
    text = re.sub(r"\s*https://t\.co/\w+", "", text).strip()
    user = (((tw.get("core") or {}).get("user_results") or {}).get("result") or {}).get("core") or {}
    name, handle = str(user.get("name") or ""), str(user.get("screen_name") or "")
    pid = str(tw.get("rest_id") or "")
    ms = det.get("created_at_ms")
    when = ""
    if isinstance(ms, (int, float)) and ms > 0:
        when = _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c = tw.get("counts") or {}
    counts = ", ".join(f"{_n(c.get(k))} {label}" for k, label in (
        ("reply_count", "replies"), ("retweet_count", "reposts"), ("favorite_count", "likes"),
        ("quote_count", "quotes")) if _n(c.get(k)))
    views = _n((tw.get("views") or {}).get("count"))
    if views:
        counts = f"{counts}, {views} views" if counts else f"{views} views"
    if not text and not media:
        return None
    who = f"{name} (@{handle})" if name and handle else (f"@{handle}" if handle else name)
    head = " | ".join(x for x in (who, when, counts) if x)
    kinds = sorted({str(m.get("type") or "media") for m in media})
    body = text + (f"\n  [{', '.join(kinds)} attached]" if kinds else "")
    link = f"https://x.com/{handle}/status/{pid}" if handle and pid else ""
    block = f"- {head}\n  " + body.replace("\n", "\n  ") + (f"\n  {link}" if link else "")
    return pid or text[:80], block, int(ms) if isinstance(ms, (int, float)) else 0


def _x_strings(node, key: str, depth: int = 0):
    if depth > 8:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str):
                yield v
            elif isinstance(v, (dict, list)):
                yield from _x_strings(v, key, depth + 1)
    elif isinstance(node, list):
        for v in node:
            yield from _x_strings(v, key, depth + 1)


def x_posts_text(html: str, url: str = "") -> str:
    """The posts (and, on a profile, the account) an X page was served with:
    author, time, text, counts, link. "" when the page streamed none -- a
    login wall, a search page."""
    posts: dict[str, tuple[str, int]] = {}
    order: list[str] = []
    users: dict[str, dict] = {}
    for data in _x_payloads(html):
        for tw in _x_walk(data):
            got = _x_post(tw)
            if got is None:
                continue
            pid, block, ms = got
            # The same post may arrive twice (a hero copy and a thread copy);
            # the fuller copy is kept, in first-seen order.
            if pid not in posts:
                order.append(pid)
                posts[pid] = (block, ms)
            elif len(block) > len(posts[pid][0]):
                posts[pid] = (block, ms)
        for u in _x_users(data):
            handle = str(u["core"]["screen_name"]).lower()
            if len(json.dumps(u)) > len(json.dumps(users.get(handle, {}))):
                users[handle] = u
    if not posts:
        return ""
    path = [p for p in (urlsplit(url).path or "").split("/") if p]
    out: list[str] = []
    status_id = path[2] if len(path) >= 3 and path[1] == "status" else ""
    owner = users.get(path[0].lower()) if path else None
    if owner and not status_id:
        core = owner["core"]
        line = f"{core.get('name') or ''} (@{core.get('screen_name')})".strip()
        bio = str((owner.get("profile_bio") or {}).get("description") or "")
        rel = owner.get("relationship_counts") or {}
        facts = ", ".join(x for x in (
            f"{_n(rel.get('followers'))} followers" if _n(rel.get("followers")) else "",
            f"{_n(rel.get('following'))} following" if _n(rel.get("following")) else "",
            f"{_n((owner.get('tweet_counts') or {}).get('tweets'))} posts"
            if _n((owner.get("tweet_counts") or {}).get("tweets")) else "",
            str((owner.get("location") or {}).get("location") or "")) if x)
        out.append("\n".join(x for x in (line, bio, facts) if x))
    if status_id and status_id in posts:
        out.append("## Post\n\n" + posts[status_id][0])
        rest = [posts[p][0] for p in order if p != status_id]
        if rest:
            out.append("## Replies\n\n" + "\n".join(rest[:_MAX_POSTS]))
    else:
        out.append("## Posts\n\n" + "\n".join(posts[p][0] for p in order[:_MAX_POSTS]))
        if owner:
            out.append("Only the posts X serves to a signed-out visitor are listed; "
                       "the full timeline needs a signed-in session.")
    return "\n\n".join(out)


# ── doc_from_fetch hook ────────────────────────────────────────────────────

def site_text(url: str, html: str, text: str, settings) -> str:
    """``text``, or the page's own data when a site reader has it. A reader
    that finds nothing leaves ``text`` alone."""
    if getattr(settings, "extract_youtube_lists", False) and is_youtube_list(url) \
            and "ytInitialData" in html:
        got = youtube_list_text(html, url)
        if got:
            # The visible markup of these pages is chrome (a footer); keep
            # any real prose there was, after the rows.
            return got if len(text.strip()) < 600 else f"{got}\n\n{text}"
    if getattr(settings, "extract_x_posts", False) and is_x(url):
        got = x_posts_text(html, url)
        if got:
            return got
        if "__xClientTextFormatters" in text:
            return X_SCRIPT_JUNK.sub(" ", text)
    return text
