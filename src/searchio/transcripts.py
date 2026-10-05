"""YouTube caption transcripts.

The YouTube provider's own leg (:mod:`searchio.providers.youtube`) is
search-only: it reads ``ytInitialData`` off the results page and stops
there. Pulling a transcript is a different primitive -- no search, one
video, no page at all -- so it lives here, and the Engine, the HTTP server,
the MCP server and the CLI all call into it.

Every *direct* caption path is gated as of 2026-09: the timedtext
``baseUrl`` from ``ytInitialPlayerResponse.captions`` answers 200 with zero
bytes (PO-token gate, ``xoaf=4`` included), and the Innertube
``youtubei/v1/get_transcript`` POST fails its precondition check in every
body/header shape. ``youtube-transcript-api`` maintains the PO-token
handshake internally and is the proven leg (control video dQw4w9WgXcQ: 61
snippets, ~0.5s), so the fetch runs in a worker thread via
:func:`asyncio.to_thread` and the event loop never blocks on YouTube. If a
future update breaks it, the arms race moved again -- re-probe the direct
paths before assuming this is still the only way.

The segment shape deliberately mirrors SwarmIO's ``media_sidecar`` captions
verb (``start`` / ``end`` / ``text``), so a caller can treat a searchio
transcript and a SwarmIO caption payload identically.
"""

from __future__ import annotations

import asyncio
import html
import re
from urllib.parse import parse_qs, urlsplit

from .errors import ConfigError, TranscriptError
from .models import Transcript, TranscriptSegment

_FULL_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_KINDS = ("shorts", "embed", "live", "v")
_HOST_PREFIX = re.compile(r"^(www\.|m\.|music\.)")

# youtube-transcript-api exception names -> failure kind. Mapped by name (not
# imported) so a yta upgrade that renames one degrades to "upstream", never
# an ImportError at module load; and so tests can fake them with same-named
# classes instead of constructing real yta exceptions.
_UNAVAILABLE = frozenset({
    "VideoUnavailable", "VideoUnplayable", "AgeRestricted",
    "TranscriptsDisabled", "NoTranscriptFound", "InvalidVideoId",
})
_BLOCKED = frozenset({"RequestBlocked", "IpBlocked", "PoTokenRequired"})


def extract_video_id(target: str) -> str:
    """Accept a bare id or any common YouTube URL shape, return the id.

    Raises :class:`TranscriptError` with kind ``invalid_target`` for anything
    else -- a malformed string is the caller's bug, not YouTube's answer.
    """
    t = (target or "").strip()
    if _FULL_ID.fullmatch(t):
        return t
    if "youtu" not in t:
        raise TranscriptError("invalid_target", f"not a YouTube video id or URL: {target!r}")
    parts = urlsplit(t if "//" in t else f"https://{t}")
    host = _HOST_PREFIX.sub("", parts.netloc.lower())
    if host == "youtu.be":
        m = _FULL_ID.fullmatch(parts.path.strip("/"))
        if m:
            return m.group(0)
    if host in ("youtube.com", "youtube-nocookie.com"):
        segs = [s for s in parts.path.split("/") if s]
        if segs and segs[0] == "watch":
            m = _FULL_ID.fullmatch(parse_qs(parts.query).get("v", [""])[0])
            if m:
                return m.group(0)
        elif len(segs) >= 2 and segs[0] in _PATH_KINDS:
            m = _FULL_ID.fullmatch(segs[1])
            if m:
                return m.group(0)
    raise TranscriptError("invalid_target", f"could not find a video id in {target!r}")


def _load_api():
    """Import youtube-transcript-api at call time, not module time.

    It is a declared dependency, so this should never fire; kept as a clear
    error rather than an AttributeError deep in a thread, for editable and
    partial installs.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError(
            "youtube-transcript-api is required for transcripts "
            "(pip install youtube-transcript-api)"
        ) from exc
    return YouTubeTranscriptApi()


def _clean(text: str) -> str:
    """One segment, one line: captions embed newlines and HTML entities."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _map_error(exc: Exception) -> TranscriptError:
    name = type(exc).__name__
    if name in _UNAVAILABLE:
        kind = "unavailable"
    elif name in _BLOCKED:
        kind = "blocked"
    else:
        kind = "upstream"
    return TranscriptError(kind, f"{name}: {str(exc)[:400]}")


async def fetch_transcript(
    target: str,
    *,
    languages: list[str] | tuple[str, ...] = ("en",),
) -> Transcript:
    """Fetch the caption track of one YouTube video.

    ``target`` is a watch / youtu.be / shorts / embed / live URL or a bare
    11-char video id. ``languages`` is the preference order: the first
    language YouTube actually has wins (``de,en`` reads a German video in
    German). A video without captions, a blocked caption request, or a
    garbage target is a typed :class:`TranscriptError` -- never an honest
    empty, so a caller can tell "no transcript" apart from "nothing came
    back".
    """
    video_id = extract_video_id(target)
    langs = [l.strip() for l in languages if l and l.strip()] or ["en"]
    api = _load_api()
    try:
        fetched = await asyncio.to_thread(api.fetch, video_id, languages=langs)
    except Exception as exc:  # noqa: BLE001 -- every yta failure becomes a typed error
        raise _map_error(exc) from exc

    segments: list[TranscriptSegment] = []
    for snip in fetched.snippets:
        text = _clean(snip.text)
        if not text:
            continue
        start = round(float(snip.start), 2)
        segments.append(TranscriptSegment(
            start=start,
            end=round(float(snip.start) + float(snip.duration), 2),
            text=text,
        ))
    if not segments:
        raise TranscriptError(
            "unavailable", f"caption track for {video_id} parsed to zero segments"
        )
    return Transcript(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        lang=getattr(fetched, "language_code", "") or langs[0],
        language=getattr(fetched, "language", "") or "",
        source="captions",
        segment_count=len(segments),
        duration_sec=segments[-1].end,
        text=" ".join(s.text for s in segments),
        segments=segments,
    )
