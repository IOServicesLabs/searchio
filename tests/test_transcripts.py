"""YouTube transcripts: id extraction, normalization, error mapping.

Offline by construction: the youtube-transcript-api client is stubbed at
the ``_load_api`` seam, so nothing here touches YouTube. The live proof
(control video dQw4w9WgXcQ, 61 snippets, ~0.5s) lives in bench/ytwall.py;
run it to re-check the library still wins the PO-token arms race.
"""

from __future__ import annotations

import pytest

import searchio.transcripts as tr
from searchio.errors import ConfigError, TranscriptError


# ── id extraction ────────────────────────────────────────────────────────────


def test_bare_id_passthrough():
    assert tr.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_watch_url_with_extra_params():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=43s&list=RD"
    assert tr.extract_video_id(url) == "dQw4w9WgXcQ"


def test_youtu_be():
    assert tr.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert tr.extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=1") == "dQw4w9WgXcQ"


def test_shorts_embed_live_v_paths():
    for p in ("shorts", "embed", "live", "v"):
        assert tr.extract_video_id(f"https://www.youtube.com/{p}/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_subdomains_music_and_mobile():
    assert tr.extract_video_id("https://music.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert tr.extract_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_bare_host_without_scheme():
    assert tr.extract_video_id("youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


@pytest.mark.parametrize("bad", [
    "", "   ", "hello", "dQw4w9WgXc",          # 10 chars
    "dQw4w9WgXcQQ",                            # 12 chars
    "https://example.com/watch?v=dQw4w9WgXcQ", # wrong host
    "https://youtu.be/short",
    "https://www.youtube.com/feed/trending",
])
def test_invalid_targets_raise_invalid_target(bad):
    with pytest.raises(TranscriptError) as exc:
        tr.extract_video_id(bad)
    assert exc.value.kind == "invalid_target"


# ── fetch normalization ──────────────────────────────────────────────────────


class FakeSnippet:
    def __init__(self, text, start, duration):
        self.text, self.start, self.duration = text, start, duration


class FakeFetched:
    language_code = "en"
    language = "English"

    def __init__(self, snippets):
        self.snippets = snippets


class FakeApi:
    """Same duck-type surface youtube-transcript-api's client exposes."""

    def __init__(self, fetched=None, exc=None):
        self._fetched, self._exc = fetched, exc
        self.calls: list[tuple] = []

    def fetch(self, video_id, *, languages):
        self.calls.append((video_id, list(languages)))
        if self._exc is not None:
            raise self._exc
        return self._fetched


def _fetched():
    return FakeFetched([
        FakeSnippet("[&#9835;]  hello\n   world &#9835;", 1.005, 2.0),
        FakeSnippet("", 3.0, 1.0),  # blank after cleaning: dropped
        FakeSnippet("next line", 3.006, 1.5),
    ])


async def test_fetch_normalizes_and_threads_languages(monkeypatch):
    api = FakeApi(fetched=_fetched())
    monkeypatch.setattr(tr, "_load_api", lambda: api)
    t = await tr.fetch_transcript("https://youtu.be/dQw4w9WgXcQ", languages=["de", "en"])
    assert api.calls == [("dQw4w9WgXcQ", ["de", "en"])]
    assert t.video_id == "dQw4w9WgXcQ"
    assert t.url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert t.lang == "en" and t.language == "English"
    assert t.source == "captions"
    # newline -> space, entities decoded, whitespace squeezed, blanks dropped
    assert [s.text for s in t.segments] == ["[♫] hello world ♫", "next line"]
    assert [(s.start, s.end) for s in t.segments] == [(1.0, 3.0), (3.01, 4.51)]
    assert t.segment_count == 2
    assert t.duration_sec == 4.51
    assert t.text == "[♫] hello world ♫ next line"


async def test_default_language_is_en(monkeypatch):
    api = FakeApi(fetched=_fetched())
    monkeypatch.setattr(tr, "_load_api", lambda: api)
    await tr.fetch_transcript("dQw4w9WgXcQ")
    assert api.calls[0][1] == ["en"]


async def test_empty_languages_falls_back_to_en(monkeypatch):
    api = FakeApi(fetched=_fetched())
    monkeypatch.setattr(tr, "_load_api", lambda: api)
    await tr.fetch_transcript("dQw4w9WgXcQ", languages=["", "  "])
    assert api.calls[0][1] == ["en"]


async def test_track_that_parses_to_nothing_is_unavailable(monkeypatch):
    api = FakeApi(fetched=FakeFetched([]))
    monkeypatch.setattr(tr, "_load_api", lambda: api)
    with pytest.raises(TranscriptError) as exc:
        await tr.fetch_transcript("dQw4w9WgXcQ")
    assert exc.value.kind == "unavailable"


# ── error mapping ────────────────────────────────────────────────────────────

# Name-based mapping means same-named stand-ins exercise the real branches;
# a yta rename degrades to "upstream", never an ImportError.


async def test_unavailable_mapping(monkeypatch):
    class NoTranscriptFound(Exception):
        pass

    monkeypatch.setattr(tr, "_load_api", lambda: FakeApi(exc=NoTranscriptFound("none")))
    with pytest.raises(TranscriptError) as exc:
        await tr.fetch_transcript("dQw4w9WgXcQ")
    assert exc.value.kind == "unavailable"
    assert "NoTranscriptFound" in str(exc.value)


async def test_blocked_mapping(monkeypatch):
    class IpBlocked(Exception):
        pass

    monkeypatch.setattr(tr, "_load_api", lambda: FakeApi(exc=IpBlocked("429")))
    with pytest.raises(TranscriptError) as exc:
        await tr.fetch_transcript("dQw4w9WgXcQ")
    assert exc.value.kind == "blocked"


async def test_unexpected_error_is_upstream(monkeypatch):
    class YouTubeDataUnparsable(Exception):
        pass

    monkeypatch.setattr(tr, "_load_api", lambda: FakeApi(exc=YouTubeDataUnparsable("shape changed")))
    with pytest.raises(TranscriptError) as exc:
        await tr.fetch_transcript("dQw4w9WgXcQ")
    assert exc.value.kind == "upstream"


async def test_missing_dependency_is_a_config_error(monkeypatch):
    def _raise():
        raise ConfigError("youtube-transcript-api is required for transcripts")

    monkeypatch.setattr(tr, "_load_api", _raise)
    with pytest.raises(ConfigError):
        await tr.fetch_transcript("dQw4w9WgXcQ")


# ── engine wiring ────────────────────────────────────────────────────────────


async def test_engine_transcript_delegates(monkeypatch):
    """Engine.transcript must thread (target, lang) into fetch_transcript;
    a lazy-import typo here would ship as a NameError in production."""
    calls: list[dict] = []

    async def _fake(target, *, languages):
        calls.append({"target": target, "languages": languages})
        from searchio.models import Transcript

        return Transcript(video_id="dQw4w9WgXcQ")

    monkeypatch.setattr(tr, "fetch_transcript", _fake)
    from searchio.engine import Engine

    eng = Engine.__new__(Engine)  # no ladder/router: this method uses neither
    out = await eng.transcript(" https://youtu.be/dQw4w9WgXcQ ", lang=" de ")
    assert out.video_id == "dQw4w9WgXcQ"
    assert calls == [{"target": " https://youtu.be/dQw4w9WgXcQ ", "languages": ["de"]}]
