"""The trust-cookie jar: cheap-tier responses bank retained cookie names.

The cookie-retention lever (user proposal, 2026-09-17): every tier's
responses -- 200s and challenge responses alike, which is where vendors set
their visitor ids -- contribute the retained names (CLEARANCE_COOKIES +
TRUST_COOKIES; analytics excluded by policy) to the per-domain clearance
store, so the next fetch on a host opens with continuity instead of cold.

These pin the harvest helper's filter/merge/UA semantics, the RFC-6265
scoping it inherits from _HopCookies (foreign Domain rejected, Secure
honored), and the tier-0 wiring (a real _tier0_document over a stubbed
_open_stream banks from both 200 and 403 responses without disturbing the
returned document).
"""

from __future__ import annotations

import httpx
import pytest

from searchio.config import Settings
from searchio.net.clearance import CLEARANCE_COOKIES, RETAINED_COOKIES, TRUST_COOKIES
from searchio.net.ladder import Ladder

from tests.test_ladder_router import HTML, GOOD, FakeLadder


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,  # keep tests fast; pacing is tested separately
        per_domain_burst=1000,
        sidecar_autostart=False,
    )

UA_PERSONA = "Mozilla/5.0 (X11; Linux x86_64) PersonaChrome/131.0"
UA_BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) BrowserChrome/131.0"


def harvest(lad, url, set_cookie_headers, ua=UA_PERSONA):
    host = url.split("/")[2]
    jar = Ladder._jar_from_set_cookie(set_cookie_headers, host)
    lad._harvest_trust(url, jar, ua)


class TestRetainedSet:
    """The retention policy itself: names in, names out, by design."""

    async def test_retained_is_clearance_plus_trust(self):
        assert RETAINED_COOKIES == CLEARANCE_COOKIES | TRUST_COOKIES
        assert "bm_sv" in TRUST_COOKIES and "_pxvid" in TRUST_COOKIES
        # The anti-bot trust names stay...
        assert "cf_clearance" in CLEARANCE_COOKIES
        assert "_abck" in CLEARANCE_COOKIES and "bm_sz" in CLEARANCE_COOKIES
        # ...and analytics/trackers stay OUT (they carry no trust; replaying
        # trackers makes the client more distinctive, not less).
        for name in ("_ga", "_gid", "_gcl_au", "NID", "MUID"):
            assert name not in RETAINED_COOKIES


class TestHarvestTrust:
    """Direct _harvest_trust semantics over the real clearance store."""

    async def test_banks_trust_names_from_a_plain_response(self, settings):
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "https://example.com/x", [
                "bm_sv=akamai-value; Domain=example.com; Secure",
                "_pxvid=px-visitor; Domain=example.com",
            ])
            row = lad.clearance.get("example.com")
            assert row is not None
            assert row.cookies == {"bm_sv": "akamai-value", "_pxvid": "px-visitor"}
            assert row.user_agent == UA_PERSONA
        finally:
            await lad.close()

    async def test_clearance_names_banked_too(self, settings):
        # A cheap-tier CHALLENGE response (403/429) is where datadome/PX set
        # their ids -- harvest must not be success-gated.
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "https://example.com/x", ["datadome=dd-value; Domain=example.com"])
            row = lad.clearance.get("example.com")
            assert row is not None and row.cookies == {"datadome": "dd-value"}
        finally:
            await lad.close()

    async def test_analytics_names_not_retained(self, settings):
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "https://example.com/x", [
                "_ga=tracker; Domain=example.com",
                "NID=google; Domain=example.com",
                "_pxvid=keeper; Domain=example.com",
            ])
            row = lad.clearance.get("example.com")
            assert row is not None and row.cookies == {"_pxvid": "keeper"}
        finally:
            await lad.close()

    async def test_merge_new_wins_per_name_existing_kept(self, settings):
        lad = FakeLadder(settings, {})
        try:
            lad.clearance.put("example.com", {"_abck": "old-abck", "bm_sz": "old-sz"},
                              UA_BROWSER)
            harvest(lad, "https://example.com/x", ["bm_sz=new-sz; Domain=example.com"])
            row = lad.clearance.get("example.com")
            # The unchanged browser-earned cookie survives; the changed name
            # takes the new value.
            assert row.cookies == {"_abck": "old-abck", "bm_sz": "new-sz"}
        finally:
            await lad.close()

    async def test_clearance_class_harvest_updates_the_row_ua(self, settings):
        # bm_sz is UA-bound: a harvest carrying it was earned by the request
        # whose UA we hold, so that UA owns the row now.
        lad = FakeLadder(settings, {})
        try:
            lad.clearance.put("example.com", {"cf_clearance": "cf"}, UA_BROWSER)
            harvest(lad, "https://example.com/x", ["bm_sz=sz; Domain=example.com"])
            row = lad.clearance.get("example.com")
            assert row.user_agent == UA_PERSONA
            assert row.cookies == {"cf_clearance": "cf", "bm_sz": "sz"}
        finally:
            await lad.close()

    async def test_trust_only_harvest_keeps_the_existing_pin(self, settings):
        # _pxvid/_pxhd/bm_sv are not UA-bound; a pure trust-name harvest must
        # leave the bug-72/73 pin (earned against cf_clearance) untouched.
        lad = FakeLadder(settings, {})
        try:
            lad.clearance.put("example.com", {"cf_clearance": "cf"}, UA_BROWSER)
            harvest(lad, "https://example.com/x", ["_pxvid=v; Domain=example.com"])
            row = lad.clearance.get("example.com")
            assert row.user_agent == UA_BROWSER
            assert row.cookies == {"cf_clearance": "cf", "_pxvid": "v"}
        finally:
            await lad.close()

    async def test_foreign_domain_attribute_rejected_at_collection(self, settings):
        # _HopCookies refuses a Domain the setter had no right to (RFC 6265
        # s5.3) -- the jar harvest inherits that for free.
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "https://example.com/x",
                    ["bm_sz=evil; Domain=other.com"])
            assert lad.clearance.get("example.com") is None
        finally:
            await lad.close()

    async def test_secure_cookie_not_harvested_over_http(self, settings):
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "http://example.com/x", ["bm_sv=s; Secure"])
            assert lad.clearance.get("example.com") is None
        finally:
            await lad.close()

    async def test_garbage_header_never_raises(self, settings):
        lad = FakeLadder(settings, {})
        try:
            harvest(lad, "https://example.com/x", ["no-equals-sign", "", "=novalue"])
            assert lad.clearance.get("example.com") is None
        finally:
            await lad.close()


class _FakeStreamResponse:
    """The slice of httpx.Response _tier0_document + _read_capped touch."""

    def __init__(self, url: str, status: int, body: bytes, set_cookie: list[str]):
        pairs = [("content-type", HTML)]
        for sc in set_cookie:
            pairs.append(("set-cookie", sc))
        self._headers = httpx.Headers(pairs)
        self.status_code = status
        self.url = url
        self.charset_encoding = None
        self._body = body

    @property
    def headers(self) -> httpx.Headers:
        return self._headers

    async def aiter_bytes(self, _n: int):
        yield self._body

    async def aclose(self) -> None:
        return None


class TestTier0HarvestWiring:
    """A real _tier0_document over a stubbed _open_stream banks as it passes."""

    async def test_tier0_200_response_banks_retained_names(self, settings):
        lad = FakeLadder(settings, {})

        async def fake_open_stream(url, headers, timeout):
            return _FakeStreamResponse(url, 200, GOOD.encode(), [
                "bm_sv=warm; Domain=example.com; Secure",
                "_ga=ignored; Domain=example.com",
            ])

        lad._open_stream = fake_open_stream
        try:
            out = await lad._tier0_document("https://example.com/x")
            assert out[0] == 200 and "Plenty of real" in out[2]
            row = lad.clearance.get("example.com")
            assert row is not None and row.cookies == {"bm_sv": "warm"}
        finally:
            await lad.close()

    async def test_tier0_challenge_response_banks_too(self, settings):
        # The 403 that starts a climb sets the vendor ids that gate it --
        # harvest on the refusal, not just the success.
        lad = FakeLadder(settings, {})

        async def fake_open_stream(url, headers, timeout):
            return _FakeStreamResponse(url, 403, b"<html>denied</html>", [
                "datadome=from-403; Domain=example.com",
            ])

        lad._open_stream = fake_open_stream
        try:
            out = await lad._tier0_document("https://example.com/x")
            assert out[0] == 403
            row = lad.clearance.get("example.com")
            assert row is not None and row.cookies == {"datadome": "from-403"}
        finally:
            await lad.close()

    async def test_tier0_harvest_never_disturbs_the_document(self, settings):
        lad = FakeLadder(settings, {})

        async def fake_open_stream(url, headers, timeout):
            return _FakeStreamResponse(url, 200, GOOD.encode(), [])

        lad._open_stream = fake_open_stream
        try:
            status, headers, text, ctype, final = await lad._tier0_document(
                "https://example.com/x")
            assert (status, text, ctype, final) == (200, GOOD, HTML, "https://example.com/x")
            assert lad.clearance.get("example.com") is None, "nothing retained, no row"
        finally:
            await lad.close()
