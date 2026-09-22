"""Persona currency and fingerprint/UA agreement (iteration 64). Red first."""

from __future__ import annotations

import re

import pytest

from searchio.net import persona as persona_mod

_MAJOR = re.compile(r"Chrome/(\d+)\.")


def _supported() -> set[str]:
    from curl_cffi.requests.impersonate import BrowserType

    return {b.value for b in BrowserType}


class TestPersonasAreCurrentAndSelfConsistent:
    # Bug 141: every Chromium persona claimed Chrome 131 (November 2024) --
    # in 2026 a two-year-old browser is itself a signal, and curl_cffi ships
    # fingerprints up to Chrome 150. The UA major, the client-hint major and
    # the impersonation target must agree, and the target must be one this
    # curl_cffi actually implements.
    @pytest.mark.parametrize("p", [p for p in persona_mod.PERSONAS if p.sec_ch_ua])
    def test_chromium_persona_agrees_with_itself(self, p):
        ua_major = int(_MAJOR.search(p.ua).group(1))
        assert f'v="{ua_major}"' in p.sec_ch_ua
        assert p.impersonate == f"chrome{ua_major}", (p.name, p.impersonate)
        assert p.impersonate in _supported()

    def test_chromium_personas_are_not_years_old(self):
        for p in persona_mod.PERSONAS:
            if p.sec_ch_ua:
                assert int(_MAJOR.search(p.ua).group(1)) >= 142, p.name


class TestImpersonationFollowsThePinnedUA:
    # A banked clearance pins the BROWSER's UA (say Chrome 145) onto the
    # tier-1 request; the TLS fingerprint must follow it to the nearest
    # supported target instead of staying on the persona's.
    @pytest.mark.parametrize("ua,want", [
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36", "chrome145"),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36", "chrome142"),
        ("Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0", "chrome131"),
        ("Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/999.0.0.0 Safari/537.36", "chrome150"),
    ])
    def test_nearest_supported_chrome(self, ua, want):
        assert persona_mod.impersonate_for_ua(ua, "chrome131") == want

    @pytest.mark.parametrize("ua", ["Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 Version/17.0 Safari/605.1.15",
                                    "Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/130.0", "", "garbage"])
    def test_non_chromium_keeps_the_default(self, ua):
        assert persona_mod.impersonate_for_ua(ua, "safari17_0") == "safari17_0"
