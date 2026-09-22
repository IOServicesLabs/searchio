"""An invalid certificate is a decided fetch, not a transient (iteration
70, from the false-green review of the suite: the tls control rows could
not demand a certificate reason because the product lost it). Red first."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.errors import TransientError
from searchio.net.ladder import Ladder, _is_cert_failure

CERT_MSGS = [
    "tier0 ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1006)",
    "tier1 Failed to perform, curl: (60) SSL certificate problem: self signed certificate",
    "tier2: fetch: request failed: error sending request: invalid peer certificate: UnknownIssuer",
    "tier1 curl: (60) SSL: no alternative certificate subject name matches target host name",
]
NOT_CERT = [
    "tier0 ConnectError: All connection attempts failed",
    "tier1 Timeout: Failed to perform, curl: (28) Operation timed out",
    "tier0 ConnectError: [SSL: TLSV1_ALERT_PROTOCOL_VERSION] tlsv1 alert protocol version",
    "tier2: fetch: request failed: error sending request",
]


class TestCertFailureIsRecognised:
    @pytest.mark.parametrize("msg", CERT_MSGS)
    def test_cert_messages(self, msg):
        assert _is_cert_failure(msg)

    @pytest.mark.parametrize("msg", NOT_CERT)
    def test_other_transients_are_not_cert_failures(self, msg):
        assert not _is_cert_failure(msg)


class TestCertFailureDoesNotClimb:
    async def test_no_retry_no_climb_and_the_reason_survives(self, tmp_path):
        # Bug 157: a self-signed certificate at tier 0 was retried, climbed
        # to tier 1 and then to the engine, and the FINAL error was the
        # engine's "request failed: error sending request" -- the reason
        # gone, and a browser tier booted for a fault no browser can fix.
        s = Settings(state_dir=tmp_path, cache_enabled=False, max_tier=2, robots_policy="off",
                     sidecar_autostart=False)
        lad = Ladder(s)
        attempts: list[int] = []

        async def try_tier(tier, url, **kw):
            attempts.append(tier)
            raise TransientError(CERT_MSGS[0])
        lad._try_tier = try_tier
        try:
            with pytest.raises(TransientError) as ei:
                await lad.fetch("https://self-signed.example/x", use_cache=False)
            assert str(ei.value).startswith("tls_invalid_cert:"), str(ei.value)
            assert "certificate" in str(ei.value).lower()
            assert attempts == [0], attempts
        finally:
            await lad.close()
