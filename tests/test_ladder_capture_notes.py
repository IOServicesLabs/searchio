"""Clearance capture instrumentation: every outcome leaves evidence.

The 2026-09-18 realtor opener pass banked nothing while fourteen other
domains bank fine through the same ``_capture_clearance`` call, and the
old silent ``return``/``except: pass`` made the miss untraceable. Now the
call returns its outcome (banked / empty / error), bumps a per-outcome
counter, and appends a bounded note (counts + names + domains only, never
values) surfaced through ``Ladder.stats()["capture_notes"]``.

The empty path settles 6s and re-snapshots once (Akamai's sensor POST can
land after load-event), so a capture that fails on the first jar but
succeeds on the settled jar reports ``banked`` with a distinct counter --
the timing race turns into a banked row instead of a mystery. The
still-empty note discriminates what remains: ``first_party=0`` means the
jar had no cookies for the target domain at all (wrong-jar snapshot);
``retained_anywhere`` lists RETAINED names seen on any domain, so a
retained cookie scoped somewhere irrelevant shows up there without being
banked (domain-shape bug); neither means the sensor never set them.

These pins hold that contract on both the happy path and each failure
mode, and guard the values-never-logged rule (cookies are credentials).
"""

from __future__ import annotations

import asyncio

import pytest

from searchio.config import Settings
from searchio.net.ladder import Ladder

from tests.test_ladder_router import FakeLadder


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path,
        cache_enabled=False,
        robots_policy="off",
        max_tier=2,
        per_domain_rps=1000.0,
        per_domain_burst=1000,
        sidecar_autostart=False,
    )


@pytest.fixture(autouse=True)
def _no_settle_sleep(monkeypatch):
    """The 6s settle delay is a live-path courtesy, not a unit-test cost."""
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


class _JarClient:
    """Stands in for a SidecarClient: serves canned cookie jars/UAs.

    ``jars`` supplies a per-call sequence: call N gets jars[min(N,
    len-1)], so the last jar repeats. ``fail`` raises on every call.
    """

    def __init__(self, jar=None, ua="BrowserChrome/131.0", fail=None,
                 jars=None):
        self._jar = jar or []
        self._ua = ua
        self._fail = fail
        self._jars = list(jars) if jars is not None else None
        self._calls = 0

    async def cookies(self):
        if self._fail:
            raise self._fail
        if self._jars is not None:
            jar = self._jars[min(self._calls, len(self._jars) - 1)]
            self._calls += 1
            return jar
        return self._jar

    async def user_agent(self):
        return self._ua


def jar(*rows):
    return [dict(name=n, value=v, domain=d) for n, v, d in rows]


class TestCaptureOutcomes:
    async def test_banked_puts_row_and_reports_banked(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(jar(
            ("_abck", "akamai-token", ".realtor.com"),
            ("_ga", "tracker", ".realtor.com"),
        ))
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "banked"
        row = lad.clearance.get("realtor.com")
        assert row is not None and "_abck" in row.cookies
        assert lad._stats.get("clearance_captured") == 1
        assert lad._stats.get("clearance_capture_empty", 0) == 0
        # First jar was enough -- the settle retry never ran.
        assert lad._stats.get("clearance_capture_settle_retry", 0) == 0

    async def test_empty_first_jar_banks_after_settle(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(jars=[
            # t=0: sensor POST still in flight -- nothing retained yet.
            jar(("srchd", "pre", ".realtor.com")),
            # t=+6s: the sensor landed _abck between the snapshots.
            jar(("_abck", "akamai-token", ".realtor.com")),
        ])
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "banked"
        row = lad.clearance.get("realtor.com")
        assert row is not None and "_abck" in row.cookies
        assert lad._stats.get("clearance_capture_settle_retry") == 1
        assert lad._stats.get("clearance_capture_banked_after_settle") == 1
        assert lad._stats.get("clearance_capture_empty", 0) == 0
        assert any("settle" in n for n in lad.stats()["capture_notes"])

    async def test_empty_jar_bumps_counter_and_notes_evidence(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(jar(
            ("sessionid", "login-class", ".realtor.com"),
            ("_ga", "tracker", ".realtor.com"),
        ))
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "empty"
        assert lad._stats.get("clearance_capture_empty") == 1
        # The settle retry ran and still found nothing.
        assert lad._stats.get("clearance_capture_settle_retry") == 1
        (note,) = lad.stats()["capture_notes"]
        assert "realtor.com" in note
        # First-party cookies exist (domain matches) but no retained names.
        assert "first_party=2" in note
        assert "retained_anywhere=[]" in note
        # Login/session names are never banked and never logged with values.
        assert lad.clearance.get("realtor.com") is None
        assert "login-class" not in note and "tracker" not in note

    async def test_empty_wrong_jar_zero_first_party(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(jar(
            ("177_dsp_uid", "ad-junk", ".33across.com"),
            ("_ga", "tracker", ".2o7.net"),
        ))
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "empty"
        (note,) = lad.stats()["capture_notes"]
        assert "first_party=0" in note
        assert "retained_anywhere=[]" in note
        assert "retained_sites=[]" in note

    async def test_retained_on_wrong_domain_is_evidence_not_banked(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(jar(
            ("_abck", "akamai-token", ".some-ad.net"),
        ))
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "empty"
        assert lad.clearance.get("realtor.com") is None
        (note,) = lad.stats()["capture_notes"]
        # The retained name IS in the jar -- on a domain that can never
        # serve realtor.com. retained_sites pins the exact scoping.
        assert "retained_anywhere=['_abck']" in note
        assert "retained_sites=['_abck@.some-ad.net']" in note
        assert "first_party=0" in note
        assert "akamai-token" not in note  # values never logged

    async def test_error_is_counted_never_raised(self, settings):
        lad = FakeLadder(settings, {})
        client = _JarClient(fail=RuntimeError("jar rpc timeout"))
        outcome = await lad._capture_clearance("realtor.com", client)
        assert outcome == "error"
        assert lad._stats.get("clearance_capture_error") == 1
        (note,) = lad.stats()["capture_notes"]
        assert "RuntimeError" in note and "jar rpc timeout" in note
        assert lad.clearance.get("realtor.com") is None

    async def test_notes_ring_is_bounded(self, settings):
        lad = FakeLadder(settings, {})
        for i in range(12):
            await lad._capture_clearance(f"host{i}.example", _JarClient())
        notes = lad.stats()["capture_notes"]
        assert len(notes) == 8
        assert "host11.example" in notes[-1]
        assert "host0.example" not in notes
