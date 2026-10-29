"""Provider health persisted across processes (``persist_health``).

A breaker that lives in memory is reset by every CLI invocation: a run that
starts thirty seconds after DuckDuckGo walled this egress re-learns the wall
with a multi-second refusal (and a fresh block signal), and the latency EMAs
that rank providers start from the same 500 ms guess every time. The rows
here carry the breaker state and the EMAs over; the in-process
:class:`~searchio.providers.base.Health` stays the source of truth while a
process runs.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from ._sqlite import connect_locked

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS health (
    provider       TEXT PRIMARY KEY,
    calls          INTEGER NOT NULL,
    failures       INTEGER NOT NULL,
    consecutive    INTEGER NOT NULL,
    ema_latency_ms REAL NOT NULL,
    ema_success    REAL NOT NULL,
    opened_wall    REAL NOT NULL,  -- breaker trip time as wall clock, 0 = closed
    last_error     TEXT NOT NULL,
    saved          REAL NOT NULL
);
"""

#: Rows older than this are history, not health: a provider that broke a
#: day ago is not presumed broken now.
MAX_AGE_S = 6 * 3600


class HealthStore:
    def __init__(self, path: Path) -> None:
        self._db = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = connect_locked(str(path))
            db.executescript(_SCHEMA)
            db.execute("PRAGMA journal_mode=WAL")
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as exc:
            log.warning("health store %s: %s (memory-only)", path, exc)

    def load(self, providers) -> None:
        """Seed each provider's Health from its saved row, if recent."""
        if self._db is None:
            return
        try:
            rows = self._db.execute(
                "SELECT provider, calls, failures, consecutive, ema_latency_ms, ema_success, "
                "opened_wall, last_error, saved FROM health").fetchall()
        except sqlite3.Error:
            return
        by_name = {p.name: p for p in providers}
        now_wall, now_mono = time.time(), time.monotonic()
        for name, calls, fails, consec, lat, succ, opened, err, saved in rows:
            p = by_name.get(name)
            if p is None or now_wall - float(saved) > MAX_AGE_S:
                continue
            h = p.health
            try:
                h.calls, h.failures = int(calls), int(fails)
                h.consecutive_failures = int(consec)
                h.ema_latency_ms, h.ema_success = float(lat), float(succ)
                h.last_error = str(err or "")
                # The trip time moves from wall clock back onto this
                # process's monotonic clock, so the cooldown keeps running
                # from when it really started.
                h.opened_at = (now_mono - (now_wall - float(opened))) if float(opened) else 0.0
            except (TypeError, ValueError):
                continue

    def save(self, p) -> None:
        if self._db is None:
            return
        h = p.health
        opened_wall = (time.time() - (time.monotonic() - h.opened_at)) if h.opened_at else 0.0
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO health VALUES (?,?,?,?,?,?,?,?,?)",
                (p.name, h.calls, h.failures, h.consecutive_failures, h.ema_latency_ms,
                 h.ema_success, opened_wall, h.last_error, time.time()))
            self._db.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


class HoldStore:
    """``persist_holds``: provider holds a host stated (a quota's reset)
    carried across processes. OpenAlex's spent daily budget answers 429 with
    a reset hours away; in memory only, every new CLI process re-asks and
    re-learns it, spending a fan-out slot on a guaranteed refusal."""

    def __init__(self, path: Path) -> None:
        self._db = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = connect_locked(str(path))
            db.execute("CREATE TABLE IF NOT EXISTS holds (provider TEXT PRIMARY KEY, until REAL)")
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as exc:
            log.warning("hold store %s: %s (memory-only)", path, exc)

    def load(self, providers) -> None:
        """Re-open each held provider's breaker for what is left of its hold."""
        if self._db is None:
            return
        try:
            rows = dict(self._db.execute("SELECT provider, until FROM holds").fetchall())
        except sqlite3.Error:
            return
        now = time.time()
        for p in providers:
            left = rows.get(p.name, 0) - now
            if left >= 1:
                p.health.hold(left)

    def put(self, provider: str, seconds: float) -> None:
        if self._db is None:
            return
        try:
            self._db.execute("INSERT OR REPLACE INTO holds VALUES (?, ?)",
                             (provider, time.time() + seconds))
            self._db.commit()
        except sqlite3.Error:
            pass

    def clear(self, provider: str) -> None:
        if self._db is None:
            return
        try:
            self._db.execute("DELETE FROM holds WHERE provider = ?", (provider,))
            self._db.commit()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
            self._db = None
