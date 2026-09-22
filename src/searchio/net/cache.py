"""Content cache backed by SQLite.

Every cache hit is a request some site does not receive. That makes this the
cheapest anti-blocking measure available -- cheaper than any fingerprint work,
because traffic that never leaves the process cannot be profiled, rate-limited,
or challenged. During development especially, where the same query gets run
fifty times, it is the difference between a warm relationship with a host and a
cooled-off IP.

SQLite rather than a dict because the useful lifetime spans processes: a CLI
invocation, the server, and a test run should all share one body of
already-paid-for fetches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

from ._sqlite import connect_locked
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key      TEXT PRIMARY KEY,
    url      TEXT NOT NULL,
    payload  TEXT NOT NULL,
    tier     INTEGER NOT NULL DEFAULT 0,
    created  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_created ON entries(created);
"""


def _key(url: str, variant: str = "") -> str:
    return hashlib.blake2b(f"{url}|{variant}".encode(), digest_size=16).hexdigest()


class Cache:
    """TTL cache for fetch results and provider responses.

    One connection per instance; the async call sites are single-threaded.
    check_same_thread is disabled because asyncio may hand blocking work to a
    worker thread, not to invite concurrent writers.
    """

    def __init__(self, path: Path, ttl_s: int = 3600, enabled: bool = True) -> None:
        self.ttl_s = ttl_s
        self.enabled = enabled
        self.path = path
        self._db: sqlite3.Connection | None = None
        #: Failures are counted, not raised: a cache is an optimization, and
        #: every sqlite error is a miss (bug 67 -- a corrupt file used to
        #: raise out of Ladder construction; "database is locked" from a
        #: concurrent CLI run raised out of put() AFTER a successful fetch).
        self.errors = 0
        self.last_error = ""
        if enabled:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                db = connect_locked(str(path))  # serialized across threads (bug 154)
                db.executescript(_SCHEMA)
                # WAL keeps a reader (the server) from blocking a writer (a
                # CLI run) against the same file.
                db.execute("PRAGMA journal_mode=WAL")
                db.commit()
                self._db = db
            except (sqlite3.Error, OSError) as exc:
                self._fail(exc, "cache disabled for this process")

    def _fail(self, exc: BaseException, what: str) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        if self.errors == 1:
            log.warning("cache %s: %s (%s)", self.path, self.last_error, what)

    def get(self, url: str, variant: str = "") -> Any | None:
        if not self._db:
            return None
        try:
            row = self._db.execute(
                "SELECT payload, created FROM entries WHERE key = ?", (_key(url, variant),)
            ).fetchone()
        except sqlite3.Error as exc:
            self._fail(exc, "read served as a miss")
            return None
        if not row:
            return None
        payload, created = row
        if time.time() - created > self.ttl_s:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    def put(self, url: str, value: Any, variant: str = "", tier: int = 0) -> None:
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO entries (key, url, payload, tier, created) "
                "VALUES (?,?,?,?,?)",
                (_key(url, variant), url, json.dumps(value, default=str), tier, time.time()),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            self._fail(exc, "write dropped")
            try:
                self._db.rollback()
            except sqlite3.Error:
                pass

    def purge_expired(self) -> int:
        if not self._db:
            return 0
        try:
            cur = self._db.execute(
                "DELETE FROM entries WHERE created < ?", (time.time() - self.ttl_s,)
            )
            self._db.commit()
            return cur.rowcount
        except sqlite3.Error as exc:
            self._fail(exc, "purge skipped")
            return 0

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"entries": 0, "errors": self.errors}
        if self.last_error:
            out["error"] = self.last_error
        if not self._db:
            return out
        try:
            (n,) = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()
            out["entries"] = n
        except sqlite3.Error as exc:
            self._fail(exc, "stats unavailable")
            out["errors"] = self.errors
            out["error"] = self.last_error
        return out

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
