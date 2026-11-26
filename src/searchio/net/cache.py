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

#: cache_compress: payloads at least this long are stored zstd-compressed,
#: as a BLOB with this magic in front (SQLite columns are dynamically typed;
#: a TEXT row from before the knob still reads back as JSON).
_ZMAGIC = b"\x00z1"
_ZMIN = 1024

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

    def __init__(self, path: Path, ttl_s: int = 3600, enabled: bool = True,
                 *, compress: bool = False, purge_on_open: bool = False,
                 max_mb: int = 0) -> None:
        self.ttl_s = ttl_s
        self.enabled = enabled
        self.path = path
        self.compress = compress
        #: > 0: past this many MiB of payload, the oldest rows are evicted.
        self.max_bytes = max(0, int(max_mb)) * 1024 * 1024
        self._puts = 0
        self._zc = self._zd = None
        if compress:
            try:
                import zstandard

                self._zc = zstandard.ZstdCompressor(level=3)
                self._zd = zstandard.ZstdDecompressor()
            except Exception:  # noqa: BLE001 -- no codec = plain rows
                self.compress = False
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
            if self._db is not None and purge_on_open:
                # purge_expired had no caller: the file only ever grew.
                self.purge_expired()
                self._evict()

    def _fail(self, exc: BaseException, what: str) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        if self.errors == 1:
            log.warning("cache %s: %s (%s)", self.path, self.last_error, what)

    def get(self, url: str, variant: str = "", stale_for: float = 0.0) -> Any | None:
        """The cached value, or None. ``stale_for`` > 0 also accepts an entry
        up to that many seconds past its TTL (stale-while-revalidate)."""
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
        if time.time() - created > self.ttl_s + max(stale_for, 0.0):
            return None
        try:
            return json.loads(self._decode(payload))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _decode(self, payload: Any) -> str:
        """A row's JSON text: plain TEXT, or a zstd BLOB behind _ZMAGIC.
        Read regardless of the knob, so turning it off strands nothing."""
        if isinstance(payload, (bytes, memoryview)):
            raw = bytes(payload)
            if raw.startswith(_ZMAGIC):
                zd = self._zd
                if zd is None:
                    import zstandard

                    zd = self._zd = zstandard.ZstdDecompressor()
                try:
                    return zd.decompress(raw[len(_ZMAGIC):]).decode("utf-8")
                except Exception as exc:  # noqa: BLE001 -- a rotten row is a miss
                    raise ValueError(str(exc)) from exc
            return raw.decode("utf-8", errors="replace")
        return payload

    def _encode(self, text: str) -> Any:
        if self.compress and self._zc is not None and len(text) >= _ZMIN:
            return _ZMAGIC + self._zc.compress(text.encode("utf-8"))
        return text

    def _evict(self) -> int:
        """Drop oldest rows until the payload total is under the cap."""
        if not self._db or not self.max_bytes:
            return 0
        try:
            (total,) = self._db.execute(
                "SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM entries").fetchone()
            if total <= self.max_bytes:
                return 0
            # Evict to 90 % so one put past the line is not an evict per put.
            target = int(self.max_bytes * 0.9)
            dropped = 0
            rows = self._db.execute(
                "SELECT key, LENGTH(payload) FROM entries ORDER BY created ASC").fetchall()
            for key, size in rows:
                if total <= target:
                    break
                self._db.execute("DELETE FROM entries WHERE key = ?", (key,))
                total -= int(size or 0)
                dropped += 1
            self._db.commit()
            return dropped
        except sqlite3.Error as exc:
            self._fail(exc, "evict skipped")
            return 0

    def put(self, url: str, value: Any, variant: str = "", tier: int = 0) -> None:
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO entries (key, url, payload, tier, created) "
                "VALUES (?,?,?,?,?)",
                (_key(url, variant), url, self._encode(json.dumps(value, default=str)),
                 tier, time.time()),
            )
            self._db.commit()
            self._puts += 1
            if self.max_bytes and self._puts % 64 == 0:
                self._evict()
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
