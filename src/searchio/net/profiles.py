"""What we have learned about each domain, persisted across runs.

The payoff is starting tier: once a site has proved it needs a browser, paying
two failed round trips to rediscover that on every single request is pure
waste, and the failures themselves are block signals we would rather not
generate. So the ladder records the cheapest tier that actually worked and
starts there next time.

The opposite direction matters too. Sites relax -- a WAF rule gets tuned, a
promotion ends, an IP reputation decays -- so a domain pinned at tier 2 must be
able to walk back down. :meth:`DomainStore.should_probe` reintroduces a cheap
attempt occasionally, which costs one fast failure and can save every
subsequent browser launch for that host.
"""

from __future__ import annotations

import logging
import random
import sqlite3

from ._sqlite import connect_locked
import time
from pathlib import Path

from ..models import DomainProfile

_SCHEMA = """
CREATE TABLE IF NOT EXISTS domains (
    domain     TEXT PRIMARY KEY,
    min_tier   INTEGER NOT NULL DEFAULT 0,
    successes  INTEGER NOT NULL DEFAULT 0,
    blocks     INTEGER NOT NULL DEFAULT 0,
    vendor     TEXT NOT NULL DEFAULT '',
    last_seen  REAL NOT NULL,
    rps        REAL NOT NULL DEFAULT 0.75
);
"""

# One in this many fetches to a tier-pinned domain retries the cheap path.
PROBE_ODDS = 12

log = logging.getLogger(__name__)


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default: float) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    return out if out == out else default  # NaN is not a value


class DomainStore:
    """Persistent per-domain profiles."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self._db: sqlite3.Connection | None = None
        self._mem: dict[str, DomainProfile] = {}
        #: sqlite failures degrade to the in-memory layer and are counted,
        #: never raised (bug 71): record_success/record_block run outside
        #: any try in the ladder, right after a successful fetch.
        self.errors = 0
        self.last_error = ""
        if enabled:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                db = connect_locked(str(path))  # serialized across threads (bug 154)
                db.executescript(_SCHEMA)
                db.execute("PRAGMA journal_mode=WAL")
                db.commit()
                self._db = db
            except (sqlite3.Error, OSError) as exc:
                self._fail(exc, "domain profiles are memory-only for this process")

    def _fail(self, exc: BaseException, what: str) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        if self.errors == 1:
            log.warning("domain profiles: %s (%s)", self.last_error, what)

    def get(self, domain: str) -> DomainProfile:
        if domain in self._mem:
            return self._mem[domain]
        prof = DomainProfile(domain=domain)
        if self._db:
            row = None
            try:
                row = self._db.execute(
                    "SELECT min_tier, successes, blocks, vendor, last_seen, rps "
                    "FROM domains WHERE domain = ?",
                    (domain,),
                ).fetchone()
            except sqlite3.Error as exc:
                self._fail(exc, "profile read served as fresh")
            if row:
                # Coerce and clamp (bug 153's profile side): a corrupted row
                # handed back min_tier 'abc' and start_tier's min() on it was
                # a TypeError in the fetch path; 99 came back as 99.
                prof = DomainProfile(
                    domain=domain,
                    min_tier=min(max(_as_int(row[0], 0), 0), 2),
                    successes=max(_as_int(row[1], 0), 0),
                    blocks=max(_as_int(row[2], 0), 0),
                    last_block_vendor=str(row[3] or ""),
                    last_seen=_as_float(row[4], 0.0),
                    rps=_as_float(row[5], 0.0),
                )
        self._mem[domain] = prof
        return prof

    def save(self, prof: DomainProfile) -> None:
        prof.last_seen = time.time()
        self._mem[prof.domain] = prof
        if not self._db:
            return
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO domains "
                "(domain, min_tier, successes, blocks, vendor, last_seen, rps) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    prof.domain,
                    prof.min_tier,
                    prof.successes,
                    prof.blocks,
                    prof.last_block_vendor,
                    prof.last_seen,
                    prof.rps,
                ),
            )
            self._db.commit()
        except sqlite3.Error as exc:
            self._fail(exc, "profile write dropped")
            try:
                self._db.rollback()
            except sqlite3.Error:
                pass

    def record_success(self, domain: str, tier: int) -> None:
        p = self.get(domain)
        p.successes += 1
        # A cheaper tier just worked, so the pin was too pessimistic.
        if tier < p.min_tier:
            p.min_tier = tier
        self.save(p)

    def record_block(self, domain: str, tier: int, vendor: str = "") -> None:
        p = self.get(domain)
        p.blocks += 1
        if vendor:
            p.last_block_vendor = vendor
        # This tier is not enough for this host; next time start above it.
        p.min_tier = max(p.min_tier, min(tier + 1, 2))
        self.save(p)

    def start_tier(self, domain: str, ceiling: int) -> int:
        """Which tier to begin at for this domain."""
        p = self.get(domain)
        tier = min(p.min_tier, ceiling)
        if tier > 0 and self.should_probe():
            return 0
        return tier

    def should_probe(self) -> bool:
        """Occasionally re-test whether a pinned domain has relaxed."""
        return random.randrange(PROBE_ODDS) == 0

    def all(self) -> list[DomainProfile]:
        if not self._db:
            return list(self._mem.values())
        try:
            rows = self._db.execute(
                "SELECT domain, min_tier, successes, blocks, vendor, last_seen, rps "
                "FROM domains ORDER BY successes + blocks DESC"
            ).fetchall()
        except sqlite3.Error as exc:
            self._fail(exc, "listing unavailable")
            return list(self._mem.values())
        return [
            DomainProfile(
                domain=r[0],
                min_tier=r[1],
                successes=r[2],
                blocks=r[3],
                last_block_vendor=r[4],
                last_seen=r[5],
                rps=r[6],
            )
            for r in rows
        ]

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
