"""Reusing the clearance a browser earns.

When tier 2 solves a challenge, the site hands back a cookie -- ``cf_clearance``
for Cloudflare, ``datadome`` for DataDome -- that says "this visitor already
passed". That cookie is the expensive part of the whole exchange: g2.com costs
~12 s and a Chromium launch to obtain, and roughly 600 ms to *use*.

Throwing it away after one request, which is what searchio did until now, means
paying the 12 s again for the very next page on that host. Keeping it turns the
browser into something you visit once per domain per session rather than once
per URL, which is the difference between "we can get through" and "we can get
through at scale".

Two constraints make this narrower than it first looks:

* **The cookie is bound to the User-Agent.** Cloudflare issues ``cf_clearance``
  against the UA that solved the challenge and rejects it under any other. So
  the browser's own UA is captured alongside the cookies and pinned on every
  request that reuses them -- which also means the persona machinery has to
  step aside for those hosts, and a mismatch here fails *worse* than not
  reusing at all.
* **It is bound to the egress IP.** Fine while everything runs from one
  machine; it is why this store is not shareable across a proxy pool without
  keying on the exit address too.
"""

from __future__ import annotations

import json
import logging
import sqlite3

from ._sqlite import connect_locked
import time
from dataclasses import dataclass
from pathlib import Path

#: Cookies that actually represent a passed challenge. Storing the whole jar
#: would work but carries session and tracking cookies we have no business
#: replaying, so the store keeps only what earns its place.
CLEARANCE_COOKIES: frozenset[str] = frozenset(
    {
        "cf_clearance",  # Cloudflare managed challenge
        "__cf_bm",  # Cloudflare bot management
        "datadome",  # DataDome
        "_px3",  # PerimeterX
        "px-cts",
        "_abck",  # Akamai
        "bm_sz",
        "ak_bmsc",
        "incap_ses",  # Imperva/Incapsula
        "visid_incap",
    }
)

#: Continuity cookies harvested from ANY tier's ordinary responses (the
#: trust-cookie jar), as opposed to CLEARANCE_COOKIES, which are earned by a
#: challenge solve and banked from a browser jar. These three are the
#: vendor-trust class: Akamai's ``bm_sv`` accompanies ``bm_sz`` on ordinary
#: (non-challenge) responses, and PerimeterX's ``_pxvid``/``_pxhd`` identify
#: the visitor across requests -- exactly the "browser that has been here
#: before" signal an adaptive gate rewards (probe_realtor_patchright.py,
#: engine ec52690: the browser's jar grew these on the way to content).
#: Ad/analytics names (_ga, _gcl_au, NID, ...) are deliberately excluded:
#: they carry no anti-bot trust, and replaying trackers makes the client
#: MORE distinctive, not less.
TRUST_COOKIES: frozenset[str] = frozenset(
    {
        "bm_sv",  # Akamai sensor-validation companion to bm_sz
        "_pxvid",  # PerimeterX visitor id
        "_pxhd",  # PerimeterX header-print id (long-lived)
    }
)

#: Every name the jar retains, for one-membership test at harvest.
RETAINED_COOKIES: frozenset[str] = CLEARANCE_COOKIES | TRUST_COOKIES

#: Clearance is short-lived by design. cf_clearance is typically ~30 minutes;
#: replaying an expired one is a wasted round trip, not a failure, so this is a
#: conservative default rather than a guess at each vendor's policy.
DEFAULT_TTL_S = 1500

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clearance (
    domain     TEXT PRIMARY KEY,
    cookies    TEXT NOT NULL,
    user_agent TEXT NOT NULL DEFAULT '',
    captured   REAL NOT NULL,
    uses       INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class Clearance:
    domain: str
    cookies: dict[str, str]
    user_agent: str = ""
    captured: float = 0.0
    uses: int = 0

    def header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def fresh(self, ttl_s: int = DEFAULT_TTL_S) -> bool:
        return bool(self.cookies) and (time.time() - self.captured) < ttl_s


def relevant_cookies(jar: list[dict], domain: str) -> dict[str, str]:
    """Pick out the retained cookies (clearance + trust) from a cookie jar.

    Used for the browser-jar capture after a tier-2 pass: the pass earned
    the full retained set -- challenge tokens AND the continuity names
    (_pxvid/bm_sv/...) a cheap tier harvest would also keep -- so both bank,
    under one retention policy. Session/login names stay excluded by design
    (test_every_known_cookie_is_a_challenge_cookie is the guard).
    """
    out: dict[str, str] = {}
    want = domain.removeprefix("www.")
    for c in jar or []:
        name = str(c.get("name") or "")
        if name not in RETAINED_COOKIES:
            continue
        cdom = str(c.get("domain") or "").lstrip(".").removeprefix("www.")
        # A cookie set on a parent domain is valid for the child; a cookie
        # set on a CHILD is not valid for the parent (bug 155: a __cf_bm
        # scoped to sub.example.com was replayed on example.com -- a browser
        # never sends it there). A cookie with no domain belongs nowhere.
        if not cdom or not (want == cdom or want.endswith("." + cdom)):
            continue
        out[name] = str(c.get("value") or "")
    return out


class ClearanceStore:
    """Per-domain clearance cookies, persisted across runs.

    On disk rather than in memory because the cookie outlives the process and
    re-earning it costs a browser launch. A stale row is harmless -- it is
    checked for freshness before use and refreshed on the next tier-2 success.
    """

    def __init__(self, path: Path, enabled: bool = True, ttl_s: int = DEFAULT_TTL_S) -> None:
        self.enabled = enabled
        self.ttl_s = ttl_s
        self._db: sqlite3.Connection | None = None
        self._mem: dict[str, Clearance] = {}
        #: sqlite failures degrade to the in-memory layer and are counted,
        #: never raised (bug 71 -- the bug-67 class): a corrupt file used to
        #: raise out of Ladder construction, a locked db out of put() after
        #: a successful fetch.
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
                self._fail(exc, "clearance store is memory-only for this process")

    def _fail(self, exc: BaseException, what: str) -> None:
        self.errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"[:200]
        if self.errors == 1:
            log.warning("clearance store: %s (%s)", self.last_error, what)

    def _exec(self, sql: str, params: tuple, what: str):
        """Run one statement + commit; a failure is counted and yields None."""
        if self._db is None:
            return None
        try:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur
        except sqlite3.Error as exc:
            self._fail(exc, what)
            try:
                self._db.rollback()
            except sqlite3.Error:
                pass
            return None

    @staticmethod
    def _from_row(domain: str, row) -> Clearance | None:
        try:
            cookies = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(cookies, dict):
            return None  # a hand-edited row must not become a header() crash
        try:
            captured = float(row[2] or 0)
            uses = int(row[3] or 0)
        except (TypeError, ValueError):
            # The never-raise guard covered sqlite errors but not the value
            # coercion (bug 153): a corrupted `captured` made get() raise
            # ValueError into the fetch path. A row that cannot be read is
            # no clearance.
            return None
        return Clearance(domain, {str(k): str(v) for k, v in cookies.items()},
                         str(row[1] or ""), captured, uses)

    def get(self, domain: str) -> Clearance | None:
        """Return usable clearance for this domain, or None."""
        if not self.enabled:
            return None
        c = self._mem.get(domain)
        if c is None and self._db is not None:
            row = None
            try:
                row = self._db.execute(
                    "SELECT cookies, user_agent, captured, uses FROM clearance WHERE domain = ?",
                    (domain,),
                ).fetchone()
            except sqlite3.Error as exc:
                self._fail(exc, "read served as a miss")
            if row:
                c = self._from_row(domain, row)
            if c:
                self._mem[domain] = c
        if c and c.fresh(self.ttl_s):
            return c
        return None

    def put(self, domain: str, cookies: dict[str, str], user_agent: str = "") -> None:
        if not self.enabled or not cookies:
            return
        c = Clearance(domain, cookies, user_agent, time.time())
        self._mem[domain] = c
        self._exec(
            "INSERT OR REPLACE INTO clearance "
            "(domain, cookies, user_agent, captured, uses) VALUES (?,?,?,?,0)",
            (domain, json.dumps(cookies), user_agent, c.captured), "write dropped")

    def record_use(self, domain: str) -> None:
        c = self._mem.get(domain)
        if c:
            c.uses += 1
        self._exec("UPDATE clearance SET uses = uses + 1 WHERE domain = ?", (domain,),
                   "use count not persisted")

    def drop(self, domain: str, *, older_than: float | None = None) -> None:
        """Forget clearance that did not work -- a replayed cookie that gets a
        403 is worse than none, because it wastes a tier and looks like a
        replay attack.

        ``older_than`` scopes the drop to the entry that could actually have
        been replayed (bug 143): a block can only invalidate a clearance that
        existed when its attempt began. One captured since -- by another
        caller's browser rescue, while this request was in flight -- is not
        the one that failed and stays.
        """
        if older_than is not None:
            cur = self._mem.get(domain) or self.get(domain)
            if cur is not None and cur.captured >= older_than:
                return
        self._mem.pop(domain, None)
        if self._exec("DELETE FROM clearance WHERE domain = ?", (domain,),
                      "drop not persisted") is None and self._db is not None:
            # The row survived on disk; make sure the next get() cannot
            # resurrect the cookie that just failed.
            self._mem[domain] = Clearance(domain, {}, "", 0.0)

    def all(self) -> list[Clearance]:
        if self._db is None:
            return list(self._mem.values())
        try:
            rows = self._db.execute(
                "SELECT domain, cookies, user_agent, captured, uses FROM clearance "
                "ORDER BY captured DESC"
            ).fetchall()
        except sqlite3.Error as exc:
            self._fail(exc, "listing unavailable")
            return list(self._mem.values())
        out = []
        for r in rows:
            c = self._from_row(r[0], r[1:])
            if c:
                out.append(c)
        return out

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
