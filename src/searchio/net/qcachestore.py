"""Fused search results across processes (persist_query_cache).

The in-memory query cache dies with its process, and a CLI search is a new
process every time: a repeat query paid the whole fan-out again -- or at best
one content-cache read, re-parse and robots check per provider. This store
keeps the fused RouteResult itself, so a repeat answers from one SQLite row.

Entries carry a WALL-clock expiry (the in-memory cache uses the monotonic
clock, which means nothing in the next process). Only answers are stored;
the router never remembers an empty fan-out.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from ._sqlite import connect_locked

#: Rows kept; the oldest go first.
MAX_ROWS = 2048


def _to_json(res) -> str:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(res):
        v = getattr(res, f.name)
        out[f.name] = [d.model_dump(mode="json") for d in v] if f.name == "docs" else v
    return json.dumps(out)


def _from_json(text: str):
    from ..models import Doc
    from ..router import RouteResult

    raw = json.loads(text)
    known = {f.name for f in dataclasses.fields(RouteResult)}
    kw = {k: v for k, v in raw.items() if k in known}
    kw["docs"] = [Doc.model_validate(d) for d in kw.get("docs") or []]
    return RouteResult(**kw)


class QueryCacheStore:
    def __init__(self, path: Path) -> None:
        self._db = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = connect_locked(str(path))
            db.execute("CREATE TABLE IF NOT EXISTS q (key TEXT PRIMARY KEY, expires REAL, body TEXT)")
            db.commit()
            self._db = db
        except Exception:  # noqa: BLE001 -- a lost store is a cold cache, never an error
            self._db = None

    @staticmethod
    def _key(key: tuple) -> str:
        return json.dumps(list(key), default=str)

    def get(self, key: tuple):
        """``(monotonic_expiry, RouteResult)`` -- possibly already expired, so
        the caller's stale-while-revalidate window applies -- or None."""
        if self._db is None:
            return None
        try:
            row = self._db.execute("SELECT expires, body FROM q WHERE key = ?",
                                   (self._key(key),)).fetchone()
            if row is None:
                return None
            return (time.monotonic() + (float(row[0]) - time.time()), _from_json(row[1]))
        except Exception:  # noqa: BLE001 -- a row that will not decode is a miss
            return None

    def put(self, key: tuple, entry: tuple) -> None:
        if self._db is None:
            return
        expires_mono, res = entry
        try:
            self._db.execute("INSERT OR REPLACE INTO q VALUES (?, ?, ?)",
                             (self._key(key), time.time() + (expires_mono - time.monotonic()),
                              _to_json(res)))
            self._db.execute(
                "DELETE FROM q WHERE key IN (SELECT key FROM q ORDER BY expires DESC "
                f"LIMIT -1 OFFSET {MAX_ROWS})")
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            except Exception:  # noqa: BLE001
                pass
            self._db = None
