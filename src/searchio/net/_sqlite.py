"""One sqlite connection, safely shared across threads.

The three small stores (cache, clearance, domain profiles) each keep a single
connection opened with ``check_same_thread=False`` so the tier-1 worker
threads can read clearance while the event-loop thread writes. sqlite3 does
not serialize concurrent use of one connection object: four threads on one
store produced "cannot commit - no transaction is active" and dropped writes
(bug 154). This wrapper serializes every statement and fully materializes
the rows inside the lock, so the callers' ``fetchone()``/``fetchall()`` never
touch a cursor another thread could disturb.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any


class _Rows:
    __slots__ = ("_rows", "rowcount")

    def __init__(self, rows: list, rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list:
        return list(self._rows)


class LockedConnection:
    """A minimal serialized facade over ``sqlite3.Connection``."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._lock = threading.RLock()

    def execute(self, sql: str, params: Any = ()) -> _Rows:
        with self._lock:
            cur = self._db.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            return _Rows(rows, cur.rowcount)

    def executescript(self, script: str) -> None:
        with self._lock:
            self._db.executescript(script)

    def commit(self) -> None:
        with self._lock:
            self._db.commit()

    def rollback(self) -> None:
        with self._lock:
            self._db.rollback()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @property
    def in_transaction(self) -> bool:
        return self._db.in_transaction


def connect_locked(path: str) -> LockedConnection:
    return LockedConnection(sqlite3.connect(path, check_same_thread=False))
