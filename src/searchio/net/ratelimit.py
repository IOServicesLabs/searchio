"""Per-domain adaptive pacing.

Politeness and evasion point the same direction here, which is the useful thing
about this module. A request pattern no human could produce -- twenty parallel
hits on one host, perfectly uniform inter-arrival times -- is exactly the signal
behavioural anti-bot systems key on. So the pacing that keeps us a good citizen
is also the pacing that keeps us unblocked, and there is no tradeoff to make.

The controller is AIMD, borrowed from TCP congestion control for the same
reason TCP uses it: we cannot see the server limit, only whether we just
crossed it. So probe upward slowly, and on evidence of refusal back off hard.

* success  -> rate += ADDITIVE (a linear probe, capped)
* blocked  -> rate *= BACKOFF  (halve, floored)

The asymmetry is the point. Recovering costs many successes; crossing the line
costs one 429.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import random
import time
from dataclasses import dataclass, field

ADDITIVE = 0.05  # rps added per successful request
BACKOFF = 0.5  # multiplier applied on a block
MAX_BUCKETS = 4096
FLOOR = 0.1  # never slower than one request per 10s
JITTER = 0.25  # +/- fraction applied to every wait


@dataclass
class _Bucket:
    """A token bucket whose refill rate the controller moves at runtime."""

    rate: float
    burst: int
    tokens: float = 0.0
    #: Per-host rate ceiling (configure); 0 = the limiter's max_rps.
    ceiling: float = 0.0
    updated: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(float(self.burst), self.tokens + (now - self.updated) * self.rate)
        self.updated = now


class DomainLimiter:
    """Token buckets keyed by domain, plus a global concurrency ceiling.

    The global semaphore matters independently of the per-domain rate: fanning
    out to forty domains at one request per second each is still forty
    concurrent sockets, and that is where a laptop file-descriptor budget and a
    home connection NAT table start to complain.
    """

    def __init__(
        self,
        *,
        rps: float = 0.75,
        burst: int = 3,
        max_rps: float = 4.0,
        concurrency: int = 24,
    ) -> None:
        # Clamp (bug 68): a burst of 0, or a rate of 0 / negative, made
        # acquire() loop forever -- tokens could never reach 1.0 -- so every
        # fetch hung silently. Settings does not validate these; the limiter
        # is the last line.
        self.default_rps = max(float(rps), FLOOR)
        self.burst = max(int(burst), 1)
        self.max_rps = max(float(max_rps), self.default_rps)
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        # A concurrency of 0 was a Semaphore(0): every fetch waited forever
        # (iteration 68 rider, the same clamp class as above).
        self._sem = asyncio.Semaphore(max(int(concurrency), 1))

    def _bucket(self, domain: str) -> _Bucket:
        b = self._buckets.get(domain)
        if b is None:
            b = _Bucket(rate=self.default_rps, burst=self.burst, tokens=float(self.burst))
            self._buckets[domain] = b
            # One bucket per host forever grew without bound on a long-lived
            # server (iteration 68 rider); the least recently used go first.
            while len(self._buckets) > MAX_BUCKETS:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(domain)
        return b

    def rate_for(self, domain: str) -> float:
        return self._bucket(domain).rate

    async def acquire(self, domain: str, max_wait: float | None = None) -> None:
        """Block until this domain may be hit again.

        ``max_wait`` (seconds): raise :class:`Throttled` instead of sleeping
        longer than that. A search provider uses it -- after a few blocks the
        host's rate sits near FLOOR, and a 10 s sleep inside a 12 s provider
        timeout is a search spent waiting to be refused.
        """
        b = self._bucket(domain)
        async with b.lock:
            while True:
                b.refill()
                if b.tokens >= 1.0:
                    b.tokens -= 1.0
                    return
                deficit = 1.0 - b.tokens
                wait = deficit / max(b.rate, FLOOR)
                if max_wait is not None and wait > max_wait:
                    from ..errors import Throttled

                    raise Throttled(
                        f"local_throttle: {domain} at {b.rate:.2f} rps would wait "
                        f"{wait:.1f}s > {max_wait:.1f}s")
                # Jitter breaks up the metronomic timing that makes automated
                # traffic legible even when every individual request is clean.
                wait *= 1.0 + random.uniform(-JITTER, JITTER)
                await asyncio.sleep(max(wait, 0.01))

    def record_success(self, domain: str) -> None:
        b = self._bucket(domain)
        b.rate = min(b.ceiling or self.max_rps, b.rate + ADDITIVE)

    def record_block(self, domain: str) -> None:
        b = self._bucket(domain)
        b.rate = max(FLOOR, b.rate * BACKOFF)
        # Surrender the burst allowance too. Backing off the refill rate while
        # three tokens are still in hand means the next three requests go out
        # immediately at full speed, which is precisely what just failed.
        b.tokens = 0.0
        b.updated = time.monotonic()

    def snapshot(self, domain: str) -> tuple[float, float]:
        """(rate, tokens) as of now -- what a record_block would take."""
        b = self._bucket(domain)
        b.refill()
        return b.rate, b.tokens

    def refund(self, domain: str, snap: tuple[float, float]) -> None:
        """Undo a record_block whose wall a better client then passed in the
        same climb (wall_refund): the refusal was about the client, not the
        pace. The rate comes back and so does the burst it surrendered, less
        one token for the extra request the climb made."""
        b = self._bucket(domain)
        b.refill()
        b.rate = max(b.rate, snap[0])
        b.tokens = min(float(b.burst), b.tokens + snap[1] - 1.0)

    def configure(self, domain: str, *, rps: float, burst: int) -> None:
        """Give ``domain`` its own documented rate and burst (api_rate_table).
        AIMD still moves it -- probing up from ``rps`` to ``max(rps,
        max_rps)``, halving on a block -- but a per-host ceiling below the
        global one (arXiv's 1-per-3 s) is never probed past."""
        b = self._bucket(domain)
        b.rate = max(FLOOR / 10, float(rps))
        b.burst = max(int(burst), 1)
        b.tokens = float(b.burst)
        b.ceiling = float(rps) if rps < self.default_rps else max(float(rps), self.max_rps)
        b.updated = time.monotonic()

    def learned(self) -> dict[str, float]:
        """Hosts whose learned rate sits below the default (persist_rates)."""
        return {d: b.rate for d, b in self._buckets.items() if b.rate < self.default_rps}

    def seed(self, domain: str, rate: float) -> None:
        """Start ``domain`` at a rate learned by an earlier process, with no
        burst in hand -- the same state record_block would have left."""
        if rate >= self.default_rps:
            return
        b = self._bucket(domain)
        b.rate = max(FLOOR, float(rate))
        b.tokens = 0.0
        b.updated = time.monotonic()

    def slot(self) -> asyncio.Semaphore:
        """Global concurrency guard, used as an async context manager."""
        return self._sem


#: persist_rates: a stored rate recovers this many rps per second of wall
#: clock (one ADDITIVE step per 10 s idle), so a host that walled us an hour
#: ago is back at the default by the time it matters.
RECOVER_RPS_PER_S = ADDITIVE / 10.0


class RateStore:
    """Learned per-domain rates across processes (persist_rates).

    A fresh process otherwise meets a host that walled the last one with a
    full burst at the default rate, and pays a wall attempt (2-3 s for a
    search engine) to relearn what the previous process already knew.
    """

    def __init__(self, path) -> None:
        from ._sqlite import connect_locked

        self._db = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = connect_locked(str(path))
            db.execute("CREATE TABLE IF NOT EXISTS rates (domain TEXT PRIMARY KEY, rate REAL, at REAL)")
            db.commit()
            self._db = db
        except Exception:  # noqa: BLE001 -- a lost rate store costs one relearn, never a fetch
            self._db = None

    def load_into(self, limiter: DomainLimiter) -> int:
        if self._db is None:
            return 0
        now = time.time()
        n = 0
        try:
            rows = self._db.execute("SELECT domain, rate, at FROM rates").fetchall()
        except Exception:  # noqa: BLE001
            return 0
        for domain, rate, at in rows:
            rate = float(rate) + max(0.0, now - float(at)) * RECOVER_RPS_PER_S
            if rate < limiter.default_rps:
                limiter.seed(domain, rate)
                n += 1
        return n

    def save(self, limiter: DomainLimiter) -> None:
        if self._db is None:
            return
        now = time.time()
        try:
            self._db.execute("DELETE FROM rates")
            for d, r in limiter.learned().items():
                self._db.execute("INSERT OR REPLACE INTO rates VALUES (?, ?, ?)", (d, r, now))
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
