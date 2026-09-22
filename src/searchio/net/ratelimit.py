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

    async def acquire(self, domain: str) -> None:
        """Block until this domain may be hit again."""
        b = self._bucket(domain)
        async with b.lock:
            while True:
                b.refill()
                if b.tokens >= 1.0:
                    b.tokens -= 1.0
                    return
                deficit = 1.0 - b.tokens
                wait = deficit / max(b.rate, FLOOR)
                # Jitter breaks up the metronomic timing that makes automated
                # traffic legible even when every individual request is clean.
                wait *= 1.0 + random.uniform(-JITTER, JITTER)
                await asyncio.sleep(max(wait, 0.01))

    def record_success(self, domain: str) -> None:
        b = self._bucket(domain)
        b.rate = min(self.max_rps, b.rate + ADDITIVE)

    def record_block(self, domain: str) -> None:
        b = self._bucket(domain)
        b.rate = max(FLOOR, b.rate * BACKOFF)
        # Surrender the burst allowance too. Backing off the refill rate while
        # three tokens are still in hand means the next three requests go out
        # immediately at full speed, which is precisely what just failed.
        b.tokens = 0.0
        b.updated = time.monotonic()

    def slot(self) -> asyncio.Semaphore:
        """Global concurrency guard, used as an async context manager."""
        return self._sem
