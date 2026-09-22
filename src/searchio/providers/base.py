"""The provider contract.

A provider is anything that turns a query into ranked documents: a search
engine scraped from HTML, a public JSON API, or a commercial SERP service. The
router only needs four facts about each one -- what it can answer, what it
costs, whether it is configured, and how healthy it has been lately -- so the
interface stays deliberately small.

Providers must not raise for ordinary emptiness. A provider with nothing to say
returns ``[]``; exceptions are reserved for real failures, because the router
counts them against a circuit breaker.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

from ..models import Capability, Doc, Query


@dataclass
class Health:
    """Rolling health for one provider, driving the router's circuit breaker.

    ``ema_latency`` and ``ema_success`` are exponential moving averages rather
    than windows because they need no bookkeeping and decay old evidence
    naturally -- a provider that broke an hour ago should not be punished
    forever once it starts answering again.
    """

    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    ema_latency_ms: float = 500.0
    ema_success: float = 1.0
    opened_at: float = 0.0  # circuit-breaker trip time, 0 when closed
    last_error: str = ""

    ALPHA = 0.3
    TRIP_AFTER = 3  # consecutive failures before opening
    COOLDOWN_S = 120.0

    def record(self, ok: bool, latency_ms: float, error: str = "") -> None:
        self.calls += 1
        self.ema_latency_ms = (1 - self.ALPHA) * self.ema_latency_ms + self.ALPHA * latency_ms
        self.ema_success = (1 - self.ALPHA) * self.ema_success + self.ALPHA * (1.0 if ok else 0.0)
        if ok:
            self.consecutive_failures = 0
            if self.opened_at:
                self.opened_at = 0.0  # half-open probe succeeded; close it
        else:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = error[:200]
            if self.consecutive_failures >= self.TRIP_AFTER:
                self.opened_at = time.monotonic()

    @property
    def open(self) -> bool:
        """True while the breaker is holding calls back."""
        if not self.opened_at:
            return False
        if time.monotonic() - self.opened_at > self.COOLDOWN_S:
            # Cooldown elapsed: allow one probe through (half-open).
            return False
        return True

    @property
    def score(self) -> float:
        """0..1 desirability. Success dominates; latency is a tiebreak."""
        if self.open:
            return 0.0
        latency_factor = 1.0 / (1.0 + self.ema_latency_ms / 2000.0)
        return 0.8 * self.ema_success + 0.2 * latency_factor


class Provider(abc.ABC):
    """One source of search results."""

    name: str = "provider"
    caps: frozenset[Capability] = frozenset({Capability.WEB})
    #: Dollars per thousand queries. Zero means keyless/free; the router uses
    #: this to prefer free sources until quality demands a paid one.
    cost_per_1k: float = 0.0
    #: Settings attribute holding this provider's key, or None if keyless.
    key_field: str | None = None
    #: Providers that hit one host hard should declare it so the ladder's
    #: per-domain limiter can pace them together.
    primary_domain: str = ""

    def __init__(self) -> None:
        self.health = Health()

    def configured(self, settings) -> bool:
        """Whether this provider can run at all in the current environment."""
        if self.key_field is None:
            return True
        return bool(getattr(settings, self.key_field, ""))

    def supports(self, intent: str) -> bool:
        return any(c.value == intent for c in self.caps)

    @abc.abstractmethod
    async def search(self, q: Query, ctx: "ProviderContext") -> list[Doc]:
        """Return ranked documents. Empty is a valid answer; raise only on failure."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}>"


@dataclass
class ProviderContext:
    """Shared machinery handed to every provider call.

    Providers get the ladder rather than an HTTP client of their own, so that
    every outbound request -- including a provider scraping a SERP -- goes
    through the same pacing, caching, block detection, and tier escalation.
    Bypassing it would put the one class of traffic most likely to be blocked
    outside the machinery built to keep it unblocked.
    """

    ladder: object
    settings: object
    session_id: str = ""
    #: The router, when one exists. Providers that need to *find* pages before
    #: working on them (the marketplace adapter, say) should discover through
    #: this rather than calling one engine directly, so discovery inherits the
    #: same failover every other search gets. Optional, because a provider must
    #: still work when constructed standalone in a test.
    router: object | None = None
    extra: dict = field(default_factory=dict)
