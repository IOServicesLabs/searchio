"""Spending limits for a swarm run.

A research swarm is the most expensive thing in this package by a wide margin.
Anthropic's own multi-agent research system measured roughly 15x the tokens of
a single-agent chat for the same question, and that is the *intended* cost, not
a bug. So the ceiling has to be explicit, enforced, and checked in more than
one place -- a budget that is only consulted before dispatch will be blown
through by five workers that each looked cheap when they started.

Three independent limits, because runs fail expensively in three different
ways: a plan with too many subtasks, one worker that will not stop calling
tools, and a run that is simply taking too long to be useful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..errors import BudgetExhausted

# Claude Opus 5 list price, dollars per million tokens. Used only to report an
# estimate; nothing here bills anyone.
PRICE_IN = 5.0
PRICE_OUT = 25.0


@dataclass
class Budget:
    """Tracks and enforces what a run may spend."""

    max_tokens: int = 400_000
    max_wall_clock_s: float = 600.0
    max_workers: int = 5
    max_tool_calls_per_worker: int = 12

    #: USD per million tokens. Zero means unpriced -- report tokens only.
    price_in: float = PRICE_IN
    price_out: float = PRICE_OUT

    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    stopped_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    @property
    def priced(self) -> bool:
        return bool(self.price_in or self.price_out)

    @property
    def estimated_cost(self) -> float:
        return (self.input_tokens / 1e6) * self.price_in + (
            self.output_tokens / 1e6
        ) * self.price_out

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def exhausted(self) -> str:
        """Return a reason string if any limit is spent, else empty."""
        if self.total_tokens >= self.max_tokens:
            return f"token budget exhausted ({self.total_tokens:,}/{self.max_tokens:,})"
        if self.elapsed_s >= self.max_wall_clock_s:
            return f"wall clock exhausted ({self.elapsed_s:.0f}s/{self.max_wall_clock_s:.0f}s)"
        return ""

    def check(self) -> None:
        """Raise if the run must stop now."""
        reason = self.exhausted()
        if reason:
            self.stopped_reason = reason
            raise BudgetExhausted(reason)

    def soft_check(self) -> bool:
        """Non-raising variant for loop conditions.

        Returns True while there is room to continue. Used where stopping
        cleanly and reporting partial findings beats raising -- a swarm that
        hits its ceiling should hand back what it learned, not nothing.
        """
        reason = self.exhausted()
        if reason and not self.stopped_reason:
            self.stopped_reason = reason
        return not reason

    def summary(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4) if self.priced else None,
            "elapsed_s": round(self.elapsed_s, 1),
            "stopped_reason": self.stopped_reason,
        }
