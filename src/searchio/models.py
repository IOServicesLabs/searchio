"""The vocabulary every layer shares.

Two result shapes, deliberately kept apart:

* :class:`Doc` — something to *read*. A page with a title, a URL, and text.
* :class:`Item` — something to *buy or compare*. A listing with a price, a
  seller, and availability.

They stay separate because the useful operations differ. Docs get deduped by
URL and reranked by relevance; items get deduped by *product identity* (two
Amazon URLs for the same ASIN are one item) and ranked by price. Flattening
both into one "result" type means every consumer re-derives which kind it has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Capability(str, Enum):
    """What a provider can answer. The router matches intent to capability."""

    WEB = "web"
    NEWS = "news"
    VIDEO = "video"
    SHOPPING = "shopping"
    ACADEMIC = "academic"
    CODE = "code"
    LOCAL = "local"
    REFERENCE = "reference"
    FORUM = "forum"


Intent = Literal[
    "web", "news", "video", "shopping", "academic", "code", "local",
    "reference", "forum",
]


class Doc(BaseModel):
    """A retrieved document."""

    url: str
    title: str = ""
    snippet: str = ""
    text: str = ""  # full extracted markdown, populated only when fetched
    published: str | None = None
    source: str = ""  # provider name that surfaced it
    rank: int = 0  # rank within that provider's own result list, 0-based
    score: float = 0.0  # fused score, filled by searchio.fuse
    fetched_via: str = ""  # which ladder tier served the body
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def domain(self) -> str:
        from urllib.parse import urlsplit

        return urlsplit(self.url).netloc.lower().removeprefix("www.")


class Price(BaseModel):
    amount: float | None = None
    currency: str = "USD"
    raw: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        if self.amount is None:
            return self.raw or "?"
        return f"{self.amount:,.2f} {self.currency}"


class Item(BaseModel):
    """A normalized marketplace listing.

    ``identity`` is the cross-site join key: a GTIN/ASIN/MPN when the page
    exposes one, else a normalized brand+model slug. It is what lets the same
    pair of headphones on Amazon, Best Buy, and Target collapse into one row
    with three offers instead of three near-identical rows.
    """

    title: str
    url: str
    price: Price = Field(default_factory=Price)
    seller: str = ""
    brand: str = ""
    condition: str = ""
    availability: str = ""
    rating: float | None = None
    reviews: int | None = None
    image: str = ""
    identity: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    source: str = ""
    verified: bool = False  # detail page opened and confirmed live
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def domain(self) -> str:
        from urllib.parse import urlsplit

        return urlsplit(self.url).netloc.lower().removeprefix("www.")


class Query(BaseModel):
    """One search request as providers see it."""

    text: str
    intent: Intent = "web"
    k: int = 10
    freshness: Literal["day", "week", "month", "year", "all"] = "all"
    domains: list[str] = Field(default_factory=list)  # allowlist
    exclude_domains: list[str] = Field(default_factory=list)
    locale: str = "en-US"
    region: str = "us"


class FetchResult(BaseModel):
    """What the acquisition ladder returns."""

    url: str
    final_url: str = ""
    status: int = 0
    body: str = ""
    content_type: str = ""
    tier: int = 0
    via: str = ""  # 'http', 'impersonate', 'browser', 'cache'
    elapsed_ms: int = 0
    escalations: list[str] = Field(default_factory=list)  # why each tier was abandoned
    from_cache: bool = False
    #: True when the body was served by the challenge/fidelity sidecar
    #: (ladder ``rendered=True``, or the one-shot engine-shell auto-rescue).
    rendered: bool = False


# ── Swarm ────────────────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """One independent thread of research, handed to one worker.

    Workers never see each other. Everything a worker needs to do its job has
    to be in here, because there is no channel to ask a follow-up question.
    """

    id: str
    objective: str
    rationale: str = ""
    intent: Intent = "web"
    queries: list[str] = Field(default_factory=list)
    success_criteria: str = ""


class ResearchPlan(BaseModel):
    interpretation: str
    subtasks: list[SubTask]


class Citation(BaseModel):
    url: str
    title: str = ""
    quote: str = ""


class Finding(BaseModel):
    """One claim a worker is willing to stand behind, with its evidence."""

    claim: str
    detail: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    citations: list[Citation] = Field(default_factory=list)
    subtask_id: str = ""


class WorkerReport(BaseModel):
    subtask_id: str
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    docs: list[Doc] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


class ResearchResult(BaseModel):
    query: str
    answer: str
    plan: ResearchPlan | None = None
    findings: list[Finding] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    sources: list[Doc] = Field(default_factory=list)
    reports: list[WorkerReport] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    #: None when the backend's token prices are not configured.
    estimated_cost_usd: float | None = None
    model: str = ""
    elapsed_s: float = 0.0
    stopped_early: str = ""
    #: Everything that makes this result partial or edited (iteration 46):
    #: a crashed worker, a truncated plan, answer links no worker retrieved.
    warnings: list[str] = Field(default_factory=list)


# ── Internal bookkeeping (not part of the API surface) ───────────────────────


@dataclass
class DomainProfile:
    """What we have learned about how hard one domain is to read.

    ``min_tier`` is the payoff: after a site has proved it needs a browser we
    stop paying two failed round trips to rediscover that on every request, and
    when it later gets easier a periodic probe walks the tier back down.
    """

    domain: str
    min_tier: int = 0
    successes: int = 0
    blocks: int = 0
    last_block_vendor: str = ""
    last_seen: float = field(default_factory=time.time)
    rps: float = 0.75  # current adaptive rate, AIMD-controlled

    @property
    def block_rate(self) -> float:
        total = self.successes + self.blocks
        return self.blocks / total if total else 0.0
