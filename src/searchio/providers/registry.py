"""Which providers exist, and which of them can run right now.

Registration is explicit rather than by plugin discovery. A search system whose
provider set depends on what happens to be importable is a system whose results
are not reproducible, and "why did this query return different sources on the
server than on my laptop" is a miserable thing to debug.
"""

from __future__ import annotations

from ..models import Capability
from .apis import (
    ArxivProvider,
    Crossref,
    GitHubRepos,
    HackerNews,
    OpenAlex,
    StackExchange,
    Wikipedia,
)
from .base import Provider
from .engines import Bing, DuckDuckGo, SidecarSearch
from .facebook import FacebookMarketplace
from .marketplace import MarketplaceItems, SidecarListings
from .youtube import YouTube


def default_providers() -> list[Provider]:
    """Every provider searchio knows about, configured or not.

    The router filters by :meth:`Provider.configured` and by capability, so an
    unconfigured provider costs nothing but is visible in ``searchio providers``
    -- which is how a user finds out that adding a key would help.
    """
    return [
        # Keyless general web
        DuckDuckGo(),
        Bing(),
        # Keyless structured
        Wikipedia(),
        HackerNews(),
        YouTube(),
        ArxivProvider(),
        OpenAlex(),
        Crossref(),
        GitHubRepos(),
        StackExchange(),
        # Items
        MarketplaceItems(),
        # Browser-backed, expensive, last resort
        FacebookMarketplace(),
        SidecarSearch(),
        SidecarListings(),
    ]


class Registry:
    """Holds provider instances and their health across a process's lifetime.

    Health lives on the instances, so the registry must be long-lived for the
    circuit breakers to mean anything -- rebuilding it per request would reset
    every breaker and re-learn every outage from scratch.
    """

    def __init__(self, providers: list[Provider] | None = None) -> None:
        self._providers = providers if providers is not None else default_providers()
        self._by_name = {p.name: p for p in self._providers}

    def all(self) -> list[Provider]:
        return list(self._providers)

    def get(self, name: str) -> Provider | None:
        return self._by_name.get(name)

    def available(self, settings, intent: str) -> list[Provider]:
        """Configured providers that support ``intent`` and are not circuit-open."""
        return [
            p
            for p in self._providers
            if p.configured(settings) and p.supports(intent) and not p.health.open
        ]

    def capabilities(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {c.value: [] for c in Capability}
        for p in self._providers:
            for c in p.caps:
                out[c.value].append(p.name)
        return out
