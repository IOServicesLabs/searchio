"""Error taxonomy.

The split that matters everywhere else in the codebase is *whose fault* a
failure was, because that decides what happens next:

* :class:`Blocked` — the site refused us. Escalate to a heavier tier.
* :class:`Unusable` — the site answered, but with something no reader can use
  (an empty SPA shell, a PDF where HTML was expected). Also escalate.
* :class:`TransientError` — network flake. Retry the *same* tier.
* :class:`ProviderError` — an adapter is broken or out of quota. Take it out of
  the rotation; do not escalate, there is nothing to escalate to.

Escalating on a transient error is the expensive mistake: it spends a browser
on what a second plain GET would have answered.
"""

from __future__ import annotations


class SearchioError(Exception):
    """Base for everything this package raises."""


class ConfigError(SearchioError):
    """Missing or contradictory configuration."""


class TransientError(SearchioError):
    """Network-level flake. Retry the same tier before escalating."""


class TargetRefused(TransientError):
    """Policy refusal: the URL itself is off-limits (link-local/unspecified
    IP literal, non-http(s) scheme).

    A subclass of :class:`TransientError` so generic handling still treats
    it as "no page arrived", but the ladder keys on the exact type: a policy
    refusal is PERMANENT -- no tier can fetch a refused target -- so it is
    raised out of the tier loop immediately, never retried, never climbed.
    (Bug 25: without the type split, a redirect to the cloud metadata
    endpoint was retried at every tier and the refusal text drowned under
    the last tier's unrelated error.)
    """


class NavErrorPage(TransientError):
    """The tab committed the browser's network-error page.

    A *page-initiated* navigation (page script, meta refresh) that a real
    browser followed natively ended on ``chrome-error://`` -- the followed
    target was unreachable, and the tab now holds the error page, not
    content. A subclass of :class:`TransientError` so generic handling still
    treats it as "no page arrived", but the ladder keys on the exact type:
    this fetch is DONE. The shell fetch already proved the network path, and
    re-fetching the shell deterministically re-follows the same script into
    the same dead end -- so the tier loop neither retries nor climbs on it.
    (Bug 32: without the split, the same-tier retry spent a second browser
    navigation reaching the identical dino page.)
    """


class Blocked(SearchioError):
    """An anti-bot system refused the request.

    ``vendor`` names the system when we could identify it (cloudflare,
    datadome, perimeterx, akamai, incapsula) — useful because the right
    response differs: Cloudflare's managed challenge usually clears with a real
    browser, while a DataDome block is frequently IP reputation and a new
    browser on the same egress will fail identically.
    """

    def __init__(self, reason: str, vendor: str | None = None, status: int | None = None):
        self.reason = reason
        self.vendor = vendor
        self.status = status
        detail = f"{vendor}: {reason}" if vendor else reason
        super().__init__(f"blocked ({detail})" + (f" [HTTP {status}]" if status else ""))


class Unusable(SearchioError):
    """A response arrived but carries no readable content."""

    def __init__(self, reason: str, status: int | None = None):
        self.reason = reason
        self.status = status
        super().__init__(f"unusable ({reason})")


class ProviderError(SearchioError):
    """A provider adapter failed in a way retrying will not fix."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"provider {provider}: {message}")


class BudgetExhausted(SearchioError):
    """The run hit its token, dollar, or wall-clock ceiling."""


class SidecarUnavailable(SearchioError):
    """The SwarmIO browser sidecar could not be reached or spawned."""


class SidecarVerbError(SearchioError):
    """The sidecar answered, and said the verb failed (a dead tab, a bad
    selector, a navigation that never committed). Distinct from
    :class:`SidecarUnavailable`: the sidecar is fine, this call is not --
    and its typed readers used to hide it as ``""`` / ``[]`` (bug 101)."""


class LoginHandoffError(SearchioError):
    """A delegated headed login handoff failed.

    The handoff flow (open a headed browser, a human logs in by hand,
    inject the exported session) exists precisely so no script ever
    submits credentials — a scripted login is the single most reliable
    way to earn a checkpoint. Failures here are operator problems (nobody
    logged in in time, the headed backend is missing), never retried
    with automation: the anonymous fallback result stands instead.
    """


class LoginHandoffUnavailable(LoginHandoffError):
    """The headed backend (patchright) cannot be started here."""


class LoginHandoffTimeout(LoginHandoffError):
    """Nobody completed the login within the handoff deadline."""


class LoginHandoffCancelled(LoginHandoffError):
    """The wait on the human was cancelled (e.g. the run was stopped)."""
