"""Runtime configuration, all overridable by environment (prefix ``SEARCHIO_``).

Design rule: searchio must do something useful with an empty environment. Every
key-requiring capability is optional and degrades to a keyless path, so `pip
install` then `searchio search "..."` works with nothing configured. The only
thing an API key buys you here is the swarm (which needs Claude) — plain search
and item lookup are entirely keyless.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_state_dir() -> Path:
    return Path(os.environ.get("SEARCHIO_STATE_DIR") or (Path.home() / ".searchio"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SEARCHIO_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # ── Validation ───────────────────────────────────────────────────────────
    #: Every field here comes from the environment. A typo must fail at
    #: startup naming the field (bug 87), not surface three modules away as
    #: a hang (per_domain_burst=0 hung the limiter), a silent default
    #: (robots_policy='bogus'), or a swarm that dispatches nothing.
    @model_validator(mode="after")
    def _check_ranges(self) -> "Settings":
        problems: list[str] = []

        def at_least(name: str, floor):
            if getattr(self, name) < floor:
                problems.append(f"{name} must be >= {floor}, got {getattr(self, name)}")

        def positive(name: str):
            if not getattr(self, name) > 0:
                problems.append(f"{name} must be > 0, got {getattr(self, name)}")

        for n in ("cache_ttl_s", "clearance_ttl_s", "min_text_chars", "max_redirects"):
            at_least(n, 0)
        for n in ("per_domain_burst", "global_concurrency", "pdf_max_bytes", "pdf_max_pages",
                  "max_body_bytes", "fanout", "results_per_provider", "max_workers",
                  "worker_concurrency", "max_tool_calls_per_worker", "swarm_token_budget"):
            at_least(n, 1)
        for n in ("per_domain_rps", "fetch_timeout_s", "provider_timeout_s",
                  "sidecar_boot_timeout_s", "handoff_timeout_s", "swarm_wall_clock_s",
                  # The tier timeouts are timeouts too (bug 180): tier2 feeds
                  # int(tier2_timeout_s * 1000 / 3), so a zero or negative one
                  # is a broken timeout, not a config the operator meant.
                  "tier0_timeout_s", "tier1_timeout_s", "tier2_timeout_s"):
            positive(n)
        if self.max_tier not in (0, 1, 2):
            problems.append(f"max_tier must be 0, 1 or 2, got {self.max_tier}")
        if self.robots_policy not in ("enforce", "warn", "off"):
            problems.append(f"robots_policy must be enforce|warn|off, got {self.robots_policy!r}")
        if self.llm_provider.lower() not in ("anthropic", "deepseek", "openai_compat", "openai-compatible"):
            problems.append(f"llm_provider must be anthropic|deepseek|openai_compat, got {self.llm_provider!r}")
        # String enums that were plain str (bug 181): a typo used to disable
        # the challenge rescue silently ("patchrite" != "patchright" -> no
        # rescue) or reach the model API as a bad effort (a late 400), never a
        # config error the operator sees at startup.
        if self.sidecar_challenge.strip().lower() not in ("patchright", "engine"):
            problems.append(f"sidecar_challenge must be patchright|engine, got {self.sidecar_challenge!r}")
        for n in ("lead_effort", "worker_effort"):
            if getattr(self, n).strip().lower() not in ("low", "medium", "high"):
                problems.append(f"{n} must be low|medium|high, got {getattr(self, n)!r}")
        if problems:
            raise ValueError("; ".join(problems))
        # The rate ceiling follows the floor rather than refusing: a control
        # site legitimately runs at 1000 rps with the default 4.0 ceiling.
        if self.per_domain_max_rps < self.per_domain_rps:
            self.per_domain_max_rps = self.per_domain_rps
        return self

    # ── State ────────────────────────────────────────────────────────────────
    state_dir: Path = Field(default_factory=_default_state_dir)
    cache_ttl_s: int = 3600
    cache_enabled: bool = True
    # Clearance cookies earned by the browser, replayed on cheaper tiers.
    clearance_enabled: bool = True
    clearance_ttl_s: int = 1500

    # ── Acquisition ladder ───────────────────────────────────────────────────
    # Tier ceiling: 0 plain HTTP, 1 TLS-impersonating HTTP, 2 stealth browser.
    # Lowering this to 1 guarantees searchio never spends a browser.
    max_tier: int = 2
    fetch_timeout_s: float = 20.0
    tier0_timeout_s: float = 8.0
    tier1_timeout_s: float = 15.0
    tier2_timeout_s: float = 90.0
    max_redirects: int = 5
    # Floor for stripped visible text before a 200 counts as a real document.
    # Calibrated on the same measurements SwarmIO's http_first used: a React
    # shell strips to ~70 chars, example.com to ~142.
    min_text_chars: int = 120
    # PDF text extraction (net/pdf.py): when classify identifies a PDF, the
    # ladder re-fetches the bytes bounded and serves the extracted text
    # instead of refusing. Off = the pre-capability honest refusal.
    pdf_extraction: bool = True
    pdf_max_bytes: int = 25 * 1024 * 1024
    pdf_max_pages: int = 30
    # Hard cap on a page body, counted on the DECODED bytes at every tier:
    # a gzip bomb is kilobytes on the wire and gigabytes in memory, which is
    # precisely the shape that must never materialize. Over the cap the tier
    # refuses too_large -- never a silent truncation served as a page.
    # 32 MiB (was 4): SSR pages inline their state now -- vinted's catalog
    # answered 7.2 MB decoded and the whole fetch died on a page a browser
    # serves without blinking; the PDF path already trusts 25 MiB. Matches
    # the engine's MAX_DOCUMENT_BYTES so a tier-2 body the engine accepted
    # is never re-refused at the ladder.
    max_body_bytes: int = 32 * 1024 * 1024

    # ── Politeness ───────────────────────────────────────────────────────────
    # These are not only courtesy: a request pattern no human could produce is
    # itself the signal most anti-bot systems key on, so pacing is part of how
    # the ladder stays unblocked rather than a tax on it.
    per_domain_rps: float = 0.75
    per_domain_burst: int = 3
    per_domain_max_rps: float = 4.0
    global_concurrency: int = 24
    robots_policy: str = "warn"  # enforce | warn | off
    # Egress proxy, applied to every tier. IP reputation is the one input
    # to an anti-bot decision that no amount of fingerprint work can move,
    # so this is the lever that matters once a host has judged your address.
    # Format: http://user:pass@host:port (or socks5://...).
    proxy: str = Field("", repr=False, exclude=True)  # may carry user:pass@
    # Optional second proxy used only for the browser tier, e.g. when the
    # residential pool is metered and you would rather not spend it on the
    # cheap tiers.
    browser_proxy: str = Field("", repr=False, exclude=True)
    user_agent_contact: str = ""  # e.g. "+https://example.com/bot"

    # ── Browser sidecar (SwarmIO) ────────────────────────────────────────────
    # Point at a running sidecar, or give the script path and searchio spawns
    # one lazily on first tier-2 escalation.
    sidecar_url: str = ""
    sidecar_script: str = ""
    sidecar_port: int = 8899
    sidecar_token: str = Field("", repr=False, exclude=True)
    # Interpreter used to launch the sidecar. Empty means auto-detect one
    # that can import patchright -- searchio's own venv usually cannot.
    sidecar_python: str = ""
    sidecar_autostart: bool = True
    sidecar_boot_timeout_s: float = 120.0
    # Swap tier 2's backend from the patchright script to the Rust
    # searchio-engine sidecar (se-serve). Both speak the same wire; the
    # engine boots in milliseconds and carries no Chromium. The binary is
    # resolved from SEARCHIO_ENGINE_BIN or a sibling searchio-engine
    # checkout; if neither yields a binary, tier 2 reports unavailable
    # rather than silently falling back to patchright.
    sidecar_engine: bool = False
    # The fidelity/challenge tier: which backend answers ``rendered`` fetches
    # (ladder ``rendered=True``, the swarm's read_page_rendered tool) and
    # one-shot shell rescues when the primary tier-2 backend comes back with
    # a JS shell. "patchright" (full Chromium: module scripts, WASM, real
    # sessions, managed challenges) or "engine". Independent of
    # ``sidecar_engine`` so both backends coexist in one process: the engine
    # takes the cheap tier-2 load, patchright takes the load that genuinely
    # needs a browser. Auto-rescue only ever fires engine -> patchright --
    # escalating a real browser down to the engine would be backwards -- so
    # with patchright primary and engine challenge, only an explicit
    # ``rendered=True`` reaches the engine.
    sidecar_challenge: str = "patchright"
    sidecar_challenge_url: str = ""
    sidecar_challenge_port: int = 8901
    sidecar_challenge_auto: bool = True
    # Hosts that open on the challenge sidecar (a real browser, when the
    # primary tier-2 backend is the engine) BEFORE the cheap tiers. For the
    # "adaptive" anti-bot class the origin 429s the initial document and
    # serves content only after browser-grade behavior -- statics, sensor
    # POSTs. probe_realtor_patchright.py (engine ec52690, 2026-09-17):
    # realtor.com 429'd every cheap tier on first contact, then one
    # production patchright pass settled on the genuine 801KB SRP. The cheap
    # tiers have no behavior to continue with, so the normal climb can never
    # pass these hosts -- and iteration 19 (a 429 buys no rescue) means no
    # unaided fetch ever reaches the browser. Opening rendered is the
    # sanctioned path; the normal climb stays as the fallback when the
    # browser pass itself fails. Subdomains match an entry (www.realtor.com
    # matches "realtor.com"). Tenant-scoped fetches skip the opener: the
    # browser's cookie store is one per process (bug 136's rule), and an
    # operator-chosen routing table does not get to override tenant isolation.
    rendered_first_domains: list[str] = ["realtor.com"]

    # ── Providers ────────────────────────────────────────────────────────────
    fanout: int = 4  # providers queried in parallel per search
    results_per_provider: int = 10
    provider_timeout_s: float = 12.0
    # Optional keys. Absent => that adapter stays dormant and is never routed to.
    serpapi_key: str = Field("", repr=False, exclude=True)
    serper_key: str = Field("", repr=False, exclude=True)
    brave_key: str = Field("", repr=False, exclude=True)
    tavily_key: str = Field("", repr=False, exclude=True)
    exa_key: str = Field("", repr=False, exclude=True)
    github_token: str = Field("", repr=False, exclude=True)
    # Facebook Marketplace is geolocated by exit IP: an unpinned query answered
    # from a datacentre returns that datacentre's metro, confidently and
    # wrongly. Set the city slug as it appears in the Marketplace URL
    # ("nyc", "seattle", "la") to say which one you meant.
    facebook_city: str = ""
    # Path to a Playwright storage_state JSON. Logged out, Marketplace caps a
    # search at ~14 listings; a session lifts that. Produce the file by logging
    # in by hand through SwarmIO's interactive_login -- searchio never handles
    # the password, and a scripted login is the fastest way to get an account
    # checkpointed.
    facebook_session: str = ""
    # Run the browser with a visible window for Marketplace. A forced-invisible
    # window is a strong bot signal on the most block-prone target in the
    # system: measured 37 listing hrefs headful vs 11 headless for the
    # identical anonymous query. Off by default because servers have no
    # display; on a desktop it is worth the window. An explicit
    # SWARM_BROWSER_WINDOW_MODE / SWARM_BROWSER_HEADLESS always wins.
    facebook_headed: bool = False
    # Delegated headed login handoff (firing 40, see net/login_handoff.py).
    # When a session-gated search collapses to the silent empty feed
    # (feed_units: null) and this is on, open a headed browser, let a
    # HUMAN log in by hand, export the storage_state into the engine and
    # retry once. Off by default: a headed window popping on a server is
    # an operator decision, not a default. Credentials are never scripted
    # either way — that is precisely what the handoff exists to avoid.
    facebook_auto_handoff: bool = False
    facebook_login_url: str = "https://www.facebook.com/login"
    handoff_timeout_s: float = 300.0

    # ── Swarm ────────────────────────────────────────────────────────────────
    # Backend selection. "anthropic" is the reference implementation; any
    # OpenAI-compatible endpoint (DeepSeek, self-hosted vLLM) works via the
    # adapter in swarm/llm.py.
    llm_provider: str = "anthropic"
    anthropic_api_key: str = Field("", repr=False, exclude=True)
    model: str = "claude-opus-5"
    deepseek_api_key: str = Field("", repr=False, exclude=True)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-pro"
    # Token prices in USD per million, used only to report an estimate.
    # Left at 0 for non-Anthropic backends rather than guessing: a made-up
    # dollar figure is worse than none, because it gets believed.
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    lead_effort: str = "high"
    worker_effort: str = "low"
    max_workers: int = 5
    worker_concurrency: int = 5
    max_tool_calls_per_worker: int = 12
    swarm_token_budget: int = 400_000
    swarm_wall_clock_s: float = 600.0

    def cache_path(self) -> Path:
        return self.state_dir / "cache.sqlite3"

    def profile_path(self) -> Path:
        return self.state_dir / "domains.sqlite3"

    def clearance_path(self) -> Path:
        return self.state_dir / "clearance.sqlite3"

    def ensure_state_dir(self) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        return self.state_dir

    def prices(self) -> tuple[float, float]:
        """(input, output) USD per million tokens, or (0, 0) when unknown."""
        if self.price_in_per_mtok or self.price_out_per_mtok:
            return self.price_in_per_mtok, self.price_out_per_mtok
        if (self.llm_provider or "anthropic").lower() == "anthropic":
            return 5.0, 25.0  # Claude Opus 5 list price
        return 0.0, 0.0

    def anthropic_key(self) -> str:
        """The swarm's key, falling back to the SDK's own env var.

        An unset key is not fatal at import time — only :mod:`searchio.swarm`
        needs it, and `searchio search` must keep working without one.
        """
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(s: Settings | None) -> None:
    """Override the process-wide settings (tests, embedding callers).
    ``None`` forgets the singleton so the next get_settings() re-reads the
    environment."""
    global _settings
    _settings = s
