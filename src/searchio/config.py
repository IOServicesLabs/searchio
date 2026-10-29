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
    # Seeded browser sessions: the platform's directory of login jars
    # (Playwright storage_state + a swarmio block), exported by the cloud.
    # Empty = disabled: searchio behaves exactly as if sessions did not
    # exist. When set, a fetch that hits a wall retries at the same tier
    # with the site's jar and rotated cookies are written to
    # <dir>/writeback/ for the cloud to re-seal. See docs/SESSIONS.md.
    sessions_dir: str = ""

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

    # ── Fan-out speed (each knob is A/B-tested by bench/ab.py) ──────────────
    # Defaults follow the A/B verdicts in bench/IMPROVEMENTS.md: a knob is on
    # only where it won live or on replay without costing results.
    # Early return: once ``fanout_quorum`` providers have answered with docs,
    # wait at most ``fanout_grace_ms`` for the rest, then fuse what is in.
    # Stragglers keep running detached so their health and cache still land.
    # 0 = wait for every provider (up to provider_timeout_s).
    fanout_grace_ms: int = 0
    fanout_quorum: int = 2
    # Hedged endpoint fallback inside a provider (DDG html->lite, Bing
    # rss->html): start the fallback after this many ms instead of only after
    # the primary fails; the first acceptable answer wins. 0 = sequential.
    hedge_ms: int = 0
    # robots.txt persisted in the state dir, so a new process does not pay
    # a serial robots round trip per host before its first request.
    robots_persist: bool = True
    # Under robots_policy=warn/off a robots miss loads in the background
    # instead of gating the request (warn never refuses anyway).
    robots_nonblocking: bool = True
    # Tier 1 reuses a curl_cffi Session per worker thread and impersonation
    # target (keep-alive + TLS session reuse) instead of one-shot requests.
    tier1_pool: bool = True
    # Checked DNS answers reused for this long; 0 = resolve every dial.
    dns_cache_ttl_s: float = 0.0
    # Identical concurrent searches share one fan-out; identical concurrent
    # URL fetches share one ladder climb.
    singleflight: bool = True
    # Fused results for an identical query reused for this long, in memory.
    query_cache_ttl_s: float = 300.0
    # SQLite stores commit with synchronous=NORMAL (WAL keeps it durable
    # against a crash of the process; only an OS crash can drop the tail).
    sqlite_fast: bool = False
    # A provider whose first endpoint drew an anti-bot wall does not try its
    # sibling endpoints (DDG html->lite share one IP reputation; measured:
    # the lite attempt drew the same anomaly modal every time, ~3 s later).
    block_skip_siblings: bool = True
    # A provider refused by a wall on every attempt opens its breaker at
    # once instead of after three such searches.
    fast_trip_on_block: bool = True
    # Bing: once the HTML fallback has been degraded the last 3 times RSS
    # was, skip it after a degraded RSS (re-probed 1 time in 8).
    degraded_skip: bool = False
    # Per-provider timeout from its own latency (3x EMA + 1 s, floor 2 s,
    # ceiling provider_timeout_s) once it has a few calls of history.
    adaptive_timeout: bool = False
    # Idle keep-alive for the tier-0 connection pool. httpx's default is 5 s,
    # shorter than the gap between an agent's searches, so every search
    # paid a fresh TLS handshake per provider host.
    http_keepalive_s: float = 5.0
    # Long-lived servers (MCP, HTTP) warm the ladder at startup: imports,
    # the TLS client, robots.txt and a live connection per provider host.
    warmup_on_start: bool = True
    # A search provider's fetch fails Throttled instead of sleeping longer
    # than this on our own per-domain limiter (after blocks DDG's rate sits
    # at the 0.1 rps floor: a 10 s sleep inside a 12 s provider timeout).
    # 0 = sleep as long as the limiter says.
    serp_max_wait_s: float = 1.5
    # Provider health (breakers, latency/success EMAs) persisted in the state
    # dir: a CLI run starts knowing DDG walled this egress 30 s ago, instead of
    # re-learning it with a multi-second refusal.
    persist_health: bool = False
    # Query-cache / singleflight keys fold case and whitespace, so "Rust
    # async" and "rust  async" share one answer.
    normalize_query_keys: bool = True
    # Stale-while-revalidate for SERP fetches: an expired cached SERP up to
    # this many seconds past its TTL answers at once while a background fetch
    # refreshes it. 0 = an expired SERP is a miss. Never applies to a
    # freshness-bound query. A/B (replay): -303 ms [-354, -254] on repeats.
    serp_swr_s: float = 21600.0
    # DDG/Bing start with whichever endpoint answered most recently when the
    # usual primary has been failing.
    endpoint_learning: bool = False
    # If fewer than fanout_quorum providers have answered after this many ms,
    # launch the next-ranked provider not yet in the fan-out. 0 = off.
    backup_provider_ms: int = 0
    # Soft fan-out deadline: wait for every provider as usual, but once this
    # many ms have passed, return with whatever has answered (if anything
    # has); stragglers finish detached. Bounds the tail a slow provider adds
    # (arXiv/Crossref answering in 2.5-12 s) without the early return's
    # coverage cost on ordinary searches. 0 = off. A/B (live cold, 24 pairs):
    # p99 12046 -> 1527 ms, >2 s 8.3% -> 0%, docs 10.0 = 10.0.
    fanout_deadline_ms: int = 1500
    # Weight of latency (vs success rate) when ranking providers for the
    # fan-out; 0.2 is the stock Health.score blend. Matters only when an
    # intent has more candidates than ``fanout`` slots (code, academic).
    latency_weight: float = 0.2
    # SERP bodies over 20 KB parse on a worker thread instead of the loop.
    parse_offloop: bool = False
    # Cache writes run on a worker thread instead of committing on the
    # event loop inside the fetch.
    cache_write_behind: bool = False
    # After a search, fetch the top-k result pages in the background so the
    # read that usually follows is a cache hit (A/B, 2 s think time: read
    # p50 232 -> 29 ms). 0 = off.
    prefetch_top_k: int = 2  # b23 WIN (#119; was 1, #23)
    # Per-host limiter rates persisted (state_dir/rates.sqlite), so a new
    # process starts a host that just walled the last one at its learned
    # pace instead of a full burst at the default.
    persist_rates: bool = False
    # Run the CLI's event loop on winloop (Windows) / uvloop when installed.
    fast_loop: bool = False
    # Batch 5.
    # arXiv's export API over https directly: the http URL draws a 301 first.
    arxiv_https: bool = True
    # Ask the APIs for only the fields the parsers read (OpenAlex/Crossref
    # select=, HN attributesToRetrieve): OpenAlex 144 KB -> 31 KB a page.
    lean_api_fields: bool = False
    # Import the tier-1 client and parsers on a worker thread at startup,
    # and never on the event loop: a cold process's first tier-1 fetch used
    # to stall every other provider ~55 ms on ``import curl_cffi``.
    offloop_imports: bool = False
    # Hosts known to wall tier 0 start at tier 1 on first contact, so a
    # fresh state dir (a container without a volume) does not spend its
    # first DDG request on a wall -- and the post-wall limiter wait that
    # then fails the tier-1 retry Throttled under serp_max_wait_s.
    serp_tier_hints: bool = False
    # Breaker cooldown starts here and doubles on each re-trip up to the
    # 120 s ceiling (0 = the fixed 120 s): a transient DDG wall returns to
    # the fan-out in seconds, not minutes.
    breaker_backoff_s: float = 15.0
    # When every generalist is cooling, let the one that has cooled longest
    # through as a probe (at most once per this many seconds; 0 = off)
    # instead of fanning out with no general web engine at all.
    starved_probe_s: float = 10.0
    # Skip httpx's CLI import chain at package import. Acts in
    # searchio/__init__.py, which reads SEARCHIO_LEAN_IMPORTS itself (it must
    # run before this class's imports); declared here so it is documented.
    lean_imports: bool = True
    # Batch 6.
    # An API's rate-limit answer (429, or 403 with x-ratelimit-remaining 0)
    # ends the fetch at once -- no tier climb into the same refusal, no
    # profile pin -- and holds the provider until the stated reset.
    ratelimit_aware: bool = True  # b44: with ratelimit_503, WIN -434 ms
    #: With ratelimit_aware: a non-challenge 503 that states a numeric
    #: Retry-After is a rate limit too (no climb, held until then).
    ratelimit_503: bool = True  # b44 WIN -434 ms [-1107, -61], docs 8.4 vs 7.5
    #: ...only for documented API hosts (API_RATES): a page's 429 can be an
    #: anti-bot challenge that a better tier passes.
    ratelimit_api_only: bool = True
    # Seed the per-host limiter with each API's documented rate instead of
    # the generic 0.75 rps / burst 3 (faster for Wikipedia, OpenAlex, HN;
    # slower for arXiv's 1-per-3 s and GitHub search's 10/min).
    api_rate_table: bool = False
    # Serve an expired query-cache entry this many seconds past its TTL and
    # refresh it in the background (0 = off).
    query_cache_swr_s: float = 600.0
    # Persist fused results (state_dir/qcache.sqlite) so a new process can
    # answer a repeat query without a single provider call.
    persist_query_cache: bool = True
    # Stragglers the soft deadline left behind merge into the cached result
    # when they land, so a repeat of the query gets the whole fan-out.
    late_merge: bool = True
    # At the soft deadline, keep waiting while fewer than this many docs are
    # in hand, up to fanout_deadline_max_ms (0 = off).
    deadline_min_docs: int = 10
    fanout_deadline_max_ms: int = 3000
    # After a cooldown, one half-open probe at a time: concurrent searches
    # skip the provider while its probe is in flight.
    halfopen_single: bool = False
    # A provider failing "degraded" (answering, but only the first query
    # word: an IP-reputation verdict) is held this long instead of the
    # wall/timeout cooldown (0 = off).
    degraded_hold_s: float = 0.0
    # Marginalia's public search API as an extra keyless web generalist.
    # Opt-in: its results are licensed CC-BY-NC-SA 4.0 (non-commercial).
    marginalia_enabled: bool = False
    # Batch 7.
    # Remember that trafilatura failed to import: a broken optional
    # dependency was re-imported (and failed again, ~4 ms) on every read.
    memo_extract_import: bool = False
    # A prefetched page is also extracted (off the event loop) and the Doc
    # kept, so the read that follows skips extraction (6-30 ms a page).
    prefetch_extract: bool = True  # b76+b77 WIN (#161, re-test of #48 with trafilatura)
    # Extract pages of at least this many bytes on a worker thread, so one
    # large read does not stall every concurrent search (0 = inline).
    read_offloop_bytes: int = 0
    # A breaker tripped by timeouts or by our own limiter's Throttled holds
    # this long instead of the wall cooldown (0 = the normal rule).
    transient_hold_s: float = 0.0
    # Return before the soft deadline once k docs are in hand and every
    # provider still out has a latency EMA past that deadline.
    predictive_cut: bool = True
    # Find a page's <title> in its <head> alone (~3 ms saved on a 1 MB page).
    fast_title: bool = False
    # Keep every read's Doc keyed by its body: re-reading an unchanged
    # (cached) page skips extraction.
    read_memo: bool = True  # b72+b75 WIN (#159, re-test of #53 with trafilatura)
    # Batch 8.
    # Engine.search_many / the search_many MCP tool: searches in flight at once.
    search_many_concurrency: int = 3
    # Engine.read_many / the read_many MCP tool: reads in flight at once.
    read_many_concurrency: int = 4
    # Start the prefetch from the first generalist to answer, not after the
    # fan-out's deadline.
    prefetch_early: bool = False
    # Persist stated rate-limit holds of 60 s or more (state_dir/holds.sqlite)
    # so a new process skips a provider whose quota is spent until its reset.
    persist_holds: bool = False
    # Skip speculative prefetch while the measured link throughput is below
    # this many KB/s (0 = always prefetch).
    prefetch_min_kbps: float = 0.0
    # Cancel the soft deadline's stragglers instead of letting them finish
    # detached, freeing a slow link for the next search (their SERP is then
    # not cached; late_merge keeps them regardless).
    cancel_stragglers: bool = False
    # Tier 0 over HTTP/2 (one multiplexed connection per host). Off = HTTP/1.1
    # with a connection per request: a lost packet stalls one request only --
    # but a browser persona over HTTP/1.1 draws 403s (Wikimedia, probed).
    tier0_http2: bool = True
    # Connect phase timeout and retry (0 = the tier timeout, no retry): a
    # SYN lost on a spotty link fails over in seconds, not after 8 s.
    connect_timeout_s: float = 3.0  # b33 WIN on dead hosts (-4718 ms), neutral live
    connect_retries: int = 0
    # No response headers after this many ms: re-send the GET on a fresh
    # connection and keep the first answer (0 = off).
    stall_hedge_ms: int = 0
    # A failed DNS lookup dials the host's last vetted answer instead.
    dns_stale_if_error: bool = False
    # Network-level failures hitting several providers at once are the link
    # dropping, not the providers: not counted toward their breakers.
    link_down_guard: bool = False
    # When no provider answers, serve the query's expired cached answer if
    # it expired no more than this many seconds ago (0 = off).
    offline_stale_s: float = 0.0
    # A fetch the network failed is answered from an expired cache entry up
    # to this many seconds past its TTL (0 = off).
    stale_if_error_s: float = 0.0
    # Race TCP dials: no connection after this many ms starts another (the
    # next address, or the same on a fresh socket); first wins (0 = serial).
    connect_stagger_ms: int = 300  # b30 WIN on dead multi-address hosts, neutral live
    #: An address whose connect timed out while other dials succeeded is
    #: skipped this long; a host with no live address left fails at once
    #: instead of paying every tier's connect timeout again (~44 s here).
    dead_ip_ttl_s: float = 300.0  # b18 WIN (#113)
    #: Keep those addresses in state_dir/dead_ips.json across processes; a
    #: fresh process also counts a remote DNS answer as proof the link is up.
    dead_ip_persist: bool = False
    # Specialists (non-generalists) are asked for this fraction of k
    # (min 3; 0 = all k): fusion rarely keeps their tail.
    specialist_k_ratio: float = 0.0
    # The query cache ignores k: a cached answer fused for k >= the ask's k
    # answers it (its first k).
    qcache_k_superset: bool = False
    # With lean_api_fields: leave Crossref unselected (its select= answers slower).
    lean_skip_crossref: bool = False
    # Longest gap allowed between body chunks at tier 0 (0 = the tier timeout).
    read_stall_s: float = 0.0
    # A new search cancels the previous one's in-flight prefetches.
    prefetch_yield: bool = False
    # Tier-1 (curl_cffi) connect timeout (0 = the tier timeout covers it).
    tier1_connect_timeout_s: float = 0.0
    # Skip the robots.txt lookup for the documented JSON APIs (API_RATES
    # hosts) unless robots_policy is enforce.
    robots_skip_apis: bool = False
    # Resolve the provider hosts on a thread at CLI import, overlapping DNS
    # with the rest of startup (warms the OS resolver cache).
    dns_prefetch: bool = False
    # Under this link throughput (KB/s), read() keeps the first
    # read_soft_cap_kb of a page instead of downloading it all (0 = off).
    read_slow_kbps: float = 0.0
    read_soft_cap_kb: int = 512
    # Batch 9: lean reads (searchio.leanread) -- the same content, a fraction
    # of the bytes. Wikipedia article -> extracts API (~10x smaller on the
    # wire); arXiv PDF -> its HTML full text (2 MB -> ~150 KB); GitHub repo
    # page -> its raw README.md (56 KB -> 3 KB).
    lean_read_wikipedia: bool = True  # b9 WIN (#76)
    lean_read_arxiv: bool = True  # b9 WIN (#77)
    lean_read_github: bool = True  # b10 WIN (#78, with the #88 guard)
    # MDN doc page -> its index.json (16 KB -> 2 KB on the wire).
    lean_read_mdn: bool = False
    # GitHub file page (/blob/<ref>/<path>) -> raw.githubusercontent (95 -> 9 KB).
    lean_read_github_blob: bool = True  # b9 WIN (#84)
    # Hugging Face model/dataset/space page -> its raw README.md (91 -> 1 KB).
    lean_read_hf: bool = True  # b16 WIN (#85)
    # crates.io (a JS app that 404s a plain GET) -> docs.rs/crate/<name>/latest.
    lean_read_crates: bool = True  # b10 WIN (#86)
    # npmjs.com package page (403s tier 0) -> unpkg README (3 KB).
    lean_read_npm: bool = True  # b13 WIN (#87)
    # GitHub issue/PR page (300 KB) -> issues API + comments (2 x ~2 KB).
    lean_read_github_issue: bool = True  # b10 WIN (#89)
    # RFC HTML (1.2 MB decoded) -> rfc-editor plain text (0.5 MB, no extraction).
    lean_read_rfc: bool = False
    # Stale-while-revalidate for reads: a page cached up to this long past its
    # TTL answers at once while a background refetch replaces it. 0 = off.
    page_swr_s: float = 0.0
    # GitHub releases page (116 KB wire) -> releases API (41 KB).
    lean_read_github_releases: bool = True  # b16 WIN (#93)
    # GitHub commit page (48 KB) -> its .patch (the same diff, plain text).
    lean_read_github_commit: bool = True  # b13 WIN (#94)
    # kubernetes.io docs (67 KB) -> the page's source markdown (18 KB).
    lean_read_k8s: bool = False
    # doc.rust-lang.org mdBooks (book, reference, ...) -> source markdown.
    lean_read_rust_book: bool = False
    # Batch 12: docs sites read as their GitHub source (raw.githubusercontent,
    # no robots.txt, a fast CDN); probes are page -> source wire bytes.
    lean_read_github_wiki: bool = True  # 50 -> 9 KB; b12 WIN (#97)
    lean_read_hf_docs: bool = False  # 233 -> 5 KB
    lean_read_react: bool = False  # 54 -> 11 KB
    lean_read_vue: bool = True  # 18 -> 4 KB; b12 WIN (#100)
    lean_read_node_api: bool = False  # 28 -> 15 KB
    lean_read_django: bool = True  # 36 -> 21 KB; b12 WIN (#102)
    lean_read_fastapi: bool = False  # 18 -> 4 KB
    lean_read_pydantic: bool = True  # 40 -> 18 KB; b12 WIN (#104)
    # Learn each host's advertised markdown alternate (<link rel=alternate
    # type=text/markdown>) from one normal read; read its later pages as
    # markdown (Stripe 257 -> 1 KB, Vercel 91 -> 12 KB, Docker 83 -> 1 KB).
    lean_md_alternate: bool = True  # b13 WIN (#105)
    # Learn each host's page -> GitHub source mapping from its "Edit this
    # page" link (deno, astro, prometheus, hashicorp, tauri ...); later pages
    # there read as their raw source file.
    lean_edit_link: bool = False
    # Open connections to the top N results' hosts (a robots.txt GET on the
    # shared pool) so the read that follows skips TCP+TLS setup. 0 = off.
    preconnect_top_k: int = 4  # b11 WIN (#92)
    # Race getaddrinfo against a direct UDP query to the system's configured
    # DNS servers (searchio.net.dnsrace): here every new name stalled ~10 s
    # in the OS resolver while its own server answered in 7 ms.
    #: Page reads ask for text/markdown first; a site that negotiates
    #: (Cloudflare, Vercel, Mintlify, Stripe, GitHub docs...) answers with a
    #: body ~10x smaller that needs no extraction. Others send HTML as before.
    accept_markdown: bool = True  # b17 WIN (#111)
    #: accept_markdown also takes a text/plain reply shaped like markdown
    #: (Prisma's docs answer the markdown Accept that way).
    accept_markdown_plain: bool = True  # b18: speed NEUTRAL; on as #111's title fix
    #: A thin negotiated-markdown reply (an index page's stub) goes back to
    #: the read, which re-reads the HTML, instead of climbing tiers.
    md_stub_passthrough: bool = True  # b49 WIN (#139)
    #: A new host on the exact address set where >=2 hosts drew a tier-0
    #: 403/429 that tier 1 passed (and none passed tier 0) opens at tier 1.
    addrset_tier_memo: bool = True  # b52 WIN (#140)
    #: SERVFAIL from every one of the machine's own DNS servers ends a lookup.
    dns_servfail_fast: bool = False
    #: An https-upgraded (or memo-redirected) read whose name does not resolve
    #: fails there; the as-asked URL on the same name is not looked up again.
    upgrade_dns_final: bool = True  # b54 WIN (#142)
    #: decisive_404 also for a CDN-tagged 404/410 whose body is no challenge.
    decisive_404_cdn: bool = True  # b55 WIN (#143)
    #: Stack Exchange question pages read through the public API (question + answers).
    lean_read_stackexchange: bool = True  # b56 WIN (#144)
    #: PubMed records read through NCBI E-utilities (plain-text abstract).
    lean_read_pubmed: bool = False  # b57 faster (#145), OFF: robots.txt disallows the lean URL
    #: PMC articles read through NCBI efetch (JATS XML full text).
    lean_read_pmc: bool = False  # b58 faster (#146), OFF: robots.txt disallows the lean URL
    #: Walled publishers' DOI article pages read as Crossref + OpenAlex metadata and abstract.
    lean_read_publisher_doi: bool = True  # b59 WIN (#147)
    #: A failed or near-empty read of a URL naming a DOI answers with its Crossref/OpenAlex record.
    doi_fallback: bool = True  # b60 WIN reliability (#148)
    #: doi_fallback also for SSRN abstract pages (DOI 10.2139/ssrn.<abstract_id>).
    doi_ssrn: bool = True  # b63 WIN (#151, reliability)
    #: gitlab.com project pages read as their raw README.md.
    lean_read_gitlab: bool = False  # b61 faster (#149), OFF: robots.txt disallows the lean URL
    #: bioRxiv/medRxiv preprint pages read through api.biorxiv.org details.
    lean_read_rxiv: bool = True  # b62 WIN (#150)
    lean_read_github_tree: bool = False  # b65 LOSS (#153)
    lean_miss_memo: bool = True  # b66 WIN (#154)
    youtube_watch_details: bool = True  # b67+b70 WIN (#155, content at neutral time)
    doi_direct_memo: bool = True  # b68 WIN (#156)
    blocked_read_memo_s: float = 300.0  # b69 WIN (#157)
    thin_read_memo_s: float = 300.0  # b71 WIN (#158)
    extract_no_comments: bool = False  # b78 NEUTRAL (#162)
    extract_strip_scripts: bool = False  # b79 NEUTRAL (#163)
    #: Listing pages (cars, retail) keep their cards in schema.org JSON-LD or
    #: in per-card JSON attributes; prose extraction drops every one. Append
    #: them to the read as compact rows.
    structured_listings: bool = True  # b80 WIN (#164)
    #: Also read listing records outside JSON-LD: any JSON object in a
    #: ``data-*`` attribute or inline script state that carries a name and a
    #: price or VIN. Recognised by shape; no site's markup is known. Replaced
    #: the site-tuned listing_attrs/listing_embedded (#165/#166) at equal recall.
    listing_generic: bool = True  # b87 WIN (#174)
    #: A listing the prose already names still gets a compact "name | link"
    #: row when its detail link is missing from the text.
    listing_keep_links: bool = True  # b94 WIN (#181)
    #: An item read from a data attribute without a link takes its card's
    #: link: the first same-site href right after the attribute.
    listing_card_href: bool = True  # b95 WIN (#182)
    #: A link-less item whose id is a segment of exactly one same-site link's
    #: path takes that link (data in a script, card link rendered apart).
    listing_id_href: bool = True  # b98 WIN (#185)
    #: A vehicle row carries the record's history summary (accident and owner
    #: counts, title flags, history codes) when its data has one.
    listing_history: bool = True  # b99 WIN (#186)
    #: A listing read states the search's total from the page's own data (or
    #: that it matched nothing), when the page's count fields agree.
    listing_result_count: bool = True  # b101 WIN (#188)
    #: Run tier 2's origin warm-up + one retry when the browser's page is a
    #: bot wall the sidecar did not flag (a 200 "Access Denied"), not only
    #: when the sidecar reports a refusal.
    tier2_warmup_on_wall: bool = False  # b89 NEUTRAL (#176)
    #: Visit the origin before the deep URL on every tier-2 browser pass.
    tier2_origin_first: bool = False
    tier2_origin_dwell_s: float = 1.5
    #: When no general web engine answered a search (walled, degraded, late
    #: or cooling), ask one browser-backed provider before returning.
    search_wall_rescue: bool = True  # b92 WIN (#179)
    #: A host whose read got through only on the challenge browser's rescue
    #: opens its next reads there (for this long), skipping the tiers and the
    #: engine pass that just failed; a failed opener unlearns it.
    learn_rendered_first: bool = False  # b96 NEUTRAL (#183)
    learn_rendered_first_ttl_s: float = 1800.0
    #: The navigation state a real-browser (patchright) tier-2 fetch waits
    #: for before reading: "" = the sidecar's default (load), or
    #: "domcontentloaded" / "commit" / "networkidle".
    tier2_wait_until: str = ""  # b97 NEUTRAL (#184)
    #: Drop click-tracking parameters (utm_*, gclid, msockid, ...) from the
    #: result URLs handed back; content-selecting parameters stay.
    strip_result_trackers: bool = False
    #: Drop search results carrying under this share of the query's
    #: distinctive terms (0 = off); see engine._relevance_gate.
    relevance_gate: float = 0.25  # b81/b82 WIN (#167)
    #: Commercial web queries ("for sale", "price", "near me") skip the
    #: reference providers (Wikipedia).
    commerce_skip_reference: bool = False  # b81 LOSS (#168)
    #: Bing's HTML SERP is fetched with the Chrome-impersonating client
    #: (tier 1): httpx's fingerprint gets a brushed-off answer on hard queries.
    bing_html_tier1: bool = False  # b83 NEUTRAL (#170)
    #: Ask Bing's HTML SERP before its RSS feed (RSS is brushed off too).
    bing_html_first: bool = False  # b83/b85 NEUTRAL (#171)
    #: looks_degraded: "6-speed"-style tokens are not model codes.
    degraded_codes_strict: bool = False  # b83 NEUTRAL (#172)
    #: Semantic Scholar paper pages read through the Graph API.
    lean_read_semantic_scholar: bool = False
    #: A read whose origin moved the path (a redirect or meta refresh that
    #: kept the path's tail) teaches the move; later reads there go straight
    #: to the new place (state_dir/redirects.json). A miss reads as asked.
    redirect_memo: bool = True  # b20 WIN (#116)
    #: Read an http:// URL over https first (the redirect most origins answer
    #: with costs a round trip and a second connection); a miss reads as asked.
    https_upgrade: bool = True  # b21 WIN (#117)
    #: redirect_memo: a whole-host move (every path kept, only the origin
    #: changed: apex -> www) applies after ONE read instead of two.
    redirect_host_once: bool = False
    #: Honor robots Crawl-delay as the minimum gap BETWEEN requests to a host
    #: (arXiv: 15 s) instead of sleeping it before every request.
    crawl_delay_interval: bool = True  # b25 WIN (#122)
    #: After a block, the very next attempt of the same fetch (next tier or
    #: the jar retry) skips the per-domain permit; the AIMD backoff still
    #: governs every later fetch of the host.
    escalate_no_wait: bool = True  # b26 WIN -2509 ms [-2871, -2045]
    #: A tier-2 climb whose sidecar is already known unavailable (sticky)
    #: records it and moves on without waiting for the host's permit.
    skip_dead_tier: bool = True  # b37 WIN -793 ms [-1076, -511] (max_tier 2)
    #: A block that a higher tier then passes in the same climb hands the
    #: host's rate and burst back (less one token): a fingerprint wall is not
    #: a pace signal. A 429 or a climb that fails still backs off.
    wall_refund: bool = True  # b28 WIN -759 ms [-948, -569]
    #: With dead_ip_ttl_s: a host whose every address timed out is skipped
    #: by NAME for the TTL, so rotating DNS answers cannot re-arm the wait.
    dead_host_memo: bool = True  # b31 WIN x2 (-4215, -5550 ms)
    #: host:port whose TCP handshake proves the link is up when this process
    #: has no address that answered yet (a fresh CLI run): only then can a
    #: connect timeout mark a dead address. Empty = never probe a third party.
    link_probe: str = ""
    #: A local sidecar port is pre-checked with a 0.25 s connect, so a
    #: missing sidecar is known in ms, not after Windows' ~1 s refusal.
    sidecar_fast_probe: bool = True  # b37 WIN -782 ms [-814, -749] per cold read
    #: A 2xx thin page or empty JS mount at tier 0 skips tier 1 (it sees the
    #: same shell): straight to tier 2, or the thin fallback.
    thin_skip_tier1: bool = True  # b36 WIN -857 ms [-1135, -589]
    #: A page read stops PDF extraction once it has this many characters
    #: (the read keeps 20 000); such a cut body is never cached. 0 = off.
    pdf_read_budget: int = 24000  # b38 WIN -788 ms [-1341, -378], text identical
    #: A PDF tier 0 already downloaded whole is extracted from those bytes,
    #: not downloaded a second time.
    pdf_reuse_bytes: bool = True  # b39 WIN -148 ms [-249, -56]
    #: dns_race: an NXDOMAIN from one of the machine's own DNS servers fails
    #: the lookup at once instead of waiting out the OS resolver.
    dns_nx_fast: bool = True  # b40 WIN -10019 ms per NXDOMAIN read, neutral live
    #: A plain 404/410 at tier 0 (no wall vendor) ends the climb: the page
    #: does not exist for any client.
    decisive_404: bool = True  # b42 WIN -450 ms [-717, -249]
    #: The CLI `read` command and HTTP /read go through Engine.read (the MCP
    #: tool's path: lean reads, markdown, redirect memo, PDF budget) instead
    #: of a bare ladder fetch + extraction.
    read_via_engine: bool = True  # b48 WIN (#138)
    dns_race: bool = True  # b15 WIN (#110)
    # Extra resolvers for the race, comma-separated IPv4 ("" = system's only).
    dns_race_servers: str = ""
    # Re-ask an expired cache entry conditionally (If-None-Match /
    # If-Modified-Since): a 304 refreshes it without re-downloading the body.
    revalidate_cache: bool = False
    revalidate_max_age_s: float = 7 * 86400

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
