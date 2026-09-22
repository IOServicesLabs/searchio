"""Client for SwarmIO's browser sidecar.

searchio does not implement a stealth browser. SwarmIO already has one --
``crates/swarm-tools/python/browser_sidecar.py``, a patchright-driven Chromium
with bot-wall detection, captcha escalation, persistent profiles, session
injection, generic listing extraction, and an XHR-watching API discovery verb.
Reimplementing that here would be strictly worse than calling it.

The seam is the sidecar's own remote-worker transport: set
``SWARM_SIDECAR_HTTP_PORT`` and it serves the same JSON-RPC verb table over
HTTP instead of stdin/stdout. So this module is a thin, well-behaved client:

* attach to a sidecar someone already started, or spawn one lazily on first use
* spawn either backend: the patchright script (default) or the Rust
  ``se-serve`` engine binary (``binary=`` / ``SEARCHIO_ENGINE_BIN``) -- both
  serve the same JSON-RPC verb table, so callers pick per deployment
* speak the JSON-RPC envelope over ``POST /rpc``
* expose the handful of verbs searchio actually needs as typed methods
* degrade honestly -- if the sidecar cannot start, tier 2 is simply unavailable
  and the ladder stops at tier 1 rather than pretending it failed to fetch

Spawning is lazy on purpose. A browser costs seconds to launch and hundreds of
megabytes to hold, and most searchio queries never need one.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from ..errors import SidecarUnavailable, SidecarVerbError

# Verbs searchio uses. The sidecar exposes ~60; these are the ones that earn
# their place in a search engine.
VERB_FETCH = "fetch"  # HTTP-first, escalates to browser only if needed
VERB_GOTO = "goto"  # keeps a live tab for later interaction
VERB_SEARCH = "search_engine_results"
VERB_EXTRACT = "extract_listings"
VERB_VERIFY = "verify_listings"
VERB_RESEARCH = "research_listings"
VERB_LOCAL = "local_search"
VERB_CRAWL_MANY = "crawl_many"
VERB_DISCOVER_API = "discover_api"
VERB_READ_TEXT = "read_text"
VERB_READ_HTML = "read_html"
VERB_EVAL = "eval"
VERB_HEALTH = "health"


def default_script_path() -> Path | None:
    """Best guess at where browser_sidecar.py lives.

    Checked in order: explicit env var, a sibling SwarmIO checkout, the
    conventional GitHub clone location.
    """
    env = os.environ.get("SEARCHIO_SIDECAR_SCRIPT") or os.environ.get("SWARM_BROWSER_SIDECAR")
    if env and Path(env).is_file():
        return Path(env)
    rel = Path("crates/swarm-tools/python/browser_sidecar.py")
    for base in (
        Path.cwd().parent / "SwarmIO",
        Path.home() / "Documents" / "GitHub" / "SwarmIO",
        Path.home() / "GitHub" / "SwarmIO",
        Path.home() / "src" / "SwarmIO",
    ):
        cand = base / rel
        if cand.is_file():
            return cand
    return None


def default_engine_path() -> Path | None:
    """Best guess at where the Rust engine's se-serve binary lives.

    Checked in order: explicit env var, a well-known per-user install dir
    (``~/.searchio/bin`` -- the Docker image and release installers drop the
    binary here), then a sibling searchio-engine checkout for development.
    Returns None when no binary is findable -- engine mode is an explicit
    opt-in, so a missing binary is a clear configuration error, never a
    silent fallback to patchright.
    """
    env = os.environ.get("SEARCHIO_ENGINE_BIN")
    if env and Path(env).is_file():
        return Path(env)
    name = "se-serve.exe" if os.name == "nt" else "se-serve"
    installed = Path.home() / ".searchio" / "bin" / name
    if installed.is_file():
        return installed
    rel = Path("target") / "debug" / name
    for base in (
        Path.cwd().parent / "searchio-engine",
        Path.home() / "Documents" / "searchio-engine",
        Path.home() / "src" / "searchio-engine",
    ):
        cand = base / rel
        if cand.is_file():
            return cand
    return None


class _StderrTail:
    """Drain a child's stderr for its whole life, keeping only the tail.

    The sidecar script writes its DEBUG/INFO lines to raw stderr (its own
    docstring says so), and the Chromium tree it launches inherits the same
    pipe. With ``stderr=PIPE`` and nobody reading after boot, the pipe
    buffer fills after a few hundred lines and the next write BLOCKS -- the
    whole sidecar wedges mid-run, which surfaces to the ladder as a sidecar
    that stopped answering (bug 99; the shape behind bug 74's ReadError).
    A daemon thread reads continuously and keeps the last ``keep`` lines for
    the death message.
    """

    def __init__(self, proc: subprocess.Popen, keep: int = 200) -> None:
        import collections
        import threading

        self.lines: collections.deque[str] = collections.deque(maxlen=keep)
        self._proc = proc
        self._thread = threading.Thread(target=self._pump, name="sidecar-stderr", daemon=True)
        if proc.stderr is not None:
            self._thread.start()

    def _pump(self) -> None:
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                self.lines.append(raw.decode("utf-8", "replace").rstrip())
        except Exception:  # noqa: BLE001 -- the pipe closing is the end
            pass

    def tail(self, chars: int = 500) -> str:
        return "\n".join(self.lines)[-chars:].strip()


#: Environment names a browser sidecar never needs. Everything the parent
#: process was given -- LLM keys, search-API keys, tokens, proxy passwords
#: -- used to ride into the child and its Chromium tree (bug 100); a
#: sidecar that logs its environment, or a renderer that is compromised,
#: had them all.
_SECRET_ENV_RE = re.compile(
    r"(?:_|^)(?:API_?KEY|APIKEY|KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|AUTH)(?:_|$)",
    re.I,
)
_CHILD_ENV_KEEP_PREFIXES = ("SWARM_", "SE_SERVE_", "PLAYWRIGHT", "PATCHRIGHT", "PYTHON", "NODE")


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The parent's environment minus secrets the sidecar has no use for.

    Kept: everything that is not secret-shaped, plus the sidecar's own
    SWARM_* / SE_SERVE_* knobs even when their names look secret
    (SWARM_SIDECAR_HTTP_TOKEN is the sidecar's token by design).
    """
    src = os.environ if base is None else base
    out: dict[str, str] = {}
    for k, v in src.items():
        if k.upper().startswith(_CHILD_ENV_KEEP_PREFIXES) or not _SECRET_ENV_RE.search(k):
            out[k] = v
    return out


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a spawned sidecar and EVERY process it started.

    ``Popen.terminate`` reaches only the direct child, and on Windows the
    patchright sidecar is a tree -- python.exe -> node cli.js -> chrome.exe
    xN -- so terminating the wrapper leaked the browser (zombie headed
    Chromiums accumulating tabs across runs; the next run then ATTACHES to
    the zombie on the shared challenge port and never owns it, so the tabs
    grew without bound). ``taskkill /T`` walks the tree on Windows; POSIX
    keeps terminate-then-kill on the direct child (a tree kill there needs
    a process group the spawn does not create; the sessions this project
    runs are Windows, so the leak fix targets taskkill).
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _tree_pids(root_pid: int) -> set[int]:
    """Every PID in the process tree under ``root_pid`` (itself included),
    walked from a single CIM snapshot (Windows only; empty elsewhere).

    Taken BEFORE the tree kill, so grandchildren that get re-parented by the
    kill are still known to be ours -- and a stranger that later reuses our
    port is known not to be (bug 102).
    """
    if sys.platform != "win32":
        return {root_pid}
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {root_pid}
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen = {root_pid}
    stack = [root_pid]
    while stack:
        for kid in children.get(stack.pop(), []):
            if kid not in seen:
                seen.add(kid)
                stack.append(kid)
    return seen


def _kill_listener_on_port(port: int, allowed_pids: set[int] | None = None) -> None:
    """Kill the process still LISTENING on ``port`` -- if it is one of ours.

    ``allowed_pids`` is the spawned tree as snapshotted before the tree kill;
    a listener outside it is a stranger that reused the port after our
    process died (bug 102: a span run's shutdown belt killed the NEXT run's
    freshly booted engine on the recycled port) and is left alone. ``None``
    keeps the old unconditional behaviour for callers that have no tree.

    The belt behind close()'s tree kill: netstat -> the owning PID ->
    taskkill /T /F on it. A no-op when nothing listens (the tree kill
    already worked) and when the owner is gone by the time we look. Used
    only for a port we spawned a sidecar on, so it never targets a foreign
    server. POSIX has no equivalent here and does not need one (the sessions
    this runs in are Windows); a miss there is a plain no-op.
    """
    if sys.platform != "win32" or not port:
        return
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    needle = f":{port} "
    pids: set[str] = set()
    for line in out.splitlines():
        if "LISTENING" in line and needle in line:
            pid = line.split()[-1]
            if pid.isdigit() and pid != "0":
                pids.add(pid)
    for pid in pids:
        if allowed_pids is not None and int(pid) not in allowed_pids:
            continue  # a stranger on a recycled port, not our orphan
        try:
            subprocess.run(
                ["taskkill", "/PID", pid, "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            pass


class SidecarClient:
    """Talks to one browser sidecar, starting it if asked to.

    Instances are cheap; the browser is not. Share one client across a run.
    """

    def __init__(
        self,
        *,
        url: str = "",
        script: str = "",
        binary: str | Path | None = None,
        port: int = 8899,
        token: str = "",
        python: str = "",
        proxy: str = "",
        autostart: bool = True,
        boot_timeout_s: float = 120.0,
        request_timeout_s: float = 90.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.script = Path(script) if script else None
        # Engine mode: spawn the Rust se-serve binary instead of the
        # patchright script. Same wire, same verbs, ~100x lighter boot.
        self.binary = Path(binary) if binary else None
        self.port = port
        self.token = token
        self.python = python
        self.proxy = proxy
        self._resolved_python = ""
        self.autostart = autostart
        self.boot_timeout_s = boot_timeout_s
        self.request_timeout_s = request_timeout_s

        self._proc: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._id = 0
        self._ua = ""
        self._start_lock = asyncio.Lock()
        self._unavailable_reason = ""
        self._backend_kind = ""
        # The URL whose backend identity we have already verified via the
        # health verb. Identity is checked once per adopted URL, not per call
        # -- call() runs ensure() for every verb, and a re-probe each time
        # would double the wire traffic.
        self._verified_url = ""
        self._stderr_tail: _StderrTail | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """False once we have concluded the sidecar cannot be reached.

        Sticky by design: after one failed boot, every later tier-2 escalation
        should fail fast instead of re-attempting a 120-second browser launch.
        """
        return not self._unavailable_reason

    async def ensure(self) -> str:
        """Return a live sidecar base URL, starting one if necessary."""
        if self._unavailable_reason:
            raise SidecarUnavailable(self._unavailable_reason)
        if self.url and self._verified_url == self.url:
            # Verified once, trusted until a verb's transport fails (bug 98):
            # every verb used to pay a fresh-client GET /healthz first --
            # double wire traffic and a new connection per call -- although
            # the identity handshake was already remembered.
            return self.url
        async with self._start_lock:
            if self._unavailable_reason:
                # A caller queued behind a boot that FAILED (bug 158): the
                # sticky flag was checked only before the lock, so each
                # waiter launched its own boot with the flag already set.
                raise SidecarUnavailable(self._unavailable_reason)
            if self.url and self._verified_url == self.url:
                return self.url
            if self.url and await self._healthy(self.url):
                if await self._identity_ok(self.url, strict=False):
                    return self.url
                raise SidecarUnavailable(self._unavailable_reason)
            if self.url and not self.autostart:
                self._unavailable_reason = f"no sidecar at {self.url}"
                raise SidecarUnavailable(self._unavailable_reason)

            # Nothing configured or nothing answering: try to spawn one.
            if not self.autostart:
                self._unavailable_reason = "sidecar autostart disabled and no URL configured"
                raise SidecarUnavailable(self._unavailable_reason)

            candidate = f"http://127.0.0.1:{self.port}"
            if await self._healthy(candidate):
                # Something already serves our port. Adopting it sight-unseen
                # is how a patchright-configured client once ended up driving
                # a leftover se-serve: every patchright-only verb then failed
                # deep in provider-land with a bare "Unknown method" (caught
                # by bench/span.py web.expensive_search). Nobody explicitly
                # chose this endpoint, so verify identity strictly and refuse
                # loudly on a mismatch.
                if await self._identity_ok(candidate, strict=True):
                    self.url = candidate
                    return self.url
                raise SidecarUnavailable(self._unavailable_reason)

            if self.binary is not None:
                if not self.binary.is_file():
                    self._unavailable_reason = (
                        f"engine binary not found: {self.binary} -- set "
                        "SEARCHIO_ENGINE_BIN to a built se-serve binary"
                    )
                    raise SidecarUnavailable(self._unavailable_reason)
                await self._spawn_engine(self.binary, candidate)
                # We spawned it, so a mismatch is practically unreachable --
                # but the port could have been stolen between probe and bind.
                if await self._identity_ok(candidate, strict=True):
                    self.url = candidate
                    return self.url
                raise SidecarUnavailable(self._unavailable_reason)

            script = self.script or default_script_path()
            if not script or not Path(script).is_file():
                self._unavailable_reason = (
                    "browser_sidecar.py not found -- set SEARCHIO_SIDECAR_SCRIPT to "
                    "SwarmIO/crates/swarm-tools/python/browser_sidecar.py, or "
                    "SEARCHIO_SIDECAR_URL to a running sidecar"
                )
                raise SidecarUnavailable(self._unavailable_reason)

            await self._spawn(Path(script), candidate)
            if await self._identity_ok(candidate, strict=True):
                self.url = candidate
                return self.url
            raise SidecarUnavailable(self._unavailable_reason)

    def _interpreter(self) -> str:
        """Find a Python that can actually run the sidecar.

        ``sys.executable`` is the obvious choice and the wrong one: searchio
        runs in its own venv, the sidecar needs patchright and Chromium, and
        those live wherever SwarmIO was installed. Launching the sidecar under
        searchio's interpreter fails at import with a bare exit code 2.

        So probe for an interpreter that can import patchright, preferring the
        current one, and remember the answer.
        """
        if self.python:
            return self.python
        if self._resolved_python:
            return self._resolved_python

        candidates = [sys.executable, "python", "python3"]
        for exe in candidates:
            if not exe:
                continue
            try:
                r = subprocess.run(
                    [exe, "-c", "import patchright, sys; sys.exit(0)"],
                    capture_output=True,
                    timeout=60,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode == 0:
                self._resolved_python = exe
                return exe
        # Nothing has patchright. Return the current interpreter so the spawn
        # attempt produces a real import error the user can act on, rather than
        # us guessing at a fix.
        self._resolved_python = sys.executable
        return sys.executable

    async def _spawn(self, script: Path, expect_url: str) -> None:
        env = child_env()
        env["SWARM_SIDECAR_HTTP_PORT"] = str(self.port)
        env["SWARM_SIDECAR_HTTP_HOST"] = "127.0.0.1"
        if self.token:
            env["SWARM_SIDECAR_HTTP_TOKEN"] = self.token
        if self.proxy:
            # The sidecar reads its egress proxy from the environment at
            # launch, so this only takes effect on a sidecar we start. One
            # we attached to keeps whatever egress it was started with.
            env["SWARM_BROWSER_PROXY"] = self.proxy
        # The sidecar imports _sidecar_kernel from its own directory.
        env["PYTHONPATH"] = str(script.parent) + os.pathsep + env.get("PYTHONPATH", "")

        exe = await asyncio.to_thread(self._interpreter)
        # Keep stderr. A dead sidecar reported only as "exit code 2" is
        # undiagnosable, and the reason is almost always one line of traceback.
        self._proc = subprocess.Popen(
            [exe, str(script)],
            cwd=str(script.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        self._stderr_tail = _StderrTail(self._proc)

        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                tail = self._stderr_tail.tail()
                self._unavailable_reason = (
                    f"sidecar exited during boot (code {self._proc.returncode}, "
                    f"interpreter {exe})" + (f": {tail}" if tail else "")
                )
                raise SidecarUnavailable(self._unavailable_reason)
            if await self._healthy(expect_url):
                return
            await asyncio.sleep(1.0)
        self._unavailable_reason = f"sidecar did not become healthy within {self.boot_timeout_s}s"
        raise SidecarUnavailable(self._unavailable_reason)

    async def _spawn_engine(self, binary: Path, expect_url: str) -> None:
        """Launch the Rust se-serve binary on our port and wait for health.

        Differences from the patchright spawn, honestly stated: no interpreter
        probe (it is a native exe), no token enforcement (se-serve does not
        check Authorization -- pass one only for upstream middleware), and no
        egress-proxy variable (the engine's client does not read one yet; it
        picks up the system proxy configuration like reqwest's defaults).
        SE_SERVE_SESSION already present in the environment rides through, so
        an externally managed session file still applies to a client-spawned
        engine. Note also that close() terminates the process, which on
        Windows is a hard kill: the engine's graceful session re-save is a
        Ctrl+C path, so on-disk persistence in engine mode is driven
        explicitly through the session_save verb, not implied by close().
        """
        env = child_env()
        env["SE_SERVE_PORT"] = str(self.port)

        self._proc = subprocess.Popen(
            [str(binary), str(self.port)],
            cwd=str(binary.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        self._stderr_tail = _StderrTail(self._proc)

        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                tail = self._stderr_tail.tail()
                self._unavailable_reason = (
                    f"engine exited during boot (code {self._proc.returncode})"
                    + (f": {tail}" if tail else "")
                )
                raise SidecarUnavailable(self._unavailable_reason)
            if await self._healthy(expect_url):
                return
            await asyncio.sleep(0.5)
        self._unavailable_reason = f"engine did not become healthy within {self.boot_timeout_s}s"
        raise SidecarUnavailable(self._unavailable_reason)

    async def _healthy(self, base: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{base}/healthz")
                return r.status_code == 200 and r.json().get("ok") is True
        except Exception:
            return False

    def _expected_kind(self) -> str:
        """The backend this client is configured to drive."""
        return "engine" if self.binary is not None else "patchright"

    async def _identify(self, base: str) -> str:
        """The backend's self-reported kind: ``"engine"``, ``"patchright"``, or
        ``""`` when the responder does not name itself.

        Both backends name themselves in the health verb's ``engine`` field
        (se-serve reports ``searchio-engine``; the sidecar script reports
        ``patchright``/``playwright``). This deliberately bypasses
        :meth:`call` -- ``call`` goes through :meth:`ensure`, and ``ensure``
        is exactly where this probe is needed, so routing it through ``call``
        would deadlock on the start lock.
        """
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.post(
                    f"{base}/rpc",
                    content=json.dumps(
                        {"jsonrpc": "2.0", "id": 0, "method": VERB_HEALTH, "params": {}}
                    ),
                    headers=headers,
                )
            result = r.json().get("result") or {}
            eng = result.get("engine") or ""
            if eng == "searchio-engine":
                return "engine"
            return "patchright" if eng else ""
        except Exception:
            return ""

    async def _identity_ok(self, base: str, *, strict: bool) -> bool:
        """True when the healthy responder at ``base`` may be adopted; on a
        refusal, records the precise reason and returns False.

        ``/healthz`` alone cannot discriminate: se-serve answers it with a
        bare ``{"ok": true}``, byte-compatible with the sidecar script, so a
        stray engine on the patchright port (or vice versa) passes the health
        check and then fails every backend-specific verb with an opaque wire
        error. The health verb's self-naming is the handshake.

        ``strict`` applies to endpoints nobody explicitly chose -- the
        default-port probe and the post-spawn check -- where the responder
        must be the backend this client is configured for. An explicit
        ``url=`` is the operator's choice: any self-naming backend is adopted
        and remembered, and consumers branch on :meth:`backend_kind` (an
        engine behind an explicit URL is how the whole wire-parity suite
        drives it). A responder that does not name itself is refused on
        either path -- both known backends self-name.
        """
        if base == self._verified_url:
            return True
        kind = await self._identify(base)
        if not kind:
            self._unavailable_reason = (
                f"{base} answers /healthz but does not name itself via the "
                "health verb; refusing to attach to an unidentified server -- "
                "both known backends (patchright script, se-serve) self-name"
            )
            return False
        if not strict:
            self._verified_url = base
            self._backend_kind = kind
            return True
        want = self._expected_kind()
        if kind == want:
            self._verified_url = base
            self._backend_kind = kind
            return True
        if kind == "engine":
            hint = (
                "stop that process, or set SEARCHIO_SIDECAR_ENGINE=1 to drive "
                "the engine backend on purpose"
            )
        else:
            hint = (
                "stop that process, or unset SEARCHIO_SIDECAR_ENGINE to drive "
                "the patchright sidecar"
            )
        self._unavailable_reason = (
            f"{base} already serves the {kind} backend, but this client is "
            f"configured for {want} -- refusing to attach to the wrong "
            f"sidecar; {hint}"
        )
        return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        # Only kill a sidecar we started ourselves. One we attached to belongs
        # to someone else and may be serving other callers.
        if self._proc is not None:
            # Snapshot OUR tree before the kill (bug 102): the belt below may
            # only reap PIDs that were ours, never a stranger that reused
            # the port after we died.
            ours = _tree_pids(self._proc.pid) if self._proc.poll() is None else {self._proc.pid}
            if self._proc.poll() is None:
                _kill_tree(self._proc)
            # Verify the reap by PORT, not just by trusting the tree walk. A
            # span run occasionally left a challenge sidecar alive after a
            # clean close() (iteration 32: PID 39388/59800 held the last
            # tool-check page) even though the same spawn reaps cleanly in
            # isolation -- an intermittent taskkill/tree-walk miss (a chromium
            # renderer racing the walk, a grandchild re-parented at the wrong
            # instant). The root cause stayed unconfirmed, so this is a belt:
            # if our own port still answers after the tree kill, kill whatever
            # holds it. Gated on self._proc (a sidecar WE spawned) so a truly
            # adopted foreign sidecar is never touched -- free_port uniqueness
            # keeps every ephemeral bench sidecar on this spawned side.
            _kill_listener_on_port(self.port, ours)
            self._proc = None

    # ── RPC ──────────────────────────────────────────────────────────────────

    async def call(self, verb: str, params: dict[str, Any], *, timeout: float | None = None) -> dict:
        """Invoke one sidecar verb. Raises SidecarUnavailable on transport failure."""
        base = await self.ensure()
        if self._client is None:
            # Under the start lock (iteration 52 rider): a bare `is None`
            # check let two concurrent first verbs build two clients, one of
            # which leaked its pool for the life of the process.
            async with self._start_lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=self.request_timeout_s)
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": verb, "params": params}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            r = await self._client.post(
                f"{base}/rpc",
                content=json.dumps(payload),
                headers=headers,
                timeout=timeout or self.request_timeout_s,
            )
        except Exception as exc:
            # The next ensure() re-probes health before trusting this URL.
            self._verified_url = ""
            raise SidecarUnavailable(f"{verb}: {type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            self._verified_url = ""
            raise SidecarUnavailable(f"{verb}: HTTP {r.status_code}")
        try:
            body = r.json()
        except ValueError as exc:
            # A 200 that is not JSON is not the sidecar answering (a proxy
            # page, a half-written body): SidecarUnavailable is the class
            # the tier loop classifies; a raw JSONDecodeError took the fetch
            # down unclassified (bug 97).
            raise SidecarUnavailable(
                f"{verb}: malformed response ({r.headers.get('content-type', '?')}, "
                f"{len(r.content)} bytes): {r.text[:80]!r}") from exc
        if not isinstance(body, dict):
            raise SidecarUnavailable(f"{verb}: malformed response: {type(body).__name__} envelope")
        if "error" in body and body["error"]:
            err = body["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            # A verb-level error is the sidecar working correctly and telling us
            # the call failed, so it is a result, not a transport problem.
            return {"ok": False, "error": msg}
        result = body.get("result")
        return result if isinstance(result, dict) else {"ok": False, "error": "malformed_result"}

    async def backend_kind(self) -> str:
        """Which backend answers on this wire: ``"engine"``, ``"patchright"``,
        or ``""`` when it cannot be determined.

        The health verb names its implementation on both backends (the Rust
        engine reports ``engine="searchio-engine"``; the sidecar script
        reports ``"patchright"``/``"playwright"``), so one probe discriminates
        — including for url-connected clients, where the process behind the
        port is whoever the operator started. A client in ``binary`` mode is
        the engine without asking, and any answer is cached: consumers like
        the marketplace provider branch on this every call.
        """
        if self.binary is not None:
            return "engine"
        if self._backend_kind:
            return self._backend_kind
        try:
            h = await self.call("health", {})
        except Exception:
            return ""
        eng = h.get("engine") or ""
        self._backend_kind = "engine" if eng == "searchio-engine" else ("patchright" if eng else "")
        return self._backend_kind

    # ── typed verbs ──────────────────────────────────────────────────────────

    async def fetch(self, url: str, *, tab_id: str = "default", timeout: float | None = None,
                    timeout_ms: int | None = None) -> dict:
        """HTTP-first page fetch that escalates to the browser only if needed.

        ``timeout`` bounds the RPC call itself; ``timeout_ms`` is forwarded to
        the verb as the PER-ATTEMPT navigation budget (the patchright sidecar's
        own default is 30000). The ladder passes a third of its tier budget so
        the agent-visible budget actually bounds the browser's navigation --
        and so the soften-and-serve path (a second, degraded attempt plus the
        DOM reads) still fits inside the call window.
        """
        params: dict[str, Any] = {"url": url, "tab_id": tab_id}
        if timeout_ms is not None:
            params["timeout_ms"] = timeout_ms
        return await self.call(VERB_FETCH, params, timeout=timeout)

    async def search(self, query: str, *, engine: str = "auto", k: int = 10,
                     tab_id: str = "default") -> dict:
        return await self.call(
            VERB_SEARCH,
            {"query": query, "engine": engine, "max_results": k, "tab_id": tab_id},
        )

    async def goto(self, url: str, *, tab_id: str = "default",
                   timeout_ms: int | None = None) -> dict:
        params: dict[str, Any] = {"url": url, "tab_id": tab_id}
        if timeout_ms is not None:
            params["timeout_ms"] = timeout_ms
        return await self.call(VERB_GOTO, params)

    async def read_html(self, *, tab_id: str = "default", selector: str = "") -> str:
        """The tab's DOM as it stands right now.

        Distinct from :meth:`fetch`, which returns the *response* body. After a
        page has been scrolled and has lazy-loaded more of itself, the response
        body is the shell it started as and the DOM is what it has become, so
        anything driving a page before reading it needs this one.
        """
        params: dict[str, Any] = {"tab_id": tab_id}
        if selector:
            params["selector"] = selector
        r = await self.call(VERB_READ_HTML, params)
        if r.get("ok") is False:
            # The sidecar said the read failed (bug 101): "" here became
            # "no listings" -> "empty market" two callers up.
            raise SidecarVerbError(f"read_html[{tab_id}]: {r.get('error') or 'verb failed'}")
        html = r.get("html")
        return html if isinstance(html, str) else ""

    async def eval_js(self, js: str, *, tab_id: str = "default",
                      timeout: float | None = None) -> Any:
        """Run JavaScript in a live tab and return its value.

        A promise is awaited by the driver, so an expression of the form
        ``(async () => { ... })()`` can wait for the page to settle and report
        what it found -- which is how a scroll loop knows whether it is still
        making progress.
        """
        r = await self.call(VERB_EVAL, {"tab_id": tab_id, "js": js}, timeout=timeout)
        return r.get("result")

    async def extract_listings(self, *, tab_id: str = "default", max_items: int = 40) -> dict:
        return await self.call(VERB_EXTRACT, {"tab_id": tab_id, "max_items": max_items})

    async def verify_listings(self, urls: list[str], *, fields: list[str] | None = None) -> dict:
        return await self.call(
            VERB_VERIFY,
            {"urls_json": json.dumps(urls), "extract_fields": fields or []},
            timeout=300.0,
        )

    async def research_listings(self, query: str, **kw: Any) -> dict:
        return await self.call(VERB_RESEARCH, {"query": query, **kw}, timeout=600.0)

    async def local_search(self, query: str, location: str) -> dict:
        return await self.call(VERB_LOCAL, {"query": query, "location": location}, timeout=300.0)

    async def crawl_many(self, urls: list[str], *, concurrency: int = 4) -> dict:
        return await self.call(
            VERB_CRAWL_MANY, {"urls": urls, "concurrency": concurrency}, timeout=600.0
        )

    async def cookies(self) -> list[dict]:
        """The browser's current cookie jar."""
        r = await self.call("cookies_get", {})
        return r.get("cookies") or []

    async def user_agent(self) -> str:
        """The browser's own User-Agent.

        Needed verbatim: a cf_clearance cookie is issued against the UA that
        solved the challenge and is refused under any other, so replaying the
        cookie without this is worse than not replaying it.
        """
        if self._ua:
            return self._ua
        try:
            r = await self.call("eval", {"js": "navigator.userAgent"})
        except Exception:
            return ""
        # The verb reports ok=False when the tab sits at about:blank, which is
        # the right call for page-scraping scripts and irrelevant here --
        # navigator.userAgent is a property of the browser, not of whatever is
        # loaded. So read `result` regardless of the ok flag.
        ua = r.get("result")
        self._ua = ua.strip() if isinstance(ua, str) else ""
        return self._ua

    async def discover_api(self, url: str, *, actions: list | None = None, save: bool = True) -> dict:
        """Learn a site's own JSON API by driving its UI and watching XHR.

        Worth reaching for whenever a host will be queried repeatedly: an
        endpoint the page itself calls is cheaper, more stable, and far less
        likely to be challenged than scraping the rendered HTML.
        """
        return await self.call(
            VERB_DISCOVER_API,
            {"url": url, "actions": actions or [], "save": save},
            timeout=300.0,
        )
