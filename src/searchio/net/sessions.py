"""Seeded browser sessions: the platform's login jars, attached only on a wall.

The cloud (Swarmio_cloud, ``src/searchio_sessions.rs``) exports signed-in
cookie jars -- Playwright ``storage_state`` files plus a ``swarmio`` metadata
block -- into a directory it shares with searchio, and drains rotated cookies
back out of ``writeback/``. ``docs/SESSIONS.md`` is the binding contract; the
rules that matter for reading this module:

* **Attach on a wall, not by default.** A shared logged-in identity that
  touches every query is how an account gets rate-limited and locked. The
  ladder fetches anonymously first and only reaches for a jar when the site
  actually refuses the anonymous fetch -- and then retries the SAME cheap
  tier before anyone boots a browser.
* **One identity per logged-in site.** The platform captures cookies only,
  never the user agent, so the UA pin is ours to set: freeze one persona UA
  per site on first use and keep every jar-carrying request on it. The
  persona machinery steps aside for those hosts (the same discipline
  ``clearance.py`` pins on challenge cookies).
* **Write rotated cookies back.** A jar that is read but never refreshed
  dies in days; the cloud re-seals whatever lands in ``writeback/``.
* **Never log values.** Filenames, domains, cookie NAMES, and counts only --
  in logs, in errors, in stats.

This store is deliberately separate from ``clearance.py``: a login jar is
not clearance, has a different lifetime, and must never be replayed by logic
that assumes "this cookie proves we passed a challenge"
(``test_every_known_cookie_is_a_challenge_cookie`` is the guard). It is also
deliberately JSON-not-sqlite: the contract's shape is cloud-owned files, and
nothing here needs a query language.

Per-tenant isolation is NOT solved here (contract section 4): these are
platform accounts, so any run -- tenant-scoped or not -- that trips a wall
on that site uses the platform jar. The natural future seam is the
``session=`` tenant id ``/read`` already takes: key the store on
``(tenant, domain)`` and keep the platform jars as the fallback tenant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

from . import persona as persona_mod

log = logging.getLogger(__name__)

#: Cap mirroring the cloud's validation (SESSION_MAX_COOKIES in the drain
#: path): a jar above it would be dropped whole, so don't bother writing it.
_MAX_COOKIES = 200

#: Public suffixes a cookie may never be scoped to (RFC 6265 s5.3 rule 5) --
#: the same table ``Ladder._HopCookies`` validates redirect-hop cookies
#: against (bug 91). sessions.py cannot import ladder (ladder imports
#: sessions), so the table lives in both places; a test pins them equal.
PUBLIC_SUFFIXES: frozenset[str] = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "me.uk", "ltd.uk", "plc.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "co.nz", "org.nz",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp", "co.kr", "or.kr",
    "com.br", "net.br", "org.br", "com.cn", "net.cn", "org.cn", "com.tw",
    "co.in", "net.in", "org.in", "com.mx", "com.tr", "com.ar", "com.sg",
    "co.za", "com.hk", "com.my", "co.id", "com.ph", "com.vn", "com.eg",
    "github.io", "herokuapp.com", "azurewebsites.net", "cloudfront.net",
    "amazonaws.com", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
})


def _domain_matches(host: str, domain: str) -> bool:
    """The rule a browser follows: a parent-domain cookie is valid for a
    child; a child-scoped cookie is NOT valid for the parent."""
    return host == domain or host.endswith("." + domain)


def _is_public_suffix(domain: str) -> bool:
    return "." not in domain or domain in PUBLIC_SUFFIXES


def _norm_domain(domain: str) -> str:
    """Cookie domains compare dot- and www.-stripped (bug 155's rule)."""
    return domain.lstrip(".").removeprefix("www.").lower()


def _expired(rec: dict, now: float) -> bool:
    """Playwright convention: expires is SECONDS, -1/None = session cookie."""
    ex = rec.get("expires")
    return ex is not None and ex >= 0 and ex <= now


def _sendable(rec: dict, host: str, scheme: str, now: float) -> bool:
    """Would a browser attach this record to (host, scheme) right now?"""
    if _expired(rec, now):
        return False
    if rec.get("secure") and scheme != "https":
        return False
    dom = str(rec.get("domain") or "")
    if rec.get("host_only"):
        return host == dom
    return bool(dom) and _domain_matches(host, dom)


def _match_records(cookies: list[dict], want: str) -> list[dict]:
    """Full records a browser would send to ``want`` -- the ``relevant_cookies``
    domain rule (clearance.py) with the RETAINED_COOKIES name filter REMOVED:
    a login jar is whatever the site set, and dropping "unknown" names breaks
    sessions in ways that look like a dead account."""
    out: list[dict] = []
    want = want.removeprefix("www.")
    for c in cookies or []:
        name = str(c.get("name") or "")
        if not name:
            continue
        cdom = _norm_domain(str(c.get("domain") or ""))
        # A cookie with no domain belongs nowhere (relevant_cookies, bug 155).
        if not cdom or not (want == cdom or want.endswith("." + cdom)):
            continue
        out.append(c)
    return out


def _site_records(cookies: list[dict], key: str) -> list[dict]:
    """Every record scoped to the site's tree (the key or anything under it)
    -- the browser read-back's write-back scope. Unlike ``_match_records``
    (the send rule, child-host looking UP at parents), this keeps child-scope
    records too: the site logged in on a subdomain and that cookie belongs
    in the site's jar."""
    out: list[dict] = []
    for c in cookies or []:
        name = str(c.get("name") or "")
        if not name:
            continue
        cdom = _norm_domain(str(c.get("domain") or ""))
        if cdom and (cdom == key or cdom.endswith("." + key)):
            out.append(c)
    return out


def _parse_set_cookie(header: str, setting_host: str) -> dict | None:
    """One Set-Cookie value -> an internal snake_case record, a tombstone
    (``{"delete": True}``) for an expired/Max-Age<=0 deletion, or None for
    garbage (never raises -- a cookie is never worth a fetch).

    Domain validation mirrors ``_HopCookies.set_cookie``: the setter may only
    scope to its own host or a parent, never a public suffix (bug 91).
    """
    try:
        segs = header.split(";")
        name, eq, value = segs[0].partition("=")
        name, value = name.strip(), value.strip()
        if not name or not eq:
            return None
        host = setting_host.lower()
        domain_attr, secure, http_only = "", False, False
        max_age: int | None = None
        expires_at: float | None = None
        path = "/"
        for attr in segs[1:]:
            k, _, v = attr.partition("=")
            kl = k.strip().lower()
            if kl == "domain":
                domain_attr = v.strip().lstrip(".").lower()
            elif kl == "secure":
                secure = True
            elif kl == "httponly":
                http_only = True
            elif kl == "path" and v.strip():
                path = v.strip()
            elif kl == "max-age":
                try:
                    max_age = int(v.strip())
                except ValueError:
                    max_age = None  # s5.2: ignore unparsable attributes
            elif kl == "expires":
                try:
                    dt = parsedate_to_datetime(v.strip())
                    expires_at = dt.timestamp() if dt else None
                except (TypeError, ValueError, OverflowError):
                    expires_at = None  # unparseable -> attribute ignored
        if domain_attr:
            if not _domain_matches(host, domain_attr) or _is_public_suffix(domain_attr):
                return None
            scope, host_only_flag = domain_attr, False
        else:
            scope, host_only_flag = host, True
        if max_age is not None:
            if max_age <= 0:
                return {"name": name, "delete": True, "domain": scope,
                        "host_only": host_only_flag}
            expires: int | None = int(time.time()) + max_age
        elif expires_at is not None:
            if expires_at <= time.time():
                return {"name": name, "delete": True, "domain": scope,
                        "host_only": host_only_flag}
            expires = int(expires_at)
        else:
            expires = None
        return {"name": name, "value": value, "domain": scope, "path": path,
                "expires": expires, "secure": secure, "http_only": http_only,
                "host_only": host_only_flag}
    except Exception:  # noqa: BLE001 -- a cookie is never worth a fetch
        return None


def _normalize_playwright(cookies: list[dict]) -> list[dict]:
    """Playwright ``storage_state`` cookie records (``httpOnly`` camelCase)
    -> the internal snake_case shape. The cloud's exporter speaks Playwright;
    accepting the vault's snake_case spelling too costs one ``or``."""
    out: list[dict] = []
    for c in cookies or []:
        name = str(c.get("name") or "")
        if not name:
            continue
        raw_dom = str(c.get("domain") or "").lower()
        try:
            expires = c.get("expires")
            expires = None if expires in (None, -1) else int(float(expires))
        except (TypeError, ValueError):
            expires = None
        out.append({
            "name": name,
            "value": str(c.get("value") or ""),
            "domain": _norm_domain(raw_dom),
            "path": str(c.get("path") or "/") or "/",
            "expires": expires,
            "secure": bool(c.get("secure")),
            "http_only": bool(c.get("httpOnly") or c.get("http_only")),
            "host_only": bool(raw_dom) and not raw_dom.startswith("."),
        })
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    """tmp file in the same directory + os.replace; chmod 0600 to match the
    directory discipline (contract section 2)."""
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


@dataclass
class SiteSession:
    """One site's planted jar: a parsed cloud session file."""

    key: str                        # registrable-ish key (first cookie domain)
    path: Path                      # source file (mtime watch + notes)
    mtime: float
    platform: str                   # write-back address from the swarmio block
    label: str
    cookies: list[dict]             # internal snake_case records, NO name filter
    planted_names: frozenset[str]   # names present at load (logged-out guard)
    ua: str = ""                    # pinned UA, frozen by the store on first use
    _wb_hash: str = field(default="", repr=False)  # last write-back content hash

    def names(self) -> frozenset[str]:
        return frozenset(str(c.get("name") or "") for c in self.cookies)


class SessionStore:
    """The cloud's session directory, read-mostly and never-raise.

    Absent directory (or unset ``sessions_dir``) means "no sessions" -- a
    silent no-op, never an error (contract section 4, acceptance #1). The
    cloud rewrites a file only when its content changes, so an mtime move is
    a real change: ``for_host`` re-scans the directory cheaply (a few files)
    and re-parses only moved mtimes. A file that disappears means the session
    is gone and must stop being used within one refresh (acceptance #5).

    Errors are counted, logged with filenames only, and never raised into a
    fetch -- same degradation discipline as ``ClearanceStore``.
    """

    def __init__(self, sessions_dir: str, state_dir: Path) -> None:
        self.errors = 0
        self._lock = threading.RLock()
        self._dir = Path(sessions_dir) if sessions_dir else None
        self._sessions: dict[str, SiteSession] = {}   # key -> session
        self._files: dict[Path, tuple[float, str]] = {}  # path -> (mtime, key)
        self._key_owner: dict[str, Path] = {}  # key -> the file that owns it
        self._pins_path = Path(state_dir) / "session_pins.json"
        try:
            self._pins: dict[str, str] = json.loads(self._pins_path.read_text())
            if not isinstance(self._pins, dict):
                self._pins = {}
        except (OSError, ValueError):
            self._pins = {}

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    @property
    def sites(self) -> int:
        return len(self._sessions)

    # ── loading ──────────────────────────────────────────────────────────────

    def for_host(self, dom: str) -> SiteSession | None:
        """The session whose jar covers ``dom``, refreshing the directory
        view first. Cookie scope rules ignore ports (RFC 6265) even though
        the ladder's ``domain_of`` keeps them for pacing identity, so a
        ``host:port`` lookup matches the bare-host jar. Exact key match wins;
        otherwise the first session (in sorted-filename order) whose jar
        domain-matches -- deterministic, so same-jar collisions (x.com vs
        twitter.com) resolve either way."""
        if self._dir is None:
            return None
        with self._lock:
            self._rescan()
            dom = dom.removeprefix("www.")
            if ":" in dom:  # host:port -> host (IPv6 literals land on exact-key miss)
                dom = dom.rsplit(":", 1)[0]
            hit = self._sessions.get(dom)
            if hit is not None:
                return hit
            for key in sorted(self._sessions):
                if _domain_matches(dom, key):
                    return self._sessions[key]
            return None

    def _rescan(self) -> None:
        """Re-read the directory: new/changed files re-parsed (a CHANGED file
        replaces its session even at the same key -- the cloud rotates cookie
        values in place), deleted files dropped. Called under the lock."""
        try:
            on_disk: dict[Path, float] = {}
            for p in sorted(self._dir.glob("*.json")):
                try:
                    on_disk[p] = p.stat().st_mtime
                except OSError:
                    self.errors += 1
        except OSError:
            # Directory vanished (volume unmounted?): everything stops
            # matching until it comes back -- "no sessions", not an error.
            self._sessions.clear()
            self._files.clear()
            self._key_owner.clear()
            return
        for path in list(self._files):
            if path not in on_disk:
                _, key = self._files.pop(path)
                # Note: every rescan globs NEW Path objects -- compare with ==,
                # identity would never match across scans.
                if self._key_owner.get(key) == path:
                    self._key_owner.pop(key, None)
                    self._sessions.pop(key, None)
                    log.info("sessions: %s gone; jar dropped", path.name)
                    # A colliding file may still hold this key: force its
                    # re-parse so the key is not lost until its mtime moves.
                    for other, (_, k2) in list(self._files.items()):
                        if k2 == key:
                            self._files.pop(other)
        for path, mtime in on_disk.items():
            prev = self._files.get(path)
            if prev is not None and prev[0] == mtime:
                continue
            sess = self._parse(path, mtime)
            if sess is None:
                if prev is not None:
                    _, old_key = prev
                    self._files.pop(path, None)
                    if self._key_owner.get(old_key) == path:
                        self._key_owner.pop(old_key, None)
                        self._sessions.pop(old_key, None)
                continue
            self._files[path] = (mtime, sess.key)
            if prev is not None:
                _, old_key = prev
                if old_key != sess.key and self._key_owner.get(old_key) == path:
                    self._key_owner.pop(old_key, None)
                    self._sessions.pop(old_key, None)
            # First file for a key owns it; a later collision keeps its mtime
            # bookmark so it is not re-parsed every scan, but never displaces.
            owner = self._key_owner.get(sess.key)
            if owner is None or owner == path:
                self._key_owner[sess.key] = path
                self._sessions[sess.key] = sess

    def _parse(self, path: Path, mtime: float) -> SiteSession | None:
        try:
            doc = json.loads(path.read_text())
            if not isinstance(doc, dict):
                raise ValueError("not an object")
            raw = doc.get("cookies")
            if not isinstance(raw, list):
                raise ValueError("no cookies list")
            cookies = _normalize_playwright(raw)
            if not cookies:
                raise ValueError("empty jar")
        except (OSError, ValueError) as e:
            self.errors += 1
            log.warning("sessions: skipping unreadable %s (%s)", path.name,
                        type(e).__name__)
            return None
        swarmio = doc.get("swarmio") if isinstance(doc.get("swarmio"), dict) else {}
        key = next((str(c["domain"]) for c in cookies if c.get("domain")), "")
        if not key:
            key = path.stem
        sess = SiteSession(
            key=key,
            path=path,
            mtime=mtime,
            platform=str(swarmio.get("platform") or ""),
            label=str(swarmio.get("label") or ""),
            cookies=cookies,
            planted_names=frozenset(c["name"] for c in cookies),
        )
        # Carry a frozen pin across restarts; the cloud rewrites its file
        # but never our pin store.
        sess.ua = str(self._pins.get(key) or "")
        return sess

    # ── attaching ────────────────────────────────────────────────────────────

    def cookie_header(self, sess: SiteSession, host: str, scheme: str) -> str:
        """The Cookie header value a browser would send right now: expired
        dropped, Secure kept off http, domain-matched. Names and values only
        -- this string IS a credential and never gets logged."""
        now = time.time()
        host = host.lower().removeprefix("www.")
        return "; ".join(
            f"{c['name']}={c['value']}"
            for c in sess.cookies
            if _sendable(c, host, scheme, now)
        )

    def attachable(self, sess: SiteSession, url: str) -> bool:
        """A jar attaches only over https (login credentials, like clearance,
        never ride plaintext) and only when it actually has a live cookie
        for this URL."""
        parts = urlsplit(url)
        if parts.scheme != "https":
            return False
        return bool(self.cookie_header(sess, parts.hostname or "", parts.scheme))

    def pinned_ua(self, sess: SiteSession) -> str:
        """The site's ONE identity UA (contract 3.4). The platform captures
        cookies only, so nothing is inherited: freeze the deterministic
        persona choice for this key on first use, persist it beside our
        state (NEVER in the cloud's file -- they rewrite theirs), and reuse
        it for every jar-carrying request. A jar arriving under a rotating
        fingerprint reads as stolen, which is worse than no jar at all."""
        if sess.ua:
            return sess.ua
        with self._lock:
            if not sess.ua:
                ua = persona_mod.for_domain(sess.key).ua
                sess.ua = ua
                self._pins[sess.key] = ua
                try:
                    _atomic_write(self._pins_path,
                                  json.dumps(self._pins, indent=1).encode())
                except OSError:
                    self.errors += 1
                log.info("sessions: pinned one UA for %s (%s)", sess.key,
                         ua.split(")", 1)[0].split("(")[-1][:40])
            return sess.ua

    # ── rotation and write-back ──────────────────────────────────────────────

    def apply_response(self, sess: SiteSession, set_cookie_headers: list[str],
                       host: str, scheme: str) -> bool:
        """Fold a jar-carrying response's Set-Cookie list into the jar.
        Only names ALREADY in the jar update (contract 3.5 -- a minimal
        parse is not the place to admit new identity cookies); a tombstone
        deletes. Returns True if any jar name was touched."""
        del scheme  # scope decisions ride on the setting host; kept for symmetry
        touched = False
        with self._lock:
            for header in set_cookie_headers or []:
                rec = _parse_set_cookie(header, host)
                if rec is None:
                    continue
                name = rec["name"]
                kept = [c for c in sess.cookies
                        if not (c.get("name") == name
                                and _domain_matches(host, str(c.get("domain") or "")))]
                if rec.get("delete"):
                    if len(kept) != len(sess.cookies):
                        sess.cookies = kept
                        touched = True
                    continue
                if not any(c.get("name") == name for c in sess.cookies):
                    continue  # unknown name: not ours to grow (cheap tiers)
                sess.cookies = kept + [rec]
                touched = True
        if touched:
            log.info("sessions: %s rotated (%d names)", sess.key,
                     len(sess.cookies))
        return touched

    def site_records(self, sess: SiteSession) -> list[dict]:
        with self._lock:
            return [dict(c) for c in sess.cookies]

    def replace_site_records(self, sess: SiteSession, records: list[dict]) -> None:
        """The browser jar is ground truth: after a tier-2 pass the site's
        records are REPLACED wholesale (new names included -- unlike the
        cheap tiers' known-names-only rule)."""
        with self._lock:
            sess.cookies = [dict(c) for c in records]

    def site_cookies_from_state(self, sess: SiteSession, state: dict) -> list[dict]:
        """Site-scoped records out of a ``storage_state_get`` artifact. The
        artifact is the WHOLE session jar -- other sites' cookies must never
        leak into this site's write-back, so the site-tree filter applies
        before anything is compared or written."""
        if not isinstance(state, dict):
            return []
        raw = state.get("cookies")
        if not isinstance(raw, list):
            return []
        now = time.time()
        return [c for c in _site_records(_normalize_playwright(raw), sess.key)
                if not _expired(c, now)]

    def playwright_records(self, sess: SiteSession) -> list[dict]:
        """The site's LIVE records as Playwright ``storage_state`` cookie
        dicts (``httpOnly`` camelCase; ``expires`` omitted for session
        cookies) -- the shape ``storage_state_set`` takes verbatim. Expired
        records are pre-filtered: ``add_cookies`` warns per file on those,
        and a dead cookie is no way to open a pass."""
        now = time.time()
        out: list[dict] = []
        with self._lock:
            for c in sess.cookies:
                if _expired(c, now):
                    continue
                rec = {"name": c["name"], "value": c["value"],
                       "domain": c["domain"], "path": c.get("path") or "/",
                       "secure": bool(c.get("secure")),
                       "httpOnly": bool(c.get("http_only"))}
                if c.get("expires") is not None:
                    rec["expires"] = c["expires"]
                out.append(rec)
        return out

    def request_writeback(self, sess: SiteSession) -> bool:
        """Write the WHOLE current jar for the site to
        ``<dir>/writeback/<uuid>.json`` in the platform's snake_case shape.
        The cloud drains and deletes each file on a timer whether or not it
        could use it, so: atomic tmp+rename (a half-written file is simply
        lost), guards before writing, and never an exception into the fetch.

        Guards, in order: no platform address (missing swarmio block) -> the
        refreshed jar cannot find its vault row, skip; NONE of the planted
        names survived (the site logged the session out) -> an empty or
        stripped jar would overwrite a good one, skip; everything expired ->
        skip; over the cloud's per-jar cap -> skip; content unchanged since
        the last write-back -> skip."""
        with self._lock:
            if not sess.platform:
                return False
            names = sess.names()
            if not (sess.planted_names & names):
                log.info("sessions: %s looks logged out; not writing back",
                         sess.key)
                return False
            now = time.time()
            live = [c for c in sess.cookies if not _expired(c, now)]
            if not live or len(live) > _MAX_COOKIES:
                return False
            payload = {
                "platform": sess.platform,
                "label": sess.label,
                "cookies": [
                    {
                        "name": c["name"],
                        "value": c["value"],
                        "domain": c["domain"],
                        "path": c.get("path") or "/",
                        **({"expires": c["expires"]} if c.get("expires") is not None else {}),
                        "secure": bool(c.get("secure")),
                        "http_only": bool(c.get("http_only")),
                    }
                    for c in live
                ],
            }
            blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            digest = hashlib.sha256(blob.encode()).hexdigest()
            if digest == sess._wb_hash:
                return False
            out_dir = self._dir / "writeback"
            try:
                # The cloud creates the directory with the volume, but a fresh
                # mount (or a cloud that only drains) may not have it yet --
                # creating it is ours, same as the tmp+rename discipline.
                out_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write(out_dir / f"{uuid.uuid4().hex}.json", blob.encode())
            except OSError:
                # Windows: os.replace fails if the cloud's drain holds the
                # destination open -- the tmp is cleaned up and the next
                # rotation retries. Never propagates into the fetch.
                self.errors += 1
                return False
            sess._wb_hash = digest
            log.info("sessions: write-back queued for %s (%d cookies)",
                     sess.key, len(live))
            return True

    def close(self) -> None:
        """Flush the pin store; pins are written eagerly too, so this is a
        belt-and-braces no-op on a healthy run."""
        with self._lock:
            try:
                _atomic_write(self._pins_path,
                              json.dumps(self._pins, indent=1).encode())
            except OSError:
                self.errors += 1
