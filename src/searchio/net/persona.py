"""Coherent browser identities.

The single most common way a scraper gives itself away is not *which*
fingerprint it presents but that its fingerprint does not agree with itself: a
Chrome User-Agent over a Python TLS handshake, a macOS UA with
``Sec-CH-UA-Platform: "Windows"``, a Chrome 131 UA with Chrome 116's
``sec-ch-ua`` brand list. Anti-bot vendors do not need a secret signal to catch
that; the contradiction *is* the signal.

So a :class:`Persona` is an all-or-nothing bundle. You never pick a User-Agent;
you pick a persona, and everything downstream — TLS impersonation target, HTTP/2
settings, client hints, ``Accept-Language`` — comes from the same row.

The second rule is *stability*. Rotating identity per request is a stronger
tell than any single fingerprint: real users do not change browser between page
one and page two. :func:`for_domain` therefore pins one persona per domain per
session, so a site sees one consistent visitor.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """One self-consistent browser identity.

    ``impersonate`` is the curl_cffi target, which sets the TLS (JA3/JA4) and
    HTTP/2 fingerprint. Everything else must describe *that same browser*.
    """

    name: str
    impersonate: str
    ua: str
    sec_ch_ua: str
    platform: str
    accept_language: str = "en-US,en;q=0.9"
    accept_encoding: str = "gzip, deflate, br, zstd"

    def headers(self, *, referer: str = "", contact: str = "") -> dict[str, str]:
        """Header set for this identity, in Chrome's own emission order.

        Header *order* is fingerprinted too. curl_cffi's impersonation handles
        the low-level ordering; this dict keeps the high-level set coherent
        with the TLS target above.
        """
        h = {
            "User-Agent": self.ua,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": self.accept_language,
            "Accept-Encoding": self.accept_encoding,
            "sec-ch-ua": self.sec_ch_ua,
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": f'"{self.platform}"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site" if referer else "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        if referer:
            h["Referer"] = referer
        if contact:
            # Opt-in honesty channel. Off by default because many WAFs treat an
            # unknown token in the UA as reason enough to challenge, but when a
            # site operator asks who we are, this is how they find out.
            h["From"] = contact
        return h


# Kept deliberately small. Each row is a real, current, self-consistent
# desktop browser; a large pool of half-invented fingerprints is worse than a
# handful of accurate ones, because every invented row is a contradiction
# waiting to be noticed.
PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="chrome-win",
        impersonate="chrome150",
        ua=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="150", "Chromium";v="150", "Not_A Brand";v="24"',
        platform="Windows",
    ),
    Persona(
        name="chrome-mac",
        impersonate="chrome150",
        ua=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="150", "Chromium";v="150", "Not_A Brand";v="24"',
        platform="macOS",
    ),
    Persona(
        name="edge-win",
        # Edge IS Chromium: its TLS/H2 fingerprint is chrome<major>. The
        # old target, edge101, put a Chrome 101-era JA3 under a 131 UA --
        # the exact contradiction this module exists to prevent (bug 70).
        impersonate="chrome150",
        ua=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0"
        ),
        sec_ch_ua='"Microsoft Edge";v="150", "Chromium";v="150", "Not_A Brand";v="24"',
        platform="Windows",
    ),
    Persona(
        name="safari-mac",
        impersonate="safari17_0",
        ua=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ),
        # Safari sends no Client Hints at all. Emitting them here would be the
        # exact kind of contradiction this module exists to prevent.
        sec_ch_ua="",
        platform="macOS",
        accept_encoding="gzip, deflate, br",
    ),
)

DEFAULT = PERSONAS[0]


def for_domain(domain: str, *, session: str = "") -> Persona:
    """Pick a stable persona for ``domain``.

    Deterministic in ``(domain, session)``: the same domain gets the same
    identity for the life of a session, and two different domains are unlikely
    to share one, so a request pattern never looks like one browser
    simultaneously being four browsers.
    """
    seed = f"{session}|{domain}".encode()
    idx = int.from_bytes(hashlib.blake2b(seed, digest_size=2).digest(), "big")
    return PERSONAS[idx % len(PERSONAS)]


def by_name(name: str) -> Persona:
    for p in PERSONAS:
        if p.name == name:
            return p
    raise KeyError(name)


_CHROME_MAJOR = re.compile(r"\bChrome/(\d+)")


def hints_for_ua(ua: str) -> dict[str, str]:
    """Client Hints that agree with ``ua``: the brand list at the UA's
    Chrome major, the platform from its OS token. ``{}`` for a browser that
    sends none (Safari, Firefox) -- the caller strips instead."""
    m = _CHROME_MAJOR.search(ua or "")
    if not m or "Firefox/" in ua:
        return {}
    major = m.group(1)
    brand = "Microsoft Edge" if "Edg/" in ua else "Google Chrome"
    if "Android" in ua:
        platform, mobile = "Android", "?1"
    elif "Windows" in ua:
        platform, mobile = "Windows", "?0"
    elif "Macintosh" in ua or "Mac OS X" in ua:
        platform, mobile = "macOS", "?0"
    elif "CrOS" in ua:
        platform, mobile = "Chrome OS", "?0"
    else:
        platform, mobile = "Linux", "?0"
    return {
        "sec-ch-ua": f'"{brand}";v="{major}", "Chromium";v="{major}", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": mobile,
        "sec-ch-ua-platform": f'"{platform}"',
    }


def _chrome_targets() -> list[int]:
    """Chrome majors this curl_cffi can impersonate, ascending."""
    try:
        from curl_cffi.requests.impersonate import BrowserType
    except Exception:  # noqa: BLE001 -- optional dependency
        return []
    out = []
    for b in BrowserType:
        m = re.fullmatch(r"chrome(\d+)", b.value)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def impersonate_for_ua(ua: str, default: str) -> str:
    """The TLS/H2 fingerprint that agrees with ``ua``.

    A banked clearance pins the BROWSER's User-Agent (Chrome 145, say) onto
    a tier-1 request; sending it over the persona's Chrome 131 fingerprint
    is the contradiction bugs 70/73 removed one layer up (bug 141). Picks
    the newest supported Chrome target not above the UA's major (the
    closest fingerprint that browser could have); non-Chromium UAs keep the
    persona's target.
    """
    m = _CHROME_MAJOR.search(ua or "")
    if not m or "Firefox/" in (ua or ""):
        return default
    major = int(m.group(1))
    below = [t for t in _chrome_targets() if t <= major]
    return f"chrome{below[-1]}" if below else default


def pin_user_agent(headers: dict[str, str], ua: str) -> dict[str, str]:
    """Pin ``ua`` on a header set and make the Client Hints agree with it.

    A pinned UA over another persona's hints (Chrome 143 with v="150", or a
    Chrome UA under the Safari persona's no-hints set) is the contradiction
    that gets a clearance replay flagged (bug 73)."""
    out = {k: v for k, v in headers.items() if not k.lower().startswith("sec-ch-ua")}
    out["User-Agent"] = ua
    out.update(hints_for_ua(ua))
    return out


def safari_headers_fix(p: Persona, headers: dict[str, str]) -> dict[str, str]:
    """Strip Client Hints for browsers that do not send them."""
    if not p.sec_ch_ua:
        for k in ("sec-ch-ua", "sec-ch-ua-mobile", "sec-ch-ua-platform"):
            headers.pop(k, None)
    return headers
