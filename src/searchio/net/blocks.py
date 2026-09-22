"""Deciding whether a response is a page, a wall, or a shell.

Three outcomes matter and they are not the same thing:

* **blocked** — an anti-bot system refused us. A heavier tier may get through.
* **unusable** — the site answered honestly but with nothing readable (an empty
  SPA mount, a login gate, a PDF). A browser may fill it in.
* **ok** — usable content; stop climbing, we are done.

Getting this wrong is expensive in both directions. A false "blocked" burns a
browser launch on a page plain HTTP already answered. A false "ok" hands the
model an empty shell, which it faithfully reports as "this page has no
information" — a silent wrong answer, and much the worse failure.

The vendor identification is not decoration. What to do next genuinely differs:
a Cloudflare managed challenge usually clears in a real browser, while a
DataDome block is often IP reputation, and re-running the same egress through
Chromium just fails slower.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Vendor signatures, checked against headers+cookies first (cheap and far more
# reliable than body text, which varies by locale and customer template).
_VENDOR_HEADERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloudflare", ("cf-mitigated", "cf-chl-bypass")),
    ("datadome", ("x-datadome", "x-dd-b")),
    ("incapsula", ("x-iinfo",)),
    ("akamai", ("x-akamai-transformed",)),
)

_VENDOR_COOKIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloudflare", ("__cf_bm", "cf_clearance", "__cfduid")),
    ("datadome", ("datadome",)),
    ("perimeterx", ("_px", "_pxhd", "_pxvid")),
    ("akamai", ("_abck", "bm_sz", "ak_bmsc")),
)

# Vendor signatures that appear in the BODY, not headers or cookies. Used only
# on small responses: the vendor naming itself IS the wall there. A page big
# enough to merely *mention* the string (docs, a status postmortem) is not a
# block, so the size gate is the load-bearing part of the check.
_VENDOR_BODY: tuple[tuple[str, tuple[str, ...]], ...] = (
    # PerimeterX's hard block answers 403 with a body of exactly
    # "perimeterx_challenge" and no header or cookie tell (live: zillow.com,
    # 2026-09-16 -- the 20-byte body said the vendor's name and classify
    # still reported vendor None, so the ladder recorded an anonymous
    # http_403 where a PerimeterX block -- actionable by the challenge
    # browser -- was owed).
    ("perimeterx", ("perimeterx_challenge",)),
)


def _identify_vendor_body(body: str) -> str | None:
    """Vendor from an unambiguous self-naming string on a SMALL response."""
    if len(body) > 4000:
        return None
    low = body.lower()
    for vendor, markers in _VENDOR_BODY:
        if any(m in low for m in markers):
            return vendor
    return None

# Body markers. Ordered most- to least-specific; the generic ones ("captcha")
# only count when the page is *dominated* by the challenge — see below.
_BLOCK_MARKERS_STRONG: tuple[str, ...] = (
    "just a moment",
    "checking your browser before accessing",
    "enable javascript and cookies to continue",
    "pardon our interruption",
    "please verify you are a human",
    "why have i been blocked",
    "attention required! | cloudflare",
    "request unsuccessful. incapsula incident",
    "access to this page has been denied",
)

# Markers that live in markup rather than visible text -- a challenge widget
# can render almost no words while still being a wall.
#
# `cf-turnstile` is deliberately NOT here. The widget element stays in the DOM
# after the challenge is solved, so its presence says nothing about whether we
# got through: nowsecure.nl serves its success page ("NOWSECURE") with the
# turnstile markup still attached, and treating that as a wall throws away a
# page we had already earned. A real Cloudflare interstitial announces itself
# in visible text ("Just a moment...") and is caught by _BLOCK_MARKERS_STRONG.
_BLOCK_MARKUP: tuple[str, ...] = (
    "anomaly-modal",  # DuckDuckGo
    "px-captcha",  # PerimeterX
    # Markup-only signatures (bug 125): DataDome's challenge is a script src
    # with no visible words and Cloudflare's legacy check is a div id, so
    # checking them against VISIBLE text classified the wall as too_thin --
    # an honest refusal where a block (and its rescue) was owed.
    "captcha-delivery.com",  # DataDome
    "cf-browser-verification",  # Cloudflare (legacy)
)

#: <title> signatures of a wall. The title is the vendor's own label, so it
#: counts at any page size (a padded challenge page is still the wall).
_BLOCK_TITLES: tuple[str, ...] = (
    "just a moment",
    "attention required",
    "access denied",
    "pardon our interruption",
    "request unsuccessful",
    "are you a robot",
    "bot verification",
    "human verification",
)

#: Soft-error pages served with 200: not a wall (no rescue helps), not
#: content either. Gated on a nearly empty page like the weak markers.
_SOFT_ERROR_MARKERS: tuple[str, ...] = (
    "something went wrong",
    "page not found",
    "page you requested was not found",
    "an error occurred",
    "an unexpected error",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
)

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)

_BLOCK_MARKERS_WEAK: tuple[str, ...] = (
    "captcha",
    "unusual traffic",
    "are you a robot",
    "bot detection",
    "ddos protection",
    "access denied",
)

_JS_MARKERS: tuple[str, ...] = (
    "you need to enable javascript",
    "please enable javascript",
    "javascript is required",
    "this app requires javascript",
)

# Framework mount points. An empty one with no prose around it is a shell no
# matter how many kilobytes of inline script shipped with it.
_MOUNTS: tuple[str, ...] = (
    'id="root"',
    "id='root'",
    'id="app"',
    "id='app'",
    'id="__next"',
    'id="__nuxt"',
    "data-reactroot",
)

# Unrolled-loop form, NOT ``.*?</\1>``: the lazy version made every UNCLOSED
# <script> open scan to end-of-body, O(K*N) quadratic on a hostile page within
# the fetch cap (32 MiB since the vinted live fix) -- and classify() runs
# visible_text on every fetched body, so one crafted page wedged the whole
# ladder for minutes (bug 47, iteration 35). This matches the block content as
# [^<] runs plus any '<' that does NOT begin a matching close, then an OPTIONAL
# close so an unclosed tag still strips (consumed once, linearly). Closes on
# any of the three tags rather than the exact one -- fine for a length/keyword
# pass, and it bounds the scan.
_TAG_RE = re.compile(
    r"<(?:script|style|noscript)\b[^>]*>"
    r"[^<]*(?:<(?!/(?:script|style|noscript)\b)[^<]*)*"
    r"(?:</(?:script|style|noscript)\s*>)?",
    re.I | re.S,
)
# ``[^<>]`` not ``[^>]`` (bug 122): with no '>' in sight the old class scanned
# to end-of-body from EVERY '<', so a 2 MB run of '<' (or of unclosed "<a")
# wedged classify() for minutes -- bug 47's shape on the other regex. A tag
# cannot contain '<', so stopping at the next one loses nothing and bounds
# every scan by the distance to the next angle bracket.
_ANYTAG_RE = re.compile(r"<[^<>]+>")
_WS_RE = re.compile(r"\s+")
# C0/C1 controls and DEL are not reader-visible text. A body made of them
# (a corrupt payload, a binary answered as text/html) must measure as zero
# text, not as a page: bench/span.py ctl.garbage_body caught 1200 bytes of
# \x07\x0b\x0e\x1f scored as a clean retrieval. Tab/LF/CR stay -- they are
# ordinary formatting and the whitespace collapse handles them.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")


def visible_text(html: str) -> str:
    """Strip markup, scripts and styles; return roughly what a reader sees.

    Not a parser and not trying to be one — this feeds a length threshold and a
    keyword scan, both of which tolerate a sloppy result. Real extraction is
    :mod:`searchio.extract`'s job.
    """
    body = _TAG_RE.sub(" ", html)
    body = _ANYTAG_RE.sub(" ", body)
    body = _CTRL_RE.sub(" ", body)
    return _WS_RE.sub(" ", body).strip()


@dataclass
class Verdict:
    """The classification of one response."""

    ok: bool
    blocked: bool = False
    vendor: str | None = None
    reason: str = ""
    text_len: int = 0

    @property
    def unusable(self) -> bool:
        return not self.ok and not self.blocked


#: Body magics that outrank a claimed text content-type. A text label can be
#: a lie twice over: origins mislabel downloads, and tier-2 sidecar envelopes
#: historically fabricated ``text/html`` for every body (bench/span.py bulk
#: seed 113 caught two PDFs sailing through tier 2 as ok pages with control
#: bytes -- their ASCII operator streams clear ``min_text``). No real page
#: starts with one of these at byte zero.
_BINARY_MAGIC: tuple[tuple[str, str], ...] = (
    ("%PDF-", "application/pdf"),
    ("PK\x03\x04", "application/zip"),
    ("7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    ("\x89PNG\r\n\x1a\n", "image/png"),
    ("\xff\xd8\xff", "image/jpeg"),
    ("GIF87a", "image/gif"),
    ("GIF89a", "image/gif"),
    ("\x1f\x8b", "application/gzip"),
)

#: Content-type words the magic check is allowed to overrule. A type that
#: already names a binary format needs no sniffing; the lying cases are the
#: text-ish ones (and the missing one).
_TEXTISH = ("html", "text", "json", "xml", "csv", "plain", "javascript")


def identify_vendor(headers: dict[str, str], cookies: str = "") -> str | None:
    """Name the anti-bot system in front of a response, if any is visible.

    Presence of a vendor is *not* by itself a block — most of Cloudflare's
    customers serve normal pages through it, and a ``__cf_bm`` cookie rides
    along with a perfectly good 200. This only answers "who is standing here".
    """
    low = {k.lower(): (v or "") for k, v in headers.items()}
    for vendor, keys in _VENDOR_HEADERS:
        if any(k in low for k in keys):
            return vendor
    blob = (cookies + " " + low.get("set-cookie", "")).lower()
    for vendor, names in _VENDOR_COOKIES:
        if any(n in blob for n in names):
            return vendor
    if "cloudflare" in low.get("server", "").lower() or "cf-ray" in low:
        return "cloudflare"
    return None


def _looks_like_challenge(body: str, headers: dict[str, str]) -> bool:
    """A challenge signature: the vendor SAYING so (Cloudflare's
    ``cf-mitigated: challenge``) or a wall signature in the first 64 KB of
    raw markup."""
    if any(k.lower() == "cf-mitigated" for k in headers):
        return True
    blow = body[:65536].lower()
    return any(m in blow for m in _BLOCK_MARKERS_STRONG + _BLOCK_MARKUP + _BLOCK_TITLES)


def _challenge_dominant(text: str, marker: str) -> bool:
    """True when the challenge *is* the page, not a mention on a real one.

    A support article titled "How to fix a CAPTCHA loop" is not a block, and an
    e-commerce page with a hidden reCAPTCHA widget in its login modal is not a
    block either. A genuine interstitial is nearly empty: a heading, a line of
    explanation, maybe a ray ID.
    """
    return len(text) < 800


def classify(
    status: int,
    headers: dict[str, str],
    body: str,
    *,
    content_type: str = "",
    min_text: int = 120,
) -> Verdict:
    """Classify one HTTP response.

    Order matters: status first (cheapest and most decisive), then vendor
    signals, then body markers, then the shell heuristics — so we never scan a
    megabyte of HTML to conclude what a 403 already said.
    """
    # Header names are case-insensitive; callers hand over whatever their
    # client produced (rider, iteration 59: "Content-Type: image/png" took
    # the html path and a PNG scored as an ok page).
    headers = {str(k).lower(): (v or "") for k, v in headers.items()}
    vendor = identify_vendor(headers, headers.get("set-cookie", ""))
    ctype = (content_type or headers.get("content-type", "")).lower()

    # ── Status ───────────────────────────────────────────────────────────────
    if status in (401, 402, 407):
        return Verdict(False, reason=f"auth_required_{status}", vendor=vendor)
    if status in (403,):
        # A 403 answers "no" but says nothing about why, and the why decides
        # what to try next. Headers/cookies told us nothing here -- fall back
        # to a vendor that names itself in a small body (zillow's PX wall is
        # literally the 20-byte string "perimeterx_challenge"), so the block
        # is stamped with a vendor the rescue policy can act on.
        if vendor is None:
            vendor = _identify_vendor_body(body)
        return Verdict(False, blocked=True, vendor=vendor, reason=f"http_{status}")
    if status == 429:
        # A rate limit is an instruction (Retry-After), not a wall: booting
        # a challenge browser at it both wastes the render and hammers the
        # origin harder -- the bug-10 class in a polite costume. Refuse
        # honestly, no rescue.
        return Verdict(False, reason="http_429", vendor=vendor)
    if status == 503 and vendor and _looks_like_challenge(body, headers):
        # 503 from a WAF is a challenge; 503 from an origin is an outage --
        # and a CDN-fronted origin that is DOWN answers 503 with the vendor's
        # headers too (rider, iteration 59): only a body carrying a challenge
        # signature is a challenge, the rest is an outage no rescue can fix.
        return Verdict(False, blocked=True, vendor=vendor, reason="http_503_challenge")
    if not 200 <= status < 300:
        return Verdict(False, reason=f"http_{status}", vendor=vendor)

    # ── Content type ─────────────────────────────────────────────────────────
    # Body magic outranks a text-ish claim (see _BINARY_MAGIC). Runs before the
    # structured win below: a PDF mislabeled text/plain is still a PDF.
    if not ctype or any(t in ctype for t in _TEXTISH):
        for magic, real in _BINARY_MAGIC:
            if body.startswith(magic):
                return Verdict(False, reason=f"binary:{real}", vendor=vendor)
    # JSON is a first-class win, not a fallback: a site's own API answers with
    # data a browser render could only ever approximate, and it does not care
    # what our fingerprint looks like.
    if any(t in ctype for t in ("json", "xml", "text/plain", "csv")):
        # A structured body is a first-class win -- but an EMPTY one is not a
        # retrieval, it is a truncated/stub response wearing a data label
        # (bug 48). "[]"/"{}" are non-empty and stay ok (valid empty results);
        # a whitespace-only body falls through to an honest refusal. A SHORT
        # text/plain body is checked for wall words first (rider, iteration
        # 59): "Please complete the CAPTCHA" as text/plain was "structured".
        if "text/plain" in ctype and len(body) < 800:
            plain_low = body.lower()
            for m in _BLOCK_MARKERS_STRONG + _BLOCK_MARKERS_WEAK:
                if m in plain_low:
                    return Verdict(False, blocked=True, vendor=vendor, reason=f"marker:{m}", text_len=len(body))
        if body.strip():
            return Verdict(True, reason="structured", text_len=len(body))
    if ctype and "html" not in ctype and "text" not in ctype:
        return Verdict(False, reason=f"not_html:{ctype.split(';')[0]}", vendor=vendor)

    text = visible_text(body)
    n = len(text)
    low = text.lower()
    blow = body.lower()

    # Challenge widgets are identified from markup, not visible text, because a
    # turnstile renders almost no words. But the marker alone is not enough: a
    # 124 KB search results page that merely *references* Turnstile in some
    # unrelated script is plainly not a challenge. Require the page to also be
    # substantively empty, which every real interstitial is.
    if n < 1500:
        for m in _BLOCK_MARKUP:
            if m in blow:
                return Verdict(
                    False, blocked=True, vendor=vendor, reason=f"widget:{m}", text_len=n
                )

    # ── Bot walls ────────────────────────────────────────────────────────────
    # The <title> is the vendor's own label and counts at any size; a body
    # marker counts only on a page that is substantively empty (bug 124: a
    # real article QUOTING "checking your browser before accessing", or
    # containing the everyday "just a moment", was refused as blocked and a
    # rescue launched for a page the cheap tier had already served).
    tm = _TITLE_RE.search(body[:8192])
    title_low = re.sub(r"\s+", " ", tm.group(1)).strip().lower() if tm else ""
    for m in _BLOCK_TITLES:
        if m in title_low:
            return Verdict(False, blocked=True, vendor=vendor, reason=f"title:{m}", text_len=n)
    if n < 1500:
        for m in _BLOCK_MARKERS_STRONG:
            if m in low:
                return Verdict(False, blocked=True, vendor=vendor, reason=f"marker:{m}", text_len=n)
    for m in _BLOCK_MARKERS_WEAK:
        if m in low and _challenge_dominant(text, m):
            return Verdict(False, blocked=True, vendor=vendor, reason=f"weak:{m}", text_len=n)
    # A soft error served as 200 (rider): "Sorry! Something went wrong" over
    # a nearly empty page is neither a wall nor content.
    if n < 800:
        for m in _SOFT_ERROR_MARKERS:
            if m in low or m in title_low:
                return Verdict(False, reason=f"soft_error:{m}", vendor=vendor, text_len=n)

    # ── Mojibake ─────────────────────────────────────────────────────────────
    # A body heavy with U+FFFD replacement chars was decoded against the wrong
    # charset (the classic shape: a Shift_JIS page lossy-UTF-8'd). That is not
    # content, it is a decode failure -- handing it to the agent as an ok page
    # is the same silent-wrong-answer class as the empty shell. After the
    # bot-wall checks on purpose: wall markers are ASCII and survive a wrong
    # decode, and "blocked" drives the more useful behavior (rescue/climb).
    # text_len=0 so the ladder's thin-page fallback cannot resurrect the stew
    # as accepted_thin. Thresholds mirror the bench/span.py bulk mojibake
    # detector: real pages carry zero replacement chars (probed across tiers),
    # so a couple of strays in real prose is a page problem, heavy density is
    # a decode problem. Bug 23, iteration-15 probe: every non-browser tier
    # scored a meta-only-charset page's 156-FFFD body as a clean ok.
    n_fffd = text.count("\ufffd")
    if n and n_fffd >= 3 and n_fffd / n > 0.002:
        return Verdict(False, reason=f"mojibake:{n_fffd}/{n}", vendor=vendor, text_len=0)

    # ── Shells ───────────────────────────────────────────────────────────────
    # A JS notice is a shell only on an EMPTY page (bug 123): every comment
    # widget's footer says "please enable JavaScript", and a rich page that
    # carried one was discarded as a shell and a browser booted for nothing.
    if n < 1500:
        for m in _JS_MARKERS:
            if m in low:
                return Verdict(False, reason="js_required", text_len=n)
    if n <= 200:
        if any(mount in blow for mount in _MOUNTS):
            return Verdict(False, reason="empty_mount", text_len=n)
    if n < min_text:
        return Verdict(False, reason=f"too_thin:{n}", text_len=n)

    return Verdict(True, reason="ok", text_len=n)
