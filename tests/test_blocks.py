"""Block classification: the decision that drives every tier escalation.

These cases are the ones that were actually wrong at some point during
development, which is why they are the ones worth keeping.
"""

from __future__ import annotations

from searchio.net.blocks import classify, identify_vendor, visible_text

HTML = "text/html"


def page(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def prose(n: int = 40) -> str:
    return "Real article content that a reader would actually want to read. " * n


class TestVendor:
    def test_cloudflare_from_header(self):
        assert identify_vendor({"cf-mitigated": "challenge"}) == "cloudflare"

    def test_cloudflare_from_server(self):
        assert identify_vendor({"server": "cloudflare"}) == "cloudflare"

    def test_datadome_from_cookie(self):
        assert identify_vendor({"set-cookie": "datadome=xyz; Path=/"}) == "datadome"

    def test_akamai_from_cookie(self):
        assert identify_vendor({"set-cookie": "_abck=abc"}) == "akamai"

    def test_perimeterx_from_cookie(self):
        assert identify_vendor({"set-cookie": "_pxvid=1"}) == "perimeterx"

    def test_none_for_plain(self):
        assert identify_vendor({"content-type": HTML}) is None


class TestBlocked:
    def test_403_is_blocked(self):
        v = classify(403, {"server": "cloudflare"}, page("nope"))
        assert v.blocked and v.vendor == "cloudflare"

    def test_403_perimeterx_names_itself_in_the_body(self):
        # Live zillow.com wall (2026-09-16): the 403 body is exactly the
        # 20-byte string "perimeterx_challenge" with no PX header or cookie
        # tell. Without a body fallback the verdict is vendor None -- an
        # anonymous wall the rescue policy can't act on, when a PX challenge
        # is exactly what the challenge browser clears.
        v = classify(403, {"content-type": HTML}, "perimeterx_challenge")
        assert v.blocked and v.vendor == "perimeterx"

    def test_403_large_body_mentioning_perimeterx_is_not_stamped(self):
        # The size gate is load-bearing: a big page that merely mentions the
        # string (docs, incident postmortem) is not a PX wall.
        v = classify(403, {"content-type": HTML}, prose(120) + " perimeterx_challenge " + prose(120))
        assert v.blocked and v.vendor is None

    def test_429_is_an_honest_refusal_not_a_wall(self):
        # A rate limit is a throttle instruction (Retry-After), not a
        # challenge: blocked=True made 429 rescue-worthy, i.e. a rate limit
        # could boot the challenge browser -- waste plus abuse. Demoted in
        # iteration 19 (span ctl.http_429_honest* pins no challenge boot).
        v = classify(429, {}, page("slow down"))
        assert not v.blocked and not v.ok and v.reason == "http_429"

    def test_cloudflare_interstitial(self):
        v = classify(200, {"content-type": HTML}, page("Just a moment..."))
        assert v.blocked

    def test_challenge_widget_on_empty_page(self):
        v = classify(200, {"content-type": HTML}, page('<div class="anomaly-modal">x</div>'))
        assert v.blocked

    def test_waf_503_is_a_challenge(self):
        v = classify(503, {"cf-mitigated": "challenge"}, page("x"))
        assert v.blocked

    def test_origin_503_is_not_a_block(self):
        # No WAF fingerprint: this is the site being down, and escalating to a
        # browser would waste a launch on an outage.
        v = classify(503, {}, page("maintenance"))
        assert not v.blocked and not v.ok


class TestNotBlocked:
    def test_real_page_is_ok(self):
        assert classify(200, {"content-type": HTML}, page(prose())).ok

    def test_article_mentioning_captcha_is_not_a_block(self):
        # The regression that matters: a support article about CAPTCHAs is a
        # perfectly good page.
        v = classify(200, {"content-type": HTML}, page("<h1>Fixing a captcha loop</h1>" + prose()))
        assert v.ok, v.reason

    def test_large_page_referencing_turnstile_is_not_a_block(self):
        # A 100 KB SERP that merely references Cloudflare Turnstile in some
        # unrelated script is not a challenge page.
        body = page(prose(200) + '<script src="cf-turnstile.js"></script>')
        assert classify(200, {"content-type": HTML}, body).ok

    def test_json_is_always_usable(self):
        v = classify(200, {"content-type": "application/json"}, '{"a":1}')
        assert v.ok and v.reason == "structured"

    def test_empty_structured_body_is_not_a_page(self):
        # Bug 48 (iteration 35, DeepSeek blocks.py review): the structured
        # (json/xml/csv) win returned ok with text_len=len(body) BEFORE any
        # content check, so an EMPTY body under a json content-type scored ok
        # -- the silent false-ok this module exists to prevent (a truncated
        # or stub API response served as a retrieval). A valid-but-empty
        # result ("[]"/"{}") is real data and stays ok.
        for empty in ("", "   ", "\n\t"):
            v = classify(200, {"content-type": "application/json"}, empty)
            assert not v.ok, repr(empty)
        assert classify(200, {"content-type": "application/json"}, "[]").ok
        assert classify(200, {"content-type": "application/xml"}, "<r/>").ok


class TestUnusable:
    def test_empty_react_mount(self):
        v = classify(200, {"content-type": HTML}, page('<div id="root"></div>'))
        assert not v.ok and not v.blocked and v.reason == "empty_mount"

    def test_js_required_notice(self):
        v = classify(200, {"content-type": HTML}, page("You need to enable JavaScript to run"))
        assert not v.ok and v.reason == "js_required"

    def test_thin_page(self):
        v = classify(200, {"content-type": HTML}, page("hi"))
        assert not v.ok and v.reason.startswith("too_thin")

    def test_pdf_is_not_html(self):
        v = classify(200, {"content-type": "application/pdf"}, "%PDF-1.4")
        assert not v.ok and v.reason.startswith("not_html")

    def test_binary_magic_outranks_a_text_label(self):
        # Bug 20: tier-2 sidecar envelopes fabricated content-type text/html
        # for every body, and a PDF's ASCII operator stream cleared the body
        # heuristics -- control bytes shipped as an ok page (bench/span.py
        # bulk seed 113, two live PDFs). The magic at byte zero settles it.
        pdf = "%PDF-1.7\n" + "\x00\x01\x02obj stream BT /F1 Tf ET\n" * 400
        for label in ("text/html", "text/plain", ""):
            v = classify(200, {"content-type": label} if label else {}, pdf)
            assert not v.ok and v.reason == "binary:application/pdf", label

    def test_binary_magic_covers_the_common_downloads(self):
        cases = [
            ("PK\x03\x04" + "\x00" * 50, "binary:application/zip"),
            ("\x89PNG\r\n\x1a\n" + "\x00" * 50, "binary:image/png"),
            ("\xff\xd8\xff\xe0" + "\x00" * 50, "binary:image/jpeg"),
            ("GIF89a" + "\x00" * 50, "binary:image/gif"),
            ("\x1f\x8b\x08" + "\x00" * 50, "binary:application/gzip"),
            ("7z\xbc\xaf\x27\x1c" + "\x00" * 50, "binary:application/x-7z-compressed"),
        ]
        for body, reason in cases:
            v = classify(200, {"content-type": "text/html"}, body)
            assert not v.ok and v.reason == reason, body[:8]

    def test_magic_does_not_overrule_an_honest_binary_label(self):
        # The structured/not_html branches already handle these; the sniff
        # only arbitrates text-ish lies. A JSON content-type with a JSON body
        # must keep winning even when the payload quotes a magic string.
        v = classify(200, {"content-type": "application/json"}, '{"a":1}')
        assert v.ok

    def test_mojibake_is_unusable_not_a_page(self):
        # Bug 23: a Shift_JIS page lossy-UTF-8'd arrives heavy with U+FFFD.
        # Scoring the stew as a clean ok hands the agent decode garbage --
        # the same silent-wrong-answer class as the empty shell. text_len=0
        # pins that the ladder's thin-page fallback cannot resurrect it.
        stew = "english words " * 40 + "\ufffd" * 40
        v = classify(200, {"content-type": HTML}, page(stew))
        assert not v.ok and not v.blocked
        assert v.reason.startswith("mojibake:")
        assert v.text_len == 0

    def test_stray_replacement_chars_do_not_condemn_a_page(self):
        # A couple of U+FFFD in otherwise real prose is the page's own
        # problem, not a decode failure -- density is the signal.
        body = prose(40) + " \ufffd \ufffd "
        v = classify(200, {"content-type": HTML}, page(body))
        assert v.ok

    def test_bot_wall_outranks_mojibake(self):
        # Wall markers are ASCII and survive a wrong decode; "blocked" drives
        # the more useful behavior (rescue/climb), so it wins over mojibake.
        body = "Just a moment... " + "\ufffd" * 40
        v = classify(200, {"content-type": HTML}, page(body))
        assert not v.ok and v.blocked


def test_visible_text_strips_scripts_and_styles():
    html = "<style>a{color:red}</style><script>var x=1</script><p>Hello world</p>"
    assert visible_text(html).strip() == "Hello world"


def test_control_characters_are_not_visible_text():
    # A corrupt/binary payload answered as text/html must measure as no text,
    # not as a page full of "characters". bench/span.py ctl.garbage_body.
    garbage = "\x07\x0b\x0e\x1f" * 300
    assert visible_text(garbage) == ""
    v = classify(200, {"content-type": HTML}, garbage)
    assert not v.ok, "control-char garbage classified as a usable page"
    # ...while ordinary formatting whitespace survives untouched.
    assert visible_text("a\tb\nc\rd") == "a b c d"


class TestSolvedChallengeIsNotABlock:
    """A solved Turnstile leaves its widget in the DOM.

    nowsecure.nl serves its success page -- the word "NOWSECURE" -- with the
    cf-turnstile element still attached. Reading that markup as a wall discards
    a page the browser had already earned.
    """

    def test_turnstile_markup_alone_is_not_a_block(self):
        body = page(
            '<div class="cf-turnstile" data-sitekey="x"></div>'
            "<h1>NOWSECURE</h1><p>by nodriver</p>"
        )
        v = classify(200, {"content-type": HTML}, body, min_text=20)
        assert not v.blocked, v.reason

    def test_real_cloudflare_interstitial_still_blocks(self):
        body = page('<div class="cf-turnstile"></div><h1>Just a moment...</h1>')
        assert classify(200, {"content-type": HTML}, body).blocked

    def test_perimeterx_widget_on_an_empty_page_still_blocks(self):
        assert classify(200, {"content-type": HTML}, page('<div id="px-captcha"></div>')).blocked


class TestVisibleTextBounded:
    """visible_text must stay linear on hostile markup (bug 47, iteration 35).

    The script/style stripper used ``<(script|style|noscript)\b[^>]*>.*?</\1>``
    (DOTALL): a body with many UNCLOSED <script> opens made each open's ``.*?``
    scan to EOF -- O(K*N) quadratic, minutes on a ~3 MB body that fits inside
    the fetch cap. classify() runs visible_text on every fetched body, so
    one hostile page wedges the whole ladder. The unrolled pattern is linear.
    """

    def test_unclosed_script_opens_do_not_hang(self):
        import time

        body = "<script>" * 300000  # ~2.4 MB of unclosed opens
        t = time.perf_counter()
        out = visible_text(body)
        dt = time.perf_counter() - t
        assert dt < 5.0, f"visible_text took {dt:.1f}s -- quadratic regression"
        assert out == "", "unclosed script content must still be stripped"

    def test_scripts_and_styles_still_stripped(self):
        assert visible_text("<p>keep</p><script>evil()</script><p>this</p>") == "keep this"
        assert visible_text("a<script>x</script>b<style>y</style>c") == "a b c"
        # adjacent blocks stop at the first close, not the last
        assert visible_text("<script>a</script>MID<script>b</script>END") == "MID END"
