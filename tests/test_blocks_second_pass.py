"""blocks.py second pass (iteration 59): classify() probed with hostile,
borderline and wall-shaped bodies before the DeepSeek review landed.

Every test here bit RED before its fix.
"""

from __future__ import annotations

import time

import pytest

from searchio.net.blocks import classify

H = {"content-type": "text/html"}
PROSE = "<p>" + ("Real content sentence about kayaks and paddles. " * 60) + "</p>"


class TestClassifyIsBounded:
    # Bug 122: the any-tag regex `<[^>]+>` scanned to end-of-body from every
    # '<' when no '>' followed -- a 2 MB run of '<' (or of unclosed "<a")
    # wedged classify(), and classify runs on every fetched body (bug 47's
    # sibling on the other regex).
    @pytest.mark.parametrize("body", ["<" * 2_000_000, "<a" * 500_000, "<" + "a" * 2_000_000, "<script>" + "<" * 1_000_000],
                             ids=["lt_run", "open_a_tags", "one_long_tag", "script_then_lt"])
    def test_hostile_bodies_classify_in_bounded_time(self, body):
        t0 = time.perf_counter()
        classify(200, H, body)
        assert time.perf_counter() - t0 < 3.0


class TestMarkersNeedAnEmptyPage:
    # Bug 123: a real page carrying "please enable JavaScript" in a comments
    # footer (Disqus, every comment widget) was a js_required shell -- the
    # cheap tier's rich page discarded and a browser booted for nothing.
    def test_js_notice_on_a_real_page_is_not_a_shell(self):
        v = classify(200, H, PROSE + "<footer>Please enable JavaScript to view the comments.</footer>")
        assert v.ok and v.reason == "ok"

    def test_js_notice_on_an_empty_page_is_a_shell(self):
        v = classify(200, H, "<html><body><p>Please enable JavaScript to continue.</p></body></html>")
        assert not v.ok and v.reason == "js_required"

    # Bug 124: strong wall markers had no size gate, so an article QUOTING
    # "checking your browser before accessing" (or containing the everyday
    # "just a moment") was refused as blocked and a rescue launched.
    @pytest.mark.parametrize("phrase", ["Just a moment ago the market opened.",
                                        "Cloudflare shows 'checking your browser before accessing' on its interstitial."])
    def test_wall_phrases_inside_real_prose_are_not_a_block(self, phrase):
        v = classify(200, H, PROSE + f"<p>{phrase}</p>")
        assert v.ok and not v.blocked, v

    def test_real_interstitial_still_blocked(self):
        v = classify(200, H, "<html><head><title>Just a moment...</title></head><body><p>Checking your browser before accessing example.com</p></body></html>")
        assert v.blocked

    def test_wall_title_on_a_padded_page_is_still_a_block(self):
        # The title is the vendor's signature: a padded challenge page (some
        # ship kilobytes of explanatory text) is still the wall.
        v = classify(200, H, "<html><head><title>Attention Required! | Cloudflare</title></head><body>" + PROSE + "</body></html>")
        assert v.blocked


class TestMarkupOnlyChallenges:
    # Bug 125: DataDome's challenge is a <script src=...captcha-delivery.com>
    # with no visible words, but "captcha-delivery.com" was checked against
    # the VISIBLE text -- the wall classified as too_thin and no rescue ran.
    def test_datadome_script_is_a_block(self):
        v = classify(200, H, '<html><head><title>x</title></head><body><script src="https://ct.captcha-delivery.com/c.js"></script></body></html>')
        assert v.blocked and "captcha-delivery" in v.reason

    def test_cf_browser_verification_markup_is_a_block(self):
        v = classify(200, H, '<html><body><div id="cf-browser-verification"></div></body></html>')
        assert v.blocked


class TestSoftErrorPages:
    # Rider: Amazon's "Sorry! Something went wrong" dog page (126 visible
    # chars, over min_text) scored as an ok retrieval.
    def test_something_went_wrong_page_is_not_content(self):
        v = classify(200, H, "<html><head><title>Sorry! Something went wrong!</title></head><body><img alt='Dogs of Amazon'>"
                             "<p>Sorry! Something went wrong on our end. Please go back and try again or go to Amazon's home page.</p></body></html>")
        assert not v.ok and v.reason.startswith("soft_error") and not v.blocked

    def test_a_real_page_mentioning_an_error_is_fine(self):
        v = classify(200, H, PROSE + "<p>If something went wrong, contact support.</p>")
        assert v.ok


class TestReviewRiders:
    def test_text_plain_wall_is_not_structured_data(self):
        # DeepSeek #1: the structured short-circuit ran before the wall
        # checks, so "Please complete the CAPTCHA" as text/plain was an ok
        # "structured" retrieval.
        v = classify(200, {"content-type": "text/plain"}, "Please complete the CAPTCHA to continue", content_type="text/plain")
        assert v.blocked

    def test_text_plain_data_still_structured(self):
        v = classify(200, {"content-type": "text/plain"}, "a,b,c\n1,2,3\n", content_type="text/plain")
        assert v.ok and v.reason == "structured"

    def test_vendor_503_with_a_challenge_body_is_a_challenge(self):
        hd = {"content-type": "text/html", "server": "cloudflare"}
        v = classify(503, hd, "<html><head><title>Just a moment...</title></head><body></body></html>")
        assert v.blocked and v.reason == "http_503_challenge"

    def test_header_names_are_case_insensitive(self):
        # Drop-list rider: classify read headers["content-type"] literally,
        # so a caller passing "Content-Type: image/png" got the html path
        # and a 500-byte PNG scored as an ok page.
        v = classify(200, {"Content-Type": "image/png"}, "x" * 500)
        assert not v.ok and v.reason.startswith("not_html")
        v = classify(200, {"Content-Type": "application/json"}, "[]")
        assert v.ok and v.reason == "structured"

    def test_vendor_503_origin_outage_is_not_a_challenge(self):
        # DeepSeek #6: a CDN-fronted origin that is down answers 503 with the
        # vendor's headers; that is an outage, and booting a browser at it
        # is bug 10's shape.
        hd = {"content-type": "text/html", "server": "cloudflare"}
        v = classify(503, hd, "<html><body><h1>Origin is unreachable</h1><p>Error 523</p></body></html>")
        assert not v.blocked and v.reason == "http_503"
