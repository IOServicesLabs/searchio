"""engine.py second pass (iteration 60): the inline-operator extractor and
the locale mapper probed with what agents actually write.

Every test here bit RED before its fix.
"""

from __future__ import annotations

import pytest

from searchio.engine import _extract_operators, _locale_fields


class TestWildcardAndListedSiteOperators:
    def test_wildcard_site_is_the_suffix(self):
        # Bug 130: site:*.gov became the literal host "*.gov"; the router's
        # dot-boundary filter then matched nothing and the agent got an
        # honest-but-useless empty (bug 50's class).
        text, dom, exc = _extract_operators("site:*.gov budget 2026", None, None)
        assert (text, dom) == ("budget 2026", ["gov"])

    def test_comma_list_is_several_hosts(self):
        text, dom, exc = _extract_operators("site:nasa.gov,esa.int moon", None, None)
        assert dom == ["nasa.gov", "esa.int"] and text == "moon"

    def test_dangling_or_is_dropped(self):
        text, dom, exc = _extract_operators("site:nasa.gov OR site:esa.int moon", None, None)
        assert dom == ["nasa.gov", "esa.int"] and text == "moon"

    @pytest.mark.parametrize("q", ["moon (site:nasa.gov)", '"site:nasa.gov" moon', "moon [site:nasa.gov]"])
    def test_bracketed_or_quoted_operator_is_still_an_operator(self, q):
        text, dom, exc = _extract_operators(q, None, None)
        assert dom == ["nasa.gov"] and "site:" not in text and "moon" in text

    def test_duplicate_hosts_collapse(self):
        text, dom, exc = _extract_operators("site:NASA.gov site:nasa.gov moon", ["nasa.gov"], None)
        assert dom == ["nasa.gov"]

    def test_plain_words_untouched(self):
        for q in ("the campsite: nice", "on-site:yes", "website nasa.gov", "prerequisite: x"):
            assert _extract_operators(q, None, None) == (q, [], [])


class TestLocaleSubtags:
    # Bug 131: "zh-Hant-TW" mapped to locale "zh-HANT-TW" with NO region --
    # an invalid market tag for Bing (ignored -> en-US results) and no gl=
    # for YouTube; the region is the 2-letter (or 3-digit) subtag wherever
    # it sits, and the script subtag is title-cased.
    @pytest.mark.parametrize("tag,locale,region", [
        ("zh-Hant-TW", "zh-Hant-TW", "tw"), ("sr-Latn-RS", "sr-Latn-RS", "rs"), ("de-DE", "de-DE", "de"),
        ("de_de", "de-DE", "de"), ("DE", "de", ""), ("es-419", "es-419", ""), ("pt-br", "pt-BR", "br"),
    ])
    def test_mapping(self, tag, locale, region):
        out = _locale_fields(tag)
        assert out.get("locale") == locale and out.get("region", "") == region, out

    @pytest.mark.parametrize("tag", ["a" * 100, "123", "d", "!!", "de-DE; DROP TABLE", "en-US\nfoo"])
    def test_garbage_is_ignored(self, tag):
        assert _locale_fields(tag) == {}
