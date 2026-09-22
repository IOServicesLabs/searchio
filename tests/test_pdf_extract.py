"""Pins for the PDF text-extraction capability (net/pdf.py) and the ladder
trigger that fires it. The fixture PDFs come from build_pdf itself -- the
suite never depends on a captured binary drifting."""

from __future__ import annotations

import pytest

from searchio.net import pdf as pdf_mod
from searchio.net.blocks import classify
from searchio.net.ladder import _PDF_REASONS


# ── build/extract round-trip ────────────────────────────────────────────────


def test_roundtrip_multipage():
    data = pdf_mod.build_pdf(["alpha page one", "beta page two", "gamma page three"])
    assert pdf_mod.looks_like_pdf(data)
    text, used, total = pdf_mod.extract_text(data)
    assert (used, total) == (3, 3)
    for needle in ("alpha page one", "beta page two", "gamma page three"):
        assert needle in text


def test_roundtrip_escapes_parens_and_backslash():
    # Literal-string metacharacters must survive the Tj operand round-trip.
    data = pdf_mod.build_pdf([r"cost (est) \100"])
    text, used, total = pdf_mod.extract_text(data)
    assert (used, total) == (1, 1)
    assert "cost (est)" in text
    assert "\\100" in text


def test_page_cap_binds_and_reports():
    pages = [f"page number {i}" for i in range(40)]
    data = pdf_mod.build_pdf(pages)
    text, used, total = pdf_mod.extract_text(data, max_pages=30)
    assert (used, total) == (30, 40)
    assert "page number 0" in text
    assert "page number 29" in text
    assert "page number 30" not in text


def test_blank_pages_extract_empty():
    # The scanned-PDF shape: parseable, but no text. The ladder turns this
    # into pdf_text:no_text, an honest refusal -- no tier can OCR.
    data = pdf_mod.build_pdf(["", ""])
    text, used, total = pdf_mod.extract_text(data)
    assert (used, total) == (2, 2)
    assert not text.strip()


def test_garbage_raises_unreadable():
    with pytest.raises(pdf_mod.PdfUnreadable):
        pdf_mod.extract_text(b"%PDF-1.4 this is not a real pdf body at all")


def test_non_pdf_bytes_fail_magic_check():
    assert not pdf_mod.looks_like_pdf(b"<html><body>nope</body></html>")
    assert not pdf_mod.looks_like_pdf(b"")


# ── ladder trigger: both PDF verdict reasons must be wired ──────────────────


def _classify_pdf_envelope():
    body = "%PDF-1.4 fake"  # body decoded to str, as the tier contract hands it
    return classify(200, {"content-type": "application/pdf"}, body,
                    content_type="application/pdf", min_text=120)


def _classify_pdf_magic():
    # Mislabeled as HTML but the body carries the PDF magic -- classify's
    # byte sniff is what catches it (bug-20 fix family).
    body = "%PDF-1.4 fake"
    return classify(200, {"content-type": "text/html"}, body,
                    content_type="text/html", min_text=120)


def test_pdf_envelope_verdict_is_a_trigger_reason():
    v = _classify_pdf_envelope()
    assert not v.ok
    assert v.reason in _PDF_REASONS


def test_pdf_magic_sniff_verdict_is_a_trigger_reason():
    v = _classify_pdf_magic()
    assert not v.ok
    assert v.reason in _PDF_REASONS


def test_trigger_reasons_are_exactly_the_pdf_pair():
    assert _PDF_REASONS == ("not_html:application/pdf", "binary:application/pdf")
