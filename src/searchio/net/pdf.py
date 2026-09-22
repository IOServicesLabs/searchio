"""PDF text extraction with honesty bounds.

The ladder's tier contract hands :func:`searchio.net.blocks.classify` a
decoded ``str``, which is where PDFs die honestly today: ``not_html`` or the
body-magic sniff, an unusable verdict, a correct-but-empty refusal. Most of
the PDFs a search engine actually meets are digital-born documents whose text
extracts cleanly with pypdf -- refusing them wastes the fetch that already
happened.

This module is the whole capability: a byte-shape check, a bounded pypdf
extract, and the errors the ladder turns into honest escalations. It is
deliberately NOT wired into the tiers: extraction triggers once per fetch, on
the classify verdict, via a dedicated byte re-fetch (see ladder._pdf_text for
why the bytes are not threaded through the tier contract).

``build_pdf`` is the fixture source for the unit pins and bench/span.py's
control endpoints -- a programmatic minimal PDF (computed xref, base-14
Helvetica, one text line per page) so the suite never depends on a captured
binary fixture drifting. Production code never calls it.
"""

from __future__ import annotations

import io

#: Re-fetches larger than this are refused, not parsed: a hostile or broken
#: origin must not turn one PDF link into a memory event, and pypdf parse
#: time scales with the byte count.
DEFAULT_MAX_BYTES = 25 * 1024 * 1024

#: Extraction reads at most this many pages. Long documents front-load their
#: abstract/lead; the stamp the ladder emits says "used/total" so the caller
#: can see the truncation.
DEFAULT_MAX_PAGES = 30


class PdfUnreadable(Exception):
    """The bytes claim PDF but pypdf cannot parse them (truncated, encrypted,
    xref-rotten -- the honest answer is a refusal, not a guess)."""


class PdfTooLarge(Exception):
    """The byte fetch exceeded the configured cap."""


def looks_like_pdf(data: bytes) -> bool:
    """Byte-zero magic -- the re-fetch may have drawn a login wall or an HTML
    error page instead of the file, and only real PDF bytes go to pypdf."""
    return data.startswith(b"%PDF-")


def extract_text(
    data: bytes, max_pages: int = DEFAULT_MAX_PAGES
) -> tuple[str, int, int]:
    """Extract visible text from PDF bytes.

    Returns ``(text, pages_used, pages_total)``; ``pages_used < pages_total``
    means the page cap bound the extraction. Raises :class:`PdfUnreadable`
    when pypdf cannot parse the bytes at all. A parseable but textless PDF
    (scanned images) returns empty text -- the ladder refuses those honestly,
    no render tier can OCR either.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover -- dep is declared
        raise PdfUnreadable(f"pypdf not installed: {exc}") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        total = len(reader.pages)
        parts: list[str] = []
        used = 0
        # A cap below 1 is a cap of 1, not a negative slice that silently
        # drops the LAST pages (iteration 49 rider).
        for page in reader.pages[: max(1, int(max_pages))]:
            parts.append(page.extract_text() or "")
            used += 1
    except PdfUnreadable:
        raise
    except Exception as exc:  # pypdf raises a zoo of parse errors
        raise PdfUnreadable(str(exc)[:120]) from exc
    return "\n\n".join(parts), used, total


def _esc_pdf_text(s: str) -> str:
    """Escape a string for a PDF literal (the Tj operand)."""
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[str]) -> bytes:
    """Build a minimal, valid, multi-page PDF with one text line per page.

    Offsets for the xref table are computed from the bytes actually emitted,
    so the document stays valid no matter what the text is. One base-14 font,
    no compression -- the point is a deterministic fixture, not efficiency.
    """
    objects: list[bytes] = []
    # 1: catalog, 2: page tree, 3: font; then per page: page obj + content.
    n_pages = len(pages)
    page_obj_ids = [4 + 2 * i for i in range(n_pages)]
    kids = " ".join(f"{oid} 0 R" for oid in page_obj_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        content = (
            f"BT /F1 18 Tf 72 720 Td ({_esc_pdf_text(text)}) Tj ET".encode()
        )
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {4 + 2 * i + 1} 0 R >>"
        ).encode()
        stream = (
            b"<< /Length " + str(len(content)).encode()
            + b" >>\nstream\n" + content + b"\nendstream"
        )
        objects.append(page)
        objects.append(stream)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for oid, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{oid} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)
