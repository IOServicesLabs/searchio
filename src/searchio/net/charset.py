"""Body decoding for the cheap tiers (charset_detect, #214).

The order a browser uses: byte-order mark, the Content-Type charset, the
``<meta>`` prescan of the first 1024 bytes, then UTF-8. A body that declares
nothing and is not valid UTF-8 goes to a statistical detector
(charset-normalizer when installed) before the lossy fallback.
"""

from __future__ import annotations

import codecs
import re

#: Set by the Ladder from Settings.charset_detect.
_ON = [False]

_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_META_RE = re.compile(rb"<meta\b[^>]*?charset\s*=\s*[\"']?\s*([a-zA-Z0-9_.:\-]+)", re.I)
_MULTIBYTE = re.compile(r"cp93|shift|euc|gb|big5|cp949|cp950|iso2022|johab|hz|utf")
#: WHATWG label quirks Python's codec registry does not share: browsers decode
#: these labels as their Windows supersets.
_ALIASES = {
    "iso-8859-1": "cp1252", "latin1": "cp1252", "latin-1": "cp1252", "ascii": "cp1252",
    "us-ascii": "cp1252", "iso8859-1": "cp1252", "iso_8859-1": "cp1252",
    "shift_jis": "cp932", "shift-jis": "cp932", "sjis": "cp932", "x-sjis": "cp932",
    "gb2312": "gbk", "gb_2312-80": "gbk", "x-gbk": "gbk",
    "ks_c_5601-1987": "cp949", "euc-kr": "cp949", "iso-8859-9": "cp1254",
    "iso-8859-11": "cp874", "tis-620": "cp874", "utf8": "utf-8",
}


def _codec(label: str | None) -> str | None:
    """A usable Python codec name for a declared label, else None."""
    if not label:
        return None
    name = label.strip().strip("\"'").lower()
    name = _ALIASES.get(name, name)
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def sniff_meta(raw: bytes) -> str | None:
    m = _META_RE.search(raw[:1024])
    return _codec(m.group(1).decode("ascii", "replace")) if m else None


def _texty(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return (not ct or ct.startswith("text/")
            or any(k in ct for k in ("html", "xml", "json", "javascript")))


def decode(raw: bytes, header_charset: str | None, content_type: str = "") -> str:
    """Decode ``raw``. With the switch off this is exactly the old behaviour:
    the header charset, else lossy UTF-8."""
    if not _ON[0]:
        return raw.decode(header_charset or "utf-8", errors="replace")
    for bom, name in _BOMS:
        if raw.startswith(bom):
            return raw.decode(name, errors="replace")
    declared = _codec(header_charset)
    if declared:
        return raw.decode(declared, errors="replace")
    if "html" in (content_type or "").lower() or not content_type:
        meta = sniff_meta(raw)
        if meta:
            return raw.decode(meta, errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if _texty(content_type) and b"\x00" not in raw[:4096]:
        try:
            from charset_normalizer import from_bytes

            matches = list(from_bytes(raw[:256 * 1024]))
            best = matches[0] if matches else None
            enc = _codec(best.encoding) if best is not None else None
            # Single-byte code pages are near-indistinguishable to a detector
            # (French scored as cp1257). When windows-1252 is a candidate it
            # wins: it is every browser's fallback for an undeclared page.
            if enc and not _MULTIBYTE.match(enc) and any(
                    m.encoding == "cp1252" for m in matches):
                enc = "cp1252"
            if enc and enc != "utf-8":
                return raw.decode(enc, errors="replace")
        except Exception:  # noqa: BLE001 -- the detector is best-effort
            pass
    return raw.decode("utf-8", errors="replace")
