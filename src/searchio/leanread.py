"""Cheaper representations of pages agents read often (batch 9).

A read of a Wikipedia article ships ~110 KB of skin and navigation on the wire
to deliver ~30 KB of prose; an arXiv PDF link ships 2 MB of PDF; a GitHub repo
page ships ~56 KB of chrome around a 3 KB README. Each site publishes the same
content in a lean form -- the MediaWiki extracts API, arXiv's HTML full text,
the raw README -- and on a slow link the difference is seconds per read.

Each rewrite has its own knob (so each is A/B'd alone) and returns
``(lean_url, build)``: ``build(body)`` turns the lean response into
``(title, text)``, or None when the body is not what the rewrite expects --
the caller then reads the original URL the normal way. A rewrite never
changes what the agent asked for: the Doc keeps the ORIGINAL url.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Callable

_WIKI = re.compile(r"^https?://([a-z\-]{2,12})\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)$")
_ARXIV_PDF = re.compile(r"^https?://(?:export\.)?arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)(?:\.pdf)?/?$")
_GH_REPO = re.compile(r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/?$")
_GH_BLOB = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/blob/([^?#]+?)/?(?:[?#].*)?$")
_HF = re.compile(r"^https?://huggingface\.co/(datasets/|spaces/)?([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/?$")
_CRATES = re.compile(r"^https?://crates\.io/crates/([A-Za-z0-9_\-]+)/?$")
_NPM = re.compile(r"^https?://(?:www\.)?npmjs\.com/package/((?:@[A-Za-z0-9_.\-]+/)?[A-Za-z0-9_.\-]+)/?$")
_GH_ISSUE = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/(?:issues|pull)/([0-9]+)/?(?:[?#].*)?$")
_RFC = re.compile(r"^https?://(?:www\.rfc-editor\.org/rfc/rfc|datatracker\.ietf\.org/doc/html/rfc"
                  r"|tools\.ietf\.org/html/rfc)([0-9]{1,5})(?:\.html)?/?(?:#.*)?$")
_GH_RELEASES = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/releases(?:/tag/([^/?#]+))?/?$")
_GH_COMMIT = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/commit/([0-9a-f]{7,40})/?$")
_K8S = re.compile(r"^https?://kubernetes\.io/(?:([a-z]{2}(?:-[a-z]{2})?)/)?docs/([^?#]*?)/?(?:[?#].*)?$")
_RUST_BOOK = re.compile(r"^https?://doc\.rust-lang\.org/(?:stable/|beta/|nightly/)?"
                        r"(book|reference|nomicon|rust-by-example|edition-guide)/"
                        r"([A-Za-z0-9_\-/]*?)(?:\.html)?(?:[?#].*)?$")
_GH_TREE = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/tree/([^/?#]+)((?:/[^?#]*?)?)/?(?:[?#].*)?$")
_GH_WIKI = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)/wiki(?:/([^/?#]+))?/?(?:[?#].*)?$")
_HF_DOCS = re.compile(r"^https?://huggingface\.co/docs/([a-z_\-]+)/(?:(?:main|v[0-9][0-9.]*)/)?"
                      r"(?:([a-z]{2})/)?([A-Za-z0-9_\-/]+?)/?(?:[?#].*)?$")
_REACT = re.compile(r"^https?://react\.dev/((?:reference|learn|blog|community)(?:/[A-Za-z0-9_\-]+)*)/?(?:[?#].*)?$")
_VUE = re.compile(r"^https?://vuejs\.org/((?:guide|api|tutorial|examples|about|ecosystem)/[A-Za-z0-9_\-/]*?)"
                  r"(?:\.html)?/?(?:[?#].*)?$")
_NODE_API = re.compile(r"^https?://nodejs\.org/(?:dist/latest/docs/)?api/([a-z0-9_]+)\.html(?:[?#].*)?$")
_DJANGO = re.compile(r"^https?://docs\.djangoproject\.com/en/([0-9]+\.[0-9]+|dev)/([a-z0-9_\-/]+?)/?(?:[?#].*)?$")
_FASTAPI = re.compile(r"^https?://fastapi\.tiangolo\.com/(?:([a-z]{2}(?:-[a-z]{2,4})?)/(?=[a-z]))?"
                      r"([a-z0-9_\-/]*?)/?(?:[?#].*)?$")
_STACK = re.compile(r"^https?://(?:www\.)?(stackoverflow\.com|superuser\.com|serverfault\.com|askubuntu\.com"
                    r"|mathoverflow\.net|[a-z0-9]+\.stackexchange\.com)/(?:questions|q)/([0-9]+)"
                    r"(?:/[^?#]*)?/?(?:[?#].*)?$")
_PUBMED = re.compile(r"^https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/([0-9]{1,9})/?(?:[?#].*)?$")
_PMC = re.compile(r"^https?://(?:pmc\.ncbi\.nlm\.nih\.gov|(?:www\.)?ncbi\.nlm\.nih\.gov/pmc)"
                  r"/articles/PMC([0-9]{1,9})/?(?:[?#].*)?$")
#: Publishers whose article pages refuse this client at tier 0 AND tier 1
#: (Cloudflare/Akamai 403, 2026-10-28 probe) and carry the DOI in the URL.
_DOI_HOSTS = ("onlinelibrary.wiley.com", "www.ahajournals.org", "www.tandfonline.com",
              "journals.sagepub.com", "pubs.acs.org", "www.nejm.org", "www.science.org", "www.pnas.org")
_PUB_DOI = re.compile(r"^https?://(" + "|".join(re.escape(h) for h in _DOI_HOSTS) + r")"
                      r"/doi/(?:(?:full|abs|epdf|pdf|epub|reader)/)?(10\.[0-9]{4,9}/[^?#]+?)/?(?:[?#].*)?$")
_GITLAB = re.compile(r"^https?://gitlab\.com/((?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+)/?$")
_GITLAB_RESERVED = {"explore", "users", "help", "dashboard", "groups", "search", "admin", "projects",
                    "-", "api", "assets", "uploads"}
_RXIV = re.compile(r"^https?://(?:www\.|connect\.)?(biorxiv|medrxiv)\.org/content/(10\.1101/[0-9.]+?)"
                   r"(?:v([0-9]+))?(?:\.(?:full|abstract|article-info|article-metrics|supplementary-material)"
                   r"(?:\.pdf)?|\.full-text)?/?(?:[?#].*)?$")
_S2 = re.compile(r"^https?://(?:www\.)?semanticscholar\.org/paper/(?:[^/?#]+/)?([0-9a-f]{40})/?(?:[?#].*)?$")
_PYDANTIC = re.compile(r"^https?://docs\.pydantic\.dev/(?:latest|dev)/([a-z0-9_\-/]+?)/?(?:[?#].*)?$")
#: huggingface.co/docs/<lib> -> the repo whose docs/source holds it.
_HF_DOC_REPOS = {"transformers": "transformers", "diffusers": "diffusers", "datasets": "datasets",
                 "peft": "peft", "accelerate": "accelerate", "trl": "trl",
                 "huggingface_hub": "huggingface_hub", "optimum": "optimum",
                 "evaluate": "evaluate", "safetensors": "safetensors", "smolagents": "smolagents"}
_RAW = "https://raw.githubusercontent.com"
#: A heading's explicit anchor (``{#id}`` / ``{ #id }``, VitePress and mkdocs).
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")
_PATH_SUFFIX = re.compile(r"\s*\(/[^()\s]*\)$")
_MDN = re.compile(r"^(https://developer\.mozilla\.org/[a-zA-Z\-]{2,8}/docs/[^?#]+?)/?$")
#: GitHub top-level paths that are not owners.
_GH_RESERVED = {"topics", "trending", "orgs", "settings", "marketplace", "explore",
                "features", "sponsors", "collections", "about", "pricing", "search"}

Build = Callable[[str], "tuple[str, str] | None"]


def _html_page(text: str) -> bool:
    """A served HTML document (an error page), not markdown that opens with
    inline HTML like ``<p align="center">`` -- many READMEs do."""
    head = text[:200].lstrip().lower()
    return head.startswith(("<!doctype", "<html", "<?xml"))


def _wiki(url: str) -> tuple[str, Build] | None:
    m = _WIKI.match(url)
    if not m:
        return None
    lang, raw_title = m.groups()
    title = urllib.parse.unquote(raw_title)
    if ":" in title:  # Special:, File:, Talk: ... -- not articles
        return None
    api = (f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts"
           f"&explaintext=1&redirects=1&format=json&formatversion=2"
           f"&titles={urllib.parse.quote(title)}")

    def build(body: str):
        try:
            pages = json.loads(body)["query"]["pages"]
        except (ValueError, KeyError, TypeError):
            return None
        page = pages[0] if pages else {}
        text = (page.get("extract") or "").strip()
        if page.get("missing") or len(text) < 200:
            return None
        return page.get("title") or title.replace("_", " "), text

    return api, build


def _arxiv(url: str) -> tuple[str, Build] | None:
    m = _ARXIV_PDF.match(url)
    if not m:
        return None
    lean = f"https://arxiv.org/html/{m.group(1)}"

    def build(body: str):
        from .extract import title_of, to_markdown

        if "ltx_document" not in body and "ltx_page_main" not in body:
            return None  # no HTML rendering for this paper: read the PDF
        text = to_markdown(body, lean)
        if len(text) < 500:
            return None
        return title_of(body), text

    return lean, build


def _github(url: str) -> tuple[str, Build] | None:
    m = _GH_REPO.match(url)
    if not m or m.group(1).lower() in _GH_RESERVED:
        return None
    owner, repo = m.groups()
    lean = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md"

    def build(body: str):
        text = body.strip()
        if len(text) < 80 or _html_page(text):
            return None  # 404 page, or no README.md (README.rst ...): read the page
        return f"{owner}/{repo}", text

    return lean, build


def _gitlab(url: str) -> tuple[str, Build] | None:
    """A gitlab.com project page -> its raw README.md. The page is ~58 KB of
    chrome that renders the README with JS (a read got ~300 chars of nav)."""
    m = _GITLAB.match(url)
    if not m:
        return None
    path = m.group(1)
    segs = path.split("/")
    if segs[0].lower() in _GITLAB_RESERVED or "-" in segs or path.endswith(".git"):
        return None
    lean = f"https://gitlab.com/{path}/-/raw/HEAD/README.md"

    def build(body: str):
        text = body.strip()
        if len(text) < 80 or _html_page(text):
            return None  # no README.md (README.rst, a sign-in page): read the page
        return path, text

    return lean, build


def _github_blob(url: str) -> tuple[str, Build] | None:
    m = _GH_BLOB.match(url)
    if not m:
        return None
    owner, repo, ref_path = m.groups()
    lean = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref_path}"
    name = ref_path.rsplit("/", 1)[-1]

    def build(body: str):
        if not body.strip():
            return None
        return f"{name} - {owner}/{repo}", body

    return lean, build


def _mdn(url: str) -> tuple[str, Build] | None:
    m = _MDN.match(url)
    if not m or m.group(1).endswith("/index.json"):
        return None
    lean = m.group(1) + "/index.json"

    def build(body: str):
        from .net.blocks import visible_text

        try:
            doc = json.loads(body)["doc"]
        except (ValueError, KeyError, TypeError):
            return None
        parts = []
        for sec in doc.get("body") or []:
            val = sec.get("value") or {}
            if val.get("title"):
                parts.append(f"## {val['title']}")
            if isinstance(val.get("content"), str):
                parts.append(visible_text(val["content"]))
        text = "\n\n".join(p for p in parts if p).strip()
        if len(text) < 200:
            return None
        return doc.get("title") or "", text

    return lean, build


def _hf(url: str) -> tuple[str, Build] | None:
    m = _HF.match(url)
    if not m:
        return None
    kind, owner, name = m.groups()
    if not kind and owner in ("docs", "blog", "models", "datasets", "spaces", "papers",
                              "learn", "join", "login", "pricing", "settings"):
        return None
    lean = f"https://huggingface.co/{kind or ''}{owner}/{name}/raw/main/README.md"

    def build(body: str):
        text = body.strip()
        if len(text) < 40 or _html_page(text):
            return None
        return f"{owner}/{name}", text

    return lean, build


def _crates(url: str) -> tuple[str, Build] | None:
    m = _CRATES.match(url)
    if not m:
        return None
    lean = f"https://docs.rs/crate/{m.group(1)}/latest"

    def build(body: str):
        from .extract import title_of, to_markdown

        text = to_markdown(body, lean)
        if len(text) < 200:
            return None
        return title_of(body) or m.group(1), text

    return lean, build


def _npm(url: str) -> list[tuple[str, Build]] | None:
    m = _NPM.match(url)
    if not m:
        return None
    pkg = m.group(1)

    def build(body: str):
        text = body.strip()
        if len(text) < 40 or _html_page(text):
            return None
        return pkg, text

    # unpkg resolves the README's spelling and the latest version itself.
    return [(f"https://unpkg.com/{pkg}/README.md", build)]


def _github_issue(url: str):
    m = _GH_ISSUE.match(url)
    if not m:
        return None
    owner, repo, num = m.groups()
    api = f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
    # The issues API serves pull requests too; the comments are a second,
    # concurrent call. Unauthenticated it is 60/h: a 403 falls back to the page.
    lean = (api, api + "/comments?per_page=100")

    def build(issue_body: str, comments_body: str):
        try:
            issue, comments = json.loads(issue_body), json.loads(comments_body)
        except ValueError:
            return None
        if not isinstance(issue, dict) or "title" not in issue or not isinstance(comments, list):
            return None

        def who(o):
            return (o.get("user") or {}).get("login", "?")

        parts = [f"# {issue['title']} #{num}",
                 f"{issue.get('state', '')} - opened by {who(issue)} {issue.get('created_at', '')}",
                 (issue.get("body") or "").strip()]
        for c in comments:
            if isinstance(c, dict):
                parts.append(f"---\n**{who(c)}** {c.get('created_at', '')}\n\n"
                             f"{(c.get('body') or '').strip()}")
        return f"{issue['title']} - {owner}/{repo}#{num}", "\n\n".join(p for p in parts if p)

    return lean, build


def _github_tree(url: str):
    """A GitHub directory page -> the contents API listing plus that
    directory's raw README.md, concurrently; the listing alone when the
    directory has no README.md. The page is a heavy React app (and
    github.com's robots.txt disallows /*/tree/; the API and raw hosts do not).
    A ref with a slash (``feature/x``) misses the API and reads the page."""
    m = _GH_TREE.match(url)
    if not m or m.group(1).lower() in _GH_RESERVED:
        return None
    owner, repo, ref, sub = m.groups()
    sub = sub.strip("/")
    api = (f"https://api.github.com/repos/{owner}/{repo}/contents/{sub}"
           f"?ref={urllib.parse.quote(ref, safe='')}")
    readme = f"{_RAW}/{owner}/{repo}/{ref}/{sub + '/' if sub else ''}README.md"
    where = f"{owner}/{repo}/{sub}" if sub else f"{owner}/{repo}"

    def listing(body: str):
        try:
            items = json.loads(body)
        except ValueError:
            return None
        if not isinstance(items, list) or not items or not all(
                isinstance(i, dict) and "name" in i and "type" in i for i in items):
            return None
        items.sort(key=lambda i: (i["type"] != "dir", i["name"].lower()))
        rows = [f"- {i['name']}{'/' if i['type'] == 'dir' else ''}" for i in items]
        return f"# {where} ({ref})\n\n" + "\n".join(rows)

    def build_list(body: str):
        text = listing(body)
        return (f"{where} at {ref}", text) if text else None

    def build_both(body: str, readme_body: str):
        text = listing(body)
        rd = readme_body.strip()
        if not text or len(rd) < 80 or _html_page(rd):
            return None
        return f"{where} at {ref}", f"{text}\n\n---\n\n{rd}"

    return [((api, readme), build_both), (api, build_list)]


def _rfc(url: str) -> tuple[str, Build] | None:
    m = _RFC.match(url)
    if not m:
        return None
    num = int(m.group(1))
    lean = f"https://www.rfc-editor.org/rfc/rfc{num}.txt"

    def build(body: str):
        text = body.strip()
        if len(text) < 500 or _html_page(text):
            return None
        title = f"RFC {num}"
        lines = [ln.strip() for ln in text.lstrip("﻿").strip().splitlines()[:60]]
        if "" in lines:
            # The title is the first line after the blank that ends the header block.
            rest = [ln for ln in lines[lines.index(""):] if ln]
            if rest and rest[0] != "Abstract":
                title = f"RFC {num}: {rest[0]}"
        return title, text

    return lean, build


def _github_releases(url: str):
    m = _GH_RELEASES.match(url)
    if not m:
        return None
    owner, repo, tag = m.groups()
    base = f"https://api.github.com/repos/{owner}/{repo}/releases"
    lean = f"{base}/tags/{tag}" if tag else f"{base}?per_page=10"

    def build(body: str):
        try:
            data = json.loads(body)
        except ValueError:
            return None
        rels = [data] if isinstance(data, dict) else data
        if not isinstance(rels, list) or not rels or not all(
                isinstance(r, dict) and "tag_name" in r for r in rels):
            return None
        parts = [f"## {r.get('name') or r['tag_name']} ({r['tag_name']}) "
                 f"{r.get('published_at') or ''}\n\n{(r.get('body') or '').strip()}"
                 for r in rels]
        head = f"{tag} - " if tag else "Releases - "
        return f"{head}{owner}/{repo}", "\n\n".join(parts)

    return lean, build


def _github_commit(url: str) -> tuple[str, Build] | None:
    m = _GH_COMMIT.match(url)
    if not m:
        return None
    owner, repo, sha = m.groups()
    lean = f"https://github.com/{owner}/{repo}/commit/{sha}.patch"

    def build(body: str):
        if not body.startswith("From ") or _html_page(body):
            return None
        subject = next((ln[len("Subject: "):] for ln in body.splitlines()[:12]
                        if ln.startswith("Subject: ")), sha)
        subject = re.sub(r"^\[PATCH[^\]]*\]\s*", "", subject)
        return f"{subject} - {owner}/{repo}@{sha[:7]}", body

    return lean, build


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    """Hugo/Jekyll ``---`` front matter -> (flat key: value, rest)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    meta = {}
    for ln in text[3:end].splitlines():
        k, sep, v = ln.partition(":")
        if sep and k and not k.startswith((" ", "-")):
            meta[k.strip()] = v.strip().strip("\"'")
    return meta, text[end + 4:].lstrip("-\n")


def _k8s(url: str):
    m = _K8S.match(url)
    if not m:
        return None
    lang, path = m.group(1) or "en", m.group(2).strip("/")
    if not path:
        return None
    base = f"https://raw.githubusercontent.com/kubernetes/website/HEAD/content/{lang}/docs/{path}"

    def build(body: str):
        if _html_page(body) or body.startswith("404"):
            return None
        meta, text = _front_matter(body.strip())
        if len(text) < 200:
            return None
        return meta.get("title") or path.rsplit("/", 1)[-1], text

    # A page is <path>.md; a section page is <path>/_index.md.
    return [(f"{base}.md", build), (f"{base}/_index.md", build)]


def _rust_book(url: str) -> tuple[str, Build] | None:
    m = _RUST_BOOK.match(url)
    if not m:
        return None
    book, page = m.groups()
    if page in ("", "index", "print", "title-page"):
        return None
    lean = f"https://raw.githubusercontent.com/rust-lang/{book}/HEAD/src/{page}.md"

    def build(body: str):
        text = body.strip()
        if len(text) < 100 or _html_page(text) or text.startswith("404"):
            return None
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), page)
        return title, text

    return lean, build


def _md_title(text: str, fallback: str) -> str:
    """A markdown/rst page's title: front matter, a ``# `` heading, or an
    rst title (a line underlined with ``===``)."""
    meta, body = _front_matter(text)
    if meta.get("title"):
        return _ANCHOR.sub("", meta["title"])
    lines = body.splitlines()[:40]
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            # "Title (/docs/the/page)": some hosts append the page path.
            return _PATH_SUFFIX.sub("", _ANCHOR.sub("", ln[2:].strip()))
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if ln.strip() and len(nxt) >= 3 and set(nxt.strip()) <= set("=#*") and nxt.strip():
            return ln.strip()
    return fallback.replace("-", " ").replace("_", " ").capitalize()


def _source_build(fallback: str, min_len: int = 200) -> Build:
    """Build for a docs page read as its source file (markdown or rst)."""

    def build(body: str):
        text = body.strip()
        if len(text) < min_len or _html_page(text) or text.startswith("404"):
            return None
        title = _md_title(text, fallback)
        _meta, rest = _front_matter(text)
        return title, rest.strip() or text

    return build


def _candidates(base: str, exts: tuple[str, ...], fallback: str):
    build = _source_build(fallback)
    return [(base + ext, build) for ext in exts]


def _github_wiki(url: str):
    m = _GH_WIKI.match(url)
    if not m:
        return None
    owner, repo, page = m.groups()
    page = page or "Home"
    return _candidates(f"{_RAW}/wiki/{owner}/{repo}/{page}", (".md",), page.replace("-", " "))


def _hf_docs(url: str):
    m = _HF_DOCS.match(url)
    if not m:
        return None
    lib, lang, path = m.groups()
    repo = _HF_DOC_REPOS.get(lib)
    if repo is None or path in ("index", ""):
        return None
    root = f"{_RAW}/huggingface/{repo}/main/docs/source"
    # Most repos keep docs/source/<lang>/; a few (peft) have no language level.
    return (_candidates(f"{root}/{lang or 'en'}/{path}", (".md", ".mdx"), path.rsplit("/", 1)[-1])
            + ([] if lang not in (None, "en") else _candidates(f"{root}/{path}", (".md", ".mdx"), path.rsplit("/", 1)[-1])))


def _react(url: str):
    m = _REACT.match(url)
    if not m:
        return None
    path = m.group(1)
    base = f"{_RAW}/reactjs/react.dev/main/src/content/{path}"
    return _candidates(base, (".md", "/index.md"), path.rsplit("/", 1)[-1])


def _vue(url: str):
    m = _VUE.match(url)
    if not m:
        return None
    path = m.group(1).rstrip("/")
    base = f"{_RAW}/vuejs/docs/main/src/{path}"
    return _candidates(base, (".md", "/index.md"), path.rsplit("/", 1)[-1])


def _node_api(url: str):
    m = _NODE_API.match(url)
    if not m or m.group(1) in ("index", "all"):
        return None
    # main's doc/api, not the release's: the same module docs, plus any
    # not-yet-released additions (marked "Added in: REPLACEME").
    return _candidates(f"{_RAW}/nodejs/node/main/doc/api/{m.group(1)}", (".md",), m.group(1))


def _django(url: str):
    m = _DJANGO.match(url)
    if not m:
        return None
    ver, path = m.groups()
    branch = "main" if ver == "dev" else f"stable/{ver}.x"
    base = f"{_RAW}/django/django/{branch}/docs/{path}"
    return _candidates(base, (".txt", "/index.txt"), path.rsplit("/", 1)[-1])


def _fastapi(url: str):
    m = _FASTAPI.match(url)
    if not m:
        return None
    lang, path = m.group(1) or "en", m.group(2).strip("/")
    base = f"{_RAW}/fastapi/fastapi/master/docs/{lang}/docs"
    if not path:
        return _candidates(f"{base}/index", (".md",), "FastAPI")
    return _candidates(f"{base}/{path}", (".md", "/index.md"), path.rsplit("/", 1)[-1])


def _pydantic(url: str):
    m = _PYDANTIC.match(url)
    if not m or m.group(1).startswith("api/"):
        return None  # api/ pages are generated from docstrings: no source file
    path = m.group(1)
    base = f"{_RAW}/pydantic/pydantic/main/docs/{path}"
    return _candidates(base, (".md", "/index.md"), path.rsplit("/", 1)[-1])


def _stack_html(html: str) -> str:
    from .extract import to_markdown

    return to_markdown(f"<html><body>{html}</body></html>", "").strip()


def _stackexchange(url: str):
    """A Stack Exchange question -> the public API's question + answers (two
    concurrent calls). The site pages answer this client with a Cloudflare
    challenge at tier 0 AND tier 1; the API serves the same posts as JSON."""
    m = _STACK.match(url)
    if not m:
        return None
    host, qid = m.groups()
    site = host.split(".stackexchange.com")[0] if host.endswith(".stackexchange.com") else host.split(".")[0]
    api = "https://api.stackexchange.com/2.3/questions/" + qid
    q = f"?site={site}&filter=withbody"
    lean = (api + q, api + "/answers" + q + "&sort=votes&order=desc&pagesize=10")

    def build(q_body: str, a_body: str):
        try:
            qd, ad = json.loads(q_body), json.loads(a_body)
        except ValueError:
            return None
        items = qd.get("items") if isinstance(qd, dict) else None
        if not items or not isinstance(ad, dict) or not isinstance(ad.get("items"), list):
            return None
        import html as _h

        qi = items[0]
        title = _h.unescape(qi.get("title", ""))
        parts = [f"# {title}",
                 f"score {qi.get('score', 0)} - tags: {', '.join(qi.get('tags') or [])}",
                 _stack_html(qi.get("body", ""))]
        for a in ad["items"]:
            mark = " (accepted)" if a.get("is_accepted") else ""
            parts.append(f"---\n**Answer{mark}** score {a.get('score', 0)}\n\n{_stack_html(a.get('body', ''))}")
        return title, "\n\n".join(p for p in parts if p)

    return lean, build


def _pubmed(url: str):
    """A PubMed record -> NCBI E-utilities' plain-text abstract. The page
    answers this client with a Google reCAPTCHA challenge; efetch serves the
    citation, authors, abstract and identifiers (~3 KB, ~0.3 s)."""
    m = _PUBMED.match(url)
    if not m:
        return None
    pmid = m.group(1)
    lean = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text")

    def build(body: str):
        text = (body or "").strip()
        if not text.startswith("1. ") or _html_page(text) or f"PMID: {pmid}" not in text:
            return None
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        title = " ".join(paras[1].split()) if len(paras) > 1 else f"PMID {pmid}"
        return title, text

    return lean, build


def _jats_text(el) -> str:
    return " ".join("".join(el.itertext()).split())


def _pmc(url: str):
    """A PMC article -> NCBI efetch's JATS XML, rendered as title, abstract
    and the body's section headings + paragraphs. The page answers this
    client with a Google reCAPTCHA challenge, like PubMed's."""
    m = _PMC.match(url)
    if not m:
        return None
    pmcid = m.group(1)
    lean = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"

    def build(body: str):
        import xml.etree.ElementTree as ET

        if "<!ENTITY" in (body or ""):
            return None  # never expand an internal DTD subset
        try:
            root = ET.fromstring((body or "").encode("utf-8"))
        except ET.ParseError:
            return None
        art = root.find(".//article")
        if art is None:
            return None
        t_el = art.find(".//front//article-title")
        title = _jats_text(t_el) if t_el is not None else f"PMC{pmcid}"
        parts = [f"# {title}"]
        for ab in art.findall(".//front//abstract"):
            parts.append("## Abstract")
            parts.extend(_jats_text(p) for p in ab.iter("p"))
        bod = art.find("body")
        if bod is not None:
            for el in bod.iter():
                if el.tag == "title":
                    parts.append("## " + _jats_text(el))
                elif el.tag == "p":
                    parts.append(_jats_text(el))
        text = "\n\n".join(p for p in parts if p)
        return (title, text) if len(parts) > 2 else None

    return lean, build


def _strip_tags(s: str) -> str:
    import html as _h

    return " ".join(_h.unescape(re.sub(r"<[^>]+>", " ", s or "")).split())


def _publisher_doi(url: str):
    """A walled publisher's article page -> its Crossref record (title,
    authors, venue, year, abstract) with OpenAlex fetched alongside for the
    abstract Crossref lacks and an open-access copy's URL. The page itself
    is a 403 at every tier this client has; this is the paper's metadata
    and abstract, never its full text. OpenAlex refusing (its per-IP daily
    budget) leaves the Crossref-only candidate."""
    m = _PUB_DOI.match(url)
    if not m:
        return None
    return doi_candidates(urllib.parse.unquote(m.group(2)))


#: A DOI inside a URL path (``/doi/full/10.1145/3686994``, ``/article/10.1088/...``).
_DOI_IN_URL = re.compile(r"(10\.[0-9]{4,9}/[^\s?#&]+)")
_DOI_TAIL = re.compile(r"(?:/(?:full|abstract|abs|pdf|epdf|epub|fulltext|summary|meta|html))+/?$|\.pdf$|/$")


def doi_in_url(url: str) -> str | None:
    """doi_fallback: the DOI a URL's path names, if any."""
    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
    m = _DOI_IN_URL.search(path)
    if not m:
        return None
    doi = _DOI_TAIL.sub("", m.group(1))
    return doi if "/" in doi and len(doi.split("/", 1)[1]) >= 2 else None


def doi_candidates(doi: str):
    """Crossref (+ OpenAlex alongside) lean candidates for one DOI."""
    q = urllib.parse.quote(doi, safe="/")
    cr = f"https://api.crossref.org/works/{q}"
    oa = (f"https://api.openalex.org/works/doi:{q}"
          "?select=title,abstract_inverted_index,open_access,publication_year")

    def build(cr_body: str, oa_body: str = ""):
        try:
            msg = json.loads(cr_body).get("message")
        except (ValueError, AttributeError):
            return None
        if not isinstance(msg, dict) or not msg.get("title"):
            return None
        title = _strip_tags((msg.get("title") or [""])[0])
        authors = ", ".join(" ".join(x for x in (a.get("given"), a.get("family")) if x)
                            for a in (msg.get("author") or [])[:12] if isinstance(a, dict))
        venue = (msg.get("container-title") or [""])[0]
        year = ((msg.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        abstract = _strip_tags(msg.get("abstract") or "")
        oa_url = ""
        if oa_body:
            try:
                w = json.loads(oa_body)
            except ValueError:
                w = {}
            inv = w.get("abstract_inverted_index") if isinstance(w, dict) else None
            if not abstract and isinstance(inv, dict) and inv:
                pos = sorted((i, word) for word, idx in inv.items() for i in idx)
                abstract = " ".join(word for _, word in pos)
            oa_url = ((w.get("open_access") or {}).get("oa_url") or "") if isinstance(w, dict) else ""
        parts = [f"# {title}", ", ".join(str(x) for x in (venue, year) if x), authors,
                 f"DOI: {doi}", "## Abstract" if abstract else "", abstract,
                 f"Open-access copy: {oa_url}" if oa_url else ""]
        return title, "\n\n".join(p for p in parts if p)

    return [((cr, oa), build), (cr, build)]


def _rxiv(url: str):
    """A bioRxiv/medRxiv preprint page -> the server's details API (title,
    authors, date, category, abstract of the asked version, else the latest).
    The page drew a 3.5 s tier-1 climb and ~800 chars of navigation."""
    m = _RXIV.match(url)
    if not m:
        return None
    server, doi, ver = m.groups()
    lean = f"https://api.biorxiv.org/details/{server}/{doi}"

    def build(body: str):
        try:
            coll = json.loads(body).get("collection")
        except (ValueError, AttributeError):
            return None
        if not isinstance(coll, list) or not coll:
            return None
        pick = next((c for c in coll if ver and str(c.get("version")) == ver), coll[-1])
        title = (pick.get("title") or "").strip()
        if not title:
            return None
        parts = [f"# {title}", f"{server} {pick.get('date', '')} v{pick.get('version', '')} - "
                 f"{pick.get('category', '')}", (pick.get("authors") or "").strip(),
                 f"DOI: {doi}", "## Abstract", (pick.get("abstract") or "").strip()]
        return title, "\n\n".join(p for p in parts if p)

    return lean, build


def _semantic_scholar(url: str):
    """A Semantic Scholar paper page (a JS app: tier 0 reads 0 chars, no
    browser here) -> the Graph API record: title, venue, year, authors,
    TLDR, abstract, ids, open-access PDF. Unkeyed it is a shared pool; a 429
    falls back to the page."""
    m = _S2.match(url)
    if not m:
        return None
    pid = m.group(1)
    lean = (f"https://api.semanticscholar.org/graph/v1/paper/{pid}"
            "?fields=title,abstract,authors,year,venue,externalIds,openAccessPdf,tldr")

    def build(body: str):
        try:
            p = json.loads(body)
        except ValueError:
            return None
        if not isinstance(p, dict) or not p.get("title"):
            return None
        ids = p.get("externalIds") or {}
        tldr = (p.get("tldr") or {}).get("text") or ""
        pdf = (p.get("openAccessPdf") or {}).get("url") or ""
        parts = [f"# {p['title']}", ", ".join(str(x) for x in (p.get("venue"), p.get("year")) if x),
                 ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:12] if isinstance(a, dict)),
                 " | ".join(f"{k}: {v}" for k, v in ids.items() if k in ("DOI", "ArXiv", "PubMed")),
                 f"TLDR: {tldr}" if tldr else "", "## Abstract" if p.get("abstract") else "",
                 p.get("abstract") or "", f"Open-access PDF: {pdf}" if pdf else ""]
        return p["title"], "\n\n".join(x for x in parts if x)

    return lean, build


#: knob -> rewrite
REWRITES: tuple[tuple[str, Callable[[str], "tuple[str, Build] | None"]], ...] = (
    ("lean_read_wikipedia", _wiki),
    ("lean_read_arxiv", _arxiv),
    ("lean_read_github", _github),
    ("lean_read_mdn", _mdn),
    ("lean_read_github_blob", _github_blob),
    ("lean_read_hf", _hf),
    ("lean_read_crates", _crates),
    ("lean_read_npm", _npm),
    ("lean_read_github_issue", _github_issue),
    ("lean_read_rfc", _rfc),
    ("lean_read_github_releases", _github_releases),
    ("lean_read_github_commit", _github_commit),
    ("lean_read_k8s", _k8s),
    ("lean_read_rust_book", _rust_book),
    ("lean_read_github_wiki", _github_wiki),
    ("lean_read_hf_docs", _hf_docs),
    ("lean_read_react", _react),
    ("lean_read_vue", _vue),
    ("lean_read_node_api", _node_api),
    ("lean_read_django", _django),
    ("lean_read_fastapi", _fastapi),
    ("lean_read_pydantic", _pydantic),
    ("lean_read_stackexchange", _stackexchange),
    ("lean_read_pubmed", _pubmed),
    ("lean_read_pmc", _pmc),
    ("lean_read_publisher_doi", _publisher_doi),
    ("lean_read_gitlab", _gitlab),
    ("lean_read_rxiv", _rxiv),
    ("lean_read_github_tree", _github_tree),
    ("lean_read_semantic_scholar", _semantic_scholar),
)


def lean_targets(url: str, settings) -> list[tuple[str, Build]]:
    """Candidate lean fetches for ``url``, tried in order (empty: none)."""
    for knob, rewrite in REWRITES:
        if getattr(settings, knob, False):
            hit = rewrite(url)
            if hit is not None:
                return hit if isinstance(hit, list) else [hit]
    return []


def lean_target(url: str, settings) -> tuple[str, Build] | None:
    """The first candidate (what a prefetch warms)."""
    hits = lean_targets(url, settings)
    return hits[0] if hits else None



# ── lean_md_alternate: learned, not hand-written ────────────────────────────
#
# Many docs sites (Mintlify, Vercel, Docker, Supabase, Cloudflare, GitHub
# Docs, ...) advertise a markdown rendering of every page in the head:
# ``<link rel="alternate" type="text/markdown" href="...">``. One normal read
# on such a host shows how its markdown URL is built from the page path; the
# template is kept per host (persisted), and later reads there fetch the
# markdown instead of the page.

_ALT_LINK = re.compile(r"<link\b[^>]*>", re.I)
_ATTR = re.compile(r"""([a-zA-Z\-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")


def markdown_alternate(page_url: str, html: str) -> str | None:
    """The page's advertised markdown alternate, absolute, or None."""
    head = html[:65536]
    end = head.lower().find("</head>")
    if end > 0:
        head = head[:end]
    for tag in _ALT_LINK.findall(head):
        attrs = {m.group(1).lower(): (m.group(2) or m.group(3) or m.group(4) or "")
                 for m in _ATTR.finditer(tag)}
        if (attrs.get("type", "").lower() == "text/markdown"
                and "alternate" in attrs.get("rel", "").lower().split() and attrs.get("href")):
            return urllib.parse.urljoin(page_url, attrs["href"])
    return None


def alternate_template(page_url: str, alt_url: str) -> str | None:
    """``alt_url`` with the page's path replaced by ``{path}`` -- the rule
    for every page on the host -- or None when the alternate is not built
    from the path (then it holds for this page only)."""
    page = urllib.parse.urlsplit(page_url)
    path = page.path.rstrip("/")
    if page.query or len(path) < 2:
        return None
    alt = urllib.parse.urlsplit(alt_url)
    if alt.scheme not in ("http", "https"):
        return None
    if alt_url.count(path) != 1:
        return None
    return alt_url.replace("{", "{{").replace("}", "}}").replace(path, "{path}", 1)


def apply_template(template: str, url: str) -> str | None:
    parts = urllib.parse.urlsplit(url)
    path = parts.path.rstrip("/")
    if parts.query or len(path) < 2:
        return None
    return template.format(path=path)


def markdown_build(fallback: str) -> Build:
    def build(body: str):
        text = body.strip()
        if len(text) < 200 or _html_page(text):
            return None
        return _md_title(text, fallback), text

    return build


class AltStore:
    """host -> lean rule (a markdown-alternate template string, or an
    edit-link rule dict), persisted as JSON in the state dir."""

    def __init__(self, path) -> None:
        self.path = path
        try:
            self.map: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.map = {}

    def get(self, host: str):
        return self.map.get(host)

    def put(self, host: str, template) -> None:
        if self.map.get(host) == template:
            return
        self.map[host] = template
        try:
            self.path.write_text(json.dumps(self.map), encoding="utf-8")
        except OSError:
            pass



# ── lean_edit_link: the "Edit this page on GitHub" link as a lean source ────
#
# A docs page that links its own source (``github.com/<o>/<r>/edit/<ref>/<file>``)
# shows how the host maps page paths to files: the file path ends with the
# page path's tail. The mapping is kept per host, like a markdown alternate.

_EDIT_LINK = re.compile(
    r"""href=["'](https://github\.com/([^/"']+)/([^/"']+)/(?:edit|blob)/([^/"']+)/"""
    r"""([^"'#?]+\.(md|mdx|markdown|rst)))["']""")
_INDEX_NAMES = ("index", "readme", "_index")


def _segments(path: str) -> list[str]:
    return [p for p in path.strip("/").split("/") if p]


def edit_rule(page_url: str, html: str) -> dict | None:
    """A host rule learned from the page's edit link, or None. The rule:
    page paths under ``page_prefix`` map to ``raw_prefix/<rest><style><ext>``."""
    page = urllib.parse.urlsplit(page_url)
    psegs = _segments(re.sub(r"\.html?$", "", page.path))
    if page.query or not psegs:
        return None
    for m in _EDIT_LINK.finditer(html[:400000]):
        owner, repo, ref, fpath, ext = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        if " " in fpath or "%" in fpath:
            continue
        fsegs = _segments(fpath)
        index = fsegs[-1].rsplit(".", 1)[0].lower() in _INDEX_NAMES
        stem = fsegs[:-1] if index else fsegs[:-1] + [fsegs[-1].rsplit(".", 1)[0]]
        k = 0
        while k < min(len(stem), len(psegs)) and stem[-1 - k] == psegs[-1 - k]:
            k += 1
        if k == 0:
            continue  # not this page's source (a CONTRIBUTING or CoC link)
        return {"page_prefix": "/".join(psegs[:-k]),
                "raw_prefix": "/".join([f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}"]
                                       + stem[:-k]),
                "ext": "." + ext, "index_first": index,
                "index_name": fsegs[-1].rsplit(".", 1)[0] if index else "index"}
    return None


def apply_edit_rule(rule: dict, url: str) -> list[str]:
    parts = urllib.parse.urlsplit(url)
    psegs = _segments(re.sub(r"\.html?$", "", parts.path))
    pre = _segments(rule["page_prefix"])
    if parts.query or psegs[:len(pre)] != pre or len(psegs) == len(pre):
        return []
    rest = "/".join(psegs[len(pre):])
    flat = f"{rule['raw_prefix']}/{rest}{rule['ext']}"
    nested = f"{rule['raw_prefix']}/{rest}/{rule['index_name']}{rule['ext']}"
    return [nested, flat] if rule["index_first"] else [flat, nested]


# ── youtube_watch_details: the watch page's own player JSON ─────────────────
#
# A youtube.com/watch page extracts to ~240 chars of footer links: the video
# is described only inside the ``ytInitialPlayerResponse`` JSON the same HTML
# carries. Reading it needs no request beyond the page already fetched.

_YT_WATCH = re.compile(r"^https?://(?:www\.|m\.)?youtube\.com/(?:watch\?|shorts/|live/)", re.I)
_YT_PLAYER = re.compile(r"ytInitialPlayerResponse\s*=\s*")


def youtube_watch_details(url: str, html: str) -> tuple[str, str] | None:
    """(title, text) from a watch page's player response, or None."""
    if not _YT_WATCH.match(url):
        return None
    m = _YT_PLAYER.search(html)
    if not m:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(html, m.end())
    except ValueError:
        return None
    v = data.get("videoDetails") if isinstance(data, dict) else None
    if not isinstance(v, dict) or not v.get("title"):
        return None
    micro = (data.get("microformat") or {}).get("playerMicroformatRenderer") or {}
    secs = int(v.get("lengthSeconds") or 0)
    length = f"{secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}" if secs >= 3600 \
        else f"{secs // 60}:{secs % 60:02d}"
    facts = [f"Channel: {v.get('author', '')}",
             f"Length: {length}" if secs else "",
             f"Published: {(micro.get('publishDate') or '')[:10]}" if micro.get("publishDate") else "",
             f"Views: {int(v['viewCount']):,}" if str(v.get("viewCount", "")).isdigit() else "",
             f"Category: {micro['category']}" if micro.get("category") else ""]
    lines = [f"# {v['title']}", " | ".join(f for f in facts if f)]
    desc = (v.get("shortDescription") or "").strip()
    if desc:
        lines.append(desc)
    return v["title"], "\n\n".join(lines)
