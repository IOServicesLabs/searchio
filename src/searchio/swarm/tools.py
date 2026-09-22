"""The tools a research worker can call, and the bridge that executes them.

Tool design here is mostly about *what not to return*. A worker's context is
the scarcest resource in the system: hand it a 3 MB Amazon page and the page is
the run. So every tool returns the smallest thing that still supports a
decision -- search returns titles and snippets, never bodies; ``read_page``
returns extracted prose with a hard character cap and says so when it truncates.

``submit_report`` is how a worker finishes. Making the terminal step a tool call
with a strict schema means the structured result comes out of the same loop
that did the research, instead of a second summarisation call that can quietly
disagree with it.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import TransientError
from ..extract import to_markdown
from ..models import Citation, Doc, Finding, Item, WorkerReport

#: Hard cap on page text handed back to a model, in characters.
PAGE_CHAR_LIMIT = 12_000
SNIPPET_LIMIT = 240
TITLE_LIMIT = 200


def tool_defs() -> list[dict]:
    """Anthropic tool definitions for a research worker.

    ``strict`` is set so tool inputs validate exactly against the schema; the
    loop below parses them with ``json.loads`` regardless, because escaping in
    tool arguments is not something to pattern-match on.
    """
    return [
        {
            "name": "web_search",
            "description": (
                "Search the web and get ranked results with titles and snippets. "
                "Results are fused across several independent providers, so a result "
                "found by more than one is more likely to be right. Use short keyword "
                "queries (2-6 words), not sentences. Run several narrow searches "
                "rather than one broad one."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Short keyword query."},
                    "intent": {
                        "type": "string",
                        "enum": ["web", "news", "video", "shopping", "academic", "code",
                                 "local", "reference", "forum"],
                        "description": (
                            "Routes to the right sources: 'academic' uses OpenAlex/arXiv/"
                            "Crossref, 'code' uses GitHub/StackOverflow, 'reference' uses "
                            "Wikipedia, 'shopping' uses marketplaces, 'video' uses YouTube."
                        ),
                    },
                    "k": {"type": "integer", "description": "Results to return, 1-15."},
                    "freshness": {
                        "type": "string",
                        "enum": ["all", "day", "week", "month", "year"],
                        "description": (
                            "Recency bound. 'day'/'week' for news and prices, "
                            "'all' (default) for evergreen topics."
                        ),
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Restrict to these domains (like site:). Off-domain "
                            "results are dropped after the fan-out, so the "
                            "restriction is a guarantee, not a hint."
                        ),
                    },
                    "locale": {
                        "type": "string",
                        "description": (
                            "Language/market of the question as a BCP-47 tag "
                            "(\"de-DE\", \"ja-JP\", \"pt-BR\"). Use it whenever the "
                            "question is not in English: providers then search that "
                            "language's web. Default en-US."
                        ),
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Never return results from these domains.",
                    },
                },
                "required": ["query", "intent"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "read_page",
            "description": (
                "Fetch one URL and return its readable text as markdown. Use this only "
                "after a search result looks genuinely worth reading -- it costs a "
                "network round trip and a large slice of your context. Handles "
                "bot-protected pages automatically by escalating transports. If the "
                "result says the page is an empty JavaScript shell, use "
                "read_page_rendered on the same URL instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "read_page_rendered",
            "description": (
                "Fetch one URL through the full-browser fidelity tier (real "
                "Chromium: executes every kind of JavaScript including module "
                "scripts and WASM, carries real session state, solves managed "
                "challenges). This is 10-100x slower and heavier than read_page, "
                "so use it ONLY when read_page came back as an empty/JS-shell "
                "page or you already know the content only exists after scripts "
                "run. For ordinary pages read_page is always the right call."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL."},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "find_items",
            "description": (
                "Find purchasable listings for a product across marketplaces and return "
                "normalized records with price, seller, availability and rating. Offers "
                "for the same product are merged, so each row is one product with its "
                "alternative offers attached. Use for any 'what does X cost' or "
                "'where can I buy X' question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product name and key attributes."},
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional marketplace domains to restrict to.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "submit_report",
            "description": (
                "Finish this subtask and return your findings. Call this exactly once, "
                "when you have enough evidence or have concluded the evidence is not "
                "available. Every claim must carry at least one citation URL you "
                "actually retrieved. If you could not answer, say so in gaps rather "
                "than guessing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "2-4 sentences answering the subtask objective.",
                    },
                    "findings": {
                        "type": "array",
                        "description": "Discrete claims, each independently supported.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "detail": {"type": "string"},
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "citations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "url": {"type": "string"},
                                            "title": {"type": "string"},
                                            "quote": {"type": "string"},
                                        },
                                        "required": ["url"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["claim", "confidence", "citations"],
                            "additionalProperties": False,
                        },
                    },
                    "gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What you could not establish, and why.",
                    },
                },
                "required": ["summary", "findings"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


class ToolBridge:
    """Executes worker tool calls against the searchio engine.

    Holds the documents and items a worker touched so the orchestrator can
    assemble sources without re-fetching anything.
    """

    def __init__(self, engine, *, subtask_id: str = "") -> None:
        self.engine = engine
        self.subtask_id = subtask_id
        self.seen_docs: dict[str, Doc] = {}
        self.items: list[Item] = []
        self.calls = 0
        self.report: WorkerReport | None = None

    async def dispatch(self, name: str, raw_input: Any) -> tuple[str, bool]:
        """Run one tool. Returns (result_text, is_error)."""
        self.calls += 1
        try:
            # Inside the try (rider, iteration 42): a malformed non-dict input
            # used to escape as JSONDecodeError and take the worker loop down.
            args = raw_input if isinstance(raw_input, dict) else json.loads(raw_input or "{}")
            if isinstance(args, dict) and "_error" in args:
                # The backend could not parse the model's arguments (bug 82):
                # say so, instead of running the tool with none of them.
                return f"Tool call rejected: {args['_error']}", True
            if name == "web_search":
                return await self._web_search(args), False
            if name == "read_page":
                return await self._read_page(args)
            if name == "read_page_rendered":
                return await self._read_page(args, rendered=True)
            if name == "find_items":
                return await self._find_items(args), False
            if name == "submit_report":
                return self._submit(args)
        except Exception as exc:
            # Tool failures are information, not crashes: a worker that learns
            # a page is blocked should try another source, so the error goes
            # back into the loop rather than ending the run.
            return f"{type(exc).__name__}: {exc}"[:500], True
        return f"unknown tool {name}", True

    # ── implementations ──────────────────────────────────────────────────────

    def _circuit_open_note(self, intent: str) -> str:
        """Name the providers that would serve ``intent`` but sit circuit-open.

        An empty answer with no recorded failure usually means attrition: the
        engine providers tripped their breakers earlier this session and only
        clean-empty APIs answered. The stock advice ("try different keywords")
        then sends an agent into a rephrase loop that burns its whole turn
        budget (bench/span.py llm.price_check, three consecutive runs), so the
        honest version prescribes a different remedy.
        """
        registry = getattr(self.engine.router, "registry", None)
        if registry is None:
            return ""
        open_prov = [
            p.name
            for p in registry.all()
            if p.configured(self.engine.s) and p.supports(intent) and p.health.open
        ]
        if not open_prov:
            return ""
        return (
            f" Providers for intent {intent!r} are circuit-open from earlier "
            f"failures this session: {', '.join(open_prov)}. Rephrasing the "
            f"query will not bring them back -- use sources you already "
            f"retrieved, a different tool, or submit a partial report."
        )

    async def _web_search(self, args: dict) -> str:
        text = str(args.get("query", ""))[:300]
        intent = args.get("intent", "web")
        # Engine.search, never router.search directly: the engine is where
        # inline operators (site:x / -site:x inside the query text) get
        # extracted into the structured fields the router post-filter
        # enforces. This bridge used to bypass it, so tool callers -- the
        # agents who write operators inline most -- silently lost them.
        res = await self.engine.search(
            text,
            intent=intent,
            k=max(1, min(int(args.get("k") or 8), 15)),
            domains=args.get("domains") or None,
            exclude_domains=args.get("exclude_domains") or None,
            freshness=args.get("freshness") or "all",
            locale=str(args.get("locale") or "")[:16],
        )
        if not res.docs:
            if res.failed:
                return (f"No results for {text!r} (providers failed: "
                        f"{res.failed}). Try different keywords or another intent.")
            note = self._circuit_open_note(intent)
            if note:
                return f"No results for {text!r}.{note}"
            filtered = dict(getattr(res, "filtered", None) or {})
            filtered_stale = dict(getattr(res, "filtered_stale", None) or {})
            if filtered or filtered_stale:
                # Everything answered was dropped by the query's own
                # restrictions (off-domain, or provably stale for the
                # freshness bound): the guarantees held, the engines just
                # would not honor the hints this run. Say how to widen.
                bits = []
                if filtered:
                    bits.append(f"{sum(filtered.values())} off-domain "
                                f"{dict(filtered)}")
                if filtered_stale:
                    bits.append(f"{sum(filtered_stale.values())} stale-dated "
                                f"{dict(filtered_stale)}")
                return (f"No results for {text!r}: restrictions dropped "
                        + "; ".join(bits)
                        + ". The filters held; broaden domains/freshness or "
                          "remove operators if the restriction was not intended.")
            return (f"No results for {text!r}. Try different keywords or "
                    f"another intent.")
        lines = [f"{len(res.docs)} results (sources: {', '.join(res.used)})"]
        for i, d in enumerate(res.docs, 1):
            # Never replace a doc read_page already filled with text by the
            # bare search hit (bug 106): the report's sources lost the page.
            prev = self.seen_docs.get(d.url)
            if prev is None or not prev.text:
                self.seen_docs[d.url] = d
            agree = d.meta.get("agreement", 1)
            mark = f" [confirmed by {agree} providers]" if agree > 1 else ""
            # A hostile <title> times k=15 must not flood the context (rider).
            lines.append(f"{i}. {d.title[:TITLE_LIMIT]}{mark}\n   {d.url}\n   {d.snippet[:SNIPPET_LIMIT]}")
        return "\n".join(lines)

    async def _read_page(self, args: dict, *, rendered: bool = False) -> tuple[str, bool]:
        url = str(args.get("url", ""))
        if not url.startswith("http"):
            return "Invalid URL: must be absolute http(s).", True
        try:
            # rendered=True forces tier 2: the tool's contract IS a browser
            # pass. Letting a cheap tier answer re-serves the exact shell the
            # model just escalated away from (bench/span.py llm.price_check:
            # the same Amazon shell three times, rendered budget burned).
            res = await self.engine.ladder.fetch(
                url, rendered=rendered, force_tier=2 if rendered else None
            )
        except TransientError as exc:
            # The hard-failure version of the shell lesson: every transport
            # came back empty, which is what a JS shell does to parse-only
            # backends. Point at the fix so the model can escalate instead
            # of giving up or blindly retrying the same call.
            if not rendered and "no usable content" in str(exc):
                return (
                    f"{type(exc).__name__}: {exc}\n\n[note: every transport came "
                    "back empty -- this page is likely a JavaScript shell. If you "
                    "need its content, call read_page_rendered on the same URL.]"
                ), True
            raise
        text = to_markdown(res.body, url)
        doc = self.seen_docs.get(url) or Doc(url=url, source="read_page")
        doc.text = text
        self.seen_docs[url] = doc
        via = f"{res.via} (tier {res.tier})"
        if res.rendered:
            via += ", full-browser render"
        header = f"[{url}] fetched via {via}"
        if len(text) > PAGE_CHAR_LIMIT:
            out = (
                f"{header}, truncated to {PAGE_CHAR_LIMIT} of {len(text)} chars\n\n"
                + text[:PAGE_CHAR_LIMIT]
                + "\n\n[...truncated...]"
            )
        else:
            out = f"{header}\n\n{text}"
        # The escalation lesson, inline: a near-empty result from the cheap
        # path is almost always a JS shell, and the fix is one tool call away.
        # Only hint when we did NOT already pay for a render -- a rendered
        # page with no text has nothing to escalate to.
        # A cached shell is still a shell (iteration 56 rider): the reader
        # cannot know an earlier task already saw the hint.
        if not rendered and len(text.strip()) < 300:
            out += (
                "\n\n[note: this page has almost no visible text -- it is "
                "likely a JavaScript shell. If you need its content, call "
                "read_page_rendered on the same URL.]"
            )
        return out, False

    async def _find_items(self, args: dict) -> str:
        from ..errors import ProviderError

        try:
            items = await self.engine.find_items(
                str(args.get("query", ""))[:300], domains=args.get("domains") or None
            )
        except ProviderError as exc:
            # The provider refused (bug 120): say so, with the same
            # circuit-open guidance an empty answer carries.
            return f"Listings lookup failed: {str(exc)[:300]}.{self._circuit_open_note('shopping')}"
        # Keep the report's item list free of the same listing arriving from
        # two calls (rider); merge_items already collapsed within one call.
        known = {it.url for it in self.items}
        self.items.extend(it for it in items if it.url not in known)
        # A listing the model was shown IS retrieved (bug 105): citing it used
        # to be dropped by submit_report as "never retrieved in this task".
        for it in items:
            if it.url and it.url not in self.seen_docs:
                self.seen_docs[it.url] = Doc(url=it.url, title=it.title, snippet=str(it.price),
                                             source=it.source or "find_items")
        if not items:
            note = self._circuit_open_note("shopping")
            if note:
                return f"No listings found.{note}"
            return "No listings found. Try a broader product name."
        # Say what is SHOWN: the header used to claim N listings and render
        # 15, so the model believed it had seen all N (rider).
        shown = items[:15]
        lines = [f"showing {len(shown)} of {len(items)} listings:"
                 if len(items) > len(shown) else f"{len(items)} listings:"]
        for i, it in enumerate(shown, 1):
            bits = [f"{i}. {it.title[:110]}", f"   {it.price} — {it.seller or it.domain}"]
            if it.verified:
                bits.append("   ✓ verified (detail page confirmed)")
            if it.rating is not None:
                bits.append(f"   rating {it.rating} ({it.reviews or 0} reviews)")
            if it.availability:
                bits.append(f"   {it.availability}")
            if it.meta.get("offer_count"):
                bits.append(f"   {it.meta['offer_count']} offers across sites")
            bits.append(f"   {it.url}")
            lines.append("\n".join(bits))
        return "\n".join(lines)

    def _submit(self, args: dict) -> tuple[str, bool]:
        from ..fuse import canonical_url

        # A blank submission is not a report (bug 160, caught live): the
        # schema marks summary and findings required, but OpenAI-compatible
        # backends do not enforce strict schemas, and a worker that had read
        # five pages called submit_report({}) -- the bridge said "Report
        # recorded." and the subtask closed as an unforced success with no
        # summary, no findings and no citations. Refuse it as a tool error
        # so the model gets the turn back; a forced-turn blank then lands
        # in the fallback report's error (bug 128's path).
        summary = str(args.get("summary") or "").strip()
        raw_findings = args.get("findings") or []
        if not isinstance(raw_findings, list) or any(not isinstance(f, dict) for f in raw_findings):
            return ("Rejected: findings must be a list of objects, each with a claim, "
                    "a confidence and its citations. Nothing was recorded."), True
        raw_gaps = args.get("gaps") or []
        if isinstance(raw_gaps, str):
            raw_gaps = [raw_gaps]
        has_claim = any(str(f.get("claim") or "").strip() for f in raw_findings)
        has_gap = any(str(g or "").strip() for g in raw_gaps)
        if not (summary or has_claim or has_gap):
            return ("Rejected: an empty report. submit_report needs a summary "
                    "(2-4 sentences answering the objective) and findings whose "
                    "citations are URLs you retrieved in this task, or gaps saying "
                    "what you could not establish. Nothing was recorded; call it "
                    "again with content."), True

        # The contract: "a citation URL you actually retrieved". The bridge
        # used to accept any truthy url, so a URL the model never searched or
        # read -- fabricated provenance, the exact shape bench/span.py's LLM
        # leg kept catching (llm.price_check cited a target.com/s?searchTerm=
        # page it never fetched) -- walked into the WorkerReport (bug 62).
        # Match on the canonical URL so tracking params / www. / scheme do
        # not reject the page the model really read; record every drop as a
        # gap so the report stays honest about its own evidence.
        seen = {canonical_url(u) for u in self.seen_docs}
        dropped: list[str] = []
        findings = []
        for f in raw_findings:
            cites = []
            cited_keys: set[str] = set()
            for c in f.get("citations") or []:
                url = str(c.get("url") or "")
                if not url:
                    continue
                key = canonical_url(url)
                if key not in seen:
                    dropped.append(url)
                    continue
                if key in cited_keys:
                    # One page is one source (iteration 56 rider): the same
                    # URL three times over was three "sources" to the reader.
                    continue
                cited_keys.add(key)
                cites.append(Citation(url=url, title=str(c.get("title", "")),
                                      quote=str(c.get("quote", ""))[:400]))
            findings.append(
                Finding(
                    claim=str(f.get("claim", "")),
                    detail=str(f.get("detail", "")),
                    confidence=f.get("confidence", "medium"),
                    subtask_id=self.subtask_id,
                    citations=cites,
                )
            )
        gaps = [str(g) for g in raw_gaps]  # a plain string is one gap, not its characters (rider)
        note = ""
        if dropped:
            shown = ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else "")
            note = (f" {len(dropped)} citation(s) dropped: URL never retrieved in "
                    f"this task ({shown}).")
            gaps.append(f"{len(dropped)} citation(s) dropped as never retrieved: {shown}")
        self.report = WorkerReport(
            subtask_id=self.subtask_id,
            summary=str(args.get("summary", "")),
            findings=findings,
            gaps=gaps,
            items=list(self.items),
            docs=list(self.seen_docs.values()),
            tool_calls=self.calls,
        )
        return "Report recorded." + note, False
