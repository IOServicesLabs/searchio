"""The lead agent: plan, dispatch, synthesise.

Orchestrator-worker rather than a peer-to-peer swarm. The workers never see
each other, and every decision about what happens next lives here. That is a
real constraint and it buys two things: workers run in genuine parallel because
there is nothing to coordinate, and the lead's context holds a handful of
reports instead of the entire cross-talk of five agents.

The run has three phases:

1. **Plan** -- decompose the question into independent subtasks. Independence is
   the whole game: two subtasks that need each other's answers cannot run in
   parallel, so a bad decomposition silently serialises the swarm.
2. **Dispatch** -- run workers concurrently under a shared budget. One wave
   only: a follow-up wave that chases the gaps the first wave reported is the
   obvious extension, and deliberately not here yet, because it doubles the
   worst-case cost of a run and wants a real stopping rule before it earns that.
3. **Synthesise** -- write the answer from the reports, citing only what workers
   actually retrieved.

Progress is streamed through an optional callback so a CLI or an SSE endpoint
can show the swarm working, which matters when a run legitimately takes a
minute.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from ..errors import ProviderError
from ..fuse import dedupe_docs, merge_items
from ..models import Doc, Finding, ResearchPlan, ResearchResult, SubTask, WorkerReport
from .budget import Budget
from .worker import Worker

Progress = Callable[[str, dict], Any] | None

PLANNER_SYSTEM = """You decompose a research question into INDEPENDENT subtasks for parallel workers.

The single most important property is independence: each subtask must be
answerable without knowing any other subtask's result. Workers run simultaneously
and cannot communicate, so a subtask that depends on another one's answer will
fail. If a question is inherently sequential, fold the sequence into ONE subtask
rather than splitting it.

Scale the plan to the question:
- A simple factual lookup needs 1 subtask. Do not inflate it.
- A comparison needs one subtask per thing compared, plus at most one for the
  comparison criteria.
- An open-ended question needs 3-5 subtasks covering genuinely different angles,
  not five rephrasings of the same search.

Each subtask gets a concrete objective, 2-4 short keyword search queries (2-6
words each, never sentences), and a clear success criterion. Choose the intent
that routes to the right sources."""

SYNTH_SYSTEM = """You write the final answer from parallel workers' reports.

- Answer the question that was asked, directly, in the first sentence or two.
- Cite with inline markdown links to URLs that appear in the workers' findings.
  Never cite a URL that is not in their reports.
- Where workers disagree or evidence was thin, say so plainly. A hedge that
  reflects real uncertainty is worth more than false confidence.
- Report gaps the workers flagged rather than papering over them.
- Match length to the question: a factual lookup gets a short paragraph, an
  open-ended comparison gets structure. Do not pad."""

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "interpretation": {
            "type": "string",
            "description": "What the user is actually asking, in one sentence.",
        },
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "objective": {"type": "string"},
                    "rationale": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "enum": ["web", "news", "video", "shopping", "academic", "code",
                                 "local", "reference", "forum"],
                    },
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "success_criteria": {"type": "string"},
                },
                "required": ["id", "objective", "intent", "queries", "success_criteria"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["interpretation", "subtasks"],
    "additionalProperties": False,
}


log = logging.getLogger(__name__)

#: A plan longer than this is not a plan; the planner prompt asks for a
#: handful and the budget dispatches max_workers of them.
MAX_PLAN_SUBTASKS = 50

_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"(?<![(\w])https?://[^\s)\]>\"']+")


def _url_key(url: str) -> str:
    u = url.strip().rstrip(".,;:!?")
    u = u.split("#", 1)[0]
    if "://" in u:
        scheme, rest = u.split("://", 1)
        host, _, path = rest.partition("/")
        u = f"{scheme.lower()}://{host.lower()}/{path}"
    return u.rstrip("/")


def neutralize_unretrieved_links(answer: str, retrieved: set[str]) -> tuple[str, int]:
    """Strip every link in ``answer`` whose URL no worker retrieved (bug 79).

    Bug 62 made findings' citations provable; the free-text answer was the
    remaining door for an invented URL. A markdown link keeps its text, a
    bare URL becomes a marker; the count goes into the result's warnings.
    """
    keys = {_url_key(u) for u in retrieved}
    dropped = 0

    def md(m: re.Match) -> str:
        nonlocal dropped
        if _url_key(m.group(2)) in keys:
            return m.group(0)
        dropped += 1
        return m.group(1)

    out = _MD_LINK.sub(md, answer)

    def bare(m: re.Match) -> str:
        nonlocal dropped
        if _url_key(m.group(0)) in keys:
            return m.group(0)
        dropped += 1
        return "[unverified link removed]"

    out = _BARE_URL.sub(bare, out)
    return out, dropped


class Orchestrator:
    """Runs a full research swarm for one question."""

    def __init__(self, engine, client, *, model: str = "", progress: Progress = None) -> None:
        self.engine = engine
        self.client = client
        self.s = engine.s
        # The backend factory resolves which model id is valid for this
        # client; falling back to settings.model keeps direct construction
        # (and the tests) working unchanged.
        self.model = model or engine.s.model
        self._progress = progress
        self._progress_failures = 0
        self._plan_warnings: list[str] = []

    async def _emit(self, event: str, data: dict) -> None:
        if self._progress is None:
            return
        try:
            out = self._progress(event, data)
            if asyncio.iscoroutine(out):
                await out
        except Exception as exc:  # noqa: BLE001
            # A listener is an observer: its failure must not end the run
            # (bug 77's second door). Say so once.
            self._progress_failures += 1
            if self._progress_failures == 1:
                log.warning("progress listener failed on %s: %s", event, exc)

    # ── phases ───────────────────────────────────────────────────────────────

    async def plan(self, question: str, budget: Budget) -> ResearchPlan:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": question}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.s.lead_effort, "format": {
                "type": "json_schema", "schema": PLAN_SCHEMA
            }},
        )
        budget.record(resp.usage.input_tokens, resp.usage.output_tokens)
        if getattr(resp, "stop_reason", "") == "refusal":
            raise ProviderError("planner", "model declined to plan this question")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        # The planner's text is model output, not a trusted document (bug
        # 78): garbage, a missing key or a malformed subtask used to escape
        # as JSONDecodeError / KeyError / ValidationError -- a bare 500 on
        # /research, an opaque error event on the stream.
        try:
            data = json.loads(text)
            raw = data["subtasks"]
            interpretation = str(data.get("interpretation") or "")
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ProviderError(
                "planner", f"unusable plan ({type(exc).__name__}: {exc}); head={text[:120]!r}"
            ) from exc
        if not isinstance(raw, list):
            raise ProviderError("planner", f"unusable plan: subtasks is {type(raw).__name__}")
        subtasks: list[SubTask] = []
        seen: set[str] = set()
        ids: set[str] = set()
        for st in raw[:MAX_PLAN_SUBTASKS]:
            try:
                sub = SubTask(**st) if isinstance(st, dict) else None
            except (ValidationError, TypeError):
                sub = None
            if sub is None:
                continue
            key = " ".join(sub.objective.split()).lower()
            if not key or key in seen:
                continue  # empty or duplicate objective: one worker is enough
            seen.add(key)
            if not sub.id or sub.id in ids:
                # Skip ids already taken (bug 185): f"s{len(subtasks)+1}" could
                # reproduce a model-given id -- if subtask 1 kept "s2" and
                # subtask 2 needed a new one, s{1+1} was "s2" again, so two
                # subtasks shared an id and their reports collided.
                n = len(subtasks) + 1
                while f"s{n}" in ids:
                    n += 1
                sub = sub.model_copy(update={"id": f"s{n}"})
            ids.add(sub.id)
            subtasks.append(sub)
        if not subtasks:
            raise ProviderError("planner", "plan has no usable subtasks")
        self._plan_warnings = []
        if len(subtasks) > budget.max_workers:
            self._plan_warnings.append(
                f"plan truncated: {len(subtasks)} subtasks planned, "
                f"{budget.max_workers} dispatched (max_workers)")
            subtasks = subtasks[: budget.max_workers]
        return ResearchPlan(interpretation=interpretation, subtasks=subtasks)

    async def dispatch(self, plan: ResearchPlan, budget: Budget) -> list[WorkerReport]:
        """Run every subtask concurrently, bounded by the configured limit."""
        sem = asyncio.Semaphore(self.s.worker_concurrency)

        async def one(st: SubTask) -> WorkerReport:
            async with sem:
                await self._emit("worker_start", {"id": st.id, "objective": st.objective})
                try:
                    report = await Worker(
                        self.engine, self.client, st, budget,
                        effort=self.s.worker_effort, model=self.model,
                    ).run()
                except Exception as exc:  # noqa: BLE001
                    # One worker crashing must not lose the wave (bug 77):
                    # gather() without return_exceptions aborted /research
                    # for everyone and left the siblings running as orphans.
                    report = WorkerReport(
                        subtask_id=st.id, summary="",
                        error=f"worker crashed: {type(exc).__name__}: {exc}"[:300])
                await self._emit(
                    "worker_done",
                    {
                        "id": st.id,
                        "findings": len(report.findings),
                        "sources": len(report.docs),
                        "tool_calls": report.tool_calls,
                        "error": report.error,
                    },
                )
                return report

        return list(await asyncio.gather(*(one(st) for st in plan.subtasks)))

    async def synthesise(
        self, question: str, plan: ResearchPlan, reports: list[WorkerReport], budget: Budget
    ) -> str:
        brief = [f"# Question\n{question}\n", f"# Interpretation\n{plan.interpretation}\n"]
        for r in reports:
            st = next((s for s in plan.subtasks if s.id == r.subtask_id), None)
            brief.append(f"\n## Worker {r.subtask_id}: {st.objective if st else ''}")
            if r.error:
                brief.append(f"FAILED: {r.error}")
                continue
            brief.append(r.summary)
            for f in r.findings:
                cites = " ".join(f"[{c.title or c.url}]({c.url})" for c in f.citations)
                brief.append(f"- ({f.confidence}) {f.claim}"
                             + (f" — {f.detail}" if f.detail else "")
                             + (f"\n  sources: {cites}" if cites else ""))
            if r.items:
                brief.append("Listings found:")
                for it in r.items[:12]:
                    brief.append(f"- {it.title[:100]} — {it.price} — {it.url}")
            if r.gaps:
                brief.append("Gaps: " + "; ".join(r.gaps))

        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=SYNTH_SYSTEM,
            messages=[{"role": "user", "content": "\n".join(brief)}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.s.lead_effort},
        )
        budget.record(resp.usage.input_tokens, resp.usage.output_tokens)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if getattr(resp, "stop_reason", "") == "refusal" or not text:
            # A refusal or an empty reply at the LAST step used to ship as
            # answer="" with no warning -- a successful-looking nothing
            # (bug 147). The findings are still the findings: render them
            # deterministically and say why.
            why = "declined" if getattr(resp, "stop_reason", "") == "refusal" else "returned nothing"
            self._plan_warnings.append(f"synthesis {why}; answer assembled from the findings")
            return self._fallback_answer(question, plan, reports)
        return text

    @staticmethod
    def _fallback_answer(question: str, plan: ResearchPlan, reports: list[WorkerReport]) -> str:
        out = [f"No synthesised answer was produced for: {question}", ""]
        for r in reports:
            if r.error:
                out.append(f"- worker {r.subtask_id} failed: {r.error}")
                continue
            if r.summary:
                out.append(f"- {r.summary}")
            for f in r.findings:
                cites = " ".join(f"[{c.title or c.url}]({c.url})" for c in f.citations if c.url)
                out.append(f"  - ({f.confidence}) {f.claim}" + (f" {cites}" if cites else ""))
            for it in r.items[:12]:
                out.append(f"  - {it.title[:100]} — {it.price} — {it.url}")
            if r.gaps:
                out.append("  - gaps: " + "; ".join(r.gaps))
        return "\n".join(out).strip()

    # ── entry point ──────────────────────────────────────────────────────────

    async def run(self, question: str, *, budget: Budget | None = None) -> ResearchResult:
        price_in, price_out = self.s.prices()
        b = budget or Budget(
            max_tokens=self.s.swarm_token_budget,
            max_wall_clock_s=self.s.swarm_wall_clock_s,
            max_workers=self.s.max_workers,
            max_tool_calls_per_worker=self.s.max_tool_calls_per_worker,
            price_in=price_in,
            price_out=price_out,
        )

        await self._emit("planning", {"question": question})
        plan = await self.plan(question, b)
        await self._emit(
            "planned",
            {
                "interpretation": plan.interpretation,
                "subtasks": [{"id": s.id, "objective": s.objective} for s in plan.subtasks],
            },
        )

        reports = await self.dispatch(plan, b)

        findings: list[Finding] = []
        docs: list[Doc] = []
        items = []
        for r in reports:
            findings.extend(r.findings)
            docs.extend(r.docs)
            items.extend(r.items)

        await self._emit("synthesising", {"findings": len(findings), "sources": len(docs)})
        answer = await self.synthesise(question, plan, reports, b)

        warnings = list(self._plan_warnings)
        for r in reports:
            if r.error:
                warnings.append(f"worker {r.subtask_id} failed: {r.error}")
        # Listing URLs are retrieved too (bug 146): the brief hands the model
        # every listing's url, and the guard then stripped those same links.
        retrieved = {d.url for d in docs} | {
            c.url for f in findings for c in f.citations if c.url} | {
            it.url for it in items if it.url}
        answer, dropped = neutralize_unretrieved_links(answer, retrieved)
        if dropped:
            warnings.append(
                f"{dropped} link(s) in the answer pointed at URLs not retrieved by "
                f"any worker and were removed")

        return ResearchResult(
            query=question,
            answer=answer,
            plan=plan,
            findings=findings,
            items=merge_items(items),
            sources=dedupe_docs(docs),
            reports=reports,
            input_tokens=b.input_tokens,
            output_tokens=b.output_tokens,
            estimated_cost_usd=round(b.estimated_cost, 4) if b.priced else None,
            model=self.model,
            elapsed_s=b.elapsed_s,
            stopped_early=b.stopped_reason,
            warnings=warnings,
        )
