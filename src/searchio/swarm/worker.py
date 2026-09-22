"""One research worker: a bounded tool loop with its own context.

A worker knows nothing about the other workers. It gets a self-contained
objective, a fresh context window, and a tool budget, and it returns a report.
That isolation is what lets the swarm actually run in parallel -- workers that
can talk to each other have to be scheduled around each other, and their
combined chatter lands in the lead agent's context, which is the resource the
whole design exists to protect.

The loop is written out rather than delegated to the SDK's tool runner because
of what happens at the edges: a worker that exhausts its tool budget without
calling ``submit_report`` still has to produce a usable report, and a worker
that hits the run's token ceiling mid-loop has to stop cleanly and hand back
partial findings. Those are the cases that decide whether a swarm degrades
gracefully or returns nothing.
"""

from __future__ import annotations


from ..models import SubTask, WorkerReport
from .budget import Budget
from .tools import ToolBridge, tool_defs

WORKER_SYSTEM = """You are a research worker in a parallel swarm. You have ONE subtask.

How to work:
- Start with 2-3 short keyword searches from different angles. Search queries are
  keywords (2-6 words), never sentences.
- Pick the `intent` that matches what you need: `academic` for papers, `code` for
  libraries and errors, `reference` for definitions, `shopping` for prices,
  `video` for YouTube, `forum` for candid real-world experience.
- Reach for the search knobs the question implies: `freshness` = "day" or "week"
  for anything recent (news, prices, "this week"); `intent` = "forum" for opinions;
  put `site:example.com` in the query (or use `domains`) when the question names a
  site; set `locale` (e.g. "de-DE") and search in the question's own language when
  the question is not in English, and prefer sources in that language.
- Only call read_page when a result genuinely looks decisive. Reading is expensive
  and most questions are answered by comparing several snippets.
- Prefer sources that more than one provider surfaced, and prefer primary sources
  (documentation, the paper, the seller's own page) over commentary about them.
- When two sources disagree, say so in your findings instead of silently picking one.

Rules:
- Every claim needs a citation URL you actually retrieved in this session. Never
  cite a URL you did not open or see in search results.
- Do not speculate. If the evidence is not there, put it in `gaps`.
- You cannot ask anyone questions. Work with what you can retrieve.
- Call submit_report exactly once, when done. That ends your task."""


def worker_system(max_calls: int) -> str:
    """WORKER_SYSTEM plus the concrete budget and the stopping rule.

    An agent that cannot see its budget cannot spend it deliberately.
    bench/span.py run 080002: three of four DeepSeek questions rephrase-looped
    to the turn cap under provider attrition -- the tools were honestly saying
    "circuit-open, rephrasing will not help" and the prompt never said when to
    stop, so the model kept rephrasing. The force-submit net catches the work;
    this line is what makes the net rare.
    """
    return WORKER_SYSTEM + (
        f"\n\nBudget: at most {max_calls} tool calls, and the counter is "
        f"running now. If a search comes back empty with a circuit-open "
        f"note, that source is DOWN, not your keywords wrong: at most one "
        f"rephrase, then work from what you already retrieved, try another "
        f"tool, or submit_report with the gap named."
    )


class Worker:
    """Runs one subtask to completion."""

    def __init__(self, engine, client, subtask: SubTask, budget: Budget, *,
                 effort: str = "low", model: str = ""):
        self.engine = engine
        self.client = client
        self.subtask = subtask
        self.budget = budget
        self.effort = effort
        self.model = model or engine.s.model
        self.system = worker_system(budget.max_tool_calls_per_worker)
        #: The forced submit turn's last failure (exception / refusal), for
        #: the fallback report's error field (bug 64).
        self._force_failure = ""

    async def run(self) -> WorkerReport:
        bridge = ToolBridge(self.engine, subtask_id=self.subtask.id)
        messages: list[dict] = [{"role": "user", "content": self._prompt()}]
        max_calls = self.budget.max_tool_calls_per_worker
        budget_stopped = False
        prose_turns = 0

        while bridge.report is None and bridge.calls < max_calls:
            if not self.budget.soft_check():
                budget_stopped = True
                break
            try:
                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    system=self.system,
                    messages=messages,
                    tools=tool_defs(),
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                )
            except Exception as exc:
                return WorkerReport(
                    subtask_id=self.subtask.id,
                    summary="",
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    tool_calls=bridge.calls,
                )

            self.budget.record(resp.usage.input_tokens, resp.usage.output_tokens)

            if resp.stop_reason == "refusal":
                return WorkerReport(
                    subtask_id=self.subtask.id,
                    summary="",
                    error="model declined this subtask",
                    tool_calls=bridge.calls,
                )

            # Thinking blocks must be echoed back unchanged to continue the turn.
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                # The model answered in prose instead of finishing properly.
                # Nudge, but not forever (bug 127): a model that keeps
                # answering in prose never moves bridge.calls, so the loop's
                # only bound was the token budget -- dozens of 8k-token turns
                # for one worker. Two prose turns, then the forced submit.
                prose_turns += 1
                if prose_turns >= 2:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "Call submit_report now with what you have established.",
                    }
                )
                continue

            # All tool results for one assistant turn go back in ONE user
            # message; splitting them teaches the model to stop calling tools
            # in parallel.
            results = []
            for tu in tool_uses:
                # Every tool_use id needs a tool_result (the API rejects the
                # next turn otherwise), but nothing EXECUTES past the cap or
                # after a report exists (bug 63): the cap was checked only at
                # the top of the turn, so a parallel batch ran to cap+N past
                # the budget the system prompt promises, and tools after
                # submit_report in the same turn ran after the report was
                # built. submit_report itself is always allowed -- it is the
                # terminal step the force-submit would spend an extra call on.
                if bridge.report is not None:
                    text, is_error = ("Report already submitted; this call was "
                                      "not executed."), True
                elif bridge.calls >= max_calls and tu.name != "submit_report":
                    text, is_error = (f"Tool budget of {max_calls} calls is "
                                      "exhausted; this call was not executed. "
                                      "Call submit_report with what you have."), True
                else:
                    text, is_error = await bridge.dispatch(tu.name, tu.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})

        if bridge.report is not None:
            return bridge.report

        # Out of tool calls with no report. This is the common failure, not a
        # rare one: a worker will happily spend its whole budget searching, and
        # the terminal step competes for the same budget as the research. So
        # spend one more call that can ONLY submit -- the worker has the
        # evidence in context either way, and the difference between a report
        # and no report is the difference between findings and a shrug.
        self._force_failure = ""
        # A budget stop means STOP (bug 126) -- but the loop also exits on the
        # tool-call cap or the prose cap, which leave budget_stopped False
        # while the token/wall budget may already be spent (bug 184, bug 126's
        # sibling): the forced turn then ran and overspent. Re-check the budget
        # so the one harvesting call happens only when there is room for it;
        # soft_check is a pure predicate, safe to call again.
        if not budget_stopped and self.budget.soft_check():
            report = await self._force_submit(messages, bridge)
            if report is not None:
                return report

        # Even that failed. Everything retrieved is still worth keeping.
        return WorkerReport(
            subtask_id=self.subtask.id,
            summary=(
                f"Did not complete within {bridge.calls} tool calls. "
                f"Retrieved {len(bridge.seen_docs)} sources without concluding."
            ),
            docs=list(bridge.seen_docs.values()),
            items=list(bridge.items),
            gaps=[f"Subtask incomplete: {self.subtask.objective}"],
            tool_calls=bridge.calls,
            # A forced turn that died (API exception) or was declined is a
            # failure the orchestrator must see -- it used to read as a clean
            # incomplete with error="" (bug 64). An honest "no submit came
            # back" stays a plain incomplete: the summary and gaps say so.
            error=self._force_failure
            or ("" if bridge.calls >= max_calls else self.budget.stopped_reason),
        )

    async def _force_submit(self, messages: list[dict], bridge: ToolBridge):
        """One last turn whose only available tool is ``submit_report``.

        Offering a single tool is the reliable way to get the terminal step:
        forced ``tool_choice`` is not supported on every model or backend, so it
        is attempted and then retried without, and with only one tool on the
        table a model that calls anything at all calls the right thing.
        """
        submit_only = [t for t in tool_defs() if t["name"] == "submit_report"]
        convo = messages + [
            {
                "role": "user",
                "content": (
                    "You have used your tool budget. Stop researching and call "
                    "submit_report now, using only what you already retrieved in "
                    "this session. Cite the URLs you actually saw. Put anything "
                    "you could not establish in gaps."
                ),
            }
        ]
        for tool_choice in ({"type": "tool", "name": "submit_report"}, None):
            kwargs = dict(
                model=self.model,
                max_tokens=8192,
                system=self.system,
                messages=convo,
                tools=submit_only,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
            )
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            try:
                resp = await self.client.messages.create(**kwargs)
            except Exception as exc:
                self._force_failure = f"force-submit failed: {type(exc).__name__}: {exc}"[:300]
                continue
            self.budget.record(resp.usage.input_tokens, resp.usage.output_tokens)
            if getattr(resp, "stop_reason", "") == "refusal":
                self._force_failure = "model declined the forced submit_report turn"
                continue
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use" and b.name == "submit_report":
                    text, is_error = await bridge.dispatch(b.name, b.input)
                    if bridge.report is not None:
                        bridge.report.tool_calls = bridge.calls
                        return bridge.report
                    if is_error:
                        # The forced submit came back and was REJECTED
                        # (malformed arguments, a validation failure): that
                        # is the failure the fallback report must name, not
                        # error="" (bug 128 -- bug 64's shape one layer down).
                        self._force_failure = f"forced submit_report rejected: {text}"[:300]
        return None

    def _prompt(self) -> str:
        parts = [
            f"# Subtask {self.subtask.id}",
            f"\n## Objective\n{self.subtask.objective}",
        ]
        if self.subtask.rationale:
            parts.append(f"\n## Why this matters to the overall question\n{self.subtask.rationale}")
        if self.subtask.queries:
            parts.append(
                "\n## Suggested starting queries\n"
                + "\n".join(f"- {q}" for q in self.subtask.queries)
            )
        if self.subtask.success_criteria:
            parts.append(f"\n## Done when\n{self.subtask.success_criteria}")
        parts.append(f"\n## Default search intent\n{self.subtask.intent}")
        return "\n".join(parts)
