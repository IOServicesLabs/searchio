"""The forced submit must respect the budget on EVERY exit path (iteration
87, worker.py third pass). Bug 126 stopped the forced turn after a
soft_check-triggered stop; bug 184 is its sibling -- a loop that exits via
the tool-call cap (or the prose cap) while the token/wall budget is ALSO
exhausted left budget_stopped False, so the forced turn ran and overspent.
Red first."""

from __future__ import annotations

import pytest

from searchio.config import Settings
from searchio.models import SubTask
from searchio.swarm.budget import Budget
from searchio.swarm.worker import Worker

from tests.test_worker_second_pass import FakeClient, FakeEngine, response, tool_use


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, max_tool_calls_per_worker=4)


SUB = SubTask(id="s1", objective="Find out X", intent="web")


class TestForcedSubmitRespectsTheBudgetOnACapExit:
    async def test_a_tool_cap_exit_with_an_exhausted_budget_does_not_force_submit(self, settings):
        # Four searches, each recording 4k tokens: soft_check passes at the
        # TOP of every iteration (12k < 15k before the 4th call), so the loop
        # exits on bridge.calls >= max_calls, not soft_check -- budget_stopped
        # stays False -- and by then the budget is exhausted (16k >= 15k). The
        # forced turn must NOT run.
        searches = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")],
                             in_tok=2000, out_tok=2000) for i in range(4)]
        # These would be consumed only if the forced turn wrongly ran.
        forced = [response([tool_use("submit_report", {"summary": "late", "findings": []}, tid="tf")])
                  for _ in range(2)]
        client = FakeClient(searches + forced)
        budget = Budget(max_tokens=15_000, max_tool_calls_per_worker=4)
        report = await Worker(FakeEngine(settings), client, SUB, budget).run()
        assert client.calls == 4, f"the forced turn overspent an exhausted budget: {client.calls} calls"
        # The worker still hands back the incomplete fallback with what it
        # retrieved (a tool-cap exit's error is "" by design, bug 64); the
        # point is the forced turn did not run.
        assert report is not None and "Did not complete" in report.summary, report.summary

    async def test_a_cap_exit_with_budget_left_still_force_submits(self, settings):
        # The other side of the gate (regression guard for the common case,
        # bug 63): a tool-cap exit while the budget HAS room must still spend
        # the one harvesting submit-only turn.
        searches = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")],
                             in_tok=10, out_tok=10) for i in range(4)]
        forced = response([tool_use("submit_report", {"summary": "forced", "findings": []}, tid="tf")])
        client = FakeClient(searches + [forced])
        report = await Worker(FakeEngine(settings), client, SUB,
                              Budget(max_tokens=1_000_000, max_tool_calls_per_worker=4)).run()
        assert report.summary == "forced", report
        assert client.calls == 5
