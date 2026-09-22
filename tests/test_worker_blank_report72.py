"""A blank submit_report must not close a worker (iteration 72, caught live:
run 71's llm.non_english read five German pages, then the model called
submit_report({}) and the bridge answered "Report recorded." -- a report
with no summary, no findings and no gaps went out as an unforced success
with 0 citations). Red first."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.models import SubTask
from searchio.swarm.budget import Budget
from searchio.swarm.tools import ToolBridge
from searchio.swarm.worker import Worker

from tests.test_worker_second_pass import FakeClient, FakeEngine, response, tool_use


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, max_tool_calls_per_worker=4)


SUB = SubTask(id="s1", objective="Find out X", intent="web")


class TestTheBridgeRefusesABlankReport:
    @pytest.mark.parametrize("args", [
        {},
        {"summary": "", "findings": []},
        {"summary": "   ", "findings": [], "gaps": []},
        {"summary": "", "findings": [{"claim": "", "confidence": "low", "citations": []}]},
    ], ids=["empty", "blank-summary", "whitespace", "blank-finding"])
    async def test_nothing_is_recorded_and_the_model_is_told_what_a_report_needs(self, settings, args):
        bridge = ToolBridge(FakeEngine(settings), subtask_id="s1")
        text, is_error = await bridge.dispatch("submit_report", args)
        assert is_error, text
        assert bridge.report is None, "a blank report closed the worker"
        assert "summary" in text and "gaps" in text, text
        assert "Report recorded" not in text, text

    @pytest.mark.parametrize("args", [
        {"summary": "Nothing conclusive.", "findings": []},
        {"summary": "", "findings": [], "gaps": ["No German source named the threshold."]},
        {"summary": "", "findings": [{"claim": "X is Y", "confidence": "low", "citations": []}]},
    ], ids=["summary-only", "gaps-only", "claim-only"])
    async def test_an_honest_thin_report_is_still_a_report(self, settings, args):
        bridge = ToolBridge(FakeEngine(settings), subtask_id="s1")
        text, is_error = await bridge.dispatch("submit_report", args)
        assert not is_error, text
        assert bridge.report is not None


class TestTheWorkerKeepsGoingAfterABlankSubmit:
    async def test_the_model_gets_a_second_chance_and_the_real_report_wins(self, settings):
        client = FakeClient([
            response([tool_use("web_search", {"query": "x", "intent": "web"}, tid="t1")]),
            response([tool_use("submit_report", {}, tid="t2")]),
            response([tool_use("submit_report", {"summary": "real", "findings": [],
                                                 "gaps": ["one thing"]}, tid="t3")]),
        ])
        report = await Worker(FakeEngine(settings), client, SUB, Budget(max_tool_calls_per_worker=4)).run()
        assert report.summary == "real", report
        assert client.calls == 3, client.calls

    async def test_a_blank_forced_submit_is_named_as_the_failure(self, settings):
        never = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")])
                 for i in range(4)]
        blank = [response([tool_use("submit_report", {}, tid="tf1")]),
                 response([tool_use("submit_report", {}, tid="tf2")])]
        client = FakeClient(never + blank)
        report = await Worker(FakeEngine(settings), client, SUB, Budget(max_tool_calls_per_worker=4)).run()
        assert report.error and "rejected" in report.error, report.error
        assert report.docs, "the retrieved pages are still kept"
