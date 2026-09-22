"""worker.py second pass (iteration 59, DeepSeek): the forced submit and
the nudge loop. Every test here bit RED before its fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.models import SubTask
from searchio.swarm.budget import Budget
from searchio.swarm.worker import Worker


@pytest.fixture
def settings(tmp_path):
    return Settings(state_dir=tmp_path, cache_enabled=False, max_tool_calls_per_worker=4)


class FakeEngine:
    def __init__(self, settings):
        self.s = settings
        self.router = SimpleNamespace()
        self.ladder = SimpleNamespace()

    async def search(self, text, **kw):
        from searchio.models import Doc

        return SimpleNamespace(docs=[Doc(url="https://a.com/1", title="R", snippet="s", source="stub")],
                               used=["stub"], failed={}, filtered={}, per_provider={}, elapsed_ms=1)


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.calls += 1
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)


def response(content, *, stop_reason="tool_use", in_tok=100, out_tok=50):
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok))


def tool_use(name, inp, tid="t1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=tid)


def prose(text="Here is what I found so far."):
    return SimpleNamespace(type="text", text=text)


SUB = SubTask(id="s1", objective="Find out X", intent="web")


class TestForcedSubmitRespectsTheBudgetStop:
    async def test_no_extra_calls_after_the_token_budget_stops_the_worker(self, settings):
        # Bug 126 (DeepSeek #1): the loop broke on soft_check() but the
        # forced-submit turn ran anyway -- up to two more 8k-token calls per
        # worker AFTER the shared budget said stop.
        client = FakeClient([
            response([tool_use("web_search", {"query": "x", "intent": "web"})], in_tok=10_000, out_tok=10_000),
            response([tool_use("submit_report", {"summary": "late", "findings": []}, tid="t2")]),
            response([tool_use("submit_report", {"summary": "late", "findings": []}, tid="t3")]),
        ])
        budget = Budget(max_tokens=15_000)
        report = await Worker(FakeEngine(settings), client, SUB, budget).run()
        assert client.calls == 1, client.calls
        assert report.error.startswith("token budget")

    async def test_forced_submit_still_runs_when_only_tool_calls_ran_out(self, settings):
        never = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")]) for i in range(4)]
        forced = response([tool_use("submit_report", {"summary": "forced", "findings": []}, tid="tf")])
        client = FakeClient(never + [forced])
        report = await Worker(FakeEngine(settings), client, SUB, Budget(max_tool_calls_per_worker=4)).run()
        assert report.summary == "forced"


class TestProseOnlyModelIsBounded:
    async def test_repeated_prose_turns_do_not_loop_until_the_tokens_die(self, settings):
        # Bug 127 (DeepSeek #4): a model that answered in prose every turn was
        # nudged every turn -- bridge.calls never moved, so the only bound was
        # the token budget: dozens of 8k-token turns for one worker.
        client = FakeClient([response([prose()]) for _ in range(30)])
        report = await Worker(FakeEngine(settings), client, SUB, Budget(max_tool_calls_per_worker=4)).run()
        assert client.calls <= 5, client.calls
        assert report is not None


class TestForcedSubmitFailureIsVisible:
    async def test_rejected_forced_submit_names_the_rejection(self, settings):
        # Bug 128 (DeepSeek #5): the forced turn's submit_report was
        # dispatched and its rejection ignored; the fallback report went out
        # with error="" -- bug 64's shape one layer down.
        never = [response([tool_use("web_search", {"query": "x", "intent": "web"}, tid=f"t{i}")]) for i in range(4)]
        bad = [response([tool_use("submit_report", {"_error": "malformed JSON arguments"}, tid="tf1")]),
               response([tool_use("submit_report", {"_error": "malformed JSON arguments"}, tid="tf2")])]
        client = FakeClient(never + bad)
        report = await Worker(FakeEngine(settings), client, SUB, Budget(max_tool_calls_per_worker=4)).run()
        assert report.error and "malformed" in report.error, report.error
