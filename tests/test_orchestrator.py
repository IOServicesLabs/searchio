"""The lead agent's contract: a crashing worker, a bad plan, a lying answer.

Iteration 46 of the span suite (DeepSeek design review of server.py +
swarm/orchestrator.py). Every test here bit RED before its fix.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from searchio.config import Settings
from searchio.errors import ProviderError
from searchio.models import Citation, Doc, Finding, WorkerReport
from searchio.swarm import orchestrator as orch_mod
from searchio.swarm.budget import Budget
from searchio.swarm.orchestrator import Orchestrator


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)


def text_response(text, *, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def plan_json(*objectives, ids=None):
    ids = ids or [f"s{i}" for i in range(len(objectives))]
    return json.dumps({"interpretation": "i",
                       "subtasks": [{"id": i, "objective": o} for i, o in zip(ids, objectives)]})


class FakeWorker:
    raise_for: set[str] = set()

    def __init__(self, engine, client, st, budget, *, effort="", model=""):
        self.st = st

    async def run(self):
        if self.st.id in self.raise_for:
            raise RuntimeError("worker crashed hard")
        url = f"https://{self.st.id}.example/1"
        return WorkerReport(
            subtask_id=self.st.id, summary=f"summary {self.st.id}",
            docs=[Doc(url=url, title=self.st.id, snippet="", source="stub")],
            findings=[Finding(claim="c", confidence="high",
                              citations=[Citation(url=url, title=self.st.id)])],
        )


@pytest.fixture
def eng(tmp_path):
    return SimpleNamespace(s=Settings(state_dir=tmp_path, cache_enabled=False))


@pytest.fixture(autouse=True)
def fake_worker(monkeypatch):
    FakeWorker.raise_for = set()
    monkeypatch.setattr(orch_mod, "Worker", FakeWorker)


def budget(max_workers=2):
    return Budget(max_workers=max_workers)


class TestWaveSurvivesAWorkerCrash:
    async def test_crash_becomes_a_failed_report_not_a_lost_wave(self, eng):
        # Bug 77: dispatch gathered the workers without return_exceptions, so
        # one worker raising aborted /research for everyone -- the finished
        # siblings' reports were thrown away and the still-running ones kept
        # burning tokens as orphans.
        FakeWorker.raise_for = {"s1"}
        client = FakeClient([text_response(plan_json("find a", "find b")),
                             text_response("Answer: see [a](https://s0.example/1).")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget())
        assert [r.subtask_id for r in res.reports] == ["s0", "s1"]
        assert "RuntimeError" in res.reports[1].error
        assert any("s1" in w and "failed" in w for w in res.warnings), res.warnings
        assert res.answer.startswith("Answer")

    async def test_progress_callback_failure_does_not_kill_the_run(self, eng):
        calls = []

        def progress(event, data):
            calls.append(event)
            raise ValueError("listener died")

        client = FakeClient([text_response(plan_json("find a")), text_response("Answer.")])
        res = await Orchestrator(eng, client, model="m", progress=progress).run("q?", budget=budget())
        assert res.answer == "Answer." and "planning" in calls


class TestPlanIsValidated:
    """Bug 78: the planner's text went straight into json.loads / data["subtasks"]
    / SubTask(**st) -- garbage or a refusal escaped as JSONDecodeError /
    KeyError (a bare 500 on /research); an empty plan dispatched nothing and
    synthesised an "answer" grounded in nothing; duplicate subtasks ran twice;
    a plan longer than max_workers was truncated silently.
    """

    async def test_garbage_and_refusal_are_provider_errors(self, eng):
        with pytest.raises(ProviderError, match="planner"):
            await Orchestrator(eng, FakeClient([text_response("not json at all")]), model="m").run(
                "q?", budget=budget())
        with pytest.raises(ProviderError, match="declined"):
            await Orchestrator(eng, FakeClient([text_response("", stop_reason="refusal")]),
                               model="m").run("q?", budget=budget())
        with pytest.raises(ProviderError, match="no usable subtasks"):
            await Orchestrator(eng, FakeClient([text_response(plan_json())]), model="m").run(
                "q?", budget=budget())

    async def test_reassigned_ids_never_collide_with_a_model_given_id(self, eng):
        # Bug 185: the reassignment used f"s{len(subtasks)+1}", which can
        # reproduce an id the model already gave -- if subtask 1 keeps id "s2"
        # (subtasks len -> 1) and subtask 2 needs a new id, s{1+1} is "s2"
        # again, so two subtasks shared one id and their WorkerReports
        # collided downstream. The reassignment must skip taken ids.
        client = FakeClient([text_response(plan_json("find a", "find b", ids=["s2", ""])),
                             text_response("Answer.")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget(max_workers=4))
        ids = [s.id for s in res.plan.subtasks]
        assert len(ids) == 2 and len(set(ids)) == 2, ids

    async def test_duplicates_dropped_and_truncation_recorded(self, eng):
        client = FakeClient([text_response(plan_json("find a", "Find  A", "find b", "find c", "find d")),
                             text_response("Answer.")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget(max_workers=2))
        assert [s.objective for s in res.plan.subtasks] == ["find a", "find b"]
        assert any("truncated" in w and "4" in w and "2" in w for w in res.warnings), res.warnings


class TestAnswerLinksAreGrounded:
    async def test_unretrieved_links_are_neutralized_and_reported(self, eng):
        # Bug 79: the synthesis text went out verbatim -- a markdown link to a
        # URL no worker ever retrieved (the LLM's invention) reached the
        # agent as a citation. Bug 62 closed this for findings; the answer
        # text was the remaining door.
        client = FakeClient([
            text_response(plan_json("find a")),
            text_response("Real [a](https://s0.example/1) and fake [b](https://never.example/x) "
                          "plus bare https://also-fake.example/y here."),
        ])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget())
        assert "https://s0.example/1" in res.answer
        assert "never.example" not in res.answer and "also-fake.example" not in res.answer
        assert "[b]" not in res.answer and " b " in res.answer or "b " in res.answer
        assert any("2 link" in w and "not retrieved" in w for w in res.warnings), res.warnings


class TestSynthesisIsHonest:
    """Iteration 66 (Muse second pass on the orchestrator)."""

    async def test_listing_urls_survive_the_link_guard(self, eng, monkeypatch):
        # Bug 146: the synthesis brief hands the model every listing's URL
        # ("Listings found: ... -- url"), then the unretrieved-link guard
        # stripped those same URLs from the answer because only docs and
        # citations counted as retrieved.
        from searchio.models import Item, Price

        class ItemWorker(FakeWorker):
            async def run(self):
                return WorkerReport(
                    subtask_id=self.st.id, summary="listings",
                    items=[Item(title="Sony WH-1000XM5", url="https://shop.example/p/1",
                                price=Price(amount=299.0, currency="USD"))],
                )
        monkeypatch.setattr(orch_mod, "Worker", ItemWorker)
        client = FakeClient([text_response(plan_json("find it")),
                             text_response("Cheapest: [Sony WH-1000XM5](https://shop.example/p/1) at $299.")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget(1))
        assert "https://shop.example/p/1" in res.answer, res.answer

    async def test_refused_synthesis_is_not_a_blank_success(self, eng):
        # Bug 147: a refusal (or an empty reply) at the synthesis step came
        # back as answer="" with no warning -- a successful-looking nothing.
        client = FakeClient([text_response(plan_json("find a")),
                             text_response("", stop_reason="refusal")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget(1))
        assert res.answer.strip(), "answer must carry the findings when synthesis declines"
        assert "https://s0.example/1" in res.answer
        assert any("synthes" in w.lower() for w in res.warnings), res.warnings

    async def test_empty_synthesis_falls_back_to_the_findings(self, eng):
        client = FakeClient([text_response(plan_json("find a")), text_response("   ")])
        res = await Orchestrator(eng, client, model="m").run("q?", budget=budget(1))
        assert "https://s0.example/1" in res.answer
        assert any("synthes" in w.lower() for w in res.warnings), res.warnings
