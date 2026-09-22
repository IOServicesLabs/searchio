"""The operator surface: Settings validation, secret hygiene, CLI exit codes.

Iteration 49 of the span suite (DeepSeek operator review: cli.py + config.py
+ net/pdf.py). Every test here bit RED before its fix.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from searchio.config import Settings, set_settings


class TestSettingsValidation:
    """Bug 87: Settings validated nothing. per_domain_burst=0 hung the
    limiter (bug 68 clamped it downstream), robots_policy='bogus' silently
    meant 'warn'-ish, max_workers=0 dispatched nothing, timeouts of 0 and
    negative TTLs flowed straight into consumers. The environment is where
    these come from (SEARCHIO_*), and a typo must fail at startup with the
    field's name, not surface as a hang three modules away.
    """

    @pytest.mark.parametrize("kw", [
        {"per_domain_rps": -1}, {"per_domain_rps": 0}, {"per_domain_burst": 0},
        {"cache_ttl_s": -5}, {"clearance_ttl_s": -1}, {"max_workers": 0},
        {"max_tool_calls_per_worker": -3}, {"worker_concurrency": 0},
        {"robots_policy": "bogus"}, {"llm_provider": "bogus"}, {"min_text_chars": -1},
        {"pdf_max_bytes": 0}, {"pdf_max_pages": 0}, {"max_body_bytes": 0},
        {"fetch_timeout_s": 0}, {"provider_timeout_s": -2}, {"handoff_timeout_s": -1},
        {"swarm_wall_clock_s": 0}, {"swarm_token_budget": 0}, {"max_tier": 3},
        {"max_tier": -1}, {"fanout": 0}, {"results_per_provider": 0},
        {"global_concurrency": 0}, {"max_redirects": -1},
        # Bug 180: the tier timeouts ARE timeouts (tier2 -> int(x*1000/3)),
        # but only their siblings were positive-checked; a zero/negative one
        # slipped through as a broken timeout instead of a startup error.
        {"tier0_timeout_s": 0}, {"tier1_timeout_s": -1}, {"tier2_timeout_s": 0},
        # Bug 181: string enums accepted any value -- a typo disabled the
        # challenge rescue or produced a late API error, never a config error.
        {"sidecar_challenge": "patchrite"}, {"lead_effort": "hihg"},
        {"worker_effort": "bogus"},
    ])
    def test_bad_values_fail_at_construction_naming_the_field(self, kw, tmp_path):
        field = next(iter(kw))
        with pytest.raises(ValidationError) as exc:
            Settings(state_dir=tmp_path, **kw)
        assert field in str(exc.value), str(exc.value)[:200]

    def test_defaults_and_sane_overrides_pass(self, tmp_path):
        s = Settings(state_dir=tmp_path)
        assert s.robots_policy == "warn" and s.per_domain_burst >= 1
        s = Settings(state_dir=tmp_path, per_domain_rps=1000.0, per_domain_burst=1000, robots_policy="off")
        assert s.per_domain_max_rps >= 1000.0  # the ceiling follows the floor, not a refusal
        # The valid enum values and tier timeouts still pass (no false reject).
        s = Settings(state_dir=tmp_path, tier2_timeout_s=30.0, sidecar_challenge="engine",
                     lead_effort="medium", worker_effort="high")
        assert s.sidecar_challenge == "engine" and s.lead_effort == "medium"


class TestSecretsNeverEcho:
    def test_keys_absent_from_repr_str_and_dump(self, tmp_path):
        # Bug 88: every key field was a plain str, so repr(settings) /
        # model_dump() -- a debug print, a log line, an error message with
        # the settings attached -- carried the API keys and a proxy URL's
        # embedded password.
        s = Settings(state_dir=tmp_path, anthropic_api_key="sk-ant-SECRET1", deepseek_api_key="sk-SECRET2",
                     serpapi_key="SECRET3", github_token="ghp_SECRET4", sidecar_token="SECRET5",
                     proxy="http://user:PWSECRET6@proxy.example:8080")
        blob = repr(s) + str(s) + json.dumps(s.model_dump(), default=str)
        assert "SECRET" not in blob, blob[:300]
        # ...while the code that needs them still reads them.
        assert s.anthropic_key() == "sk-ant-SECRET1" and s.deepseek_api_key == "sk-SECRET2"
        assert s.proxy.endswith("@proxy.example:8080")


class TestCliExpectedFailuresAreOneLiners:
    """Bug 89: cli._run mapped ConfigError/SearchioError/Ctrl-C to exit codes
    but let a pydantic ValidationError (an invalid --intent, a bad setting
    from the environment) escape as a rich traceback -- exit 1 with forty
    lines of frames for an expected input error.
    """

    def test_invalid_intent_is_a_usage_error_not_a_traceback(self, tmp_path, monkeypatch):
        from searchio.cli import app

        set_settings(Settings(state_dir=tmp_path, cache_enabled=False))
        res = CliRunner().invoke(app, ["search", "x", "--intent", "bogus"])
        assert res.exit_code == 2, res.output[:300]
        assert "Traceback" not in res.output and "intent" in res.output

    def test_research_without_a_key_is_a_config_error(self, tmp_path, monkeypatch):
        from searchio.cli import app

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        set_settings(Settings(state_dir=tmp_path, cache_enabled=False, anthropic_api_key=""))
        res = CliRunner().invoke(app, ["research", "what is x?"])
        assert res.exit_code == 2, res.output[:300]
        assert "Traceback" not in res.output and "config" in res.output.lower()

    def test_bad_environment_setting_is_a_config_error(self, tmp_path, monkeypatch):
        from searchio import config as cfg
        from searchio.cli import app

        monkeypatch.setenv("SEARCHIO_PER_DOMAIN_BURST", "0")
        monkeypatch.setenv("SEARCHIO_STATE_DIR", str(tmp_path))
        cfg.set_settings(None)  # force a fresh Settings() from the environment
        res = CliRunner().invoke(app, ["providers"])
        assert res.exit_code == 2, res.output[:300]
        assert "Traceback" not in res.output and "per_domain_burst" in res.output


from types import SimpleNamespace as _NS  # noqa: E402

from searchio.models import Doc, ResearchResult  # noqa: E402


class _FakeEngine:
    """Engine stand-in for the CLI: no network, long strings to defeat wrapping."""

    def __init__(self, settings):
        self.s = settings

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def search(self, text, **kw):
        doc = Doc(url="https://a.example/" + "p" * 300, title="T" * 200, snippet="s" * 400, source="stub")
        return _NS(docs=[doc], used=["stub"], failed={}, elapsed_ms=1)

    async def research(self, question, *, progress=None):
        if progress:
            progress("planning", {"question": question})
            progress("synthesising", {"findings": 0, "sources": 0})
        return ResearchResult(query=question, answer="A" * 500)


class TestJsonIsForPipes:
    """Bug 90 (review candidates 2 + 6): --json went through rich's
    console.print(JSON(...)), which wraps at the terminal width (80 in a
    pipe) -- a 300-character URL split across lines is not JSON any more;
    `research --json` printed its progress lines on stdout BEFORE the
    payload; and every error line went to stdout too. An agent or a shell
    piping --json got a corrupt document. stdout is the payload, stderr is
    everything else.
    """

    def test_search_json_is_parseable_on_stdout_alone(self, tmp_path, monkeypatch):
        from searchio import cli

        monkeypatch.setattr(cli, "Engine", _FakeEngine)
        set_settings(Settings(state_dir=tmp_path, cache_enabled=False))
        res = CliRunner().invoke(app_of(cli), ["search", "x", "--json"])
        assert res.exit_code == 0, res.output[:300]
        data = json.loads(res.stdout)  # used to fail: rich wrapped the 300-char url
        assert data["results"][0]["url"].startswith("https://a.example/ppp")

    def test_research_json_keeps_progress_off_stdout(self, tmp_path, monkeypatch):
        from searchio import cli

        monkeypatch.setattr(cli, "Engine", _FakeEngine)
        set_settings(Settings(state_dir=tmp_path, cache_enabled=False))
        res = CliRunner().invoke(app_of(cli), ["research", "what is x?", "--json"])
        assert res.exit_code == 0, res.output[:300]
        data = json.loads(res.stdout)
        assert data["answer"].startswith("AAAA") and "planning" not in res.stdout
        assert "planning" in res.stderr

    def test_errors_go_to_stderr(self, tmp_path, monkeypatch):
        from searchio import cli

        set_settings(Settings(state_dir=tmp_path, cache_enabled=False))
        res = CliRunner().invoke(app_of(cli), ["search", "x", "--intent", "bogus", "--json"])
        assert res.exit_code == 2
        assert res.stdout.strip() == "" and "intent" in res.stderr


    def test_all_providers_failed_is_exit_1(self, tmp_path, monkeypatch):
        # Rider (dropped candidate): `search` printed "No results." and
        # exited 0 when every provider had failed -- the CLI twin of bug 75.
        from searchio import cli

        class Dead(_FakeEngine):
            async def search(self, text, **kw):
                return _NS(docs=[], used=[], failed={"bing": "circuit open", "ddg": "429"}, elapsed_ms=1)

        monkeypatch.setattr(cli, "Engine", Dead)
        set_settings(Settings(state_dir=tmp_path, cache_enabled=False))
        res = CliRunner().invoke(app_of(cli), ["search", "x"])
        assert res.exit_code == 1, res.output[:200]
        assert "all providers failed" in res.stderr and "bing" in res.stderr


def app_of(cli):
    return cli.app


class TestPdfPageCapIsSane:
    def test_negative_or_zero_max_pages_reads_one_page(self):
        # Rider (review candidate 4): pages[:max_pages] with a negative cap
        # silently dropped the LAST page(s) instead of bounding the read.
        from searchio.net.pdf import build_pdf, extract_text

        pdf = build_pdf(["one", "two", "three"])
        text, used, total = extract_text(pdf, max_pages=-1)
        assert (used, total) == (1, 3) and "one" in text
        text, used, total = extract_text(pdf, max_pages=0)
        assert used == 1



class TestItemCommandsNameCooling:
    """Bug 182: the CLI `items`/`marketplace` commands printed a calm
    "No listings found." and exited 0 when every shopping/local provider was
    circuit-open from earlier failures -- the CLI twin of bug 75/164, which
    `search`, /search and the tool bridge all name. They now report the
    cooldown and exit 1; a genuine empty (nobody cooling) stays exit 0.
    """

    def _engine(self, tmp_path):
        from searchio.engine import Engine
        return Engine(Settings(state_dir=tmp_path, cache_enabled=False))

    async def test_cooling_shopping_providers_are_reported(self, tmp_path):
        import time

        from searchio.cli import _cooling
        eng = self._engine(tmp_path)
        shopping = [p for p in eng.router.registry.all()
                    if p.configured(eng.s) and p.supports("shopping")]
        assert shopping, "the fixture needs at least one shopping provider"
        for p in shopping:
            p.health.opened_at = time.monotonic()
            p.health.consecutive_failures = 3
        cooling = _cooling(eng, "shopping")
        assert set(cooling) == {p.name for p in shopping}, cooling
        # A healthy provider is not reported.
        for p in shopping:
            p.health.opened_at = 0.0
        assert _cooling(eng, "shopping") == {}

    async def test_report_no_items_exits_one_when_cooling_else_zero(self, tmp_path):
        import time

        import typer

        from searchio.cli import _report_no_items
        eng = self._engine(tmp_path)
        # No cooling -> honest empty, no exception (exit 0).
        _report_no_items(eng, "shopping")
        # All shopping providers cooling -> Exit(1).
        for p in eng.router.registry.all():
            if p.supports("shopping"):
                p.health.opened_at = time.monotonic()
                p.health.consecutive_failures = 3
        with pytest.raises(typer.Exit) as exc:
            _report_no_items(eng, "shopping")
        assert exc.value.exit_code == 1
