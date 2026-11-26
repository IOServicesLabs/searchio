"""Browser batch 4 (#214+). One class per knob; knob off pins the old
behaviour, knob on pins the new one."""

from __future__ import annotations

import codecs

import pytest

from searchio.config import Settings
from searchio.net import charset as charset_mod
from searchio.net import sidecar as sidecar_mod
from searchio.net.ladder import Ladder
from tests._ab_baseline import BASELINE

FR = ("Les élèves français ont déjà mangé à la crèmerie "
      "près de l'école, où le café est très apprécié. ")
JP = ("日本語のページです。これはテスト用の"
      "文章であり、文字コードの判定を確認します。 ")
ZH = ("这是一个中文网页，用于测试字符编码的自动"
      "检测功能是否正常工作。 ")


def page(text: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body><article><p>{text * 12}</p></article></body></html>"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        state_dir=tmp_path, cache_enabled=False, robots_policy="off", max_tier=2,
        per_domain_rps=1000.0, per_domain_burst=1000, sidecar_autostart=False,
        clearance_enabled=False, sidecar_challenge_auto=False, **BASELINE,
    )


@pytest.fixture
def detect():
    old = charset_mod._ON[0]
    charset_mod._ON[0] = True
    yield
    charset_mod._ON[0] = old


@pytest.fixture
def no_detect():
    old = charset_mod._ON[0]
    charset_mod._ON[0] = False
    yield
    charset_mod._ON[0] = old


class TestCharsetDetect:
    def test_off_is_header_charset_else_lossy_utf8(self, no_detect):
        raw = page(JP).encode("shift_jis")
        assert charset_mod.decode(raw, None, "text/html") == raw.decode("utf-8", "replace")
        assert "�" in charset_mod.decode(raw, None, "text/html")
        assert charset_mod.decode(raw, "shift_jis", "text/html") == page(JP)

    @pytest.mark.parametrize("enc,text", [("cp1252", FR), ("shift_jis", JP), ("gbk", ZH)])
    def test_undeclared_legacy_body_is_detected(self, detect, enc, text):
        got = charset_mod.decode(page(text).encode(enc), None, "text/html")
        assert got == page(text)
        assert "�" not in got

    def test_meta_prescan(self, detect):
        src = page(JP, '<meta charset="Shift_JIS">')
        assert charset_mod.decode(src.encode("shift_jis"), None, "text/html") == src
        src = page(ZH, '<meta http-equiv="Content-Type" content="text/html; charset=gb2312">')
        assert charset_mod.decode(src.encode("gbk"), None, "text/html") == src

    def test_bom_outranks_a_wrong_header(self, detect):
        raw = codecs.BOM_UTF16_LE + "héllo".encode("utf-16-le")
        assert charset_mod.decode(raw, "iso-8859-1", "text/html") == "héllo"
        raw = codecs.BOM_UTF8 + "café".encode()
        assert charset_mod.decode(raw, None, "text/plain") == "café"

    def test_header_wins_over_meta_and_unknown_label_does_not_raise(self, detect):
        src = page(FR, '<meta charset="shift_jis">')
        assert charset_mod.decode(src.encode("cp1252"), "windows-1252", "text/html") == src
        assert charset_mod.decode(b"plain ascii", "x-no-such-charset", "text/html") == "plain ascii"

    def test_latin1_label_decodes_as_windows_1252(self, detect):
        # WHATWG: iso-8859-1 means windows-1252 (curly quotes at 0x93/0x94).
        assert charset_mod.decode(b"\x93hi\x94", "iso-8859-1", "text/html") == "“hi”"

    def test_valid_utf8_and_binary_are_untouched(self, detect):
        raw = page(JP).encode()
        assert charset_mod.decode(raw, None, "text/html") == page(JP)
        blob = b"\x89PNG\r\n\x00\xff\xfe\x80"
        assert charset_mod.decode(blob, None, "image/png") == blob.decode("utf-8", "replace")

    def test_ladder_sets_the_switch_and_the_engine_environment(self, settings):
        Ladder(settings)
        assert charset_mod._ON[0] is False
        assert not sidecar_mod._ENGINE_ENV.get("SE_CHARSET_DETECT")
        Ladder(settings.model_copy(update={"charset_detect": True}))
        assert charset_mod._ON[0] is True
        assert sidecar_mod._ENGINE_ENV["SE_CHARSET_DETECT"] == "1"
        Ladder(settings)
        assert charset_mod._ON[0] is False


class TestEngineLiveDom:
    """#216: the render rung asks the engine for its live DOM only when the
    knob is on; the engine side is pinned by se-serve's livedom_tests."""

    async def _render_params(self, settings, **knobs):
        from tests.test_browser_ab_3 import SHELL, NoHttpLadder, _Engine, _with

        eng = _Engine(page=SHELL)
        lad = NoHttpLadder(_with(settings, engine_render_rung=True, engine_render_prelude=True, **knobs),
                           sidecar=eng)
        res = await lad.fetch("https://spa.example/live", use_cache=False)
        await lad.close()
        assert "readable prose" in res.body
        return [p for v, p in eng.calls if v == "fetch"][0]

    async def test_off_the_param_is_not_sent(self, settings):
        # Shipped on since the batch-two re-measure; `settings` is the baseline.
        assert settings.engine_live_dom is False
        assert BASELINE["engine_live_dom"] is False
        assert "live_dom" not in await self._render_params(settings)

    async def test_on_the_render_asks_for_the_live_dom(self, settings):
        sent = await self._render_params(settings, engine_live_dom=True)
        assert sent["live_dom"] is True and sent["prelude"] is True
