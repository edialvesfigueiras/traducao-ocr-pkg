
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    detectar_lingua_auto, detectar_lingua,
    _detectar_unicode, _detectar_langid,
    _ISO_PARA_CHAVE,
)


class TestDeteccaoUnicode:

    def test_bengali_exclusivo(self):
        candidatos = _detectar_unicode("আমি বাংলায় লিখছি")
        assert candidatos
        assert candidatos[0][0] == "bengali"
        assert candidatos[0][1] >= 0.8

    def test_hindi_devanagari(self):
        candidatos = _detectar_unicode("नमस्ते यह हिंदी में लिखा हुआ है")
        assert candidatos
        linguas = [l for l, _ in candidatos]
        assert any(l in ("hindi", "marathi", "nepali") for l in linguas)

    def test_coreano_hangul(self):
        candidatos = _detectar_unicode("안녕하세요 오늘 날씨가 좋습니다")
        assert candidatos
        assert candidatos[0][0] == "coreano"

    def test_grego(self):
        candidatos = _detectar_unicode("Αυτό είναι ελληνικό κείμενο")
        assert candidatos
        assert candidatos[0][0] == "grego"

    def test_hebraico(self):
        candidatos = _detectar_unicode("שלום עולם זה טקסט בעברית")
        assert candidatos
        assert candidatos[0][0] == "hebraico"

    def test_tailandes(self):
        candidatos = _detectar_unicode("สวัสดีครับ วันนี้อากาศดี")
        assert candidatos
        assert candidatos[0][0] == "tailandes"

    def test_japones_hiragana(self):
        candidatos = _detectar_unicode("こんにちは お元気ですか")
        assert candidatos
        assert candidatos[0][0] == "japones"

    def test_cirilico_partilhado(self):
        candidatos = _detectar_unicode("Привет мир")
        assert candidatos
        assert candidatos[0][1] < 0.6

    def test_arabe_partilhado(self):
        candidatos = _detectar_unicode("مرحبا بالعالم")
        assert candidatos
        linguas = [l for l, _ in candidatos]
        assert any(l in ("arabe", "persa", "urdu", "pashto") for l in linguas)

    def test_latino_baixa_confianca(self):
        candidatos = _detectar_unicode("Hello world this is a test")
        assert candidatos
        assert candidatos[0][1] < 0.20

    def test_texto_vazio(self):
        candidatos = _detectar_unicode("")
        assert candidatos == []

    def test_texto_numeros(self):
        candidatos = _detectar_unicode("12345 67890")
        assert candidatos == []

    def test_cjk_chinês(self):
        candidatos = _detectar_unicode("这是中文文本测试")
        assert candidatos
        linguas = [l for l, _ in candidatos]
        assert any(l in ("chines_s", "chines_t") for l in linguas)

    def test_georgiano(self):
        candidatos = _detectar_unicode("გამარჯობა მსოფლიო")
        assert candidatos
        assert candidatos[0][0] == "georgiano"

    def test_armenio(self):
        candidatos = _detectar_unicode("Բարেdelays աdelays")
        candidatos = _detectar_unicode("Բարdelays Հայաստdelays")
        candidatos = _detectar_unicode("\u0532\u0561\u0580\u0565\u0582")
        assert candidatos
        assert candidatos[0][0] == "armenio"


class TestDeteccaoLangid:

    def test_langid_russo(self):
        lingua, conf = _detectar_langid("Привет мир как дела сегодня утром дома")
        if lingua is not None:
            assert lingua in _ISO_PARA_CHAVE.values()
            assert 0.0 <= conf <= 1.0

    def test_langid_texto_curto(self):
        lingua, conf = _detectar_langid("ab")
        assert lingua is None or conf < 0.5

    def test_langid_texto_vazio(self):
        lingua, conf = _detectar_langid("")
        assert lingua is None
        assert conf == 0.0


class TestDeteccaoHibrida:

    def test_bengali_alta_confianca(self):
        lingua, conf = detectar_lingua_auto("আমি বাংলায় লিখছি এই হলো পরীক্ষা")
        assert lingua == "bengali"
        assert conf >= 0.5

    def test_grego_alta_confianca(self):
        lingua, conf = detectar_lingua_auto("Αυτό είναι ελληνικό κείμενο δοκιμή")
        assert lingua == "grego"
        assert conf >= 0.5

    def test_coreano_alta_confianca(self):
        lingua, conf = detectar_lingua_auto("안녕하세요 오늘 날씨가 좋습니다 감사합니다")
        assert lingua == "coreano"
        assert conf >= 0.5

    def test_tailandes(self):
        lingua, conf = detectar_lingua_auto("สวัสดีครับ วันนี้อากาศดีมาก")
        assert lingua == "tailandes"
        assert conf >= 0.5

    def test_japones(self):
        lingua, conf = detectar_lingua_auto("こんにちは お元気ですか 今日は天気がいいです")
        assert lingua == "japones"
        assert conf >= 0.5

    def test_cirilico_detecta_algo(self):
        lingua, conf = detectar_lingua_auto("Привет мир как дела сегодня утром")
        assert lingua is not None
        assert lingua in ("russo", "ucraniano", "bulgaro", "serbio",
                          "macedonio", "bielorrusso", "cazaque")

    def test_arabe_detecta_algo(self):
        lingua, conf = detectar_lingua_auto("مرحبا بالعالم هذا نص عربي طويل بعض الشيء")
        assert lingua is not None
        assert lingua in ("arabe", "persa", "urdu", "pashto")

    def test_texto_curto_retorna_none(self):
        lingua, conf = detectar_lingua_auto("abc")
        assert lingua is None
        assert conf == 0.0

    def test_texto_vazio(self):
        lingua, conf = detectar_lingua_auto("")
        assert lingua is None
        assert conf == 0.0

    def test_texto_so_espacos(self):
        lingua, conf = detectar_lingua_auto("    ")
        assert lingua is None
        assert conf == 0.0

    def test_retorna_tupla(self):
        resultado = detectar_lingua_auto("আমি বাংলায় লিখছি")
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2

    def test_confianca_entre_0_e_1(self):
        _, conf = detectar_lingua_auto("আমি বাংলায় লিখছি এই পরীক্ষা")
        assert 0.0 <= conf <= 1.0

    def test_chineses_cjk(self):
        lingua, conf = detectar_lingua_auto("这是中文文本测试句子")
        assert lingua is not None
        assert lingua in ("chines_s", "chines_t", "japones")

    def test_hindi_devanagari(self):
        lingua, conf = detectar_lingua_auto("नमस्ते यह हिंदी में लिखा हुआ है आज")
        assert lingua is not None
        assert lingua in ("hindi", "marathi", "nepali")

    def test_tamil(self):
        lingua, conf = detectar_lingua_auto("வணக்கம் இது தமிழ் உரை சோதனை")
        assert lingua == "tamil"

    def test_telugu(self):
        lingua, conf = detectar_lingua_auto("హలో ఇది తెలుగు పరీక్ష")
        assert lingua == "telugu"


class TestAPILegada:

    def test_detectar_lingua_wrapper(self):
        resultado = detectar_lingua("আমি বাংলায় লিখছি এই পরীক্ষা")
        assert resultado == "bengali"

    def test_detectar_lingua_none_para_curto(self):
        resultado = detectar_lingua("ab")
        assert resultado is None


class TestMapeamentoISO:

    def test_iso_para_chave_nao_vazio(self):
        assert len(_ISO_PARA_CHAVE) > 30

    def test_iso_bengali(self):
        assert _ISO_PARA_CHAVE["bn"] == "bengali"

    def test_iso_russo(self):
        assert _ISO_PARA_CHAVE["ru"] == "russo"

    def test_iso_arabe(self):
        assert _ISO_PARA_CHAVE["ar"] == "arabe"

    def test_iso_japones(self):
        assert _ISO_PARA_CHAVE["ja"] == "japones"

    def test_iso_espanhol(self):
        assert _ISO_PARA_CHAVE["es"] == "espanhol"

    def test_iso_frances(self):
        assert _ISO_PARA_CHAVE["fr"] == "frances"
