
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.limpeza import (
    limpar_texto_ocr, normalizar_texto, dividir_em_paragrafos, _linha_e_lixo,
)


class TestLinhaLixo:
    def test_url_http(self):
        assert _linha_e_lixo("http://example.com")

    def test_url_youtube(self):
        assert _linha_e_lixo("youtube.com/watch?v=123")

    def test_url_telegram(self):
        assert _linha_e_lixo("t.me/canal_qualquer")

    def test_url_facebook(self):
        assert _linha_e_lixo("facebook.com/pagina")

    def test_url_instagram(self):
        assert _linha_e_lixo("Siga no instagram")

    def test_subscribe(self):
        assert _linha_e_lixo("Please subscribe and share")

    def test_texto_bengali_nao_e_lixo(self):
        assert not _linha_e_lixo("আমি বাংলায় গান গাই")

    def test_texto_russo_nao_e_lixo(self):
        assert not _linha_e_lixo("Привет, как дела? Все хорошо.")

    def test_texto_arabe_nao_e_lixo(self):
        assert not _linha_e_lixo("مرحبا بالعالم العربي")

    def test_linha_vazia(self):
        assert not _linha_e_lixo("")

    def test_so_simbolos(self):
        assert _linha_e_lixo("___---***")

    def test_underscore_sem_espacos(self):
        assert _linha_e_lixo("watermark_channel_name")

    def test_ascii_curto(self):
        assert _linha_e_lixo("abc123")

    def test_ascii_longo_nao_e_lixo(self):
        assert not _linha_e_lixo("This is a longer line that should not be classified as junk content")


class TestNormalizarTexto:
    def test_nfc_normalization(self):
        resultado = normalizar_texto("caf\u0065\u0301")
        assert "é" in resultado

    def test_remove_zero_width(self):
        resultado = normalizar_texto("texto\u200Bcom\u200Bzero\u200Bwidth")
        assert "\u200B" not in resultado

    def test_remove_tabs(self):
        resultado = normalizar_texto("texto\tcom\ttabs")
        assert "\t" not in resultado

    def test_colapsa_espacos(self):
        resultado = normalizar_texto("texto   com   espaços")
        assert "   " not in resultado

    def test_pipe_para_danda_com_script_indico(self):
        resultado = normalizar_texto("বাংলা|texto")
        assert "।" in resultado

    def test_pipe_nao_muda_sem_script_indico(self):
        resultado = normalizar_texto("texto|mais")
        assert "|" in resultado

    def test_remove_soft_hyphen(self):
        resultado = normalizar_texto("texto\u00ADcom\u00ADhyphen")
        assert "\u00AD" not in resultado


class TestLimparTextoOcr:
    def test_remove_urls(self):
        texto = "আমি বাংলায় গান গাই importante\nhttp://spam.com\nআরও টেক্সট"
        resultado = limpar_texto_ocr(texto)
        assert "http" not in resultado
        assert "আমি" in resultado

    def test_remove_watermarks_underscore(self):
        texto = "Texto real\ncanal_nome_watermark\nMais texto"
        resultado = limpar_texto_ocr(texto)
        assert "watermark" not in resultado

    def test_colapsa_linhas_vazias(self):
        texto = "Parágrafo 1\n\n\n\n\nParágrafo 2"
        resultado = limpar_texto_ocr(texto)
        assert "\n\n\n" not in resultado

    def test_preserva_texto_real(self):
        texto = "আমি বাংলায় গান গাই\nএটা একটা পরীক্ষা"
        resultado = limpar_texto_ocr(texto)
        assert "আমি" in resultado

    def test_texto_vazio(self):
        assert limpar_texto_ocr("") == ""

    def test_so_lixo(self):
        resultado = limpar_texto_ocr("http://spam.com\nwww.junk.org")
        assert resultado == "" or len(resultado) < 5


class TestDividirParagrafos:
    def test_dois_paragrafos(self):
        texto = "Primeiro parágrafo aqui.\n\nSegundo parágrafo aqui."
        resultado = dividir_em_paragrafos(texto)
        assert len(resultado) == 2

    def test_ignora_paragrafos_curtos(self):
        texto = "OK\n\nSegundo parágrafo completo."
        resultado = dividir_em_paragrafos(texto)
        assert len(resultado) == 1

    def test_junta_linhas_no_paragrafo(self):
        texto = "Linha 1 do parágrafo\nLinha 2 do mesmo\n\nOutro parágrafo."
        resultado = dividir_em_paragrafos(texto)
        assert len(resultado) == 2
        assert "Linha 1" in resultado[0]
        assert "Linha 2" in resultado[0]

    def test_texto_vazio(self):
        assert dividir_em_paragrafos("") == []

    def test_um_paragrafo(self):
        resultado = dividir_em_paragrafos("Texto simples sem quebras.")
        assert len(resultado) == 1

    def test_multiplas_linhas_vazias(self):
        texto = "Parágrafo A\n\n\n\n\nParágrafo B"
        resultado = dividir_em_paragrafos(texto)
        assert len(resultado) == 2
