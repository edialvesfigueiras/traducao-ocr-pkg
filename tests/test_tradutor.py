
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tradutor import (
    _hash_texto, _segmentar_texto,
    carregar_cache, guardar_cache, limpar_cache,
    num_entradas_cache,
)
from src.modelo_router import (
    obter_backend, obter_nome_modelo, listar_modelos_disponiveis,
)
from src.config import MODELOS_ESPECIALIZADOS, MODELO_PIVOT_EN_PT


class TestHashTexto:
    def test_hash_determinista(self):
        h1 = _hash_texto("texto", "bengali")
        h2 = _hash_texto("texto", "bengali")
        assert h1 == h2

    def test_hash_diferente_lingua(self):
        h1 = _hash_texto("texto", "bengali")
        h2 = _hash_texto("texto", "russo")
        assert h1 != h2

    def test_hash_diferente_texto(self):
        h1 = _hash_texto("texto1", "bengali")
        h2 = _hash_texto("texto2", "bengali")
        assert h1 != h2

    def test_hash_md5_format(self):
        h = _hash_texto("qualquer", "qualquer")
        assert len(h) == 32


class TestSegmentarTexto:
    def test_texto_curto_nao_segmenta(self):
        texto = "Texto curto."
        resultado = _segmentar_texto(texto)
        assert len(resultado) == 1
        assert resultado[0] == texto

    def test_texto_longo_segmenta(self):
        texto = ". ".join(["Frase número " + str(i) for i in range(200)])
        resultado = _segmentar_texto(texto)
        assert len(resultado) >= 2

    def test_retorna_lista(self):
        resultado = _segmentar_texto("Texto simples.")
        assert isinstance(resultado, list)


class TestCache:
    def test_limpar_cache(self):
        limpar_cache()
        assert num_entradas_cache() == 0


class TestTradutorMocked:

    @patch("src.tradutor._model")
    @patch("src.tradutor._modelo_usado", "facebook/nllb-200-distilled-600M")
    def test_modelo_usado_retorna_valor(self, mock_model):
        from src.tradutor import get_modelo_usado
        assert get_modelo_usado() == "facebook/nllb-200-distilled-600M"

    def test_get_device_retorna_string(self):
        from src.tradutor import get_device
        device = get_device()
        assert device in ("cpu", "cuda")


class TestObterBackend:

    def test_lingua_sem_modelo_especializado_retorna_nllb(self):
        assert obter_backend("bengali") == "nllb"

    def test_lingua_sem_modelo_especializado_birmanes(self):
        assert obter_backend("birmanes") == "nllb"

    def test_lingua_inexistente_retorna_nllb(self):
        assert obter_backend("klingon") == "nllb"

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_espanhol_directo_se_em_cache(self, mock_cache):
        assert obter_backend("espanhol") == "opus-mt-directo"

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_frances_directo_se_em_cache(self, mock_cache):
        assert obter_backend("frances") == "opus-mt-directo"

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_italiano_directo_se_em_cache(self, mock_cache):
        assert obter_backend("italiano") == "opus-mt-directo"

    @patch("src.modelo_router.modelo_em_cache", return_value=False)
    def test_espanhol_sem_cache_retorna_nllb(self, mock_cache):
        assert obter_backend("espanhol") == "nllb"

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_russo_pivot_se_em_cache(self, mock_cache):
        assert obter_backend("russo") == "opus-mt-pivot"

    @patch("src.modelo_router.modelo_em_cache")
    def test_russo_sem_pivot_retorna_nllb(self, mock_cache):
        def side_effect(nome):
            if nome == MODELO_PIVOT_EN_PT:
                return False
            return True
        mock_cache.side_effect = side_effect
        assert obter_backend("russo") == "nllb"


class TestObterNomeModelo:

    def test_nllb_para_lingua_sem_especializado(self):
        nome = obter_nome_modelo("bengali")
        assert "NLLB" in nome

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_espanhol_mostra_directo(self, mock_cache):
        nome = obter_nome_modelo("espanhol")
        assert "directo" in nome.lower() or "PT" in nome

    @patch("src.modelo_router.modelo_em_cache", return_value=True)
    def test_russo_mostra_pivot(self, mock_cache):
        nome = obter_nome_modelo("russo")
        assert "pivot" in nome.lower() or "en" in nome.lower()


class TestListarModelos:

    def test_retorna_lista(self):
        modelos = listar_modelos_disponiveis()
        assert isinstance(modelos, list)

    def test_inclui_pivot(self):
        modelos = listar_modelos_disponiveis()
        linguas = [m["lingua"] for m in modelos]
        assert "_pivot_en_pt" in linguas

    def test_inclui_todas_linguas_especializadas(self):
        modelos = listar_modelos_disponiveis()
        linguas = {m["lingua"] for m in modelos}
        for lingua in MODELOS_ESPECIALIZADOS:
            assert lingua in linguas, f"{lingua} não encontrado na listagem"

    def test_cada_modelo_tem_campos(self):
        modelos = listar_modelos_disponiveis()
        for m in modelos:
            assert "lingua" in m
            assert "nome_lingua" in m
            assert "modelo" in m
            assert "directo_pt" in m
            assert "em_cache" in m

    def test_directo_pt_para_espanhol(self):
        modelos = listar_modelos_disponiveis()
        for m in modelos:
            if m["lingua"] == "espanhol":
                assert m["directo_pt"] is True
                break
        else:
            pytest.fail("Espanhol não encontrado")

    def test_pivot_para_russo(self):
        modelos = listar_modelos_disponiveis()
        for m in modelos:
            if m["lingua"] == "russo":
                assert m["directo_pt"] is False
                break
        else:
            pytest.fail("Russo não encontrado")


class TestModelosConfig:

    def test_modelos_especializados_nao_vazio(self):
        assert len(MODELOS_ESPECIALIZADOS) > 0

    def test_cada_modelo_tem_campos(self):
        for lingua, config in MODELOS_ESPECIALIZADOS.items():
            assert "modelo" in config, f"{lingua} sem 'modelo'"
            assert "directo_pt" in config, f"{lingua} sem 'directo_pt'"

    def test_modelo_pivot_definido(self):
        assert MODELO_PIVOT_EN_PT.startswith("Helsinki-NLP/")

    def test_directos_sao_para_pt(self):
        for lingua, config in MODELOS_ESPECIALIZADOS.items():
            if config["directo_pt"]:
                assert "-pt" in config["modelo"], \
                    f"{lingua}: modelo directo deve ter '-pt' no nome"

    def test_pivots_sao_para_en(self):
        for lingua, config in MODELOS_ESPECIALIZADOS.items():
            if not config["directo_pt"]:
                assert "-en" in config["modelo"], \
                    f"{lingua}: modelo pivot deve ter '-en' no nome"
