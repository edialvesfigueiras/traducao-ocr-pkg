
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import LINGUAS, SCRIPTS, PARES_DIRECTOS, obter_lingua, listar_linguas


class TestLinguas:
    def test_linguas_nao_vazio(self):
        assert len(LINGUAS) > 40

    def test_cada_lingua_tem_5_campos(self):
        for chave, valor in LINGUAS.items():
            assert len(valor) == 5, f"Língua '{chave}' tem {len(valor)} campos, esperado 5"

    def test_cada_lingua_tem_nome(self):
        for chave, (nome, _, _, _, _) in LINGUAS.items():
            assert nome, f"Língua '{chave}' sem nome"

    def test_cada_lingua_tem_nllb_code(self):
        for chave, (_, nllb, _, _, _) in LINGUAS.items():
            assert nllb, f"Língua '{chave}' sem NLLB code"
            assert "_" in nllb, f"NLLB code '{nllb}' inválido para '{chave}'"

    def test_cada_lingua_tem_tesseract_code(self):
        for chave, (_, _, tess, _, _) in LINGUAS.items():
            assert tess, f"Língua '{chave}' sem Tesseract code"

    def test_cada_lingua_tem_script(self):
        for chave, (_, _, _, script, _) in LINGUAS.items():
            assert script, f"Língua '{chave}' sem script"


class TestScripts:
    def test_scripts_nao_vazio(self):
        assert len(SCRIPTS) > 5

    def test_todos_scripts_tem_linguas(self):
        for script, items in SCRIPTS.items():
            assert len(items) > 0, f"Script '{script}' sem línguas"

    def test_bengali_em_scripts(self):
        found = False
        for script, items in SCRIPTS.items():
            for chave, nome in items:
                if chave == "bengali":
                    found = True
        assert found, "Bengali não encontrado em SCRIPTS"


class TestObterLingua:
    def test_obter_lingua_valida(self):
        nome, nllb, tess, script, urange = obter_lingua("russo")
        assert nome == "Russo"
        assert nllb == "rus_Cyrl"
        assert "rus" in tess

    def test_obter_lingua_invalida(self):
        with pytest.raises(ValueError, match="não suportada"):
            obter_lingua("klingon")

    def test_obter_lingua_bengali(self):
        nome, nllb, tess, script, urange = obter_lingua("bengali")
        assert nome == "Bengali"
        assert nllb == "ben_Beng"
        assert urange is not None


class TestParesDirectos:
    def test_pares_directos_nao_vazio(self):
        assert len(PARES_DIRECTOS) > 5

    def test_pares_directos_existem_em_linguas(self):
        for chave in PARES_DIRECTOS:
            assert chave in LINGUAS, f"Par directo '{chave}' não existe em LINGUAS"

    def test_russo_e_par_directo(self):
        assert "russo" in PARES_DIRECTOS

    def test_espanhol_e_par_directo(self):
        assert "espanhol" in PARES_DIRECTOS
