
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np

from src.ocr import (
    preprocessar_imagem, preprocessar_imagem_pipeline,
    calcular_metricas_ocr, HAS_CV2,
)
from src.config import (
    OCR_CONFIG, OCR_CONFIG_DEFAULT, obter_ocr_config,
    SCRIPT_TO_TESSERACT, SCRIPT_OSD_TO_INTERNO,
    OCR_QUALIDADE_EXCELENTE, OCR_QUALIDADE_BOM, OCR_QUALIDADE_FRACO,
)


class TestOCRConfig:

    def test_config_latino_existe(self):
        assert "latino" in OCR_CONFIG

    def test_config_cirilico_existe(self):
        assert "cirilico" in OCR_CONFIG

    def test_config_arabe_existe(self):
        assert "arabe" in OCR_CONFIG

    def test_config_bengali_existe(self):
        assert "bengali" in OCR_CONFIG

    def test_config_cjk_existe(self):
        assert "cjk" in OCR_CONFIG

    def test_config_default_tem_psm(self):
        assert "psm" in OCR_CONFIG_DEFAULT
        assert isinstance(OCR_CONFIG_DEFAULT["psm"], list)

    def test_obter_config_existente(self):
        config = obter_ocr_config("latino")
        assert "psm" in config
        assert "threshold" in config
        assert "dilate" in config
        assert "invert" in config

    def test_obter_config_inexistente_retorna_default(self):
        config = obter_ocr_config("script_inventado")
        assert config == OCR_CONFIG_DEFAULT

    def test_arabe_sem_dilate(self):
        config = obter_ocr_config("arabe")
        assert config["dilate"] is False

    def test_arabe_tem_invert(self):
        config = obter_ocr_config("arabe")
        assert config["invert"] is True

    def test_latino_sem_dilate(self):
        config = obter_ocr_config("latino")
        assert config["dilate"] is False

    def test_todos_configs_tem_campos(self):
        for script, config in OCR_CONFIG.items():
            assert "psm" in config, f"{script} sem psm"
            assert "threshold" in config, f"{script} sem threshold"
            assert "dilate" in config, f"{script} sem dilate"
            assert "invert" in config, f"{script} sem invert"


class TestScriptMapping:

    def test_cyrillic_mapping(self):
        assert "Cyrillic" in SCRIPT_TO_TESSERACT
        langs = SCRIPT_TO_TESSERACT["Cyrillic"]
        assert "rus+eng" in langs

    def test_arabic_mapping(self):
        assert "Arabic" in SCRIPT_TO_TESSERACT
        langs = SCRIPT_TO_TESSERACT["Arabic"]
        assert "ara+eng" in langs

    def test_bengali_mapping(self):
        assert "Bengali" in SCRIPT_TO_TESSERACT
        langs = SCRIPT_TO_TESSERACT["Bengali"]
        assert "ben+eng" in langs

    def test_latin_mapping(self):
        assert "Latin" in SCRIPT_TO_TESSERACT
        langs = SCRIPT_TO_TESSERACT["Latin"]
        assert "eng" in langs

    def test_osd_to_interno_cyrillic(self):
        assert SCRIPT_OSD_TO_INTERNO["Cyrillic"] == "cirilico"

    def test_osd_to_interno_bengali(self):
        assert SCRIPT_OSD_TO_INTERNO["Bengali"] == "bengali"

    def test_osd_to_interno_latin(self):
        assert SCRIPT_OSD_TO_INTERNO["Latin"] == "latino"

    def test_osd_to_interno_han(self):
        assert SCRIPT_OSD_TO_INTERNO["Han"] == "cjk"

    def test_todos_osd_tem_interno(self):
        for osd_name in SCRIPT_TO_TESSERACT:
            assert osd_name in SCRIPT_OSD_TO_INTERNO, \
                f"Script OSD '{osd_name}' sem mapeamento interno"


class TestQualidadeThresholds:
    def test_excelente_maior_que_bom(self):
        assert OCR_QUALIDADE_EXCELENTE > OCR_QUALIDADE_BOM

    def test_bom_maior_que_fraco(self):
        assert OCR_QUALIDADE_BOM > OCR_QUALIDADE_FRACO

    def test_fraco_positivo(self):
        assert OCR_QUALIDADE_FRACO > 0


class TestPreprocessamento:

    def test_preprocessar_imagem_pil(self):
        img = Image.new("RGB", (200, 200), color="white")
        resultado = preprocessar_imagem(img)
        assert isinstance(resultado, Image.Image)

    def test_preprocessar_imagem_pequena_escala(self):
        img = Image.new("RGB", (50, 50), color="white")
        resultado = preprocessar_imagem(img)
        assert resultado.size[0] >= 50

    def test_pipeline_retorna_estagios(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        img = Image.new("RGB", (200, 200), color="white")
        resultado, estagios = preprocessar_imagem_pipeline(img)
        assert isinstance(resultado, Image.Image)
        assert isinstance(estagios, list)

    def test_pipeline_estagios_sao_tuplas(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        img = Image.new("RGB", (200, 200), color="white")
        _, estagios = preprocessar_imagem_pipeline(img)
        for item in estagios:
            assert isinstance(item, tuple)
            assert len(item) == 2
            nome, img_estagio = item
            assert isinstance(nome, str)
            assert isinstance(img_estagio, Image.Image)

    def test_pipeline_com_script_latino(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        img = Image.new("RGB", (200, 200), color="white")
        resultado, _ = preprocessar_imagem_pipeline(img, script="latino")
        assert isinstance(resultado, Image.Image)

    def test_pipeline_com_script_bengali(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        img = Image.new("RGB", (200, 200), color="white")
        resultado, _ = preprocessar_imagem_pipeline(img, script="bengali")
        assert isinstance(resultado, Image.Image)

    def test_pipeline_com_script_arabe(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        img = Image.new("RGB", (200, 200), color="white")
        resultado, _ = preprocessar_imagem_pipeline(img, script="arabe")
        assert isinstance(resultado, Image.Image)

    def test_imagem_cinza_aceite(self):
        img = Image.new("L", (200, 200), color=128)
        resultado = preprocessar_imagem(img)
        assert isinstance(resultado, Image.Image)

    def test_imagem_com_texto_sintetico(self):
        if not HAS_CV2:
            pytest.skip("OpenCV não disponível")
        arr = np.ones((200, 400, 3), dtype=np.uint8) * 255
        arr[80:120, 50:350] = 0
        img = Image.fromarray(arr)
        resultado, estagios = preprocessar_imagem_pipeline(img)
        assert isinstance(resultado, Image.Image)
        assert len(estagios) > 0


class TestMetricasOCR:

    def test_metricas_retorna_dict(self):
        img = Image.new("RGB", (200, 50), color="white")
        try:
            metricas = calcular_metricas_ocr(img, lang="eng")
        except Exception:
            pytest.skip("Tesseract não disponível")

        assert isinstance(metricas, dict)
        assert "confianca_media" in metricas
        assert "pct_alta_confianca" in metricas
        assert "qualidade" in metricas
        assert "total_palavras" in metricas

    def test_metricas_confianca_entre_0_100(self):
        img = Image.new("RGB", (200, 50), color="white")
        try:
            metricas = calcular_metricas_ocr(img, lang="eng")
        except Exception:
            pytest.skip("Tesseract não disponível")

        assert 0 <= metricas["confianca_media"] <= 100
        assert 0 <= metricas["pct_alta_confianca"] <= 100

    def test_metricas_qualidade_string(self):
        img = Image.new("RGB", (200, 50), color="white")
        try:
            metricas = calcular_metricas_ocr(img, lang="eng")
        except Exception:
            pytest.skip("Tesseract não disponível")

        assert metricas["qualidade"] in ("Excelente", "Bom", "Fraco", "Muito fraco", "N/A")
