
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr import HAS_CV2, preprocessar_imagem


class TestPreprocessamento:
    def test_preprocessar_pil_image(self):
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        resultado = preprocessar_imagem(img)
        assert isinstance(resultado, Image.Image)

    def test_preprocessar_escala_imagem_pequena(self):
        from PIL import Image
        img = Image.new("RGB", (50, 50), color="white")
        resultado = preprocessar_imagem(img)
        assert resultado.size[0] >= 50 or resultado.size[1] >= 50

    def test_has_cv2_is_bool(self):
        assert isinstance(HAS_CV2, bool)
