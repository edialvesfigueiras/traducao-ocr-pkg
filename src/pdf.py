
from PIL import Image

try:
    import fitz as pymupdf
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


def extrair_paginas_pdf(caminho_pdf: str, dpi: int = 300) -> list[Image.Image]:
    if not HAS_PDF:
        raise RuntimeError("pymupdf não instalado. python -m pip install pymupdf")
    doc = pymupdf.open(caminho_pdf)
    imagens = []
    for pagina in doc:
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pagina.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imagens.append(img)
    doc.close()
    return imagens
