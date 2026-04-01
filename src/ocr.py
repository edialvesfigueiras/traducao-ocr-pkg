
import os
import re
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import easyocr as _easyocr_mod
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

from .config import (
    TESSERACT_CMD, OCR_CONFIG_DEFAULT, LINGUAS,
    SCRIPT_TO_TESSERACT, SCRIPT_OSD_TO_INTERNO,
    OCR_QUALIDADE_EXCELENTE, OCR_QUALIDADE_BOM, OCR_QUALIDADE_FRACO,
    obter_ocr_config,
)

log = logging.getLogger("ocr")

if pytesseract and os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def _estagio_super_resolucao(img: np.ndarray, verbose: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) >= 1000:
        return img
    scale = max(2.0, 1500 / max(h, w))
    scale = min(scale, 4.0)
    resultado = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    if verbose:
        log.info(f"[preprocessing] upscale: {scale:.1f}x ({w}x{h} -> {resultado.shape[1]}x{resultado.shape[0]})")
    return resultado


def _estagio_perspectiva(img: np.ndarray, verbose: bool = False) -> np.ndarray:
    gray = _to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    maior = max(contours, key=cv2.contourArea)
    area_img = gray.shape[0] * gray.shape[1]

    if cv2.contourArea(maior) < area_img * 0.3:
        return img

    peri = cv2.arcLength(maior, True)
    approx = cv2.approxPolyDP(maior, 0.02 * peri, True)

    if len(approx) != 4:
        return img

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    w_new = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    h_new = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))

    if w_new < 50 or h_new < 50:
        return img

    dst = np.array([[0, 0], [w_new, 0], [w_new, h_new], [0, h_new]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    resultado = cv2.warpPerspective(img, M, (w_new, h_new))

    if verbose:
        log.info(f"[preprocessing] perspectiva corrigida ({w_new}x{h_new})")
    return resultado


def _estagio_deskew(gray: np.ndarray, verbose: bool = False) -> np.ndarray:
    melhor_angulo = 0.0
    melhor_variancia = 0.0

    _, temp_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    for angulo_10 in range(-150, 151, 5):
        angulo = angulo_10 / 10.0
        h, w = temp_bin.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
        rotated = cv2.warpAffine(temp_bin, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        proj = np.sum(rotated, axis=1)
        variancia = np.var(proj)
        if variancia > melhor_variancia:
            melhor_variancia = variancia
            melhor_angulo = angulo

    if abs(melhor_angulo) < 0.3:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), melhor_angulo, 1.0)
    resultado = cv2.warpAffine(gray, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    if verbose:
        log.info(f"[preprocessing] deskew: {melhor_angulo:.1f} graus")
    return resultado


def _estagio_remover_sombras(img: np.ndarray, verbose: bool = False) -> np.ndarray:
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        resultado = clahe.apply(img)
        if verbose:
            log.info("[preprocessing] sombras removidas (grayscale CLAHE)")
        return resultado

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    resultado = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if verbose:
        log.info("[preprocessing] sombras removidas (LAB CLAHE)")
    return resultado


def _estagio_binarizacao(gray: np.ndarray, threshold: int = 140,
                          script: str = "latino",
                          verbose: bool = False) -> np.ndarray:
    scripts_diacriticos = {"arabe", "hebraico", "devanagari", "bengali",
                           "tamil", "telugu", "kannada", "malaiala",
                           "gujarati", "gurmukhi", "thai", "myanmar"}
    conservador = script in scripts_diacriticos

    if conservador:
        block_size = 21
        c_val = 8
    else:
        block_size = 31
        c_val = 15

    candidatos = {}

    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidatos["otsu"] = otsu
    except Exception:
        pass

    try:
        adaptivo = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=block_size, C=c_val)
        candidatos["adaptivo"] = adaptivo
    except Exception:
        pass

    try:
        from skimage.filters import threshold_sauvola
        window = 21 if conservador else 31
        k_val = 0.3 if conservador else 0.2
        thresh_map = threshold_sauvola(gray, window_size=window, k=k_val)
        sauvola = ((gray > thresh_map) * 255).astype(np.uint8)
        candidatos["sauvola"] = sauvola
    except ImportError:
        pass
    except Exception:
        pass

    if not candidatos:
        _, resultado = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return resultado

    melhor_nome = ""
    melhor_score = -1
    for nome, img in candidatos.items():
        score = np.sum(img < 128)
        ratio = score / img.size
        if ratio > 0.6:
            score = score * 0.3
        if score > melhor_score:
            melhor_score = score
            melhor_nome = nome

    if verbose:
        log.info(f"[preprocessing] binarizacao: {melhor_nome} (de {len(candidatos)} candidatos, "
                 f"{'conservador' if conservador else 'standard'})")
    return candidatos[melhor_nome]


def _estagio_recortar_margens(binary: np.ndarray, verbose: bool = False) -> np.ndarray:
    coords = cv2.findNonZero(255 - binary)
    if coords is None or len(coords) < 50:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    margin = 10
    y1 = max(0, y - margin)
    y2 = min(binary.shape[0], y + h + margin)
    x1 = max(0, x - margin)
    x2 = min(binary.shape[1], x + w + margin)
    resultado = binary[y1:y2, x1:x2]
    if verbose and (x > margin * 3 or y > margin * 3):
        log.info(f"[preprocessing] margens recortadas: ({x},{y}) {w}x{h}")
    return resultado


def _estagio_remover_ruido(binary: np.ndarray, script: str = "latino",
                            verbose: bool = False) -> np.ndarray:
    scripts_diacriticos = {"arabe", "hebraico", "devanagari", "bengali",
                           "tamil", "telugu", "thai", "myanmar"}
    if script in scripts_diacriticos:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        resultado = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if verbose:
            log.info("[preprocessing] ruido removido (suave, preserva diacriticos)")
    else:
        resultado = cv2.medianBlur(binary, 3)
        if verbose:
            log.info("[preprocessing] ruido salt-and-pepper removido")
    return resultado


def _estagio_dilatar(binary: np.ndarray, verbose: bool = False) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    invertida = cv2.bitwise_not(binary)
    dilatada = cv2.dilate(invertida, kernel, iterations=1)
    resultado = cv2.bitwise_not(dilatada)
    if verbose:
        log.info("[preprocessing] dilatacao aplicada (tracos finos)")
    return resultado


def preprocessar_imagem_pipeline(caminho_ou_img, script: str = "latino",
                                  verbose: bool = False) -> tuple[Image.Image, list[tuple[str, Image.Image]]]:
    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img.convert("RGB")
    else:
        img = Image.open(caminho_ou_img).convert("RGB")

    if not HAS_CV2:
        resultado = _preprocessar_pil(img)
        return resultado, [("original", img), ("final", resultado)]

    ocr_cfg = obter_ocr_config(script)
    arr = np.array(img)
    estagios = [("original", img)]

    arr = _estagio_super_resolucao(arr, verbose)
    estagios.append(("super-resolucao", Image.fromarray(arr)))

    arr = _estagio_perspectiva(arr, verbose)
    estagios.append(("perspectiva", Image.fromarray(arr)))

    gray = _to_gray(arr)

    gray = _estagio_remover_sombras(gray, verbose)
    estagios.append(("sombras", Image.fromarray(gray)))

    gray = _estagio_deskew(gray, verbose)
    estagios.append(("deskew", Image.fromarray(gray)))

    _scripts_diacriticos = {"arabe", "hebraico", "devanagari", "bengali",
                            "tamil", "telugu", "thai", "myanmar"}
    h_denoise = 5 if script in _scripts_diacriticos else 10
    gray = cv2.fastNlMeansDenoising(gray, h=h_denoise, templateWindowSize=7,
                                     searchWindowSize=21)

    binary = _estagio_binarizacao(gray, ocr_cfg.get("threshold", 140),
                                   script=script, verbose=verbose)
    estagios.append(("binarizacao", Image.fromarray(binary)))

    binary = _estagio_recortar_margens(binary, verbose)

    binary = _estagio_remover_ruido(binary, script=script, verbose=verbose)

    if ocr_cfg.get("dilate", False):
        binary = _estagio_dilatar(binary, verbose)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    resultado = Image.fromarray(binary)
    estagios.append(("final", resultado))
    return resultado, estagios


def preprocessar_imagem(caminho_ou_img, script: str = "latino") -> Image.Image:
    resultado, _ = preprocessar_imagem_pipeline(caminho_ou_img, script)
    return resultado


def _preprocessar_pil(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    w, h = gray.size
    if max(w, h) < 1500:
        scale = max(2, int(1500 / max(w, h)))
        gray = gray.resize((w * scale, h * scale), Image.LANCZOS)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    gray = gray.point(lambda x: 0 if x < 140 else 255, "1")
    return gray


def detectar_script_osd(caminho_ou_img) -> str | None:
    if pytesseract is None:
        return None

    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img
    else:
        img = Image.open(caminho_ou_img)

    try:
        osd = pytesseract.image_to_osd(img, config="--psm 0")
        for line in osd.split("\n"):
            if "Script:" in line:
                script = line.split(":")[-1].strip()
                if script and script != "NULL":
                    return script
    except Exception:
        pass
    return None


def _extrair_palavras_data(img, lang: str, config: str) -> list[dict]:
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config,
            output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    palavras = []
    for i in range(len(data["text"])):
        texto = data["text"][i].strip()
        conf = int(data["conf"][i])
        if texto and conf >= 0:
            palavras.append({
                "palavra": texto,
                "confianca": conf,
                "bloco": data["block_num"][i],
                "par": data["par_num"][i],
                "linha": data["line_num"][i],
                "palavra_num": data["word_num"][i],
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })
    return palavras


def _merge_resultados(todos_resultados: list[list[dict]],
                       min_confianca: int = 20) -> tuple[str, list[dict]]:
    melhor_por_posicao = {}

    for resultado in todos_resultados:
        for pal in resultado:
            cx = (pal["left"] + pal["width"] // 2) // 15
            cy = (pal["top"] + pal["height"] // 2) // 15
            key = (cy, cx)

            if key not in melhor_por_posicao or pal["confianca"] > melhor_por_posicao[key]["confianca"]:
                melhor_por_posicao[key] = pal

    palavras_ordenadas = sorted(
        melhor_por_posicao.values(),
        key=lambda p: (p["top"], p["left"]))

    palavras_filtradas = [p for p in palavras_ordenadas if p["confianca"] >= min_confianca]

    if not palavras_filtradas:
        return "", []

    linhas = {}
    for p in palavras_filtradas:
        line_key = p["top"] // 12
        linhas.setdefault(line_key, []).append(p)

    texto_linhas = []
    for key in sorted(linhas.keys()):
        palavras_linha = sorted(linhas[key], key=lambda p: p["left"])
        texto_linhas.append(" ".join(p["palavra"] for p in palavras_linha))

    texto = "\n".join(texto_linhas)
    return texto, palavras_filtradas


def extrair_texto(caminho_ou_img, lang: str = "ben+eng",
                  script: str = "latino", min_confianca: int = 20,
                  verbose: bool = False) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract não instalado.")

    if isinstance(caminho_ou_img, Image.Image):
        img_original = caminho_ou_img
    else:
        img_original = Image.open(caminho_ou_img)

    ocr_cfg = obter_ocr_config(script)
    psm_list = ocr_cfg.get("psm", [6, 3])
    testar_invertida = ocr_cfg.get("invert", False)

    img_proc = preprocessar_imagem(img_original.copy(), script)
    variantes = [
        ("preprocessada", img_proc),
        ("original", img_original),
    ]
    if testar_invertida:
        variantes.append(("invertida", ImageOps.invert(img_proc.convert("L"))))

    todos_resultados = []
    melhor_texto_simples = ""
    melhor_score_simples = 0

    _rtl_scripts = {"arabe", "hebraico", "persa", "urdu", "pashto"}
    is_rtl = script in _rtl_scripts

    for psm in psm_list:
        extra_cfg = " -c preserve_interword_spaces=1"
        if is_rtl:
            extra_cfg += " -c textord_old_baselines=0"
            extra_cfg += " -c textord_old_xheight=0"
        config = f"--oem 3 --psm {psm}{extra_cfg}"
        for nome_var, img_var in variantes:
            palavras = _extrair_palavras_data(img_var, lang, config)
            if palavras:
                todos_resultados.append(palavras)
                if verbose:
                    conf_media = sum(p["confianca"] for p in palavras) / len(palavras)
                    log.info(f"[ocr] PSM {psm} | {nome_var}: {len(palavras)} palavras, conf media {conf_media:.0f}%")

            try:
                texto = pytesseract.image_to_string(img_var, lang=lang, config=config).strip()
                if len(texto) > melhor_score_simples:
                    melhor_texto_simples = texto
                    melhor_score_simples = len(texto)
            except Exception:
                pass

    if todos_resultados:
        texto_votado, _ = _merge_resultados(todos_resultados, min_confianca)
        if len(texto_votado) >= len(melhor_texto_simples) * 0.7:
            return texto_votado

    return melhor_texto_simples


def extrair_texto_auto(caminho_ou_img, verbose: bool = False,
                        min_confianca: int = 20) -> tuple[str, str | None, float]:
    if pytesseract is None:
        raise RuntimeError("pytesseract não instalado.")

    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img
    else:
        img = Image.open(caminho_ou_img)

    script_osd = detectar_script_osd(img)
    if verbose and script_osd:
        log.info(f"[deteccao] Script visual (OSD): {script_osd}")

    if script_osd and script_osd in SCRIPT_TO_TESSERACT:
        candidatos = SCRIPT_TO_TESSERACT[script_osd]
        script_interno = SCRIPT_OSD_TO_INTERNO.get(script_osd, "latino")
    else:
        candidatos = ["eng"]
        script_interno = "latino"

    melhor_texto = ""
    melhor_conf = 0.0
    melhor_lang = candidatos[0] if candidatos else "eng"

    for lang in candidatos:
        try:
            texto = extrair_texto(img, lang=lang, script=script_interno,
                                   min_confianca=min_confianca, verbose=verbose)
            metricas = calcular_metricas_ocr(img, lang=lang, script=script_interno)
            conf = metricas.get("confianca_media", 0)

            if verbose:
                log.info(f"[deteccao] OCR {lang}: {len(texto)} chars, conf {conf:.0f}%")

            if conf > melhor_conf and len(texto) > 10:
                melhor_texto = texto
                melhor_conf = conf
                melhor_lang = lang
        except Exception:
            continue

    lingua_detectada = None
    if melhor_texto:
        from .utils import detectar_lingua_auto
        lingua_detectada, _ = detectar_lingua_auto(melhor_texto)

    if verbose:
        log.info(f"[deteccao] Resultado: lang_ocr={melhor_lang}, lingua={lingua_detectada}, conf={melhor_conf:.0f}%")

    return melhor_texto, lingua_detectada, melhor_conf


def calcular_metricas_ocr(caminho_ou_img, lang: str = "ben+eng",
                           script: str = "latino") -> dict:
    if pytesseract is None:
        return {"confianca_media": 0, "pct_alta_confianca": 0,
                "racio_texto_ruido": 0, "qualidade": "N/A", "total_palavras": 0}

    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img
    else:
        img = Image.open(caminho_ou_img)

    img_proc = preprocessar_imagem(img.copy(), script)

    try:
        data = pytesseract.image_to_data(
            img_proc, lang=lang,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT)
    except Exception:
        return {"confianca_media": 0, "pct_alta_confianca": 0,
                "racio_texto_ruido": 0, "qualidade": "N/A", "total_palavras": 0}

    confiancas = []
    texto_total = ""
    for i in range(len(data["text"])):
        texto = data["text"][i].strip()
        conf = int(data["conf"][i])
        if texto and conf >= 0:
            confiancas.append(conf)
            texto_total += texto

    if not confiancas:
        return {"confianca_media": 0, "pct_alta_confianca": 0,
                "racio_texto_ruido": 0, "qualidade": "Muito fraco", "total_palavras": 0}

    conf_media = sum(confiancas) / len(confiancas)
    pct_alta = sum(1 for c in confiancas if c > 60) / len(confiancas) * 100

    urange = None
    for _, (_, _, _, s, ur) in LINGUAS.items():
        if s == script and ur:
            urange = ur
            break

    racio = 100.0
    if urange and texto_total:
        chars_script = len(re.findall(f"[{urange}]", texto_total))
        racio = (chars_script / len(texto_total) * 100) if texto_total else 0

    if conf_media >= OCR_QUALIDADE_EXCELENTE:
        qualidade = "Excelente"
    elif conf_media >= OCR_QUALIDADE_BOM:
        qualidade = "Bom"
    elif conf_media >= OCR_QUALIDADE_FRACO:
        qualidade = "Fraco"
    else:
        qualidade = "Muito fraco"

    return {
        "confianca_media": round(conf_media, 1),
        "pct_alta_confianca": round(pct_alta, 1),
        "racio_texto_ruido": round(racio, 1),
        "qualidade": qualidade,
        "total_palavras": len(confiancas),
    }


def extrair_confianca(caminho_ou_img, lang: str = "ben+eng",
                       script: str = "latino") -> list[dict]:
    if pytesseract is None:
        raise RuntimeError("pytesseract não instalado.")

    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img
    else:
        img = Image.open(caminho_ou_img)

    img_proc = preprocessar_imagem(img.copy(), script)
    data = pytesseract.image_to_data(
        img_proc, lang=lang,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT)

    palavras = []
    for i in range(len(data["text"])):
        texto = data["text"][i].strip()
        if texto:
            palavras.append({
                "palavra": texto,
                "confianca": int(data["conf"][i]),
                "bloco": data["block_num"][i],
                "linha": data["line_num"][i],
            })
    return palavras
