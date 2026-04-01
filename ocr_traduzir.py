
import os
import re
import sys
import glob
import json
import hashlib
import logging
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _log in [
    "transformers.modeling_utils",
    "transformers.generation.utils",
    "transformers.tokenization_utils_base",
    "huggingface_hub.file_download",
    "huggingface_hub._commit_api",
    "huggingface_hub.utils._auth",
]:
    logging.getLogger(_log).setLevel(logging.ERROR)

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import fitz as pymupdf
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LINGUAS = {
    "bengali":      ("Bengali",       "ben_Beng", "ben+eng", "bengali",    r"\u0980-\u09FF"),
    "hindi":        ("Hindi",         "hin_Deva", "hin+eng", "devanagari", r"\u0900-\u097F"),
    "urdu":         ("Urdu",          "urd_Arab", "urd+eng", "arabe",      r"\u0600-\u06FF"),
    "tamil":        ("Tâmil",         "tam_Taml", "tam+eng", "tamil",      r"\u0B80-\u0BFF"),
    "telugu":       ("Telugu",        "tel_Telu", "tel+eng", "telugu",     r"\u0C00-\u0C7F"),
    "kannada":      ("Canarim",       "kan_Knda", "kan+eng", "kannada",    r"\u0C80-\u0CFF"),
    "malaiala":     ("Malaiala",      "mal_Mlym", "mal+eng", "malaiala",   r"\u0D00-\u0D7F"),
    "gujarati":     ("Guzerate",      "guj_Gujr", "guj+eng", "gujarati",  r"\u0A80-\u0AFF"),
    "punjabi":      ("Punjabi",       "pan_Guru", "pan+eng", "gurmukhi",   r"\u0A00-\u0A7F"),
    "nepali":       ("Nepalês",       "npi_Deva", "nep+eng", "devanagari", r"\u0900-\u097F"),
    "singales":     ("Cingalês",      "sin_Sinh", "sin+eng", "sinhala",    r"\u0D80-\u0DFF"),
    "marathi":      ("Marata",        "mar_Deva", "mar+eng", "devanagari", r"\u0900-\u097F"),

    "russo":        ("Russo",         "rus_Cyrl", "rus+eng", "cirilico",   r"\u0400-\u04FF"),
    "ucraniano":    ("Ucraniano",     "ukr_Cyrl", "ukr+eng", "cirilico",   r"\u0400-\u04FF"),
    "bulgaro":      ("Búlgaro",       "bul_Cyrl", "bul+eng", "cirilico",   r"\u0400-\u04FF"),
    "serbio":       ("Sérvio",        "srp_Cyrl", "srp+eng", "cirilico",   r"\u0400-\u04FF"),
    "macedonio":    ("Macedónio",     "mkd_Cyrl", "mkd+eng", "cirilico",   r"\u0400-\u04FF"),
    "bielorrusso":  ("Bielorrusso",   "bel_Cyrl", "bel+eng", "cirilico",   r"\u0400-\u04FF"),
    "cazaque":      ("Cazaque",       "kaz_Cyrl", "kaz+eng", "cirilico",   r"\u0400-\u04FF"),

    "arabe":        ("Árabe",         "arb_Arab", "ara+eng", "arabe",      r"\u0600-\u06FF"),
    "persa":        ("Persa/Farsi",   "pes_Arab", "fas+eng", "arabe",      r"\u0600-\u06FF"),
    "pashto":       ("Pashto",        "pbt_Arab", "pus+eng", "arabe",      r"\u0600-\u06FF"),

    "chines_s":     ("Chinês Simpl.", "zho_Hans", "chi_sim+eng", "cjk",    r"\u4E00-\u9FFF"),
    "chines_t":     ("Chinês Trad.",  "zho_Hant", "chi_tra+eng", "cjk",    r"\u4E00-\u9FFF"),
    "japones":      ("Japonês",       "jpn_Jpan", "jpn+eng", "japones",    r"\u3040-\u309F\u30A0-\u30FF"),
    "coreano":      ("Coreano",       "kor_Hang", "kor+eng", "hangul",     r"\uAC00-\uD7AF"),

    "tailandes":    ("Tailandês",     "tha_Thai", "tha+eng", "thai",       r"\u0E00-\u0E7F"),
    "vietnamita":   ("Vietnamita",    "vie_Latn", "vie+eng", "latino",     None),
    "birman":       ("Birmanês",      "mya_Mymr", "mya+eng", "myanmar",   r"\u1000-\u109F"),
    "khmer":        ("Khmer",         "khm_Khmr", "khm+eng", "khmer",     r"\u1780-\u17FF"),
    "lao":          ("Laociano",      "lao_Laoo", "lao+eng", "lao",       r"\u0E80-\u0EFF"),

    "grego":        ("Grego",         "ell_Grek", "ell+eng", "grego",      r"\u0370-\u03FF"),
    "hebraico":     ("Hebraico",      "heb_Hebr", "heb+eng", "hebraico",   r"\u0590-\u05FF"),
    "georgiano":    ("Georgiano",     "kat_Geor", "kat+eng", "georgiano",  r"\u10A0-\u10FF"),
    "armenio":      ("Arménio",       "hye_Armn", "hye+eng", "armenio",    r"\u0530-\u058F"),
    "etiope":       ("Amárico",       "amh_Ethi", "amh+eng", "etiope",     r"\u1200-\u137F"),
    "tibetano":     ("Tibetano",      "bod_Tibt", "bod+eng", "tibetano",   r"\u0F00-\u0FFF"),

    "turco":        ("Turco",         "tur_Latn", "tur+eng", "latino",     None),
    "indonesio":    ("Indonésio",     "ind_Latn", "ind+eng", "latino",     None),
    "malaio":       ("Malaio",        "zsm_Latn", "msa+eng", "latino",     None),
    "swahili":      ("Suaíli",        "swh_Latn", "swa+eng", "latino",     None),
}

SCRIPTS = {}
for chave, (nome, nllb, tess, script, _) in LINGUAS.items():
    SCRIPTS.setdefault(script, []).append((chave, nome))


def listar_linguas():
    print(f"{'Chave':<16} {'Língua':<20} {'NLLB':<12} {'Tesseract':<14} {'Script'}")
    print("-" * 78)
    for chave, (nome, nllb, tess, script, _) in sorted(LINGUAS.items()):
        print(f"{chave:<16} {nome:<20} {nllb:<12} {tess:<14} {script}")


def obter_lingua(chave: str) -> tuple:
    if chave not in LINGUAS:
        raise ValueError(
            f"Língua '{chave}' não suportada. "
            f"Use --listar-linguas para ver opções."
        )
    return LINGUAS[chave]


def detectar_lingua(texto: str) -> str | None:
    contagens = {}
    for chave, (_, _, _, script, urange) in LINGUAS.items():
        if urange is None:
            continue
        pattern = f"[{urange}]"
        n = len(re.findall(pattern, texto))
        if n > 0:
            contagens[chave] = n

    if not contagens:
        return None

    melhor = max(contagens, key=contagens.get)
    return melhor


TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

MODELOS_PREFERIDOS = [
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-distilled-600M",
]

PIVOT_LANG = "eng_Latn"
TGT_LANG   = "por_Latn"

_cache_traducoes: dict[str, str] = {}
CACHE_DIR = Path.home() / ".cache" / "ocr_traduzir"
CACHE_FILE = CACHE_DIR / "traducoes_cache.json"


def _carregar_cache():
    global _cache_traducoes
    if CACHE_FILE.exists():
        try:
            _cache_traducoes = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache_traducoes = {}


def _guardar_cache():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if len(_cache_traducoes) > 2000:
            chaves = list(_cache_traducoes.keys())
            for c in chaves[:len(chaves) - 2000]:
                del _cache_traducoes[c]
        CACHE_FILE.write_text(
            json.dumps(_cache_traducoes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _hash_texto(texto: str, lingua: str = "") -> str:
    return hashlib.md5(f"{lingua}:{texto}".encode("utf-8")).hexdigest()


def _preprocessar_cv2(img_array: np.ndarray) -> np.ndarray:
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    h, w = gray.shape[:2]
    if max(h, w) < 1500:
        scale = max(2.0, 1500 / max(h, w))
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                         searchWindowSize=21)

    coords = np.column_stack(np.where(denoised < 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if 0.5 < abs(angle) < 15:
            center = (denoised.shape[1] // 2, denoised.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(
                denoised, M, (denoised.shape[1], denoised.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=15)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


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


def preprocessar_imagem(caminho_ou_img) -> Image.Image:
    if isinstance(caminho_ou_img, Image.Image):
        img = caminho_ou_img.convert("RGB")
    else:
        img = Image.open(caminho_ou_img).convert("RGB")

    if HAS_CV2:
        return Image.fromarray(_preprocessar_cv2(np.array(img)))
    else:
        return _preprocessar_pil(img)


def _contar_script(texto: str, unicode_range: str | None) -> int:
    if unicode_range is None:
        return len(texto)
    return len(re.findall(f"[{unicode_range}]", texto))


def extrair_texto(caminho_ou_img, lang: str = "ben+eng") -> str:
    if isinstance(caminho_ou_img, Image.Image):
        img_original = caminho_ou_img
    else:
        img_original = Image.open(caminho_ou_img)

    configs = [
        "--oem 3 --psm 6 -c preserve_interword_spaces=1",
        "--oem 3 --psm 3 -c preserve_interword_spaces=1",
        "--oem 3 --psm 4 -c preserve_interword_spaces=1",
    ]

    melhor_texto = ""
    melhor_score = 0

    for config in configs:
        for img_input in [img_original]:
            try:
                img_proc = preprocessar_imagem(img_input.copy())
                texto_proc = pytesseract.image_to_string(
                    img_proc, lang=lang, config=config).strip()
                score = len(texto_proc)
                if score > melhor_score:
                    melhor_texto = texto_proc
                    melhor_score = score
            except Exception:
                pass

            try:
                texto_orig = pytesseract.image_to_string(
                    img_original, lang=lang, config=config).strip()
                score = len(texto_orig)
                if score > melhor_score:
                    melhor_texto = texto_orig
                    melhor_score = score
            except Exception:
                pass

    return melhor_texto


def extrair_paginas_pdf(caminho_pdf: str, dpi: int = 300) -> list[Image.Image]:
    if not HAS_PDF:
        raise RuntimeError("pymupdf não instalado. pip install pymupdf")
    doc = pymupdf.open(caminho_pdf)
    imagens = []
    for pagina in doc:
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = pagina.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        imagens.append(img)
    doc.close()
    return imagens


def _linha_e_lixo(linha: str) -> bool:
    l = linha.strip()
    if not l:
        return False
    if any(x in l.lower() for x in [
        "http", "youtube", ".com", ".org", ".net", "www.",
        "t.me", "t.co", "fb.com", "instagram", "twitter",
        "facebook", "whatsapp", "tiktok", "telegram",
    ]):
        return True
    if re.fullmatch(r"[a-zA-Z0-9\s_.@#/\-]+", l) and len(l) < 40:
        return True
    if re.fullmatch(r"[\s\W_]+", l):
        return True
    if "_" in l and " " not in l:
        return True
    return False


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[\u200B-\u200F\u202A-\u202E\uFEFF\u00AD]", "", texto)
    texto = re.sub(r"\t", " ", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    texto = texto.replace("|", "।")
    texto = re.sub(r"\s+([।,;:!?])", r"\1", texto)
    return texto.strip()


def limpar_texto_ocr(texto: str) -> str:
    linhas = []
    for linha in texto.split("\n"):
        if _linha_e_lixo(linha):
            if not linha.strip():
                linhas.append("")
        else:
            linhas.append(linha)

    resultado = []
    vazia_anterior = False
    for l in linhas:
        if not l.strip():
            if not vazia_anterior:
                resultado.append("")
            vazia_anterior = True
        else:
            resultado.append(l)
            vazia_anterior = False

    return normalizar_texto("\n".join(resultado).strip())


_model = None
_tokenizers: dict[str, object] = {}
_lang_ids: dict[str, int] = {}
_modelo_usado = None


def _modelo_em_cache(nome: str) -> bool:
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    modelo_dir = cache_base / f"models--{nome.replace('/', '--')}"
    if modelo_dir.exists():
        snapshots = modelo_dir / "snapshots"
        if snapshots.exists() and any(snapshots.iterdir()):
            for snap in snapshots.iterdir():
                model_files = list(snap.glob("*.bin")) + list(snap.glob("*.safetensors"))
                if model_files:
                    total = sum(f.stat().st_size for f in model_files)
                    if total > 500_000_000:
                        return True
    return False


def _obter_tokenizer(lang_code: str):
    global _tokenizers, _lang_ids
    if lang_code in _tokenizers:
        return _tokenizers[lang_code], _lang_ids[lang_code]

    _carregar_modelo()
    tok = AutoTokenizer.from_pretrained(_modelo_usado, src_lang=lang_code)
    _tokenizers[lang_code] = tok

    for tgt in [PIVOT_LANG, TGT_LANG]:
        if tgt not in _lang_ids:
            lid = tok.convert_tokens_to_ids(tgt)
            if lid != tok.unk_token_id:
                _lang_ids[tgt] = lid

    return tok, _lang_ids.get(PIVOT_LANG), _lang_ids.get(TGT_LANG)


def _carregar_modelo():
    global _model, _modelo_usado

    if _model is not None:
        return

    modelo_a_usar = None
    for nome in MODELOS_PREFERIDOS:
        if _modelo_em_cache(nome):
            modelo_a_usar = nome
            break

    if modelo_a_usar is None:
        modelo_a_usar = MODELOS_PREFERIDOS[-1]

    for tentativa in [modelo_a_usar] + MODELOS_PREFERIDOS[::-1]:
        try:
            _modelo_usado = tentativa
            print(f"A carregar {tentativa} ({DEVICE})... ", end="", flush=True)

            _model = AutoModelForSeq2SeqLM.from_pretrained(tentativa)
            _model.generation_config.max_length = None
            _model.eval()
            _model.to(DEVICE)
            print("OK")
            return
        except Exception as e:
            print(f"FALHOU ({e})")
            _model = None
            if tentativa != MODELOS_PREFERIDOS[-1]:
                print("  A tentar alternativa...", flush=True)

    raise RuntimeError("Não foi possível carregar nenhum modelo NLLB.")


def _gerar(tokenizer, inputs, forced_bos_id: int,
           max_tokens: int = 512, beams: int = 5) -> str:
    with torch.no_grad():
        output_ids = _model.generate(
            **{k: v.to(DEVICE) for k, v in inputs.items()},
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=max_tokens,
            num_beams=beams,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def traduzir_para_eng(texto: str, src_lang: str) -> str:
    _carregar_modelo()
    tok_src, pivot_id, _ = _obter_tokenizer(src_lang)
    inputs = tok_src(texto, return_tensors="pt", truncation=True, max_length=512)
    return _gerar(tok_src, inputs, pivot_id)


def traduzir_eng_para_por(texto: str) -> str:
    _carregar_modelo()
    tok_eng, _, target_id = _obter_tokenizer(PIVOT_LANG)
    inputs = tok_eng(texto, return_tensors="pt", truncation=True, max_length=512)
    return _gerar(tok_eng, inputs, target_id)


def traduzir(texto: str, lingua: str = "bengali",
             verbose: bool = False) -> str:
    _, nllb_code, _, _, _ = obter_lingua(lingua)

    chave = _hash_texto(texto, lingua)
    if chave in _cache_traducoes:
        if verbose:
            print(f"  [cache]")
        return _cache_traducoes[chave]

    texto_en = traduzir_para_eng(texto, nllb_code)
    if verbose:
        print(f"  [EN] {texto_en}")

    texto_pt = traduzir_eng_para_por(texto_en)
    resultado = pos_processar_pt(texto_pt)

    _cache_traducoes[chave] = resultado
    return resultado


_SUBSTITUICOES_LEXICAIS = [
    (r"\bVocê\b", "O senhor"), (r"\bvocê\b", "o senhor"),
    (r"\bVocês\b", "Os senhores"), (r"\bvocês\b", "os senhores"),
    (r"\bmenino\b", "rapaz"), (r"\bmenina\b", "rapariga"),
    (r"\bmeninos\b", "rapazes"), (r"\bmeninas\b", "raparigas"),
    (r"\bgaroto\b", "rapaz"), (r"\bgarota\b", "rapariga"),
    (r"\bgarotos\b", "rapazes"), (r"\bgarotas\b", "raparigas"),
    (r"\bmoleque\b", "miúdo"), (r"\bmoleques\b", "miúdos"),
    (r"\bcelular\b", "telemóvel"), (r"\bcelulares\b", "telemóveis"),
    (r"\bônibus\b", "autocarro"), (r"\btrem\b", "comboio"),
    (r"\btrens\b", "comboios"), (r"\bmetrô\b", "metro"),
    (r"\bbanheiro\b", "casa de banho"), (r"\bbanheiros\b", "casas de banho"),
    (r"\bchuveiro\b", "duche"), (r"\bgeladeira\b", "frigorífico"),
    (r"\bsorvete\b", "gelado"), (r"\bsorvetes\b", "gelados"),
    (r"\bsuco\b", "sumo"), (r"\bsucos\b", "sumos"),
    (r"\bxícara\b", "chávena"), (r"\bxícaras\b", "chávenas"),
    (r"\bcalçada\b", "passeio"), (r"\bvitrine\b", "montra"),
    (r"\bponto de ônibus\b", "paragem de autocarro"),
    (r"\bcarona\b", "boleia"), (r"\bcaronas\b", "boleias"),
    (r"\bacademia\b", "ginásio"), (r"\bnotebook\b", "portátil"),
    (r"\bcafé da manhã\b", "pequeno-almoço"),
    (r"\bCafé da manhã\b", "Pequeno-almoço"),
    (r"\bbiscoito\b", "bolacha"), (r"\bbiscoitos\b", "bolachas"),
    (r"\btênis\b", "sapatilhas"), (r"\bmoletom\b", "camisola"),
    (r"\bcontato\b", "contacto"), (r"\bcontatos\b", "contactos"),
    (r"\bfato\b(?!\s+de\s+banho)", "facto"), (r"\bfatos\b", "factos"),
    (r"\bseção\b", "secção"), (r"\bseções\b", "secções"),
    (r"\brecepção\b", "receção"), (r"\binfecção\b", "infeção"),
    (r"\ba gente\b", "nós"), (r"\brelacionamento\b", "relação"),
    (r"\brelacionamentos\b", "relações"),
    (r"\bpersonagem\b", "carácter"), (r"\bpersonagens\b", "caracteres"),
    (r"\bestúpida\b", "elegante"), (r"\bestúpido\b", "elegante"),
    (r"\bdieta\b", "vida"),
]

_SUBSTITUICOES_COMPILADAS = [
    (re.compile(p), r) for p, r in _SUBSTITUICOES_LEXICAIS
]

_REGEX_GERUNDIO = re.compile(
    r"\b(est(?:á|ão|ou|ava|avam|ive|iveram|eja|ejam))"
    r"\s+(\w+(?:ando|endo|indo))\b", re.IGNORECASE)

_REGEX_GERUNDIO_SOLTO = re.compile(
    r"\b((?:fico|fica|ficam|ficou|ficaram|ficava|ficavam"
    r"|ando|anda|andam|andou|andaram"
    r"|continuo|continua|continuam|continuou"
    r"|segue|seguem|seguiu|vai|vão|foi|foram))"
    r"\s+(\w+(?:ando|endo|indo))\b", re.IGNORECASE)


def _converter_gerundio(match: re.Match) -> str:
    prefixo = match.group(1)
    gerundio = match.group(2)
    if prefixo.lower().endswith(("ando", "endo", "indo")):
        return match.group(0)
    if gerundio.endswith("ando"):
        inf = gerundio[:-4] + "ar"
    elif gerundio.endswith("endo"):
        inf = gerundio[:-4] + "er"
    elif gerundio.endswith("indo"):
        inf = gerundio[:-4] + "ir"
    else:
        return match.group(0)
    return f"{prefixo} a {inf}"


def pos_processar_pt(texto: str) -> str:
    for p, r in _SUBSTITUICOES_COMPILADAS:
        texto = p.sub(r, texto)
    texto = _REGEX_GERUNDIO.sub(_converter_gerundio, texto)
    texto = _REGEX_GERUNDIO_SOLTO.sub(_converter_gerundio, texto)
    texto = re.sub(r"  +", " ", texto)
    texto = re.sub(r" ([.,;:!?])", r"\1", texto)
    texto = re.sub(r"(?<=[.!?]\s)([a-záàâãéêíóôõúç])",
                   lambda m: m.group(1).upper(), texto)
    if texto and texto[0].islower():
        texto = texto[0].upper() + texto[1:]
    return texto.strip()


def _dividir_em_paragrafos(texto: str) -> list[str]:
    paragrafos = re.split(r"\n\s*\n", texto)
    resultado = []
    for p in paragrafos:
        linhas = [l.strip() for l in p.split("\n") if l.strip()]
        if not linhas:
            continue
        bloco = " ".join(linhas)
        if len(bloco) < 3:
            continue
        resultado.append(bloco)
    return resultado


def traduzir_texto(texto: str, lingua: str = "bengali",
                   verbose: bool = False) -> str:
    texto_limpo = limpar_texto_ocr(texto)
    if verbose:
        print(f"\n- Texto limpo ({len(texto_limpo)} chars) -")
        print(texto_limpo)

    paragrafos = _dividir_em_paragrafos(texto_limpo)
    if not paragrafos:
        print("[aviso] Nenhum parágrafo encontrado.")
        return ""

    n = len(paragrafos)
    print(f"\n- Tradução ({n} parágrafos, língua: {lingua}) -")
    resultados = []

    for i, p in enumerate(paragrafos, 1):
        try:
            trad = traduzir(p, lingua=lingua, verbose=verbose)
            resultados.append(trad)
            print(f"[ok §{i}/{n}] {trad}")
        except Exception as e:
            print(f"[erro §{i}] {p[:40]!r} → {e}")

    _guardar_cache()
    return "\n\n".join(resultados)


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def processar_imagem(caminho_ou_img, lingua: str = "bengali",
                     lang_ocr: str | None = None,
                     verbose: bool = False, nome: str = "") -> str:
    _, _, tess_default, _, _ = obter_lingua(lingua)
    lang_ocr = lang_ocr or tess_default

    label = nome or (caminho_ou_img if isinstance(caminho_ou_img, str) else "imagem")
    print(f"\n{'='*60}\nFicheiro: {label}\nLíngua: {lingua}\n{'='*60}")

    print("A extrair texto (OCR)...")
    texto_ocr = extrair_texto(caminho_ou_img, lang=lang_ocr)

    if not texto_ocr.strip():
        print("  [aviso] OCR não extraiu texto.")
        return ""

    print(f"- Texto OCR ({len(texto_ocr)} chars) -")
    print(texto_ocr)

    texto_pt = traduzir_texto(texto_ocr, lingua=lingua, verbose=verbose)

    print(f"\n{'='*60}\nRESULTADO PT-PT\n{'='*60}")
    print(texto_pt)
    return texto_pt


def processar_pdf(caminho: str, lingua: str = "bengali",
                  lang_ocr: str | None = None,
                  verbose: bool = False) -> str:
    imagens = extrair_paginas_pdf(caminho)
    print(f"PDF: {caminho} — {len(imagens)} páginas")
    resultados = []
    for i, img in enumerate(imagens, 1):
        res = processar_imagem(
            img, lingua=lingua, lang_ocr=lang_ocr, verbose=verbose,
            nome=f"{caminho} — Página {i}/{len(imagens)}")
        if res:
            resultados.append(f"- Página {i} -\n{res}")
    return "\n\n".join(resultados)


def processar_pasta(pasta: str, lingua: str = "bengali",
                    lang_ocr: str | None = None,
                    verbose: bool = False) -> dict[str, str]:
    extensoes = EXTENSOES_IMAGEM | ({".pdf"} if HAS_PDF else set())
    ficheiros = sorted(
        f for f in glob.glob(os.path.join(pasta, "*"))
        if Path(f).suffix.lower() in extensoes
    )
    if not ficheiros:
        print(f"Nenhum ficheiro suportado em: {pasta}")
        return {}

    print(f"{len(ficheiros)} ficheiros em: {pasta}")
    resultados = {}
    for f in ficheiros:
        if f.lower().endswith(".pdf"):
            resultados[f] = processar_pdf(f, lingua, lang_ocr, verbose)
        else:
            resultados[f] = processar_imagem(f, lingua, lang_ocr, verbose)
    return resultados


def _imprimir_info():
    print(f"Dispositivo: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    print(f"OpenCV: {'sim' if HAS_CV2 else 'não'}")
    print(f"PDF: {'sim' if HAS_PDF else 'não'}")
    print(f"Línguas: {len(LINGUAS)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="OCR + Tradução multilingue offline → Português (PT-PT)")
    parser.add_argument("input", nargs="?", default=None)
    parser.add_argument("--lingua", "-l", default="bengali",
                        help="Língua de origem (default: bengali)")
    parser.add_argument("--lang-ocr", default=None,
                        help="Override Tesseract lang (ex: rus+eng)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--texto", action="store_true",
                        help="Input é ficheiro de texto")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--listar-linguas", action="store_true")
    parser.add_argument("--limpar-cache", action="store_true")
    args = parser.parse_args()

    if args.listar_linguas:
        listar_linguas()
        sys.exit(0)
    if args.info:
        _imprimir_info()
        sys.exit(0)
    if args.limpar_cache:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        print("Cache limpa.")
        sys.exit(0)
    if not args.input:
        parser.print_help()
        sys.exit(1)

    _carregar_cache()
    _imprimir_info()

    texto_final = ""

    if args.texto:
        texto = Path(args.input).read_text(encoding="utf-8")
        texto_final = traduzir_texto(texto, lingua=args.lingua,
                                     verbose=args.verbose)
    elif os.path.isdir(args.input):
        res = processar_pasta(args.input, args.lingua, args.lang_ocr,
                              args.verbose)
        texto_final = "\n\n".join(
            f"# {os.path.basename(f)}\n{t}" for f, t in res.items() if t)
    elif args.input.lower().endswith(".pdf"):
        texto_final = processar_pdf(args.input, args.lingua, args.lang_ocr,
                                    args.verbose)
    elif os.path.isfile(args.input):
        texto_final = processar_imagem(args.input, args.lingua, args.lang_ocr,
                                       args.verbose)
    else:
        print(f"Erro: '{args.input}' não encontrado.")
        sys.exit(1)

    if args.output and texto_final:
        Path(args.output).write_text(texto_final, encoding="utf-8")
        print(f"\nGuardado em: {args.output}")


if __name__ == "__main__":
    main()
