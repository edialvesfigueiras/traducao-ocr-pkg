
import os
import logging
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _log_name in [
    "transformers.modeling_utils",
    "transformers.generation.utils",
    "transformers.tokenization_utils_base",
    "huggingface_hub.file_download",
    "huggingface_hub._commit_api",
    "huggingface_hub.utils._auth",
]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
CACHE_DIR = Path.home() / ".cache" / "ocr_traduzir"
CACHE_FILE = CACHE_DIR / "traducoes_cache.json"
LOGO_PATH = SCRIPT_DIR / "logopj.png"

MODELOS_PREFERIDOS = [
    "facebook/nllb-200-1.3B",
    "facebook/nllb-200-distilled-600M",
]

PIVOT_LANG = "eng_Latn"
TGT_LANG = "por_Latn"

NUM_BEAMS_DEFAULT = 4
MAX_TOKENS_DEFAULT = 512
BATCH_SIZE_DEFAULT = 8

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

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

    "espanhol":     ("Espanhol",      "spa_Latn", "spa+eng", "latino",     None),
    "frances":      ("Francês",       "fra_Latn", "fra+eng", "latino",     None),
    "italiano":     ("Italiano",      "ita_Latn", "ita+eng", "latino",     None),
    "alemao":       ("Alemão",        "deu_Latn", "deu+eng", "latino",     None),
    "romeno":       ("Romeno",        "ron_Latn", "ron+eng", "latino",     None),
    "polaco":       ("Polaco",        "pol_Latn", "pol+eng", "latino",     None),
    "checo":        ("Checo",         "ces_Latn", "ces+eng", "latino",     None),
    "holandes":     ("Holandês",      "nld_Latn", "nld+eng", "latino",     None),
}

PARES_DIRECTOS = {
    "espanhol", "frances", "italiano", "alemao", "romeno",
    "russo", "polaco", "checo", "holandes",
    "arabe", "turco", "hindi",
}

SCRIPTS = {}
for _chave, (_nome, _nllb, _tess, _script, _) in LINGUAS.items():
    SCRIPTS.setdefault(_script, []).append((_chave, _nome))

MODELOS_ESPECIALIZADOS = {
    "espanhol":   {"modelo": "Helsinki-NLP/opus-mt-es-pt", "directo_pt": True},
    "frances":    {"modelo": "Helsinki-NLP/opus-mt-fr-pt", "directo_pt": True},
    "italiano":   {"modelo": "Helsinki-NLP/opus-mt-it-pt", "directo_pt": True},

    "alemao":     {"modelo": "Helsinki-NLP/opus-mt-de-en", "directo_pt": False},
    "romeno":     {"modelo": "Helsinki-NLP/opus-mt-roa-en", "directo_pt": False},
    "polaco":     {"modelo": "Helsinki-NLP/opus-mt-pl-en", "directo_pt": False},
    "holandes":   {"modelo": "Helsinki-NLP/opus-mt-nl-en", "directo_pt": False},
    "checo":      {"modelo": "Helsinki-NLP/opus-mt-cs-en", "directo_pt": False},
    "russo":      {"modelo": "Helsinki-NLP/opus-mt-ru-en", "directo_pt": False},
    "ucraniano":  {"modelo": "Helsinki-NLP/opus-mt-uk-en", "directo_pt": False},
    "bulgaro":    {"modelo": "Helsinki-NLP/opus-mt-bg-en", "directo_pt": False},
    "arabe":      {"modelo": "Helsinki-NLP/opus-mt-ar-en", "directo_pt": False},
    "chines_s":   {"modelo": "Helsinki-NLP/opus-mt-zh-en", "directo_pt": False},
    "chines_t":   {"modelo": "Helsinki-NLP/opus-mt-zh-en", "directo_pt": False},
    "japones":    {"modelo": "Helsinki-NLP/opus-mt-jap-en", "directo_pt": False},
    "coreano":    {"modelo": "Helsinki-NLP/opus-mt-ko-en", "directo_pt": False},
    "hindi":      {"modelo": "Helsinki-NLP/opus-mt-hi-en", "directo_pt": False},
    "turco":      {"modelo": "Helsinki-NLP/opus-mt-tr-en", "directo_pt": False},
    "hebraico":   {"modelo": "Helsinki-NLP/opus-mt-he-en", "directo_pt": False},
}

MODELO_PIVOT_EN_PT = "Helsinki-NLP/opus-mt-en-pt"


OCR_CONFIG = {
    "cirilico":    {"psm": [6, 3],     "threshold": 140, "dilate": False, "invert": False},
    "arabe":       {"psm": [6, 4, 3],  "threshold": 150, "dilate": False, "invert": True},
    "devanagari":  {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "bengali":     {"psm": [6, 3],     "threshold": 140, "dilate": True,  "invert": False},
    "tamil":       {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "telugu":      {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "kannada":     {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "malaiala":    {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "gujarati":    {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "gurmukhi":    {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "sinhala":     {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "cjk":         {"psm": [6, 3],     "threshold": 128, "dilate": True,  "invert": False},
    "japones":     {"psm": [6, 3],     "threshold": 128, "dilate": True,  "invert": False},
    "hangul":      {"psm": [6, 3],     "threshold": 130, "dilate": True,  "invert": False},
    "thai":        {"psm": [6, 3],     "threshold": 135, "dilate": False, "invert": False},
    "myanmar":     {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "khmer":       {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "lao":         {"psm": [6, 3],     "threshold": 135, "dilate": False, "invert": False},
    "grego":       {"psm": [6, 3],     "threshold": 140, "dilate": False, "invert": False},
    "hebraico":    {"psm": [6, 4],     "threshold": 135, "dilate": False, "invert": True},
    "georgiano":   {"psm": [6, 3],     "threshold": 140, "dilate": False, "invert": False},
    "armenio":     {"psm": [6, 3],     "threshold": 140, "dilate": False, "invert": False},
    "etiope":      {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "tibetano":    {"psm": [6, 3],     "threshold": 135, "dilate": True,  "invert": False},
    "latino":      {"psm": [6, 3],     "threshold": 140, "dilate": False, "invert": False},
}

OCR_CONFIG_DEFAULT = {"psm": [6, 3], "threshold": 140, "dilate": False, "invert": False}


def obter_ocr_config(script: str) -> dict:
    return OCR_CONFIG.get(script, OCR_CONFIG_DEFAULT)


SCRIPT_TO_TESSERACT = {
    "Cyrillic":    ["rus+eng", "ukr+eng", "bul+eng", "srp+eng"],
    "Arabic":      ["ara+eng", "fas+eng", "urd+eng"],
    "Devanagari":  ["hin+eng", "mar+eng", "nep+eng"],
    "Bengali":     ["ben+eng"],
    "Han":         ["chi_sim+eng", "chi_tra+eng"],
    "Hangul":      ["kor+eng"],
    "Japanese":    ["jpn+eng"],
    "Greek":       ["ell+eng"],
    "Hebrew":      ["heb+eng"],
    "Thai":        ["tha+eng"],
    "Georgian":    ["kat+eng"],
    "Armenian":    ["hye+eng"],
    "Tamil":       ["tam+eng"],
    "Telugu":      ["tel+eng"],
    "Kannada":     ["kan+eng"],
    "Malayalam":   ["mal+eng"],
    "Gujarati":    ["guj+eng"],
    "Gurmukhi":    ["pan+eng"],
    "Sinhala":     ["sin+eng"],
    "Myanmar":     ["mya+eng"],
    "Khmer":       ["khm+eng"],
    "Lao":         ["lao+eng"],
    "Tibetan":     ["bod+eng"],
    "Ethiopic":    ["amh+eng"],
    "Latin":       ["eng", "tur+eng", "vie+eng", "ind+eng", "spa+eng", "fra+eng",
                    "deu+eng", "ita+eng", "ron+eng", "pol+eng", "ces+eng", "nld+eng"],
}

SCRIPT_OSD_TO_INTERNO = {
    "Cyrillic": "cirilico", "Arabic": "arabe", "Devanagari": "devanagari",
    "Bengali": "bengali", "Han": "cjk", "Hangul": "hangul",
    "Japanese": "japones", "Greek": "grego", "Hebrew": "hebraico",
    "Thai": "thai", "Georgian": "georgiano", "Armenian": "armenio",
    "Tamil": "tamil", "Telugu": "telugu", "Kannada": "kannada",
    "Malayalam": "malaiala", "Gujarati": "gujarati", "Gurmukhi": "gurmukhi",
    "Sinhala": "sinhala", "Myanmar": "myanmar", "Khmer": "khmer",
    "Lao": "lao", "Tibetan": "tibetano", "Ethiopic": "etiope",
    "Latin": "latino",
}

OCR_QUALIDADE_EXCELENTE = 80
OCR_QUALIDADE_BOM = 60
OCR_QUALIDADE_FRACO = 40


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
