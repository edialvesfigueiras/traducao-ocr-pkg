
import re
import logging
from pathlib import Path

from .config import LINGUAS, CACHE_DIR

log = logging.getLogger(__name__)

_ISO_PARA_CHAVE = {
    "bn": "bengali", "hi": "hindi", "ur": "urdu", "ta": "tamil",
    "te": "telugu", "kn": "kannada", "ml": "malaiala", "gu": "gujarati",
    "pa": "punjabi", "ne": "nepali", "si": "singales", "mr": "marathi",
    "ru": "russo", "uk": "ucraniano", "bg": "bulgaro", "sr": "serbio",
    "mk": "macedonio", "be": "bielorrusso", "kk": "cazaque",
    "ar": "arabe", "fa": "persa", "ps": "pashto",
    "zh": "chines_s", "ja": "japones", "ko": "coreano",
    "th": "tailandes", "vi": "vietnamita", "my": "birman",
    "km": "khmer", "lo": "lao",
    "el": "grego", "he": "hebraico", "ka": "georgiano",
    "hy": "armenio", "am": "etiope", "bo": "tibetano",
    "tr": "turco", "id": "indonesio", "ms": "malaio", "sw": "swahili",
    "es": "espanhol", "fr": "frances", "it": "italiano",
    "de": "alemao", "ro": "romeno", "pl": "polaco",
    "cs": "checo", "nl": "holandes",
}

_UNICODE_RANGES = [
    ("bengali",     r"[\u0980-\u09FF]",           ["bengali"],                          0.95),
    ("devanagari",  r"[\u0900-\u097F]",           ["hindi", "marathi", "nepali"],       0.85),
    ("tamil",       r"[\u0B80-\u0BFF]",           ["tamil"],                            0.95),
    ("telugu",      r"[\u0C00-\u0C7F]",           ["telugu"],                           0.95),
    ("kannada",     r"[\u0C80-\u0CFF]",           ["kannada"],                          0.95),
    ("malaiala",    r"[\u0D00-\u0D7F]",           ["malaiala"],                         0.95),
    ("gujarati",    r"[\u0A80-\u0AFF]",           ["gujarati"],                         0.95),
    ("gurmukhi",    r"[\u0A00-\u0A7F]",           ["punjabi"],                          0.95),
    ("sinhala",     r"[\u0D80-\u0DFF]",           ["singales"],                         0.95),
    ("hangul",      r"[\uAC00-\uD7AF\u1100-\u11FF]", ["coreano"],                      0.95),
    ("thai",        r"[\u0E00-\u0E7F]",           ["tailandes"],                        0.95),
    ("lao",         r"[\u0E80-\u0EFF]",           ["lao"],                              0.95),
    ("myanmar",     r"[\u1000-\u109F]",           ["birman"],                           0.95),
    ("khmer",       r"[\u1780-\u17FF]",           ["khmer"],                            0.95),
    ("georgiano",   r"[\u10A0-\u10FF]",           ["georgiano"],                        0.95),
    ("armenio",     r"[\u0530-\u058F]",           ["armenio"],                          0.95),
    ("etiope",      r"[\u1200-\u137F]",           ["etiope"],                           0.95),
    ("tibetano",    r"[\u0F00-\u0FFF]",           ["tibetano"],                         0.95),
    ("grego",       r"[\u0370-\u03FF]",           ["grego"],                            0.90),
    ("hebraico",    r"[\u0590-\u05FF]",           ["hebraico"],                         0.90),

    ("cirilico",    r"[\u0400-\u04FF]",           ["russo", "ucraniano", "bulgaro",
                                                    "serbio", "macedonio", "bielorrusso",
                                                    "cazaque"],                          0.50),
    ("arabe",       r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]",
                                                  ["arabe", "persa", "urdu", "pashto"], 0.50),
    ("cjk",         r"[\u4E00-\u9FFF\u3400-\u4DBF]",
                                                  ["chines_s", "chines_t"],             0.60),
    ("japones",     r"[\u3040-\u309F\u30A0-\u30FF]",
                                                  ["japones"],                           0.90),

    ("latino",      r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žȀ-ȳ]",
                                                  ["espanhol", "frances", "italiano",
                                                   "alemao", "romeno", "polaco", "checo",
                                                   "holandes", "turco", "indonesio",
                                                   "malaio", "swahili", "vietnamita"],  0.15),
]

_UNICODE_COMPILADOS = [
    (nome, re.compile(pattern), linguas, peso)
    for nome, pattern, linguas, peso in _UNICODE_RANGES
]


def _detectar_unicode(texto: str) -> list[tuple[str, float]]:
    contagens_script = {}
    for nome, regex, linguas, peso in _UNICODE_COMPILADOS:
        n = len(regex.findall(texto))
        if n > 0:
            contagens_script[nome] = (n, linguas, peso)

    if not contagens_script:
        return []

    total_chars = sum(c for c, _, _ in contagens_script.values())
    if total_chars == 0:
        return []

    resultados = []
    for nome, (count, linguas, peso_base) in contagens_script.items():
        proporcao = count / total_chars
        score = proporcao * peso_base

        if len(linguas) == 1:
            resultados.append((linguas[0], score))
        else:
            for lingua in linguas:
                resultados.append((lingua, score / len(linguas)))

    agregado = {}
    for lingua, score in resultados:
        agregado[lingua] = agregado.get(lingua, 0) + score

    return sorted(agregado.items(), key=lambda x: x[1], reverse=True)


_fasttext_modelo = None
_fasttext_carregado = False

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_FASTTEXT_PATH = CACHE_DIR / "lid.176.ftz"


def _obter_fasttext_path() -> Path | None:
    if _FASTTEXT_PATH.exists():
        return _FASTTEXT_PATH
    return None


def _carregar_fasttext():
    global _fasttext_modelo, _fasttext_carregado
    if _fasttext_carregado:
        return _fasttext_modelo
    _fasttext_carregado = True

    model_path = _obter_fasttext_path()
    if model_path is None:
        log.debug("Modelo fasttext lid.176.ftz não encontrado em %s", _FASTTEXT_PATH)
        return None

    try:
        import fasttext
        _fasttext_modelo = fasttext.load_model(str(model_path))
        log.debug("Modelo fasttext carregado: %s", model_path)
    except ImportError:
        log.debug("Biblioteca fasttext não instalada")
    except Exception as e:
        log.warning("Erro ao carregar fasttext: %s", e)

    return _fasttext_modelo


def _detectar_fasttext(texto: str) -> tuple[str | None, float]:
    modelo = _carregar_fasttext()
    if modelo is None:
        return None, 0.0

    try:
        texto_limpo = texto.replace("\n", " ").strip()
        if len(texto_limpo) < 3:
            return None, 0.0

        predicoes = modelo.predict(texto_limpo, k=3)
        labels, probs = predicoes

        for label, prob in zip(labels, probs):
            iso = label.replace("__label__", "")
            if iso in _ISO_PARA_CHAVE:
                return _ISO_PARA_CHAVE[iso], float(prob)

        return None, 0.0
    except Exception as e:
        log.debug("Erro fasttext: %s", e)
        return None, 0.0


_langid_modulo = None
_langid_carregado = False


def _carregar_langid():
    global _langid_modulo, _langid_carregado
    if _langid_carregado:
        return _langid_modulo
    _langid_carregado = True
    try:
        import langid as _lid
        _langid_modulo = _lid
    except ImportError:
        log.debug("Biblioteca langid não instalada")
    return _langid_modulo


def _detectar_langid(texto: str) -> tuple[str | None, float]:
    lid = _carregar_langid()
    if lid is None:
        return None, 0.0

    try:
        iso, conf_raw = lid.classify(texto)
        conf_clamped = max(-1000.0, min(-10.0, conf_raw))
        conf = (conf_clamped + 1000.0) / 990.0

        if iso in _ISO_PARA_CHAVE:
            return _ISO_PARA_CHAVE[iso], conf
        return None, 0.0
    except Exception as e:
        log.debug("Erro langid: %s", e)
        return None, 0.0


def detectar_lingua_auto(texto: str) -> tuple[str | None, float]:
    if not texto or len(texto.strip()) < 5:
        return None, 0.0

    texto_limpo = texto.strip()

    unicode_candidatos = _detectar_unicode(texto_limpo)
    unicode_top = unicode_candidatos[0] if unicode_candidatos else (None, 0.0)

    if unicode_top[1] >= 0.90:
        log.debug("Unicode: %s (%.2f) — script exclusivo, alta confiança",
                  unicode_top[0], unicode_top[1])
        pass

    ft_lingua, ft_conf = _detectar_fasttext(texto_limpo)

    lid_lingua, lid_conf = _detectar_langid(texto_limpo)

    log.debug("Detecção — Unicode: %s (%.2f), fasttext: %s (%.2f), langid: %s (%.2f)",
              unicode_top[0], unicode_top[1],
              ft_lingua, ft_conf,
              lid_lingua, lid_conf)

    return _consenso(unicode_top, unicode_candidatos, ft_lingua, ft_conf, lid_lingua, lid_conf)


def _consenso(
    unicode_top: tuple[str | None, float],
    unicode_candidatos: list[tuple[str, float]],
    ft_lingua: str | None,
    ft_conf: float,
    lid_lingua: str | None,
    lid_conf: float,
) -> tuple[str | None, float]:
    unicode_lingua, unicode_score = unicode_top
    unicode_linguas_set = {l for l, _ in unicode_candidatos}

    def _script_de(lingua: str | None) -> str | None:
        if lingua and lingua in LINGUAS:
            return LINGUAS[lingua][3]
        return None

    if ft_lingua is not None and ft_conf > 0.3:
        ft_script = _script_de(ft_lingua)
        unicode_script = _script_de(unicode_lingua)

        if ft_lingua == lid_lingua:
            conf = min(1.0, (ft_conf + lid_conf) / 2 + 0.15)

            if ft_lingua in unicode_linguas_set or ft_script == unicode_script:
                return ft_lingua, conf
            elif unicode_score < 0.3:
                return ft_lingua, conf * 0.9
            else:
                if unicode_score >= 0.85 and len(unicode_linguas_set) == 1:
                    return unicode_lingua, unicode_score
                return ft_lingua, conf * 0.7

        if ft_lingua == unicode_lingua:
            conf = min(1.0, ft_conf * 0.7 + unicode_score * 0.3 + 0.1)
            return ft_lingua, conf

        if ft_lingua in unicode_linguas_set:
            conf = min(1.0, ft_conf * 0.8 + 0.05)
            return ft_lingua, conf

        if ft_script == unicode_script:
            return ft_lingua, ft_conf * 0.85

        if unicode_score >= 0.85:
            if lid_lingua == unicode_lingua or lid_lingua in unicode_linguas_set:
                return lid_lingua if lid_lingua in unicode_linguas_set else unicode_lingua, unicode_score * 0.9
            return unicode_lingua, unicode_score * 0.7
        return ft_lingua, ft_conf * 0.5

    if lid_lingua is not None and lid_conf > 0.3:
        lid_script = _script_de(lid_lingua)
        unicode_script = _script_de(unicode_lingua)

        if lid_lingua == unicode_lingua:
            conf = min(1.0, lid_conf * 0.6 + unicode_score * 0.4)
            return lid_lingua, conf

        if lid_lingua in unicode_linguas_set:
            return lid_lingua, lid_conf * 0.7

        if lid_script == unicode_script:
            return lid_lingua, lid_conf * 0.6

        if unicode_score >= 0.85:
            return unicode_lingua, unicode_score * 0.7
        return lid_lingua, lid_conf * 0.4

    if unicode_lingua is not None and unicode_score > 0.2:
        return unicode_lingua, unicode_score
    if unicode_lingua is not None:
        return unicode_lingua, unicode_score * 0.5

    return None, 0.0


def detectar_lingua(texto: str) -> str | None:
    lingua, _ = detectar_lingua_auto(texto)
    return lingua


def descarregar_fasttext(verbose: bool = False) -> bool:
    if _FASTTEXT_PATH.exists():
        if verbose:
            log.info("Modelo fasttext já existe: %s", _FASTTEXT_PATH)
        return True

    try:
        import urllib.request
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if verbose:
            log.info("A descarregar fasttext lid.176.ftz (~900KB)...")
            print(f"A descarregar {_FASTTEXT_URL} ...")

        tmp_path = _FASTTEXT_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(_FASTTEXT_URL, tmp_path)
        tmp_path.rename(_FASTTEXT_PATH)

        if verbose:
            log.info("Modelo fasttext guardado em %s", _FASTTEXT_PATH)
        return True
    except Exception as e:
        log.warning("Falha ao descarregar fasttext: %s", e)
        if verbose:
            print(f"Falha ao descarregar modelo fasttext: {e}")
        return False
