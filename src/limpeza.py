
import re
import unicodedata


def _tem_script_nao_latino(texto: str) -> bool:
    return bool(re.search(
        r"[\u0400-\u04FF"
        r"\u0600-\u06FF"
        r"\u0750-\u077F"
        r"\u0900-\u097F"
        r"\u0980-\u09FF"
        r"\u0A00-\u0A7F"
        r"\u0A80-\u0AFF"
        r"\u0B80-\u0BFF"
        r"\u0C00-\u0C7F"
        r"\u0C80-\u0CFF"
        r"\u0D00-\u0D7F"
        r"\u0E00-\u0E7F"
        r"\u0E80-\u0EFF"
        r"\u1000-\u109F"
        r"\u10A0-\u10FF"
        r"\u1780-\u17FF"
        r"\u3040-\u30FF"
        r"\u4E00-\u9FFF"
        r"\uAC00-\uD7AF"
        r"\u0370-\u03FF"
        r"\u0590-\u05FF"
        r"\uFB50-\uFDFF"
        r"\uFE70-\uFEFF"
        r"]", texto))


def _linha_e_lixo(linha: str) -> bool:
    l = linha.strip()
    if not l:
        return False
    if any(x in l.lower() for x in [
        "http", "youtube", ".com", ".org", ".net", "www.",
        "t.me", "t.co", "fb.com", "instagram", "twitter",
        "facebook", "whatsapp", "tiktok", "telegram",
        "subscribe", "follow", "like", "share",
    ]):
        return True
    if _tem_script_nao_latino(l):
        return False
    if re.fullmatch(r"[a-zA-Z0-9\s_.@#/\-]+", l) and len(l) < 40:
        return True
    if re.fullmatch(r"[\s\W_]+", l):
        return True
    if "_" in l and " " not in l:
        return True
    return False


def normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"[\u200B\uFEFF\u00AD]", "", texto)
    texto = re.sub(r"\t", " ", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    if re.search(r"[\u0900-\u0DFF]", texto):
        texto = texto.replace("|", "।")
    texto = re.sub(r"\s+([।,;:!?\u061F\u060C\u061B])", r"\1", texto)
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


def dividir_em_paragrafos(texto: str) -> list[str]:
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
