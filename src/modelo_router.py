
import time
import logging
from pathlib import Path

from .config import (
    MODELOS_ESPECIALIZADOS, MODELO_PIVOT_EN_PT, MODELOS_PREFERIDOS,
    LINGUAS,
)

log = logging.getLogger(__name__)

_opus_modelo = None
_opus_tokenizer = None
_opus_nome = None

_pivot_modelo = None
_pivot_tokenizer = None
_pivot_carregado = False

_torch = None
_transformers = None


def _lazy_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _lazy_transformers():
    global _transformers
    if _transformers is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _transformers = type("T", (), {
            "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
            "AutoTokenizer": AutoTokenizer,
        })()
    return _transformers


def modelo_em_cache(nome: str) -> bool:
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    modelo_dir = cache_base / f"models--{nome.replace('/', '--')}"
    if not modelo_dir.exists():
        return False
    snapshots = modelo_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir():
            model_files = (list(snap.glob("*.bin")) +
                           list(snap.glob("*.safetensors")) +
                           list(snap.glob("opus.spm")))
            if model_files:
                return True
    return False


def listar_modelos_disponiveis() -> list[dict]:
    resultado = []

    for lingua, config in sorted(MODELOS_ESPECIALIZADOS.items()):
        nome_lingua = LINGUAS[lingua][0] if lingua in LINGUAS else lingua
        em_cache = modelo_em_cache(config["modelo"])
        resultado.append({
            "lingua": lingua,
            "nome_lingua": nome_lingua,
            "modelo": config["modelo"],
            "directo_pt": config["directo_pt"],
            "em_cache": em_cache,
        })

    resultado.append({
        "lingua": "_pivot_en_pt",
        "nome_lingua": "Pivot EN->PT",
        "modelo": MODELO_PIVOT_EN_PT,
        "directo_pt": True,
        "em_cache": modelo_em_cache(MODELO_PIVOT_EN_PT),
    })

    return resultado


def obter_backend(lingua: str) -> str:
    if lingua not in MODELOS_ESPECIALIZADOS:
        return "nllb"

    config = MODELOS_ESPECIALIZADOS[lingua]
    modelo_nome = config["modelo"]

    if not modelo_em_cache(modelo_nome):
        log.debug("Modelo %s não em cache, fallback NLLB", modelo_nome)
        return "nllb"

    if config["directo_pt"]:
        return "opus-mt-directo"

    if not modelo_em_cache(MODELO_PIVOT_EN_PT):
        log.debug("Modelo pivot %s não em cache, fallback NLLB",
                  MODELO_PIVOT_EN_PT)
        return "nllb"

    return "opus-mt-pivot"


def obter_nome_modelo(lingua: str) -> str:
    backend = obter_backend(lingua)
    if backend == "nllb":
        return "NLLB-200 (genérico)"
    config = MODELOS_ESPECIALIZADOS[lingua]
    nome = config["modelo"].split("/")[-1]
    if backend == "opus-mt-directo":
        return f"{nome} (directo → PT)"
    return f"{nome} + en→pt (pivot)"


def _get_device() -> str:
    torch = _lazy_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _carregar_opus_mt(nome_modelo: str):
    torch = _lazy_torch()
    tf = _lazy_transformers()
    device = _get_device()

    t0 = time.perf_counter()
    log.info("A carregar %s (%s)...", nome_modelo, device)
    print(f"  A carregar {nome_modelo} ({device})... ", end="", flush=True)

    tokenizer = tf.AutoTokenizer.from_pretrained(nome_modelo)

    if device == "cuda":
        model = tf.AutoModelForSeq2SeqLM.from_pretrained(
            nome_modelo, torch_dtype=torch.float16)
    else:
        model = tf.AutoModelForSeq2SeqLM.from_pretrained(nome_modelo)
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            print("(INT8) ", end="", flush=True)
        except Exception:
            pass

    model.eval()
    model.to(device)

    dt = time.perf_counter() - t0
    print(f"OK ({dt:.1f}s)")
    log.info("Modelo %s carregado em %.1fs", nome_modelo, dt)

    return model, tokenizer


def _garantir_opus_mt(lingua: str):
    global _opus_modelo, _opus_tokenizer, _opus_nome

    config = MODELOS_ESPECIALIZADOS[lingua]
    nome_necessario = config["modelo"]

    if _opus_nome == nome_necessario and _opus_modelo is not None:
        return _opus_modelo, _opus_tokenizer

    if _opus_modelo is not None:
        log.info("A descarregar %s para carregar %s", _opus_nome, nome_necessario)
        _opus_modelo = None
        _opus_tokenizer = None
        _opus_nome = None
        torch = _lazy_torch()
        if hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    _opus_modelo, _opus_tokenizer = _carregar_opus_mt(nome_necessario)
    _opus_nome = nome_necessario

    return _opus_modelo, _opus_tokenizer


def _garantir_pivot():
    global _pivot_modelo, _pivot_tokenizer, _pivot_carregado

    if _pivot_carregado:
        return _pivot_modelo, _pivot_tokenizer

    _pivot_modelo, _pivot_tokenizer = _carregar_opus_mt(MODELO_PIVOT_EN_PT)
    _pivot_carregado = True

    return _pivot_modelo, _pivot_tokenizer


def _gerar_opus_mt(model, tokenizer, texto: str, max_tokens: int = 512) -> str:
    torch = _lazy_torch()
    device = _get_device()

    inputs = tokenizer(texto, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)

    with torch.no_grad():
        output_ids = model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=4,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _gerar_opus_mt_batch(model, tokenizer, textos: list[str],
                          max_tokens: int = 512) -> list[str]:
    torch = _lazy_torch()
    device = _get_device()

    inputs = tokenizer(textos, return_tensors="pt", truncation=True,
                       max_length=512, padding=True)

    with torch.no_grad():
        output_ids = model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=4,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def traduzir_opus_mt(texto: str, lingua: str,
                      verbose: bool = False) -> tuple[str, str]:
    config = MODELOS_ESPECIALIZADOS[lingua]
    model, tokenizer = _garantir_opus_mt(lingua)

    if config["directo_pt"]:
        texto_pt = _gerar_opus_mt(model, tokenizer, texto)
        if verbose:
            print(f"  [opus-mt directo] {texto_pt[:80]}")
        return "", texto_pt

    texto_en = _gerar_opus_mt(model, tokenizer, texto)
    if verbose:
        print(f"  [opus-mt → EN] {texto_en[:80]}")

    pivot_model, pivot_tok = _garantir_pivot()
    texto_pt = _gerar_opus_mt(pivot_model, pivot_tok, texto_en)
    if verbose:
        print(f"  [opus-mt → PT] {texto_pt[:80]}")

    return texto_en, texto_pt


def traduzir_opus_mt_batch(textos: list[str], lingua: str,
                            batch_size: int = 8,
                            verbose: bool = False) -> tuple[list[str], list[str]]:
    config = MODELOS_ESPECIALIZADOS[lingua]
    model, tokenizer = _garantir_opus_mt(lingua)

    if config["directo_pt"]:
        resultados_pt = []
        for i in range(0, len(textos), batch_size):
            batch = textos[i:i + batch_size]
            resultados_pt.extend(_gerar_opus_mt_batch(model, tokenizer, batch))
        return ["[opus-mt directo]"] * len(textos), resultados_pt

    resultados_en = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        resultados_en.extend(_gerar_opus_mt_batch(model, tokenizer, batch))

    if verbose:
        print(f"  Batch opus-mt X→EN: {len(textos)} segmentos")

    pivot_model, pivot_tok = _garantir_pivot()
    resultados_pt = []
    for i in range(0, len(resultados_en), batch_size):
        batch = resultados_en[i:i + batch_size]
        resultados_pt.extend(_gerar_opus_mt_batch(pivot_model, pivot_tok, batch))

    if verbose:
        print(f"  Batch opus-mt EN→PT: {len(resultados_en)} segmentos")

    return resultados_en, resultados_pt


def descarregar_modelo(lingua: str, verbose: bool = True) -> bool:
    if lingua not in MODELOS_ESPECIALIZADOS:
        if verbose:
            print(f"Sem modelo especializado para '{lingua}' — usa NLLB.")
        return False

    config = MODELOS_ESPECIALIZADOS[lingua]
    nome = config["modelo"]

    if modelo_em_cache(nome):
        if verbose:
            print(f"  {nome}: já em cache")
    else:
        if verbose:
            print(f"  A descarregar {nome}...")
        try:
            tf = _lazy_transformers()
            tf.AutoTokenizer.from_pretrained(nome)
            tf.AutoModelForSeq2SeqLM.from_pretrained(nome)
            if verbose:
                print(f"  {nome}: OK")
        except Exception as e:
            if verbose:
                print(f"  {nome}: FALHOU ({e})")
            return False

    if not config["directo_pt"]:
        if modelo_em_cache(MODELO_PIVOT_EN_PT):
            if verbose:
                print(f"  {MODELO_PIVOT_EN_PT}: já em cache")
        else:
            if verbose:
                print(f"  A descarregar {MODELO_PIVOT_EN_PT}...")
            try:
                tf = _lazy_transformers()
                tf.AutoTokenizer.from_pretrained(MODELO_PIVOT_EN_PT)
                tf.AutoModelForSeq2SeqLM.from_pretrained(MODELO_PIVOT_EN_PT)
                if verbose:
                    print(f"  {MODELO_PIVOT_EN_PT}: OK")
            except Exception as e:
                if verbose:
                    print(f"  {MODELO_PIVOT_EN_PT}: FALHOU ({e})")
                return False

    return True
