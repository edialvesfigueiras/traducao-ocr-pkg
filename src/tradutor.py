
import json
import time
import hashlib
import re
from pathlib import Path

from .config import (
    MODELOS_PREFERIDOS, PIVOT_LANG, TGT_LANG,
    CACHE_DIR, CACHE_FILE,
    NUM_BEAMS_DEFAULT, MAX_TOKENS_DEFAULT, BATCH_SIZE_DEFAULT,
    LINGUAS, PARES_DIRECTOS,
    obter_lingua,
)

_torch = None
_transformers = None
_model = None
_tokenizers: dict[str, object] = {}
_lang_ids: dict[str, int] = {}
_modelo_usado = None
_device = None
_cache_traducoes: dict[str, str] = {}


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


def get_device() -> str:
    global _device
    if _device is None:
        torch = _lazy_torch()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_modelo_usado() -> str | None:
    return _modelo_usado


def carregar_cache():
    global _cache_traducoes
    if CACHE_FILE.exists():
        try:
            _cache_traducoes = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            _cache_traducoes = {}


def guardar_cache():
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


def limpar_cache():
    global _cache_traducoes
    _cache_traducoes.clear()
    guardar_cache()


def num_entradas_cache() -> int:
    return len(_cache_traducoes)


def _hash_texto(texto: str, lingua: str = "") -> str:
    return hashlib.md5(f"{lingua}:{texto}".encode("utf-8")).hexdigest()


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


def carregar_modelo():
    global _model, _modelo_usado

    if _model is not None:
        return

    torch = _lazy_torch()
    tf = _lazy_transformers()
    device = get_device()

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
            t0 = time.perf_counter()
            print(f"A carregar {tentativa} ({device})... ", end="", flush=True)

            if device == "cuda":
                _model = tf.AutoModelForSeq2SeqLM.from_pretrained(
                    tentativa, torch_dtype=torch.float16)
            else:
                _model = tf.AutoModelForSeq2SeqLM.from_pretrained(tentativa)
                try:
                    _model = torch.quantization.quantize_dynamic(
                        _model, {torch.nn.Linear}, dtype=torch.qint8)
                    print("(INT8) ", end="", flush=True)
                except Exception:
                    pass

            _model.generation_config.max_length = None
            _model.eval()
            _model.to(device)

            if hasattr(torch, "compile") and device == "cuda":
                try:
                    _model = torch.compile(_model)
                    print("(compiled) ", end="", flush=True)
                except Exception:
                    pass

            dt = time.perf_counter() - t0
            print(f"OK ({dt:.1f}s)")
            return
        except Exception as e:
            print(f"FALHOU ({e})")
            _model = None
            if tentativa != MODELOS_PREFERIDOS[-1]:
                print("  A tentar alternativa...", flush=True)

    raise RuntimeError("Não foi possível carregar nenhum modelo NLLB.")


def _obter_tokenizer(lang_code: str):
    global _tokenizers, _lang_ids
    if lang_code in _tokenizers:
        return _tokenizers[lang_code], _lang_ids.get(PIVOT_LANG), _lang_ids.get(TGT_LANG)

    tf = _lazy_transformers()
    carregar_modelo()
    tok = tf.AutoTokenizer.from_pretrained(_modelo_usado, src_lang=lang_code)
    _tokenizers[lang_code] = tok

    for tgt in [PIVOT_LANG, TGT_LANG]:
        if tgt not in _lang_ids:
            lid = tok.convert_tokens_to_ids(tgt)
            if lid != tok.unk_token_id:
                _lang_ids[tgt] = lid

    return tok, _lang_ids.get(PIVOT_LANG), _lang_ids.get(TGT_LANG)


def _segmentar_texto(texto: str, max_tokens: int = 400) -> list[str]:
    if len(texto) < max_tokens * 3:
        return [texto]

    try:
        import nltk
        try:
            frases = nltk.sent_tokenize(texto)
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            frases = nltk.sent_tokenize(texto)
    except ImportError:
        frases = re.split(
            r'(?<=[.!?।॥\u3002\uff01\uff1f\u061F\u06D4])\s+', texto)
        if len(frases) <= 1:
            frases = re.split(r'(?<=[,;:，；：\u060C\u061B])\s+', texto)

    if len(frases) <= 1:
        return [texto]

    segmentos = []
    actual = []
    actual_len = 0
    limite = max_tokens * 3

    for frase in frases:
        if actual_len + len(frase) > limite and actual:
            segmentos.append(" ".join(actual))
            actual = []
            actual_len = 0
        actual.append(frase)
        actual_len += len(frase) + 1

    if actual:
        segmentos.append(" ".join(actual))

    return segmentos


def _gerar_single(tokenizer, inputs, forced_bos_id: int,
                  max_tokens: int = MAX_TOKENS_DEFAULT,
                  beams: int = NUM_BEAMS_DEFAULT) -> str:
    torch = _lazy_torch()
    device = get_device()
    with torch.no_grad():
        output_ids = _model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=max_tokens,
            num_beams=beams,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def _gerar_batch(tokenizer, textos: list[str], forced_bos_id: int,
                 max_tokens: int = MAX_TOKENS_DEFAULT,
                 beams: int = NUM_BEAMS_DEFAULT) -> list[str]:
    torch = _lazy_torch()
    device = get_device()

    inputs = tokenizer(
        textos, return_tensors="pt", truncation=True,
        max_length=512, padding=True)

    with torch.no_grad():
        output_ids = _model.generate(
            **{k: v.to(device) for k, v in inputs.items()},
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=max_tokens,
            num_beams=beams,
            do_sample=False,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.3,
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def traduzir_para_eng(texto: str, src_lang: str) -> str:
    carregar_modelo()
    tok_src, pivot_id, _ = _obter_tokenizer(src_lang)
    inputs = tok_src(texto, return_tensors="pt", truncation=True, max_length=512)
    return _gerar_single(tok_src, inputs, pivot_id)


def traduzir_eng_para_por(texto: str) -> str:
    carregar_modelo()
    tok_eng, _, target_id = _obter_tokenizer(PIVOT_LANG)
    inputs = tok_eng(texto, return_tensors="pt", truncation=True, max_length=512)
    return _gerar_single(tok_eng, inputs, target_id)


def traduzir_directo(texto: str, src_lang: str) -> str:
    carregar_modelo()
    tok_src, _, target_id = _obter_tokenizer(src_lang)
    inputs = tok_src(texto, return_tensors="pt", truncation=True, max_length=512)
    return _gerar_single(tok_src, inputs, target_id)


def traduzir_batch_para_eng(textos: list[str], src_lang: str,
                            batch_size: int = BATCH_SIZE_DEFAULT) -> list[str]:
    carregar_modelo()
    tok_src, pivot_id, _ = _obter_tokenizer(src_lang)
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        resultados.extend(_gerar_batch(tok_src, batch, pivot_id))
    return resultados


def traduzir_batch_eng_para_por(textos: list[str],
                                batch_size: int = BATCH_SIZE_DEFAULT) -> list[str]:
    carregar_modelo()
    tok_eng, _, target_id = _obter_tokenizer(PIVOT_LANG)
    resultados = []
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i + batch_size]
        resultados.extend(_gerar_batch(tok_eng, batch, target_id))
    return resultados


def traduzir(texto: str, lingua: str = "bengali",
             verbose: bool = False, directo: bool = False) -> str:
    from .pos_processamento import pos_processar_pt
    from .modelo_router import obter_backend, traduzir_opus_mt

    _, nllb_code, _, _, _ = obter_lingua(lingua)
    backend = obter_backend(lingua)

    cache_suffix = lingua + (":dir" if directo else "")
    if backend.startswith("opus-mt"):
        cache_suffix += f":opus"
    chave = _hash_texto(texto, cache_suffix)
    if chave in _cache_traducoes:
        if verbose:
            print("  [cache]")
        return _cache_traducoes[chave]

    segmentos = _segmentar_texto(texto)

    resultados_pt = []
    for seg in segmentos:
        if backend.startswith("opus-mt"):
            _, texto_pt = traduzir_opus_mt(seg, lingua, verbose=verbose)
        elif directo and lingua in PARES_DIRECTOS:
            texto_pt = traduzir_directo(seg, nllb_code)
            if verbose:
                print(f"  [NLLB directo] {texto_pt[:80]}")
        else:
            texto_en = traduzir_para_eng(seg, nllb_code)
            if verbose:
                print(f"  [NLLB → EN] {texto_en[:80]}")
            texto_pt = traduzir_eng_para_por(texto_en)

        resultados_pt.append(pos_processar_pt(texto_pt))

    resultado = " ".join(resultados_pt)
    _cache_traducoes[chave] = resultado
    return resultado


def traduzir_paragrafos(paragrafos: list[str], lingua: str = "bengali",
                        verbose: bool = False, directo: bool = False,
                        callback=None) -> tuple[list[str], list[str]]:
    from .pos_processamento import pos_processar_pt
    from .modelo_router import obter_backend, traduzir_opus_mt_batch, obter_nome_modelo

    _, nllb_code, _, _, _ = obter_lingua(lingua)
    backend = obter_backend(lingua)

    if verbose:
        print(f"  Backend: {obter_nome_modelo(lingua)}")

    cache_suffix = lingua + (":dir" if directo else "")
    if backend.startswith("opus-mt"):
        cache_suffix += ":opus"

    resultados_en = [""] * len(paragrafos)
    resultados_pt = [""] * len(paragrafos)
    indices_novos = []
    textos_novos = []

    for i, p in enumerate(paragrafos):
        chave = _hash_texto(p, cache_suffix)
        if chave in _cache_traducoes:
            resultados_pt[i] = _cache_traducoes[chave]
            resultados_en[i] = "[cache]"
        else:
            indices_novos.append(i)
            segs = _segmentar_texto(p)
            textos_novos.append(segs)

    if textos_novos:
        all_segs = [seg for segs in textos_novos for seg in segs]
        seg_counts = [len(segs) for segs in textos_novos]

        if backend.startswith("opus-mt"):
            t0 = time.perf_counter()
            all_en, all_pt = traduzir_opus_mt_batch(
                all_segs, lingua, verbose=verbose)
            dt = time.perf_counter() - t0
            if verbose:
                print(f"  Batch opus-mt: {dt:.1f}s ({len(all_segs)} segmentos)")

            pos = 0
            for j, idx in enumerate(indices_novos):
                count = seg_counts[j]
                en_parts = all_en[pos:pos + count]
                pt_parts = all_pt[pos:pos + count]
                en_text = " ".join(en_parts)
                pt_text = " ".join(pos_processar_pt(p) for p in pt_parts)
                resultados_en[idx] = en_text
                resultados_pt[idx] = pt_text
                _cache_traducoes[_hash_texto(paragrafos[idx], cache_suffix)] = pt_text
                pos += count
                if callback:
                    callback(idx + 1, len(paragrafos))

        elif directo and lingua in PARES_DIRECTOS:
            carregar_modelo()
            tok_src, _, target_id = _obter_tokenizer(nllb_code)
            all_pt = []
            for bi in range(0, len(all_segs), BATCH_SIZE_DEFAULT):
                batch = all_segs[bi:bi + BATCH_SIZE_DEFAULT]
                all_pt.extend(_gerar_batch(tok_src, batch, target_id))

            pos = 0
            for j, idx in enumerate(indices_novos):
                count = seg_counts[j]
                pt_parts = all_pt[pos:pos + count]
                pt_text = " ".join(pos_processar_pt(p) for p in pt_parts)
                resultados_pt[idx] = pt_text
                resultados_en[idx] = "[NLLB directo]"
                _cache_traducoes[_hash_texto(paragrafos[idx], cache_suffix)] = pt_text
                pos += count
                if callback:
                    callback(idx + 1, len(paragrafos))
        else:
            carregar_modelo()
            t0 = time.perf_counter()
            all_en = traduzir_batch_para_eng(all_segs, nllb_code)
            t1 = time.perf_counter()
            if verbose:
                print(f"  Batch NLLB src→eng: {t1 - t0:.1f}s ({len(all_segs)} segmentos)")

            all_pt = traduzir_batch_eng_para_por(all_en)
            t2 = time.perf_counter()
            if verbose:
                print(f"  Batch NLLB eng→por: {t2 - t1:.1f}s")

            pos = 0
            for j, idx in enumerate(indices_novos):
                count = seg_counts[j]
                en_parts = all_en[pos:pos + count]
                pt_parts = all_pt[pos:pos + count]
                en_text = " ".join(en_parts)
                pt_text = " ".join(pos_processar_pt(p) for p in pt_parts)
                resultados_en[idx] = en_text
                resultados_pt[idx] = pt_text
                _cache_traducoes[_hash_texto(paragrafos[idx], cache_suffix)] = pt_text
                pos += count
                if callback:
                    callback(idx + 1, len(paragrafos))
    else:
        if callback:
            callback(len(paragrafos), len(paragrafos))

    guardar_cache()
    return resultados_en, resultados_pt
