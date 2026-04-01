
import os
import sys
import time
import glob
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.config import (
    LINGUAS, EXTENSOES_IMAGEM, MODELOS_ESPECIALIZADOS,
    listar_linguas, obter_lingua,
)
from src.ocr import extrair_texto, extrair_texto_auto, calcular_metricas_ocr, HAS_CV2
from src.limpeza import limpar_texto_ocr, dividir_em_paragrafos
from src.tradutor import (
    carregar_modelo, carregar_cache, guardar_cache,
    traduzir, traduzir_paragrafos,
    get_device, get_modelo_usado,
)
from src.pdf import HAS_PDF, extrair_paginas_pdf
from src.utils import detectar_lingua_auto
from src.modelo_router import (
    listar_modelos_disponiveis, descarregar_modelo, obter_nome_modelo,
)


def _imprimir_info():
    device = get_device()
    print(f"Dispositivo: {device}", end="")
    if device == "cuda":
        import torch
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"OpenCV: {'sim' if HAS_CV2 else 'não'}")
    print(f"PDF: {'sim' if HAS_PDF else 'não'}")
    print(f"Línguas: {len(LINGUAS)}")

    camadas = ["Unicode ranges"]
    try:
        import fasttext
        from src.utils import _FASTTEXT_PATH
        if _FASTTEXT_PATH.exists():
            camadas.append("fasttext lid.176.ftz")
    except ImportError:
        pass
    try:
        import langid
        camadas.append("langid")
    except ImportError:
        pass
    print(f"Detecção de língua: {' + '.join(camadas)}")
    print()


def processar_imagem_auto(caminho_ou_img, verbose: bool = False,
                          nome: str = "", directo: bool = False) -> str:
    label = nome or (caminho_ou_img if isinstance(caminho_ou_img, str) else "imagem")
    print(f"\n{'='*60}\nFicheiro: {label}\nModo: Detecção automática\n{'='*60}")

    t0 = time.perf_counter()
    print("A detectar língua e extrair texto (OCR)...")
    texto_ocr, lingua_detectada, confianca = extrair_texto_auto(
        caminho_ou_img, verbose=verbose)
    t_ocr = time.perf_counter() - t0

    if lingua_detectada:
        nome_lingua = LINGUAS[lingua_detectada][0] if lingua_detectada in LINGUAS else lingua_detectada
        print(f"  Língua detectada: {nome_lingua} ({lingua_detectada}) — confiança: {confianca:.0f}%")
    else:
        print("  [aviso] Não foi possível detectar a língua.")
        if texto_ocr:
            lingua_detectada_texto, conf_texto = detectar_lingua_auto(texto_ocr)
            if lingua_detectada_texto:
                lingua_detectada = lingua_detectada_texto
                nome_lingua = LINGUAS[lingua_detectada][0] if lingua_detectada in LINGUAS else lingua_detectada
                print(f"  Fallback detecção por texto: {nome_lingua} ({lingua_detectada}) — confiança: {conf_texto:.2f}")
            else:
                print("  [erro] Impossível determinar língua. Use --lingua para especificar.")
                return ""
        else:
            print("  [erro] OCR não extraiu texto.")
            return ""

    if not texto_ocr.strip():
        print("  [aviso] OCR não extraiu texto.")
        return ""

    print(f"- Texto OCR ({len(texto_ocr)} chars, {t_ocr:.1f}s) -")
    if verbose:
        print(texto_ocr)

    texto_limpo = limpar_texto_ocr(texto_ocr)
    paragrafos = dividir_em_paragrafos(texto_limpo)

    if not paragrafos:
        print("[aviso] Nenhum parágrafo encontrado.")
        return ""

    n = len(paragrafos)
    print(f"\n- Tradução ({n} parágrafos, língua: {lingua_detectada}) -")

    t1 = time.perf_counter()
    lista_en, lista_pt = traduzir_paragrafos(
        paragrafos, lingua=lingua_detectada, verbose=verbose, directo=directo,
        callback=lambda i, total: print(f"  [{i}/{total}]", end="\r", flush=True))
    t_trad = time.perf_counter() - t1

    texto_pt = "\n\n".join(lista_pt)
    print(f"\n{'='*60}")
    print(f"RESULTADO PT-PT (OCR: {t_ocr:.1f}s, Tradução: {t_trad:.1f}s, Total: {t_ocr + t_trad:.1f}s)")
    print(f"{'='*60}")
    print(texto_pt)
    return texto_pt


def processar_imagem(caminho_ou_img, lingua: str = "bengali",
                     lang_ocr: str | None = None,
                     verbose: bool = False, nome: str = "",
                     directo: bool = False) -> str:
    _, _, tess_default, _, _ = obter_lingua(lingua)
    lang_ocr = lang_ocr or tess_default

    label = nome or (caminho_ou_img if isinstance(caminho_ou_img, str) else "imagem")
    print(f"\n{'='*60}\nFicheiro: {label}\nLíngua: {lingua}\n{'='*60}")

    t0 = time.perf_counter()
    print("A extrair texto (OCR)...")
    texto_ocr = extrair_texto(caminho_ou_img, lang=lang_ocr)
    t_ocr = time.perf_counter() - t0

    if not texto_ocr.strip():
        print("  [aviso] OCR não extraiu texto.")
        return ""

    print(f"- Texto OCR ({len(texto_ocr)} chars, {t_ocr:.1f}s) -")
    if verbose:
        print(texto_ocr)

    texto_limpo = limpar_texto_ocr(texto_ocr)
    paragrafos = dividir_em_paragrafos(texto_limpo)

    if not paragrafos:
        print("[aviso] Nenhum parágrafo encontrado.")
        return ""

    n = len(paragrafos)
    print(f"\n- Tradução ({n} parágrafos) -")

    t1 = time.perf_counter()
    lista_en, lista_pt = traduzir_paragrafos(
        paragrafos, lingua=lingua, verbose=verbose, directo=directo,
        callback=lambda i, total: print(f"  [{i}/{total}]", end="\r", flush=True))
    t_trad = time.perf_counter() - t1

    texto_pt = "\n\n".join(lista_pt)
    print(f"\n{'='*60}")
    print(f"RESULTADO PT-PT (OCR: {t_ocr:.1f}s, Tradução: {t_trad:.1f}s, Total: {t_ocr + t_trad:.1f}s)")
    print(f"{'='*60}")
    print(texto_pt)
    return texto_pt


def processar_pdf(caminho: str, lingua: str = "bengali",
                  lang_ocr: str | None = None,
                  verbose: bool = False,
                  directo: bool = False) -> str:
    imagens = extrair_paginas_pdf(caminho)
    print(f"PDF: {caminho} — {len(imagens)} páginas")
    resultados = []
    for i, img in enumerate(imagens, 1):
        res = processar_imagem(
            img, lingua=lingua, lang_ocr=lang_ocr, verbose=verbose,
            nome=f"{caminho} — Página {i}/{len(imagens)}",
            directo=directo)
        if res:
            resultados.append(f"- Página {i} -\n{res}")
    return "\n\n".join(resultados)


def processar_pasta(pasta: str, lingua: str = "bengali",
                    lang_ocr: str | None = None,
                    verbose: bool = False,
                    directo: bool = False) -> dict[str, str]:
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
            resultados[f] = processar_pdf(f, lingua, lang_ocr, verbose, directo)
        else:
            resultados[f] = processar_imagem(f, lingua, lang_ocr, verbose,
                                              directo=directo)
    return resultados


GRUPOS_LINGUAS = {
    "europeus": ["espanhol", "frances", "italiano", "alemao", "romeno",
                 "polaco", "holandes", "checo"],
    "cirilicos": ["russo", "ucraniano", "bulgaro"],
    "asiaticos": ["chines_s", "chines_t", "japones", "coreano", "hindi"],
    "outros": ["arabe", "turco", "hebraico"],
}


def _listar_modelos():
    modelos = listar_modelos_disponiveis()
    print(f"\n{'Lingua':<20} {'Modelo':<40} {'Tipo':<15} {'Cache'}")
    print("-" * 1)
    for m in modelos:
        tipo = "directo->PT" if m["directo_pt"] else "pivot via EN"
        cache = "SIM" if m["em_cache"] else "-"
        print(f"{m['nome_lingua']:<20} {m['modelo']:<40} {tipo:<15} {cache}")
    print()

    em_cache = sum(1 for m in modelos if m["em_cache"])
    total = len(modelos)
    print(f"Em cache: {em_cache}/{total}")
    print(f"\nUsar --descarregar-modelo <lingua> para descarregar.")
    print(f"Grupos: --descarregar-modelo --europeus | --cirilicos | --asiaticos | --outros | --todos")


def _descarregar_modelos(args_linguas: list[str]):
    linguas = []

    for arg in args_linguas:
        arg_clean = arg.lstrip("-")
        if arg_clean == "todos":
            linguas = list(MODELOS_ESPECIALIZADOS.keys())
            break
        elif arg_clean in GRUPOS_LINGUAS:
            linguas.extend(GRUPOS_LINGUAS[arg_clean])
        elif arg_clean in MODELOS_ESPECIALIZADOS:
            linguas.append(arg_clean)
        else:
            print(f"[aviso] '{arg_clean}' nao reconhecido. Linguas disponiveis:")
            for l in sorted(MODELOS_ESPECIALIZADOS.keys()):
                print(f"  {l}")
            print(f"Grupos: europeus, cirilicos, asiaticos, outros, todos")
            return

    if not linguas:
        print("Uso: --descarregar-modelo espanhol frances russo")
        print("     --descarregar-modelo --europeus")
        print("     --descarregar-modelo --todos")
        return

    vistas = set()
    linguas_unicas = []
    for l in linguas:
        if l not in vistas:
            vistas.add(l)
            linguas_unicas.append(l)

    print(f"\nA descarregar modelos para {len(linguas_unicas)} linguas...\n")
    ok = 0
    for lingua in linguas_unicas:
        nome_l = LINGUAS[lingua][0] if lingua in LINGUAS else lingua
        print(f"[{lingua}] {nome_l}")
        if descarregar_modelo(lingua, verbose=True):
            ok += 1
        print()

    print(f"\nResultado: {ok}/{len(linguas_unicas)} modelos descarregados com sucesso.")


def main():
    parser = argparse.ArgumentParser(
        description="OCR + Traducao multilingue offline -> Portugues (PT-PT)")
    parser.add_argument("input", nargs="?", default=None)
    parser.add_argument("--lingua", "-l", default="bengali",
                        help="Língua de origem (default: bengali)")
    parser.add_argument("--lang-ocr", default=None,
                        help="Override Tesseract lang (ex: rus+eng)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--texto", action="store_true",
                        help="Input é ficheiro de texto")
    parser.add_argument("--directo", action="store_true",
                        help="Tradução directa (sem pivot inglês) para línguas suportadas")
    parser.add_argument("--auto", "-a", action="store_true",
                        help="Detecção automática de língua (ignora --lingua)")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--listar-linguas", action="store_true")
    parser.add_argument("--listar-modelos", action="store_true",
                        help="Lista modelos especializados e estado de cache")
    parser.add_argument("--descarregar-modelo", nargs="*", default=None,
                        metavar="LINGUA",
                        help="Descarrega modelos especializados (ex: espanhol frances russo)")
    parser.add_argument("--limpar-cache", action="store_true")
    args = parser.parse_args()

    if args.listar_linguas:
        listar_linguas()
        sys.exit(0)
    if args.listar_modelos:
        _listar_modelos()
        sys.exit(0)
    if args.descarregar_modelo is not None:
        _descarregar_modelos(args.descarregar_modelo)
        sys.exit(0)
    if args.info:
        _imprimir_info()
        sys.exit(0)
    if args.limpar_cache:
        from src.tradutor import limpar_cache
        limpar_cache()
        print("Cache limpa.")
        sys.exit(0)
    if not args.input:
        parser.print_help()
        sys.exit(1)

    carregar_cache()
    _imprimir_info()

    texto_final = ""

    t_total = time.perf_counter()

    if args.texto:
        texto = Path(args.input).read_text(encoding="utf-8")
        lingua = args.lingua
        if args.auto:
            lingua_det, conf = detectar_lingua_auto(texto)
            if lingua_det:
                lingua = lingua_det
                nome_l = LINGUAS[lingua][0] if lingua in LINGUAS else lingua
                print(f"Língua detectada: {nome_l} ({lingua}) — confiança: {conf:.2f}")
            else:
                print(f"[aviso] Detecção automática falhou, a usar: {lingua}")

        texto_limpo = limpar_texto_ocr(texto)
        paragrafos = dividir_em_paragrafos(texto_limpo)
        if paragrafos:
            _, lista_pt = traduzir_paragrafos(
                paragrafos, lingua=lingua,
                verbose=args.verbose, directo=args.directo)
            texto_final = "\n\n".join(lista_pt)
    elif args.auto and os.path.isfile(args.input) and not args.input.lower().endswith(".pdf"):
        texto_final = processar_imagem_auto(args.input, args.verbose,
                                             directo=args.directo)
    elif os.path.isdir(args.input):
        if args.auto:
            extensoes = EXTENSOES_IMAGEM | ({".pdf"} if HAS_PDF else set())
            ficheiros = sorted(
                f for f in glob.glob(os.path.join(args.input, "*"))
                if Path(f).suffix.lower() in extensoes
            )
            if not ficheiros:
                print(f"Nenhum ficheiro suportado em: {args.input}")
            else:
                print(f"{len(ficheiros)} ficheiros em: {args.input} (modo auto)")
                resultados = {}
                for f in ficheiros:
                    resultados[f] = processar_imagem_auto(
                        f, args.verbose, nome=os.path.basename(f),
                        directo=args.directo)
                texto_final = "\n\n".join(
                    f"# {os.path.basename(f)}\n{t}" for f, t in resultados.items() if t)
        else:
            res = processar_pasta(args.input, args.lingua, args.lang_ocr,
                                  args.verbose, args.directo)
            texto_final = "\n\n".join(
                f"# {os.path.basename(f)}\n{t}" for f, t in res.items() if t)
    elif args.input.lower().endswith(".pdf"):
        texto_final = processar_pdf(args.input, args.lingua, args.lang_ocr,
                                    args.verbose, args.directo)
    elif os.path.isfile(args.input):
        texto_final = processar_imagem(args.input, args.lingua, args.lang_ocr,
                                       args.verbose, directo=args.directo)
    else:
        print(f"Erro: '{args.input}' não encontrado.")
        sys.exit(1)

    dt = time.perf_counter() - t_total
    print(f"\nTempo total: {dt:.1f}s")

    if args.output and texto_final:
        Path(args.output).write_text(texto_final, encoding="utf-8")
        print(f"Guardado em: {args.output}")


if __name__ == "__main__":
    main()
