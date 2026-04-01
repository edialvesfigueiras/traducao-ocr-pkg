
import os
import sys
import time
import base64
import html as html_mod
import tempfile
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import streamlit as st
from PIL import Image

from src.config import (
    LINGUAS, SCRIPTS, EXTENSOES_IMAGEM, PARES_DIRECTOS, LOGO_PATH,
    obter_lingua,
)
from src.ocr import (
    extrair_texto, extrair_texto_auto, extrair_confianca,
    preprocessar_imagem_pipeline,
    calcular_metricas_ocr, detectar_script_osd, HAS_CV2,
)
from src.limpeza import limpar_texto_ocr, dividir_em_paragrafos
from src.tradutor import (
    carregar_modelo, carregar_cache, guardar_cache, limpar_cache,
    traduzir_paragrafos, num_entradas_cache,
    get_device, get_modelo_usado,
)
from src.pos_processamento import pos_processar_pt
from src.pdf import HAS_PDF, extrair_paginas_pdf
from src.utils import detectar_lingua_auto
from src.modelo_router import obter_backend, obter_nome_modelo
from src.config import OCR_QUALIDADE_EXCELENTE, OCR_QUALIDADE_BOM, OCR_QUALIDADE_FRACO


st.set_page_config(
    page_title="OCR + Tradução Multilingue",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_B64 = ""
if LOGO_PATH.exists():
    LOGO_B64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")

if "historico" not in st.session_state:
    st.session_state.historico = []


_CSS = """
<style>
    .app-header {
        background: #1b1f23;
        padding: 1.4rem 2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #2f81f7;
        display: flex;
        align-items: center;
        gap: 1.4rem;
    }
    .app-header img { width: 52px; height: auto; flex-shrink: 0; }
    .app-header h1 {
        color: #f0f0f0; font-size: 1.4rem; font-weight: 600;
        margin: 0; letter-spacing: -0.01em;
    }
    .app-header p { color: #8b949e; font-size: 0.85rem; margin: 0.25rem 0 0 0; }

    .result-card {
        border: 1px solid #d0d7de;
        border-radius: 6px; padding: 1.25rem 1.5rem; margin: 0.6rem 0;
    }
    .result-card h3 {
        font-size: 0.75rem; font-weight: 600; color: #656d76;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.6rem; padding-bottom: 0.4rem;
        border-bottom: 1px solid #d0d7de;
    }
    .result-card .content { font-size: 0.95rem; line-height: 1.75; }
    .result-card.english .content { color: #656d76; }
    .result-card.portuguese { border-left: 3px solid #2f81f7; }
    .result-card.portuguese .content { font-weight: 500; }

    .stats-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.8rem 0; }
    .stat-pill {
        background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 3px;
        padding: 0.2rem 0.6rem; font-size: 0.75rem; color: #656d76;
        font-family: 'IBM Plex Mono', monospace;
    }

    .section-label {
        font-size: 0.8rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em;
        margin: 1rem 0 0.4rem 0;
        opacity: 0.7;
    }

    .step-indicator { display: flex; align-items: center; gap: 0.4rem; padding: 0.5rem 0; font-family: 'IBM Plex Mono', monospace; }
    .step-dot { width: 7px; height: 7px; border-radius: 50%; background: #d0d7de; }
    .step-dot.active { background: #2f81f7; }
    .step-dot.done { background: #1a7f37; }
    .step-label { font-size: 0.75rem; color: #656d76; }
    .step-label.active { color: #2f81f7; font-weight: 600; }
    .step-label.done { color: #1a7f37; }
    .step-sep { color: #d0d7de; margin: 0 0.15rem; font-size: 0.7rem; }

    .divider { border: none; border-top: 1px solid #d0d7de; margin: 1.5rem 0; }

    .log-box {
        background: #f6f8fa; border: 1px solid #d0d7de;
        border-radius: 4px; padding: 0.8rem 1rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem; color: #656d76;
        max-height: 200px; overflow-y: auto;
        margin: 0.5rem 0;
    }

    .timing-badge {
        display: inline-block; background: #f6f8fa;
        border: 1px solid #d0d7de; border-radius: 3px;
        padding: 0.15rem 0.5rem; font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        color: #656d76; margin: 0.2rem 0.3rem 0.2rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


_logo_tag = f'<img src="data:image/png;base64,{LOGO_B64}" alt="PJ">' if LOGO_B64 else ""
st.markdown(f"""
<div class="app-header">
    {_logo_tag}
    <div>
        <h1>OCR + Tradução Multilingue → Português (PT-PT)</h1>
        <p>Pipeline offline  ·  {len(LINGUAS)} línguas  ·  Tesseract OCR  ·  NLLB-200</p>
    </div>
</div>
""", unsafe_allow_html=True)


_opcoes_lingua = {}
for script, items in sorted(SCRIPTS.items()):
    for chave, nome in items:
        _opcoes_lingua[f"{nome}  ({script})"] = chave

_lista_opcoes = list(_opcoes_lingua.keys())
_idx_default = next(
    (i for i, k in enumerate(_lista_opcoes) if _opcoes_lingua[k] == "bengali"), 0
)

with st.sidebar:
    st.header("Configurações")

    lingua_sel = st.selectbox(
        "Lingua de origem",
        _lista_opcoes,
        index=_idx_default,
        key="sel_lingua",
    )
    lingua_chave = _opcoes_lingua[lingua_sel]
    lingua_nome, lingua_nllb, lingua_tess, lingua_script, _ = \
        obter_lingua(lingua_chave)

    st.caption(f"NLLB: {lingua_nllb} / Tesseract: {lingua_tess}")

    _backend = obter_backend(lingua_chave)
    _modelo_nome = obter_nome_modelo(lingua_chave)
    if _backend.startswith("opus-mt"):
        st.success(f"Modelo especializado: {_modelo_nome}")
    else:
        st.caption(f"Modelo: {_modelo_nome}")

    auto_detectar = st.toggle(
        "Auto-detectar lingua",
        value=True,
        key="toggle_auto",
        help="Deteccao hibrida: Unicode + fasttext + langid",
    )

    pode_directo = lingua_chave in PARES_DIRECTOS
    usar_directo = st.toggle(
        "Traducao directa (sem pivot ingles)",
        value=False,
        key="toggle_directo",
        disabled=not pode_directo,
        help="Disponivel para: " + ", ".join(sorted(PARES_DIRECTOS))
              if pode_directo else "Nao disponivel para esta lingua",
    )

    st.divider()

    ocr_lang_override = st.text_input(
        "Override Tesseract lang",
        value="",
        key="ocr_override",
        placeholder=lingua_tess,
        help=f"Deixar vazio para usar '{lingua_tess}'. "
             f"Ex: rus+eng, ara+eng, chi_sim+eng",
    )
    ocr_lang = ocr_lang_override.strip() if ocr_lang_override.strip() else lingua_tess

    mostrar_ingles = st.toggle(
        "Mostrar traducao inglesa intermedia",
        value=True,
        key="toggle_eng",
    )

    mostrar_confianca = st.toggle(
        "Mostrar confianca OCR por palavra",
        value=False,
        key="toggle_conf",
    )

    mostrar_preprocessado = st.toggle(
        "Mostrar pipeline pre-processamento",
        value=False,
        key="toggle_preproc",
    )

    st.divider()

    device = get_device()
    info_lines = [f"Dispositivo: {device}"]
    if device == "cuda":
        import torch
        info_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    info_lines.append(f"OpenCV: {'Sim' if HAS_CV2 else 'Nao'}")
    info_lines.append(f"PDF: {'Sim' if HAS_PDF else 'Nao'}")
    st.caption(" | ".join(info_lines))

    carregar_cache()
    st.caption(f"Cache: {num_entradas_cache()} entradas")
    if st.button("Limpar cache", key="btn_cache"):
        limpar_cache()
        st.success("Cache limpa.")
        st.rerun()

    st.divider()
    st.caption("Policia Judiciaria — UNCT | Pipeline 100% offline")


def _render_step(steps):
    html = '<div class="step-indicator">'
    for i, s in enumerate(steps):
        cls = s.get("status", "")
        html += f'<span class="step-dot {cls}"></span>'
        html += f'<span class="step-label {cls}">{s["label"]}</span>'
        if i < len(steps) - 1:
            html += '<span class="step-sep">›</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _render_result_card(titulo, conteudo, tipo=""):
    safe = html_mod.escape(conteudo).replace("\n", "<br>")
    st.markdown(f"""
    <div class="result-card {tipo}">
        <h3>{titulo}</h3>
        <div class="content">{safe}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_timing(tempos: dict):
    html_parts = []
    for label, valor in tempos.items():
        html_parts.append(f'<span class="timing-badge">{label}: {valor:.1f}s</span>')
    st.markdown(" ".join(html_parts), unsafe_allow_html=True)


def _render_ocr_metricas(metricas: dict):
    conf = metricas.get("confianca_media", 0)
    pct_alta = metricas.get("pct_alta_confianca", 0)
    qualidade = metricas.get("qualidade", "desconhecida")
    total = metricas.get("total_palavras", 0)

    if conf >= OCR_QUALIDADE_EXCELENTE:
        cor = "#1a7f37"
        label = "Excelente"
    elif conf >= OCR_QUALIDADE_BOM:
        cor = "#2f81f7"
        label = "Bom"
    elif conf >= OCR_QUALIDADE_FRACO:
        cor = "#d29922"
        label = "Fraco"
    else:
        cor = "#cf222e"
        label = "Muito fraco"

    st.markdown(f"""
    <div class="stats-row">
        <span class="stat-pill" style="border-left:3px solid {cor};">
            Qualidade OCR: {label} ({conf:.0f}%)
        </span>
        <span class="stat-pill">{pct_alta:.0f}% palavras confiantes</span>
        <span class="stat-pill">{total} palavras</span>
    </div>
    """, unsafe_allow_html=True)


def _render_preprocessamento(img, script: str):
    try:
        _, estagios = preprocessar_imagem_pipeline(img, script=script, verbose=False)
        if estagios:
            with st.expander(f"Pipeline de pré-processamento ({len(estagios)} estágios)", expanded=False):
                for nome_estagio, img_estagio in estagios:
                    st.caption(nome_estagio)
                    st.image(img_estagio, use_container_width=True)
    except Exception as e:
        st.caption(f"Pré-processamento indisponível: {e}")


def _adicionar_historico(entrada: dict):
    entrada["timestamp"] = datetime.now().strftime("%H:%M:%S")
    entrada["data"] = datetime.now().strftime("%Y-%m-%d")
    st.session_state.historico.append(entrada)
    if len(st.session_state.historico) > 50:
        st.session_state.historico = st.session_state.historico[-50:]


def _processar_e_mostrar(texto_ocr, placeholder, lingua, log_container=None):
    texto_limpo = limpar_texto_ocr(texto_ocr)
    paragrafos = dividir_em_paragrafos(texto_limpo)

    if not paragrafos:
        st.warning("Nenhum parágrafo encontrado.")
        return None, None, None, {}

    logs = []
    def _log(msg):
        logs.append(msg)
        if log_container:
            log_container.markdown(
                '<div class="log-box">' + "<br>".join(logs) + '</div>',
                unsafe_allow_html=True)

    lingua_usada = lingua
    if auto_detectar:
        t0 = time.perf_counter()
        detectada, conf_det = detectar_lingua_auto(texto_limpo)
        dt = time.perf_counter() - t0
        if detectada and detectada in LINGUAS:
            lingua_usada = detectada
            nome_det = LINGUAS[detectada][0]
            conf_pct = conf_det * 100
            if conf_pct >= 70:
                st.success(f"Língua detectada: {nome_det} (confiança: {conf_pct:.0f}%)")
            elif conf_pct >= 40:
                st.info(f"Língua detectada: {nome_det} (confiança: {conf_pct:.0f}%)")
            else:
                st.warning(f"Língua detectada: {nome_det} (confiança baixa: {conf_pct:.0f}%) "
                           f"— considere seleccionar manualmente na barra lateral")
            _log(f"Detecção: {nome_det} ({conf_pct:.0f}%, {dt:.2f}s)")

    total = len(paragrafos)
    _log(f"Parágrafos: {total}")

    progress_bar = placeholder.progress(0, text="A preparar modelo...")
    t0 = time.perf_counter()
    carregar_modelo()
    t_modelo = time.perf_counter() - t0
    _log(f"Modelo carregado ({t_modelo:.1f}s)")

    progress_bar.progress(10, text=f"A traduzir 0/{total}...")

    def _callback(i, total):
        progress_bar.progress(
            10 + int((i / total) * 85),
            text=f"A traduzir {i}/{total}...")
        _log(f"Traduzido {i}/{total}")

    t1 = time.perf_counter()
    lista_en, lista_pt = traduzir_paragrafos(
        paragrafos, lingua=lingua_usada,
        directo=usar_directo,
        callback=_callback)
    t_trad = time.perf_counter() - t1

    guardar_cache()
    progress_bar.progress(100, text="Concluído")
    time.sleep(0.3)
    placeholder.empty()

    tempos = {"Modelo": t_modelo, "Tradução": t_trad, "Total": t_modelo + t_trad}
    _log(f"Concluído em {t_modelo + t_trad:.1f}s")

    texto_en = "\n\n".join(lista_en)
    texto_pt = "\n\n".join(lista_pt)

    return texto_limpo, texto_en, texto_pt, tempos


def _render_exportacao(dados: dict, prefixo: str, key_suffix: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.download_button(
            "Download TXT", data=dados.get("texto_pt", ""),
            file_name=f"{prefixo}_pt_{ts}.txt", mime="text/plain",
            width='stretch', key=f"dl_txt_{key_suffix}")

    with col2:
        try:
            from src.exportar import exportar_docx
            docx_bytes = exportar_docx(dados)
            st.download_button(
                "Download DOCX", data=docx_bytes,
                file_name=f"{prefixo}_relatorio_{ts}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                width='stretch', key=f"dl_docx_{key_suffix}")
        except ImportError:
            st.caption("python-docx não instalado")

    with col3:
        try:
            from src.exportar import exportar_excel
            xlsx_bytes = exportar_excel(dados)
            st.download_button(
                "Download Excel", data=xlsx_bytes,
                file_name=f"{prefixo}_dados_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch', key=f"dl_xlsx_{key_suffix}")
        except ImportError:
            st.caption("openpyxl não instalado")

    with col4:
        try:
            from src.exportar import exportar_pdf
            pdf_bytes = exportar_pdf(dados)
            st.download_button(
                "Download PDF", data=pdf_bytes,
                file_name=f"{prefixo}_relatorio_{ts}.pdf",
                mime="application/pdf",
                width='stretch', key=f"dl_pdf_{key_suffix}")
        except ImportError:
            st.caption("reportlab não instalado")


tab_imagem, tab_ocr, tab_detectar, tab_texto, tab_lote, tab_historico, tab_linguas, tab_sobre = st.tabs([
    "Imagem + Traducao",
    "Apenas OCR",
    "Detectar lingua",
    "Texto directo",
    "Processamento em lote",
    "Historico",
    "Linguas suportadas",
    "Documentacao",
])


with tab_imagem:
    tipos = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
    if HAS_PDF:
        tipos.append("pdf")

    ficheiros = st.file_uploader(
        "Imagens ou PDF",
        type=tipos,
        accept_multiple_files=True,
        key="uploader_img",
    )

    processar_imgs = st.button(
        "Processar", type="primary",
        disabled=not ficheiros,
        key="btn_proc_imgs",
    )

    if ficheiros and processar_imgs:
        for ficheiro in ficheiros:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown(f"### {ficheiro.name}")

            is_pdf = ficheiro.name.lower().endswith(".pdf")

            if is_pdf and HAS_PDF:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(ficheiro.read())
                    tmp_path = tmp.name

                try:
                    imagens_pdf = extrair_paginas_pdf(tmp_path)
                    st.caption(f"{len(imagens_pdf)} página(s)")

                    for pg_idx, img_pg in enumerate(imagens_pdf):
                        st.markdown(f"**Página {pg_idx + 1}**")
                        col_img, col_res = st.columns([1, 1.5])

                        with col_img:
                            st.image(img_pg, caption=f"Página {pg_idx+1}",
                                     use_container_width=True)
                            if mostrar_preprocessado:
                                _render_preprocessamento(img_pg.copy(), lingua_script)

                        with col_res:
                            t0_ocr = time.perf_counter()
                            with st.spinner("OCR..."):
                                texto_ocr = extrair_texto(img_pg, lang=ocr_lang)
                            t_ocr = time.perf_counter() - t0_ocr

                            if not texto_ocr.strip():
                                st.warning("Sem texto nesta página.")
                                continue

                            st.markdown('<p class="section-label">Texto OCR</p>',
                                        unsafe_allow_html=True)
                            st.code(texto_ocr, language=None)
                            _render_timing({"OCR": t_ocr})

                            try:
                                metricas = calcular_metricas_ocr(img_pg, lang=ocr_lang,
                                                                  script=lingua_script)
                                _render_ocr_metricas(metricas)
                            except Exception:
                                pass

                            if mostrar_confianca:
                                conf_data = extrair_confianca(img_pg, lang=ocr_lang)
                                if conf_data:
                                    import pandas as pd
                                    df = pd.DataFrame(conf_data)
                                    st.dataframe(df, use_container_width=True)

                            log_cont = st.empty()
                            ph = st.empty()
                            try:
                                tl, te, tp, tempos = _processar_e_mostrar(
                                    texto_ocr, ph, lingua_chave, log_cont)
                            except Exception as e:
                                st.error(f"Erro: {e}")
                                tp = None

                            if tp:
                                tempos["OCR"] = t_ocr
                                tempos["Total"] = t_ocr + tempos.get("Tradução", 0)
                                _render_timing(tempos)

                                if mostrar_ingles and te:
                                    _render_result_card("Inglês (intermédio)", te, "english")
                                _render_result_card("Português (PT-PT)", tp, "portuguese")

                                tp_editado = st.text_area(
                                    "Editar resultado (PT-PT):",
                                    value=tp, height=150,
                                    key=f"edit_pdf_{pg_idx}")

                                paragrafos_orig = dividir_em_paragrafos(limpar_texto_ocr(texto_ocr))
                                dados_export = {
                                    "ficheiro": ficheiro.name,
                                    "lingua": lingua_nome,
                                    "modelo": get_modelo_usado() or "N/A",
                                    "texto_ocr": texto_ocr,
                                    "texto_en": te,
                                    "texto_pt": tp_editado,
                                    "paragrafos_orig": paragrafos_orig,
                                    "paragrafos_en": te.split("\n\n") if te else [],
                                    "paragrafos_pt": tp_editado.split("\n\n"),
                                }
                                _render_exportacao(dados_export,
                                                   Path(ficheiro.name).stem,
                                                   f"pdf_{pg_idx}")

                                _adicionar_historico({
                                    "tipo": "imagem+traducao",
                                    "ficheiro": ficheiro.name,
                                    "pagina": pg_idx + 1,
                                    "lingua": lingua_nome,
                                    "modelo": get_modelo_usado() or "N/A",
                                    "texto_ocr": texto_ocr,
                                    "texto_en": te,
                                    "texto_pt": tp_editado,
                                    "tempos": tempos,
                                })
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            else:
                img = Image.open(ficheiro)

                col_img_area, col_res = st.columns([1, 1.5])

                with col_img_area:
                    st.image(img, caption=ficheiro.name,
                             use_container_width=True)
                    if mostrar_preprocessado:
                        _render_preprocessamento(img.copy(), lingua_script)

                with col_res:
                    _render_step([
                        {"label": "Imagem", "status": "done"},
                        {"label": "OCR", "status": "active"},
                        {"label": f"{lingua_chave} → eng", "status": ""},
                        {"label": "eng → por", "status": ""},
                        {"label": "PT-PT", "status": ""},
                    ])

                    t0_ocr = time.perf_counter()
                    with st.spinner("OCR..."):
                        tmp_path = os.path.join(
                            tempfile.gettempdir(),
                            f"ocr_{hash(ficheiro.name)}{Path(ficheiro.name).suffix}")
                        try:
                            img.save(tmp_path)
                            texto_ocr = extrair_texto(tmp_path, lang=ocr_lang)
                        except Exception as e:
                            st.error(f"Erro OCR: {e}")
                            texto_ocr = ""
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                    t_ocr = time.perf_counter() - t0_ocr

                    if not texto_ocr.strip():
                        st.warning("Sem texto extraído.")
                        continue

                    st.markdown('<p class="section-label">Texto OCR extraído</p>',
                                unsafe_allow_html=True)
                    st.code(texto_ocr, language=None)
                    _render_timing({"OCR": t_ocr})

                    try:
                        img_metricas = Image.open(ficheiro)
                        metricas = calcular_metricas_ocr(img_metricas, lang=ocr_lang,
                                                          script=lingua_script)
                        _render_ocr_metricas(metricas)
                    except Exception:
                        pass

                    if mostrar_confianca:
                        try:
                            img_tmp = Image.open(ficheiro)
                            conf_data = extrair_confianca(img_tmp, lang=ocr_lang)
                            if conf_data:
                                import pandas as pd
                                df = pd.DataFrame(conf_data)
                                st.dataframe(df, use_container_width=True)
                        except Exception:
                            pass

                    _render_step([
                        {"label": "Imagem", "status": "done"},
                        {"label": "OCR", "status": "done"},
                        {"label": f"{lingua_chave} → eng", "status": "active"},
                        {"label": "eng → por", "status": ""},
                        {"label": "PT-PT", "status": ""},
                    ])

                    log_cont = st.empty()
                    ph = st.empty()
                    try:
                        texto_limpo, texto_en, texto_pt, tempos = \
                            _processar_e_mostrar(texto_ocr, ph, lingua_chave, log_cont)
                    except Exception as e:
                        st.error(f"Erro: {e}")
                        texto_pt = None

                    if texto_pt:
                        tempos["OCR"] = t_ocr
                        tempos["Total"] = t_ocr + tempos.get("Tradução", 0)

                        _render_step([
                            {"label": "Imagem", "status": "done"},
                            {"label": "OCR", "status": "done"},
                            {"label": f"{lingua_chave} → eng", "status": "done"},
                            {"label": "eng → por", "status": "done"},
                            {"label": "PT-PT", "status": "done"},
                        ])
                        _render_timing(tempos)

                        if mostrar_ingles and texto_en:
                            col_en, col_pt = st.columns(2)
                            with col_en:
                                _render_result_card("Inglês (passo intermédio)", texto_en, "english")
                            with col_pt:
                                _render_result_card("Português (PT-PT) — Resultado", texto_pt, "portuguese")
                        else:
                            _render_result_card("Português (PT-PT) — Resultado", texto_pt, "portuguese")

                        st.markdown(f"""
                        <div class="stats-row">
                            <span class="stat-pill">Língua: {lingua_nome}</span>
                            <span class="stat-pill">OCR: {len(texto_ocr)} chars</span>
                            <span class="stat-pill">Modelo: {get_modelo_usado() or 'N/A'}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        texto_pt_editado = st.text_area(
                            "Editar resultado (PT-PT):",
                            value=texto_pt, height=150,
                            key=f"edit_img_{ficheiro.name}")

                        st.markdown("-")

                        paragrafos_orig = dividir_em_paragrafos(texto_limpo) if texto_limpo else []
                        dados_export = {
                            "ficheiro": ficheiro.name,
                            "lingua": lingua_nome,
                            "modelo": get_modelo_usado() or "N/A",
                            "texto_ocr": texto_ocr,
                            "texto_en": texto_en,
                            "texto_pt": texto_pt_editado,
                            "paragrafos_orig": paragrafos_orig,
                            "paragrafos_en": texto_en.split("\n\n") if texto_en else [],
                            "paragrafos_pt": texto_pt_editado.split("\n\n"),
                        }
                        _render_exportacao(dados_export,
                                           Path(ficheiro.name).stem,
                                           ficheiro.name)

                        _adicionar_historico({
                            "tipo": "imagem+traducao",
                            "ficheiro": ficheiro.name,
                            "lingua": lingua_nome,
                            "modelo": get_modelo_usado() or "N/A",
                            "texto_ocr": texto_ocr,
                            "texto_en": texto_en,
                            "texto_pt": texto_pt_editado,
                            "tempos": tempos,
                        })


with tab_ocr:
    st.caption("Extrair texto de imagens ou PDF sem traduzir. "
               "Pode seleccionar e copiar partes do texto extraido.")

    tipos_ocr = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"]
    if HAS_PDF:
        tipos_ocr.append("pdf")

    ficheiros_ocr = st.file_uploader(
        "Imagens ou PDF",
        type=tipos_ocr,
        accept_multiple_files=True,
        key="uploader_ocr_only",
    )

    col_ocr_opts1, col_ocr_opts2 = st.columns(2)
    with col_ocr_opts1:
        ocr_only_lang = st.text_input(
            "Tesseract lang",
            value=ocr_lang,
            key="ocr_only_lang",
            help="Ex: ara+eng, rus+eng, chi_sim+eng, ben+eng",
        )
    with col_ocr_opts2:
        ocr_only_limpar = st.toggle(
            "Limpar texto OCR (remover lixo/normalizar)",
            value=True,
            key="ocr_only_limpar",
        )

    btn_ocr_only = st.button(
        "Extrair texto",
        type="primary",
        disabled=not ficheiros_ocr,
        key="btn_ocr_only",
    )

    if ficheiros_ocr and btn_ocr_only:
        for ficheiro_ocr in ficheiros_ocr:
            st.markdown("-")
            st.markdown(f"### {ficheiro_ocr.name}")

            is_pdf_ocr = ficheiro_ocr.name.lower().endswith(".pdf")

            imagens_para_ocr = []

            if is_pdf_ocr and HAS_PDF:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    tmp.write(ficheiro_ocr.read())
                    tmp_path_ocr = tmp.name
                try:
                    paginas = extrair_paginas_pdf(tmp_path_ocr)
                    for pg_i, pg_img in enumerate(paginas):
                        imagens_para_ocr.append((f"Pagina {pg_i + 1}", pg_img))
                finally:
                    try:
                        os.unlink(tmp_path_ocr)
                    except OSError:
                        pass
            else:
                img_ocr = Image.open(ficheiro_ocr)
                imagens_para_ocr.append((ficheiro_ocr.name, img_ocr))

            textos_extraidos = []

            for label_ocr, img_para_ocr in imagens_para_ocr:
                if len(imagens_para_ocr) > 1:
                    st.markdown(f"**{label_ocr}**")

                col_img_ocr, col_txt_ocr = st.columns([1, 2])

                with col_img_ocr:
                    st.image(img_para_ocr, caption=label_ocr,
                             use_container_width=True)

                    if mostrar_preprocessado:
                        _render_preprocessamento(img_para_ocr.copy(), lingua_script)

                with col_txt_ocr:
                    t0_ocr_only = time.perf_counter()
                    with st.spinner("A extrair texto..."):
                        tmp_ocr_path = os.path.join(
                            tempfile.gettempdir(),
                            f"ocr_only_{hash(label_ocr)}.png")
                        try:
                            img_para_ocr.save(tmp_ocr_path)
                            texto_ocr_raw = extrair_texto(
                                tmp_ocr_path,
                                lang=ocr_only_lang.strip() or "eng",
                                script=lingua_script,
                            )
                        except Exception as e:
                            st.error(f"Erro OCR: {e}")
                            texto_ocr_raw = ""
                        finally:
                            try:
                                os.unlink(tmp_ocr_path)
                            except OSError:
                                pass
                    dt_ocr_only = time.perf_counter() - t0_ocr_only

                    if not texto_ocr_raw.strip():
                        st.warning("Sem texto extraido.")
                        continue

                    if ocr_only_limpar:
                        texto_final_ocr = limpar_texto_ocr(texto_ocr_raw)
                    else:
                        texto_final_ocr = texto_ocr_raw

                    textos_extraidos.append((label_ocr, texto_final_ocr))

                    _render_timing({"OCR": dt_ocr_only})
                    try:
                        img_m = Image.open(ficheiro_ocr) if not is_pdf_ocr else img_para_ocr
                        metricas = calcular_metricas_ocr(
                            img_m,
                            lang=ocr_only_lang.strip() or "eng",
                            script=lingua_script)
                        _render_ocr_metricas(metricas)
                    except Exception:
                        pass

                    st.text_area(
                        "Texto extraido (seleccione e copie o que precisar):",
                        value=texto_final_ocr,
                        height=300,
                        key=f"ocr_only_txt_{ficheiro_ocr.name}_{label_ocr}",
                    )

                    if mostrar_confianca:
                        try:
                            conf_data = extrair_confianca(
                                img_para_ocr,
                                lang=ocr_only_lang.strip() or "eng")
                            if conf_data:
                                import pandas as pd
                                df = pd.DataFrame(conf_data)
                                st.dataframe(df, use_container_width=True)
                        except Exception:
                            pass

            if textos_extraidos:
                st.markdown("-")
                texto_completo = "\n\n".join(
                    f"- {lbl} -\n{txt}" if len(textos_extraidos) > 1 else txt
                    for lbl, txt in textos_extraidos
                )
                ts_ocr = datetime.now().strftime("%Y%m%d_%H%M%S")

                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "Download texto completo (TXT)",
                        data=texto_completo,
                        file_name=f"ocr_{Path(ficheiro_ocr.name).stem}_{ts_ocr}.txt",
                        mime="text/plain",
                        key=f"dl_ocr_only_{ficheiro_ocr.name}",
                    )
                with col_dl2:
                    st.caption(f"{len(textos_extraidos)} bloco(s), "
                               f"{sum(len(t) for _, t in textos_extraidos)} caracteres")


with tab_detectar:
    st.caption("Identificar a lingua de um texto ou imagem. "
               "Usa deteccao hibrida: Unicode + fasttext + langid.")

    modo_det = st.radio(
        "Fonte do texto",
        ["Colar texto", "Imagem (OCR + deteccao)"],
        horizontal=True,
        key="radio_modo_det",
    )

    if modo_det == "Colar texto":
        with st.form("form_detectar_texto", clear_on_submit=False):
            texto_det = st.text_area(
                "Texto para identificar",
                height=200,
                placeholder="Colar texto em qualquer lingua...",
                key="ta_detectar",
            )
            btn_detectar = st.form_submit_button(
                "Detectar lingua", type="primary", use_container_width=True)

        if btn_detectar and texto_det.strip():
            t0_det = time.perf_counter()
            lingua_det, conf_det = detectar_lingua_auto(texto_det.strip())
            dt_det = time.perf_counter() - t0_det

            if lingua_det and lingua_det in LINGUAS:
                nome_det, nllb_det, tess_det, script_det, _ = LINGUAS[lingua_det]
                conf_pct = conf_det * 100

                if conf_pct >= 70:
                    st.success(f"Lingua detectada: **{nome_det}** (confianca: {conf_pct:.0f}%)")
                elif conf_pct >= 40:
                    st.info(f"Lingua detectada: **{nome_det}** (confianca: {conf_pct:.0f}%)")
                else:
                    st.warning(f"Lingua detectada: **{nome_det}** (confianca baixa: {conf_pct:.0f}%)")

                st.markdown(f"""
                <div class="stats-row">
                    <span class="stat-pill">Chave: {lingua_det}</span>
                    <span class="stat-pill">Script: {script_det}</span>
                    <span class="stat-pill">NLLB: {nllb_det}</span>
                    <span class="stat-pill">Tesseract: {tess_det}</span>
                    <span class="stat-pill">Tempo: {dt_det:.3f}s</span>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Detalhes da deteccao (3 camadas)"):
                    from src.utils import _detectar_unicode, _detectar_fasttext, _detectar_langid

                    unicode_res = _detectar_unicode(texto_det.strip())
                    if unicode_res:
                        st.markdown("**Camada 1 — Unicode ranges:**")
                        for ling, score in unicode_res[:5]:
                            nome_u = LINGUAS[ling][0] if ling in LINGUAS else ling
                            bar_pct = min(100, int(score * 100))
                            st.progress(bar_pct / 100, text=f"{nome_u} ({ling}): {score:.2f}")
                    else:
                        st.caption("Unicode: sem resultados")

                    ft_ling, ft_conf = _detectar_fasttext(texto_det.strip())
                    if ft_ling:
                        nome_ft = LINGUAS[ft_ling][0] if ft_ling in LINGUAS else ft_ling
                        st.markdown(f"**Camada 2 — fasttext:** {nome_ft} ({ft_ling}) — confianca: {ft_conf:.2f}")
                    else:
                        st.caption("fasttext: indisponivel ou sem resultado")

                    lid_ling, lid_conf = _detectar_langid(texto_det.strip())
                    if lid_ling:
                        nome_lid = LINGUAS[lid_ling][0] if lid_ling in LINGUAS else lid_ling
                        st.markdown(f"**Camada 3 — langid:** {nome_lid} ({lid_ling}) — confianca: {lid_conf:.2f}")
                    else:
                        st.caption("langid: indisponivel ou sem resultado")

            else:
                st.error("Nao foi possivel identificar a lingua.")
                st.caption(f"Tempo: {dt_det:.3f}s")

    else:
        ficheiro_det = st.file_uploader(
            "Imagem para detectar lingua",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            key="uploader_det_img",
        )

        det_tess_lang = st.text_input(
            "Tesseract lang para OCR inicial",
            value="eng",
            key="det_tess_lang",
            help="Use 'osd' para deteccao de script, ou lang packs como 'ara+eng'",
        )

        btn_det_img = st.button(
            "Detectar lingua da imagem",
            type="primary",
            disabled=not ficheiro_det,
            key="btn_det_img",
        )

        if ficheiro_det and btn_det_img:
            img_det = Image.open(ficheiro_det)

            col_img_det, col_res_det = st.columns([1, 2])

            with col_img_det:
                st.image(img_det, caption=ficheiro_det.name,
                         use_container_width=True)

                with st.spinner("A detectar script visual (OSD)..."):
                    script_osd = detectar_script_osd(img_det)
                if script_osd:
                    st.info(f"Script visual (Tesseract OSD): **{script_osd}**")
                else:
                    st.caption("OSD: script nao detectado")

            with col_res_det:
                t0_ocr_det = time.perf_counter()
                with st.spinner("A extrair texto (OCR)..."):
                    tmp_det_path = os.path.join(
                        tempfile.gettempdir(),
                        f"det_{hash(ficheiro_det.name)}.png")
                    try:
                        img_det.save(tmp_det_path)
                        texto_ocr_det = extrair_texto(
                            tmp_det_path,
                            lang=det_tess_lang.strip() or "eng",
                        )
                    except Exception as e:
                        st.error(f"Erro OCR: {e}")
                        texto_ocr_det = ""
                    finally:
                        try:
                            os.unlink(tmp_det_path)
                        except OSError:
                            pass
                dt_ocr_det = time.perf_counter() - t0_ocr_det

                if not texto_ocr_det.strip():
                    st.warning("OCR nao extraiu texto. Tente outro Tesseract lang.")
                else:
                    st.text_area(
                        "Texto extraido pelo OCR:",
                        value=texto_ocr_det,
                        height=150,
                        key="det_ocr_texto",
                    )
                    _render_timing({"OCR": dt_ocr_det})

                    t0_det2 = time.perf_counter()
                    lingua_det2, conf_det2 = detectar_lingua_auto(texto_ocr_det.strip())
                    dt_det2 = time.perf_counter() - t0_det2

                    if lingua_det2 and lingua_det2 in LINGUAS:
                        nome_det2, nllb_det2, tess_det2, script_det2, _ = LINGUAS[lingua_det2]
                        conf_pct2 = conf_det2 * 100

                        if conf_pct2 >= 70:
                            st.success(f"Lingua detectada: **{nome_det2}** (confianca: {conf_pct2:.0f}%)")
                        elif conf_pct2 >= 40:
                            st.info(f"Lingua detectada: **{nome_det2}** (confianca: {conf_pct2:.0f}%)")
                        else:
                            st.warning(f"Lingua detectada: **{nome_det2}** (confianca baixa: {conf_pct2:.0f}%)")

                        st.markdown(f"""
                        <div class="stats-row">
                            <span class="stat-pill">Chave: {lingua_det2}</span>
                            <span class="stat-pill">Script: {script_det2}</span>
                            <span class="stat-pill">NLLB: {nllb_det2}</span>
                            <span class="stat-pill">Tesseract recomendado: {tess_det2}</span>
                            <span class="stat-pill">Deteccao: {dt_det2:.3f}s</span>
                        </div>
                        """, unsafe_allow_html=True)

                        if tess_det2 != det_tess_lang.strip():
                            st.info(f"Sugestao: refazer OCR com `{tess_det2}` para melhores resultados.")
                    else:
                        st.error("Nao foi possivel identificar a lingua do texto extraido.")


with tab_texto:
    st.caption(f"Traduzir texto directamente (sem OCR). Língua: {lingua_nome}")

    with st.form("form_texto", clear_on_submit=False):
        texto_input = st.text_area(
            "Texto de origem",
            height=250,
            placeholder="Colar texto aqui...",
            key="ta_texto",
        )
        btn_traduzir = st.form_submit_button(
            "Traduzir", type="primary", use_container_width=True)

    if btn_traduzir and texto_input.strip():
            log_cont = st.empty()
            ph = st.empty()
            try:
                tl, te, tp, tempos = _processar_e_mostrar(
                    texto_input, ph, lingua_chave, log_cont)
            except Exception as e:
                st.error(f"Erro: {e}")
                tp = None

            if tp:
                _render_timing(tempos)

                col1, col2 = st.columns(2)
                with col1:
                    if mostrar_ingles and te:
                        _render_result_card("Inglês", te, "english")
                with col2:
                    _render_result_card("Português (PT-PT)", tp, "portuguese")

                tp_editado = st.text_area(
                    "Editar resultado (PT-PT):", value=tp,
                    height=150, key="edit_texto")

                dados_export = {
                    "ficheiro": "texto_directo",
                    "lingua": lingua_nome,
                    "modelo": get_modelo_usado() or "N/A",
                    "texto_ocr": texto_input,
                    "texto_en": te,
                    "texto_pt": tp_editado,
                    "paragrafos_orig": dividir_em_paragrafos(tl) if tl else [],
                    "paragrafos_en": te.split("\n\n") if te else [],
                    "paragrafos_pt": tp_editado.split("\n\n"),
                }
                _render_exportacao(dados_export, "texto", "texto")

                _adicionar_historico({
                    "tipo": "texto_directo",
                    "ficheiro": "Texto directo",
                    "lingua": lingua_nome,
                    "modelo": get_modelo_usado() or "N/A",
                    "texto_ocr": texto_input,
                    "texto_en": te,
                    "texto_pt": tp_editado,
                    "tempos": tempos,
                })


with tab_lote:
    st.caption(f"Processar pasta. Língua: {lingua_nome}")

    col_path, col_browse = st.columns([4, 1])
    with col_browse:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Procurar...", key="btn_browse_pasta", width='stretch'):
            try:
                import tkinter as tk
                from tkinter import filedialog
                _root = tk.Tk()
                _root.withdraw()
                _root.attributes("-topmost", True)
                _pasta_sel = filedialog.askdirectory(
                    title="Seleccionar pasta com imagens",
                    parent=_root)
                _root.destroy()
                if _pasta_sel:
                    st.session_state["input_pasta"] = _pasta_sel
                    st.rerun()
            except Exception as e:
                st.error(f"Erro ao abrir diálogo: {e}")
    with col_path:
        caminho_pasta = st.text_input(
            "Caminho da pasta",
            placeholder=r"D:\casos\caso_123\imagens",
            key="input_pasta",
        )

    pasta_final = caminho_pasta.strip()

    if st.button("Processar pasta", type="primary",
                  disabled=not pasta_final, key="btn_lote"):
        pasta = Path(pasta_final)
        if not pasta.is_dir():
            st.error(f"Pasta não encontrada: {pasta}")
        else:
            extensoes = EXTENSOES_IMAGEM
            if HAS_PDF:
                extensoes = extensoes | {".pdf"}
            flist = sorted(
                f for f in pasta.glob("*") if f.suffix.lower() in extensoes)

            if not flist:
                st.warning("Nenhum ficheiro suportado.")
            else:
                st.caption(f"{len(flist)} ficheiros")
                resultados = {}
                prog = st.progress(0, text="A iniciar...")

                _, nllb, _, _, _ = obter_lingua(lingua_chave)

                for fi, fp in enumerate(flist):
                    prog.progress(int((fi / len(flist)) * 100),
                                  text=f"{fp.name} ({fi+1}/{len(flist)})")
                    try:
                        t0 = time.perf_counter()
                        ocr = extrair_texto(str(fp), lang=ocr_lang)
                        if ocr.strip():
                            limpo = limpar_texto_ocr(ocr)
                            paras = dividir_em_paragrafos(limpo)
                            _, lista_pt = traduzir_paragrafos(
                                paras, lingua=lingua_chave, directo=usar_directo)
                            resultados[fp.name] = "\n\n".join(lista_pt)
                        dt = time.perf_counter() - t0
                    except Exception as e:
                        resultados[fp.name] = f"[erro] {e}"

                prog.progress(100, text="Concluído")
                guardar_cache()

                for nome, texto in resultados.items():
                    with st.expander(nome, expanded=True):
                        _render_result_card("Português (PT-PT)", texto, "portuguese")

                rel = "\n\n".join(f"# {n}\n{t}" for n, t in resultados.items())
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("Download completo", data=rel,
                    file_name=f"lote_{pasta.name}_{ts}.txt",
                    mime="text/plain", width='stretch', key="dl_lote")


with tab_historico:
    st.caption(f"Ultimas {len(st.session_state.historico)} operacoes desta sessao")

    if not st.session_state.historico:
        st.info("Nenhuma operacao realizada nesta sessao.")
    else:
        col_hist_actions = st.columns([1, 1, 3])
        with col_hist_actions[0]:
            if st.button("Limpar historico", key="btn_limpar_hist"):
                st.session_state.historico.clear()
                st.rerun()
        with col_hist_actions[1]:
            _hist_completo = []
            for _hitem in reversed(st.session_state.historico):
                _hbloco = [f"{'='*60}",
                           f"[{_hitem.get('data','')} {_hitem.get('timestamp','')}] "
                           f"{_hitem.get('ficheiro','N/A')} — {_hitem.get('lingua','')}",
                           f"{'='*60}"]
                if _hitem.get("texto_ocr"):
                    _hbloco.append(f"\n- TEXTO ORIGINAL / OCR -\n{_hitem['texto_ocr']}")
                if _hitem.get("texto_en"):
                    _hbloco.append(f"\n- INGLES (INTERMEDIO) -\n{_hitem['texto_en']}")
                if _hitem.get("texto_pt"):
                    _hbloco.append(f"\n- PORTUGUES (PT-PT) -\n{_hitem['texto_pt']}")
                _hbloco.append("")
                _hist_completo.append("\n".join(_hbloco))
            _hist_txt = "\n\n".join(_hist_completo)
            _hist_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download historico (TXT)",
                data=_hist_txt,
                file_name=f"historico_{_hist_ts}.txt",
                mime="text/plain",
                key="dl_historico_completo",
            )

        for i, item in enumerate(reversed(st.session_state.historico)):
            tipo = item.get("tipo", "")
            pagina = item.get("pagina")
            titulo_pag = f" — Pag. {pagina}" if pagina else ""
            titulo = (f"{item.get('timestamp', '')} | "
                      f"{item.get('ficheiro', 'N/A')}{titulo_pag} | "
                      f"{item.get('lingua', '')}")

            with st.expander(titulo, expanded=(i == 0)):
                tempos = item.get("tempos", {})
                if tempos:
                    _render_timing(tempos)

                meta_parts = []
                if item.get("lingua"):
                    meta_parts.append(f"Lingua: {item['lingua']}")
                if item.get("modelo"):
                    meta_parts.append(f"Modelo: {item['modelo']}")
                if item.get("tipo"):
                    meta_parts.append(f"Tipo: {item['tipo']}")
                if meta_parts:
                    st.caption(" | ".join(meta_parts))

                texto_ocr_hist = item.get("texto_ocr", "")
                if texto_ocr_hist:
                    st.markdown("**Texto original / OCR:**")
                    st.text_area(
                        "Original",
                        value=texto_ocr_hist,
                        height=150,
                        key=f"hist_ocr_{i}",
                        label_visibility="collapsed",
                    )

                texto_en_hist = item.get("texto_en", "")
                if texto_en_hist and texto_en_hist not in ("[cache]", "[directo]"):
                    st.markdown("**Ingles (passo intermedio):**")
                    st.text_area(
                        "Ingles",
                        value=texto_en_hist,
                        height=120,
                        key=f"hist_en_{i}",
                        label_visibility="collapsed",
                    )

                texto_pt_hist = item.get("texto_pt", "")
                if texto_pt_hist:
                    st.markdown("**Portugues (PT-PT) — Resultado:**")
                    st.text_area(
                        "Portugues",
                        value=texto_pt_hist,
                        height=150,
                        key=f"hist_pt_{i}",
                        label_visibility="collapsed",
                    )

                if texto_pt_hist:
                    _dl_nome = Path(item.get("ficheiro", "resultado")).stem
                    _dl_ts = item.get("timestamp", "").replace(":", "")
                    _dl_conteudo = ""
                    if texto_ocr_hist:
                        _dl_conteudo += f"- TEXTO ORIGINAL / OCR -\n{texto_ocr_hist}\n\n"
                    if texto_en_hist and texto_en_hist not in ("[cache]", "[directo]"):
                        _dl_conteudo += f"- INGLES -\n{texto_en_hist}\n\n"
                    _dl_conteudo += f"- PORTUGUES (PT-PT) -\n{texto_pt_hist}\n"
                    st.download_button(
                        "Download este resultado (TXT)",
                        data=_dl_conteudo,
                        file_name=f"{_dl_nome}_{_dl_ts}.txt",
                        mime="text/plain",
                        key=f"dl_hist_{i}",
                    )


with tab_linguas:
    st.markdown("### Línguas suportadas")
    st.caption(f"{len(LINGUAS)} línguas disponíveis")

    for script, items in sorted(SCRIPTS.items()):
        st.markdown(f"**{script.capitalize()}**")
        cols = st.columns(4)
        for idx, (chave, nome) in enumerate(sorted(items, key=lambda x: x[1])):
            _, nllb, tess, _, _ = LINGUAS[chave]
            with cols[idx % 4]:
                directo_tag = " [directo]" if chave in PARES_DIRECTOS else ""
                st.markdown(f"`{chave}` — {nome}{directo_tag}")


with tab_sobre:
    col_info, col_arch = st.columns(2)

    with col_info:
        st.markdown(f"""
        ### Pipeline

        Tradução multilingue para português europeu (PT-PT),
        completamente offline. Suporta {len(LINGUAS)} línguas.

        **Fluxo:**
        1. Pré-processamento de imagem (OpenCV / PIL)
        2. OCR via Tesseract (LSTM, OEM 3)
        3. Limpeza e normalização Unicode
        4. Tradução: Origem → Inglês → Português (NLLB-200, 2 passos)
           ou tradução directa para línguas europeias
        5. Pós-processamento PT-BR → PT-PT (150+ regras)

        **Performance:**
        - Batch translation (múltiplos parágrafos por forward pass)
        - Quantização INT8 em CPU
        - float16 + torch.compile() em GPU
        - Cache inteligente de traduções
        - Segmentação automática de textos longos

        **Exportação:**
        DOCX (relatório PJ), Excel (tabela bilingue), PDF, TXT
        """)

    with col_arch:
        st.markdown("""
        ### Componentes

        | Componente | Detalhe |
        |---|---|
        | Tesseract OCR | Engine LSTM (OEM 3) |
        | NLLB-200 | 600M / 1.3B (auto) |
        | Pipeline | 2 passos ou directo |
        | Pós-processamento | 150+ regras PT-PT |
        | Batch | Até 8 parágrafos/batch |
        | Quantização | INT8 CPU / FP16 GPU |

        ### Instalação

        ```
        python -m pip install -r requirements.txt
        python -m streamlit run app.py
        ```

        ### CLI

        ```
        python cli.py img.jpg --lingua russo
        python cli.py img.jpg --auto
        python cli.py img.jpg --lingua espanhol --directo
        python cli.py --listar-linguas
        python cli.py --info
        ```
        """)

    st.markdown("-")
    st.markdown(f"""
    <div style="text-align:center; padding:0.8rem;">
        <span class="small-text">
            OCR + Tradução Multilingue · {len(LINGUAS)} línguas ·
            Offline · NLLB-200 · Tesseract · PJ — UNCT
        </span>
    </div>
    """, unsafe_allow_html=True)
