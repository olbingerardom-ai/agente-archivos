"""
Agente de análisis de archivos con chat, tablas y gráficas (Streamlit)
---------------------------------------------------------------------

▶ Qué hace
- Cargas 1..N archivos (CSV, XLSX, JSON, TXT, PDF).
- Previsualiza tablas, limpia tipos y detecta columnas.
- Preguntas en lenguaje natural; responde citando fuentes (archivo/página/hoja).
- Pide gráficas/tablas/pivots y las genera con matplotlib/pandas.
- Exporta resultados (CSV/PNG) y mantiene el historial de chat en la sesión.

▶ Requisitos (local)
pip install streamlit pandas numpy openpyxl pypdf faiss-cpu matplotlib tiktoken
pip install --upgrade openai

▶ Ejecutar (local)
- Define la variable de entorno OPENAI_API_KEY
- streamlit run app.py

Este proyecto está listo para desplegarse en Streamlit Cloud.
"""
import io
import os
import re
import json
import uuid
import typing as t
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Dependencias opcionales ===
try:
    from pypdf import PdfReader  # extracción de texto PDF
except Exception:
    PdfReader = None

try:
    import faiss  # indexación semántica
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    import tiktoken  # conteo de tokens (opcional)
except Exception:
    tiktoken = None

from openai import OpenAI

# ---------------------- Utilidades ----------------------
@dataclass
class Chunk:
    doc_id: str
    source: str  # nombre de archivo o hoja
    location: str  # p.ej. "p.3" o "hoja: Ventas"
    text: str


def _bytes_to_df(file) -> t.List[pd.DataFrame]:
    name = file.name.lower()
    data = file.read()
    dfs: t.List[pd.DataFrame] = []
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
        dfs.append(df)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(data))
        for sheet in xls.sheet_names:
            dfs.append(xls.parse(sheet_name=sheet))
    elif name.endswith(".json"):
        obj = json.loads(data.decode("utf-8"))
        if isinstance(obj, list):
            dfs.append(pd.json_normalize(obj))
        else:
            dfs.append(pd.json_normalize(obj))
    return dfs


def _read_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def _read_pdf(file) -> str:
    if PdfReader is None:
        return "(Instala pypdf para leer PDFs)"
    data = file.read()
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        parts.append(f"[p.{i}]\n" + txt)
    return "\n\n".join(parts)


# ---------------------- Embeddings + FAISS ----------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el entorno.")
    return OpenAI(api_key=api_key)


EMBED_MODEL = "text-embedding-3-small"  # costo bajo, suficiente para RAG
CHAT_MODEL = "gpt-4o-mini"              # rápido/preciso para orquestación


def embed_texts(texts: t.List[str]) -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    return vecs


@dataclass
class VectorIndex:
    index: t.Any
    chunks: t.List[Chunk]


def build_index(chunks: t.List[Chunk]) -> VectorIndex:
    if not FAISS_OK:
        raise RuntimeError("Instala faiss-cpu para indexación semántica.")
    vecs = embed_texts([c.text for c in chunks])
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    # normaliza para similitud coseno
    faiss.normalize_L2(vecs)
    idx.add(vecs)
    return VectorIndex(index=idx, chunks=chunks)


def search_index(vx: VectorIndex, query: str, k: int = 6) -> t.List[Chunk]:
    q = embed_texts([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = vx.index.search(q, k)
    results = []
    for i in I[0]:
        if i == -1:
            continue
        results.append(vx.chunks[i])
    return results


# ---------------------- División en chunks ----------------------
def chunk_text(text: str, source: str, doc_id: str, max_chars: int = 1200) -> t.List[Chunk]:
    parts: t.List[Chunk] = []
    start = 0
    page_hint = None
    # Extrae [p.N] si existen para conservar citas
    page_markers = list(re.finditer(r"\[p\.(\d+)\]", text))
    marker_positions = {m.start(): m.group(1) for m in page_markers}

    while start < len(text):
        end = min(start + max_chars, len(text))
        seg = text[start:end]
        last_nl = seg.rfind("\n")
        if last_nl > 400:
            end = start + last_nl
            seg = text[start:end]
        nearest_markers = [pos for pos in marker_positions.keys() if pos <= start]
        if nearest_markers:
            last_pos = max(nearest_markers)
            page_hint = marker_positions[last_pos]
        parts.append(Chunk(doc_id=doc_id, source=source, location=f"p.{page_hint}" if page_hint else "—", text=seg))
        start = end
    return parts


def df_to_text(df: pd.DataFrame, source: str, doc_id: str, max_rows: int = 30) -> t.List[Chunk]:
    head = df.head(max_rows).to_markdown(index=False)
    stats = df.describe(include="all", datetime_is_numeric=True).T.fillna("")
    stats_md = stats.to_markdown()
    text = f"[tabla]\narchivo: {source}\n\nMuestra:\n{head}\n\nEstadísticas:\n{stats_md}"
    return chunk_text(text, source=source, doc_id=doc_id)


# ---------------------- LLM Orquestación ----------------------
SYSTEM_PROMPT = (
    "Eres un analista de datos. Responde con precisión usando SOLO el contexto. "
    "Si el usuario pide una gráfica o tabla, describe brevemente qué harás y devuelve también un bloque JSON con 'action' y 'spec'. "
    "'action' puede ser 'answer' | 'plot' | 'table' | 'pivot'. "
    "Para 'plot', 'spec' debe incluir: kind(line|bar|hist|box|scatter), x, y, agg (opcional), hue (opcional). "
    "Para 'table', 'spec' debe incluir: columns (lista) y filtros (opcional). "
    "Para 'pivot', 'spec' debe incluir: index, columns, values, aggfunc. "
    "Incluye siempre una lista 'citations' con las fuentes (archivo y ubicación) que usaste."
)


def llm_answer(query: str, context_blocks: t.List[Chunk]) -> dict:
    client = get_openai_client()
    ctx_txt = "\n\n".join([f"FUENTE: {c.source} ({c.location})\n{c.text}" for c in context_blocks])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{ctx_txt}"},
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.2)
    content = resp.choices[0].message.content
    # Intenta extraer bloque JSON {action, spec, citations, answer}
    action = "answer"
    citations = [f"{c.source} ({c.location})" for c in context_blocks[:3]]
    spec = {}
    answer = content
    m = re.search(r"\{[\s\S]*\}", content)
    if m:
        try:
            js = json.loads(m.group(0))
            action = js.get("action", action)
            spec = js.get("spec", spec)
            citations = js.get("citations", citations)
            answer = js.get("answer", content)
        except Exception:
            pass
    return {"action": action, "spec": spec, "citations": citations, "answer": answer}


# ---------------------- Acciones sobre DataFrames ----------------------
def safe_plot(df: pd.DataFrame, spec: dict):
    kind = spec.get("kind", "line")
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("hue")
    agg = spec.get("agg")

    if not x or not y:
        st.warning("Falta 'x' o 'y' en la especificación de la gráfica.")
        return

    dfx = df.copy()
    # Conversión de fechas si procede
    for col in [x, y, hue]:
        if col in dfx.columns and dfx[col].dtype == object:
            try:
                dfx[col] = pd.to_datetime(dfx[col])
            except Exception:
                pass

    if agg and isinstance(y, str):
        dfx = dfx.groupby(x, as_index=False)[y].agg(agg)

    plt.figure()
    if kind == "line":
        if hue and hue in dfx.columns:
            for key, grp in dfx.groupby(hue):
                plt.plot(grp[x], grp[y], label=str(key))
            plt.legend()
        else:
            plt.plot(dfx[x], dfx[y])
    elif kind == "bar":
        if agg and isinstance(y, str):
            dfx = dfx.groupby(x, as_index=False)[y].agg(agg)
        plt.bar(dfx[x], dfx[y])
        plt.xticks(rotation=45, ha="right")
    elif kind == "hist":
        plt.hist(dfx[y].dropna())
    elif kind == "box":
        dfx[[y]].plot(kind="box")
    elif kind == "scatter":
        plt.scatter(dfx[x], dfx[y])
    else:
        st.warning(f"Tipo de gráfica no soportado: {kind}")
        return
    st.pyplot(plt.gcf())


def safe_table(df: pd.DataFrame, spec: dict):
    cols = spec.get("columns")
    filtros = spec.get("filtros") or {}
    dfx = df.copy()
    for k, v in filtros.items():
        if k in dfx.columns:
            dfx = dfx[dfx[k] == v]
    if cols:
        cols = [c for c in cols if c in dfx.columns]
        dfx = dfx[cols]
    st.dataframe(dfx)
    csv = dfx.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="tabla.csv", mime="text/csv")


def safe_pivot(df: pd.DataFrame, spec: dict):
    idx = spec.get("index")
    cols = spec.get("columns")
    vals = spec.get("values")
    agg = spec.get("aggfunc", "sum")
    try:
        pt = pd.pivot_table(df, index=idx, columns=cols, values=vals, aggfunc=agg)
        st.dataframe(pt.reset_index())
        csv = pt.to_csv().encode("utf-8")
        st.download_button("Descargar pivot CSV", data=csv, file_name="pivot.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error creando pivot: {e}")


# ---------------------- Interfaz Streamlit ----------------------
st.set_page_config(page_title="Agente de Archivos", layout="wide")
st.title("📁 Agente de Archivos: tablas, gráficas y respuestas")

with st.sidebar:
    st.markdown("**Config**")
    model = st.selectbox("Modelo LLM", [CHAT_MODEL, "gpt-4.1-mini", "gpt-4o"])
    if model != CHAT_MODEL:
        global CHAT_MODEL
        CHAT_MODEL = model
    st.caption("Requiere OPENAI_API_KEY en el entorno.")
    st.markdown("---")
    st.markdown("**Carga de archivos**")
    files = st.file_uploader(
        "Sube CSV, XLSX, JSON, TXT, PDF (puedes varios)",
        type=["csv", "xlsx", "xls", "json", "txt", "pdf"],
        accept_multiple_files=True,
    )

if "vx" not in st.session_state:
    st.session_state.vx = None
if "dfs" not in st.session_state:
    st.session_state.dfs = {}  # key -> DataFrame
if "texts" not in st.session_state:
    st.session_state.texts = []  # lista de (name, text)
if "chat" not in st.session_state:
    st.session_state.chat = []  # [(role, content)]

# Ingesta
if files:
    chunks: t.List[Chunk] = []
    for f in files:
        name = f.name
        if name.lower().endswith((".csv", ".xlsx", ".xls", ".json")):
            try:
                for i, df in enumerate(_bytes_to_df(f)):
                    key = f"{name}#sheet{i+1}"
                    st.session_state.dfs[key] = df
                    chunks += df_to_text(df, source=key, doc_id=name)
            except Exception as e:
                st.warning(f"No pude leer {name}: {e}")
        elif name.lower().endswith(".txt"):
            try:
                txt = _read_txt(f)
                st.session_state.texts.append((name, txt))
                chunks += chunk_text(txt, source=name, doc_id=name)
            except Exception as e:
                st.warning(f"No pude leer {name}: {e}")
        elif name.lower().endswith(".pdf"):
            try:
                txt = _read_pdf(f)
                st.session_state.texts.append((name, txt))
                chunks += chunk_text(txt, source=name, doc_id=name)
            except Exception as e:
                st.warning(f"No pude leer {name}: {e}")
    if chunks:
        try:
            st.session_state.vx = build_index(chunks)
            st.success("Índice semántico construido ✅")
        except Exception as e:
            st.error(f"No se pudo construir el índice: {e}")

# Tabs principales
T1, T2, T3 = st.tabs(["Chat", "Tablas & Gráficas", "Fuentes"])

with T1:
    st.subheader("Chat con tus archivos")
    q = st.text_input("Escribe tu pregunta o pide una gráfica/pivot…")
    ask = st.button("Preguntar", type="primary")

    if ask and q:
        st.session_state.chat.append(("user", q))
        if st.session_state.vx is None:
            st.warning("Primero sube archivos para crear el índice.")
        else:
            ctx = search_index(st.session_state.vx, q, k=6)
            ans = llm_answer(q, ctx)
            st.session_state.chat.append(("assistant", ans))

    # Render del historial
    for role, payload in st.session_state.chat:
        if role == "user":
            st.chat_message("user").markdown(payload)
        else:
            ans = payload  # dict
            with st.chat_message("assistant"):
                st.markdown(ans.get("answer", ""))
                cits = ans.get("citations", [])
                if cits:
                    st.caption("Fuentes: " + "; ".join(cits))
                action = ans.get("action")
                spec = ans.get("spec", {})
                if action in {"plot", "table", "pivot"} and st.session_state.dfs:
                    df_key = st.selectbox("Selecciona la tabla base", list(st.session_state.dfs.keys()), key=str(uuid.uuid4()))
                    df = st.session_state.dfs[df_key]
                    if action == "plot":
                        safe_plot(df, spec)
                    elif action == "table":
                        safe_table(df, spec)
                    elif action == "pivot":
                        safe_pivot(df, spec)

with T2:
    st.subheader("Explora y grafica rápidamente")
    if st.session_state.dfs:
        df_key = st.selectbox("Elige una tabla", list(st.session_state.dfs.keys()))
        df = st.session_state.dfs[df_key]
        st.dataframe(df.head(200))
        st.markdown("---")
        with st.expander("Gráfica rápida"):
            kind = st.selectbox("Tipo", ["line", "bar", "hist", "box", "scatter"])
            x = st.selectbox("Eje X", ["(ninguno)"] + list(df.columns))
            y = st.selectbox("Eje Y", ["(ninguno)"] + list(df.columns))
            hue = st.selectbox("Categoría (opcional)", ["(ninguno)"] + list(df.columns))
            agg = st.selectbox("Agregación (opcional)", ["(ninguna)", "sum", "mean", "count", "max", "min"])
            if st.button("Dibujar"):
                spec = {
                    "kind": kind,
                    "x": None if x == "(ninguno)" else x,
                    "y": None if y == "(ninguno)" else y,
                    "hue": None if hue == "(ninguno)" else hue,
                    "agg": None if agg == "(ninguna)" else agg,
                }
                safe_plot(df, spec)
    else:
        st.info("Sube archivos para explorar.")

with T3:
    st.subheader("Fuentes y texto indexado")
    if st.session_state.texts:
        for name, txt in st.session_state.texts:
            with st.expander(name):
                st.text(txt[:5000] + ("\n..." if len(txt) > 5000 else ""))
    if st.session_state.dfs:
        for key, df in st.session_state.dfs.items():
            with st.expander(f"{key} (preview)"):
                st.dataframe(df.head(50))

st.markdown("---")
st.caption("Hecho con ❤️ por Kai. Este demo no guarda tus archivos en servidor: viven en la sesión de Streamlit.")
