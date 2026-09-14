"""Streamlit UI for the Multi-Agent Data & Knowledge Graph Engine.

Wires together the three agents:
  Agent 1 (agents.router)      - FAISS + NetworkX retrieval
  Agent 2 (agents.text_to_sql) - Groq-powered Text-to-SQL + profiling
  Agent 3 (agents.synthesizer) - result summary + graph visualization
"""

import os
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

from utils.graph_builder import build_knowledge_graph
from utils.faiss_indexer import VectorIndexer
from agents.router import route_query
from agents.text_to_sql import generate_sql_and_profile, SQLSafetyError
from agents.synthesizer import generate_graph_plot, summarize_result

load_dotenv()

st.set_page_config(layout="wide", page_title="Multi-Agent Data & Knowledge Graph Engine", page_icon="🕸️")

DB_PATH = "data/chinook.db"
DOCS_DIR = "data/documents"


@st.cache_resource(show_spinner="Building knowledge graph and vector index...")
def load_engine():
    """Loads the knowledge graph + vector index once per session.

    Raised exceptions are allowed to propagate here (Streamlit's cache
    machinery handles that fine) - the caller renders them with full detail
    instead of the previous one-line generic message.
    """
    graph = build_knowledge_graph(DB_PATH)
    indexer = VectorIndexer(db_path=DB_PATH, docs_dir=DOCS_DIR)
    return graph, indexer


def render_startup_error(exc: Exception):
    st.error("The engine failed to start up. See the details below for the exact cause.")
    if not os.path.exists(DB_PATH):
        st.warning(f"`{DB_PATH}` was not found. Make sure the Chinook (or your own) SQLite database is in `data/`.")
    if not os.getenv("GROQ_API_KEY"):
        st.warning("`GROQ_API_KEY` is not set. Create a `.env` file in the project root with `GROQ_API_KEY=gsk_...`.")
    with st.expander("Full error / traceback", expanded=True):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="python")


st.title("🕸️ Multi-Agent Data & Knowledge Graph Engine")
st.caption("FAISS semantic retrieval + NetworkX knowledge graph + Groq Text-to-SQL agents, over a local SQLite database.")

try:
    graph, indexer = load_engine()
except Exception as e:
    render_startup_error(e)
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: engine status
# ---------------------------------------------------------------------------
schema_chunks = sum(1 for d in indexer.docs if d["type"] == "schema")
document_chunks = sum(1 for d in indexer.docs if d["type"] == "document")

st.sidebar.title("Engine Status")
st.sidebar.metric("Indexed Schema Tables", schema_chunks)
st.sidebar.metric("Indexed Document Chunks", document_chunks)
st.sidebar.metric("Knowledge Graph Nodes", len(graph.nodes()))
st.sidebar.metric("Knowledge Graph Edges", len(graph.edges()))
if indexer.build_latency_ms is not None:
    st.sidebar.caption(f"Vector index build time: {indexer.build_latency_ms:.1f} ms")

st.sidebar.divider()
groq_ready = bool(os.getenv("GROQ_API_KEY"))
st.sidebar.markdown(f"**GROQ_API_KEY:** {'✅ configured' if groq_ready else '❌ missing'}")
st.sidebar.markdown(f"**Model override (`GROQ_MODEL`):** `{os.getenv('GROQ_MODEL', 'auto-fallback chain')}`")

if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.history:
    last_latency = st.session_state.history[-1].get("total_latency_ms")
    if last_latency:
        st.sidebar.metric("Last Query Latency", f"{last_latency:.0f} ms")

# ---------------------------------------------------------------------------
# Main chat panel
# ---------------------------------------------------------------------------
query = st.chat_input("Ask a question about your enterprise data...")

if query:
    pipeline_start = time.perf_counter()
    result = {"query": query}
    try:
        with st.spinner("Agent 1: retrieving schema & document context..."):
            context = route_query(query, indexer, graph)

        with st.spinner("Agent 2: generating SQL and profiling results..."):
            sql_query, df, profile, meta = generate_sql_and_profile(query, context, db_path=DB_PATH)

        with st.spinner("Agent 3: synthesizing summary and graph..."):
            summary = summarize_result(query, sql_query, df, profile)
            fig = generate_graph_plot(context["graph_edges"], highlight_nodes=context.get("tables"))

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000
        result.update({
            "ok": True,
            "context": context,
            "sql_query": sql_query,
            "df": df,
            "profile": profile,
            "meta": meta,
            "summary": summary,
            "fig": fig,
            "total_latency_ms": total_latency_ms,
        })
    except SQLSafetyError as e:
        result.update({"ok": False, "error_kind": "safety", "error": str(e)})
    except Exception as e:
        result.update({
            "ok": False, "error_kind": "runtime", "error": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        })

    st.session_state.history.append(result)

# Render most recent result (kept simple/single-turn on screen; history is
# still available in st.session_state.history for future multi-turn UI).
if st.session_state.history:
    latest = st.session_state.history[-1]
    st.markdown(f"**Query:** {latest['query']}")

    if not latest.get("ok"):
        if latest.get("error_kind") == "safety":
            st.warning(f"Blocked for safety: {latest['error']}")
        else:
            st.error(f"Pipeline error: {latest['error']}")
            with st.expander("Full traceback"):
                st.code(latest.get("traceback", "no traceback captured"), language="python")
    else:
        tab_answer, tab_sql, tab_graph, tab_trace = st.tabs(
            ["💬 Answer", "🧮 SQL & Data", "🕸️ Knowledge Graph", "🔎 Agent Trace"]
        )

        with tab_answer:
            st.markdown(latest["summary"])
            st.caption(f"Total pipeline latency: {latest['total_latency_ms']:.0f} ms · Model used: `{latest['meta'].get('model_used')}`")
            if latest["context"].get("document_context"):
                with st.expander("Retrieved document context"):
                    for chunk in latest["context"]["document_context"]:
                        st.markdown(f"> {chunk}")

        with tab_sql:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("Generated SQL")
                st.code(latest["sql_query"], language="sql")
                st.subheader("Result")
                st.dataframe(latest["df"], use_container_width=True)
            with col2:
                st.subheader("Dataset Profile")
                st.json(latest["profile"])

        with tab_graph:
            st.subheader("Entity Knowledge Subgraph")
            st.caption("Orange nodes are the tables FAISS matched directly; gray nodes were pulled in via foreign-key traversal.")
            st.pyplot(latest["fig"])

        with tab_trace:
            st.subheader("Agent Execution Trace")
            st.caption("Every retrieval/model call in this run, with latency, for debugging and for the performance metrics in the README.")
            st.json(latest["meta"].get("steps", []))
else:
    st.info("Ask a question in the chat box below to run the full Router → Text-to-SQL → Synthesizer pipeline.")
