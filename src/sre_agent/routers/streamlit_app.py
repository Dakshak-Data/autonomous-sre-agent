"""
Streamlit UI Router: Visual SRE Command Center for Autonomous SRE Agent.

Architecture:
  Streamlit UI (routers/streamlit_app.py)
       │
       ├──▶ IncidentController.investigate()   (Tab 1: Full Autonomous SRE Investigation)
       ├──▶ HippoGraphService.query()          (Tab 2: Multi-Hop Knowledge Graph Explorer)
       ├──▶ EngramMemoryService.recall()       (Tab 3: Persistent Episodic Memory Viewer)
       └──▶ HybridRetrievalService.search()    (Tab 4: Vespa BM25 + Vector Hybrid Search)

Run with:
  cd sre-agent && PYTHONPATH=src .venv/bin/streamlit run src/sre_agent/routers/streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure `src/` is importable when launched via `streamlit run`
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sre_agent.config.constants import DEFAULT_ENGRAM_DB, DEFAULT_WORKFLOW_DB
from sre_agent.controllers.incident_controller import IncidentController, build_default_topology
from sre_agent.helpers.embeddings import get_embedder
from sre_agent.services.graph_service import HippoGraphService
from sre_agent.services.memory_service import EngramMemoryService
from sre_agent.services.retrieval_service import HybridRetrievalService, seed_sample_logs

load_dotenv()


# --- Cached Shared Resources (Loaded Once for Fast UI) ---
@st.cache_resource
def load_shared_services() -> tuple[HippoGraphService, HybridRetrievalService]:
    embedder = get_embedder()
    graph = build_default_topology()
    retriever = HybridRetrievalService(embedder=embedder)
    seed_sample_logs(retriever)
    return graph, retriever


def check_env_status() -> dict[str, bool]:
    """Safely check which credentials exist in .env without ever displaying secrets."""
    supa = os.getenv("SUPABASE_DB_URL", "").strip()
    return {
        "Gemini API": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "Groq API": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "Vercel Token": bool(os.getenv("VERCEL_TOKEN", "").strip()),
        "Supabase DB": bool(supa) and "[YOUR-DATABASE-PASSWORD]" not in supa,
        "GitHub Token": bool(os.getenv("GITHUB_TOKEN", "").strip()),
    }


def render_sidebar() -> tuple[bool, bool, bool, str | None]:
    """Render sidebar controls, connection status badges, and mode toggles."""
    st.sidebar.title("Autonomous SRE Agent")
    st.sidebar.caption("Zero-LangChain • Adapter Architecture")

    st.sidebar.subheader("Execution Mode")
    live_llm = st.sidebar.toggle(
        "Live LLM Mode (Gemini + Groq)",
        value=False,
        help="OFF = Zero-token offline mock mode. ON = Calls your real Gemini + Groq APIs with auto-failover.",
    )
    live_cloud = st.sidebar.toggle(
        "Live Cloud Tools (Vercel + Supabase + GitHub)",
        value=False,
        help="OFF = Safe sandbox telemetry. ON = Queries your live Vercel app, Supabase DB & GitHub repo.",
    )
    use_cache = st.sidebar.toggle(
        "Semantic Cache (0-Token Repeat)",
        value=True,
        help="ON = Repeated similar queries within 5 min return instantly from Engram SQLite with 0 LLM tokens!",
    )

    workflow_id_input = st.sidebar.text_input(
        "Workflow ID (Durable Replay)",
        value="",
        placeholder="e.g. inc-2026-001 (optional)",
        help="Re-use the same Workflow ID to watch Temporal-style SQLite checkpoint replay!",
    ).strip()

    st.sidebar.divider()
    st.sidebar.subheader("Connected Services (.env)")
    for label, ready in check_env_status().items():
        icon = "🟢" if ready else "⚪"
        status_text = "Ready" if ready else "Not Set"
        st.sidebar.write(f"{icon} **{label}:** {status_text}")

    st.sidebar.divider()
    st.sidebar.caption("Built with Pure Python Harness, Dynamic Tool Loading, HippoRAG, Engram & Hybrid Search.")
    return live_llm, live_cloud, use_cache, (workflow_id_input or None)


def render_investigator_tab(
    live_llm: bool,
    live_cloud: bool,
    use_cache: bool,
    workflow_id: str | None,
) -> None:
    """Tab 1: Run end-to-end autonomous SRE investigations."""
    st.subheader("Autonomous Production Incident Investigator")
    st.caption(
        "Runs Durable Step 1 (Engram Recall) -> Step 2 (HippoRAG Graph) -> "
        "Step 3 (Dynamic Tools + Harness Loop) -> Step 4 (Save to Engram)."
    )

    # Read target app details safely from .env (with generic public defaults)
    app_name = os.getenv("TARGET_APP_NAME", "my-web-app").strip()
    app_url = os.getenv("TARGET_APP_URL", "https://my-web-app.vercel.app").strip()

    # Quick 1-click preset incident prompts
    st.write("**Quick Test Prompts:**")
    c1, c2, c3 = st.columns(3)
    if c1.button("Live Vercel Health", use_container_width=True):
        st.session_state["sre_query"] = (
            f"Check live health and latest Vercel deployment logs of my {app_name} app at {app_url}"
        )
    if c2.button("Supabase Slow Queries", use_container_width=True):
        st.session_state["sre_query"] = (
            "Inspect my Supabase PostgreSQL database for slow queries, locks, and missing indexes"
        )
    if c3.button("504 Timeout Root Cause", use_container_width=True):
        st.session_state["sre_query"] = (
            "Vercel /api/checkout is throwing 504 timeout and Supabase queries are very slow. "
            "Find the root cause."
        )

    query = st.text_area(
        "Describe the incident or SRE task:",
        key="sre_query",
        value=st.session_state.get(
            "sre_query",
            f"Check health and latest deployment logs of my Vercel {app_name} app at {app_url}",
        ),
        height=90,
    )

    mode_badge = "LIVE LLM" if live_llm else "ZERO-TOKEN MOCK LLM"
    cloud_badge = "LIVE CLOUD" if live_cloud else "SANDBOX TOOLS"
    run_clicked = st.button(
        f"Run Investigation ({mode_badge} | {cloud_badge})",
        type="primary",
        use_container_width=True,
    )

    if run_clicked and query.strip():
        controller = IncidentController(data_dir=".")
        log_buffer = io.StringIO()

        with st.spinner("Running Autonomous SRE Pipeline..."):
            try:
                with contextlib.redirect_stdout(log_buffer):
                    result = controller.investigate(
                        query=query.strip(),
                        workflow_id=workflow_id,
                        live_llm=live_llm,
                        live_cloud=live_cloud,
                        auto_approve=False,
                        use_semantic_cache=use_cache,
                    )
                st.session_state["last_result"] = result
                st.session_state["last_trace"] = log_buffer.getvalue()
            except Exception as exc:
                st.error(f"Investigation failed: {exc}")
                with st.expander("Execution Logs"):
                    st.code(log_buffer.getvalue() or "(no logs)")
                return

    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.divider()

        # KPI Telemetry Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agent Steps", res.steps_taken)
        m2.metric("Dynamic Tools Loaded", f"{len(res.tools_injected)} / 6")
        m3.metric("LLM Tokens Used", res.total_llm_tokens)
        m4.metric("Tokens Saved (Trim+Compact)", res.tokens_saved_by_truncation + res.tokens_saved_by_compaction)

        st.markdown("### Final SRE Diagnostic Report")
        st.info(res.final_answer)

        c_left, c_right = st.columns(2)
        c_left.write(f"**Tools Injected:** `{res.tools_injected}`")
        c_left.write(f"**Tools Executed:** `{res.tools_executed}`")
        c_right.write(f"**Models Used:** `{res.models_used}`")
        c_right.write(f"**Auto-Compactions:** `{res.compactions_triggered}`")

        with st.expander("View Live Harness & Durable Execution Trace", expanded=True):
            st.code(st.session_state.get("last_trace", ""), language="text")


def render_hippo_tab(graph: HippoGraphService) -> None:
    """Tab 2: Interactive HippoRAG Multi-Hop Personalized PageRank Explorer (0 LLM tokens)."""
    st.subheader("HippoRAG Multi-Hop Knowledge Graph (0 LLM Tokens)")
    st.caption("Uses Personalized PageRank over your Vercel + Supabase + GitHub app topology.")

    hop_query = st.text_input(
        "Test a symptom query against the Knowledge Graph:",
        value="database table queries slow on live vercel app",
    )
    if hop_query.strip():
        result = graph.query(hop_query.strip(), top_k_seeds=1, top_k_results=6)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Step 1: Semantic Seed Node Matched**")
            for node, score in result.seed_nodes:
                st.success(f"`{node}` (cosine similarity = {score:.4f})")

            st.write("**Step 2: Personalized PageRank Multi-Hop Entities**")
            st.table([{"Entity": n, "PPR Score": s} for n, s in result.ranked_nodes])

        with col2:
            st.write("**Connected Topology Paths**")
            for triple in result.relevant_triples:
                st.code(triple.to_text(), language="text")

    with st.expander("Export Neo4j Cypher Queries"):
        st.code("\n".join(graph.to_neo4j_cypher()), language="cypher")


def render_engram_tab() -> None:
    """Tab 3: Inspect SQLite Engram Long-Term Episodic Memory & Durable Checkpoints."""
    st.subheader("Engram Long-Term Episodic Memory (SQLite)")

    db_path = Path(DEFAULT_ENGRAM_DB)
    if not db_path.exists():
        st.info("No past incidents stored yet. Run an investigation in Tab 1 to create your first Engram memory!")
        return

    search_q = st.text_input("Test Semantic Memory Recall (0 LLM tokens):", value="vercel database timeout")
    if search_q.strip():
        mem = EngramMemoryService(db_path=db_path, embedder=get_embedder())
        try:
            recalls = mem.recall(search_q.strip(), top_k=3, min_score=0.35)
        finally:
            mem.close()

        if recalls:
            st.write("**Top Semantic Memory Matches:**")
            for m in recalls:
                with st.expander(f"Score {m.score:.2f} | Recalled {m.recall_count}x | {m.incident_query[:80]}"):
                    st.write(f"**Timestamp:** `{m.timestamp}`")
                    st.write(f"**Affected Tools/Entities:** `{m.affected_entities}`")
                    st.write(f"**Saved Resolution:**\n{m.resolution}")
        else:
            st.warning("No matching past memories above similarity threshold.")

    st.divider()
    st.write("**All Stored Durable Workflow Checkpoints (`sre_workflow_state.db`):**")
    wf_path = Path(DEFAULT_WORKFLOW_DB)
    if wf_path.exists():
        conn = sqlite3.connect(str(wf_path))
        rows = conn.execute(
            "SELECT workflow_id, step_name, completed_at FROM workflow_events ORDER BY completed_at DESC LIMIT 20"
        ).fetchall()
        conn.close()
        if rows:
            st.table([{"Workflow ID": r[0], "Completed Step": r[1], "Timestamp": r[2]} for r in rows])
        else:
            st.caption("No durable workflow checkpoints recorded yet.")
    else:
        st.caption("No durable workflow database created yet.")


def render_hybrid_tab(retriever: HybridRetrievalService) -> None:
    """Tab 4: Interactive Hybrid Log Search (Vespa BM25 + Dense Vector RRF)."""
    st.subheader("Hybrid Log Retrieval Playground (Vespa Pattern: BM25 + Vector RRF)")
    st.caption("Try exact error codes like `PGRST301` OR conceptual phrases like `database ran out of connections`.")

    q = st.text_input("Search Incident Logs (0 LLM tokens):", value="PGRST301")
    if q.strip():
        hits = retriever.search(q.strip(), top_k=5)
        st.table(
            [
                {
                    "Source": h.doc.source,
                    "RRF Score": h.rrf_score,
                    "BM25 (Keyword)": h.bm25_score,
                    "Vector (Semantic)": h.vector_score,
                    "Log Content": h.doc.content,
                }
                for h in hits
            ]
        )

    with st.expander("View Production Vespa Schema (.sd)"):
        st.code(HybridRetrievalService.generate_vespa_schema(), language="text")


def main() -> None:
    st.set_page_config(
        page_title="Autonomous SRE Agent",
        page_icon="🛡️",
        layout="wide",
    )

    live_llm, live_cloud, use_cache, workflow_id = render_sidebar()
    graph, retriever = load_shared_services()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🤖 SRE Investigator",
            "🕸️ HippoRAG Graph",
            "🧠 Engram Memory & Checkpoints",
            "🔍 Hybrid Log Search",
        ]
    )

    with tab1:
        render_investigator_tab(
            live_llm=live_llm,
            live_cloud=live_cloud,
            use_cache=use_cache,
            workflow_id=workflow_id,
        )
    with tab2:
        render_hippo_tab(graph=graph)
    with tab3:
        render_engram_tab()
    with tab4:
        render_hybrid_tab(retriever=retriever)


if __name__ == "__main__":
    main()
