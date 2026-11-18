"""
Step 5 Demo: ZERO-TOKEN Offline Test of Hybrid Search + Clean Service Adapters!
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.adapters.github_adapter import GitHubAdapter
from sre_agent.adapters.supabase_adapter import SupabaseAdapter
from sre_agent.adapters.vercel_adapter import VercelAdapter
from sre_agent.helpers.embeddings import get_embedder
from sre_agent.services.retrieval_service import HybridRetrievalService, seed_sample_logs
from sre_agent.services.tool_service import ToolService, register_sre_tools


def main() -> None:
    embedder = get_embedder()

    print("=" * 75)
    print("PART 1: HYBRID LOG RETRIEVER (BM25 + Dense Vector + Reciprocal Rank Fusion)")
    print("=" * 75)
    retrieval = HybridRetrievalService(embedder=embedder)
    seed_sample_logs(retrieval)

    q1 = "PGRST301"
    print(f"\nQuery A (Exact Error Code): '{q1}'")
    for hit in retrieval.search(q1, top_k=2):
        print(
            f"  rrf={hit.rrf_score:.4f} | bm25={hit.bm25_score:.2f} | vec={hit.vector_score:.3f} "
            f"| [{hit.doc.source}] {hit.doc.content}"
        )

    q2 = "database connection pool ran out of slots"
    print(f"\nQuery B (Conceptual — zero shared keywords): '{q2}'")
    for hit in retrieval.search(q2, top_k=2):
        print(
            f"  rrf={hit.rrf_score:.4f} | bm25={hit.bm25_score:.2f} | vec={hit.vector_score:.3f} "
            f"| [{hit.doc.source}] {hit.doc.content}"
        )

    print("\n" + "=" * 75)
    print("PART 2: CLOUD ADAPTERS REGISTERED IN TOOL SERVICE (Sandbox Mode)")
    print("=" * 75)
    tools = ToolService(embedder=embedder)
    register_sre_tools(
        tool_service=tools,
        vercel=VercelAdapter(live=False),
        supabase=SupabaseAdapter(live=False),
        github=GitHubAdapter(live=False),
        retrieval_service=retrieval,
    )
    print(f"Total SRE Cloud Tools Registered: {len(tools)}")
    print("\n" + tools.execute_tool("search_incident_logs", {"query": "PGRST301 timeout", "top_k": 2}))
    print("\n" + tools.execute_tool("check_supabase_slow_queries", {"min_duration_ms": 500}))


if __name__ == "__main__":
    main()
