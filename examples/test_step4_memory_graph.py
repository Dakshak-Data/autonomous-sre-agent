"""
Step 4 Demo: ZERO-TOKEN Offline Test of Engram Memory + HippoRAG Knowledge Graph!

Demonstrates:
  1. Engram Long-Term Episodic Memory (SQLite):
       - Stores 2 past incidents from "last week".
       - Re-opens SQLite (simulating a brand-new terminal session days later!).
       - Recalls the exact matching past fix with similarity score & reinforcement count.
  2. HippoRAG Multi-Hop Knowledge Graph (Personalized PageRank):
       - Maps Vercel routes -> Supabase tables -> DB Indexes -> GitHub Commits.
       - Query mentions ONLY "Vercel checkout timeout" (ZERO mention of GitHub or indexes!).
       - Personalized PageRank traverses 2-3 hops and discovers BOTH the missing DB index
         AND the culprit GitHub commit automatically!
  3. Prints ready-to-run Neo4j Cypher export queries.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.helpers.embeddings import get_embedder
from sre_agent.services.graph_service import HippoGraphService
from sre_agent.services.memory_service import EngramMemoryService


def main() -> None:
    embedder = get_embedder()

    print("=" * 75)
    print("PART 1: ENGRAM LONG-TERM EPISODIC MEMORY (SQLite Persistence Test)")
    print("=" * 75)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "sre_engram.db"

        # Session 1 (Last Tuesday): Agent resolves 2 incidents and saves them
        mem_session1 = EngramMemoryService(db_path=db_file, embedder=embedder)
        mem_session1.store_episode(
            incident_query="Vercel /api/users throwing 504 timeout due to slow Supabase query",
            root_cause="Missing B-Tree index on users.email caused 8.2s sequential scan",
            resolution="Ran CREATE INDEX CONCURRENTLY idx_users_email ON users(email)",
            affected_entities=["route:/api/users", "table:users"],
        )
        mem_session1.store_episode(
            incident_query="Vercel auth login failing with 401 Invalid JWT signature",
            root_cause="SUPABASE_JWT_SECRET environment variable drifted after rotation",
            resolution="Updated SUPABASE_JWT_SECRET in Vercel project env vars and redeployed",
            affected_entities=["route:/api/auth/login", "config:SUPABASE_JWT_SECRET"],
        )
        mem_session1.close()
        print("Session 1 closed. 2 past incidents persisted to SQLite.\n")

        # Session 2 (5 days later — fresh process!): New similar incident occurs
        mem_session2 = EngramMemoryService(db_path=db_file, embedder=embedder)
        new_incident = "My Vercel API endpoint is timing out (504) while querying Supabase!"
        print(f"NEW INCIDENT (Session 2): {new_incident}\n")

        recalls = mem_session2.recall(new_incident, top_k=1, min_score=0.55)
        print(EngramMemoryService.format_for_prompt(recalls))
        mem_session2.close()

    print("\n" + "=" * 75)
    print("PART 2: HIPPORAG MULTI-HOP KNOWLEDGE GRAPH (Personalized PageRank)")
    print("=" * 75)

    graph = HippoGraphService(embedder=embedder)

    # Describe key app entities
    graph.add_node_description("route:/api/checkout", "Vercel serverless API endpoint for checkout payments")
    graph.add_node_description("route:/api/health", "Vercel lightweight healthcheck endpoint")
    graph.add_node_description("table:orders", "Supabase PostgreSQL table storing customer orders")
    graph.add_node_description("index:orders.customer_email", "Missing database B-Tree index causing slow table scan")
    graph.add_node_description("commit:a1b2c3", "GitHub PR #42 that added customer_email lookup filter")
    graph.add_node_description("service:stripe", "External Stripe payment gateway API")

    # Add topology edges (Subject -> Relation -> Object)
    graph.add_triple("commit:a1b2c3", "MODIFIES", "route:/api/checkout")
    graph.add_triple("route:/api/checkout", "QUERIES", "table:orders")
    graph.add_triple("table:orders", "NEEDS_INDEX", "index:orders.customer_email")
    graph.add_triple("route:/api/checkout", "CALLS", "service:stripe")
    graph.add_triple("route:/api/health", "PINGS", "table:orders")

    # Notice: Query ONLY mentions Vercel checkout timeout — ZERO mention of GitHub or Index!
    hop_query = "Vercel checkout payment endpoint is throwing 504 Gateway Timeout"
    print(f"QUERY (Notice: NO mention of GitHub or DB index!): {hop_query}\n")

    result = graph.query(hop_query, top_k_seeds=1, top_k_results=5)
    print(f"Step 1 — Seed Node Matched: {result.seed_nodes}\n")
    print("Step 2 — Multi-Hop Personalized PageRank Output:")
    print(result.format_for_prompt())

    print("\n" + "=" * 75)
    print("PART 3: NEO4J CYPHER EXPORT (Ready for Neo4j Community Edition)")
    print("=" * 75)
    for cypher in graph.to_neo4j_cypher()[:3]:
        print(f"  {cypher}")


if __name__ == "__main__":
    main()
