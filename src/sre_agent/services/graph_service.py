"""
HippoRAG Graph Service: Multi-Hop Knowledge Graph + Personalized PageRank + Neo4j Export.
"""

from __future__ import annotations

import os

import numpy as np
from fastembed import TextEmbedding

from sre_agent.helpers.embeddings import cosine_similarity, embed_text, get_embedder
from sre_agent.schemas.models import GraphTriple, HippoRetrievalResult


class HippoGraphService:
    """Multi-hop associative graph retrieval using Personalized PageRank (PPR)."""

    def __init__(self, embedder: TextEmbedding | None = None) -> None:
        self._embedder = embedder or get_embedder()
        self._triples: list[GraphTriple] = []
        self._nodes: list[str] = []
        self._node_to_idx: dict[str, int] = {}
        self._node_embeddings: dict[str, np.ndarray] = {}

    def _ensure_node(self, name: str, description: str | None = None) -> int:
        if name not in self._node_to_idx:
            self._node_to_idx[name] = len(self._nodes)
            self._nodes.append(name)
            desc = description or name.replace(":", " ").replace("_", " ").replace("/", " ")
            self._node_embeddings[name] = embed_text(f"{name} {desc}", embedder=self._embedder)
        elif description:
            self._node_embeddings[name] = embed_text(f"{name} {description}", embedder=self._embedder)
        return self._node_to_idx[name]

    def add_node_description(self, name: str, description: str) -> None:
        self._ensure_node(name, description=description)

    def add_triple(self, subject: str, relation: str, obj: str) -> None:
        self._ensure_node(subject)
        self._ensure_node(obj)
        triple = GraphTriple(subject=subject, relation=relation.upper(), obj=obj)
        if triple not in self._triples:
            self._triples.append(triple)

    def _build_transition_matrix(self) -> np.ndarray:
        n = len(self._nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        for t in self._triples:
            i = self._node_to_idx[t.subject]
            j = self._node_to_idx[t.obj]
            adj[j, i] += 1.0  # Forward hop
            adj[i, j] += 0.8  # Reverse hop (so upstream commits get credit too)

        col_sums = adj.sum(axis=0)
        for col in range(n):
            adj[:, col] = adj[:, col] / col_sums[col] if col_sums[col] > 0 else (1.0 / n)
        return adj

    def query(
        self,
        query_text: str,
        top_k_seeds: int = 2,
        top_k_results: int = 5,
        damping: float = 0.85,
        iterations: int = 25,
    ) -> HippoRetrievalResult:
        if not self._nodes:
            return HippoRetrievalResult([], [], [])

        q_vec = embed_text(query_text, embedder=self._embedder)
        sim_scores = [
            (node, cosine_similarity(q_vec, self._node_embeddings[node]))
            for node in self._nodes
        ]
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        seeds = sim_scores[:top_k_seeds]

        n = len(self._nodes)
        seed_vec = np.zeros(n, dtype=np.float32)
        for node, score in seeds:
            seed_vec[self._node_to_idx[node]] = max(score, 0.01)
        seed_vec /= seed_vec.sum()

        transition = self._build_transition_matrix()
        p = seed_vec.copy()
        for _ in range(iterations):
            p = (1.0 - damping) * seed_vec + damping * (transition @ p)

        ranked = sorted(
            [(self._nodes[i], round(float(p[i]), 4)) for i in range(n)],
            key=lambda x: x[1],
            reverse=True,
        )[:top_k_results]

        top_nodes = {node for node, _ in ranked}
        triples = [t for t in self._triples if t.subject in top_nodes and t.obj in top_nodes]

        return HippoRetrievalResult(
            seed_nodes=[(name, round(score, 4)) for name, score in seeds],
            ranked_nodes=ranked,
            relevant_triples=triples,
        )

    def to_neo4j_cypher(self) -> list[str]:
        queries: list[str] = []
        for t in self._triples:
            rel = "".join(c if c.isalnum() or c == "_" else "_" for c in t.relation)
            queries.append(
                f"MERGE (a:Entity {{name: '{t.subject}'}}) "
                f"MERGE (b:Entity {{name: '{t.obj}'}}) "
                f"MERGE (a)-[:{rel}]->(b);"
            )
        return queries

    def sync_to_neo4j(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> int:
        uri = uri or os.getenv("NEO4J_URI")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "")
        if not uri:
            raise RuntimeError("NEO4J_URI is not configured.")

        from neo4j import GraphDatabase  # type: ignore[import-not-found]

        cypher_list = self.to_neo4j_cypher()
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                for q in cypher_list:
                    session.run(q)
        return len(cypher_list)
