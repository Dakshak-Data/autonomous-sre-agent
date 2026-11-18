"""
Hybrid Retrieval Service: Vespa Pattern (BM25 + Dense Vector + Reciprocal Rank Fusion).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from fastembed import TextEmbedding

from sre_agent.helpers.embeddings import cosine_similarity, embed_batch, embed_text, get_embedder
from sre_agent.schemas.models import LogDocument, RetrievalHit

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_./:-]+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


class HybridRetrievalService:
    """In-memory BM25 + Dense Vector hybrid search fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        embedder: TextEmbedding | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self._embedder = embedder or get_embedder()
        self._k1 = bm25_k1
        self._b = bm25_b

        self._docs: list[LogDocument] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._embeddings: list[np.ndarray] = []

    def add_documents(self, docs: list[LogDocument]) -> None:
        if not docs:
            return
        texts = [f"[{d.source}] {d.content}" for d in docs]
        vectors = embed_batch(texts, embedder=self._embedder)

        for doc, vec in zip(docs, vectors):
            tokens = _tokenize(f"{doc.source} {doc.content}")
            tf = Counter(tokens)
            for term in tf:
                self._doc_freq[term] += 1

            self._docs.append(doc)
            self._doc_tokens.append(tokens)
            self._doc_term_freqs.append(tf)
            self._embeddings.append(vec)

    def _bm25_scores(self, query: str) -> list[float]:
        n_docs = len(self._docs)
        if n_docs == 0:
            return []

        query_terms = _tokenize(query)
        avg_dl = sum(len(t) for t in self._doc_tokens) / n_docs
        scores = [0.0] * n_docs

        for term in query_terms:
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for idx, tf_map in enumerate(self._doc_term_freqs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = len(self._doc_tokens[idx])
                denom = tf + self._k1 * (1.0 - self._b + self._b * (dl / max(avg_dl, 1.0)))
                scores[idx] += idf * ((tf * (self._k1 + 1.0)) / denom)
        return scores

    def search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> list[RetrievalHit]:
        if not self._docs:
            return []

        n = len(self._docs)
        bm25 = self._bm25_scores(query)
        q_vec = embed_text(query, embedder=self._embedder)
        vec_scores = [cosine_similarity(q_vec, emb) for emb in self._embeddings]

        bm25_order = sorted(range(n), key=lambda i: bm25[i], reverse=True)
        vec_order = sorted(range(n), key=lambda i: vec_scores[i], reverse=True)
        bm25_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_order)}
        vec_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(vec_order)}

        hits: list[RetrievalHit] = []
        for idx in range(n):
            bm25_term = (1.0 / (rrf_k + bm25_rank[idx])) if bm25[idx] > 0 else 0.0
            vec_term = 1.0 / (rrf_k + vec_rank[idx])
            hits.append(
                RetrievalHit(
                    doc=self._docs[idx],
                    rrf_score=round(bm25_term + vec_term, 5),
                    bm25_score=round(bm25[idx], 4),
                    vector_score=round(vec_scores[idx], 4),
                )
            )

        hits.sort(key=lambda h: h.rrf_score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def generate_vespa_schema() -> str:
        """Production Vespa Schema Definition (.sd) for scaling to millions of logs."""
        return """schema sre_logs {
    document sre_logs {
        field doc_id type string { indexing: summary | attribute }
        field source type string { indexing: summary | attribute }
        field content type string { indexing: summary | index }
        field embedding type tensor<float>(x[384]) {
            indexing: attribute | index
            attribute { distance-metric: angular }
        }
    }
    rank-profile hybrid_rrf {
        inputs {
            query(q_emb) tensor<float>(x[384])
        }
        first-phase {
            expression: bm25(content) + closeness(field, embedding)
        }
    }
}"""


def seed_sample_logs(retrieval_service: HybridRetrievalService) -> None:
    """Populate realistic Vercel + Supabase + GitHub logs for local demos."""
    retrieval_service.add_documents(
        [
            LogDocument("log_1", "vercel", "INFO GET /health 200 OK latency=4ms"),
            LogDocument("log_2", "vercel", "ERROR POST /api/checkout HTTP 504 Gateway Timeout after 10000ms"),
            LogDocument("log_3", "vercel", "ERROR PostgREST code PGRST301 JWT expired or upstream DB timeout"),
            LogDocument("log_4", "supabase", "FATAL: sorry, too many clients already (max_connections=50 reached)"),
            LogDocument("log_5", "supabase", "WARN slow query 9400ms: Seq Scan on orders WHERE customer_email=$1"),
            LogDocument("log_6", "github", "Commit a1b2c3d merged PR #42: added customer_email lookup to checkout API"),
        ]
    )
