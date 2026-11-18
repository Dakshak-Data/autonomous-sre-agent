"""
Engram Memory Service: Persistent SQLite Episodic Memory + Semantic Cache.

Features:
  1. Semantic Cache Short-Circuit: If a >=92% similar query was answered recently
     (within TTL), returns the cached resolution directly -> 0 LLM tokens spent!
  2. Deduplication on Write: Updates existing near-identical engrams instead of
     storing duplicate copies that bloat future prompts.
  3. Compact Prompt Formatting: Keeps recalled memory snippets concise (~50 tokens).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from sre_agent.config.constants import (
    DEFAULT_ENGRAM_DB,
    DEFAULT_SEMANTIC_CACHE_THRESHOLD,
    DEFAULT_SEMANTIC_CACHE_TTL_SECONDS,
)
from sre_agent.helpers.embeddings import cosine_similarity, embed_text, get_embedder
from sre_agent.schemas.models import MemoryRecall


class EngramMemoryService:
    """Stores resolved incidents in SQLite with semantic caching and deduplication."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_ENGRAM_DB,
        embedder: TextEmbedding | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self._embedder = embedder or get_embedder()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS engrams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_query TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                resolution TEXT NOT NULL,
                affected_entities TEXT NOT NULL,
                embedding BLOB NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def store_episode(
        self,
        incident_query: str,
        root_cause: str,
        resolution: str,
        affected_entities: list[str] | None = None,
        dedup_threshold: float = DEFAULT_SEMANTIC_CACHE_THRESHOLD,
    ) -> int:
        """
        Save a resolved incident. If a near-identical incident already exists,
        UPDATE it in-place instead of creating duplicate bloat rows!
        """
        entities = affected_entities or []
        q_vec = embed_text(incident_query, embedder=self._embedder)
        now_iso = datetime.now(timezone.utc).isoformat()

        # Check for existing duplicate by query vector similarity
        existing_rows = self._conn.execute("SELECT id, embedding FROM engrams").fetchall()
        for row_id, emb_blob in existing_rows:
            existing_vec = np.frombuffer(emb_blob, dtype=np.float32)
            if cosine_similarity(q_vec, existing_vec) >= dedup_threshold:
                self._conn.execute(
                    """
                    UPDATE engrams
                    SET timestamp = ?, root_cause = ?, resolution = ?, affected_entities = ?
                    WHERE id = ?
                    """,
                    (now_iso, root_cause, resolution, json.dumps(entities), row_id),
                )
                self._conn.commit()
                return int(row_id)

        cur = self._conn.execute(
            """
            INSERT INTO engrams
                (timestamp, incident_query, root_cause, resolution, affected_entities, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now_iso, incident_query, root_cause, resolution, json.dumps(entities), q_vec.tobytes()),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def check_semantic_cache(
        self,
        query: str,
        min_score: float = DEFAULT_SEMANTIC_CACHE_THRESHOLD,
        ttl_seconds: int = DEFAULT_SEMANTIC_CACHE_TTL_SECONDS,
    ) -> MemoryRecall | None:
        """
        Fast-path Semantic Cache: returns a fresh cached MemoryRecall if a nearly identical
        query (>= min_score) was answered within the last `ttl_seconds` -> 0 LLM tokens!
        """
        recalls = self.recall(query, top_k=1, min_score=min_score)
        if not recalls:
            return None

        best = recalls[0]
        try:
            saved_at = datetime.fromisoformat(best.timestamp)
            age_seconds = (datetime.now(timezone.utc) - saved_at).total_seconds()
            if age_seconds <= ttl_seconds:
                return best
        except ValueError:
            return None
        return None

    def recall(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.50,
    ) -> list[MemoryRecall]:
        rows = self._conn.execute(
            """
            SELECT id, timestamp, incident_query, root_cause, resolution,
                   affected_entities, embedding, recall_count
            FROM engrams
            """
        ).fetchall()
        if not rows:
            return []

        q_vec = embed_text(query, embedder=self._embedder)
        matches: list[MemoryRecall] = []

        for ep_id, ts, inc_q, rc, res, ent_json, emb_blob, r_count in rows:
            ep_vec = np.frombuffer(emb_blob, dtype=np.float32)
            score = cosine_similarity(q_vec, ep_vec)
            if score >= min_score:
                matches.append(
                    MemoryRecall(
                        id=ep_id,
                        timestamp=ts,
                        incident_query=inc_q,
                        root_cause=rc,
                        resolution=res,
                        affected_entities=json.loads(ent_json),
                        score=round(score, 4),
                        recall_count=r_count + 1,
                    )
                )

        matches.sort(key=lambda m: m.score, reverse=True)
        top_matches = matches[:top_k]

        for m in top_matches:
            self._conn.execute("UPDATE engrams SET recall_count = recall_count + 1 WHERE id = ?", (m.id,))
        if top_matches:
            self._conn.commit()

        return top_matches

    @staticmethod
    def format_for_prompt(recalls: list[MemoryRecall], max_chars_per_item: int = 220) -> str:
        """Compact memory summary so injected context never bloats token usage."""
        if not recalls:
            return ""
        lines = ["[ENGRAM MEMORY — COMPACT PAST INCIDENTS]"]
        for m in recalls:
            compact_summary = m.resolution.replace("\n", " ").strip()[:max_chars_per_item]
            lines.append(
                f"- (sim={m.score:.2f}) Q: {m.incident_query[:80]} | Finding: {compact_summary}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
