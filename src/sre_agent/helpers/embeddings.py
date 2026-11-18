"""
Embedding Helper: Shared local ONNX embedder + Cosine Similarity math.
Loaded once and reused across ToolService, MemoryService, GraphService, and RetrievalService!
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from sre_agent.config.constants import DEFAULT_EMBED_MODEL

_SHARED_EMBEDDER: TextEmbedding | None = None


def get_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> TextEmbedding:
    """Return a shared local TextEmbedding instance so the model is loaded only once."""
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = TextEmbedding(model_name=model_name)
    return _SHARED_EMBEDDER


def embed_text(text: str, embedder: TextEmbedding | None = None) -> np.ndarray:
    """Convert a string into an L2-normalized NumPy float32 vector."""
    model = embedder or get_embedder()
    vec = next(model.embed([text]))
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def embed_batch(texts: list[str], embedder: TextEmbedding | None = None) -> list[np.ndarray]:
    """Convert multiple strings into L2-normalized NumPy float32 vectors in one batch."""
    if not texts:
        return []
    model = embedder or get_embedder()
    vectors: list[np.ndarray] = []
    for vec in model.embed(texts):
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        vectors.append(arr / norm if norm > 0 else arr)
    return vectors


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Exact Cosine Similarity for L2-normalized vectors (simple dot product)."""
    return float(np.dot(vec_a, vec_b))
