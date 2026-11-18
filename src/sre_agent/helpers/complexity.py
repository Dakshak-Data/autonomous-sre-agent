"""
Semantic Complexity Router Helper (Zero-Hardcoded Keywords, Zero API Tokens!).

How it works:
  1. Uses our shared local ONNX embedder (`helpers/embeddings.py`) to compare the
     user's query against two semantic concept anchors:
       - EASY_ANCHOR    -> Simple status checks, URL pings, commit lists, table counts
       - COMPLEX_ANCHOR -> Crash debugging, root-cause analysis, slow DB, rollbacks, fixes
  2. Routing Rules:
       - Confident EASY match (`easy_score > complex_score` AND `easy_score >= 0.55`)
         -> tier="fast" (Groq primary, Gemini fallback)
       - COMPLEX match (`complex_score >= easy_score`)
         -> tier="smart" (Gemini primary, Groq fallback)
       - UNKNOWN / Ambiguous (`easy_score < 0.55` and `complex_score < 0.55`)
         -> Safe SRE Default: tier="smart" (Gemini!)
       - Runtime Escalation: If >=2 tools already ran, upgrade to tier="smart" (Gemini!).
"""

from __future__ import annotations

import numpy as np

from sre_agent.helpers.embeddings import cosine_similarity, embed_text

_EASY_ANCHOR = (
    "check website health status, ping live URL, list recent git commits, "
    "show database tables, count rows, quick summary"
)
_COMPLEX_ANCHOR = (
    "investigate production outage, diagnose 500 or 504 timeout crash, find root cause "
    "across services, debug slow PostgreSQL query locks or missing indexes, rollback deployment, fix bug"
)

_MIN_CONFIDENT_EASY_SCORE = 0.55

_EASY_VEC: np.ndarray | None = None
_COMPLEX_VEC: np.ndarray | None = None


def _get_anchor_vectors() -> tuple[np.ndarray, np.ndarray]:
    global _EASY_VEC, _COMPLEX_VEC
    if _EASY_VEC is None or _COMPLEX_VEC is None:
        _EASY_VEC = embed_text(_EASY_ANCHOR)
        _COMPLEX_VEC = embed_text(_COMPLEX_ANCHOR)
    return _EASY_VEC, _COMPLEX_VEC


def classify_task_tier(query: str, step: int = 1, tools_executed_so_far: int = 0) -> tuple[str, str]:
    """
    Semantically classify task complexity into 'fast' (Groq) or 'smart' (Gemini).
    Zero hardcoded keywords, zero API tokens.
    """
    # Runtime escalation: multi-step investigation automatically upgrades to SMART
    if tools_executed_so_far >= 2:
        return "smart", f"runtime escalation after {tools_executed_so_far} tool steps"

    easy_vec, complex_vec = _get_anchor_vectors()
    query_vec = embed_text(query)

    easy_score = round(cosine_similarity(query_vec, easy_vec), 3)
    complex_score = round(cosine_similarity(query_vec, complex_vec), 3)

    # Confident simple task -> Groq FAST tier
    if easy_score > complex_score and easy_score >= _MIN_CONFIDENT_EASY_SCORE:
        return "fast", f"semantic easy={easy_score} > complex={complex_score}"

    # Unknown / low-confidence query -> Safe SRE Default SMART (Gemini)
    if easy_score < _MIN_CONFIDENT_EASY_SCORE and complex_score < _MIN_CONFIDENT_EASY_SCORE:
        return "smart", f"unknown task (easy={easy_score}, complex={complex_score}) -> safe default SMART"

    # Complex task -> Gemini SMART tier
    return "smart", f"semantic complex={complex_score} >= easy={easy_score}"
