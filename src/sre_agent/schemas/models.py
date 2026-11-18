"""
Schemas / Data Models: Plain Python dataclasses shared across layers.
Zero inheritance, zero hidden magic — easy to read and autocomplete in any IDE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ToolDefinition:
    """Metadata, JSON schema, Python callable, and embedding vector for one tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    embedding: np.ndarray

    def to_llm_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ModelEndpoint:
    """One LLM provider + model slot in the failover router pool."""

    provider: str
    model: str
    base_url: str
    api_key_env: str
    tier: str = "smart"
    cooldown_until: float = 0.0
    failure_count: int = 0

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"

    def is_healthy(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return current >= self.cooldown_until

    def mark_failed(self, cooldown_seconds: float) -> None:
        self.failure_count += 1
        self.cooldown_until = time.time() + cooldown_seconds

    def mark_healthy(self) -> None:
        self.failure_count = 0
        self.cooldown_until = 0.0


@dataclass
class ToolCallRequest:
    """Normalized representation of an LLM function/tool call."""

    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    """Standardized response returned by LLMAdapter regardless of provider."""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class HarnessResult:
    """Complete audit report produced after an agent investigation finishes."""

    final_answer: str
    steps_taken: int
    tools_injected: list[str] = field(default_factory=list)
    tools_executed: list[str] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    total_llm_tokens: int = 0
    compactions_triggered: int = 0
    tokens_saved_by_truncation: int = 0
    tokens_saved_by_compaction: int = 0


@dataclass
class MemoryRecall:
    """One recalled past incident episode from Engram SQLite memory."""

    id: int
    timestamp: str
    incident_query: str
    root_cause: str
    resolution: str
    affected_entities: list[str]
    score: float
    recall_count: int


@dataclass(frozen=True)
class GraphTriple:
    """Directed edge in the HippoRAG system topology knowledge graph."""

    subject: str
    relation: str
    obj: str

    def to_text(self) -> str:
        return f"({self.subject}) -[{self.relation}]-> ({self.obj})"


@dataclass
class HippoRetrievalResult:
    """Multi-hop Personalized PageRank output from HippoGraphService."""

    seed_nodes: list[tuple[str, float]]
    ranked_nodes: list[tuple[str, float]]
    relevant_triples: list[GraphTriple]

    def format_for_prompt(self) -> str:
        lines = ["[HIPPORAG MULTI-HOP TOPOLOGY CONTEXT]"]
        lines.append("Top Linked Entities (Personalized PageRank):")
        for node, score in self.ranked_nodes:
            lines.append(f"  - {node} (ppr_score={score:.4f})")
        lines.append("Connected System Paths:")
        for t in self.relevant_triples:
            lines.append(f"  * {t.to_text()}")
        return "\n".join(lines)


@dataclass
class LogDocument:
    """One indexed log line or commit record for Hybrid Search."""

    doc_id: str
    source: str
    content: str


@dataclass
class RetrievalHit:
    """One ranked result returned by HybridRetrievalService."""

    doc: LogDocument
    rrf_score: float
    bm25_score: float
    vector_score: float
