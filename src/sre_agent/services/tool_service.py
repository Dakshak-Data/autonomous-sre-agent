"""
Tool Service: Dynamic Semantic Tool Registry + Clean Adapter Binding.
Uses helpers/embeddings.py for fast in-memory Cosine Similarity tool filtering.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from fastembed import TextEmbedding

from sre_agent.adapters.github_adapter import GitHubAdapter
from sre_agent.adapters.supabase_adapter import SupabaseAdapter
from sre_agent.adapters.vercel_adapter import VercelAdapter
from sre_agent.helpers.embeddings import cosine_similarity, embed_text, get_embedder
from sre_agent.schemas.models import ToolDefinition
from sre_agent.services.retrieval_service import HybridRetrievalService


class ToolService:
    """Registers tools and dynamically loads ONLY Top-K matching tools per query."""

    def __init__(self, embedder: TextEmbedding | None = None) -> None:
        self._embedder = embedder or get_embedder()
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a Python function as an AI tool."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            vector = embed_text(f"{name}: {description}", embedder=self._embedder)
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                embedding=vector,
            )
            return func

        return decorator

    def get_relevant_tools(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.35,
    ) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
        """Select Top-K relevant tool schemas using Cosine Similarity."""
        if not self._tools:
            return [], []

        query_vec = embed_text(query, embedder=self._embedder)
        scored = [
            (tool, cosine_similarity(query_vec, tool.embedding))
            for tool in self._tools.values()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        selected = [
            tool.to_llm_schema()
            for tool, score in scored[:top_k]
            if score >= min_score
        ]
        debug_scores = [(tool.name, round(score, 4)) for tool, score in scored]
        return selected, debug_scores

    def execute_tool(self, name: str, arguments: dict[str, Any] | str) -> str:
        """Safely execute a registered tool without crashing the agent loop."""
        tool = self._tools.get(name)
        if not tool:
            return f"ERROR: Tool '{name}' is not registered."

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"ERROR: Invalid JSON arguments for '{name}': {exc}"

        try:
            result = tool.func(**arguments)
            return result if isinstance(result, str) else json.dumps(result, indent=2)
        except TypeError as exc:
            return f"ERROR: Wrong arguments passed to '{name}': {exc}"
        except Exception as exc:
            return f"ERROR: Tool '{name}' failed: {exc}"

    def list_all_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


def register_sre_tools(
    tool_service: ToolService,
    vercel: VercelAdapter,
    supabase: SupabaseAdapter,
    github: GitHubAdapter,
    retrieval_service: HybridRetrievalService | None = None,
) -> None:
    """
    Bind explicit Adapter methods into ToolService.
    Ctrl+Click any adapter call below to jump directly to its implementation!
    """

    @tool_service.register(
        name="check_endpoint_health",
        description=(
            "Send an HTTP GET probe to a live Vercel website or API URL to check HTTP status code, "
            "response latency in milliseconds, and server availability."
        ),
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full https:// URL to probe"}},
            "required": ["url"],
        },
    )
    def check_endpoint_health(url: str) -> str:
        return vercel.check_endpoint_health(url=url)

    @tool_service.register(
        name="get_vercel_runtime_logs",
        description=(
            "Fetch recent Vercel deployments, runtime error logs, HTTP 500 crashes, and "
            "504 serverless function timeouts from Vercel API."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Vercel project name or ID"},
                "limit": {"type": "integer", "description": "Max log lines to return"},
            },
            "required": ["project_name"],
        },
    )
    def get_vercel_runtime_logs(project_name: str, limit: int = 15) -> str:
        return vercel.get_runtime_logs(project_name=project_name, limit=limit)

    @tool_service.register(
        name="check_supabase_slow_queries",
        description=(
            "Inspect Supabase PostgreSQL database for slow SQL queries, active locks, "
            "connection exhaustion, and sequential scans caused by missing indexes."
        ),
        parameters={
            "type": "object",
            "properties": {"min_duration_ms": {"type": "integer"}},
            "required": [],
        },
    )
    def check_supabase_slow_queries(min_duration_ms: int = 500) -> str:
        return supabase.check_slow_queries(min_duration_ms=min_duration_ms)

    @tool_service.register(
        name="get_github_recent_commits",
        description=(
            "Fetch recent git commits, commit SHAs, authors, and messages from a GitHub repository "
            "to identify which code deployment introduced a bug."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "GitHub repository in 'owner/repo' format"},
                "count": {"type": "integer"},
            },
            "required": ["repo"],
        },
    )
    def get_github_recent_commits(repo: str, count: int = 5) -> str:
        return github.get_recent_commits(repo=repo, count=count)

    @tool_service.register(
        name="search_incident_logs",
        description=(
            "Run hybrid BM25 + dense vector search across indexed Vercel, Supabase, and system logs "
            "to find exact error codes (e.g. PGRST301, 504) or conceptual failure patterns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    )
    def search_incident_logs(query: str, top_k: int = 3) -> str:
        if retrieval_service is None:
            return "No hybrid log retriever attached."
        hits = retrieval_service.search(query=query, top_k=top_k)
        if not hits:
            return "No matching logs found."
        lines = [f"[HYBRID LOG SEARCH] Top {len(hits)} hits for '{query}':"]
        for h in hits:
            lines.append(
                f"  - [{h.doc.source}] (rrf={h.rrf_score}, bm25={h.bm25_score}, vec={h.vector_score}) "
                f"{h.doc.content}"
            )
        return "\n".join(lines)

    @tool_service.register(
        name="rollback_vercel_deployment",
        description=(
            "DESTRUCTIVE ACTION: Immediately rollback a live Vercel production deployment to the "
            "previous stable build. Requires human approval."
        ),
        parameters={
            "type": "object",
            "properties": {"project_name": {"type": "string"}},
            "required": ["project_name"],
        },
    )
    def rollback_vercel_deployment(project_name: str) -> str:
        return vercel.rollback_deployment(project_name=project_name)
