"""
Incident Controller: Orchestrates the 4-Step Durable SRE Investigation Pipeline.

Flow (every call is directly Ctrl+Clickable to its Service or Adapter!):
  Step 1 -> EngramMemoryService.recall()        (Past incident memory)
  Step 2 -> HippoGraphService.query()           (Multi-hop topology discovery)
  Step 3 -> AgentService.run()                  (Dynamic Tools + LLMAdapter + ContextService)
  Step 4 -> EngramMemoryService.store_episode() (Save new learning to SQLite)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

from sre_agent.adapters.durable_adapter import DurableWorkflowRunner
from sre_agent.adapters.github_adapter import GitHubAdapter
from sre_agent.adapters.llm_adapter import LLMAdapter
from sre_agent.adapters.supabase_adapter import SupabaseAdapter
from sre_agent.adapters.vercel_adapter import VercelAdapter
from sre_agent.config.constants import (
    DEFAULT_ENGRAM_DB,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_SRE_SYSTEM_PROMPT,
    DEFAULT_TOP_K_TOOLS,
    DEFAULT_WORKFLOW_DB,
)
from sre_agent.helpers.embeddings import get_embedder
from sre_agent.schemas.models import HarnessResult
from sre_agent.services.agent_service import AgentService
from sre_agent.services.context_service import ContextService
from sre_agent.services.graph_service import HippoGraphService
from sre_agent.services.memory_service import EngramMemoryService
from sre_agent.services.retrieval_service import HybridRetrievalService, seed_sample_logs
from sre_agent.services.tool_service import ToolService, register_sre_tools


def build_default_topology() -> HippoGraphService:
    """Seed the application topology graph dynamically from environment variables."""
    load_dotenv()
    app_name = os.getenv("TARGET_APP_NAME", "my-web-app").strip()
    app_url = os.getenv("TARGET_APP_URL", "https://my-web-app.vercel.app").strip()
    repo_name = os.getenv("TARGET_GITHUB_REPO", "your-org/my-web-app").strip()
    raw_tables = os.getenv("TARGET_DB_TABLES", "users,orders,payments,sessions")
    tables = [t.strip() for t in raw_tables.split(",") if t.strip()]

    graph = HippoGraphService(embedder=get_embedder())
    repo_node = f"repo:{repo_name}"
    app_node = f"app:{app_name}-vercel"

    graph.add_node_description(repo_node, f"GitHub repository for {app_name}")
    graph.add_node_description(app_node, f"Live Vercel production deployment at {app_url}")
    graph.add_triple(repo_node, "DEPLOYS_TO", app_node)

    for tbl in tables:
        tbl_node = f"table:{tbl}"
        graph.add_node_description(tbl_node, f"Supabase PostgreSQL database table '{tbl}'")
        graph.add_triple(app_node, "QUERIES", tbl_node)

    return graph


def build_offline_mock_llm() -> LLMAdapter:
    """Zero-token mock LLM adapter for offline demos and CI tests."""
    step_state = {"n": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        step_state["n"] += 1
        step = step_state["n"]
        if step == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_vercel_runtime_logs",
                                            "arguments": '{"project_name": "my-vercel-app"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 210, "completion_tokens": 22},
                },
            )
        if step == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "check_supabase_slow_queries",
                                            "arguments": '{"min_duration_ms": 500}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 290, "completion_tokens": 20},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "INCIDENT ROOT CAUSE REPORT:\n"
                                "1. Vercel `/api/checkout` hit HTTP 504 Gateway Timeout (10s limit).\n"
                                "2. Supabase `orders` table is doing a 9.4s Sequential Scan on `customer_email`.\n"
                                "3. HippoRAG linked this to GitHub commit `a1b2c3d` (PR #42).\n"
                                "RECOMMENDED FIX:\n"
                                "  CREATE INDEX CONCURRENTLY idx_orders_customer_email ON orders(customer_email);"
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 340, "completion_tokens": 75},
            },
        )

    return LLMAdapter.from_env(
        http_client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
        api_keys_override={"GEMINI_API_KEY": "mock-g", "GROQ_API_KEY": "mock-q"},
    )


class IncidentController:
    """Handles SRE investigation requests from CLI Router (or future HTTP/FastAPI Router)."""

    def __init__(self, data_dir: str | Path = ".") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def investigate(
        self,
        query: str,
        workflow_id: str | None = None,
        live_llm: bool = False,
        live_cloud: bool = False,
        auto_approve: bool = False,
        use_semantic_cache: bool = True,
    ) -> HarnessResult:
        """Run or resume a durable SRE investigation workflow."""
        wf_id = workflow_id or f"inc-{uuid.uuid4().hex[:8]}"
        embedder = get_embedder()

        # 1. Initialize Adapters & Services via clean explicit composition
        durable = DurableWorkflowRunner(db_path=self.data_dir / DEFAULT_WORKFLOW_DB)
        memory = EngramMemoryService(db_path=self.data_dir / DEFAULT_ENGRAM_DB, embedder=embedder)

        print(f"\n=== AUTONOMOUS SRE AGENT | Workflow ID: {wf_id} | LLM: {'LIVE' if live_llm else 'MOCK (0 tokens)'} ===\n")

        try:
            # FAST-PATH: Semantic Cache Hit -> 0 LLM Tokens Spent on Repeated Queries!
            if use_semantic_cache and workflow_id is None:
                cached = memory.check_semantic_cache(query)
                if cached is not None:
                    print(
                        f"[ENGRAM SEMANTIC CACHE HIT] Similarity={cached.score:.2f} "
                        f"(recalled {cached.recall_count}x) -> Returning cached report with 0 LLM tokens!"
                    )
                    return HarnessResult(
                        final_answer=f"[CACHED FROM ENGRAM MEMORY — 0 TOKENS USED]\n\n{cached.resolution}",
                        steps_taken=0,
                        tools_injected=[],
                        tools_executed=cached.affected_entities,
                        models_used=["engram/semantic-cache (0 tokens)"],
                        total_llm_tokens=0,
                    )

            graph = build_default_topology()
            retrieval = HybridRetrievalService(embedder=embedder)
            seed_sample_logs(retrieval)

            tools = ToolService(embedder=embedder)
            register_sre_tools(
                tool_service=tools,
                vercel=VercelAdapter(live=live_cloud),
                supabase=SupabaseAdapter(live=live_cloud),
                github=GitHubAdapter(live=live_cloud),
                retrieval_service=retrieval,
            )

            llm = LLMAdapter.from_env() if live_llm else build_offline_mock_llm()

            # Durable Activity 1: Engram Long-Term Memory Recall (compact ~50 tokens!)
            memory_context = durable.run_activity(
                wf_id,
                "step_1_engram_recall",
                lambda: EngramMemoryService.format_for_prompt(
                    memory.recall(query, top_k=1, min_score=0.55)
                ),
            )

            # Durable Activity 2: HippoRAG Multi-Hop Topology Discovery
            graph_context = durable.run_activity(
                wf_id,
                "step_2_hipporag_ppr",
                lambda: graph.query(query, top_k_seeds=1, top_k_results=5).format_for_prompt(),
            )

            # Build Working Memory & Agent Service
            system_prompt = "\n\n".join(
                part for part in [DEFAULT_SRE_SYSTEM_PROMPT, memory_context, graph_context] if part
            )
            context = ContextService(system_prompt=system_prompt, max_context_tokens=DEFAULT_MAX_CONTEXT_TOKENS)
            agent = AgentService(tools=tools, llm=llm, context=context, top_k_tools=DEFAULT_TOP_K_TOOLS)

            def ask_human_approval(tool_name: str, args_json: str) -> bool:
                if auto_approve:
                    return True
                ans = input(f"\n[APPROVAL REQUIRED] Run destructive tool '{tool_name}({args_json})'? [y/N]: ")
                return ans.strip().lower() == "y"

            # Durable Activity 3: Run Core Agentic Harness Loop
            raw_result = durable.run_activity(
                wf_id,
                "step_3_agent_harness",
                lambda: asdict(agent.run(user_query=query, approval_callback=ask_human_approval)),
            )
            result = HarnessResult(**raw_result)

            # Durable Activity 4: Save Episode to Engram Memory
            durable.run_activity(
                wf_id,
                "step_4_engram_store",
                lambda: memory.store_episode(
                    incident_query=query,
                    root_cause=result.final_answer[:300],
                    resolution=result.final_answer,
                    affected_entities=result.tools_executed,
                ),
            )

            return result
        finally:
            memory.close()
            durable.close()
