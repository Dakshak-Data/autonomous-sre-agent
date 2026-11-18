"""
Step 3 Demo: ZERO-TOKEN End-to-End Autonomous SRE Harness Test!

Wires together Module 1 + Module 2 + Module 3 and proves EVERY feature offline:
  1. Dynamic Tool Loading (only 2 of 5 tools injected into LLM prompt)
  2. Multi-step Tool Execution Loop (Step 1: Vercel Logs -> Step 2: Supabase DB -> Step 3: Final Report)
  3. Huge Log Head+Tail Truncation (2,500+ token Vercel log trimmed cleanly)
  4. Automatic Context Compaction using FAST tier model when budget crosses 75%
  5. Human-in-the-Loop Safety Guardrail blocking unapproved Vercel Rollback!
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.adapters.llm_adapter import LLMAdapter
from sre_agent.config.constants import DEFAULT_SRE_SYSTEM_PROMPT
from sre_agent.services.agent_service import AgentService
from sre_agent.services.context_service import ContextService
from sre_agent.services.tool_service import ToolService


def build_registry() -> ToolService:
    registry = ToolService()

    @registry.register(
        name="get_vercel_runtime_logs",
        description="Fetch runtime error logs, HTTP 500 crashes, and 504 timeouts from Vercel.",
        parameters={
            "type": "object",
            "properties": {"deployment_id": {"type": "string"}},
            "required": ["deployment_id"],
        },
    )
    def get_vercel_runtime_logs(deployment_id: str) -> str:
        # Simulate a MASSIVE noisy production log (~2,500 tokens!)
        header = f"=== VERCEL DEPLOYMENT {deployment_id} RUNTIME LOGS ===\n"
        noise = "\n".join(
            f"2026-11-18T03:10:{i:02d}Z INFO GET /health 200 OK latency=4ms req_id=req_{i:04d}"
            for i in range(120)
        )
        crash_tail = (
            "\n2026-11-18T03:14:58Z ERROR POST /api/checkout -> HTTP 504 Gateway Timeout "
            "(Function exceeded 10,000ms waiting on Supabase Postgres query!)"
        )
        return header + noise + crash_tail

    @registry.register(
        name="check_supabase_slow_queries",
        description="Inspect Supabase PostgreSQL database for slow SQL queries and missing indexes.",
        parameters={
            "type": "object",
            "properties": {"min_duration_ms": {"type": "integer"}},
            "required": [],
        },
    )
    def check_supabase_slow_queries(min_duration_ms: int = 500) -> str:
        return (
            f"Found slow query (> {min_duration_ms}ms): "
            "SELECT * FROM orders WHERE customer_email = $1 | avg_time=9,400ms | "
            "Seq Scan on orders (1.2M rows) — MISSING INDEX on orders(customer_email)!"
        )

    @registry.register(
        name="rollback_vercel_deployment",
        description="Trigger an instant destructive production rollback on Vercel to previous build.",
        parameters={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
    )
    def rollback_vercel_deployment(project_id: str) -> str:
        return f"Rolled back Vercel project {project_id}."

    @registry.register(
        name="get_github_recent_commits",
        description="Fetch recent git commits and pull requests from GitHub.",
        parameters={"type": "object", "properties": {"repo": {"type": "string"}}},
    )
    def get_github_recent_commits(repo: str) -> str:
        return f"Latest commit on {repo}: added customer_email lookup."

    @registry.register(
        name="check_ssl_certificate",
        description="Check HTTPS SSL/TLS certificate expiry for a domain.",
        parameters={"type": "object", "properties": {"domain": {"type": "string"}}},
    )
    def check_ssl_certificate(domain: str) -> str:
        return f"SSL valid for {domain}."

    return registry


def main() -> None:
    turn_counter = {"smart_calls": 0}

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        model = body["model"]

        # If this is a FAST-tier compaction call (Groq Qwen or Gemini Flash-Lite):
        if model in ("qwen/qwen3.8-27b", "gemini-2.5-flash-lite"):
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "- Vercel /api/checkout hit 504 Gateway Timeout (10s limit).\n"
                                    "- Supabase orders table is doing a slow 9.4s Seq Scan on customer_email."
                                ),
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 35},
                },
            )

        # Smart-tier SRE investigation sequence:
        turn_counter["smart_calls"] += 1
        step = turn_counter["smart_calls"]

        if step == 1:
            # Step 1: Call Vercel Logs tool
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_vercel_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_vercel_runtime_logs",
                                            "arguments": '{"deployment_id": "dpl_prod_99"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 220, "completion_tokens": 24},
                },
            )

        if step == 2:
            # Step 2: Call Supabase Slow Query tool
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_supa_2",
                                        "type": "function",
                                        "function": {
                                            "name": "check_supabase_slow_queries",
                                            "arguments": '{"min_duration_ms": 1000}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 680, "completion_tokens": 22},
                },
            )

        # Step 3: Final Root-Cause Answer!
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "ROOT CAUSE CONFIRMED:\n"
                                "1. Vercel `/api/checkout` is timing out (HTTP 504 at 10s).\n"
                                "2. Supabase `orders` table is doing a 9.4s Sequential Scan on `customer_email`.\n"
                                "RECOMMENDED FIX:\n"
                                "  CREATE INDEX CONCURRENTLY idx_orders_customer_email ON orders(customer_email);"
                            ),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 310, "completion_tokens": 65},
            },
        )

    print("Setting up SRE Harness (0 real API tokens will be spent)...\n")
    registry = build_registry()
    mock_client = httpx.Client(transport=httpx.MockTransport(mock_llm_handler))
    router = LLMAdapter.from_env(
        http_client=mock_client,
        api_keys_override={"GEMINI_API_KEY": "mock-g", "GROQ_API_KEY": "mock-q"},
    )
    # Intentionally tight budget (900 tokens) so we can watch Auto-Compaction trigger LIVE!
    context = ContextService(
        system_prompt=DEFAULT_SRE_SYSTEM_PROMPT,
        max_context_tokens=900,
        max_tool_output_tokens=450,
        compaction_threshold=0.75,
        keep_recent_messages=2,
    )

    harness = AgentService(tools=registry, llm=router, context=context, top_k_tools=2)

    print("=" * 75)
    print("TEST A: Multi-Step Autonomous Investigation + Truncation + Auto-Compaction")
    print("=" * 75)
    result = harness.run(
        "My Vercel production app has 504 timeouts and Supabase database queries are very slow. Find the root cause."
    )
    print("\n--- FINAL AGENT REPORT ---")
    print(result.final_answer)
    print("\n--- HARNESS TELEMETRY ---")
    print(f"  Steps Taken               : {result.steps_taken}")
    print(f"  Dynamic Tools Loaded      : {result.tools_injected} (out of {len(registry)})")
    print(f"  Tools Executed            : {result.tools_executed}")
    print(f"  Tokens Saved (Log Trim)   : ~{result.tokens_saved_by_truncation} tokens")
    print(f"  Auto-Compactions Run      : {result.compactions_triggered}")
    print(f"  Tokens Saved (Compaction) : ~{result.tokens_saved_by_compaction} tokens")

    print("\n" + "=" * 75)
    print("TEST B: Human-in-the-Loop Safety Guardrail (Destructive Rollback Test)")
    print("=" * 75)

    def mock_rollback_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        has_tool_result = any(m["role"] == "tool" for m in body["messages"])
        if not has_tool_result:
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_rb_1",
                                        "type": "function",
                                        "function": {
                                            "name": "rollback_vercel_deployment",
                                            "arguments": '{"project_id": "prj_live_1"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                },
            )
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Understood — rollback was cancelled because human approval was denied.",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 130, "completion_tokens": 15},
            },
        )

    rb_router = LLMAdapter.from_env(
        http_client=httpx.Client(transport=httpx.MockTransport(mock_rollback_handler)),
        api_keys_override={"GEMINI_API_KEY": "mock-g"},
    )
    rb_harness = AgentService(tools=registry, llm=rb_router, top_k_tools=2)
    rb_res = rb_harness.run(
        "Rollback the broken Vercel deployment immediately!",
        approval_callback=lambda name, args: False,  # Human says NO!
    )
    print(f"  Final Response: {rb_res.final_answer}")


if __name__ == "__main__":
    main()
