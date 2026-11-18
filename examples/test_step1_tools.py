"""
Step 1 Demo: Dynamic Tool Loading in Action!

Run this script to see:
  1. 6 different SRE tools registered in memory.
  2. Different user queries tested against the registry.
  3. Only the relevant Top-2 tools selected (saving ~65% schema tokens).
  4. Safe execution of the selected tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src/ to Python path so we can import sre_agent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.services.tool_service import ToolService


def build_demo_registry() -> ToolService:
    registry = ToolService()

    @registry.register(
        name="get_vercel_runtime_logs",
        description=(
            "Fetch runtime error logs, HTTP 500 crashes, and 504 serverless function "
            "timeouts from a live Vercel deployment."
        ),
        parameters={
            "type": "object",
            "properties": {
                "deployment_id": {"type": "string", "description": "Vercel deployment ID"},
                "limit": {"type": "integer", "description": "Number of recent log lines"},
            },
            "required": ["deployment_id"],
        },
    )
    def get_vercel_runtime_logs(deployment_id: str, limit: int = 20) -> str:
        return f"[MOCK VERCEL] Found 504 Timeout on /api/users in deployment {deployment_id} (last {limit} lines)."

    @registry.register(
        name="check_supabase_slow_queries",
        description=(
            "Inspect Supabase PostgreSQL database for slow SQL queries, missing table indexes, "
            "high CPU latency, and blocked table locks."
        ),
        parameters={
            "type": "object",
            "properties": {
                "min_duration_ms": {"type": "integer", "description": "Minimum query time in ms"},
            },
            "required": [],
        },
    )
    def check_supabase_slow_queries(min_duration_ms: int = 500) -> str:
        return (
            f"[MOCK SUPABASE] Slow query > {min_duration_ms}ms found: "
            "SELECT * FROM users WHERE email = $1 (took 8,200ms, missing index on users.email)."
        )

    @registry.register(
        name="get_github_recent_commits",
        description=(
            "Fetch recent git commits, pull requests, author names, and code diffs "
            "from the connected GitHub repository."
        ),
        parameters={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "GitHub owner/repo name"},
                "count": {"type": "integer", "description": "Number of commits to fetch"},
            },
            "required": ["repo"],
        },
    )
    def get_github_recent_commits(repo: str, count: int = 5) -> str:
        return f"[MOCK GITHUB] Latest {count} commits on {repo}: #a1b2c3 'Added unindexed email search filter'."

    @registry.register(
        name="rollback_vercel_deployment",
        description=(
            "Trigger an instant production rollback on Vercel to restore the previous "
            "working build after a broken release."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_id": {"type": "string", "description": "Vercel project ID"},
            },
            "required": ["project_id"],
        },
    )
    def rollback_vercel_deployment(project_id: str) -> str:
        return f"[MOCK VERCEL] Rolled back project {project_id} to previous stable deployment."

    @registry.register(
        name="check_ssl_certificate",
        description=(
            "Check HTTPS SSL/TLS certificate expiration date, domain DNS records, "
            "and domain security health."
        ),
        parameters={
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain name to inspect"},
            },
            "required": ["domain"],
        },
    )
    def check_ssl_certificate(domain: str) -> str:
        return f"[MOCK SSL] Certificate for {domain} is valid for 64 more days."

    @registry.register(
        name="create_github_pull_request",
        description=(
            "Create a new git branch, apply a code patch, and open a Pull Request "
            "on GitHub to fix a confirmed software bug."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Pull request title"},
                "branch": {"type": "string", "description": "Branch name"},
            },
            "required": ["title", "branch"],
        },
    )
    def create_github_pull_request(title: str, branch: str) -> str:
        return f"[MOCK GITHUB] Opened PR '{title}' from branch '{branch}'."

    return registry


def main() -> None:
    print("Initializing Dynamic Tool Registry (loading lightweight local embedder)...")
    registry = build_demo_registry()
    print(f"Registered {len(registry)} tools.\n")

    all_schemas_chars = len(json.dumps(registry.list_all_schemas()))

    test_queries = [
        "My Supabase database queries are very slow and locking tables",
        "Production website on Vercel is throwing 504 timeout errors",
        "Check who pushed the latest code changes on GitHub",
    ]

    for query in test_queries:
        print("=" * 75)
        print(f"USER QUERY: {query}")
        selected_schemas, scores = registry.get_relevant_tools(query, top_k=2, min_score=0.45)

        print("\nSimilarity Scores across ALL 6 tools:")
        for tool_name, score in scores:
            marker = " <-- SELECTED" if any(s["function"]["name"] == tool_name for s in selected_schemas) else ""
            print(f"  {score:.4f} | {tool_name}{marker}")

        selected_chars = len(json.dumps(selected_schemas))
        saved_pct = (1 - selected_chars / all_schemas_chars) * 100
        print(f"\nContext Saved: ~{saved_pct:.1f}% fewer tool-schema tokens sent to LLM!")

    print("\n" + "=" * 75)
    print("Testing Safe Tool Execution:")
    output = registry.execute_tool("check_supabase_slow_queries", {"min_duration_ms": 1000})
    print(f"  Result: {output}")

    bad_output = registry.execute_tool("non_existent_tool", {})
    print(f"  Error Handling Test: {bad_output}")


if __name__ == "__main__":
    main()
