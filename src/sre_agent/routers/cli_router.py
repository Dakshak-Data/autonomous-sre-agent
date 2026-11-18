"""
CLI Router: Parses terminal commands & flags and delegates directly to IncidentController.
"""

from __future__ import annotations

import argparse

from sre_agent.controllers.incident_controller import IncidentController


def run_cli() -> None:
    """Parse CLI arguments and dispatch to IncidentController.investigate()."""
    parser = argparse.ArgumentParser(
        prog="sre-agent",
        description="Autonomous SRE & Production Incident Investigator CLI",
    )
    parser.add_argument("query", help="Incident description or question to investigate")
    parser.add_argument("--live", action="store_true", help="Use live Gemini/Groq LLM API keys from .env")
    parser.add_argument("--live-cloud", action="store_true", help="Query real Vercel/Supabase/GitHub APIs")
    parser.add_argument("--workflow-id", default=None, help="Workflow ID to start or resume from checkpoint")
    parser.add_argument("--auto-approve", action="store_true", help="Skip interactive confirmation prompts")
    args = parser.parse_args()

    controller = IncidentController()
    result = controller.investigate(
        query=args.query,
        workflow_id=args.workflow_id,
        live_llm=args.live,
        live_cloud=args.live_cloud,
        auto_approve=args.auto_approve,
    )

    print("\n" + "=" * 75)
    print("FINAL SRE REPORT")
    print("=" * 75)
    print(result.final_answer)
    print("\n" + "-" * 75)
    print(
        f"Stats | Steps={result.steps_taken} | ToolsLoaded={result.tools_injected} | "
        f"Models={result.models_used} | Tokens={result.total_llm_tokens}"
    )
