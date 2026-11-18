"""Routers Layer: Entrypoints (CLI Router now, easy to add FastAPI Router later)."""

from sre_agent.routers.cli_router import run_cli

__all__ = ["run_cli"]
