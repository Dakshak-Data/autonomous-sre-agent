"""Entrypoint for `python -m sre_agent`."""

from sre_agent.routers.cli_router import run_cli

if __name__ == "__main__":
    run_cli()
