"""
GitHub Adapter: Clean, plain Python class for GitHub REST API commit inspection.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from sre_agent.config.constants import GITHUB_API_BASE_URL


class GitHubAdapter:
    """Adapter for inspecting recent GitHub commits and deployment changes."""

    def __init__(self, token: str | None = None, live: bool = False) -> None:
        self.token = (token if token is not None else os.getenv("GITHUB_TOKEN", "")).strip()
        self.live = live

    def get_recent_commits(self, repo: str, count: int = 5) -> str:
        """Fetch latest commits from a GitHub repo ('owner/repo')."""
        if not self.live:
            return (
                f"[SANDBOX GITHUB] Recent {count} commits on {repo}:\n"
                "  - a1b2c3d (2h ago) feat(checkout): lookup orders by customer_email filter\n"
                "  - f9e8d7c (1d ago) chore: upgrade nextjs dependencies"
            )

        headers: dict[str, Any] = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        resp = httpx.get(
            f"{GITHUB_API_BASE_URL}/repos/{repo}/commits",
            headers=headers,
            params={"per_page": count},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return f"[LIVE GITHUB ERROR] HTTP {resp.status_code}: {resp.text[:200]}"

        lines = [f"[LIVE GITHUB] Latest commits on {repo}:"]
        for item in resp.json():
            sha = item["sha"][:7]
            msg = item["commit"]["message"].splitlines()[0]
            author = item["commit"]["author"]["name"]
            lines.append(f"  - {sha} ({author}): {msg}")
        return "\n".join(lines)
