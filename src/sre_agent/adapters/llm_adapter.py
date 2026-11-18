"""
LLM Adapter: Multi-Provider OpenAI-Compatible Gateway with Circuit Breaker & Auto-Failover.
Uses config/constants.py + schemas/models.py. Zero inheritance, pure composition.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from sre_agent.config.constants import (
    DEFAULT_AUTH_COOLDOWN_SECONDS,
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_MODEL_CONFIGS,
)
from sre_agent.schemas.models import LLMResponse, ModelEndpoint, ToolCallRequest


class AllModelsFailedError(RuntimeError):
    """Raised only when every configured endpoint across all providers fails."""


class LLMAdapter:
    """
    Fault-tolerant Multi-Provider LLM Router (Google Gemini + Groq + any OpenAI-compatible API).
    Supports task-tier routing ('smart' vs 'fast') and automatic 429 cooldown failover.
    """

    def __init__(
        self,
        endpoints: list[ModelEndpoint],
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
        api_keys_override: dict[str, str] | None = None,
    ) -> None:
        load_dotenv()
        self.endpoints = endpoints
        self.cooldown_seconds = cooldown_seconds
        self.timeout_seconds = timeout_seconds
        self._client = http_client or httpx.Client(timeout=timeout_seconds)
        self._api_keys_override = api_keys_override or {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_log: list[str] = []

    @classmethod
    def from_env(
        cls,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        http_client: httpx.Client | None = None,
        api_keys_override: dict[str, str] | None = None,
    ) -> LLMAdapter:
        """Build default Google Gemini + Groq pool using constants.DEFAULT_MODEL_CONFIGS."""
        load_dotenv()
        overrides = api_keys_override or {}
        all_endpoints = [
            ModelEndpoint(provider, model, base_url, key_env, tier=tier)
            for provider, model, base_url, key_env, tier in DEFAULT_MODEL_CONFIGS
        ]
        active = [
            ep for ep in all_endpoints
            if overrides.get(ep.api_key_env) or os.getenv(ep.api_key_env)
        ]
        return cls(
            endpoints=active,
            cooldown_seconds=cooldown_seconds,
            http_client=http_client,
            api_keys_override=overrides,
        )

    def _get_api_key(self, env_var: str) -> str:
        return self._api_keys_override.get(env_var) or os.getenv(env_var, "")

    def _select_candidates(self, tier: str) -> list[ModelEndpoint]:
        now = time.time()
        primary = [ep for ep in self.endpoints if ep.tier == tier and ep.is_healthy(now)]
        secondary = [ep for ep in self.endpoints if ep.tier != tier and ep.is_healthy(now)]

        seen: set[str] = set()
        ordered: list[ModelEndpoint] = []
        for ep in primary + secondary:
            if ep.id not in seen:
                seen.add(ep.id)
                ordered.append(ep)
        return ordered

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tier: str = "smart",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send chat completion through the multi-provider pool with automatic failover."""
        candidates = self._select_candidates(tier)
        if not candidates:
            raise AllModelsFailedError(
                f"No healthy endpoints available for tier='{tier}'. Check API keys or cooldowns."
            )

        errors: list[str] = []

        for ep in candidates:
            api_key = self._get_api_key(ep.api_key_env)
            if not api_key:
                errors.append(f"{ep.id}: missing {ep.api_key_env}")
                continue

            payload: dict[str, Any] = {
                "model": ep.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            url = f"{ep.base_url.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            start = time.perf_counter()
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                latency_ms = round((time.perf_counter() - start) * 1000, 1)

                if resp.status_code == 429 or resp.status_code >= 500:
                    ep.mark_failed(self.cooldown_seconds)
                    self.request_log.append(
                        f"[FAILOVER] {ep.id} returned HTTP {resp.status_code}. "
                        f"Put on {self.cooldown_seconds}s cooldown. Switching to next model..."
                    )
                    errors.append(f"{ep.id}: HTTP {resp.status_code}")
                    continue

                if resp.status_code in (401, 403):
                    ep.mark_failed(DEFAULT_AUTH_COOLDOWN_SECONDS)
                    self.request_log.append(
                        f"[AUTH FAIL] {ep.id} returned HTTP {resp.status_code}. Skipping provider."
                    )
                    errors.append(f"{ep.id}: HTTP {resp.status_code}")
                    continue

                if resp.status_code != 200:
                    errors.append(f"{ep.id}: HTTP {resp.status_code} - {resp.text[:120]}")
                    continue

                data = resp.json()
                choice = data["choices"][0]["message"]
                usage = data.get("usage", {})

                parsed_tools = [
                    ToolCallRequest(
                        id=tc.get("id", "call_unknown"),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    )
                    for tc in (choice.get("tool_calls") or [])
                ]

                prompt_toks = int(usage.get("prompt_tokens", 0))
                comp_toks = int(usage.get("completion_tokens", 0))
                self.total_prompt_tokens += prompt_toks
                self.total_completion_tokens += comp_toks

                ep.mark_healthy()
                self.request_log.append(
                    f"[OK] {ep.id} (tier={tier}) answered in {latency_ms}ms "
                    f"[{prompt_toks}+{comp_toks} tokens]"
                )

                return LLMResponse(
                    content=choice.get("content"),
                    tool_calls=parsed_tools,
                    provider=ep.provider,
                    model=ep.model,
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    latency_ms=latency_ms,
                )

            except httpx.HTTPError as exc:
                ep.mark_failed(self.cooldown_seconds)
                self.request_log.append(f"[NETWORK FAIL] {ep.id}: {exc}. Switching to next model...")
                errors.append(f"{ep.id}: {exc}")

        raise AllModelsFailedError(
            "All candidate models failed:\n  - " + "\n  - ".join(errors)
        )

    def get_health_status(self) -> list[dict[str, Any]]:
        now = time.time()
        return [
            {
                "endpoint": ep.id,
                "tier": ep.tier,
                "status": "HEALTHY" if ep.is_healthy(now) else f"COOLDOWN ({int(ep.cooldown_until - now)}s left)",
                "failures": ep.failure_count,
            }
            for ep in self.endpoints
        ]
