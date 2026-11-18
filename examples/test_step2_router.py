"""
Step 2 Demo: ZERO-TOKEN Offline Test of Multi-Provider Smart Model Router!

Zero real API calls are made! Uses httpx.MockTransport to simulate:
  1. Smart-tier routing -> Picks Google Gemini 2.5 Flash (returns a tool call).
  2. Fast-tier routing  -> Picks Google Gemini 2.5 Flash-Lite (for cheap log summaries).
  3. HTTP 429 Rate Limit on Google -> Circuit Breaker activates -> Instant Failover to Groq Qwen!
  4. Circuit Breaker memory -> Next request SKIPS rate-limited Google and goes straight to Groq!
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.adapters.llm_adapter import LLMAdapter


def main() -> None:
    # Toggle to simulate Google hitting a 429 rate limit
    simulate_google_rate_limit = {"active": False}
    http_calls_made: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        model = body["model"]
        host = request.url.host
        http_calls_made.append(f"{host} -> {model}")

        # Simulate Google 429 Rate Limit when enabled
        if simulate_google_rate_limit["active"] and "googleapis.com" in host:
            return httpx.Response(
                status_code=429,
                json={"error": {"message": "Quota exceeded (simulated 429)"}},
            )

        # Simulate a tool call if tools were provided
        if body.get("tools"):
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
                                        "id": "call_001",
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
                    "usage": {"prompt_tokens": 180, "completion_tokens": 25},
                },
            )

        # Normal text response
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"[MOCK RESPONSE from {model}] Summary complete!",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 90, "completion_tokens": 18},
            },
        )

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))

    router = LLMAdapter.from_env(
        cooldown_seconds=60.0,
        http_client=mock_client,
        # Fake keys so this test works 100% offline without touching real quotas
        api_keys_override={"GEMINI_API_KEY": "mock-gemini", "GROQ_API_KEY": "mock-groq"},
    )

    print("=== TEST 1: Smart Tier (Tool Calling) ===")
    res1 = router.chat(
        messages=[{"role": "user", "content": "Check Supabase slow queries"}],
        tools=[{"type": "function", "function": {"name": "check_supabase_slow_queries"}}],
        tier="smart",
    )
    print(f"  Answered by: {res1.provider}/{res1.model}")
    print(f"  Tool call  : {res1.tool_calls[0].name}({res1.tool_calls[0].arguments})")
    print(f"  Tokens used: {res1.total_tokens} (0 real tokens spent!)\n")

    print("=== TEST 2: Fast Tier (Cheap Log Summary) ===")
    res2 = router.chat(
        messages=[{"role": "user", "content": "Summarize these 50 log lines"}],
        tier="fast",
    )
    print(f"  Answered by: {res2.provider}/{res2.model}")
    print(f"  Content    : {res2.content}\n")

    print("=== TEST 3: Google hits 429 Rate Limit -> Auto-Failover to Groq! ===")
    simulate_google_rate_limit["active"] = True
    res3 = router.chat(
        messages=[{"role": "user", "content": "Investigate Vercel 504 crash"}],
        tier="smart",
    )
    print(f"  Answered by: {res3.provider}/{res3.model}")
    print(f"  Content    : {res3.content}")
    print("  Router Events:")
    for log in router.request_log[-2:]:
        print(f"    {log}\n")

    print("=== TEST 4: Circuit Breaker Proof (Google is skipped immediately!) ===")
    calls_before = len(http_calls_made)
    res4 = router.chat(
        messages=[{"role": "user", "content": "Next turn in investigation"}],
        tier="smart",
    )
    new_calls = http_calls_made[calls_before:]
    print(f"  HTTP calls made on Turn 4: {new_calls}  <-- Notice Google was SKIPPED!")
    print(f"  Answered by: {res4.provider}/{res4.model}\n")

    print("=== Live Endpoint Health Table ===")
    for row in router.get_health_status():
        print(f"  {row['tier']:5s} | {row['endpoint']:30s} | {row['status']}")


if __name__ == "__main__":
    main()
