"""
Context Service: Manages agent working memory, token budgets, log trimming & auto-compaction.
Delegates token math cleanly to helpers/tokens.py.
"""

from __future__ import annotations

from typing import Any

from sre_agent.adapters.llm_adapter import LLMAdapter
from sre_agent.config.constants import (
    COMPACTION_SYSTEM_PROMPT,
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_KEEP_RECENT_MESSAGES,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_TOOL_OUTPUT_TOKENS,
)
from sre_agent.helpers.tokens import count_messages_tokens, truncate_head_tail


class ContextService:
    """Working memory manager with strict token budgeting and automatic summarization."""

    def __init__(
        self,
        system_prompt: str,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_tool_output_tokens: int = DEFAULT_MAX_TOOL_OUTPUT_TOKENS,
        compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD,
        keep_recent_messages: int = DEFAULT_KEEP_RECENT_MESSAGES,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_tool_output_tokens = max_tool_output_tokens
        self.compaction_threshold = compaction_threshold
        self.keep_recent_messages = keep_recent_messages

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.compactions_triggered = 0
        self.tokens_saved_by_truncation = 0
        self.tokens_saved_by_compaction = 0

    @property
    def current_tokens(self) -> int:
        return count_messages_tokens(self.messages)

    @property
    def budget_usage_pct(self) -> float:
        return (self.current_tokens / self.max_context_tokens) * 100.0

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, tool_name: str, raw_output: str) -> str:
        safe_output, omitted = truncate_head_tail(raw_output, self.max_tool_output_tokens)
        self.tokens_saved_by_truncation += omitted
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": safe_output,
            }
        )
        return safe_output

    def maybe_compact(self, llm: LLMAdapter | None = None) -> bool:
        """Compact older conversation turns using the FAST model tier when budget >= 75%."""
        if self.current_tokens < int(self.max_context_tokens * self.compaction_threshold):
            return False
        if len(self.messages) <= self.keep_recent_messages + 2:
            return False

        system_msg = self.messages[0]
        older_msgs = self.messages[1 : -self.keep_recent_messages]
        recent_msgs = self.messages[-self.keep_recent_messages :]

        # Never split between an assistant tool_call and its corresponding tool response
        while recent_msgs and recent_msgs[0].get("role") == "tool":
            older_msgs.append(recent_msgs.pop(0))

        if not older_msgs:
            return False

        tokens_before = self.current_tokens
        transcript = "\n".join(
            f"[{m['role'].upper()}]: {str(m.get('content') or m.get('tool_calls', ''))[:400]}"
            for m in older_msgs
        )

        if llm is not None:
            summary_resp = llm.chat(
                messages=[
                    {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                tier="fast",
                max_tokens=250,
            )
            summary_text = summary_resp.content or transcript[:500]
        else:
            summary_text = transcript[:500] + "..."

        compacted_msg = {
            "role": "system",
            "content": f"[COMPACTED EARLIER INVESTIGATION CONTEXT]\n{summary_text}",
        }
        self.messages = [system_msg, compacted_msg] + recent_msgs
        self.compactions_triggered += 1
        self.tokens_saved_by_compaction += max(0, tokens_before - self.current_tokens)
        return True

    def get_messages(self) -> list[dict[str, Any]]:
        return self.messages
