"""
Agent Service (Core Harness): Pure-Python Multi-Step Execution Loop (Zero LangChain!).
Explicitly composes ToolService + LLMAdapter + ContextService.
"""

from __future__ import annotations

from typing import Callable

from sre_agent.adapters.llm_adapter import LLMAdapter
from sre_agent.config.constants import (
    DEFAULT_MAX_AGENT_STEPS,
    DEFAULT_MIN_TOOL_SCORE,
    DEFAULT_SRE_SYSTEM_PROMPT,
    DEFAULT_TOP_K_TOOLS,
    DESTRUCTIVE_TOOLS,
)
from sre_agent.helpers.complexity import classify_task_tier
from sre_agent.helpers.tokens import count_text_tokens
from sre_agent.schemas.models import HarnessResult
from sre_agent.services.context_service import ContextService
from sre_agent.services.tool_service import ToolService

ApprovalCallback = Callable[[str, str], bool]


class AgentService:
    """Core Agent Harness loop with dynamic tool loading, budgeting, and safety guardrails."""

    def __init__(
        self,
        tools: ToolService,
        llm: LLMAdapter,
        context: ContextService | None = None,
        max_steps: int = DEFAULT_MAX_AGENT_STEPS,
        top_k_tools: int = DEFAULT_TOP_K_TOOLS,
        min_tool_score: float = DEFAULT_MIN_TOOL_SCORE,
        require_approval_tools: set[str] | None = None,
        verbose: bool = True,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.context = context or ContextService(system_prompt=DEFAULT_SRE_SYSTEM_PROMPT)
        self.max_steps = max_steps
        self.top_k_tools = top_k_tools
        self.min_tool_score = min_tool_score
        self.require_approval_tools = (
            require_approval_tools if require_approval_tools is not None else set(DESTRUCTIVE_TOOLS)
        )
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(
        self,
        user_query: str,
        approval_callback: ApprovalCallback | None = None,
    ) -> HarnessResult:
        """Run a complete multi-step SRE investigation for user_query."""
        self.context.add_user(user_query)

        # 1. Dynamic Tool Loading (only inject Top-K matching tools!)
        selected_schemas, _ = self.tools.get_relevant_tools(
            query=user_query,
            top_k=self.top_k_tools,
            min_score=self.min_tool_score,
        )
        injected_names = [s["function"]["name"] for s in selected_schemas]
        self._log(
            f"[HARNESS] Dynamically loaded {len(injected_names)}/{len(self.tools)} tools: "
            f"{injected_names}"
        )

        tools_executed: list[str] = []
        models_used: list[str] = []
        total_tokens = 0

        # 2. Pure-Python Execution Loop
        for step in range(1, self.max_steps + 1):
            if self.context.maybe_compact(llm=self.llm):
                self._log(
                    f"[CONTEXT] Auto-compacted history! "
                    f"Saved ~{self.context.tokens_saved_by_compaction} tokens."
                )

            # 2a. Complexity-Aware Task Routing (Simple -> Groq 'fast' | Complex -> Gemini 'smart')
            tier, tier_reason = classify_task_tier(
                query=user_query,
                step=step,
                tools_executed_so_far=len(tools_executed),
            )

            self._log(
                f"[STEP {step}/{self.max_steps}] Tier={tier.upper()} ({tier_reason}) | "
                f"Context: {self.context.current_tokens}/{self.context.max_context_tokens} tokens "
                f"({self.context.budget_usage_pct:.1f}%)"
            )

            response = self.llm.chat(
                messages=self.context.get_messages(),
                tools=selected_schemas if selected_schemas else None,
                tier=tier,
            )
            total_tokens += response.total_tokens
            model_id = f"{response.provider}/{response.model}"
            models_used.append(model_id)

            if response.tool_calls:
                raw_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in response.tool_calls
                ]
                self.context.add_assistant(content=response.content, tool_calls=raw_calls)

                for tc in response.tool_calls:
                    self._log(f"  -> LLM ({model_id}) requested tool: {tc.name}({tc.arguments})")

                    if tc.name in self.require_approval_tools:
                        approved = (
                            approval_callback(tc.name, tc.arguments)
                            if approval_callback is not None
                            else False
                        )
                        if not approved:
                            denied_msg = (
                                f"BLOCKED BY SAFETY GUARDRAIL: Human denied approval to run "
                                f"destructive tool '{tc.name}'."
                            )
                            self._log(f"  !! {denied_msg}")
                            self.context.add_tool_result(tc.id, tc.name, denied_msg)
                            continue

                    raw_output = self.tools.execute_tool(tc.name, tc.arguments)
                    tools_executed.append(tc.name)
                    trimmed = self.context.add_tool_result(tc.id, tc.name, raw_output)
                    self._log(f"  <- Tool output stored ({count_text_tokens(trimmed)} tokens)")

                continue

            final_text = response.content or "(Empty response)"
            self.context.add_assistant(content=final_text)
            self._log(f"[DONE] Investigation finished in {step} steps via {model_id}.")

            return HarnessResult(
                final_answer=final_text,
                steps_taken=step,
                tools_injected=injected_names,
                tools_executed=tools_executed,
                models_used=models_used,
                total_llm_tokens=total_tokens,
                compactions_triggered=self.context.compactions_triggered,
                tokens_saved_by_truncation=self.context.tokens_saved_by_truncation,
                tokens_saved_by_compaction=self.context.tokens_saved_by_compaction,
            )

        return HarnessResult(
            final_answer="Stopped: reached maximum safety step limit without final answer.",
            steps_taken=self.max_steps,
            tools_injected=injected_names,
            tools_executed=tools_executed,
            models_used=models_used,
            total_llm_tokens=total_tokens,
            compactions_triggered=self.context.compactions_triggered,
            tokens_saved_by_truncation=self.context.tokens_saved_by_truncation,
            tokens_saved_by_compaction=self.context.tokens_saved_by_compaction,
        )
