"""
nanoletta/governor.py — Governor layer hooks for the agent loop.

This module provides optional governance hooks that the agent loop calls
at critical points during reasoning. Each hook is a no-op by default and
becomes active only when a GovernanceRuntime is injected.

Hook points in the agent loop (agent.py):
1. pre_step       — before any LLM calls
2. post_draft     — after LLM generates response (consistency / safety check)
3. post_tool      — after tool execution (correction / learning)
4. post_step      — after full step completes (tracking, learning)
5. build_context  — during context assembly (extra block injected into system prompt)
6. daemon_cycle   — called by external daemon (proactive / initiative logic)

The governor is stateless — all state lives in the runtime or in AgentState.
This is a thin dispatch layer.

Design:
- All hooks return GovernorResult so callers can inspect outcomes
- All hooks catch exceptions and return safe defaults
- No hook can crash the agent loop
- Hooks are individually toggleable via GovernorConfig
- Wire in custom logic by passing a GovernanceRuntime to Governor()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nanoletta.types import AgentState, Message, ToolCall

_log = logging.getLogger("nanoletta.governor")


@runtime_checkable
class GovernanceRuntime(Protocol):
    """Protocol for optional governance logic injected into the Governor.

    Implement this interface to wire in consistency checks, correction
    detection, proactive outreach, or any other governance behaviour
    without modifying the core agent loop.

    All methods are optional in the sense that the Governor falls back
    to no-ops when no runtime is provided. If a runtime is provided
    and raises, the exception is caught and logged — it cannot crash
    the agent loop.

    Example::

        class MyRuntime:
            def on_open_question(self, user_msg, ai_response, person="user"):
                return {}  # or track open questions in a DB

            def on_draft_response(self, draft, self_model, history, doctrine):
                # check for contradictions
                return {"ok": True}

            def on_correction(self, user_msg, ai_response, correction_type):
                return {}

            def get_context_block(self, max_items=3):
                return None  # or inject text into the system prompt

            def on_daemon_cycle(self, commitments, emotional_state,
                                homeostasis_state, last_user_ts, chat_id):
                return {}

        governor = Governor(runtime=MyRuntime())
    """

    def on_open_question(
        self,
        user_message: str,
        ai_response: str,
        person: str = "user",
    ) -> dict[str, Any]: ...

    def on_draft_response(
        self,
        draft_response: str,
        self_model_payload: Any,
        conversation_history: list[dict[str, Any]],
        doctrine_snippets: list[dict[str, Any]] | None,
    ) -> dict[str, Any]: ...

    def on_correction(
        self,
        user_message: str,
        ai_response: str,
        correction_type: str,
    ) -> dict[str, Any]: ...

    def get_context_block(self, max_items: int = 3) -> str | None: ...

    def on_daemon_cycle(
        self,
        commitments: list[dict[str, Any]] | None,
        emotional_state: dict[str, float] | None,
        homeostasis_state: dict[str, float] | None,
        last_user_message_ts: str | None,
        chat_id: str,
    ) -> dict[str, Any]: ...


@dataclass
class GovernorConfig:
    """Controls which governor hooks are active.

    All hooks enabled by default. Disable individually for
    performance or during testing.
    """

    enable_consistency_check: bool = True
    enable_correction_bridge: bool = True
    enable_open_questions: bool = True
    enable_consciousness_block: bool = True
    enable_initiative: bool = True
    person: str = "user"  # passed to open-question and correction hooks


@dataclass
class GovernorResult:
    """Result from a governor hook invocation."""

    hook: str
    active: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Governor:
    """Governor layer — optional governance hooks for the agent loop.

    Pass a GovernanceRuntime to activate hooks. Without one, all hooks
    are no-ops and the agent loop runs unmodified.

    Args:
        config: Toggle individual hooks and set the person label.
        runtime: Optional GovernanceRuntime implementation. If None,
                 all hooks return inactive GovernorResults.

    Example::

        governor = Governor(runtime=MyRuntime())
        agent = Agent(..., governor=governor)

    Override any method instead of using a runtime if you prefer
    subclassing over composition.
    """

    def __init__(
        self,
        config: GovernorConfig | None = None,
        runtime: GovernanceRuntime | None = None,
    ) -> None:
        self.config = config or GovernorConfig()
        self.runtime = runtime

    # ------------------------------------------------------------------
    # Hook 1: pre_step
    # ------------------------------------------------------------------

    async def pre_step(
        self,
        user_message: str,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Called before any LLM calls.

        Typical uses: open question detection, pre-step state validation.
        """
        if not self.config.enable_open_questions or self.runtime is None:
            return GovernorResult(hook="pre_step", active=False)

        try:
            result = self.runtime.on_open_question(
                user_message, "", person=self.config.person
            )
            return GovernorResult(hook="pre_step", result=result)
        except Exception as exc:
            _log.debug("pre_step hook failed: %s", exc)
            return GovernorResult(hook="pre_step", error=str(exc))

    # ------------------------------------------------------------------
    # Hook 2: post_draft
    # ------------------------------------------------------------------

    async def post_draft(
        self,
        draft_response: str,
        agent_state: AgentState,
        conversation_history: list[Message] | None = None,
        doctrine_snippets: list[dict[str, Any]] | None = None,
    ) -> GovernorResult:
        """Called after the LLM generates a draft response.

        Typical uses: consistency checking, contradiction detection,
        safety filtering before the response is committed to history.
        """
        if not self.config.enable_consistency_check or self.runtime is None:
            return GovernorResult(hook="post_draft", active=False)

        try:
            history_dicts = [
                {"role": m.role, "content": m.content}
                for m in (conversation_history or [])
            ]
            result = self.runtime.on_draft_response(
                draft_response=draft_response,
                self_model_payload=agent_state.metadata.get("self_model"),
                conversation_history=history_dicts,
                doctrine_snippets=doctrine_snippets,
            )
            return GovernorResult(hook="post_draft", result=result)
        except Exception as exc:
            _log.debug("post_draft hook failed: %s", exc)
            return GovernorResult(hook="post_draft", error=str(exc))

    # ------------------------------------------------------------------
    # Hook 3: post_tool
    # ------------------------------------------------------------------

    async def post_tool(
        self,
        user_message: str,
        ai_response: str,
        tool_call: ToolCall,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Called after a tool executes.

        Only fires for memory-editing tools by default. Typical uses:
        correction detection, learning from memory edits.
        """
        if not self.config.enable_correction_bridge or self.runtime is None:
            return GovernorResult(hook="post_tool", active=False)

        memory_tools = {"memory_str_replace", "memory_insert", "memory_rethink", "memory_delete"}
        if tool_call.name not in memory_tools:
            return GovernorResult(hook="post_tool", active=False)

        try:
            result = self.runtime.on_correction(user_message, ai_response, "INSTRUCTION")
            return GovernorResult(hook="post_tool", result=result)
        except Exception as exc:
            _log.debug("post_tool hook failed: %s", exc)
            return GovernorResult(hook="post_tool", error=str(exc))

    # ------------------------------------------------------------------
    # Hook 4: post_step
    # ------------------------------------------------------------------

    async def post_step(
        self,
        user_message: str,
        ai_response: str,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Called after the full reasoning step completes.

        Typical uses: open question tracking, post-step learning.
        """
        if not self.config.enable_open_questions or self.runtime is None:
            return GovernorResult(hook="post_step", active=False)

        try:
            result = self.runtime.on_open_question(
                user_message, ai_response, person=self.config.person
            )
            return GovernorResult(hook="post_step", result=result)
        except Exception as exc:
            _log.debug("post_step hook failed: %s", exc)
            return GovernorResult(hook="post_step", error=str(exc))

    # ------------------------------------------------------------------
    # Hook 5: build_context_block
    # ------------------------------------------------------------------

    def build_consciousness_block(self) -> str | None:
        """Called during context assembly.

        Returns an optional text block injected into the system prompt.
        Typical use: surface active cognitive state, reminders, or
        running context that should influence the next LLM call.
        """
        if not self.config.enable_consciousness_block or self.runtime is None:
            return None

        try:
            return self.runtime.get_context_block(max_items=3)
        except Exception as exc:
            _log.debug("context_block hook failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Hook 6: daemon_cycle
    # ------------------------------------------------------------------

    async def daemon_cycle(
        self,
        commitments: list[dict[str, Any]] | None = None,
        emotional_state: dict[str, float] | None = None,
        homeostasis_state: dict[str, float] | None = None,
        last_user_message_ts: str | None = None,
        chat_id: str = "",
    ) -> GovernorResult:
        """Called by an external daemon once per heartbeat cycle.

        Typical use: proactive outreach decisions, initiative logic.
        """
        if not self.config.enable_initiative or self.runtime is None:
            return GovernorResult(hook="daemon_cycle", active=False)

        try:
            result = self.runtime.on_daemon_cycle(
                commitments=commitments,
                emotional_state=emotional_state,
                homeostasis_state=homeostasis_state,
                last_user_message_ts=last_user_message_ts,
                chat_id=chat_id,
            )
            return GovernorResult(hook="daemon_cycle", result=result)
        except Exception as exc:
            _log.debug("daemon_cycle hook failed: %s", exc)
            return GovernorResult(hook="daemon_cycle", error=str(exc))
