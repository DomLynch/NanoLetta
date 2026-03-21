"""
nanoletta/governor.py — Governor layer hooks for the agent loop.

This module provides the consciousness/governance hooks that the agent
loop calls at critical points during reasoning. Each hook is optional
and degrades gracefully if the backing module is unavailable.

Hook points in the agent loop (agent.py):
1. pre_step       — before any LLM calls (open question detection)
2. post_draft     — after LLM generates response (consistency check)
3. post_tool      — after tool execution (correction bridge)
4. post_step      — after full step completes (learning, tracking)
5. build_context  — during context assembly (consciousness block)
6. daemon_cycle   — called by external daemon (initiative engine)

The governor is stateless — all state lives in the backing modules
or in AgentState. This is a thin dispatch layer.

Design:
- All hooks return dicts so callers can inspect results
- All hooks catch exceptions and return safe defaults
- No hook can crash the agent loop
- Hooks are individually toggleable via GovernorConfig
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from nanoletta.types import AgentState, Message, ToolCall

_log = logging.getLogger("nanoletta.governor")


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


@dataclass
class GovernorResult:
    """Result from a governor hook invocation."""

    hook: str
    active: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Governor:
    """Governor layer — consciousness and governance hooks for the agent loop.

    Wraps Brain's consciousness modules (0-5) behind a clean interface.
    Each method corresponds to one hook point in the agent loop.

    When Brain modules are available, delegates to them.
    When they're not (e.g., running standalone NanoLetta), all hooks
    are no-ops that return safe defaults.
    """

    def __init__(self, config: GovernorConfig | None = None) -> None:
        self.config = config or GovernorConfig()

    async def pre_step(
        self,
        user_message: str,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Hook 1: Called before any LLM calls.

        Used for:
        - Open question detection from user message
        - Pre-step state validation
        """
        if not self.config.enable_open_questions:
            return GovernorResult(hook="pre_step", active=False)

        try:
            from skills.consciousness_runtime import on_open_question
            result = on_open_question(user_message, "", person="Dominic")
            return GovernorResult(hook="pre_step", result=result)
        except ImportError:
            return GovernorResult(hook="pre_step", active=False, error="consciousness_runtime not available")
        except Exception as exc:
            _log.debug("pre_step hook failed: %s", exc)
            return GovernorResult(hook="pre_step", error=str(exc))

    async def post_draft(
        self,
        draft_response: str,
        agent_state: AgentState,
        conversation_history: list[Message] | None = None,
        doctrine_snippets: list[dict[str, Any]] | None = None,
    ) -> GovernorResult:
        """Hook 2: Called after LLM generates a draft response.

        Used for:
        - Self-consistency checking (Module 4)
        - Contradiction detection against doctrine/self-model/conversation
        """
        if not self.config.enable_consistency_check:
            return GovernorResult(hook="post_draft", active=False)

        try:
            from skills.consciousness_runtime import on_draft_response
            history_dicts = [{"role": m.role, "content": m.content} for m in (conversation_history or [])]
            result = on_draft_response(
                draft_response=draft_response,
                self_model_payload=agent_state.metadata.get("self_model"),
                conversation_history=history_dicts,
                doctrine_snippets=doctrine_snippets,
            )
            return GovernorResult(hook="post_draft", result=result)
        except ImportError:
            return GovernorResult(hook="post_draft", active=False, error="consciousness_runtime not available")
        except Exception as exc:
            _log.debug("post_draft hook failed: %s", exc)
            return GovernorResult(hook="post_draft", error=str(exc))

    async def post_tool(
        self,
        user_message: str,
        ai_response: str,
        tool_call: ToolCall,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Hook 3: Called after tool execution.

        Used for:
        - Correction bridge (Module 1) when memory was edited
        - Learning from corrections/instructions
        """
        if not self.config.enable_correction_bridge:
            return GovernorResult(hook="post_tool", active=False)

        # Only fire for memory-editing tools
        memory_tools = {"memory_str_replace", "memory_insert", "memory_rethink", "memory_delete"}
        if tool_call.name not in memory_tools:
            return GovernorResult(hook="post_tool", active=False)

        try:
            from skills.consciousness_runtime import on_correction
            result = on_correction(user_message, ai_response, "INSTRUCTION")
            return GovernorResult(hook="post_tool", result=result)
        except ImportError:
            return GovernorResult(hook="post_tool", active=False, error="consciousness_runtime not available")
        except Exception as exc:
            _log.debug("post_tool hook failed: %s", exc)
            return GovernorResult(hook="post_tool", error=str(exc))

    async def post_step(
        self,
        user_message: str,
        ai_response: str,
        agent_state: AgentState,
    ) -> GovernorResult:
        """Hook 4: Called after the full step completes.

        Used for:
        - Open question tracking (Module 2) from the exchange
        - Post-step learning
        """
        if not self.config.enable_open_questions:
            return GovernorResult(hook="post_step", active=False)

        try:
            from skills.consciousness_runtime import on_open_question
            result = on_open_question(user_message, ai_response, person="Dominic")
            return GovernorResult(hook="post_step", result=result)
        except ImportError:
            return GovernorResult(hook="post_step", active=False, error="consciousness_runtime not available")
        except Exception as exc:
            _log.debug("post_step hook failed: %s", exc)
            return GovernorResult(hook="post_step", error=str(exc))

    def build_consciousness_block(self) -> str | None:
        """Hook 5: Called during context assembly.

        Returns a formatted block of active cognitive residue
        to inject into the system prompt.
        """
        if not self.config.enable_consciousness_block:
            return None

        try:
            from skills.consciousness_runtime import get_consciousness_block
            return get_consciousness_block(max_items=3)
        except ImportError:
            return None
        except Exception as exc:
            _log.debug("consciousness_block hook failed: %s", exc)
            return None

    async def daemon_cycle(
        self,
        commitments: list[dict[str, Any]] | None = None,
        emotional_state: dict[str, float] | None = None,
        homeostasis_state: dict[str, float] | None = None,
        last_user_message_ts: str | None = None,
        chat_id: str = "",
    ) -> GovernorResult:
        """Hook 6: Called by external daemon once per cycle.

        Used for:
        - Initiative engine (Module 3) proactive outreach decisions
        """
        if not self.config.enable_initiative:
            return GovernorResult(hook="daemon_cycle", active=False)

        try:
            from skills.consciousness_runtime import on_daemon_cycle
            result = on_daemon_cycle(
                commitments=commitments,
                emotional_state=emotional_state,
                homeostasis_state=homeostasis_state,
                last_user_message_ts=last_user_message_ts,
                chat_id=chat_id,
            )
            return GovernorResult(hook="daemon_cycle", result=result)
        except ImportError:
            return GovernorResult(hook="daemon_cycle", active=False, error="consciousness_runtime not available")
        except Exception as exc:
            _log.debug("daemon_cycle hook failed: %s", exc)
            return GovernorResult(hook="daemon_cycle", error=str(exc))
