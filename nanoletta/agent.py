"""
nanoletta/agent.py — The cognitive agent loop.

This is the core reasoning cycle, transplanted from Letta's letta_agent.py
(1,969 LOC) and stripped to the essential logic (~250 LOC).

What was removed:
- Telemetry/metrics/span tracking (~200 LOC)
- Step manager logging (~100 LOC)
- Approval/denial workflow (~80 LOC)
- Dry run mode (~20 LOC)
- Job cancellation checks (~15 LOC)
- Multi-agent routing (~30 LOC)
- Run ID tracking (~20 LOC)
- Error recovery with step progression enum (~100 LOC)

What remains:
- The reasoning loop: message → LLM → tool calls → memory edits → persist
- Continuation logic: should the agent keep stepping?
- Context building: system prompt + memory blocks + conversation history
- Tool execution dispatch with result handling

Governor hooks are clearly marked for Module 5 integration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from nanoletta.interfaces import BlockStore, SessionStore, ToolExecutor, AgentRepo
from nanoletta.interfaces import LLMClient  # noqa: used as type annotation
from nanoletta.types import AgentState, Message, ToolCall, ToolResult

_log = logging.getLogger("nanoletta.agent")

# Default max reasoning steps before forced stop
DEFAULT_MAX_STEPS = 10

# Built-in tool that signals the agent wants to stop and send a message
SEND_MESSAGE_TOOL = "send_message"


@dataclass
class AgentResponse:
    """Final response after the full reasoning loop completes."""

    content: str = ""
    steps: int = 0
    messages: list[Message] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


class Agent:
    """Minimal cognitive agent with self-editing memory.

    The reasoning cycle:
    1. Build context from system prompt + memory blocks + conversation history
    2. Send to LLM with available tools
    3. Parse tool calls from response
    4. Execute tools (including memory self-editing)
    5. Persist messages and updated memory
    6. If tool signals continuation, go to step 1

    Governor hooks (marked with # GOVERNOR HOOK) allow consciousness
    modules to inject checks at critical points without modifying
    the core loop logic.
    """

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        agent_repo: AgentRepo,
        session_store: SessionStore,
        block_store: BlockStore,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm_client
        self.tools = tool_executor
        self.repo = agent_repo
        self.sessions = session_store
        self.blocks = block_store

    async def step(
        self,
        user_message: str,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> AgentResponse:
        """Run the full reasoning loop for a user message.

        Args:
            user_message: The user's input text.
            max_steps: Maximum LLM calls before forced stop.

        Returns:
            AgentResponse with the final reply and metadata.
        """
        # Load current state
        state = await self.repo.load(self.agent_id)

        # Load conversation history
        history = await self.sessions.get_messages(self.agent_id)

        # Create user message
        user_msg = Message(role="user", content=user_message, agent_id=self.agent_id)
        history.append(user_msg)

        # GOVERNOR HOOK 1: Pre-step (before any LLM calls)
        # e.g., on_open_question(user_message, ...)

        response = AgentResponse()
        response.messages.append(user_msg)
        all_new_messages: list[Message] = [user_msg]

        for step_num in range(max_steps):
            _log.debug("Step %d/%d for agent %s", step_num + 1, max_steps, self.agent_id)

            # Build context for LLM
            context_messages = self._build_context(state, history)
            tool_schemas = [t.to_openai_schema() for t in state.tools]

            # Call LLM
            llm_response = await self.llm.request(
                messages=context_messages,
                tools=tool_schemas,
                config=state.llm_config,
            )

            # Track usage
            usage = llm_response.usage
            response.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            response.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            response.usage["total_tokens"] += usage.get("total_tokens", 0)
            response.steps += 1

            # Parse tool calls
            tool_calls = llm_response.tool_calls

            if not tool_calls:
                # No tool calls — LLM wants to respond with text only
                content = llm_response.content

                # GOVERNOR HOOK 2: Post-draft consistency check
                # e.g., on_draft_response(content, state, history, doctrine)

                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    agent_id=self.agent_id,
                )
                history.append(assistant_msg)
                all_new_messages.append(assistant_msg)
                response.content = content
                response.stop_reason = "end_turn"
                break

            # Process first tool call (single tool call per step for simplicity)
            tc = tool_calls[0]

            # Create assistant message with tool call
            assistant_msg = Message(
                role="assistant",
                content=llm_response.content or "",
                agent_id=self.agent_id,
                tool_calls=[{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    },
                }],
            )
            history.append(assistant_msg)
            all_new_messages.append(assistant_msg)

            # Execute tool
            _log.debug("Executing tool: %s", tc.name)
            result = await self.tools.execute(tc, state)

            # GOVERNOR HOOK 3: Post-tool execution
            # e.g., on_correction(user_message, ...) if memory was edited

            # Create tool response message
            tool_msg = Message(
                role="tool",
                content=result.output if result.success else f"Error: {result.error}",
                agent_id=self.agent_id,
                tool_call_id=tc.id,
                name=tc.name,
            )
            history.append(tool_msg)
            all_new_messages.append(tool_msg)

            # Decide continuation
            should_continue = self._should_continue(tc, result)

            if tc.name == SEND_MESSAGE_TOOL and result.success:
                # Agent used send_message — extract the message as the response
                try:
                    msg_content = tc.arguments.get("message", "") if isinstance(tc.arguments, dict) else ""
                    response.content = msg_content
                except (AttributeError, TypeError):
                    response.content = result.output
                response.stop_reason = "end_turn"

                if not should_continue:
                    break
            elif not should_continue:
                response.stop_reason = "end_turn"
                break
        else:
            # Hit max_steps
            response.stop_reason = "max_steps"
            _log.warning("Agent %s hit max_steps (%d)", self.agent_id, max_steps)

        # Persist all new messages
        await self.sessions.save_messages(self.agent_id, all_new_messages)

        # Persist updated memory blocks (tools may have modified them in-place)
        await self.blocks.save_blocks(self.agent_id, state.blocks)

        # Save full agent state
        await self.repo.save(state)

        # GOVERNOR HOOK 4: Post-step
        # e.g., on_open_question(user_message, response.content)

        _log.info(
            "Agent %s completed: steps=%d stop=%s content_len=%d",
            self.agent_id, response.steps, response.stop_reason,
            len(response.content),
        )

        response.messages = all_new_messages
        return response

    def _build_context(
        self,
        state: AgentState,
        history: list[Message],
    ) -> list[dict[str, Any]]:
        """Build the message context for the LLM request.

        Structure:
        1. System message (system prompt + compiled memory blocks)
        2. Conversation history (last N messages, within context window)

        GOVERNOR HOOK 5: Consciousness block injection
        e.g., get_consciousness_block() appended to system message
        """
        # Compile memory blocks into a single string
        memory_str = state.compile_memory()

        # Build system message
        system_content = state.system_prompt
        if memory_str:
            system_content += f"\n\n{memory_str}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
        ]

        # Add conversation history (converted to OpenAI format)
        # Rough context window guard: keep last N messages to stay under limit
        # Uses ~4 chars per token as a rough estimate
        max_history_chars = state.llm_config.context_window * 3  # ~75% of window for history
        total_chars = len(system_content)
        history_to_include: list[dict[str, Any]] = []

        for msg in reversed(history):
            msg_dict = msg.to_openai_dict()
            msg_chars = len(msg_dict.get("content", ""))
            if total_chars + msg_chars > max_history_chars:
                break
            history_to_include.append(msg_dict)
            total_chars += msg_chars

        history_to_include.reverse()
        messages.extend(history_to_include)

        return messages

    def _should_continue(self, tool_call: ToolCall, result: ToolResult) -> bool:
        """Decide if the agent should take another step after this tool call.

        Continuation rules:
        - send_message: stop (agent has spoken)
        - Memory editing tools: continue (agent is thinking/preparing)
        - Failed tools: stop (prevent infinite retry loops)
        - Everything else: continue (agent is working)
        """
        if not result.success:
            _log.warning("Tool %s failed: %s — stopping", tool_call.name, result.error)
            return False

        # Agent sent a message — stop after this
        if tool_call.name == SEND_MESSAGE_TOOL:
            return False

        # Memory tools and other internal tools — keep going
        return True
