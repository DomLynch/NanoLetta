"""
nanoletta/interfaces.py — Protocol interfaces for the cognitive kernel.

Five thin contracts that decouple the agent loop from concrete implementations.
Each interface has exactly one job. Implementations can be swapped without
touching the loop logic.

Interfaces:
1. LLMClient     — send messages to an LLM, get a response
2. BlockStore     — persist and retrieve memory blocks
3. SessionStore   — persist and retrieve conversation messages
4. ToolExecutor   — execute a tool call and return the result
5. AgentRepo      — load and save full agent state

Design decisions:
- Protocol over ABC: structural subtyping, no inheritance required
- All methods async: IO-bound operations should not block
- Minimal signatures: only the parameters the loop actually uses
- No framework types: plain dicts where possible, nanoletta types where needed
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nanoletta.types import AgentState, Block, LLMConfig, Message, Tool, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# 1. LLMClient — talk to a language model
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Send messages to an LLM and get a completion response.

    The response must include at minimum:
    - content: str (the text response)
    - tool_calls: list[ToolCall] (parsed tool invocations, may be empty)

    Implementations: OpenAI-compatible API, Anthropic, Ollama, mock for tests.
    """

    async def request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: LLMConfig,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format message dicts [{role, content, ...}]
            tools: OpenAI-format tool schemas [{type, function: {name, ...}}]
            config: Model configuration (model name, temperature, etc.)

        Returns:
            Raw completion response dict with at minimum:
            {
                "content": str,
                "tool_calls": [{"id": str, "function": {"name": str, "arguments": str}}],
                "usage": {"prompt_tokens": int, "completion_tokens": int},
            }
        """
        ...


# ---------------------------------------------------------------------------
# 2. BlockStore — memory block persistence
# ---------------------------------------------------------------------------

@runtime_checkable
class BlockStore(Protocol):
    """Persist and retrieve memory blocks for an agent.

    Blocks are the agent's self-editing working memory. This store
    handles durability — the agent loop reads and writes blocks
    through this interface.
    """

    async def get_blocks(self, agent_id: str) -> dict[str, Block]:
        """Load all blocks for an agent.

        Returns:
            Dict mapping label → Block.
        """
        ...

    async def save_block(self, agent_id: str, block: Block) -> None:
        """Save or update a single block.

        Upserts by (agent_id, label). Creates if not exists.
        """
        ...

    async def save_blocks(self, agent_id: str, blocks: dict[str, Block]) -> None:
        """Save all blocks atomically.

        Replaces the entire block set for this agent.
        """
        ...


# ---------------------------------------------------------------------------
# 3. SessionStore — conversation message persistence
# ---------------------------------------------------------------------------

@runtime_checkable
class SessionStore(Protocol):
    """Persist and retrieve conversation messages.

    Messages are append-only within a session. The agent loop
    appends new messages after each step and reads recent history
    for context building.
    """

    async def get_messages(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> list[Message]:
        """Load recent messages for an agent.

        Returns messages in chronological order (oldest first).
        Limit controls how many recent messages to return.
        """
        ...

    async def save_messages(
        self,
        agent_id: str,
        messages: list[Message],
    ) -> None:
        """Append messages to the conversation history.

        Messages are saved in order. IDs should be pre-assigned.
        """
        ...

    async def count_messages(self, agent_id: str) -> int:
        """Return total message count for an agent."""
        ...


# ---------------------------------------------------------------------------
# 4. ToolExecutor — execute tool calls
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolExecutor(Protocol):
    """Execute a tool call and return the result.

    The executor handles both built-in tools (memory editing,
    message sending, search) and custom tools.

    Built-in tools that modify agent state (memory_str_replace,
    memory_insert, etc.) receive the AgentState and modify it
    in-place. The caller is responsible for persisting changes.
    """

    async def execute(
        self,
        tool_call: ToolCall,
        agent_state: AgentState,
    ) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: Parsed tool invocation (name + arguments).
            agent_state: Current agent state. Memory-editing tools
                         modify this in-place.

        Returns:
            ToolResult with output string and success/error status.
        """
        ...


# ---------------------------------------------------------------------------
# 5. AgentRepo — full agent state persistence
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentRepo(Protocol):
    """Load and save complete agent state.

    This is the top-level persistence interface. Implementations
    may delegate to BlockStore + SessionStore internally, or
    handle everything in one store.

    The agent loop calls load() at the start and save() after
    each step (or batch of steps).
    """

    async def load(self, agent_id: str) -> AgentState:
        """Load full agent state by ID.

        Raises KeyError if agent does not exist.
        """
        ...

    async def save(self, state: AgentState) -> None:
        """Save full agent state.

        Upserts by state.id. Creates if new, updates if existing.
        """
        ...

    async def exists(self, agent_id: str) -> bool:
        """Check if an agent exists."""
        ...

    async def create(
        self,
        name: str,
        system_prompt: str,
        blocks: dict[str, Block] | None = None,
        tools: list[Tool] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> AgentState:
        """Create a new agent with initial state.

        Returns the created AgentState with assigned ID.
        """
        ...
