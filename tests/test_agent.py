"""
Tests for nanoletta/agent.py — The cognitive agent loop.

Covers:
1. Basic message → response flow
2. Tool execution within the loop
3. Memory self-editing via tools
4. send_message stops the loop
5. Max steps enforcement
6. Failed tool stops the loop
7. No tool calls = text response
8. Context building (system prompt + memory + history)
9. Messages are persisted after step
10. Usage tracking across steps
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from nanoletta.types import AgentState, Block, LLMConfig, Message, Tool, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Mock implementations of interfaces
# ---------------------------------------------------------------------------

@dataclass
class MockLLMResponse:
    """Mock LLMResponse conforming to the Protocol."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


class MockLLMClient:
    """Mock LLM that returns pre-configured responses."""

    def __init__(self, responses: list[MockLLMResponse] | None = None):
        self.responses = list(responses or [])
        self._call_count = 0
        self.last_messages: list[dict[str, Any]] = []
        self.last_tools: list[dict[str, Any]] = []

    async def request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: LLMConfig,
    ) -> MockLLMResponse:
        self.last_messages = messages
        self.last_tools = tools
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
        else:
            # Default: send_message with "Done"
            resp = MockLLMResponse(
                tool_calls=[ToolCall(id="call_default", name="send_message", arguments={"message": "Done"})]
            )
        self._call_count += 1
        return resp


class MockToolExecutor:
    """Mock tool executor that tracks calls."""

    def __init__(self, results: dict[str, ToolResult] | None = None):
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, tool_call: ToolCall, agent_state: AgentState) -> ToolResult:
        self.calls.append((tool_call.name, tool_call.arguments))

        if tool_call.name in self.results:
            return self.results[tool_call.name]

        # Default: success with tool name echo
        return ToolResult(output=f"Executed {tool_call.name}", success=True)


class MockAgentRepo:
    """Mock agent repository storing state in memory."""

    def __init__(self, state: AgentState | None = None):
        self._state = state or _default_state()

    async def load(self, agent_id: str) -> AgentState:
        return self._state

    async def save(self, state: AgentState) -> None:
        self._state = state

    async def exists(self, agent_id: str) -> bool:
        return True

    async def create(self, **kwargs: Any) -> AgentState:
        return self._state


class MockSessionStore:
    """Mock session store that tracks messages."""

    def __init__(self):
        self.messages: dict[str, list[Message]] = {}

    async def get_messages(self, agent_id: str, limit: int = 50) -> list[Message]:
        return list(self.messages.get(agent_id, []))

    async def save_messages(self, agent_id: str, messages: list[Message]) -> None:
        if agent_id not in self.messages:
            self.messages[agent_id] = []
        self.messages[agent_id].extend(messages)

    async def count_messages(self, agent_id: str) -> int:
        return len(self.messages.get(agent_id, []))


class MockBlockStore:
    """Mock block store."""

    def __init__(self):
        self.saved_blocks: dict[str, dict[str, Block]] = {}

    async def get_blocks(self, agent_id: str) -> dict[str, Block]:
        return self.saved_blocks.get(agent_id, {})

    async def save_block(self, agent_id: str, block: Block) -> None:
        if agent_id not in self.saved_blocks:
            self.saved_blocks[agent_id] = {}
        self.saved_blocks[agent_id][block.label] = block

    async def save_blocks(self, agent_id: str, blocks: dict[str, Block]) -> None:
        self.saved_blocks[agent_id] = dict(blocks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_state() -> AgentState:
    return AgentState(
        id="agent_001",
        name="test_agent",
        system_prompt="You are a helpful assistant.",
        blocks={
            "persona": Block(label="persona", value="I am a test agent"),
        },
        tools=[
            Tool(
                name="send_message",
                description="Send a message to the user",
                parameters={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            ),
        ],
        llm_config=LLMConfig(model="test-model"),
    )


def _make_agent(
    llm_responses: list[MockLLMResponse] | None = None,
    tool_results: dict[str, ToolResult] | None = None,
    state: AgentState | None = None,
):
    """Create an Agent with mock dependencies."""
    from nanoletta.agent import Agent

    llm = MockLLMClient(llm_responses or [])
    tools = MockToolExecutor(tool_results)
    repo = MockAgentRepo(state)
    sessions = MockSessionStore()
    blocks = MockBlockStore()

    agent = Agent(
        agent_id="agent_001",
        llm_client=llm,
        tool_executor=tools,
        agent_repo=repo,
        session_store=sessions,
        block_store=blocks,
    )
    return agent, llm, tools, repo, sessions, blocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBasicFlow:
    async def test_simple_send_message(self):
        """Agent receives message, calls send_message, returns content."""
        agent, llm, tools, *_ = _make_agent(
            llm_responses=[
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_1", name="send_message", arguments={"message": "Hello back!"})]
                ),
            ]
        )

        result = await agent.step("Hello")

        assert result.content == "Hello back!"
        assert result.steps == 1
        assert result.stop_reason == "end_turn"
        assert len(tools.calls) == 1
        assert tools.calls[0][0] == "send_message"

    async def test_no_tool_calls_text_response(self):
        """When LLM returns text without tool calls, use it as response."""
        agent, *_ = _make_agent(
            llm_responses=[
                MockLLMResponse(content="Just a text reply", tool_calls=[]),
            ]
        )

        result = await agent.step("Hi")

        assert result.content == "Just a text reply"
        assert result.steps == 1
        assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
class TestToolExecution:
    async def test_tool_then_send_message(self):
        """Agent calls a tool, then sends a message (2 steps)."""
        agent, llm, tools, *_ = _make_agent(
            llm_responses=[
                # Step 1: call a memory tool
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_1", name="memory_str_replace", arguments={"label": "persona", "old": "test", "new": "real"})]
                ),
                # Step 2: send message
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_2", name="send_message", arguments={"message": "Updated memory!"})]
                ),
            ]
        )

        result = await agent.step("Update yourself")

        assert result.content == "Updated memory!"
        assert result.steps == 2
        assert len(tools.calls) == 2
        assert tools.calls[0][0] == "memory_str_replace"
        assert tools.calls[1][0] == "send_message"

    async def test_failed_tool_stops_loop(self):
        """If a tool fails, the agent stops (no infinite retry)."""
        agent, llm, tools, *_ = _make_agent(
            llm_responses=[
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_1", name="broken_tool", arguments={})]
                ),
            ],
            tool_results={
                "broken_tool": ToolResult(output="", success=False, error="Something broke"),
            },
        )

        result = await agent.step("Do something")

        assert result.steps == 1
        assert result.stop_reason == "end_turn"
        # Agent should NOT retry


@pytest.mark.asyncio
class TestMaxSteps:
    async def test_max_steps_enforced(self):
        """Agent stops after max_steps even if tools want to continue."""
        # Every response is a memory tool (wants to continue)
        responses = [
            MockLLMResponse(
                tool_calls=[ToolCall(id=f"call_{i}", name="memory_insert", arguments={"label": "log", "value": f"step {i}"})]
            )
            for i in range(20)
        ]

        agent, *_ = _make_agent(llm_responses=responses)

        result = await agent.step("Keep going", max_steps=3)

        assert result.steps == 3
        assert result.stop_reason == "max_steps"


@pytest.mark.asyncio
class TestContextBuilding:
    async def test_system_prompt_includes_memory(self):
        """Context includes system prompt + compiled memory blocks."""
        agent, llm, *_ = _make_agent(
            state=AgentState(
                id="agent_001",
                name="test",
                system_prompt="You are helpful.",
                blocks={
                    "persona": Block(label="persona", value="I am NanoLetta"),
                    "human": Block(label="human", value="The user is Dominic"),
                },
                tools=[Tool(name="send_message", description="Send message")],
                llm_config=LLMConfig(model="test"),
            ),
        )

        await agent.step("Hi")

        # Check the system message sent to LLM
        system_msg = llm.last_messages[0]
        assert system_msg["role"] == "system"
        assert "You are helpful." in system_msg["content"]
        assert "[persona]" in system_msg["content"]
        assert "I am NanoLetta" in system_msg["content"]
        assert "[human]" in system_msg["content"]
        assert "The user is Dominic" in system_msg["content"]

    async def test_user_message_in_context(self):
        """User's message appears in the context sent to LLM."""
        agent, llm, *_ = _make_agent()

        await agent.step("What is 2+2?")

        # Find user message in context
        user_msgs = [m for m in llm.last_messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "What is 2+2?"


@pytest.mark.asyncio
class TestPersistence:
    async def test_messages_persisted_after_step(self):
        """All new messages are saved to session store after step."""
        agent, _, _, _, sessions, _ = _make_agent(
            llm_responses=[
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_1", name="send_message", arguments={"message": "Reply"})]
                ),
            ]
        )

        await agent.step("Hello")

        saved = sessions.messages.get("agent_001", [])
        # Should have: user msg, assistant msg (with tool call), tool response
        assert len(saved) >= 2
        roles = [m.role for m in saved]
        assert "user" in roles
        assert "assistant" in roles

    async def test_blocks_persisted_after_step(self):
        """Updated blocks are saved to block store after step."""
        agent, _, _, _, _, blocks = _make_agent()

        await agent.step("Do something")

        saved = blocks.saved_blocks.get("agent_001", {})
        assert "persona" in saved

    async def test_agent_state_saved(self):
        """Agent state is saved via repo after step."""
        agent, _, _, repo, _, _ = _make_agent()

        await agent.step("Test")

        # Repo should have been called with save
        saved_state = repo._state
        assert saved_state.id == "agent_001"


@pytest.mark.asyncio
class TestContextWindow:
    async def test_long_history_truncated(self):
        """When history exceeds context window, older messages are dropped."""
        from nanoletta.agent import Agent

        # Create state with tiny context window (1000 chars ~= 250 tokens)
        state = AgentState(
            id="agent_001",
            name="test",
            system_prompt="Short prompt.",
            blocks={},
            tools=[Tool(name="send_message", description="Send")],
            llm_config=LLMConfig(model="test", context_window=250),  # ~1000 chars
        )

        llm = MockLLMClient([
            MockLLMResponse(
                tool_calls=[ToolCall(id="call_1", name="send_message", arguments={"message": "Hi"})]
            ),
        ])
        tools = MockToolExecutor()
        repo = MockAgentRepo(state)
        sessions = MockSessionStore()
        blocks = MockBlockStore()

        # Pre-fill history with many long messages
        sessions.messages["agent_001"] = [
            Message(role="user", content="x" * 200, agent_id="agent_001")
            for _ in range(20)
        ]

        agent = Agent("agent_001", llm, tools, repo, sessions, blocks)
        await agent.step("New message")

        # The context sent to LLM should NOT include all 20 old messages
        # (that would be 4000+ chars, way over the 750 char budget for history)
        total_context_chars = sum(len(m.get("content", "")) for m in llm.last_messages)
        assert total_context_chars < 2000  # Well under full history size of 4200+


@pytest.mark.asyncio
class TestUsageTracking:
    async def test_usage_accumulated_across_steps(self):
        """Usage tokens accumulate across multiple steps."""
        agent, *_ = _make_agent(
            llm_responses=[
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_1", name="memory_insert", arguments={"label": "x", "value": "y"})],
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                ),
                MockLLMResponse(
                    tool_calls=[ToolCall(id="call_2", name="send_message", arguments={"message": "Done"})],
                    usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
                ),
            ]
        )

        result = await agent.step("Work")

        assert result.usage["prompt_tokens"] == 220
        assert result.usage["completion_tokens"] == 80
        assert result.usage["total_tokens"] == 300
        assert result.steps == 2
