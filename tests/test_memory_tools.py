"""
Tests for nanoletta/memory_tools.py — Built-in tool executor with memory editing.

Covers:
1. send_message returns message text
2. memory_str_replace: unique find-and-replace
3. memory_str_replace: rejects non-unique old_string
4. memory_str_replace: rejects missing old_string
5. memory_str_replace: enforces read-only
6. memory_insert: insert at specific line
7. memory_insert: append at end (-1)
8. memory_rethink: wholesale replace
9. memory_rethink: creates new block
10. memory_delete: delete line range
11. conversation_search: finds matching messages
12. Unknown tool returns error
13. Custom handler for external tools
14. Tool schemas generation
"""

from __future__ import annotations

from typing import Any

import pytest

from nanoletta.types import AgentState, Block, Message, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_with_blocks(**blocks: str) -> AgentState:
    """Create an AgentState with named blocks."""
    return AgentState(
        id="agent_001",
        name="test",
        blocks={
            label: Block(label=label, value=value)
            for label, value in blocks.items()
        },
    )


def _state_with_readonly(label: str, value: str) -> AgentState:
    return AgentState(
        id="agent_001",
        blocks={label: Block(label=label, value=value, read_only=True)},
    )


class MockSessionStore:
    def __init__(self, messages: list[Message] | None = None):
        self._messages = messages or []

    async def get_messages(self, agent_id: str, limit: int = 50) -> list[Message]:
        return self._messages[:limit]

    async def save_messages(self, agent_id: str, messages: list[Message]) -> None:
        self._messages.extend(messages)

    async def count_messages(self, agent_id: str) -> int:
        return len(self._messages)


def _executor(session_store=None, custom_handler=None):
    from nanoletta.memory_tools import MemoryToolExecutor
    return MemoryToolExecutor(session_store=session_store, custom_handler=custom_handler)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSendMessage:
    async def test_returns_message(self):
        ex = _executor()
        result = await ex.execute(
            ToolCall(id="1", name="send_message", arguments={"message": "Hello!"}),
            _state_with_blocks(),
        )
        assert result.success
        assert result.output == "Hello!"

    async def test_empty_message_fails(self):
        ex = _executor()
        result = await ex.execute(
            ToolCall(id="1", name="send_message", arguments={"message": ""}),
            _state_with_blocks(),
        )
        assert not result.success
        assert "requires" in result.error


@pytest.mark.asyncio
class TestMemoryStrReplace:
    async def test_unique_replacement(self):
        state = _state_with_blocks(persona="I am a test agent")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_str_replace", arguments={
                "label": "persona",
                "old_string": "test agent",
                "new_string": "cognitive twin",
            }),
            state,
        )

        assert result.success
        assert state.get_block("persona").value == "I am a cognitive twin"

    async def test_missing_old_string_fails(self):
        state = _state_with_blocks(persona="I am a test agent")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_str_replace", arguments={
                "label": "persona",
                "old_string": "not found text",
                "new_string": "replacement",
            }),
            state,
        )

        assert not result.success
        assert "not found" in result.error

    async def test_non_unique_old_string_fails(self):
        state = _state_with_blocks(persona="hello hello world")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_str_replace", arguments={
                "label": "persona",
                "old_string": "hello",
                "new_string": "hi",
            }),
            state,
        )

        assert not result.success
        assert "2 times" in result.error

    async def test_read_only_block_fails(self):
        state = _state_with_readonly("locked", "sacred text")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_str_replace", arguments={
                "label": "locked",
                "old_string": "sacred",
                "new_string": "profane",
            }),
            state,
        )

        assert not result.success
        assert "read-only" in result.error
        assert state.get_block("locked").value == "sacred text"

    async def test_missing_label_fails(self):
        ex = _executor()
        result = await ex.execute(
            ToolCall(id="1", name="memory_str_replace", arguments={
                "label": "",
                "old_string": "x",
                "new_string": "y",
            }),
            _state_with_blocks(),
        )
        assert not result.success


@pytest.mark.asyncio
class TestMemoryInsert:
    async def test_insert_at_line(self):
        state = _state_with_blocks(notes="line 0\nline 1\nline 2")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_insert", arguments={
                "label": "notes",
                "new_string": "inserted",
                "insert_line": 1,
            }),
            state,
        )

        assert result.success
        lines = state.get_block("notes").value.split("\n")
        assert lines[0] == "line 0"
        assert lines[1] == "inserted"
        assert lines[2] == "line 1"

    async def test_append_at_end(self):
        state = _state_with_blocks(notes="line 0\nline 1")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_insert", arguments={
                "label": "notes",
                "new_string": "appended",
                "insert_line": -1,
            }),
            state,
        )

        assert result.success
        assert state.get_block("notes").value.endswith("appended")

    async def test_out_of_range_fails(self):
        state = _state_with_blocks(notes="line 0")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_insert", arguments={
                "label": "notes",
                "new_string": "bad",
                "insert_line": 99,
            }),
            state,
        )

        assert not result.success
        assert "out of range" in result.error


@pytest.mark.asyncio
class TestMemoryRethink:
    async def test_wholesale_replace(self):
        state = _state_with_blocks(persona="old content")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_rethink", arguments={
                "label": "persona",
                "new_memory": "completely new content",
            }),
            state,
        )

        assert result.success
        assert state.get_block("persona").value == "completely new content"

    async def test_creates_new_block(self):
        state = _state_with_blocks()  # empty blocks
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_rethink", arguments={
                "label": "new_block",
                "new_memory": "fresh content",
            }),
            state,
        )

        assert result.success
        assert state.get_block("new_block").value == "fresh content"


@pytest.mark.asyncio
class TestMemoryDelete:
    async def test_delete_line_range(self):
        state = _state_with_blocks(notes="line 0\nline 1\nline 2\nline 3")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_delete", arguments={
                "label": "notes",
                "start_line": 1,
                "end_line": 2,
            }),
            state,
        )

        assert result.success
        lines = state.get_block("notes").value.split("\n")
        assert lines == ["line 0", "line 3"]

    async def test_out_of_range_fails(self):
        state = _state_with_blocks(notes="line 0")
        ex = _executor()

        result = await ex.execute(
            ToolCall(id="1", name="memory_delete", arguments={
                "label": "notes",
                "start_line": 5,
            }),
            state,
        )

        assert not result.success
        assert "out of range" in result.error


@pytest.mark.asyncio
class TestConversationSearch:
    async def test_finds_matching_messages(self):
        sessions = MockSessionStore(messages=[
            Message(role="user", content="What about investing in crypto?"),
            Message(role="assistant", content="I think crypto is risky"),
            Message(role="user", content="Tell me about dogs"),
        ])
        ex = _executor(session_store=sessions)
        state = _state_with_blocks()

        result = await ex.execute(
            ToolCall(id="1", name="conversation_search", arguments={"query": "crypto"}),
            state,
        )

        assert result.success
        assert "crypto" in result.output.lower()
        assert "dogs" not in result.output.lower()

    async def test_no_session_store(self):
        ex = _executor()  # no session store
        result = await ex.execute(
            ToolCall(id="1", name="conversation_search", arguments={"query": "test"}),
            _state_with_blocks(),
        )
        assert result.success
        assert "not available" in result.output


@pytest.mark.asyncio
class TestUnknownAndCustomTools:
    async def test_unknown_tool_fails(self):
        ex = _executor()
        result = await ex.execute(
            ToolCall(id="1", name="nonexistent_tool", arguments={}),
            _state_with_blocks(),
        )
        assert not result.success
        assert "Unknown tool" in result.error

    async def test_custom_handler(self):
        async def my_tool(tc: ToolCall, state: AgentState) -> ToolResult:
            return ToolResult(output=f"Custom: {tc.name}", success=True)

        ex = _executor(custom_handler=my_tool)
        result = await ex.execute(
            ToolCall(id="1", name="my_custom_tool", arguments={}),
            _state_with_blocks(),
        )
        assert result.success
        assert "Custom: my_custom_tool" in result.output


@pytest.mark.asyncio
class TestToolSchemas:
    async def test_builtin_schemas_generated(self):
        from nanoletta.memory_tools import get_builtin_tool_schemas

        schemas = get_builtin_tool_schemas()

        assert len(schemas) == 6
        names = {s["function"]["name"] for s in schemas}
        assert "send_message" in names
        assert "memory_str_replace" in names
        assert "memory_insert" in names
        assert "memory_rethink" in names
        assert "memory_delete" in names
        assert "conversation_search" in names

        # All should have function type
        for s in schemas:
            assert s["type"] == "function"
            assert "parameters" in s["function"]
