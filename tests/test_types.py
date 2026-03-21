"""
Tests for nanoletta/types.py — Core data types.

Covers:
- Block creation, validation, read-only enforcement
- Tool schema generation
- Message OpenAI format conversion
- LLMConfig local detection
- AgentState block CRUD, memory compilation, limit enforcement
- ToolCall and ToolResult construction
"""

from __future__ import annotations

import pytest

from nanoletta.types import (
    AgentState,
    Block,
    LLMConfig,
    Message,
    Tool,
    ToolCall,
    ToolResult,
    _new_id,
    _now_iso,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_new_id_is_12_chars(self):
        assert len(_new_id()) == 12

    def test_new_id_is_unique(self):
        ids = {_new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_now_iso_contains_timezone(self):
        ts = _now_iso()
        assert "+" in ts or "Z" in ts


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class TestBlock:
    def test_create_basic(self):
        b = Block(label="persona", value="I am an agent")
        assert b.label == "persona"
        assert b.value == "I am an agent"
        assert b.read_only is False

    def test_create_empty_value(self):
        b = Block(label="empty")
        assert b.value == ""

    def test_limit_enforcement(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            Block(label="big", value="x" * 6000, limit=5000)

    def test_limit_exact_boundary(self):
        # Exactly at limit should work
        b = Block(label="exact", value="x" * 5000, limit=5000)
        assert len(b.value) == 5000

    def test_custom_limit(self):
        b = Block(label="small", value="hello", limit=100)
        assert b.limit == 100

    def test_read_only_flag(self):
        b = Block(label="ro", value="locked", read_only=True)
        assert b.read_only is True


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    def test_create_basic(self):
        t = Tool(name="search", description="Search the web")
        assert t.name == "search"

    def test_openai_schema(self):
        t = Tool(
            name="send_message",
            description="Send a message",
            parameters={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        )
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "send_message"
        assert "properties" in schema["function"]["parameters"]

    def test_empty_parameters(self):
        t = Tool(name="noop")
        schema = t.to_openai_schema()
        assert schema["function"]["parameters"] == {}


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_create_user_message(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"
        assert len(m.id) == 12

    def test_auto_timestamp(self):
        m = Message(role="system", content="test")
        assert m.created_at is not None
        assert "T" in m.created_at

    def test_openai_dict_user(self):
        m = Message(role="user", content="hi")
        d = m.to_openai_dict()
        assert d == {"role": "user", "content": "hi"}

    def test_openai_dict_tool_response(self):
        m = Message(
            role="tool",
            content="result text",
            tool_call_id="call_123",
            name="search",
        )
        d = m.to_openai_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_123"
        assert d["name"] == "search"

    def test_openai_dict_assistant_with_tool_calls(self):
        m = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        )
        d = m.to_openai_dict()
        assert d["role"] == "assistant"
        assert len(d["tool_calls"]) == 1

    def test_openai_dict_no_extra_fields(self):
        """User messages should not include tool_call_id or tool_calls."""
        m = Message(role="user", content="test")
        d = m.to_openai_dict()
        assert "tool_call_id" not in d
        assert "tool_calls" not in d
        assert "name" not in d


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.model == "gpt-4o"
        assert c.temperature == 0.7

    def test_is_local_ollama(self):
        c = LLMConfig(base_url="http://localhost:11434/v1")
        assert c.is_local is True

    def test_is_local_127(self):
        c = LLMConfig(base_url="http://127.0.0.1:11434/v1")
        assert c.is_local is True

    def test_is_not_local(self):
        c = LLMConfig(base_url="https://api.openai.com/v1")
        assert c.is_local is False


# ---------------------------------------------------------------------------
# ToolCall and ToolResult
# ---------------------------------------------------------------------------

class TestToolCallAndResult:
    def test_tool_call_creation(self):
        tc = ToolCall(id="call_1", name="search", arguments={"query": "hello"})
        assert tc.name == "search"
        assert tc.arguments["query"] == "hello"

    def test_tool_result_success(self):
        tr = ToolResult(output="found 3 results", success=True)
        assert tr.success is True

    def test_tool_result_error(self):
        tr = ToolResult(output="", success=False, error="not found")
        assert tr.success is False
        assert tr.error == "not found"


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------

class TestAgentState:
    def test_create_default(self):
        state = AgentState()
        assert len(state.id) == 12
        assert state.name == "agent"
        assert state.blocks == {}
        assert state.tools == []

    def test_get_block(self):
        state = AgentState(blocks={"persona": Block(label="persona", value="test")})
        block = state.get_block("persona")
        assert block.value == "test"

    def test_get_block_missing_raises(self):
        state = AgentState()
        with pytest.raises(KeyError, match="No block"):
            state.get_block("nonexistent")

    def test_set_block(self):
        state = AgentState()
        state.set_block(Block(label="new", value="hello"))
        assert state.get_block("new").value == "hello"

    def test_set_block_overwrites(self):
        state = AgentState(blocks={"x": Block(label="x", value="old")})
        state.set_block(Block(label="x", value="new"))
        assert state.get_block("x").value == "new"

    def test_update_block_value(self):
        state = AgentState(blocks={"persona": Block(label="persona", value="old")})
        state.update_block_value("persona", "new")
        assert state.get_block("persona").value == "new"

    def test_update_block_read_only_raises(self):
        state = AgentState(
            blocks={"locked": Block(label="locked", value="x", read_only=True)}
        )
        with pytest.raises(ValueError, match="read-only"):
            state.update_block_value("locked", "changed")

    def test_update_block_exceeds_limit_raises(self):
        state = AgentState(
            blocks={"small": Block(label="small", value="x", limit=10)}
        )
        with pytest.raises(ValueError, match="exceeds limit"):
            state.update_block_value("small", "x" * 20)

    def test_compile_memory_basic(self):
        state = AgentState(
            blocks={
                "persona": Block(label="persona", value="I am an agent"),
                "human": Block(label="human", value="The user is Dominic"),
            }
        )
        memory = state.compile_memory()
        assert "[persona]" in memory
        assert "I am an agent" in memory
        assert "[human]" in memory
        assert "The user is Dominic" in memory

    def test_compile_memory_skips_empty(self):
        state = AgentState(
            blocks={
                "full": Block(label="full", value="content"),
                "empty": Block(label="empty", value=""),
            }
        )
        memory = state.compile_memory()
        assert "[full]" in memory
        assert "[empty]" not in memory

    def test_compile_memory_includes_description(self):
        state = AgentState(
            blocks={
                "stance": Block(
                    label="stance",
                    value="listen",
                    description="Current agent posture",
                ),
            }
        )
        memory = state.compile_memory()
        assert "(Current agent posture)" in memory
