"""
Tests for nanoletta/store.py — SQLite persistence.

Covers:
1. Agent CRUD (create, load, save, exists)
2. Block persistence (save, get, read-only preserved)
3. Message persistence (save, get, count, ordering)
4. Round-trip: create agent → modify blocks → save → reload → verify
5. Schema creation on fresh DB
"""

import os
import tempfile

import pytest

from nanoletta.types import AgentState, Block, LLMConfig, Message, Tool


@pytest.fixture
def store(tmp_path):
    from nanoletta.store import SQLiteStore
    db_path = tmp_path / "test.db"
    s = SQLiteStore(db_path=db_path)
    yield s
    s.close()


@pytest.mark.asyncio
class TestAgentRepo:
    async def test_create_agent(self, store):
        state = await store.create(
            name="test_agent",
            system_prompt="You are helpful.",
            blocks={"persona": Block(label="persona", value="I am NanoLetta")},
            tools=[Tool(name="send_message", description="Send")],
            llm_config=LLMConfig(model="gpt-4o"),
        )

        assert state.id
        assert state.name == "test_agent"
        assert await store.exists(state.id)

    async def test_load_agent(self, store):
        created = await store.create(
            name="loader",
            system_prompt="Test prompt",
            blocks={"human": Block(label="human", value="Dominic")},
        )

        loaded = await store.load(created.id)

        assert loaded.id == created.id
        assert loaded.name == "loader"
        assert loaded.system_prompt == "Test prompt"
        assert loaded.get_block("human").value == "Dominic"

    async def test_load_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="not found"):
            await store.load("nonexistent_id")

    async def test_save_updates_existing(self, store):
        state = await store.create(name="updater", system_prompt="v1")

        state.system_prompt = "v2"
        await store.save(state)

        reloaded = await store.load(state.id)
        assert reloaded.system_prompt == "v2"

    async def test_exists_false_for_missing(self, store):
        assert not await store.exists("nonexistent")

    async def test_tools_round_trip(self, store):
        tools = [
            Tool(name="send_message", description="Send", parameters={"type": "object"}),
            Tool(name="search", description="Search things"),
        ]
        state = await store.create(name="tooled", system_prompt="", tools=tools)

        loaded = await store.load(state.id)
        assert len(loaded.tools) == 2
        assert loaded.tools[0].name == "send_message"
        assert loaded.tools[1].name == "search"

    async def test_llm_config_round_trip(self, store):
        config = LLMConfig(model="ollama/qwen", temperature=0.3, base_url="http://localhost:11434/v1")
        state = await store.create(name="configured", system_prompt="", llm_config=config)

        loaded = await store.load(state.id)
        assert loaded.llm_config.model == "ollama/qwen"
        assert loaded.llm_config.temperature == 0.3
        assert loaded.llm_config.is_local


@pytest.mark.asyncio
class TestBlockStore:
    async def test_save_and_get_blocks(self, store):
        blocks = {
            "persona": Block(label="persona", value="I am an agent"),
            "human": Block(label="human", value="The user is Dom"),
        }
        await store.save_blocks("agent_1", blocks)

        loaded = await store.get_blocks("agent_1")
        assert len(loaded) == 2
        assert loaded["persona"].value == "I am an agent"
        assert loaded["human"].value == "The user is Dom"

    async def test_save_single_block(self, store):
        await store.save_block("agent_1", Block(label="notes", value="hello"))

        loaded = await store.get_blocks("agent_1")
        assert "notes" in loaded
        assert loaded["notes"].value == "hello"

    async def test_read_only_preserved(self, store):
        await store.save_block("agent_1", Block(label="locked", value="sacred", read_only=True))

        loaded = await store.get_blocks("agent_1")
        assert loaded["locked"].read_only is True

    async def test_empty_blocks_for_unknown_agent(self, store):
        blocks = await store.get_blocks("unknown")
        assert blocks == {}


@pytest.mark.asyncio
class TestSessionStore:
    async def test_save_and_get_messages(self, store):
        msgs = [
            Message(role="user", content="Hello", agent_id="agent_1"),
            Message(role="assistant", content="Hi back", agent_id="agent_1"),
        ]
        await store.save_messages("agent_1", msgs)

        loaded = await store.get_messages("agent_1")
        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert loaded[1].role == "assistant"

    async def test_message_count(self, store):
        msgs = [Message(role="user", content=f"msg {i}", agent_id="agent_1") for i in range(5)]
        await store.save_messages("agent_1", msgs)

        count = await store.count_messages("agent_1")
        assert count == 5

    async def test_messages_ordered_by_time(self, store):
        msgs = [
            Message(role="user", content="first", agent_id="agent_1", created_at="2026-01-01T00:00:00"),
            Message(role="user", content="second", agent_id="agent_1", created_at="2026-01-01T01:00:00"),
        ]
        await store.save_messages("agent_1", msgs)

        loaded = await store.get_messages("agent_1")
        assert loaded[0].content == "first"
        assert loaded[1].content == "second"

    async def test_limit_respected(self, store):
        msgs = [Message(role="user", content=f"msg {i}", agent_id="agent_1") for i in range(20)]
        await store.save_messages("agent_1", msgs)

        loaded = await store.get_messages("agent_1", limit=5)
        assert len(loaded) == 5

    async def test_limit_returns_most_recent(self, store):
        """With limit, should return the MOST RECENT N messages, not the oldest."""
        msgs = [
            Message(role="user", content=f"msg {i}", agent_id="agent_1",
                    created_at=f"2026-01-01T{i:02d}:00:00")
            for i in range(10)
        ]
        await store.save_messages("agent_1", msgs)

        loaded = await store.get_messages("agent_1", limit=3)
        assert len(loaded) == 3
        # Should be msg 7, 8, 9 (most recent), ordered oldest→newest
        assert loaded[0].content == "msg 7"
        assert loaded[1].content == "msg 8"
        assert loaded[2].content == "msg 9"

    async def test_tool_calls_round_trip(self, store):
        msg = Message(
            role="assistant",
            content="",
            agent_id="agent_1",
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
        )
        await store.save_messages("agent_1", [msg])

        loaded = await store.get_messages("agent_1")
        assert len(loaded[0].tool_calls) == 1
        assert loaded[0].tool_calls[0]["function"]["name"] == "test"


@pytest.mark.asyncio
class TestRoundTrip:
    async def test_full_agent_lifecycle(self, store):
        """Create agent → modify blocks → save → reload → verify."""
        # Create
        state = await store.create(
            name="lifecycle_test",
            system_prompt="Be helpful",
            blocks={
                "persona": Block(label="persona", value="I am v1"),
                "human": Block(label="human", value="User is Dom"),
            },
            tools=[Tool(name="send_message")],
            llm_config=LLMConfig(model="test-model"),
        )

        # Modify blocks in-place (simulating agent tool execution)
        state.update_block_value("persona", "I am v2 — updated during reasoning")

        # Save
        await store.save(state)

        # Save some messages
        await store.save_messages(state.id, [
            Message(role="user", content="Hi", agent_id=state.id),
            Message(role="assistant", content="Hello!", agent_id=state.id),
        ])

        # Reload from scratch
        reloaded = await store.load(state.id)

        assert reloaded.name == "lifecycle_test"
        assert reloaded.get_block("persona").value == "I am v2 — updated during reasoning"
        assert reloaded.get_block("human").value == "User is Dom"
        assert len(reloaded.tools) == 1
        assert reloaded.llm_config.model == "test-model"

        # Check messages
        msgs = await store.get_messages(state.id)
        assert len(msgs) == 2
        assert msgs[0].content == "Hi"
        assert msgs[1].content == "Hello!"
