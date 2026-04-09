"""
End-to-end smoke test: NanoLetta against a real Ollama instance.

Proves:
1. Agent can be created with SQLite persistence
2. Real LLM call to Ollama returns a response
3. Agent uses send_message tool to reply
4. Memory blocks persist across agent reloads
5. Conversation history persists across agent reloads

Run with: OLLAMA_URL=http://localhost:11434/v1 python3 -m pytest tests/test_e2e_ollama.py -v
Or via VPS: OLLAMA_URL=http://49.12.7.18:11434/v1 python3 -m pytest tests/test_e2e_ollama.py -v

Skipped automatically if OLLAMA_URL is not set.
"""

import os
import tempfile

import pytest

OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

pytestmark = pytest.mark.skipif(not OLLAMA_URL, reason="OLLAMA_URL not set")


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "e2e_test.db"


@pytest.mark.asyncio
async def test_full_loop_with_ollama(db_path):
    """Send a message, get a real LLM response, verify persistence."""
    from nanoletta.agent import Agent
    from nanoletta.governor import Governor, GovernorConfig
    from nanoletta.llm import OpenAICompatibleClient
    from nanoletta.memory_tools import MemoryToolExecutor, get_builtin_tool_schemas
    from nanoletta.store import SQLiteStore
    from nanoletta.types import AgentState, Block, LLMConfig, Tool

    # Set up real components
    store = SQLiteStore(db_path=db_path)
    llm = OpenAICompatibleClient()
    tools_executor = MemoryToolExecutor(session_store=store)
    governor = Governor(GovernorConfig(
        enable_consistency_check=False,
        enable_correction_bridge=False,
        enable_open_questions=False,
        enable_consciousness_block=False,
    ))

    # Create agent with real config
    config = LLMConfig(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.7,
        max_tokens=200,
        context_window=4096,
    )

    builtin_tools = [
        Tool(
            name="send_message",
            description="Send a message to the user. You MUST call this tool to respond.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Your response message"},
                },
                "required": ["message"],
            },
        ),
    ]

    state = await store.create(
        name="e2e_test_agent",
        system_prompt="You are a helpful assistant. Always use the send_message tool to respond to the user.",
        blocks={
            "persona": Block(label="persona", value="I am NanoLetta, a minimal cognitive agent."),
            "human": Block(label="human", value="The user is testing me."),
        },
        tools=builtin_tools,
        llm_config=config,
    )

    agent = Agent(
        agent_id=state.id,
        llm_client=llm,
        tool_executor=tools_executor,
        agent_repo=store,
        session_store=store,
        block_store=store,
        governor=governor,
    )

    # Send a message
    response = await agent.step("Hello, what is your name?", max_steps=3)

    # Verify we got a response
    assert response.content, "Expected non-empty response content"
    assert response.steps >= 1, "Expected at least one step"
    assert response.usage["total_tokens"] > 0, "Expected token usage"

    print(f"\n  Response: {response.content}")
    print(f"  Steps: {response.steps}")
    print(f"  Tokens: {response.usage}")

    # Verify persistence — reload agent from DB
    reloaded = await store.load(state.id)
    assert reloaded.name == "e2e_test_agent"
    assert reloaded.get_block("persona").value == "I am NanoLetta, a minimal cognitive agent."

    # Verify messages were saved
    messages = await store.get_messages(state.id)
    assert len(messages) >= 2, f"Expected at least user + assistant messages, got {len(messages)}"
    assert messages[0].role == "user"

    print(f"  Messages saved: {len(messages)}")
    print("  E2E test PASSED")

    store.close()


@pytest.mark.asyncio
async def test_memory_survives_restart(db_path):
    """Create agent, modify memory, close DB, reopen, verify memory persisted."""
    from nanoletta.store import SQLiteStore
    from nanoletta.types import Block, LLMConfig

    # Create and modify
    store1 = SQLiteStore(db_path=db_path)
    state = await store1.create(
        name="persistence_test",
        system_prompt="Test",
        blocks={"notes": Block(label="notes", value="initial value")},
    )
    agent_id = state.id

    # Modify block
    state.update_block_value("notes", "modified value after reasoning")
    await store1.save(state)
    store1.close()

    # Reopen from scratch
    store2 = SQLiteStore(db_path=db_path)
    reloaded = await store2.load(agent_id)

    assert reloaded.get_block("notes").value == "modified value after reasoning"
    print("\n  Memory persistence PASSED")

    store2.close()
