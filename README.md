# nanoletta

> Letta's cognitive agent loop. 134,000 lines stripped to 1,900.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License-Apache--2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen.svg)](#tests)

NanoLetta is the cognitive kernel extracted from [Letta](https://github.com/letta-ai/letta). It keeps the one thing that actually matters — the reasoning loop — and removes everything else.

```
message → LLM → tool calls → memory edits → persist → repeat
```

One dependency. Zero infrastructure. Runs anywhere Python runs.

---

## Size comparison

| Component | Letta | NanoLetta | Reduction |
|-----------|-------|-----------|-----------|
| Agent reasoning loop | 1,969 LOC | 311 LOC | 84% |
| Memory / block system | ~2,000 LOC | 280 LOC | 86% |
| LLM client | ~800 LOC | 176 LOC | 78% |
| Persistence | ~3,000 LOC | 258 LOC | 91% |
| **Total** | **~134,000 LOC** | **~1,900 LOC** | **99%** |

What was cut: telemetry, multi-tenant routing, approval workflows, job scheduling, REST API surface, database migrations, Docker orchestration, span tracking, dry-run mode, step-progression enums, multi-agent routing, run ID tracking.

What remains: the cognition.

---

## Install

```bash
pip install httpx  # only external dependency
```

Copy the `nanoletta/` folder into your project. No pip package yet — the code is small enough to own directly.

```
nanoletta/
├── agent.py        # Reasoning loop
├── governor.py     # Extension hooks
├── interfaces.py   # Protocol definitions
├── llm.py          # OpenAI-compatible client
├── memory_tools.py # Built-in memory tools
├── store.py        # SQLite persistence
└── types.py        # Data model
```

---

## Quick start

```python
import asyncio
from nanoletta.agent import Agent
from nanoletta.store import SQLiteStore
from nanoletta.llm import OpenAICompatibleClient
from nanoletta.memory_tools import MemoryToolExecutor
from nanoletta.types import AgentState, Block, LLMConfig, Tool

async def main():
    # Persistence (SQLite, zero config)
    store = SQLiteStore("agent.db")

    # LLM client (OpenAI-compatible — works with Ollama, OpenRouter, etc.)
    llm = OpenAICompatibleClient(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )

    # Tool executor (built-in memory tools + your custom tools)
    async def my_tools(tool_call, state):
        # Add your custom tools here
        return None  # falls through to memory tools

    tools = MemoryToolExecutor(session_store=store, custom_handler=my_tools)

    # Create an agent with persistent memory blocks
    state = AgentState(
        agent_id="my-agent",
        system_prompt="You are a helpful assistant with persistent memory.",
        blocks=[
            Block(label="persona", value="Friendly, concise, honest."),
            Block(label="user_notes", value=""),  # Agent writes here
        ],
        tools=[],
        llm_config=LLMConfig(model="llama3.2", context_window=8192),
    )
    await store.save(state)

    # Run the agent
    agent = Agent(
        agent_id="my-agent",
        llm_client=llm,
        tool_executor=tools,
        agent_repo=store,
        session_store=store,
        block_store=store,
    )

    response = await agent.step("Hello! What do you remember about me?")
    print(response.content)

asyncio.run(main())
```

---

## How it works

### The reasoning loop

Each call to `agent.step()` runs this cycle until the agent sends a message or hits `max_steps`:

```
1. Load agent state + memory blocks from SQLite
2. Build context: system prompt + memory blocks + conversation history
3. Call LLM with available tools
4. Parse tool calls from response
5. Execute tools (memory edits, custom tools)
6. Persist messages + updated blocks
7. If agent called send_message → return response
   If agent used a thinking tool → go to step 2
```

### Self-editing memory

The agent can modify its own memory blocks mid-conversation using built-in tools:

| Tool | What it does |
|------|-------------|
| `send_message` | Send a reply to the user |
| `memory_str_replace` | Edit a memory block (find & replace) |
| `memory_insert` | Insert text into a memory block |
| `memory_rethink` | Rewrite a memory block from scratch |
| `memory_delete` | Remove text from a memory block |
| `conversation_search` | Search conversation history |

### Governor hooks

Six hook points let you inject governance logic without modifying the core loop. Pass a `GovernanceRuntime` to activate them — without one, all hooks are no-ops:

```python
from nanoletta.governor import Governor, GovernorConfig, GovernanceRuntime

class MyRuntime:
    def on_open_question(self, user_msg, ai_response, person="user"):
        # Track unanswered questions, update a knowledge gap log, etc.
        return {}

    def on_draft_response(self, draft, self_model, history, doctrine):
        # Consistency / safety check before response is committed
        # Return {"block": True} to signal the response should be held
        return {"ok": True}

    def on_correction(self, user_msg, ai_response, correction_type):
        # Called when the agent edits its own memory blocks
        return {}

    def get_context_block(self, max_items=3):
        # Return a string injected into the system prompt each turn
        return None  # or "Active reminders:\n- ..."

    def on_daemon_cycle(self, commitments, emotional_state,
                        homeostasis_state, last_user_ts, chat_id):
        # Proactive outreach / initiative decisions
        return {}

governor = Governor(
    config=GovernorConfig(person="Alice"),
    runtime=MyRuntime(),
)
agent = Agent(..., governor=governor)
```

| Hook | When it fires |
|------|--------------|
| `pre_step` | Before any LLM call |
| `post_draft` | After LLM generates a response |
| `post_tool` | After a memory-editing tool executes |
| `post_step` | After the full step completes |
| `build_consciousness_block` | During context assembly (injects text into system prompt) |
| `daemon_cycle` | External heartbeat (proactive / initiative logic) |

All hooks fail silently — a broken runtime cannot crash the agent loop. Subclass `Governor` instead of using a runtime if you prefer inheritance over composition.

---

## Persistence

NanoLetta uses SQLite with three tables:

```sql
agents    -- AgentState (system prompt, block refs, tool defs, LLM config)
blocks    -- Memory blocks (label, value, editable by the agent)
messages  -- Conversation history (role, content, tool calls)
```

WAL mode is enabled by default. Works fine for single-user agents. For concurrent sessions, wrap with a connection pool or swap in the `BlockStore` / `SessionStore` protocols with your preferred backend.

---

## Tests

```bash
# Unit tests (no LLM required)
python3 -m pytest tests/ -q --ignore=tests/test_e2e_ollama.py

# End-to-end test (requires Ollama)
OLLAMA_URL=http://localhost:11434/v1 OLLAMA_MODEL=llama3.2 \
  python3 -m pytest tests/test_e2e_ollama.py -v -s
```

135 unit tests. Covers types, agent loop, memory tools, SQLite store, LLM client, governor hooks.

---

## What was removed from Letta

NanoLetta is **not** a fork of the Letta API. It is a transplant of one specific module — `letta/agent/letta_agent.py` — with the essential logic kept and everything deployment-specific removed:

- **Telemetry / span tracking** (~200 LOC removed) — no OpenTelemetry dependency
- **Step progression enum + error recovery** (~100 LOC) — simplified to a clean loop
- **Approval / denial workflows** (~80 LOC) — not needed for single-agent use
- **Multi-agent routing** (~30 LOC) — out of scope
- **Job cancellation + run ID tracking** (~35 LOC) — removed
- **Dry run mode** (~20 LOC) — removed
- **REST API, database migrations, Docker** — entirely separate from the cognitive kernel

The `Governor` replaces Letta's step-manager callbacks with a cleaner hook interface.

---

## Part of a suite

NanoLetta pairs naturally with:

- **[Temporal](https://github.com/DomLynch/Temporal)** — temporal knowledge graph (Graphiti → 2.8k LOC). Knows *when* facts were true.
- **[Lucid](https://github.com/DomLynch/Lucid)** — semantic memory runtime (Hindsight → 2k LOC). `retain() / recall() / reflect()`.

Wire them into NanoLetta via custom tools and the Governor hooks.

---

## Extending NanoLetta

NanoLetta is designed to be extended through interfaces, not forked:

```python
from nanoletta.interfaces import ToolExecutor
from nanoletta.types import ToolCall, ToolResult, AgentState

class MyToolExecutor(ToolExecutor):
    async def execute(self, tool_call: ToolCall, state: AgentState) -> ToolResult:
        match tool_call.name:
            case "web_search":
                result = await do_web_search(tool_call.arguments["query"])
                return ToolResult(output=result, success=True)
            case _:
                return ToolResult(output="unknown tool", success=False)
```

Swap any of the five protocol interfaces (`LLMClient`, `BlockStore`, `SessionStore`, `ToolExecutor`, `AgentRepo`) with your own implementation.

---

## Requirements

- Python 3.11+
- `httpx` (for the built-in LLM client — omit if you bring your own)
- Any OpenAI-compatible LLM endpoint (Ollama, OpenRouter, OpenAI, Anthropic-compat, etc.)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Acknowledgments

The cognitive loop design and memory block concept come from [Letta](https://github.com/letta-ai/letta) (Apache 2.0), originally MemGPT. NanoLetta is an independent extraction — not affiliated with Letta AI.
