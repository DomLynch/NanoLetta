"""
nanoletta/store.py — SQLite persistence for agents, blocks, and messages.

Implements all three persistence interfaces (AgentRepo, BlockStore,
SessionStore) using a single SQLite database. Designed for single-user,
single-agent use cases.

Replaces Letta's:
- agent_manager.py (3,554 LOC) → ~50 LOC
- message_manager.py (1,329 LOC) → ~50 LOC
- block_manager.py (1,054 LOC) → ~40 LOC
- 50+ ORM models (4,905 LOC) → 3 tables
- alembic/ (10,577 LOC) → inline CREATE TABLE

Total Letta persistence: ~21,419 LOC → ~250 LOC
"""

import json
import logging
import sqlite3
from pathlib import Path
from nanoletta.types import AgentState, Block, LLMConfig, Message, Tool

_log = logging.getLogger("nanoletta.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    llm_config TEXT NOT NULL DEFAULT '{}',
    tools TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS blocks (
    agent_id TEXT NOT NULL,
    label TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    read_only INTEGER NOT NULL DEFAULT 0,
    block_limit INTEGER NOT NULL DEFAULT 5000,
    PRIMARY KEY (agent_id, label)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_call_id TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL DEFAULT '',
    tool_calls TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_id, created_at);
"""


class SQLiteStore:
    """Unified persistence using SQLite.

    Implements AgentRepo, BlockStore, and SessionStore Protocols.
    Single file, zero external dependencies beyond stdlib.
    """

    def __init__(self, db_path: str | Path = "nanoletta.db") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # AgentRepo interface
    # ------------------------------------------------------------------

    async def load(self, agent_id: str) -> AgentState:
        """Load full agent state by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if row is None:
            raise KeyError(f"Agent '{agent_id}' not found")

        blocks = await self.get_blocks(agent_id)
        tools = [Tool(**t) for t in json.loads(row["tools"])]
        llm_config = LLMConfig(**json.loads(row["llm_config"]))

        return AgentState(
            id=row["id"],
            name=row["name"],
            system_prompt=row["system_prompt"],
            blocks=blocks,
            tools=tools,
            llm_config=llm_config,
            metadata=json.loads(row["metadata"]),
        )

    async def save(self, state: AgentState) -> None:
        """Save full agent state (upsert)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO agents (id, name, system_prompt, llm_config, tools, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name,
                 system_prompt=excluded.system_prompt,
                 llm_config=excluded.llm_config,
                 tools=excluded.tools,
                 metadata=excluded.metadata,
                 updated_at=datetime('now')""",
            (
                state.id,
                state.name,
                state.system_prompt,
                json.dumps({"model": state.llm_config.model, "temperature": state.llm_config.temperature,
                            "max_tokens": state.llm_config.max_tokens, "context_window": state.llm_config.context_window,
                            "base_url": state.llm_config.base_url, "api_key": state.llm_config.api_key}),
                json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters} for t in state.tools]),
                json.dumps(state.metadata),
            ),
        )
        await self.save_blocks(state.id, state.blocks)
        conn.commit()

    async def exists(self, agent_id: str) -> bool:
        conn = self._get_conn()
        row = conn.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return row is not None

    async def create(
        self,
        name: str,
        system_prompt: str,
        blocks: dict[str, Block] | None = None,
        tools: list[Tool] | None = None,
        llm_config: LLMConfig | None = None,
    ) -> AgentState:
        """Create a new agent."""
        from nanoletta.types import _new_id
        state = AgentState(
            id=_new_id(),
            name=name,
            system_prompt=system_prompt,
            blocks=blocks or {},
            tools=tools or [],
            llm_config=llm_config or LLMConfig(),
        )
        await self.save(state)
        _log.info("Created agent '%s' (id=%s)", name, state.id)
        return state

    # ------------------------------------------------------------------
    # BlockStore interface
    # ------------------------------------------------------------------

    async def get_blocks(self, agent_id: str) -> dict[str, Block]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM blocks WHERE agent_id = ?", (agent_id,)
        ).fetchall()
        return {
            row["label"]: Block(
                label=row["label"],
                value=row["value"],
                description=row["description"],
                read_only=bool(row["read_only"]),
                limit=row["block_limit"],
            )
            for row in rows
        }

    async def save_block(self, agent_id: str, block: Block) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO blocks (agent_id, label, value, description, read_only, block_limit)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(agent_id, label) DO UPDATE SET
                 value=excluded.value,
                 description=excluded.description,
                 read_only=excluded.read_only,
                 block_limit=excluded.block_limit""",
            (agent_id, block.label, block.value, block.description, int(block.read_only), block.limit),
        )
        conn.commit()

    async def save_blocks(self, agent_id: str, blocks: dict[str, Block]) -> None:
        conn = self._get_conn()
        # Delete existing blocks for this agent, then insert all
        conn.execute("DELETE FROM blocks WHERE agent_id = ?", (agent_id,))
        for block in blocks.values():
            conn.execute(
                "INSERT INTO blocks (agent_id, label, value, description, read_only, block_limit) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, block.label, block.value, block.description, int(block.read_only), block.limit),
            )
        conn.commit()

    # ------------------------------------------------------------------
    # SessionStore interface
    # ------------------------------------------------------------------

    async def get_messages(self, agent_id: str, limit: int = 50) -> list[Message]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM messages WHERE agent_id = ? ORDER BY created_at ASC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [
            Message(
                id=row["id"],
                agent_id=row["agent_id"],
                role=row["role"],
                content=row["content"],
                tool_call_id=row["tool_call_id"],
                name=row["name"],
                tool_calls=json.loads(row["tool_calls"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def save_messages(self, agent_id: str, messages: list[Message]) -> None:
        conn = self._get_conn()
        for msg in messages:
            conn.execute(
                """INSERT OR IGNORE INTO messages
                   (id, agent_id, role, content, tool_call_id, name, tool_calls, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg.id, agent_id, msg.role, msg.content, msg.tool_call_id,
                 msg.name, json.dumps(msg.tool_calls), msg.created_at),
            )
        conn.commit()

    async def count_messages(self, agent_id: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return row["cnt"] if row else 0
