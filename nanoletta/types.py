"""
nanoletta/types.py — Core data types.

Minimal data models for the cognitive agent kernel.
No ORM coupling, no Pydantic validation overhead.
Plain dataclasses with explicit fields.

Design decisions:
- dataclasses over Pydantic: simpler, fewer dependencies, faster
- str IDs over UUIDs: easier debugging, human-readable
- dict metadata over typed fields: extensible without schema changes
- All fields have defaults: partial construction always works
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _new_id() -> str:
    """Generate a short, human-readable ID."""
    return uuid.uuid4().hex[:12]


def _now_iso() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Block — a labeled chunk of self-editing memory
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One memory block the agent can read and edit.

    Blocks are the agent's working memory. Each has a label (like a
    variable name) and a value (the content). The agent edits these
    via tool calls during reasoning.

    Invariants enforced on every mutation (not just construction):
    - read_only blocks reject value changes after creation
    - value length cannot exceed limit

    Examples:
        Block(label="persona", value="You are a cognitive twin of Dominic...")
        Block(label="human", value="Dominic is a founder based in Dubai...")
        Block(label="stance", value="listen")
    """

    label: str
    value: str = ""
    description: str = ""
    read_only: bool = False
    limit: int = 5000  # max chars for this block

    # Track whether __init__ is complete (for __setattr__ guard)
    _initialized: bool = False

    def __post_init__(self) -> None:
        if len(self.value) > self.limit:
            raise ValueError(
                f"Block '{self.label}' value ({len(self.value)} chars) "
                f"exceeds limit ({self.limit} chars)"
            )
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, val: object) -> None:
        """Enforce read-only and limit invariants on every mutation."""
        if name == "_initialized" or not self.__dict__.get("_initialized", False):
            # Allow setting during __init__
            object.__setattr__(self, name, val)
            return

        if name == "value":
            if self.read_only:
                raise ValueError(f"Block '{self.label}' is read-only")
            if isinstance(val, str) and len(val) > self.limit:
                raise ValueError(
                    f"Block '{self.label}' value ({len(val)} chars) "
                    f"exceeds limit ({self.limit} chars)"
                )

        if name == "read_only" and self.read_only:
            # Cannot un-set read_only once set
            if not val:
                raise ValueError(f"Block '{self.label}' is read-only and cannot be unlocked")

        object.__setattr__(self, name, val)


# ---------------------------------------------------------------------------
# Tool — a function the agent can call
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A tool/function the agent can invoke during reasoning.

    The schema follows OpenAI's function calling format:
    {"type": "function", "function": {"name": ..., "parameters": ...}}
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Message — a single turn in conversation
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """One message in the conversation history.

    Roles follow the OpenAI convention:
    - system: instructions/context
    - user: human input
    - assistant: agent response
    - tool: tool execution result
    """

    VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

    role: str  # system | user | assistant | tool
    content: str = ""
    id: str = field(default_factory=_new_id)
    agent_id: str = ""
    created_at: str = field(default_factory=_now_iso)
    tool_call_id: str = ""  # for role=tool responses
    name: str = ""  # tool name for role=tool
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid message role '{self.role}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_ROLES))}"
            )

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI chat completion message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.role == "tool" and self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.role == "tool" and self.name:
            msg["name"] = self.name

        if self.role == "assistant" and self.tool_calls:
            msg["tool_calls"] = self.tool_calls

        return msg


# ---------------------------------------------------------------------------
# LLMConfig — model configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for the LLM backend.

    model uses OpenAI-compatible format. For Ollama, prefix with
    the base URL or use the model name directly.
    """

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int = 128000
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""

    @property
    def is_local(self) -> bool:
        """Check if this points to a local model (Ollama etc)."""
        return "localhost" in self.base_url or "127.0.0.1" in self.base_url


# ---------------------------------------------------------------------------
# ToolCall — parsed tool invocation from LLM response
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A parsed tool call from the LLM's response."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ToolResult — result of executing a tool
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of executing a tool call."""

    output: str = ""
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# AgentState — everything needed to run one agent
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Complete state of a single agent.

    Deliberately minimal. 7 core fields instead of Letta's 25+.
    No org/user/project/template/identity/group coupling.

    The agent loop loads this at the start of each step,
    tools may modify blocks in-place, and persistence saves
    the updated state after each step.
    """

    id: str = field(default_factory=_new_id)
    name: str = "agent"
    system_prompt: str = ""
    blocks: dict[str, Block] = field(default_factory=dict)
    tools: list[Tool] = field(default_factory=list)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_block(self, label: str) -> Block:
        """Get a block by label. Raises KeyError if not found."""
        if label not in self.blocks:
            raise KeyError(f"No block with label '{label}'")
        return self.blocks[label]

    def set_block(self, block: Block) -> None:
        """Add or replace a block. Cannot replace a read-only block."""
        existing = self.blocks.get(block.label)
        if existing is not None and existing.read_only:
            raise ValueError(
                f"Block '{block.label}' is read-only and cannot be replaced"
            )
        self.blocks[block.label] = block

    def update_block_value(self, label: str, value: str) -> None:
        """Update a block's value.

        Delegates to Block.__setattr__ which enforces read-only and limit.
        """
        block = self.get_block(label)
        block.value = value  # Block.__setattr__ enforces invariants

    def compile_memory(self) -> str:
        """Render all blocks into a single memory string for the prompt."""
        sections: list[str] = []
        for label, block in self.blocks.items():
            if not block.value:
                continue
            header = f"[{label}]"
            if block.description:
                header += f" ({block.description})"
            sections.append(f"{header}\n{block.value}")
        return "\n\n".join(sections)
