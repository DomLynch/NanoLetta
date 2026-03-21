"""
nanoletta/memory_tools.py — Built-in tool executor with memory self-editing.

Implements the ToolExecutor interface with built-in tools for:
- send_message: agent's reply to the user
- memory_str_replace: find-and-replace within a memory block
- memory_insert: insert text at a line position in a block
- memory_rethink: wholesale replace a block's value
- memory_delete: remove lines from a block
- conversation_search: search conversation history

The agent calls these tools during reasoning to edit its own memory.
All memory operations modify AgentState.blocks in-place. The caller
(agent.py) persists the changes after the step completes.

Transplanted from Letta's core_tool_executor.py (1,068 LOC → ~200 LOC).
Removed: line number regex validation, patch/diff application, archival
memory search, file operations, multi-agent tools, composio integration.
"""

import logging
from typing import Any

from nanoletta.interfaces import SessionStore
from nanoletta.types import AgentState, Block, ToolCall, ToolResult

_log = logging.getLogger("nanoletta.memory_tools")


class MemoryToolExecutor:
    """Built-in tool executor with memory self-editing capabilities.

    Implements the ToolExecutor Protocol. Handles built-in tools
    directly and delegates unknown tools to a custom handler if provided.

    All memory operations modify agent_state.blocks in-place.
    Block invariants (read-only, limit) are enforced by Block.__setattr__.
    """

    def __init__(
        self,
        session_store: SessionStore | None = None,
        custom_handler: Any | None = None,
    ) -> None:
        """
        Args:
            session_store: For conversation_search tool. Optional.
            custom_handler: Callable(tool_call, agent_state) -> ToolResult
                           for custom/external tools. Optional.
        """
        self._sessions = session_store
        self._custom = custom_handler

        # Map of built-in tool names to handler methods
        self._builtins: dict[str, Any] = {
            "send_message": self._send_message,
            "memory_str_replace": self._memory_str_replace,
            "memory_insert": self._memory_insert,
            "memory_rethink": self._memory_rethink,
            "memory_delete": self._memory_delete,
            "conversation_search": self._conversation_search,
        }

    async def execute(
        self,
        tool_call: ToolCall,
        agent_state: AgentState,
    ) -> ToolResult:
        """Execute a tool call. Dispatches to built-in or custom handler."""
        handler = self._builtins.get(tool_call.name)

        if handler is not None:
            try:
                output = await handler(tool_call.arguments, agent_state)
                return ToolResult(output=str(output), success=True)
            except (ValueError, KeyError) as exc:
                _log.warning("Tool %s failed: %s", tool_call.name, exc)
                return ToolResult(output="", success=False, error=str(exc))
            except Exception as exc:
                _log.error("Tool %s unexpected error: %s", tool_call.name, exc)
                return ToolResult(output="", success=False, error=str(exc))

        # Try custom handler
        if self._custom is not None:
            try:
                return await self._custom(tool_call, agent_state)
            except Exception as exc:
                return ToolResult(output="", success=False, error=str(exc))

        return ToolResult(
            output="",
            success=False,
            error=f"Unknown tool: {tool_call.name}",
        )

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    async def _send_message(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Send a message to the user. Returns the message text."""
        message = args.get("message", "")
        if not message:
            raise ValueError("send_message requires a 'message' argument")
        return message

    async def _memory_str_replace(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Find and replace text within a memory block.

        Requires the old_string to appear exactly once in the block.
        This prevents ambiguous replacements.
        """
        label = args.get("label", "")
        old_string = str(args.get("old_string", "")).expandtabs()
        new_string = str(args.get("new_string", "")).expandtabs()

        if not label:
            raise ValueError("memory_str_replace requires 'label'")
        if not old_string:
            raise ValueError("memory_str_replace requires 'old_string'")

        block = agent_state.get_block(label)
        current_value = str(block.value).expandtabs()

        # Check uniqueness
        occurrences = current_value.count(old_string)
        if occurrences == 0:
            raise ValueError(
                f"old_string not found in block '{label}'. "
                f"The text must appear verbatim in the block."
            )
        if occurrences > 1:
            raise ValueError(
                f"old_string appears {occurrences} times in block '{label}'. "
                f"It must be unique. Provide more context to disambiguate."
            )

        new_value = current_value.replace(old_string, new_string)
        agent_state.update_block_value(label, new_value)

        _log.debug("memory_str_replace on '%s': %d chars → %d chars", label, len(current_value), len(new_value))
        return new_value

    async def _memory_insert(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Insert text at a specific line position in a memory block.

        insert_line: 0-indexed. -1 or omitted = append at end.
        """
        label = args.get("label", "")
        new_string = str(args.get("new_string", args.get("value", ""))).expandtabs()
        insert_line = int(args.get("insert_line", -1))

        if not label:
            raise ValueError("memory_insert requires 'label'")
        if not new_string:
            raise ValueError("memory_insert requires 'new_string' or 'value'")

        block = agent_state.get_block(label)
        current_value = str(block.value).expandtabs()
        lines = current_value.split("\n")
        n_lines = len(lines)

        if insert_line == -1:
            insert_line = n_lines
        elif insert_line < 0 or insert_line > n_lines:
            raise ValueError(
                f"insert_line {insert_line} is out of range (0-{n_lines})"
            )

        new_lines = new_string.split("\n")
        result_lines = lines[:insert_line] + new_lines + lines[insert_line:]
        new_value = "\n".join(result_lines)

        agent_state.update_block_value(label, new_value)

        _log.debug("memory_insert on '%s' at line %d", label, insert_line)
        return new_value

    async def _memory_rethink(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Wholesale replace a block's value, or create a new block.

        If the block doesn't exist, creates it.
        """
        label = args.get("label", "")
        new_memory = str(args.get("new_memory", args.get("value", "")))

        if not label:
            raise ValueError("memory_rethink requires 'label'")

        try:
            agent_state.get_block(label)
            agent_state.update_block_value(label, new_memory)
        except KeyError:
            # Create new block
            new_block = Block(label=label, value=new_memory)
            agent_state.set_block(new_block)

        _log.debug("memory_rethink on '%s': %d chars", label, len(new_memory))
        return new_memory

    async def _memory_delete(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Delete specific lines from a memory block.

        start_line and end_line are 0-indexed, inclusive.
        """
        label = args.get("label", "")
        start_line = int(args.get("start_line", 0))
        end_line = int(args.get("end_line", -1))

        if not label:
            raise ValueError("memory_delete requires 'label'")

        block = agent_state.get_block(label)
        lines = str(block.value).split("\n")

        if end_line == -1:
            end_line = len(lines) - 1

        if start_line < 0 or start_line >= len(lines):
            raise ValueError(f"start_line {start_line} out of range (0-{len(lines) - 1})")
        if end_line < start_line or end_line >= len(lines):
            raise ValueError(f"end_line {end_line} out of range ({start_line}-{len(lines) - 1})")

        result_lines = lines[:start_line] + lines[end_line + 1:]
        new_value = "\n".join(result_lines)

        agent_state.update_block_value(label, new_value)

        _log.debug("memory_delete on '%s': lines %d-%d", label, start_line, end_line)
        return new_value

    async def _conversation_search(
        self,
        args: dict[str, Any],
        agent_state: AgentState,
    ) -> str:
        """Search conversation history for matching messages."""
        query = args.get("query", "")
        limit = int(args.get("limit", 10))

        if not query:
            raise ValueError("conversation_search requires 'query'")

        if self._sessions is None:
            return "Conversation search not available (no session store configured)"

        messages = await self._sessions.get_messages(agent_state.id, limit=100)

        # Simple substring matching (can be replaced with semantic search)
        query_lower = query.lower()
        matches: list[str] = []
        for msg in messages:
            if query_lower in msg.content.lower():
                matches.append(f"[{msg.role}] {msg.content[:200]}")
                if len(matches) >= limit:
                    break

        if not matches:
            return f"No messages found matching '{query}'"

        return "\n".join(matches)


def get_builtin_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for all built-in tools.

    Use this to populate AgentState.tools at agent creation time.
    """
    from nanoletta.types import Tool

    tools = [
        Tool(
            name="send_message",
            description="Send a message to the user. Use this when you want to respond.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user",
                    },
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="memory_str_replace",
            description="Find and replace text in a memory block. The old_string must appear exactly once.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Memory block label"},
                    "old_string": {"type": "string", "description": "Text to find (must be unique)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["label", "old_string", "new_string"],
            },
        ),
        Tool(
            name="memory_insert",
            description="Insert text at a line position in a memory block. Use -1 to append.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Memory block label"},
                    "new_string": {"type": "string", "description": "Text to insert"},
                    "insert_line": {"type": "integer", "description": "Line number (0-indexed, -1 for end)"},
                },
                "required": ["label", "new_string"],
            },
        ),
        Tool(
            name="memory_rethink",
            description="Replace entire contents of a memory block, or create a new one.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Memory block label"},
                    "new_memory": {"type": "string", "description": "New content for the block"},
                },
                "required": ["label", "new_memory"],
            },
        ),
        Tool(
            name="memory_delete",
            description="Delete lines from a memory block.",
            parameters={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Memory block label"},
                    "start_line": {"type": "integer", "description": "First line to delete (0-indexed)"},
                    "end_line": {"type": "integer", "description": "Last line to delete (inclusive, -1 for last)"},
                },
                "required": ["label"],
            },
        ),
        Tool(
            name="conversation_search",
            description="Search conversation history for messages containing a query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
        ),
    ]

    return [t.to_openai_schema() for t in tools]
