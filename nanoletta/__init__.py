"""
nanoletta — Minimal cognitive agent kernel.

Extracted from Letta (134K LOC) into a lean single-agent runtime
with self-editing memory, tool execution, and governor hooks.

Usage:
    from nanoletta import AgentState, Block, LLMConfig
    from nanoletta.interfaces import LLMClient, BlockStore, SessionStore
"""

from nanoletta.types import (
    AgentState,
    Block,
    LLMConfig,
    Message,
    Tool,
    ToolCall,
    ToolResult,
)

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "Block",
    "LLMConfig",
    "Message",
    "Tool",
    "ToolCall",
    "ToolResult",
]
