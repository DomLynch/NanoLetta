"""
Tests for nanoletta/governor.py — Governor layer hooks.

Covers:
1. All hooks degrade gracefully when Brain modules unavailable
2. Hooks respect config toggles
3. GovernorResult shape is consistent
4. consciousness_block returns None when unavailable
5. post_tool only fires for memory tools
6. Exceptions don't crash hooks
"""

import sys
import types

import pytest

from nanoletta.governor import Governor, GovernorConfig, GovernorResult
from nanoletta.types import AgentState, Block, Message, ToolCall


def _default_state() -> AgentState:
    return AgentState(
        id="agent_001",
        name="test",
        blocks={"persona": Block(label="persona", value="test agent")},
    )


# ---------------------------------------------------------------------------
# Tests: Graceful degradation (no Brain modules)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGracefulDegradation:
    """All hooks return safe defaults when consciousness_runtime is missing."""

    async def test_pre_step_degrades(self):
        gov = Governor()
        result = await gov.pre_step("hello", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "pre_step"
        # Should not crash

    async def test_post_draft_degrades(self):
        gov = Governor()
        result = await gov.post_draft("draft response", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_draft"

    async def test_post_tool_degrades(self):
        gov = Governor()
        tc = ToolCall(id="1", name="memory_str_replace", arguments={})
        result = await gov.post_tool("user msg", "ai resp", tc, _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_tool"

    async def test_post_step_degrades(self):
        gov = Governor()
        result = await gov.post_step("user msg", "ai resp", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_step"

    async def test_daemon_cycle_degrades(self):
        gov = Governor()
        result = await gov.daemon_cycle()
        assert isinstance(result, GovernorResult)
        assert result.hook == "daemon_cycle"

    def test_consciousness_block_degrades(self):
        gov = Governor()
        result = gov.build_consciousness_block()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Config toggles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConfigToggles:
    async def test_disabled_consistency_check(self):
        config = GovernorConfig(enable_consistency_check=False)
        gov = Governor(config)
        result = await gov.post_draft("draft", _default_state())
        assert result.active is False

    async def test_disabled_correction_bridge(self):
        config = GovernorConfig(enable_correction_bridge=False)
        gov = Governor(config)
        tc = ToolCall(id="1", name="memory_str_replace", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert result.active is False

    async def test_disabled_open_questions(self):
        config = GovernorConfig(enable_open_questions=False)
        gov = Governor(config)
        result = await gov.pre_step("question?", _default_state())
        assert result.active is False

    def test_disabled_consciousness_block(self):
        config = GovernorConfig(enable_consciousness_block=False)
        gov = Governor(config)
        result = gov.build_consciousness_block()
        assert result is None

    async def test_disabled_initiative(self):
        config = GovernorConfig(enable_initiative=False)
        gov = Governor(config)
        result = await gov.daemon_cycle()
        assert result.active is False


# ---------------------------------------------------------------------------
# Tests: post_tool only fires for memory tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPostToolFiltering:
    async def test_memory_tool_fires(self):
        gov = Governor()
        for tool_name in ["memory_str_replace", "memory_insert", "memory_rethink", "memory_delete"]:
            tc = ToolCall(id="1", name=tool_name, arguments={})
            result = await gov.post_tool("msg", "resp", tc, _default_state())
            # Should attempt to call consciousness_runtime (will get ImportError, that's fine)
            assert result.hook == "post_tool"

    async def test_non_memory_tool_skipped(self):
        gov = Governor()
        tc = ToolCall(id="1", name="send_message", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert result.active is False  # Skipped because not a memory tool

    async def test_custom_tool_skipped(self):
        gov = Governor()
        tc = ToolCall(id="1", name="web_search", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert result.active is False


# ---------------------------------------------------------------------------
# Tests: GovernorResult shape
# ---------------------------------------------------------------------------

class TestGovernorResult:
    def test_default_result(self):
        r = GovernorResult(hook="test")
        assert r.hook == "test"
        assert r.active is True
        assert r.result == {}
        assert r.error == ""

    def test_inactive_result(self):
        r = GovernorResult(hook="test", active=False)
        assert r.active is False

    def test_error_result(self):
        r = GovernorResult(hook="test", error="something broke")
        assert r.error == "something broke"


# ---------------------------------------------------------------------------
# Tests: With mocked consciousness_runtime
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWithMockedRuntime:
    async def test_post_draft_with_mock(self):
        """When consciousness_runtime exists, post_draft delegates correctly."""
        fake_mod = types.ModuleType("skills.consciousness_runtime")
        fake_mod.on_draft_response = lambda **kwargs: {"consistency_risk": 0.3, "contradictions": [], "should_emit_signal": False}

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "skills.consciousness_runtime", fake_mod)
            mp.setitem(sys.modules, "skills", types.ModuleType("skills"))

            gov = Governor()
            result = await gov.post_draft("test draft", _default_state())

            assert result.hook == "post_draft"
            assert result.result.get("consistency_risk") == 0.3

    async def test_daemon_cycle_with_mock(self):
        """When consciousness_runtime exists, daemon_cycle delegates correctly."""
        fake_mod = types.ModuleType("skills.consciousness_runtime")
        fake_mod.on_daemon_cycle = lambda **kwargs: {"should_send": True, "message": "Check in", "priority": 0.5}

        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "skills.consciousness_runtime", fake_mod)
            mp.setitem(sys.modules, "skills", types.ModuleType("skills"))

            gov = Governor()
            result = await gov.daemon_cycle(commitments=[{"title": "Test"}])

            assert result.result.get("should_send") is True
            assert result.result.get("message") == "Check in"
