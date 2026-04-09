"""
Tests for nanoletta/governor.py — Governor layer hooks.

Covers:
1. All hooks are no-ops when no runtime is injected
2. Hooks respect config toggles
3. GovernorResult shape is consistent
4. build_consciousness_block returns None when no runtime
5. post_tool only fires for memory tools
6. Injected runtime is called correctly
7. Exceptions in runtime don't crash hooks
"""

from __future__ import annotations

import pytest

from nanoletta.governor import Governor, GovernorConfig, GovernorResult, GovernanceRuntime
from nanoletta.types import AgentState, Block, ToolCall
from typing import Any


def _default_state() -> AgentState:
    return AgentState(
        id="agent_001",
        name="test",
        blocks={"persona": Block(label="persona", value="test agent")},
    )


class _StubRuntime:
    """Minimal GovernanceRuntime for testing."""

    def __init__(self):
        self.calls: list[str] = []

    def on_open_question(self, user_message, ai_response, person="user"):
        self.calls.append("on_open_question")
        return {"tracked": True}

    def on_draft_response(self, draft_response, self_model_payload, conversation_history, doctrine_snippets):
        self.calls.append("on_draft_response")
        return {"consistency_risk": 0.3, "contradictions": [], "should_emit_signal": False}

    def on_correction(self, user_message, ai_response, correction_type):
        self.calls.append("on_correction")
        return {"recorded": True}

    def get_context_block(self, max_items=3):
        self.calls.append("get_context_block")
        return "Active reminders:\n- Stay concise"

    def on_daemon_cycle(self, commitments, emotional_state, homeostasis_state, last_user_message_ts, chat_id):
        self.calls.append("on_daemon_cycle")
        return {"should_send": True, "message": "Check in", "priority": 0.5}


class _RaisingRuntime:
    """Runtime that always raises — hooks must not propagate the exception."""

    def on_open_question(self, *a, **kw): raise RuntimeError("boom")
    def on_draft_response(self, *a, **kw): raise RuntimeError("boom")
    def on_correction(self, *a, **kw): raise RuntimeError("boom")
    def get_context_block(self, *a, **kw): raise RuntimeError("boom")
    def on_daemon_cycle(self, *a, **kw): raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Tests: No runtime — all hooks are no-ops
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestNoRuntime:
    """Without a runtime, all hooks return inactive GovernorResults."""

    async def test_pre_step_noop(self):
        gov = Governor()
        result = await gov.pre_step("hello", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "pre_step"
        assert result.active is False

    async def test_post_draft_noop(self):
        gov = Governor()
        result = await gov.post_draft("draft response", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_draft"
        assert result.active is False

    async def test_post_tool_noop(self):
        gov = Governor()
        tc = ToolCall(id="1", name="memory_str_replace", arguments={})
        result = await gov.post_tool("user msg", "ai resp", tc, _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_tool"
        assert result.active is False

    async def test_post_step_noop(self):
        gov = Governor()
        result = await gov.post_step("user msg", "ai resp", _default_state())
        assert isinstance(result, GovernorResult)
        assert result.hook == "post_step"
        assert result.active is False

    async def test_daemon_cycle_noop(self):
        gov = Governor()
        result = await gov.daemon_cycle()
        assert isinstance(result, GovernorResult)
        assert result.hook == "daemon_cycle"
        assert result.active is False

    def test_consciousness_block_noop(self):
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
        gov = Governor(config, runtime=_StubRuntime())
        result = await gov.post_draft("draft", _default_state())
        assert result.active is False

    async def test_disabled_correction_bridge(self):
        config = GovernorConfig(enable_correction_bridge=False)
        gov = Governor(config, runtime=_StubRuntime())
        tc = ToolCall(id="1", name="memory_str_replace", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert result.active is False

    async def test_disabled_open_questions(self):
        config = GovernorConfig(enable_open_questions=False)
        gov = Governor(config, runtime=_StubRuntime())
        result = await gov.pre_step("question?", _default_state())
        assert result.active is False

    def test_disabled_consciousness_block(self):
        config = GovernorConfig(enable_consciousness_block=False)
        gov = Governor(config, runtime=_StubRuntime())
        result = gov.build_consciousness_block()
        assert result is None

    async def test_disabled_initiative(self):
        config = GovernorConfig(enable_initiative=False)
        gov = Governor(config, runtime=_StubRuntime())
        result = await gov.daemon_cycle()
        assert result.active is False


# ---------------------------------------------------------------------------
# Tests: post_tool only fires for memory tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPostToolFiltering:
    async def test_memory_tool_fires(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        for tool_name in ["memory_str_replace", "memory_insert", "memory_rethink", "memory_delete"]:
            tc = ToolCall(id="1", name=tool_name, arguments={})
            result = await gov.post_tool("msg", "resp", tc, _default_state())
            assert result.hook == "post_tool"
            assert result.active is True

    async def test_non_memory_tool_skipped(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        tc = ToolCall(id="1", name="send_message", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert result.active is False
        assert "on_correction" not in runtime.calls

    async def test_custom_tool_skipped(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
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
# Tests: Runtime is called correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestWithRuntime:
    async def test_pre_step_calls_runtime(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        result = await gov.pre_step("hello?", _default_state())
        assert result.active is True
        assert result.result.get("tracked") is True
        assert "on_open_question" in runtime.calls

    async def test_post_draft_calls_runtime(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        result = await gov.post_draft("test draft", _default_state())
        assert result.active is True
        assert result.result.get("consistency_risk") == 0.3
        assert "on_draft_response" in runtime.calls

    async def test_post_step_calls_runtime(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        result = await gov.post_step("user msg", "ai resp", _default_state())
        assert result.active is True
        assert "on_open_question" in runtime.calls

    def test_consciousness_block_calls_runtime(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        block = gov.build_consciousness_block()
        assert block is not None
        assert "Active reminders" in block
        assert "get_context_block" in runtime.calls

    async def test_daemon_cycle_calls_runtime(self):
        runtime = _StubRuntime()
        gov = Governor(runtime=runtime)
        result = await gov.daemon_cycle(commitments=[{"title": "Test"}])
        assert result.active is True
        assert result.result.get("should_send") is True
        assert result.result.get("message") == "Check in"
        assert "on_daemon_cycle" in runtime.calls

    async def test_person_config_passed_to_runtime(self):
        captured: list[str] = []

        class PersonCapture:
            def on_open_question(self, user_msg, ai_resp, person="user"):
                captured.append(person)
                return {}
            def on_draft_response(self, *a, **kw): return {}
            def on_correction(self, *a, **kw): return {}
            def get_context_block(self, *a, **kw): return None
            def on_daemon_cycle(self, *a, **kw): return {}

        gov = Governor(GovernorConfig(person="Alice"), runtime=PersonCapture())
        await gov.pre_step("question", _default_state())
        assert captured == ["Alice"]


# ---------------------------------------------------------------------------
# Tests: Runtime exceptions don't crash hooks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRuntimeExceptions:
    async def test_pre_step_exception_safe(self):
        gov = Governor(runtime=_RaisingRuntime())
        result = await gov.pre_step("hello", _default_state())
        assert isinstance(result, GovernorResult)
        assert "boom" in result.error

    async def test_post_draft_exception_safe(self):
        gov = Governor(runtime=_RaisingRuntime())
        result = await gov.post_draft("draft", _default_state())
        assert isinstance(result, GovernorResult)
        assert "boom" in result.error

    async def test_post_tool_exception_safe(self):
        gov = Governor(runtime=_RaisingRuntime())
        tc = ToolCall(id="1", name="memory_str_replace", arguments={})
        result = await gov.post_tool("msg", "resp", tc, _default_state())
        assert isinstance(result, GovernorResult)
        assert "boom" in result.error

    def test_consciousness_block_exception_safe(self):
        gov = Governor(runtime=_RaisingRuntime())
        result = gov.build_consciousness_block()
        assert result is None  # exception swallowed, returns None

    async def test_daemon_cycle_exception_safe(self):
        gov = Governor(runtime=_RaisingRuntime())
        result = await gov.daemon_cycle()
        assert isinstance(result, GovernorResult)
        assert "boom" in result.error
