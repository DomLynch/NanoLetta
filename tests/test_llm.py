"""
Tests for nanoletta/llm.py — OpenAI-compatible LLM client.

Covers:
1. Response parsing (content, tool calls, usage)
2. Tool call argument parsing (JSON + malformed)
3. Empty response handling
4. Multiple tool calls
"""

import pytest

from nanoletta.llm import CompletionResponse, OpenAICompatibleClient
from nanoletta.types import LLMConfig, ToolCall


class TestCompletionResponse:
    def test_defaults(self):
        r = CompletionResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.usage["total_tokens"] == 0

    def test_with_content(self):
        r = CompletionResponse(content="Hello!", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert r.content == "Hello!"
        assert r.usage["total_tokens"] == 15

    def test_with_tool_calls(self):
        r = CompletionResponse(
            tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "test"})],
        )
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "search"


class TestResponseParsing:
    """Test the _parse_response method directly."""

    def _client(self):
        return OpenAICompatibleClient()

    def test_parse_text_response(self):
        data = {
            "choices": [{"message": {"content": "Hello world", "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = self._client()._parse_response(data)

        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.usage["total_tokens"] == 15

    def test_parse_tool_call_response(self):
        data = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "send_message",
                            "arguments": '{"message": "Hi there!"}'
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = self._client()._parse_response(data)

        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "send_message"
        assert result.tool_calls[0].arguments == {"message": "Hi there!"}

    def test_parse_multiple_tool_calls(self):
        data = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "memory_insert", "arguments": '{"label": "notes", "value": "test"}'}},
                        {"id": "call_2", "function": {"name": "send_message", "arguments": '{"message": "done"}'}},
                    ]
                }
            }],
            "usage": {},
        }
        result = self._client()._parse_response(data)

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "memory_insert"
        assert result.tool_calls[1].name == "send_message"

    def test_parse_malformed_arguments(self):
        """Malformed JSON arguments should be captured as raw string."""
        data = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_bad",
                        "function": {
                            "name": "broken",
                            "arguments": "not valid json {"
                        }
                    }]
                }
            }],
            "usage": {},
        }
        result = self._client()._parse_response(data)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"raw": "not valid json {"}

    def test_parse_empty_response(self):
        data = {"choices": [{"message": {}}], "usage": {}}
        result = self._client()._parse_response(data)

        assert result.content == ""
        assert result.tool_calls == []
        assert result.usage["total_tokens"] == 0

    def test_parse_no_choices(self):
        data = {"choices": [], "usage": {}}
        result = self._client()._parse_response(data)

        assert result.content == ""
        assert result.tool_calls == []

    def test_parse_missing_usage(self):
        data = {"choices": [{"message": {"content": "test"}}]}
        result = self._client()._parse_response(data)

        assert result.content == "test"
        assert result.usage["total_tokens"] == 0
