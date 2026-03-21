"""
nanoletta/llm.py — OpenAI-compatible LLM client.

One client that works with:
- OpenAI API (GPT-4o, etc.)
- Anthropic via OpenAI-compat proxy (or direct)
- Ollama (local models via OpenAI-compatible endpoint)
- Any OpenAI-compatible API

Replaces Letta's:
- llm_api/openai_client.py (1,418 LOC)
- llm_api/anthropic_client.py (1,838 LOC)
- llm_api/llm_client_base.py (463 LOC)
- 15+ other provider clients (~3,500 LOC)
- interfaces/ (3,677 LOC)
- local_llm/ (5,471 LOC)

Total Letta LLM layer: ~16,367 LOC → ~120 LOC
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from nanoletta.types import LLMConfig, ToolCall

_log = logging.getLogger("nanoletta.llm")


@dataclass
class CompletionResponse:
    """Response from an LLM completion request.

    Conforms to the LLMResponse Protocol defined in interfaces.py.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })


class OpenAICompatibleClient:
    """LLM client using the OpenAI chat completions API format.

    Works with any API that follows the OpenAI /v1/chat/completions spec.
    Conforms to the LLMClient Protocol defined in interfaces.py.
    """

    def __init__(self, default_config: LLMConfig | None = None) -> None:
        self._default_config = default_config

    async def request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: LLMConfig,
    ) -> CompletionResponse:
        """Send a chat completion request to an OpenAI-compatible API.

        Uses httpx for async HTTP. Falls back to urllib if httpx unavailable.
        """
        try:
            import httpx
        except ImportError:
            return await self._request_urllib(messages, tools, config)

        url = f"{config.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        _log.debug("LLM request to %s model=%s messages=%d tools=%d",
                    url, config.model, len(messages), len(tools))

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data)

    async def _request_urllib(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        config: LLMConfig,
    ) -> CompletionResponse:
        """Fallback using urllib (no httpx dependency)."""
        import urllib.request

        url = f"{config.base_url.rstrip('/')}/chat/completions"

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        _log.debug("LLM request (urllib) to %s model=%s", url, config.model)

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return self._parse_response(data)

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        """Parse OpenAI chat completion response into CompletionResponse."""
        choices = data.get("choices", [])
        if not choices:
            return CompletionResponse()

        choice = choices[0]
        message = choice.get("message", {})

        # Parse content
        content = message.get("content", "") or ""

        # Parse tool calls (may be None or missing)
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {"raw": args_str}

            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            ))

        # Parse usage
        raw_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
            "completion_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
        )
