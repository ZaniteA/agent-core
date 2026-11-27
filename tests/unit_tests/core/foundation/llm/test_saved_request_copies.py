# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Saved copies of the requests agent-core sends today (the Chat safety net).

Each setup below goes through the public ``Model`` entry point and the real
model client to the shared fake server. Its request (address, JSON body and
non-secret headers) must equal the copy saved in ``saved_requests/``. A
routing table records which client class and which first address each of a
set of configurations reaches.

The copies record today's behaviour as it is, known gaps included, so that a
change to the Chat or ChatGPT-login requests is always a deliberate one. To
update the copies after an intended change, run::

    OPENJIUWEN_UPDATE_SAVED_REQUESTS=1 python -m pytest <this file>

then review the diff of ``saved_requests/`` line by line before committing.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import (
    AssistantMessage,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from openjiuwen.core.foundation.llm.model_clients import create_model_client
from openjiuwen.core.foundation.llm.model_clients.openai_account_model_client import DEFAULT_OPENAI_ACCOUNT_BASE_URL
from openjiuwen.core.foundation.llm.schema.config import (
    IntelliRouterConfig,
    IntelliRouterDeploymentConfig,
    ReasoningConfig,
)
from openjiuwen.core.foundation.tool import ToolInfo
from tests.unit_tests.fixtures.fake_llm_servers import fake_model_server  # noqa: F401  (pytest fixture)

_SAVED_DIR = Path(__file__).parent / "saved_requests"
_UPDATE = os.getenv("OPENJIUWEN_UPDATE_SAVED_REQUESTS") == "1"
_API_BASE = "https://api.example.test/v1"
# A tiny PNG data URL; its bytes do not matter to the request shape.
_IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="

_WEATHER_TOOL = ToolInfo(
    name="get_weather",
    description="Look up the weather in a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
_WEATHER_CALL = ToolCall(id="call_1", type="function", name="get_weather", arguments='{"city": "Paris"}')
_TOOL_ROUND_TRIP = [
    UserMessage(content="What is the weather in Paris?"),
    AssistantMessage(content="", tool_calls=[_WEATHER_CALL]),
    ToolMessage(content="Sunny, 21 C.", tool_call_id="call_1"),
]
_IMAGE_QUESTION = [
    UserMessage(
        content=[
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
        ]
    )
]
_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


class _FakeOpenAIAccountAuthManager:
    """Stands in for the ChatGPT-login token store."""

    def resolve_access_token(self, *, force_refresh: bool = False) -> str:
        return "fake-access-token"


@dataclass(frozen=True)
class _Setup:
    """One model call whose request is saved."""

    name: str
    messages: Any = "Hello"
    client: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    call: dict[str, Any] = field(default_factory=dict)
    stream: bool = False


_CHAT_SETUPS = [
    _Setup("chat_plain_text", client={"custom_headers": {"X-Team": "search"}}),
    _Setup(
        "chat_plain_text_stream",
        messages=[SystemMessage(content="You are brief."), UserMessage(content="Hello")],
        stream=True,
    ),
    _Setup(
        "chat_sampling_on_openai_host",
        client={"api_base": "https://api.openai.com/v1"},
        request={"temperature": 0.3, "top_p": 0.9, "max_tokens": 256, "stop": "END"},
    ),
    _Setup("chat_tools", messages="What is the weather in Paris?", call={"tools": [_WEATHER_TOOL]}),
    _Setup(
        "chat_tool_choice",
        messages="What is the weather in Paris?",
        call={
            "tools": [_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": False,
        },
    ),
    _Setup("chat_tool_round_trip", messages=_TOOL_ROUND_TRIP, call={"tools": [_WEATHER_TOOL]}, stream=True),
    _Setup("chat_image_input", messages=_IMAGE_QUESTION),
    _Setup("chat_reasoning_effort", request={"model": "gpt-5.5", "reasoning": ReasoningConfig(effort="high")}),
    _Setup(
        "chat_thinking_off_dashscope",
        client={"client_provider": "DashScope", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        request={"model": "qwen3.6-plus", "reasoning": ReasoningConfig(mode="disabled")},
    ),
    _Setup(
        "chat_extra_body",
        request={"seed": 7},
        call={"extra_body": {"top_k": 20, "provider_option": {"region": "eu"}}},
    ),
    _Setup(
        "chat_deepseek_reasoning_content",
        client={"client_provider": "DeepSeek", "api_base": "https://api.deepseek.com"},
        request={"model": "deepseek-v4-pro"},
        messages=[
            UserMessage(content="What is the weather in Paris?"),
            AssistantMessage(content="", reasoning_content="I should look it up.", tool_calls=[_WEATHER_CALL]),
            ToolMessage(content="Sunny, 21 C.", tool_call_id="call_1"),
            AssistantMessage(content="It is sunny."),
            UserMessage(content="And tomorrow?"),
        ],
        call={"tools": [_WEATHER_TOOL]},
    ),
    _Setup("chat_structured_output", call={"response_format": _ANSWER_SCHEMA}),
    _Setup(
        "chat_openrouter_prompt_cache",
        client={"client_provider": "OpenRouter", "api_base": "https://openrouter.ai/api/v1"},
        request={"model": "anthropic/claude-sonnet-4.5"},
        messages=[SystemMessage(content="You are brief."), UserMessage(content="Hello")],
    ),
    _Setup(
        "chat_no_auth_local_server",
        client={"api_base": "http://localhost:8000/v1", "auth_mode": "none", "endpoint_profile": "vllm"},
        request={"model": "qwen3-8b"},
    ),
    _Setup(
        "chat_kv_cache_affinity",
        client={"client_provider": "AscendAffinity", "custom_headers": {"X-Gateway-Token": "gateway"}},
        call={"session_id": "session-1"},
    ),
]

_CHATGPT_LOGIN_SETUPS = [
    _Setup("chatgpt_login_plain_text"),
    _Setup(
        "chatgpt_login_stream_with_session",
        messages=[SystemMessage(content="You are brief."), UserMessage(content="Hello")],
        client={"custom_headers": {"X-Team": "search"}},
        call={"session_id": "session-1"},
        stream=True,
    ),
    _Setup(
        "chatgpt_login_tool_choice",
        messages="What is the weather in Paris?",
        call={
            "tools": [_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "parallel_tool_calls": False,
        },
    ),
    _Setup("chatgpt_login_tool_round_trip", messages=_TOOL_ROUND_TRIP, call={"tools": [_WEATHER_TOOL]}),
    _Setup("chatgpt_login_image_input", messages=_IMAGE_QUESTION),
    _Setup(
        "chatgpt_login_reasoning_and_limits",
        request={"temperature": 0.3, "max_tokens": 256, "reasoning": ReasoningConfig(effort="high")},
        call={"send_sampling_params": True, "send_max_output_tokens": True},
    ),
]


def _chat_client_config(**overrides: Any) -> ModelClientConfig:
    values = {"client_provider": "OpenAI", "api_key": "mock-api-key", "api_base": _API_BASE, "verify_ssl": False}
    return ModelClientConfig(**{**values, **overrides})


def _chatgpt_login_client_config(**overrides: Any) -> ModelClientConfig:
    values = {
        "client_provider": "OpenAIAccount",
        "api_key": "",
        "api_base": DEFAULT_OPENAI_ACCOUNT_BASE_URL,
        "verify_ssl": False,
        "openai_account_auth_manager": _FakeOpenAIAccountAuthManager(),
    }
    return ModelClientConfig(**{**values, **overrides})


def _request_config(**overrides: Any) -> ModelRequestConfig:
    return ModelRequestConfig(**{"model": "gpt-test", **overrides})


async def _run(model: Model, setup: _Setup) -> None:
    if setup.stream:
        async for _ in model.stream(setup.messages, **setup.call):
            pass
    else:
        await model.invoke(setup.messages, **setup.call)


def _assert_matches_saved_copy(name: str, actual: Any) -> None:
    path = _SAVED_DIR / f"{name}.json"
    if _UPDATE:
        _SAVED_DIR.mkdir(exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    assert path.exists(), (
        f"No saved copy at {path}. Run with OPENJIUWEN_UPDATE_SAVED_REQUESTS=1 to write it, then review it."
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert actual == saved, (
        f"The request differs from the saved copy {path.name}. If the change is intended, run with "
        "OPENJIUWEN_UPDATE_SAVED_REQUESTS=1 and review the diff."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", _CHAT_SETUPS, ids=lambda setup: setup.name)
async def test_chat_request_matches_saved_copy(fake_model_server, setup):
    model = Model(_chat_client_config(**setup.client), _request_config(**setup.request))

    await _run(model, setup)

    _assert_matches_saved_copy(setup.name, [request.saved_copy() for request in fake_model_server.requests])


@pytest.mark.asyncio
@pytest.mark.parametrize("setup", _CHATGPT_LOGIN_SETUPS, ids=lambda setup: setup.name)
async def test_chatgpt_login_request_matches_saved_copy(fake_model_server, setup):
    model = Model(_chatgpt_login_client_config(**setup.client), _request_config(**setup.request))

    await _run(model, setup)

    _assert_matches_saved_copy(setup.name, [request.saved_copy() for request in fake_model_server.requests])


# Configurations whose client class and first request address are saved. The
# Anthropic and IntelliRouter rows save the class only: the Anthropic SDK and
# the router choose their own addresses.
_ROUTES = {
    "openai_default": (_chat_client_config(), True),
    "openai_chat_completions_mode": (_chat_client_config(api_mode="chat_completions"), True),
    "openai_full_chat_address": (_chat_client_config(api_base=f"{_API_BASE}/chat/completions"), True),
    "openai_responses_mode": (_chat_client_config(api_mode="responses"), True),
    "openai_responses_mode_full_chat_address": (
        _chat_client_config(api_base=f"{_API_BASE}/chat/completions", api_mode="responses"),
        True,
    ),
    "openai_account_provider": (_chatgpt_login_client_config(), True),
    "openai_with_account_login": (
        _chat_client_config(api_mode="responses", auth_mode="openai_account_oauth", api_key="",
                            api_base=DEFAULT_OPENAI_ACCOUNT_BASE_URL,
                            openai_account_auth_manager=_FakeOpenAIAccountAuthManager()),
        True,
    ),
    "deepseek_provider": (_chat_client_config(client_provider="DeepSeek", api_base="https://api.deepseek.com"), True),
    "openrouter_responses_mode": (
        _chat_client_config(client_provider="OpenRouter", api_base="https://openrouter.ai/api/v1",
                            api_mode="responses"),
        True,
    ),
    "anthropic_provider": (
        _chat_client_config(client_provider="Anthropic", api_base="https://api.anthropic.com"),
        False,
    ),
    "anthropic_provider_responses_mode": (
        _chat_client_config(client_provider="Anthropic", api_base="https://api.anthropic.com", api_mode="responses"),
        False,
    ),
    "intelli_router": (
        ModelClientConfig(
            client_provider="intelli_router",
            verify_ssl=False,
            intelli_router=IntelliRouterConfig(
                model_group_id="group-1",
                deployments=[
                    IntelliRouterDeploymentConfig(
                        route_id="route-1",
                        model_name="qwen-turbo",
                        api_key="mock-api-key",
                        api_base="https://dashscope.aliyuncs.com",
                        provider="dashscope",
                    )
                ],
            ),
        ),
        False,
    ),
}


async def _first_request_address(fake_model_server, client) -> Optional[str]:
    try:
        await client.invoke("Hello")
    except BaseError:
        pass  # Only the address matters here, not whether the fake answer parses.
    if not fake_model_server.requests:
        return None
    first = fake_model_server.requests[0]
    return f"{first.method} {first.url}"


@pytest.mark.asyncio
async def test_routing_table_matches_saved_copy(fake_model_server):
    table = {}
    for name, (client_config, send_request) in _ROUTES.items():
        fake_model_server.requests.clear()
        client = create_model_client(client_config, _request_config())
        row = {"client": type(client).__name__}
        if send_request:
            row["first_request"] = await _first_request_address(fake_model_server, client)
        table[name] = row

    _assert_matches_saved_copy("routing_table", table)
