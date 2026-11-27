# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The API-key Responses client must build its address and headers like Chat mode."""

from __future__ import annotations

import httpx
import pytest

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.llm.schema.config import LLMAuthMode
from openjiuwen.core.foundation.llm.utils.responses_utils import OpenAIAccountResponsesError, build_headers

_RESPONSES_SSE = (
    "event: response.output_text.delta\n"
    'data: {"delta":"ok"}\n\n'
    "event: response.completed\n"
    'data: {"response":{"id":"resp-1","status":"completed",'
    '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
).encode()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _responses_model(sent_requests: list[httpx.Request], api_base: str, **client_config) -> Model:
    """A Responses-mode Model built from ``api_base`` whose HTTP calls are answered locally and recorded.

    Only the network is replaced: the client still builds its own transport, so the address
    comes from the config as in real use.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, content=_RESPONSES_SSE, headers={"content-type": "text/event-stream"})

    client_config.setdefault("api_key", "sk-test")
    model = Model(
        ModelClientConfig(
            client_provider="OpenAI",
            api_base=api_base,
            api_mode="responses",
            verify_ssl=False,
            **client_config,
        ),
        ModelRequestConfig(model="gpt-5.6-luna"),
    )
    client = model._client
    build_transport = client._make_responses_transport

    def make_transport(*, timeout):
        transport = build_transport(timeout=timeout)
        transport._transport = httpx.MockTransport(handler)
        return transport

    client._make_responses_transport = make_transport
    return model


async def _call(model: Model, *, stream: bool, **kwargs) -> None:
    if stream:
        async for _ in model.stream("hello", **kwargs):
            pass
    else:
        await model.invoke("hello", **kwargs)


@pytest.mark.parametrize(
    "api_base",
    [
        _OPENROUTER_BASE,
        f"{_OPENROUTER_BASE}/",
        f"{_OPENROUTER_BASE}/chat/completions",
        f"{_OPENROUTER_BASE}/chat/completions/",
        f"{_OPENROUTER_BASE}/responses",
        f"{_OPENROUTER_BASE}/responses/",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_request_goes_to_one_responses_path(api_base, stream):
    """A Chat-style base is cleaned up as in Chat mode, and ``/responses`` is never added twice."""
    sent_requests: list[httpx.Request] = []
    model = _responses_model(sent_requests, api_base)

    await _call(model, stream=stream)

    assert [str(request.url) for request in sent_requests] == [f"{_OPENROUTER_BASE}/responses"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_no_key_sends_no_authorization_header(stream):
    """``auth_mode="none"`` sends no ``Authorization`` header, as in Chat mode (not ``Bearer EMPTY``)."""
    sent_requests: list[httpx.Request] = []
    model = _responses_model(
        sent_requests,
        "http://localhost:11434/v1",
        api_key="",
        auth_mode=LLMAuthMode.NoneAuth,
        endpoint_profile="ollama",
    )

    await _call(model, stream=stream)

    assert len(sent_requests) == 1
    assert "authorization" not in sent_requests[0].headers


@pytest.mark.asyncio
async def test_custom_headers_without_key_send_only_the_custom_headers():
    sent_requests: list[httpx.Request] = []
    model = _responses_model(
        sent_requests,
        "https://gateway.example/v1",
        api_key="",
        auth_mode=LLMAuthMode.CustomHeaders,
        custom_headers={"X-Api-Key": "gateway-key"},
    )

    await model.invoke("hello")

    assert "authorization" not in sent_requests[0].headers
    assert sent_requests[0].headers["X-Api-Key"] == "gateway-key"


@pytest.mark.asyncio
async def test_api_key_is_still_sent_as_bearer_token():
    sent_requests: list[httpx.Request] = []
    model = _responses_model(sent_requests, _OPENROUTER_BASE, api_key="sk-test")

    await model.invoke("hello")

    assert sent_requests[0].headers["Authorization"] == "Bearer sk-test"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_session_headers_go_to_api_openai_com(stream):
    sent_requests: list[httpx.Request] = []
    model = _responses_model(sent_requests, "https://api.openai.com/v1")

    await _call(model, stream=stream, session_id="session-1")

    assert sent_requests[0].headers["session_id"] == "session-1"
    assert sent_requests[0].headers["x-client-request-id"] == "session-1"


@pytest.mark.parametrize(
    "api_base",
    [
        _OPENROUTER_BASE,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://api.deepseek.com",
        "http://localhost:8000/v1",
        # The host must be api.openai.com itself, not a name that merely starts with it.
        "https://api.openai.com.example/v1",
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_session_headers_stay_off_other_hosts(api_base, stream):
    sent_requests: list[httpx.Request] = []
    model = _responses_model(sent_requests, api_base)

    await _call(model, stream=stream, session_id="session-1")

    assert len(sent_requests) == 1
    assert "session_id" not in sent_requests[0].headers
    assert "x-client-request-id" not in sent_requests[0].headers


def test_build_headers_without_token_has_no_authorization():
    headers = build_headers(access_token=None, session_id="session-1", extra_headers={"X-Test": "value"})

    assert "Authorization" not in headers
    assert headers["X-Test"] == "value"
    assert headers["session_id"] == "session-1"


def test_build_headers_still_rejects_an_empty_token():
    """``None`` means "no key needed"; an empty token is still a mistake."""
    with pytest.raises(OpenAIAccountResponsesError, match="requires an access token"):
        build_headers(access_token="")
