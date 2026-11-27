# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Self-test of the shared fake model server, through the real model clients.

The fault tests also record what agent-core does today when a stream breaks.
Some of that is a known gap that a later change fixes on purpose: a Responses
stream that ends without ``response.completed`` counts as a finished answer.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.llm.utils.responses_transport import OpenAIAccountResponsesTransport
from tests.unit_tests.fixtures.fake_llm_servers import (  # noqa: F401  (fake_model_server is a fixture)
    REDACTED,
    FakeModelServer,
    FakeToolCall,
    StreamFault,
    fake_model_server,
    faulty_stream,
    install_fake_server,
    reset_model_client_caches,
)

_API_BASE = "https://api.example.test/v1"
_PROTOCOLS = [pytest.param(None, id="chat"), pytest.param("responses", id="responses")]
# Events sent before the answer text has arrived in full: Chat sends the role
# chunk, then the text chunk; Responses sends response.created,
# response.in_progress, output_item.added, content_part.added, then the text
# delta.
_EVENTS_UP_TO_TEXT = {None: 2, "responses": 5}
_WEATHER_CALL = FakeToolCall(name="get_weather", arguments={"city": "Paris"})


def _model(api_mode=None, **client_kwargs) -> Model:
    return Model(
        ModelClientConfig(
            client_provider="OpenAI",
            api_key="mock-api-key",
            api_base=_API_BASE,
            verify_ssl=False,
            api_mode=api_mode,
            **client_kwargs,
        ),
        ModelRequestConfig(model="gpt-test"),
    )


async def _call(model: Model, stream: bool):
    if not stream:
        return await model.invoke("hi")
    final = None
    async for chunk in model.stream("hi"):
        final = chunk if final is None else final + chunk
    return final


async def _stream_text(model: Model, received: list[str]) -> None:
    async for chunk in model.stream("hi"):
        received.append(chunk.content or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_text_reply_reaches_the_caller(fake_model_server, api_mode, stream):
    fake_model_server.add_reply(text="Hello!")

    message = await _call(_model(api_mode), stream)

    assert message.content == "Hello!"
    expected_path = "/v1/responses" if api_mode else "/v1/chat/completions"
    assert [request.path for request in fake_model_server.requests] == [expected_path]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_tool_call_and_reasoning_reach_the_caller(fake_model_server, api_mode, stream):
    fake_model_server.add_reply(reasoning="Check the weather.", tool_calls=[_WEATHER_CALL])

    message = await _call(_model(api_mode), stream)

    assert message.reasoning_content == "Check the weather."
    assert [(call.id, call.name, json.loads(call.arguments)) for call in message.tool_calls] == [
        ("call_1", "get_weather", {"city": "Paris"})
    ]


@pytest.mark.asyncio
async def test_saved_copy_keeps_agent_core_headers_and_hides_secrets(fake_model_server):
    await _model(custom_headers={"X-Team": "search", "X-Gateway-Token": "gateway"}).invoke("hi")

    saved = fake_model_server.requests[0].saved_copy()

    assert saved == {
        "method": "POST",
        "url": f"{_API_BASE}/chat/completions",
        "headers": {
            "accept": "application/json",
            "authorization": REDACTED,
            "content-type": "application/json",
            "x-gateway-token": REDACTED,
            "x-team": "search",
        },
        "body": {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-test", "stream": False},
    }


@pytest.mark.asyncio
async def test_openai_sdk_environment_does_not_reach_the_request(monkeypatch):
    # The OpenAI SDK turns these variables into OpenAI-Organization and
    # OpenAI-Project headers. Installing the fake server clears them, so a
    # saved copy is the same on every machine.
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-environment")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-environment")
    await reset_model_client_caches()
    server = install_fake_server(monkeypatch, FakeModelServer())

    try:
        await _model().invoke("hi")
    finally:
        await reset_model_client_caches()

    headers = server.requests[0].saved_copy()["headers"]
    assert not {"openai-organization", "openai-project"} & set(headers)


@pytest.mark.asyncio
async def test_script_runs_out_into_ok_and_unknown_addresses_get_404(fake_model_server):
    assert (await _model().invoke("hi")).content == "ok"

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{_API_BASE}/models")

    assert response.status_code == 404
    assert [request.path for request in fake_model_server.requests] == ["/v1/chat/completions", "/v1/models"]


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_error_reply_becomes_a_model_call_error(fake_model_server, api_mode):
    fake_model_server.add_error(400, "model not found")

    with pytest.raises(BaseError) as caught:
        await _model(api_mode).invoke("hi")

    assert caught.value.status == StatusCode.MODEL_CALL_FAILED
    assert "model not found" in str(caught.value)


@pytest.mark.asyncio
async def test_explicit_transport_hook_reaches_a_server_without_install():
    server = FakeModelServer()
    server.add_reply(text="Hello!")
    transport = OpenAIAccountResponsesTransport(base_url=_API_BASE, transport=server.transport())

    try:
        message = await transport.create_response(body={"model": "gpt-test", "input": []}, access_token="token")
    finally:
        await transport.aclose()

    assert message.content == "Hello!"
    assert server.requests[0].url == f"{_API_BASE}/responses"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_cut_stream_keeps_the_text_already_sent_and_then_fails(fake_model_server, api_mode):
    fake_model_server.add_reply(text="Hello", fault=StreamFault.cut(after=_EVENTS_UP_TO_TEXT[api_mode]))
    received: list[str] = []

    with pytest.raises(BaseError) as caught:
        await _stream_text(_model(api_mode), received)

    assert "".join(received) == "Hello"
    assert caught.value.status == StatusCode.MODEL_CALL_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_stream_without_closing_event_counts_as_finished_today(fake_model_server, api_mode):
    # Chat still got its finish and usage chunks; only ``data: [DONE]`` is
    # missing. Known gap for Responses: no finish reason and no usage, yet no
    # error. A later change makes this an error that is retried.
    fake_model_server.add_reply(text="Hello", fault=StreamFault.no_closing_event())

    message = await _call(_model(api_mode), stream=True)

    total_tokens = message.usage_metadata.total_tokens if message.usage_metadata else None
    assert (message.content, message.finish_reason, total_tokens) == (
        ("Hello", "null", None) if api_mode else ("Hello", "stop", 15)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_garbled_event_fails_the_stream(fake_model_server, api_mode):
    fake_model_server.add_reply(text="Hello", fault=StreamFault.garbled_line(after=1))

    with pytest.raises(BaseError) as caught:
        await _call(_model(api_mode), stream=True)

    assert caught.value.status == StatusCode.MODEL_CALL_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_stall_holds_the_stream_back(fake_model_server, api_mode):
    fake_model_server.add_reply(text="Hello", fault=StreamFault.stall(after=1, seconds=30))
    received: list[str] = []

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_stream_text(_model(api_mode), received), timeout=0.5)

    assert "".join(received) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", _PROTOCOLS)
async def test_short_stall_pauses_then_carries_on(fake_model_server, api_mode):
    fake_model_server.add_reply(text="Hello", fault=StreamFault.stall(after=1, seconds=0.1))
    started = time.monotonic()

    message = await _call(_model(api_mode), stream=True)

    # Lower bound only, with slack for coarse timers on Windows.
    assert time.monotonic() - started >= 0.05
    assert (message.content, message.finish_reason) == ("Hello", "stop")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [StreamFault.cut(after=4), StreamFault.garbled_line(after=-1), StreamFault.stall(after=9, seconds=0)],
    ids=["cut", "garbled_line", "stall"],
)
async def test_fault_that_never_fires_is_refused(fault):
    events = [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]

    with pytest.raises(ValueError, match="never fires on a stream of 3 events"):
        _ = [event async for event in faulty_stream(events, fault)]
