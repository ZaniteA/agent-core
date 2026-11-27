# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import BaseError, build_error
from openjiuwen.core.foundation.llm import (
    AssistantMessageChunk,
    LLMApiMode,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
)
from openjiuwen.core.foundation.llm.utils.responses_transport import OpenAIAccountResponsesTransport
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.rail.base import (
    MODEL_VISIBLE_OUTPUT_EMITTED_KEY,
    AgentCallbackContext,
)
from openjiuwen.harness.rails.model_anomaly_detection_rail import ModelAnomalyDetectionRail


_DEFAULT_ABC_REPEAT_COUNT = 54


def _make_ctx(agent=None):
    if agent is None:
        agent = MagicMock()
    return AgentCallbackContext(agent=agent, extra={})


def _make_agent() -> ReActAgent:
    return ReActAgent(card=AgentCard(description="anomaly detection rail test")).configure(
        ReActAgentConfig(
            model_config_obj=ModelRequestConfig(model="mock-model"),
            model_client_config=ModelClientConfig(
                client_provider="OpenAI",
                api_key="sk-test",
                api_base="https://mock.local/v1",
                verify_ssl=False,
            ),
            prompt_template=[{"role": "system", "content": "You are a test assistant."}],
        )
    )


class _RetryStreamModel:
    def __init__(self, mode: str):
        self.mode = mode
        self.call_count = 0
        self.last_messages = None

    async def invoke(self, **kwargs):
        raise NotImplementedError

    async def stream(self, **kwargs):
        self.call_count += 1
        self.last_messages = kwargs.get("messages")
        if self.mode in {"loop", "loop_exhausted"} and (self.mode == "loop_exhausted" or self.call_count == 1):
            for _ in range(_DEFAULT_ABC_REPEAT_COUNT):
                yield AssistantMessageChunk(reasoning_content="abc")
            return
        if self.mode in {"timeout", "timeout_exhausted"} and (self.mode == "timeout_exhausted" or self.call_count == 1):
            raise build_error(
                StatusCode.MODEL_CALL_FAILED,
                error_msg="LLM stream timeout: stream frame timeout: stage=idle_chunk",
            )
        if self.mode == "rate_limit" and self.call_count == 1:
            raise _wrapped_status_error(429, "too many requests")
        if self.mode == "rate_limit_after_text" and self.call_count == 1:
            yield AssistantMessageChunk(content="partial")
            raise _wrapped_status_error(429, "too many requests")
        yield AssistantMessageChunk(content="recovered")


class _ProviderError(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def _wrapped_status_error(status_code, message, headers=None, details=None):
    cause = _ProviderError(status_code, message, headers)
    return build_error(
        StatusCode.MODEL_CALL_FAILED,
        error_msg=message,
        cause=cause,
        details=details,
    )


@pytest.mark.asyncio
async def test_short_repeated_stream_output_below_total_threshold_is_ignored():
    rail = ModelAnomalyDetectionRail()
    ctx = _make_ctx()

    await rail.before_model_call(ctx)

    for _ in range(6):
        await rail.inspect_stream_chunk(ctx, AssistantMessageChunk(reasoning_content="abc"))


@pytest.mark.asyncio
async def test_repeated_stream_output_raises_model_error():
    rail = ModelAnomalyDetectionRail()
    ctx = _make_ctx()

    await rail.before_model_call(ctx)

    for _ in range(_DEFAULT_ABC_REPEAT_COUNT - 1):
        await rail.inspect_stream_chunk(ctx, AssistantMessageChunk(reasoning_content="abc"))

    with pytest.raises(BaseError) as exc_info:
        await rail.inspect_stream_chunk(ctx, AssistantMessageChunk(reasoning_content="abc"))

    message = str(exc_info.value)
    assert "LLM repeated stream output detected" in message
    assert "field=reasoning_content" in message
    assert f"repeat_count={_DEFAULT_ABC_REPEAT_COUNT}" in message


@pytest.mark.asyncio
async def test_single_char_repetition_raises_model_error():
    rail = ModelAnomalyDetectionRail()
    ctx = _make_ctx()

    await rail.before_model_call(ctx)
    await rail.inspect_stream_chunk(ctx, AssistantMessageChunk(content="a" * 99))

    with pytest.raises(BaseError) as exc_info:
        await rail.inspect_stream_chunk(ctx, AssistantMessageChunk(content="a"))

    message = str(exc_info.value)
    assert "LLM repeated stream output detected" in message
    assert "field=content" in message
    assert "repeat_count=100" in message


@pytest.mark.asyncio
async def test_repeat_exception_retries_twice_then_resets():
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.5, 1.0, 2.0])
    ctx = _make_ctx()
    ctx.request_retry = MagicMock()
    ctx.exception = build_error(
        StatusCode.MODEL_CALL_FAILED,
        error_msg="LLM repeated stream output detected: field=content",
    )

    await rail.on_model_exception(ctx)
    await rail.on_model_exception(ctx)
    await rail.on_model_exception(ctx)

    assert ctx.request_retry.call_count == 2
    # Backoff applied before each retry (exact schedule).
    assert [call.kwargs["delay_seconds"] for call in ctx.request_retry.call_args_list] == [0.5, 1.0]
    assert rail.repeat_retry_count == 0


@pytest.mark.asyncio
async def test_backoff_delay_follows_schedule_and_clamps():
    rail = ModelAnomalyDetectionRail(backoff_seconds=[0.5, 1.0, 2.0])
    assert rail.backoff_delay(0) == 0.5
    assert rail.backoff_delay(1) == 1.0
    assert rail.backoff_delay(2) == 2.0
    # Indices beyond the schedule clamp to the last entry.
    assert rail.backoff_delay(5) == 2.0

    # Default schedule matches the documented (0.5, 1.0, 2.0).
    default_rail = ModelAnomalyDetectionRail()
    assert [default_rail.backoff_delay(i) for i in range(3)] == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_stream_timeout_exception_retries_twice_then_resets():
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.5, 1.0, 2.0])
    ctx = _make_ctx()
    ctx.request_retry = MagicMock()
    ctx.exception = build_error(
        StatusCode.MODEL_CALL_FAILED,
        error_msg="LLM stream timeout: stream frame timeout: stage=idle_chunk",
    )

    await rail.on_model_exception(ctx)
    await rail.on_model_exception(ctx)
    await rail.on_model_exception(ctx)

    assert ctx.request_retry.call_count == 2
    assert [call.kwargs["delay_seconds"] for call in ctx.request_retry.call_args_list] == [0.5, 1.0]
    assert rail.stream_timeout_retry_count == 0


@pytest.mark.asyncio
async def test_before_invoke_resets_retry_counters():
    rail = ModelAnomalyDetectionRail()
    rail.repeat_retry_count = 1
    rail.stream_timeout_retry_count = 1
    rail.transient_retry_count = 2

    await rail.before_invoke(_make_ctx())

    assert rail.repeat_retry_count == 0
    assert rail.stream_timeout_retry_count == 0
    assert rail.transient_retry_count == 0


@pytest.mark.asyncio
async def test_rail_retries_repeated_stream_output_in_agent_streaming_path():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.0])
    await agent.register_rail(rail)
    model = _RetryStreamModel("loop")
    agent.set_llm(model)

    result = await agent.invoke({"query": "loop once"}, _streaming=True)

    assert result["result_type"] == "answer"
    assert result["output"] == "recovered"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_rail_retries_stream_timeout_in_agent_streaming_path():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.0])
    await agent.register_rail(rail)
    model = _RetryStreamModel("timeout")
    agent.set_llm(model)

    result = await agent.invoke({"query": "timeout once"}, _streaming=True)

    assert result["result_type"] == "answer"
    assert result["output"] == "recovered"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_rail_propagates_repeated_stream_output_after_retry_exhaustion():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.0])
    await agent.register_rail(rail)
    model = _RetryStreamModel("loop_exhausted")
    agent.set_llm(model)

    with pytest.raises(BaseError) as exc_info:
        await agent.invoke({"query": "loop always"}, _streaming=True)

    assert "LLM repeated stream output detected" in str(exc_info.value)
    assert model.call_count == 3


@pytest.mark.asyncio
async def test_rail_propagates_stream_timeout_after_retry_exhaustion():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(max_retries=2, backoff_seconds=[0.0])
    await agent.register_rail(rail)
    model = _RetryStreamModel("timeout_exhausted")
    agent.set_llm(model)

    with pytest.raises(BaseError) as exc_info:
        await agent.invoke({"query": "timeout always"}, _streaming=True)

    assert "LLM stream timeout" in str(exc_info.value)
    assert model.call_count == 3


def _retry_ctx(extra=None):
    ctx = _make_ctx()
    ctx.request_retry = MagicMock()
    if extra:
        ctx.extra.update(extra)
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 409, 429, 503])
async def test_transient_status_requests_retry(status_code):
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(status_code, "provider failed")

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=2.0)
    assert rail.transient_retry_count == 1
    assert rail.repeat_retry_count == 0


@pytest.mark.asyncio
async def test_transient_retries_use_exponential_base_then_reset():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(429, "too many requests")

    for _ in range(4):
        await rail.on_model_exception(ctx)

    assert [call.kwargs["delay_seconds"] for call in ctx.request_retry.call_args_list] == [2.0, 4.0, 8.0]
    assert rail.transient_retry_count == 0
    assert rail.repeat_retry_count == 0


@pytest.mark.asyncio
async def test_transient_retry_count_and_base_are_configurable():
    rail = ModelAnomalyDetectionRail(transient_max_retries=1, transient_base_delay_seconds=3.0)
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(429, "too many requests")

    await rail.on_model_exception(ctx)
    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=3.0)
    assert rail.transient_retry_count == 0


@pytest.mark.asyncio
async def test_provider_code_in_details_is_retryable():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = build_error(
        StatusCode.MODEL_CALL_FAILED,
        error_msg="provider response error: code=429",
        details={"provider_code": 429},
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=2.0)


@pytest.mark.asyncio
async def test_connection_error_without_status_is_retryable():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = build_error(
        StatusCode.MODEL_CALL_FAILED,
        error_msg="connection reset",
        cause=ConnectionError("connection reset"),
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "bad request"),
        (401, "unauthorized"),
        (429, "insufficient_quota"),
        (400, "maximum context length exceeded"),
    ],
)
async def test_non_retryable_provider_failures_do_not_retry(status_code, message):
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(status_code, message)

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_not_called()
    assert rail.transient_retry_count == 0


@pytest.mark.asyncio
async def test_x_should_retry_false_blocks_429():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(
        429,
        "too many requests",
        headers={"x-should-retry": "false"},
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_not_called()


@pytest.mark.asyncio
async def test_retry_after_header_overrides_exponential_delay():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(
        429,
        "too many requests",
        headers={"Retry-After": "5"},
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=5.0)


@pytest.mark.asyncio
async def test_retry_after_ms_is_converted_to_seconds():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(
        429,
        "too many requests",
        headers={"retry-after-ms": "1500"},
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_called_once_with(delay_seconds=1.5)


@pytest.mark.asyncio
async def test_retry_after_above_cap_does_not_retry():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx()
    ctx.exception = _wrapped_status_error(
        429,
        "too many requests",
        headers={"Retry-After": "90"},
    )

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_not_called()
    assert rail.transient_retry_count == 0


@pytest.mark.asyncio
async def test_visible_output_blocks_transient_retry():
    rail = ModelAnomalyDetectionRail()
    ctx = _retry_ctx(extra={MODEL_VISIBLE_OUTPUT_EMITTED_KEY: True})
    ctx.exception = _wrapped_status_error(429, "too many requests")

    await rail.on_model_exception(ctx)

    ctx.request_retry.assert_not_called()


@pytest.mark.asyncio
async def test_rail_retries_rate_limit_before_visible_output():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(transient_base_delay_seconds=0.0)
    await agent.register_rail(rail)
    model = _RetryStreamModel("rate_limit")
    agent.set_llm(model)

    result = await agent.invoke({"query": "rate limit once"}, _streaming=True)

    assert result["output"] == "recovered"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_rail_does_not_retry_rate_limit_after_visible_output():
    agent = _make_agent()
    rail = ModelAnomalyDetectionRail(transient_base_delay_seconds=0.0)
    await agent.register_rail(rail)
    model = _RetryStreamModel("rate_limit_after_text")
    agent.set_llm(model)

    with pytest.raises(BaseError):
        await agent.invoke({"query": "rate limit after text"}, _streaming=True)

    assert model.call_count == 1


# Responses streams that break raise ResponsesStreamInterruptedError. Until the agent gets its own
# broken-stream rule, this rail treats it like other transient failures: retried before visible
# output (tool-call arguments are not visible output), not retried after text was shown.
_RESPONSES_TEXT = 'event: response.output_text.delta\ndata: {"delta": "%s"}\n\n'
_RESPONSES_COMPLETED = 'event: response.completed\ndata: {"response": {"status": "completed"}}\n\n'
_RESPONSES_TOOL_ARGUMENTS = (
    'event: response.output_item.added\n'
    'data: {"item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "write", "arguments": ""}}\n\n'
    'event: response.function_call_arguments.delta\ndata: {"item_id": "fc_1", "delta": "{\\"path\\":"}\n\n'
)


class _DroppedBody(httpx.AsyncByteStream):
    def __init__(self, body: str):
        self._body = body.encode()

    async def __aiter__(self):
        yield self._body
        raise httpx.ReadError("connection reset by peer")


def _responses_model(monkeypatch, first_body: str, *, drop: bool) -> tuple[Model, list]:
    """A Responses-mode model whose first reply is ``first_body``; later replies answer "recovered"."""
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {"content-type": "text/event-stream"}
        if len(requests) == 1 and drop:
            return httpx.Response(200, stream=_DroppedBody(first_body), headers=headers)
        body = first_body if len(requests) == 1 else _RESPONSES_TEXT % "recovered" + _RESPONSES_COMPLETED
        return httpx.Response(200, content=body.encode(), headers=headers)

    monkeypatch.setattr(
        OpenAIAccountResponsesTransport,
        "_make_client",
        lambda _: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_mode=LLMApiMode.Responses,
            api_key="sk-test",
            api_base="https://mock.local/v1",
            verify_ssl=False,
        ),
        model_config=ModelRequestConfig(model="mock-model"),
    )
    return model, requests


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_body", "drop"),
    [("", True), (_RESPONSES_TOOL_ARGUMENTS, True), (_RESPONSES_TOOL_ARGUMENTS, False)],
    ids=["dropped_before_output", "dropped_in_tool_arguments", "no_closing_event_in_tool_arguments"],
)
async def test_rail_retries_broken_responses_stream_before_visible_output(monkeypatch, first_body, drop):
    agent = _make_agent()
    await agent.register_rail(ModelAnomalyDetectionRail(transient_base_delay_seconds=0.0))
    model, requests = _responses_model(monkeypatch, first_body, drop=drop)
    agent.set_llm(model)

    result = await agent.invoke({"query": "broken stream"}, _streaming=True)

    assert result["output"] == "recovered"
    assert len(requests) == 2
    # The half-written tool call was neither run nor kept in the conversation.
    retried_input = json.loads(requests[1].content)["input"]
    assert [item for item in retried_input if item.get("type") == "function_call"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_body", "drop"),
    [
        (_RESPONSES_TEXT % "partial", True),
        (_RESPONSES_TEXT % "partial", False),
        ('event: response.cancelled\ndata: {"response": {"status": "cancelled"}}\n\n', False),
    ],
    ids=["dropped_after_text", "no_closing_event_after_text", "cancelled"],
)
async def test_rail_does_not_retry_broken_responses_stream_after_text_or_cancel(monkeypatch, first_body, drop):
    agent = _make_agent()
    await agent.register_rail(ModelAnomalyDetectionRail(transient_base_delay_seconds=0.0))
    model, requests = _responses_model(monkeypatch, first_body, drop=drop)
    agent.set_llm(model)

    with pytest.raises(BaseError):
        await agent.invoke({"query": "broken stream"}, _streaming=True)

    assert len(requests) == 1
