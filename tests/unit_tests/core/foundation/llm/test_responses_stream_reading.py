# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from contextlib import aclosing
from unittest.mock import patch

import httpx
import pytest

from openjiuwen.core.common.exception.errors import BaseError
from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient
from openjiuwen.core.foundation.llm.utils.responses_transport import OpenAIAccountResponsesTransport
from openjiuwen.core.foundation.llm.utils.responses_utils import (
    OpenAIAccountResponsesError,
    ResponsesStreamInterruptedError,
    build_request_body,
    parse_response,
)

_MODEL = "gpt-5.6-luna"
_COMPLETED = {"type": "response.completed", "response": {"status": "completed"}}


def _call(item_id: str, call_id: str, name: str, arguments: str = "") -> dict:
    return {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": arguments}


def _message(*parts: dict, item_id: str = "msg_1") -> dict:
    return {"type": "message", "id": item_id, "role": "assistant", "content": list(parts)}


def _output_text(text: str) -> dict:
    return {"type": "output_text", "text": text}


def _reasoning(*, summary: tuple = (), content: tuple = (), item_id: str = "rs_1") -> dict:
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": [{"type": "summary_text", "text": text} for text in summary],
        "content": [{"type": "reasoning_text", "text": text} for text in content],
    }


def _added(item: dict) -> dict:
    return {"type": "response.output_item.added", "item": item}


def _done(item: dict, output_index: int = 0) -> dict:
    return {"type": "response.output_item.done", "output_index": output_index, "item": item}


def _arguments(item_id: str, delta: str) -> dict:
    return {"type": "response.function_call_arguments.delta", "item_id": item_id, "delta": delta}


def _text(delta: str, item_id: str = "msg_1") -> dict:
    return {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "delta": delta}


def _summary(delta: str, summary_index: int, item_id: str = "rs_1") -> dict:
    return {
        "type": "response.reasoning_summary_text.delta",
        "item_id": item_id,
        "output_index": 0,
        "summary_index": summary_index,
        "delta": delta,
    }


def _sse(events: list) -> bytes:
    """SSE body for ``events``; a string is sent as it is (for example a garbled block)."""
    blocks = []
    for event in events:
        if isinstance(event, str):
            blocks.append(event)
            continue
        data = {key: value for key, value in event.items() if key != "type"}
        blocks.append(f"event: {event['type']}\ndata: {json.dumps(data)}\n\n")
    return "".join(blocks).encode()


class _DroppedBody(httpx.AsyncByteStream):
    """A response body that sends its bytes and then loses the connection."""

    def __init__(self, body: bytes):
        self._body = body

    async def __aiter__(self):
        yield self._body
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _transport(events: list, *, drop: bool = False) -> OpenAIAccountResponsesTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        if drop:
            return httpx.Response(200, stream=_DroppedBody(_sse(events)), headers=headers)
        return httpx.Response(200, content=_sse(events), headers=headers)

    return OpenAIAccountResponsesTransport(transport=httpx.MockTransport(handler))


def _body() -> dict:
    return build_request_body(model=_MODEL, messages="Hello.")


async def _merged(chunks):
    merged = None
    async for chunk in chunks:
        merged = merged + chunk if merged else chunk
    return merged


async def _stream(events: list, **kwargs):
    """Merge the streamed chunks, as ReActAgent does."""
    return await _merged(_transport(events, **kwargs).stream_response(body=_body(), access_token="token"))


@pytest.mark.asyncio
async def test_parallel_tool_calls_written_at_once_stay_apart():
    events = [
        _added(_call("fc_1", "call_1", "read")),
        _added(_call("fc_2", "call_2", "write")),
        _arguments("fc_1", '{"p":'),
        _arguments("fc_2", '{"q":2}'),
        _arguments("fc_1", "1}"),
        _done(_call("fc_1", "call_1", "read", '{"p":1}')),
        _done(_call("fc_2", "call_2", "write", '{"q":2}'), output_index=1),
        _COMPLETED,
    ]
    chunks = [chunk async for chunk in _transport(events).stream_response(body=_body(), access_token="token")]

    merged = chunks[0]
    for chunk in chunks[1:]:
        merged = merged + chunk

    # Each fragment carries the position of its call, not 0 for all.
    assert [chunk.tool_calls[0].index for chunk in chunks if chunk.tool_calls] == [0, 1, 0, 1, 0, 0, 1]
    assert [(call.id, call.name, call.arguments) for call in merged.tool_calls] == [
        ("call_1", "read", '{"p":1}'),
        ("call_2", "write", '{"q":2}'),
    ]
    assert merged.finish_reason == "tool_calls"


# (case id, events before response.completed, expected content, expected reasoning)
_ITEM_END_CASES = [
    ("text_only_at_item_end", [_done(_message(_output_text("Hello")))], "Hello", ""),
    ("text_partly_streamed", [_text("Hel"), _done(_message(_output_text("Hello")))], "Hello", ""),
    ("text_fully_streamed", [_text("Hel"), _text("lo"), _done(_message(_output_text("Hello")))], "Hello", ""),
    ("text_in_two_parts", [_done(_message(_output_text("Hel"), _output_text("lo")))], "Hello", ""),
    ("done_without_content", [_text("Hello"), _done(_message())], "Hello", ""),
    # Streamed text is already concatenated downstream and cannot be taken back.
    ("done_text_differs", [_text("Hello"), _done(_message(_output_text("Goodbye")))], "Hello", ""),
    ("text_deltas_name_only_output_index", [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Hel"},
        _done(_message(_output_text("Hello"))),
    ], "Hello", ""),
    ("text_deltas_name_no_item", [
        {"type": "response.output_text.delta", "delta": "Hel"},
        _done(_message(_output_text("Hello"))),
    ], "Hello", ""),
    ("refusal_only_at_item_end", [
        _done(_message({"type": "refusal", "refusal": "I can't help."})),
    ], "I can't help.", ""),
    ("refusal_streamed", [
        {"type": "response.refusal.delta", "item_id": "msg_1", "output_index": 0, "delta": "No."},
        _done(_message({"type": "refusal", "refusal": "No."})),
    ], "No.", ""),
    ("summary_only_at_item_end", [_done(_reasoning(summary=("Plan.", "Check.")))], "", "Plan.\nCheck."),
    # DeepSeek and GLM send readable reasoning as content[] reasoning_text parts.
    ("reasoning_text_only_at_item_end", [_done(_reasoning(content=("Think.",)))], "", "Think."),
    ("summary_partly_streamed", [
        _summary("Pl", 0), _done(_reasoning(summary=("Plan.", "Check."))),
    ], "", "Plan.\nCheck."),
    ("summary_fully_streamed", [
        _summary("Plan.", 0), _summary("Check.", 1), _done(_reasoning(summary=("Plan.", "Check.")))
    ], "", "Plan.\nCheck."),
]
_ITEM_END_IDS = [case_id for case_id, _, _, _ in _ITEM_END_CASES]


@pytest.mark.asyncio
@pytest.mark.parametrize(("case_id", "events", "content", "reasoning"), _ITEM_END_CASES, ids=_ITEM_END_IDS)
async def test_text_and_reasoning_sent_at_item_end_are_kept_once(case_id, events, content, reasoning):
    streamed = await _stream(events + [_COMPLETED])
    invoked = await _transport(events + [_COMPLETED]).create_response(body=_body(), access_token="token")

    assert (streamed.content, streamed.reasoning_content) == (content, reasoning)
    assert (invoked.content, invoked.reasoning_content or "") == (content, reasoning)


@pytest.mark.asyncio
async def test_tool_call_arguments_sent_only_at_item_end_are_kept():
    """Qwen names the call when the item starts and sends the arguments only when it ends."""
    events = [
        _added(_call("fc_1", "call_1", "search")),
        _done(_call("fc_1", "call_1", "search", '{"q":"x"}')),
        _COMPLETED,
    ]

    merged = await _stream(events)

    assert [(call.name, call.arguments) for call in merged.tool_calls] == [("search", '{"q":"x"}')]


@pytest.mark.asyncio
async def test_streamed_reasoning_parts_are_joined_like_parse_response():
    first = _reasoning(summary=("First idea.", "Second idea."))
    second = _reasoning(summary=("Another item.",), item_id="rs_2")
    events = [
        _summary("First ", 0),
        _summary("idea.", 0),
        _summary("Second idea.", 1),
        _done(first),
        _summary("Another item.", 0, item_id="rs_2"),
        _done(second, output_index=1),
        _COMPLETED,
    ]

    merged = await _stream(events)

    assert merged.reasoning_content == "First idea.\nSecond idea.\nAnother item."
    assert merged.reasoning_content == parse_response({"output": [first, second]}).reasoning_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block", "event_name"),
    [
        ('event: response.output_text.delta\ndata: {"delta": "' + "x" * 5000 + "\n\n", "'response.output_text.delta'"),
        ("data: {not json\n\n", "without an event name"),
    ],
    ids=["long_named_event", "unnamed_event"],
)
async def test_garbled_event_raises_short_interrupted_error(block, event_name):
    with pytest.raises(ResponsesStreamInterruptedError) as exc_info:
        await _stream([_text("Hi"), block, _COMPLETED])

    error = exc_info.value
    assert error.reason == ResponsesStreamInterruptedError.GARBLED_EVENT
    assert event_name in str(error)
    assert len(str(error)) <= 500
    assert error.partial.content == "Hi"


@pytest.mark.asyncio
async def test_stream_without_closing_event_raises_with_partial_output():
    events = [_text("Half an ans"), _added(_call("fc_1", "call_1", "write")), _arguments("fc_1", '{"path":')]

    with pytest.raises(ResponsesStreamInterruptedError) as exc_info:
        await _stream(events)

    error = exc_info.value
    assert error.reason == ResponsesStreamInterruptedError.MISSING_CLOSING_EVENT
    assert error.partial.content == "Half an ans"
    assert [(call.name, call.arguments) for call in error.partial.tool_calls] == [("write", '{"path":')]


@pytest.mark.asyncio
async def test_empty_stream_raises_without_partial_output():
    with pytest.raises(ResponsesStreamInterruptedError) as exc_info:
        await _transport([]).create_response(body=_body(), access_token="token")

    assert exc_info.value.reason == ResponsesStreamInterruptedError.MISSING_CLOSING_EVENT
    assert exc_info.value.partial is None


@pytest.mark.asyncio
async def test_dropped_connection_raises_connection_lost():
    with pytest.raises(ResponsesStreamInterruptedError) as exc_info:
        await _stream([_text("Hel")], drop=True)

    error = exc_info.value
    assert error.reason == ResponsesStreamInterruptedError.CONNECTION_LOST
    assert "RemoteProtocolError" in str(error)
    assert isinstance(error.__cause__, httpx.RemoteProtocolError)
    assert error.partial.content == "Hel"


@pytest.mark.asyncio
async def test_cancelled_stream_is_an_error_that_is_not_retried():
    events = [_text("Hi"), {"type": "response.cancelled", "response": {"status": "cancelled"}}]

    with pytest.raises(OpenAIAccountResponsesError) as exc_info:
        await _stream(events)

    assert not isinstance(exc_info.value, ResponsesStreamInterruptedError)
    assert exc_info.value.details == {"should_retry": False}


@pytest.mark.asyncio
async def test_events_after_closing_event_are_ignored():
    events = [_text("Done."), _COMPLETED, _text(" Extra"), "data: {garbled\n\n", {"type": "error", "message": "late"}]

    merged = await _stream(events)

    assert merged.content == "Done."
    assert merged.finish_reason == "stop"


@pytest.mark.asyncio
async def test_connection_drop_after_closing_event_is_ignored():
    merged = await _stream([_text("Done."), _COMPLETED], drop=True)

    assert merged.content == "Done."
    assert merged.finish_reason == "stop"


@pytest.mark.asyncio
async def test_reader_that_stops_early_gets_no_error():
    stream = _transport([_text("Hi"), _text(" there")]).stream_response(body=_body(), access_token="token")

    async with aclosing(stream):
        first = await anext(stream)

    assert first.content == "Hi"


def _responses_client() -> OpenAIModelClient:
    return OpenAIModelClient(
        ModelRequestConfig(model=_MODEL),
        ModelClientConfig(
            client_provider="OpenAI",
            api_key="sk-test-key",
            api_base="https://api.openai.com/v1",
            api_mode="responses",
            verify_ssl=False,
        ),
    )


@pytest.mark.asyncio
async def test_model_client_reports_broken_stream_as_model_error():
    client = _responses_client()
    with patch.object(client, "_make_responses_transport", side_effect=lambda **_: _transport([_text("Hi")])):
        with pytest.raises(BaseError) as invoke_error:
            await client.invoke("Hello.")
        with pytest.raises(BaseError) as stream_error:
            await _merged(client.stream("Hello."))

    for error in (invoke_error.value, stream_error.value):
        assert isinstance(error.__cause__, ResponsesStreamInterruptedError)
        assert "ended before its closing event" in str(error)
