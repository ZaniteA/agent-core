# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fake model servers for unit tests.

``FakeModelServer`` answers model requests in memory and records each one, so
a test runs the real model client (OpenAI SDK, Responses transport, headers,
retry rules, stream reading) with no network. It answers in the format the
request address asks for:

- ``.../chat/completions``: OpenAI Chat Completions, as one JSON reply or as
  SSE chunks ending with ``data: [DONE]``. A streamed tool call sends its id
  and name first, then its arguments in pieces, as real servers do.
- ``.../responses``: OpenAI Responses SSE events in the documented order:
  ``response.created`` and ``response.in_progress``; for each output item,
  ``output_item.added``, its part, delta and ``done`` events, and
  ``output_item.done``; then ``response.completed``. The ChatGPT-login
  backend uses the same format.
- any other address: HTTP 404. The request is still recorded.

The streams stay small: each text or reasoning part arrives in one delta, and
there are no keep-alive comments, unknown events, ``response.failed`` or
``response.incomplete`` events. Add them here when a test needs them.

Script the replies in order, then read ``server.requests``::

    from tests.unit_tests.fixtures.fake_llm_servers import fake_model_server  # noqa: F401

    @pytest.mark.asyncio
    async def test_hello(fake_model_server):
        fake_model_server.add_reply(text="Hello!")
        model = Model(ModelClientConfig(...), ModelRequestConfig(model="gpt-test"))
        assert (await model.invoke("hi")).content == "Hello!"
        assert fake_model_server.requests[0].path == "/v1/chat/completions"

When the script runs out, the server answers ``"ok"``. A streamed reply can
carry a ``StreamFault`` that breaks it on purpose.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Optional, Union

import httpx
import pytest_asyncio

from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient

REDACTED = "<redacted>"

# A header whose name contains one of these words holds a secret. A saved copy
# keeps the header name, so its presence still shows, but never the value.
_SECRET_HEADER_WORDS = ("auth", "cookie", "key", "secret", "token")
# Headers that httpx or the OpenAI SDK set for themselves. They change with the
# library version or the machine, not with anything agent-core decides.
_LIBRARY_HEADERS = frozenset({"accept-encoding", "connection", "content-length", "host", "user-agent"})
_LIBRARY_HEADER_PREFIXES = ("x-stainless-",)
# Environment variables the OpenAI SDK reads into every request; for example
# ``OPENAI_ORG_ID`` becomes an ``OpenAI-Organization`` header.
_OPENAI_SDK_ENVIRONMENT = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID")

_CHAT_USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
_RESPONSES_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
_GARBLED_EVENT = b'data: {"garbled\n\n'


@dataclass(frozen=True)
class RecordedRequest:
    """One request as the fake server received it."""

    method: str
    url: str
    headers: dict[str, str]
    body: Any

    @property
    def path(self) -> str:
        return httpx.URL(self.url).path

    def saved_copy(self) -> dict[str, Any]:
        """Return the address, JSON body and non-secret headers of this request.

        Secret header values become ``<redacted>``. Headers that httpx or the
        OpenAI SDK set for themselves are left out.
        """
        headers = {}
        for name, value in sorted(self.headers.items()):
            if name in _LIBRARY_HEADERS or name.startswith(_LIBRARY_HEADER_PREFIXES):
                continue
            headers[name] = REDACTED if any(word in name for word in _SECRET_HEADER_WORDS) else value
        return {"method": self.method, "url": self.url, "headers": headers, "body": self.body}


@dataclass(frozen=True)
class FakeToolCall:
    """A tool call made by the fake model."""

    name: str
    arguments: Union[dict[str, Any], str] = field(default_factory=dict)
    call_id: str = "call_1"

    def arguments_text(self) -> str:
        return self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)


@dataclass(frozen=True)
class StreamFault:
    """One way to break a streamed reply on purpose.

    ``after`` counts the events sent before the fault: Chat chunks or
    Responses events. The closing event (``data: [DONE]`` or
    ``response.completed``) is the last event. Faults only apply to streamed
    replies; a plain JSON Chat reply ignores them.
    """

    kind: Literal["cut", "no_closing_event", "garbled_line", "stall"]
    after: int = 0
    stall_seconds: float = 0.0

    @classmethod
    def cut(cls, after: int) -> "StreamFault":
        """Drop the connection after ``after`` events, as a lost network would."""
        return cls("cut", after)

    @classmethod
    def no_closing_event(cls) -> "StreamFault":
        """Send every event except the closing one, then end the stream normally."""
        return cls("no_closing_event")

    @classmethod
    def garbled_line(cls, after: int) -> "StreamFault":
        """Insert one event whose data is not valid JSON after ``after`` events."""
        return cls("garbled_line", after)

    @classmethod
    def stall(cls, after: int, seconds: float) -> "StreamFault":
        """Send nothing for ``seconds`` after ``after`` events, then carry on."""
        return cls("stall", after, seconds)


async def faulty_stream(events: list[bytes], fault: Optional[StreamFault]) -> AsyncIterator[bytes]:
    """Send ``events`` one at a time, broken the way ``fault`` says.

    Raises:
        ValueError: When ``fault.after`` is outside ``0..len(events)``. Such a
            fault would never fire, and the test would get a normal answer.
    """
    if fault is not None and fault.kind == "no_closing_event":
        events = events[:-1]
    elif fault is not None and not 0 <= fault.after <= len(events):
        raise ValueError(
            f"StreamFault.{fault.kind}(after={fault.after}) never fires on a stream of {len(events)} events"
        )
    for index in range(len(events) + 1):
        if fault is not None and index == fault.after:
            if fault.kind == "cut":
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            if fault.kind == "garbled_line":
                yield _GARBLED_EVENT
            elif fault.kind == "stall":
                await asyncio.sleep(fault.stall_seconds)
        if index < len(events):
            yield events[index]


@dataclass(frozen=True)
class _Reply:
    status: int = 200
    error_message: str = ""
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[FakeToolCall, ...] = ()
    fault: Optional[StreamFault] = None


class FakeModelServer:
    """In-memory model server that records every request and answers from a script."""

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self._replies: list[_Reply] = []

    def add_reply(
        self,
        *,
        text: str = "",
        reasoning: str = "",
        tool_calls: tuple[FakeToolCall, ...] | list[FakeToolCall] = (),
        fault: Optional[StreamFault] = None,
    ) -> None:
        """Queue the next answer.

        ``reasoning`` goes out as Chat ``reasoning_content`` (the DeepSeek
        style) or as a Responses reasoning summary.
        """
        self._replies.append(_Reply(text=text, reasoning=reasoning, tool_calls=tuple(tool_calls), fault=fault))

    def add_error(self, status: int, message: str = "fake provider error") -> None:
        """Queue an HTTP error answer."""
        self._replies.append(_Reply(status=status, error_message=message))

    def transport(self) -> httpx.MockTransport:
        """An httpx transport that sends requests to this server, for explicit transport hooks."""
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = _json_or_none(request.content)
        self.requests.append(
            RecordedRequest(method=request.method, url=str(request.url), headers=dict(request.headers), body=body)
        )
        path = request.url.path
        if not path.endswith(("/chat/completions", "/responses")):
            return httpx.Response(404, json={"error": {"message": f"fake server has no route for {path}"}})

        reply = self._replies.pop(0) if self._replies else _Reply(text="ok")
        if reply.status != 200:
            error = {"message": reply.error_message, "type": "invalid_request_error"}
            return httpx.Response(reply.status, json={"error": error})

        body = body if isinstance(body, dict) else {}
        model = str(body.get("model") or "")
        number = len(self.requests)
        if path.endswith("/responses"):
            events = _responses_events(reply, model, number)
        elif body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            events = _chat_chunks(reply, model, number, include_usage=include_usage)
        else:
            return httpx.Response(200, json=_chat_completion(reply, model, number))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=faulty_stream(events, reply.fault),
        )


def install_fake_server(monkeypatch, server: FakeModelServer) -> FakeModelServer:
    """Send every request made through ``httpx.AsyncClient`` in this test to ``server``.

    Both OpenAI clients open their connection that way: the OpenAI SDK for
    Chat, and ``OpenAIAccountResponsesTransport`` for Responses and ChatGPT
    login. So this one patch catches their requests, while the real address
    building, headers, retry rules and stream reading still run. It is the
    only private patch point: if a client stops opening ``httpx.AsyncClient``,
    fix it here. Proxies are switched off so no request can leave the machine,
    and the OpenAI SDK's environment variables are cleared so a request does
    not depend on the machine it runs on.
    """
    for name in _OPENAI_SDK_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    transport = server.transport()
    original_init = httpx.AsyncClient.__init__

    def init_with_fake_server(client, *args, **kwargs):
        kwargs["transport"] = transport
        kwargs["trust_env"] = False
        kwargs.pop("proxy", None)
        kwargs.pop("mounts", None)
        original_init(client, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init_with_fake_server)
    return server


async def reset_model_client_caches() -> None:
    """Forget shared model connections and image-support verdicts left by other tests.

    A cached OpenAI SDK client keeps the transport it was built with, so a
    client built before ``install_fake_server`` would miss the fake server.
    """
    from openjiuwen.harness.image_modality_probe import reset_image_support_cache

    await OpenAIModelClient.aclose()
    reset_image_support_cache()


@pytest_asyncio.fixture
async def fake_model_server(monkeypatch) -> AsyncIterator[FakeModelServer]:
    """A ``FakeModelServer`` that receives every model request of the test."""
    await reset_model_client_caches()
    server = install_fake_server(monkeypatch, FakeModelServer())
    yield server
    await reset_model_client_caches()


def _json_or_none(content: bytes) -> Any:
    if not content:
        return None
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _finish_reason(reply: _Reply) -> str:
    return "tool_calls" if reply.tool_calls else "stop"


def _sse(data: Any, *, event: str = "") -> bytes:
    lines = [f"event: {event}"] if event else []
    lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
    return ("\n".join(lines) + "\n\n").encode()


def _argument_pieces(arguments: str) -> list[str]:
    """Split tool-call arguments in two, as servers stream them in pieces."""
    middle = len(arguments) // 2
    return [piece for piece in (arguments[:middle], arguments[middle:]) if piece]


def _chat_completion(reply: _Reply, model: str, number: int) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    if reply.reasoning:
        message["reasoning_content"] = reply.reasoning
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_text()},
            }
            for call in reply.tool_calls
        ]
    return {
        "id": f"chatcmpl-fake-{number}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": _finish_reason(reply)}],
        "usage": _CHAT_USAGE,
    }


def _chat_chunks(reply: _Reply, model: str, number: int, *, include_usage: bool) -> list[bytes]:
    def chunk(delta: dict[str, Any], finish_reason: Optional[str] = None) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-fake-{number}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    chunks = [chunk({"role": "assistant", "content": ""})]
    if reply.reasoning:
        chunks.append(chunk({"reasoning_content": reply.reasoning}))
    if reply.text:
        chunks.append(chunk({"content": reply.text}))
    for index, call in enumerate(reply.tool_calls):
        opening = {"index": index, "id": call.call_id, "type": "function",
                   "function": {"name": call.name, "arguments": ""}}
        chunks.append(chunk({"tool_calls": [opening]}))
        for piece in _argument_pieces(call.arguments_text()):
            chunks.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]}))
    chunks.append(chunk({}, _finish_reason(reply)))
    if include_usage:
        chunks.append({**chunk({}), "choices": [], "usage": _CHAT_USAGE})
    return [_sse(item) for item in chunks] + [_sse("[DONE]")]


def _responses_events(reply: _Reply, model: str, number: int) -> list[bytes]:
    response = {"id": f"resp_fake_{number}", "object": "response", "model": model}
    output: list[dict[str, Any]] = []
    in_progress = {**response, "status": "in_progress", "output": []}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": in_progress},
        {"type": "response.in_progress", "response": in_progress},
    ]

    def add_item(added: dict[str, Any], item_events: list[dict[str, Any]], done: dict[str, Any]) -> None:
        index = len(output)
        events.append({"type": "response.output_item.added", "output_index": index, "item": added})
        events.extend({"output_index": index, "item_id": added["id"], **event} for event in item_events)
        events.append({"type": "response.output_item.done", "output_index": index, "item": done})
        output.append(done)

    if reply.reasoning:
        part = {"type": "summary_text", "text": reply.reasoning}
        item = {"id": f"rs_fake_{number}", "type": "reasoning", "summary": []}
        add_item(
            item,
            [
                {"type": "response.reasoning_summary_part.added", "summary_index": 0, "part": {**part, "text": ""}},
                {"type": "response.reasoning_summary_text.delta", "summary_index": 0, "delta": reply.reasoning},
                {"type": "response.reasoning_summary_text.done", "summary_index": 0, "text": reply.reasoning},
                {"type": "response.reasoning_summary_part.done", "summary_index": 0, "part": part},
            ],
            {**item, "summary": [part]},
        )
    if reply.text:
        part = {"type": "output_text", "text": reply.text, "annotations": []}
        item = {"id": f"msg_fake_{number}", "type": "message", "role": "assistant", "status": "in_progress",
                "content": []}
        add_item(
            item,
            [
                {"type": "response.content_part.added", "content_index": 0, "part": {**part, "text": ""}},
                {"type": "response.output_text.delta", "content_index": 0, "delta": reply.text},
                {"type": "response.output_text.done", "content_index": 0, "text": reply.text},
                {"type": "response.content_part.done", "content_index": 0, "part": part},
            ],
            {**item, "status": "completed", "content": [part]},
        )
    for call_number, call in enumerate(reply.tool_calls, start=1):
        arguments = call.arguments_text()
        item = {"id": f"fc_fake_{number}_{call_number}", "type": "function_call", "status": "in_progress",
                "call_id": call.call_id, "name": call.name, "arguments": ""}
        deltas = [
            {"type": "response.function_call_arguments.delta", "delta": piece} for piece in _argument_pieces(arguments)
        ]
        add_item(
            item,
            deltas + [{"type": "response.function_call_arguments.done", "arguments": arguments}],
            {**item, "status": "completed", "arguments": arguments},
        )

    events.append(
        {
            "type": "response.completed",
            "response": {**response, "status": "completed", "output": output, "usage": _RESPONSES_USAGE},
        }
    )
    return [
        _sse({**event, "sequence_number": sequence}, event=event["type"])
        for sequence, event in enumerate(events)
    ]
