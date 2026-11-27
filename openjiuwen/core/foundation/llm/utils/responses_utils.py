# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Data mapping helpers for the OpenAI account Responses backend."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable, Optional, Union

import httpx
from pydantic import BaseModel

from openjiuwen.core.common.logging import llm_logger as logger
from openjiuwen.core.common.utils.header_utils import sanitize_headers
from openjiuwen.core.foundation.llm.schema.message import (
    AssistantMessage,
    BaseMessage,
    ToolMessage,
    UsageMetadata,
)
from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk
from openjiuwen.core.foundation.llm.schema.tool_call import ToolCall
from openjiuwen.core.foundation.llm.utils.provider_error import summarize_provider_error_text
from openjiuwen.core.foundation.llm.utils.request_sanitizer import clean_unicode
from openjiuwen.core.foundation.tool import ToolInfo

# Cap for provider error metadata copied into error text (e.g. OpenRouter's error.metadata.raw).
_MAX_ERROR_METADATA_CHARS = 500
# Caps for the error raised on a stream event that is not valid JSON.
_MAX_GARBLED_EVENT_MESSAGE_CHARS = 500
_MAX_GARBLED_EVENT_SAMPLE_CHARS = 200


class OpenAIAccountResponsesError(Exception):
    """Raised when Responses API data cannot be prepared or parsed.

    ``details`` is optional extra information for retry logic. For example,
    ``{"should_retry": False}`` tells the model-call retry rail not to retry.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: Any = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        self.details = details

    def __str__(self) -> str:
        return self.message


class ResponsesStreamInterruptedError(OpenAIAccountResponsesError):
    """Raised when a Responses stream breaks before its closing event.

    ``reason`` says how the stream broke:

    - :attr:`CONNECTION_LOST`: the connection dropped while the reply was streaming.
    - :attr:`MISSING_CLOSING_EVENT`: the stream ended without ``response.completed``
      or ``response.incomplete``.
    - :attr:`GARBLED_EVENT`: an event could not be read.

    ``partial`` holds what the stream delivered before it broke (text, reasoning
    and tool calls merged into one chunk), or None if nothing arrived. The reply
    is unfinished, so it must not be used as an answer.
    """

    CONNECTION_LOST = "connection_lost"
    MISSING_CLOSING_EVENT = "missing_closing_event"
    GARBLED_EVENT = "garbled_event"

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        partial: Optional[AssistantMessageChunk] = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.partial = partial


def build_request_body(
    *,
    model: str,
    messages: Union[str, list[BaseMessage], list[dict]],
    tools: Union[list[ToolInfo], list[dict], None] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    reasoning: Optional[dict[str, Any]] = None,
    include_reasoning_encrypted_content: bool = True,
    tool_choice: Union[str, dict, None] = "auto",
    parallel_tool_calls: bool = True,
    store: bool = False,
    extra_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a Responses API request body from OpenJiuwen message objects."""
    if not model:
        raise OpenAIAccountResponsesError("OpenAI account Responses request requires a model.")

    instructions, input_items = convert_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": store,
    }

    request_options = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "stop": stop,
        "reasoning": reasoning,
    }
    for key, value in request_options.items():
        if value is not None:
            body[key] = value

    response_tools = convert_tools(tools)
    if response_tools:
        body["tools"] = response_tools
        body["parallel_tool_calls"] = parallel_tool_calls
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    if include_reasoning_encrypted_content:
        body["include"] = ["reasoning.encrypted_content"]
    expanded = expand_nested_extra_body(extra_body)
    if expanded:
        body.update(expanded)
    if "tool_choice" in body:
        body["tool_choice"] = _responses_tool_choice(body["tool_choice"])
    _move_response_format_to_text(body)
    return clean_unicode(body)


def expand_nested_extra_body(extra_body: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Flatten a one-level nested ``extra_body`` into root request fields.

    Callers often pass ``model_config.model_dump()`` which may itself contain
    an ``extra_body`` key (e.g. ``{"extra_body": {"thinking": ...}}``). The
    Responses wire format needs those fields at the body root, not nested.
    Outer keys win over nested keys on conflict.
    """
    if not extra_body:
        return {}
    pending = dict(extra_body)
    nested = pending.pop("extra_body", None)
    expanded: dict[str, Any] = {}
    if isinstance(nested, dict):
        expanded.update(nested)
    expanded.update(pending)
    return expanded


def _move_response_format_to_text(body: dict[str, Any]) -> None:
    """Translate a chat-style ``response_format`` into the Responses ``text.format`` field.

    Other ``text`` settings are kept. A ``text.format`` the caller already set takes precedence.
    """
    if "response_format" not in body:
        return
    response_format = body.pop("response_format")
    if response_format is None:
        return

    text_format = _text_format_from_response_format(response_format)
    if text_format is None:
        body["response_format"] = response_format
        return

    text = dict(body["text"]) if isinstance(body.get("text"), dict) else {}
    if "format" in text:
        logger.warning("Responses request sets both response_format and text.format; keeping text.format")
    else:
        text["format"] = text_format
    body["text"] = text


def _text_format_from_response_format(response_format: Any) -> Optional[dict[str, Any]]:
    """Map a chat-style ``response_format`` dict to a Responses ``text.format`` dict.

    ``json_schema`` formats are flattened: ``name``, ``schema``, ``strict`` and ``description``
    move up next to ``type``. Other formats (``json_object``, ``text``) are used as they are.
    """
    if not isinstance(response_format, dict) or not response_format.get("type"):
        return None
    json_schema = response_format.get("json_schema")
    if response_format["type"] == "json_schema" and isinstance(json_schema, dict):
        return {"type": "json_schema", **json_schema}
    return dict(response_format)


def build_headers(
    *,
    access_token: str,
    session_id: Optional[str] = None,
    extra_headers: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Create HTTP headers for an account-backed Responses request."""
    if not access_token:
        raise OpenAIAccountResponsesError("OpenAI account Responses request requires an access token.")

    headers = sanitize_headers(extra_headers)
    headers["Authorization"] = f"Bearer {access_token}"
    headers["Accept"] = "application/json"
    headers["Content-Type"] = "application/json"
    if session_id:
        headers["session_id"] = session_id
        headers["x-client-request-id"] = session_id
    return headers


def convert_messages(messages: Union[str, list[BaseMessage], list[dict]]) -> tuple[str, list[dict[str, Any]]]:
    """Map OpenJiuwen messages to Responses ``instructions`` and ``input``."""
    instructions: list[str] = []
    inputs: list[dict[str, Any]] = []
    fallback_tool_call_index = 0

    for message in _message_dicts(messages):
        role = str(message.get("role") or "user")
        content_text = _content_as_text(message.get("content"))

        if role in {"system", "developer"}:
            if content_text:
                instructions.append(content_text)
            continue

        if role == "tool":
            tool_result = _tool_result_item(message, content_text)
            if tool_result is not None:
                inputs.append(tool_result)
            continue

        if role == "assistant":
            if content_text:
                inputs.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content_text}],
                    }
                )
            for tool_call in message.get("tool_calls") or []:
                inputs.append(_response_input_tool_call(tool_call, fallback_index=fallback_tool_call_index))
                fallback_tool_call_index += 1
            continue

        inputs.append(
            {
                "role": "user",
                "content": _user_content_parts(message.get("content"), content_text),
            }
        )

    return "\n\n".join(instructions), inputs


def convert_tools(tools: Union[list[ToolInfo], list[dict], None]) -> list[dict[str, Any]]:
    """Map tool declarations to the Responses ``tools`` shape."""
    if not tools:
        return []

    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted.append(_tool_definition(tool))
    return converted


def parse_response(payload: dict[str, Any], *, model_name: str = "") -> AssistantMessage:
    """Turn a complete Responses API payload into an ``AssistantMessage``."""
    content_parts, reasoning_parts, tool_calls = _read_output_items(payload.get("output"))
    if not content_parts and payload.get("output_text"):
        content_parts.append(str(payload.get("output_text") or ""))

    return AssistantMessage(
        content="".join(content_parts),
        tool_calls=tool_calls or None,
        usage_metadata=_usage_from_payload(payload.get("usage"), model_name=model_name),
        finish_reason=_finish_reason(payload, has_tool_calls=bool(tool_calls)),
        reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
        response_id=str(payload.get("id") or "") or None,
        response_model=str(payload.get("model") or model_name or "") or None,
        provider_metadata=_response_provider_metadata(payload),
    )


class ResponsesStreamParser:
    """Turn the Responses SSE events of one streamed response into chunks.

    Function-call arguments are streamed as they are generated
    (``response.output_item.added`` names the call, each
    ``response.function_call_arguments.delta`` extends it), so a long tool
    call shows progress instead of arriving all at once when its item
    completes; that progress is what time to first token and the stream idle
    timeout observe. ``response.output_item.done`` repeats the full item, and
    streamed chunks are concatenated downstream, so the parser remembers what
    it already emitted per item and the done event only contributes the part
    that was never streamed. This holds for tool-call arguments, answer text
    (including refusals) and reasoning, so a backend that skips the delta
    events (Qwen sends tool-call arguments only at the end of the item) still
    yields the complete item from the done event.

    Each tool-call fragment carries its item id (``ToolCall.response_item_id``),
    so calls written at the same time are merged by id, not by position.
    Reasoning parts (summary parts, ``reasoning_text`` parts, reasoning items)
    are separated by a newline, as :func:`parse_response` joins them.

    The stream ends with ``response.completed`` or ``response.incomplete``
    (:attr:`finished`); later events are ignored. ``response.failed``, ``error``
    and ``response.cancelled`` raise :class:`OpenAIAccountResponsesError`.

    One parser instance serves exactly one response stream.
    """

    def __init__(self, *, model_name: str = "") -> None:
        self._model_name = model_name
        # Response item id -> (tool call id, arguments emitted so far).
        self._streamed_calls: dict[str, tuple[str, str]] = {}
        # Response item id -> position of the call among this response's function calls.
        self._call_indexes: dict[str, int] = {}
        self._call_count = 0
        # Whether a function call was emitted, so the terminal event can report
        # ``tool_calls`` as Chat Completions does.
        self._has_function_call = False
        # Item key -> text pieces emitted for that message or reasoning item.
        self._streamed_text: dict[str, list[str]] = {}
        self._streamed_reasoning: dict[str, list[str]] = {}
        # The reasoning part the last reasoning delta belonged to; a new part starts on a new line.
        self._reasoning_part: Optional[tuple[str, str, Any]] = None
        self._reasoning_emitted = False
        self._finished = False
        self._partial = _PartialOutput()

    @property
    def finished(self) -> bool:
        """Whether the closing event (``response.completed`` or ``response.incomplete``) was parsed."""
        return self._finished

    def partial_output(self) -> Optional[AssistantMessageChunk]:
        """Return what this stream has delivered so far as one chunk, or None if nothing came."""
        return self._partial.chunk()

    def parse(self, event: Optional[dict[str, Any]]) -> Optional[AssistantMessageChunk]:
        """Turn one Responses SSE event into an ``AssistantMessageChunk``.

        Args:
            event: One decoded SSE event, or None for an empty block.

        Returns:
            The chunk the event contributes, or None when it carries nothing
            or arrives after the closing event.

        Raises:
            OpenAIAccountResponsesError: When the event reports a failed or cancelled response.
        """
        if not event or self._finished:
            return None
        chunk = self._parse_event(event)
        if chunk is not None:
            self._partial.add(chunk)
        return chunk

    def _parse_event(self, event: dict[str, Any]) -> Optional[AssistantMessageChunk]:
        event_type = str(event.get("type") or event.get("event") or "")
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            return self._text_delta_chunk(event)
        if event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
            return self._reasoning_delta_chunk(event, event_type=event_type)
        if event_type == "response.output_item.added":
            return self._added_function_call_chunk(event)
        if event_type == "response.function_call_arguments.delta":
            return self._function_call_arguments_delta_chunk(event)
        if event_type == "response.output_item.done":
            return self._done_output_item_chunk(event)
        if event_type in {"response.completed", "response.incomplete"}:
            self._finished = True
            return _terminal_stream_chunk(
                event,
                event_type=event_type,
                model_name=self._model_name,
                has_tool_calls=self._has_function_call,
            )
        if event_type in {"response.failed", "error"}:
            raise OpenAIAccountResponsesError(_stream_error_message(event))
        if event_type == "response.cancelled":
            # Someone stopped this response on purpose; asking again would not help.
            raise OpenAIAccountResponsesError(
                "Responses stream was cancelled before the response finished.",
                details={"should_retry": False},
            )
        return None

    def _text_delta_chunk(self, event: dict[str, Any]) -> Optional[AssistantMessageChunk]:
        delta = str(event.get("delta") or "")
        if not delta:
            return None
        self._streamed_text.setdefault(_delta_item_key(event), []).append(delta)
        return AssistantMessageChunk(content=delta, finish_reason="null")

    def _reasoning_delta_chunk(self, event: dict[str, Any], *, event_type: str) -> Optional[AssistantMessageChunk]:
        delta = str(event.get("delta") or "")
        if not delta:
            return None
        key = _delta_item_key(event)
        index_field = "summary_index" if event_type == "response.reasoning_summary_text.delta" else "content_index"
        part = (key, event_type, event.get(index_field))
        separator = "\n" if self._reasoning_emitted and part != self._reasoning_part else ""
        item_pieces = self._streamed_reasoning.get(key)
        if item_pieces is None:
            # A separator before an item's first part sits between items, not inside this one.
            self._streamed_reasoning[key] = [delta]
        else:
            item_pieces.extend((separator, delta))
        self._reasoning_part = part
        self._reasoning_emitted = True
        return AssistantMessageChunk(content="", reasoning_content=separator + delta, finish_reason="null")

    def _added_function_call_chunk(self, event: dict[str, Any]) -> Optional[AssistantMessageChunk]:
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return None
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            # Without an item id the deltas cannot be attributed; the done
            # event will deliver the whole call instead.
            return None
        tool_call = _tool_call_from_response_item(item, index=self._call_index(item_id))
        initial_arguments = _json_argument_string(item.get("arguments") or "")
        tool_call.arguments = initial_arguments
        self._streamed_calls[item_id] = (tool_call.id or "", initial_arguments)
        self._has_function_call = True
        return AssistantMessageChunk(content="", tool_calls=[tool_call], finish_reason="null")

    def _function_call_arguments_delta_chunk(self, event: dict[str, Any]) -> Optional[AssistantMessageChunk]:
        item_id = str(event.get("item_id") or "").strip()
        delta = str(event.get("delta") or "")
        streamed = self._streamed_calls.get(item_id)
        if streamed is None or not delta:
            return None
        call_id, emitted_arguments = streamed
        self._streamed_calls[item_id] = (call_id, emitted_arguments + delta)
        return AssistantMessageChunk(
            content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    type="function",
                    name="",
                    arguments=delta,
                    index=self._call_index(item_id),
                    response_item_id=item_id,
                ),
            ],
            finish_reason="null",
        )

    def _done_output_item_chunk(self, event: dict[str, Any]) -> Optional[AssistantMessageChunk]:
        item = event.get("item")
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type == "function_call":
            return self._done_function_call_chunk(item)
        if item_type == "message":
            streamed = _pop_streamed(self._streamed_text, event, item)
            rest = _unstreamed_rest(streamed, "".join(_message_content_text(item)), item=item)
            return AssistantMessageChunk(content=rest, finish_reason="null") if rest else None
        if item_type == "reasoning":
            streamed = _pop_streamed(self._streamed_reasoning, event, item)
            rest = _unstreamed_rest(streamed, "\n".join(_reasoning_text(item)), item=item)
            if not rest:
                return None
            if streamed is None and self._reasoning_emitted:
                rest = "\n" + rest
            self._reasoning_part = None
            self._reasoning_emitted = True
            return AssistantMessageChunk(content="", reasoning_content=rest, finish_reason="null")
        return None

    def _done_function_call_chunk(self, item: dict[str, Any]) -> AssistantMessageChunk:
        item_id = str(item.get("id") or "").strip()
        tool_call = _tool_call_from_response_item(item, index=self._call_index(item_id))
        streamed = self._streamed_calls.pop(item_id, None)
        if streamed is not None:
            _, emitted_arguments = streamed
            if not tool_call.arguments.startswith(emitted_arguments):
                # The streamed fragments are already concatenated downstream
                # and cannot be retracted; keep them rather than duplicate.
                logger.warning(
                    "Responses function call %s arguments diverged from the streamed deltas; "
                    "keeping the streamed arguments.",
                    item_id,
                )
                remainder = ""
            else:
                remainder = tool_call.arguments[len(emitted_arguments):]
            tool_call.name = ""
            tool_call.arguments = remainder
        self._has_function_call = True
        return AssistantMessageChunk(
            content="",
            tool_calls=[tool_call],
            finish_reason="tool_calls",
        )

    def _call_index(self, item_id: str) -> int:
        """Position of a function call in this response, in the order calls first appeared."""
        if item_id in self._call_indexes:
            return self._call_indexes[item_id]
        index = self._call_count
        self._call_count += 1
        if item_id:
            self._call_indexes[item_id] = index
        return index


class _PartialOutput:
    """Text, reasoning and tool calls one stream has delivered so far.

    Pieces are kept in lists, so a long stream costs no repeated copying. They are
    joined into one chunk only when a broken stream reports what it delivered.
    """

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        # (first fragment of the call, its argument pieces), in the order calls appeared.
        self._calls: list[tuple[ToolCall, list[str]]] = []
        self._call_positions: dict[str, int] = {}

    def add(self, chunk: AssistantMessageChunk) -> None:
        if isinstance(chunk.content, str) and chunk.content:
            self._content.append(chunk.content)
        if chunk.reasoning_content:
            self._reasoning.append(chunk.reasoning_content)
        for call in chunk.tool_calls or ():
            position = self._call_positions.get(call.response_item_id) if call.response_item_id else None
            if position is None:
                if call.response_item_id:
                    self._call_positions[call.response_item_id] = len(self._calls)
                self._calls.append((call, [call.arguments]))
            else:
                self._calls[position][1].append(call.arguments)

    def chunk(self) -> Optional[AssistantMessageChunk]:
        if not (self._content or self._reasoning or self._calls):
            return None
        tool_calls = [
            call.model_copy(update={"arguments": "".join(arguments)}) for call, arguments in self._calls
        ]
        return AssistantMessageChunk(
            content="".join(self._content),
            reasoning_content="".join(self._reasoning) or None,
            tool_calls=tool_calls or None,
            finish_reason="null",
        )


def _delta_item_key(event: dict[str, Any]) -> str:
    """Key of the output item a delta event belongs to: its item id, else its output index."""
    item_id = str(event.get("item_id") or "").strip()
    if item_id:
        return f"id:{item_id}"
    output_index = event.get("output_index")
    return f"index:{output_index}" if output_index is not None else ""


def _pop_streamed(streamed: dict[str, list[str]], event: dict[str, Any], item: dict[str, Any]) -> Optional[str]:
    """Remove and return the text already streamed for the item an ``output_item.done`` event closes.

    The done event names its item by ``item.id`` and ``output_index``; deltas name it by
    ``item_id``, else ``output_index``, else not at all, so each of those keys is tried.
    """
    item_id = str(item.get("id") or "").strip()
    output_index = event.get("output_index")
    keys = [f"id:{item_id}" if item_id else None, f"index:{output_index}" if output_index is not None else None, ""]
    for key in keys:
        if key is not None and key in streamed:
            return "".join(streamed.pop(key))
    return None


def _unstreamed_rest(streamed: Optional[str], full: str, *, item: dict[str, Any]) -> str:
    """Part of an item's full text that the deltas did not already deliver."""
    if streamed is None:
        return full
    if full.startswith(streamed):
        return full[len(streamed):]
    if not streamed.startswith(full):
        # The streamed pieces are already concatenated downstream and cannot be
        # taken back; keep them rather than repeat the text.
        logger.warning(
            "Responses %s item %s text differs from the streamed deltas; keeping the streamed text.",
            item.get("type"),
            item.get("id"),
        )
    return ""


def iter_sse_events(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    """Yield decoded event dictionaries from raw SSE lines."""
    block: list[str] = []
    for line in lines:
        if line:
            block.append(line)
            continue

        event = parse_sse_block(block)
        block = []
        if event is not None:
            yield event

    if block:
        event = parse_sse_block(block)
        if event is not None:
            yield event


def parse_sse_block(lines: list[str]) -> Optional[dict[str, Any]]:
    """Decode a single SSE block.

    Raises:
        ResponsesStreamInterruptedError: When the block's data is not valid JSON
            (reason :attr:`ResponsesStreamInterruptedError.GARBLED_EVENT`).
    """
    event_type = ""
    data_parts: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_type = value.strip()
        elif field == "data":
            data_parts.append(value.strip())

    if not data_parts:
        return {"type": event_type} if event_type else None

    raw_data = "\n".join(data_parts)
    if raw_data == "[DONE]":
        return None

    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise ResponsesStreamInterruptedError(
            _garbled_event_message(event_type, raw_data),
            reason=ResponsesStreamInterruptedError.GARBLED_EVENT,
        ) from exc

    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("type", event_type or parsed.get("event"))
    return parsed


def raise_for_http_error(response: httpx.Response) -> None:
    """Raise an OpenAI account error for unsuccessful HTTP responses."""
    if response.status_code < 400:
        return

    try:
        payload = response.json()
    except ValueError:
        payload = None
    detail = _http_error_message(payload)
    if not detail:
        text = (response.text or "").strip()
        if text:
            detail = summarize_provider_error_text(text, status_code=response.status_code)
    if not detail:
        detail = "Responses request failed without an error body."
    # Same shape as the OpenAI SDK's errors in Chat Completions mode, so logs and
    # text-based classifiers see the HTTP status in both API modes.
    raise OpenAIAccountResponsesError(
        f"Error code: {response.status_code} - {detail}",
        status_code=response.status_code,
        body=payload,
    )


def message_from_stream_chunk(chunk: Optional[AssistantMessageChunk]) -> AssistantMessage:
    """Convert an accumulated stream chunk into a complete assistant message."""
    if chunk is None:
        return AssistantMessage(content="", finish_reason="stop")

    # An explicit cause such as "length" or "content_filter" is kept even when
    # the message has tool calls: a call cut off by the output limit has
    # incomplete arguments.
    finish_reason = chunk.finish_reason
    if chunk.tool_calls and finish_reason in {"null", "stop"}:
        finish_reason = "tool_calls"
    elif finish_reason == "null":
        finish_reason = "stop"

    return AssistantMessage(
        content=chunk.content,
        tool_calls=chunk.tool_calls,
        usage_metadata=chunk.usage_metadata,
        finish_reason=finish_reason,
        parser_content=chunk.parser_content,
        reasoning_content=chunk.reasoning_content,
        prompt_token_ids=chunk.prompt_token_ids,
        completion_token_ids=chunk.completion_token_ids,
        logprobs=chunk.logprobs,
        response_id=chunk.response_id,
        response_model=chunk.response_model,
        provider_metadata=chunk.provider_metadata,
    )


def _message_dicts(messages: Union[str, list[BaseMessage], list[dict]]) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        raise OpenAIAccountResponsesError(f"Unsupported messages type: {type(messages).__name__}")

    converted: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, dict):
            converted.append(dict(item))
        elif isinstance(item, ToolMessage):
            converted.append({"role": item.role, "content": item.content, "tool_call_id": item.tool_call_id})
        elif isinstance(item, BaseMessage):
            converted.append(item.model_dump())
        else:
            raise OpenAIAccountResponsesError(f"Unsupported message type: {type(item).__name__}")

    if not converted:
        raise OpenAIAccountResponsesError("OpenAI account Responses request requires at least one message.")
    return converted


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    for part in content:
        text = _content_part_text(part)
        if text:
            text_parts.append(text)
    return "".join(text_parts)


def _content_part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""

    for key in ("text", "content"):
        value = part.get(key)
        if value is not None:
            return str(value)
    return ""


def _user_content_parts(content: Any, content_text: str) -> list[dict[str, Any]]:
    """Map user content to Responses ``input_text`` and ``input_image`` parts in their original order.
    Adjacent text parts are merged into one ``input_text``.
    Parts that are neither text nor images are dropped.
    """
    if not isinstance(content, list):
        return [{"type": "input_text", "text": content_text}]

    parts: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush_text() -> None:
        if pending_text:
            parts.append({"type": "input_text", "text": "".join(pending_text)})
            pending_text.clear()

    for part in content:
        image = _input_image_part(part)
        if image is not None:
            flush_text()
            parts.append(image)
            continue
        text = _content_part_text(part)
        if text:
            pending_text.append(text)
    flush_text()

    return parts or [{"type": "input_text", "text": ""}]


def _input_image_part(part: Any) -> Optional[dict[str, Any]]:
    """Convert a chat-style ``image_url`` part to a Responses ``input_image`` part."""
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None

    image_url = part.get("image_url")
    detail = None
    if isinstance(image_url, dict):
        detail = image_url.get("detail")
        image_url = image_url.get("url")
    if not isinstance(image_url, str) or not image_url:
        return None

    image: dict[str, Any] = {"type": "input_image", "image_url": image_url}
    if detail:
        image["detail"] = detail
    return image


def _tool_result_item(message: dict[str, Any], output: str) -> Optional[dict[str, Any]]:
    tool_call_id = str(message.get("tool_call_id") or message.get("call_id") or "").strip()
    if not tool_call_id:
        return None
    return {
        "type": "function_call_output",
        "call_id": tool_call_id,
        "output": output,
    }


def _response_input_tool_call(tool_call: Any, *, fallback_index: int) -> dict[str, Any]:
    if isinstance(tool_call, ToolCall):
        call_id = tool_call.id or f"call_{uuid.uuid4().hex}"
        return {
            "type": "function_call",
            "id": tool_call.response_item_id or call_id,
            "call_id": call_id,
            "name": tool_call.name,
            "arguments": _json_argument_string(tool_call.arguments),
        }

    if isinstance(tool_call, dict):
        function = _dict_or_empty(tool_call.get("function"))
        arguments = tool_call.get("arguments") or function.get("arguments") or "{}"
        name = tool_call.get("name") or function.get("name") or ""
        call_id = tool_call.get("call_id") or tool_call.get("id") or f"call_{fallback_index}"
        item_id = tool_call.get("response_item_id") or tool_call.get("id") or call_id
        return {
            "type": "function_call",
            "id": str(item_id),
            "call_id": str(call_id),
            "name": str(name),
            "arguments": _json_argument_string(arguments),
        }

    raise OpenAIAccountResponsesError(f"Unsupported tool call type: {type(tool_call).__name__}")


def _responses_tool_choice(tool_choice: Any) -> Any:
    """Flatten a Chat-style named ``tool_choice`` into the Responses shape.

    Chat Completions names the tool as ``{"type": "function", "function": {"name": ...}}``;
    Responses expects ``{"type": "function", "name": ...}``. Other values are sent unchanged.
    """
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function" and "name" not in tool_choice:
        name = _dict_or_empty(tool_choice.get("function")).get("name")
        if name:
            return {"type": "function", "name": str(name)}
    return tool_choice


def _tool_definition(tool: Union[ToolInfo, dict]) -> dict[str, Any]:
    if isinstance(tool, ToolInfo):
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": _schema_dict(tool.parameters),
        }

    if isinstance(tool, dict):
        function = _dict_or_empty(tool.get("function"))
        if function:
            return _with_strict(
                {
                    "type": "function",
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "parameters": _schema_dict(function.get("parameters") or {}),
                },
                function.get("strict"),
            )
        if tool.get("type") == "function" and tool.get("name"):
            return _with_strict(
                {
                    "type": "function",
                    "name": str(tool.get("name") or ""),
                    "description": str(tool.get("description") or ""),
                    "parameters": _schema_dict(tool.get("parameters") or {}),
                },
                tool.get("strict"),
            )

    raise OpenAIAccountResponsesError(f"Unsupported tool schema: {tool!r}")


def _with_strict(definition: dict[str, Any], strict: Any) -> dict[str, Any]:
    """Keep a caller-set ``strict``: Responses treats a missing one differently from ``false``."""
    if strict is not None:
        definition["strict"] = strict
    return definition


def _schema_dict(schema: Any) -> dict[str, Any]:
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, BaseModel):
        return schema.model_json_schema()
    return {"type": "object", "properties": {}}


def _read_output_items(output: Any) -> tuple[list[str], list[str], list[ToolCall]]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    if not isinstance(output, list):
        return content_parts, reasoning_parts, tool_calls

    for item in output:
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type") or "")
        if item_type == "message":
            content_parts.extend(_message_content_text(item))
        elif item_type == "reasoning":
            reasoning_parts.extend(_reasoning_text(item))
        elif item_type == "function_call":
            tool_calls.append(_tool_call_from_response_item(item, index=len(tool_calls)))

    return content_parts, reasoning_parts, tool_calls


def _message_content_text(item: dict[str, Any]) -> list[str]:
    content = item.get("content")
    if not isinstance(content, list):
        return []

    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"output_text", "text"} and part.get("text") is not None:
            texts.append(str(part.get("text")))
        elif part_type == "refusal" and part.get("refusal") is not None:
            # Streamed refusals (``response.refusal.delta``) are answer text too.
            texts.append(str(part.get("refusal")))
    return texts


def _reasoning_text(item: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    summary = item.get("summary")
    if isinstance(summary, list):
        for part in summary:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("text") is not None:
                texts.append(str(part.get("text")))

    # Some providers (DeepSeek and GLM, for example) send readable reasoning as
    # ``content[]`` ``reasoning_text`` parts.
    content = item.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "reasoning_text" and part.get("text") is not None:
                texts.append(str(part.get("text")))

    text = item.get("text")
    if text is not None:
        texts.append(str(text))
    return texts


def _tool_call_from_response_item(item: dict[str, Any], *, index: int) -> ToolCall:
    response_item_id = str(item.get("id") or "").strip() or None
    call_id = str(item.get("call_id") or response_item_id or "")
    return ToolCall(
        id=call_id,
        type="function",
        name=str(item.get("name") or ""),
        arguments=_json_argument_string(item.get("arguments") or "{}"),
        index=index,
        response_item_id=response_item_id,
    )


def _terminal_stream_chunk(
    event: dict[str, Any],
    *,
    event_type: str,
    model_name: str,
    has_tool_calls: bool,
) -> AssistantMessageChunk:
    response_payload = event.get("response")
    usage = response_payload.get("usage") if isinstance(response_payload, dict) else event.get("usage")
    finish_reason = _finish_reason(
        response_payload if isinstance(response_payload, dict) else {},
        has_tool_calls=has_tool_calls,
        incomplete=True if event_type == "response.incomplete" else None,
    )
    return AssistantMessageChunk(
        content="",
        usage_metadata=_usage_from_payload(usage, model_name=model_name),
        finish_reason=finish_reason,
        response_id=(
            str(response_payload.get("id") or "") or None
            if isinstance(response_payload, dict)
            else None
        ),
        response_model=(
            str(response_payload.get("model") or model_name or "") or None
            if isinstance(response_payload, dict)
            else (model_name or None)
        ),
        provider_metadata=(
            _response_provider_metadata(response_payload)
            if isinstance(response_payload, dict)
            else {}
        ),
    )


def _stream_error_message(event: dict[str, Any]) -> str:
    """Message for an ``error`` / ``response.failed`` event, keeping the provider's error code.

    ``error`` events carry ``code``/``message`` at the top level or under ``error``;
    ``response.failed`` carries them under ``response.error``.
    """
    response = event.get("response")
    candidates = [event, event.get("error"), response.get("error") if isinstance(response, dict) else None]
    errors = [item for item in candidates if isinstance(item, dict)]
    error = next((item for item in errors if item.get("message")), None)
    message = str(error["message"]) if error else "OpenAI account Responses stream failed."
    if error:
        metadata = _error_metadata_summary(error)
        if metadata:
            message = f"{message} (metadata: {metadata})"
    code = (error or {}).get("code") or next((item["code"] for item in errors if item.get("code")), None)
    return f"Error code: {code} - {message}" if code else message


def _garbled_event_message(event_type: str, raw_data: str) -> str:
    """Short error text for a stream event that is not valid JSON: the event name and a short sample."""
    name = f"'{event_type}'" if event_type else "without an event name"
    sample = raw_data
    if len(sample) > _MAX_GARBLED_EVENT_SAMPLE_CHARS:
        sample = f"{sample[:_MAX_GARBLED_EVENT_SAMPLE_CHARS]}... [{len(raw_data)} chars]"
    message = f"Responses stream event {name} is not valid JSON: {sample}"
    if len(message) > _MAX_GARBLED_EVENT_MESSAGE_CHARS:
        message = f"{message[:_MAX_GARBLED_EVENT_MESSAGE_CHARS - 3]}..."
    return message


def _error_metadata_summary(*sources: Any) -> Optional[str]:
    """Bounded one-line JSON of the first provider ``metadata`` found in ``sources``.

    OpenRouter keeps the upstream provider's error here (``metadata.raw``); its own
    ``message`` is often only "Provider returned error".
    """
    metadata = next(
        (source.get("metadata") for source in sources if isinstance(source, dict) and source.get("metadata")),
        None,
    )
    if isinstance(metadata, dict):
        # ``raw`` is often the upstream error body as an indented JSON string; decoding it
        # keeps the summary readable and inside the length cap.
        metadata = {key: _decoded_json(value) for key, value in metadata.items() if value is not None}
    if not metadata:
        return None
    if isinstance(metadata, (dict, list)):
        text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), default=str)
    else:
        text = str(metadata)
    if len(text) > _MAX_ERROR_METADATA_CHARS:
        overflow = len(text) - _MAX_ERROR_METADATA_CHARS
        text = f"{text[:_MAX_ERROR_METADATA_CHARS]}... [truncated {overflow} chars]"
    return text


def _decoded_json(value: Any) -> Any:
    """Return ``value`` parsed when it is a JSON object or array string, else unchanged."""
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _usage_from_payload(usage: Any, *, model_name: str) -> Optional[UsageMetadata]:
    if not isinstance(usage, dict):
        return None

    input_tokens = _int_or_zero(_first_present(usage, "input_tokens", "prompt_tokens"))
    output_tokens = _int_or_zero(_first_present(usage, "output_tokens", "completion_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    cache_tokens = 0
    cache_creation_tokens = None
    token_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    if isinstance(token_details, dict):
        cache_tokens = _int_or_zero(token_details.get("cached_tokens"))
        cache_creation_tokens = _optional_int(
            _first_present(token_details, "cache_creation_tokens", "cache_write_tokens")
        )

    reasoning_tokens = 0
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
    if isinstance(output_details, dict):
        reasoning_tokens = _int_or_zero(output_details.get("reasoning_tokens"))

    return UsageMetadata(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_tokens=cache_tokens,
        cache_read_tokens=cache_tokens if token_details is not None else None,
        cache_status="observed" if token_details is not None else None,
        cache_source="provider_usage" if token_details is not None else None,
        cache_authoritative=token_details is not None,
        cache_creation_input_tokens=cache_creation_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _response_provider_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-sensitive Responses API status metadata."""
    metadata: dict[str, Any] = {}
    for key in ("status", "service_tier"):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)) and value != "":
            metadata[key] = value
    incomplete_details = payload.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        metadata["incomplete_details"] = dict(incomplete_details)
    return metadata


def _finish_reason(payload: dict[str, Any], *, has_tool_calls: bool, incomplete: Optional[bool] = None) -> str:
    """Map a Responses outcome to the ``finish_reason`` Chat Completions reports.

    An incomplete response keeps its cause even when it contains tool calls:
    ``content_filter`` stays ``content_filter``, while ``max_output_tokens``,
    any other reason or no reason becomes ``length``. A completed response is
    ``tool_calls`` when it contains tool calls, otherwise ``stop``.
    ``incomplete`` defaults to the payload's ``status``.
    """
    if incomplete is None:
        incomplete = str(payload.get("status") or "") == "incomplete"
    if incomplete:
        details = payload.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_calls" if has_tool_calls else "stop"


def _http_error_message(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        message = str(error["message"])
        # OpenRouter puts upstream errors in error.metadata, its own validation errors in metadata.
        metadata = _error_metadata_summary(error, payload)
        return f"{message} (metadata: {metadata})" if metadata else message
    if isinstance(error, str):
        return error
    if isinstance(payload.get("detail"), str):
        return str(payload["detail"])
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if payload.get(key) is not None:
            return payload.get(key)
    return None


def _json_argument_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _int_or_zero(value: Any) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
