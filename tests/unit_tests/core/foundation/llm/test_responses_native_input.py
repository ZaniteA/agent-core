# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Responses mode must keep ready-made Responses items and native image and file parts."""

from __future__ import annotations

import json

import httpx
import pytest

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig, SystemMessage, UserMessage
from openjiuwen.core.foundation.llm.utils.responses_transport import OpenAIAccountResponsesTransport
from openjiuwen.core.foundation.llm.utils.responses_utils import build_request_body, convert_messages

_DATA_URL = "data:image/png;base64,iVBORw0KGgo="
_REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Add the numbers."}],
    "encrypted_content": "gAAAA-sealed",
}
_CALL = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "add", "arguments": '{"a":2,"b":2}'}
_OUTPUT = {"type": "function_call_output", "call_id": "call_1", "output": "4"}
_REFERENCE = {"type": "item_reference", "id": "msg_9"}
_PDF = {"type": "input_file", "filename": "report.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}
_RESPONSES_SSE = (
    "event: response.output_text.delta\n"
    'data: {"delta":"ok"}\n\n'
    "event: response.completed\n"
    'data: {"response":{"id":"resp-1","status":"completed",'
    '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
).encode()


def _user_content(content) -> list[dict]:
    _, inputs = convert_messages([UserMessage(content=content)])
    assert len(inputs) == 1
    assert inputs[0]["role"] == "user"
    return inputs[0]["content"]


def test_ready_made_items_are_copied_in_place():
    instructions, inputs = convert_messages([
        SystemMessage(content="Be brief."),
        UserMessage(content="What is 2+2?"),
        _REASONING,
        _CALL,
        _OUTPUT,
        _REFERENCE,
        UserMessage(content="Thanks."),
    ])

    assert instructions == "Be brief."
    assert inputs == [
        {"role": "user", "content": [{"type": "input_text", "text": "What is 2+2?"}]},
        _REASONING,
        _CALL,
        _OUTPUT,
        _REFERENCE,
        {"role": "user", "content": [{"type": "input_text", "text": "Thanks."}]},
    ]


def test_any_item_type_is_copied():
    """A dict with a ``type`` and no ``role`` is a Responses item, whatever the type."""
    item = {"type": "web_search_call", "id": "ws_1", "status": "completed"}

    _, inputs = convert_messages([UserMessage(content="Search."), item])

    assert inputs[1] == item


def test_dict_with_a_role_is_still_read_as_a_message():
    """Only role-less dicts are taken as ready-made items."""
    _, inputs = convert_messages([{"type": "message", "role": "user", "content": "Hi"}])

    assert inputs == [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]


def test_request_body_keeps_ready_made_items():
    body = build_request_body(
        model="gpt-5.6-luna",
        messages=[UserMessage(content="What is 2+2?"), _REASONING, _CALL, _OUTPUT],
    )

    assert body["input"][1:] == [_REASONING, _CALL, _OUTPUT]


def test_input_image_parts_are_kept_unchanged():
    by_url = {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "high"}
    by_file = {"type": "input_image", "file_id": "file-1"}

    parts = _user_content([{"type": "text", "text": "Look."}, by_url, by_file])

    assert parts == [{"type": "input_text", "text": "Look."}, by_url, by_file]


def test_input_file_parts_are_kept_unchanged():
    by_url = {"type": "input_file", "file_url": "https://example.com/report.pdf"}
    by_file = {"type": "input_file", "file_id": "file-1"}

    parts = _user_content([{"type": "text", "text": "Summarise."}, _PDF, by_url, by_file])

    assert parts == [{"type": "input_text", "text": "Summarise."}, _PDF, by_url, by_file]


def test_agent_core_image_part_becomes_input_image():
    """Tools and client tools write images as ``{"type": "image", "data_url": ...}``."""
    parts = _user_content([{
        "type": "image",
        "source": "read_file",
        "source_path": "chart.png",
        "mime_type": "image/png",
        "data_url": _DATA_URL,
    }])

    assert parts == [{"type": "input_image", "image_url": _DATA_URL}]


def test_agent_core_image_source_forms_become_input_image():
    parts = _user_content([
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
    ])

    assert parts == [
        {"type": "input_image", "image_url": _DATA_URL},
        {"type": "input_image", "image_url": "https://example.com/a.png"},
    ]


def test_agent_core_image_part_without_an_image_is_dropped():
    """An image part without usable image data is dropped rather than sent half-formed."""
    parts = _user_content([
        {"type": "text", "text": "Hi"},
        {"type": "image", "source": "read_file"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}},
    ])

    assert parts == [{"type": "input_text", "text": "Hi"}]


def test_native_parts_keep_their_place_between_text():
    parts = _user_content([
        {"type": "text", "text": "Read this"},
        _PDF,
        {"type": "text", "text": "and compare it"},
        {"type": "text", "text": " with this."},
        {"type": "image", "data_url": _DATA_URL},
    ])

    assert parts == [
        {"type": "input_text", "text": "Read this"},
        _PDF,
        {"type": "input_text", "text": "and compare it with this."},
        {"type": "input_image", "image_url": _DATA_URL},
    ]


@pytest.mark.asyncio
async def test_model_sends_ready_made_items_and_native_parts():
    """End to end through ``Model.invoke`` with the API-key client in Responses mode."""
    sent_bodies: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, content=_RESPONSES_SSE, headers={"content-type": "text/event-stream"})

    model = Model(
        ModelClientConfig(
            client_provider="OpenAI",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            api_mode="responses",
            verify_ssl=False,
        ),
        ModelRequestConfig(model="gpt-5.6-luna"),
    )
    model._client._make_responses_transport = lambda **_: OpenAIAccountResponsesTransport(
        transport=httpx.MockTransport(handler),
    )

    await model.invoke([
        UserMessage(content=[{"type": "text", "text": "What is in the file?"}, _PDF]),
        _REASONING,
        _CALL,
        _OUTPUT,
    ])

    assert sent_bodies[0]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "What is in the file?"}, _PDF]},
        _REASONING,
        _CALL,
        _OUTPUT,
    ]
