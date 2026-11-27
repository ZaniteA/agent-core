# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""One scripted agent task gives the same result in Chat and Responses mode.

The task: the model calls a tool, the tool returns a picture, the model
answers. A real ReActAgent and the real model client run against the shared
fake server, on the plain call path and on the streaming path. Both modes must
make the same tool call, show the model the same conversation, give the same
answer, and never send the other protocol's keys.

The last two tests record known gaps as they are today: a browser screenshot
reaches a Chat model only as a text placeholder, and an MCP tool's audio and
file content reaches it as text with the base64 data inside. Both stub the MCP
session through the client's private ``_session``, so no MCP server runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from mcp.types import AudioContent, BlobResourceContents, CallToolResult, EmbeddedResource, ImageContent

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.tool import LocalFunction, McpServerConfig, McpToolCard, StdioClient, ToolCard
from openjiuwen.core.foundation.tool.mcp.base import MCPTool
from openjiuwen.core.runner import Runner
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness.tools.base_tool import ToolOutput
from openjiuwen.harness.tools.browser_move.clients.stdio_client import BrowserMoveStdioClient
from tests.unit_tests.fixtures.fake_llm_servers import (  # noqa: F401  (fake_model_server is a fixture)
    FakeToolCall,
    fake_model_server,
)

_SYSTEM_PROMPT = "You are a camera bot."
_QUERY = "Take a picture of a cat."
_IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
_ANSWER = "A cat sits on a mat."
_CAMERA_PARAMS = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
_AUDIO_BASE64 = "UklGRiQAAABXQVZF"  # the start of a WAV file
_PDF_BASE64 = "JVBERi0xLjQK"  # the start of a PDF file

# Top-level request keys that only one protocol uses.
_CHAT_ONLY_KEYS = frozenset(
    {"messages", "stream_options", "max_tokens", "max_completion_tokens", "reasoning_effort", "response_format"}
)
_RESPONSES_ONLY_KEYS = frozenset(
    {"input", "instructions", "store", "include", "max_output_tokens", "text", "previous_response_id"}
)
# Content part and input item types that only one protocol uses.
_CHAT_ONLY_TYPES = frozenset({"text", "image_url"})
_RESPONSES_ONLY_TYPES = frozenset(
    {"input_text", "input_image", "input_file", "output_text", "function_call", "function_call_output", "reasoning"}
)
# agent-core settings that must never reach a provider (with KV-cache affinity off).
_INTERNAL_KEYS = frozenset(
    {"api_mode", "context_window", "session_id", "parent_session_id", "agent_hint", "stream_events",
     "responses_items"}
)


@pytest_asyncio.fixture
async def runner():
    await Runner.start()
    yield
    await Runner.stop()


def _model(api_mode: Optional[str]) -> Model:
    return Model(
        ModelClientConfig(
            client_provider="OpenAI",
            api_key="mock-api-key",
            api_base="https://api.example.test/v1",
            verify_ssl=False,
            api_mode=api_mode,
        ),
        ModelRequestConfig(model="gpt-test"),
    )


def _agent(agent_id: str, api_mode: Optional[str], **config: Any) -> ReActAgent:
    agent = ReActAgent(card=AgentCard(id=agent_id))
    agent.configure(ReActAgentConfig(model_name="gpt-test", max_iterations=4, **config))
    agent.set_llm(_model(api_mode))
    return agent


async def _answer(agent: ReActAgent, query: str, *, streaming: bool, conversation_id: str) -> str:
    if not streaming:
        result = await agent.invoke({"query": query}, session=None)
        return result["output"]
    frames = [frame async for frame in agent.stream({"query": query, "conversation_id": conversation_id})]
    return frames[-1].payload["output"]


def _conversation(body: dict[str, Any]) -> list[tuple]:
    """What the model is shown, written the same way for both protocols."""
    shown: list[tuple] = []

    def add_content(role: str, content: Any) -> None:
        if isinstance(content, str):
            if content:
                shown.append((role, content))
            return
        for part in content or []:
            if part["type"] in {"text", "input_text", "output_text"}:
                shown.append((role, part["text"]))
            elif part["type"] == "image_url":
                shown.append((role, "image", part["image_url"]["url"]))
            elif part["type"] == "input_image":
                shown.append((role, "image", part["image_url"]))

    if "messages" in body:
        for message in body["messages"]:
            if message["role"] == "tool":
                shown.append(("tool result", message["tool_call_id"], message["content"]))
                continue
            add_content(message["role"], message.get("content"))
            for call in message.get("tool_calls") or []:
                function = call["function"]
                shown.append(("tool call", call["id"], function["name"], json.loads(function["arguments"])))
        return shown

    add_content("system", body.get("instructions"))
    for item in body["input"]:
        if item.get("type") == "function_call":
            shown.append(("tool call", item["call_id"], item["name"], json.loads(item["arguments"])))
        elif item.get("type") == "function_call_output":
            shown.append(("tool result", item["call_id"], item["output"]))
        else:
            add_content(item["role"], item["content"])
    return shown


def _offered_tools(body: dict[str, Any]) -> tuple[list[tuple], Any]:
    """The tools the model is offered and the tool choice, the same way for both protocols."""
    tools = []
    for tool in body.get("tools") or []:
        function = tool.get("function", tool)  # Chat nests the definition; Responses does not
        tools.append((function["name"], function.get("description"), function.get("parameters")))
    return tools, body.get("tool_choice")


def _keys_and_types(value: Any, keys: set[str], types: set[str]) -> None:
    if isinstance(value, dict):
        keys.update(value)
        if isinstance(value.get("type"), str):
            types.add(value["type"])
        for item in value.values():
            _keys_and_types(item, keys, types)
    elif isinstance(value, list):
        for item in value:
            _keys_and_types(item, keys, types)


async def _run_camera_task(fake_model_server, api_mode: Optional[str], *, streaming: bool):
    pictures: list[str] = []

    def take_picture(label: str) -> ToolOutput:
        pictures.append(label)
        return ToolOutput(
            success=True,
            data={
                "content": f"Picture of {label} taken.",
                "multimodal": [{"type": "image", "source_path": "camera.png", "data_url": _IMAGE_URL}],
            },
        )

    suffix = f"{api_mode or 'chat'}_{'stream' if streaming else 'invoke'}"
    card = ToolCard(
        id=f"take_picture_{suffix}",
        name="take_picture",
        description="Take a picture of something.",
        input_params=_CAMERA_PARAMS,
    )
    agent = _agent(f"camera_agent_{suffix}", api_mode, prompt_template=[{"role": "system", "content": _SYSTEM_PROMPT}])
    Runner.resource_mgr.add_tool(LocalFunction(card=card, func=take_picture), tag=agent.card.id)
    agent.ability_manager.add(card)
    fake_model_server.add_reply(tool_calls=[FakeToolCall(name="take_picture", arguments={"label": "cat"})])
    fake_model_server.add_reply(text=_ANSWER)
    try:
        answer = await _answer(agent, _QUERY, streaming=streaming, conversation_id=f"camera_{suffix}")
    finally:
        Runner.resource_mgr.remove_tool(card.id)
    return answer, pictures, [request.body for request in fake_model_server.requests]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "stream"])
async def test_camera_task_is_the_same_in_chat_and_responses_mode(fake_model_server, runner, streaming):
    chat = await _run_camera_task(fake_model_server, None, streaming=streaming)
    fake_model_server.requests.clear()
    responses = await _run_camera_task(fake_model_server, "responses", streaming=streaming)

    chat_answer, chat_pictures, chat_bodies = chat
    responses_answer, responses_pictures, responses_bodies = responses
    assert chat_answer == responses_answer == _ANSWER
    assert chat_pictures == responses_pictures == ["cat"]
    assert [_conversation(body) for body in chat_bodies] == [_conversation(body) for body in responses_bodies]
    assert [_offered_tools(body) for body in chat_bodies] == [_offered_tools(body) for body in responses_bodies]
    camera_tool = ("take_picture", "Take a picture of something.", _CAMERA_PARAMS)
    assert _offered_tools(chat_bodies[0]) == ([camera_tool], "auto")
    assert _conversation(chat_bodies[-1]) == [
        ("system", _SYSTEM_PROMPT),
        ("user", _QUERY),
        ("tool call", "call_1", "take_picture", {"label": "cat"}),
        ("tool result", "call_1", "Picture of cat taken."),
        ("user", "Image loaded from tool: camera.png"),
        ("user", "image", _IMAGE_URL),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "stream"])
@pytest.mark.parametrize(
    ("api_mode", "other_keys", "other_types"),
    [
        pytest.param(None, _RESPONSES_ONLY_KEYS, _RESPONSES_ONLY_TYPES, id="chat"),
        pytest.param("responses", _CHAT_ONLY_KEYS, _CHAT_ONLY_TYPES, id="responses"),
    ],
)
async def test_camera_task_sends_no_key_of_the_other_protocol(
    fake_model_server, runner, streaming, api_mode, other_keys, other_types,
):
    _, _, bodies = await _run_camera_task(fake_model_server, api_mode, streaming=streaming)

    assert len(bodies) == 2
    for body in bodies:
        nested_keys: set[str] = set()
        types: set[str] = set()
        _keys_and_types(body, nested_keys, types)
        assert not set(body) & other_keys
        assert not types & other_types
        assert not nested_keys & _INTERNAL_KEYS


@pytest.mark.asyncio
async def test_browser_screenshot_reaches_chat_as_text_today(fake_model_server, runner):
    # Known gap, recorded as it is: the browser client turns the screenshot
    # into a text placeholder, so the next Chat request carries no image. A
    # later change makes screenshots reach vision models and updates this test.
    client = BrowserMoveStdioClient(
        McpServerConfig(
            server_id="browser", server_name="browser", server_path="stdio://playwright", client_type="stdio",
        )
    )
    screenshot = CallToolResult(content=[ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")])
    client._session = SimpleNamespace(call_tool=AsyncMock(return_value=screenshot))
    card = McpToolCard(
        id="browser_take_screenshot_gap",
        name="browser_take_screenshot",
        server_name="browser",
        description="Take a screenshot of the page.",
        input_params={"type": "object", "properties": {}},
    )
    agent = _agent("browser_screenshot_gap_agent", None)
    Runner.resource_mgr.add_tool(MCPTool(client, card), tag=agent.card.id)
    agent.ability_manager.add(card)
    fake_model_server.add_reply(tool_calls=[FakeToolCall(name="browser_take_screenshot")])
    fake_model_server.add_reply(text="I cannot see the image.")
    try:
        await _answer(agent, "What does the page show?", streaming=False, conversation_id="browser_gap")
    finally:
        Runner.resource_mgr.remove_tool(card.id)

    last_request = fake_model_server.requests[-1].body
    assert _conversation(last_request)[-1] == ("tool result", "call_1", "[binary content: image/png]")
    assert not [item for item in _conversation(last_request) if item[1:2] == ("image",)]


@pytest.mark.asyncio
async def test_mcp_audio_and_file_reach_chat_as_base64_text_today(fake_model_server, runner):
    # Known gap, recorded as it is: an MCP tool's audio and file content
    # becomes tool text with the base64 data inside, and no audio or file part
    # reaches the model. A later change replaces the data with extracted text
    # or a note and updates this test.
    client = StdioClient(
        McpServerConfig(
            server_id="recorder", server_name="recorder", server_path="stdio://recorder", client_type="stdio",
        )
    )
    files = CallToolResult(
        content=[
            AudioContent(type="audio", data=_AUDIO_BASE64, mimeType="audio/wav"),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(uri="file:///report.pdf", mimeType="application/pdf", blob=_PDF_BASE64),
            ),
        ]
    )
    client._session = SimpleNamespace(call_tool=AsyncMock(return_value=files))
    card = McpToolCard(
        id="recorder_fetch_files_gap",
        name="fetch_files",
        server_name="recorder",
        description="Fetch a recording and a report.",
        input_params={"type": "object", "properties": {}},
    )
    agent = _agent("mcp_files_gap_agent", None)
    Runner.resource_mgr.add_tool(MCPTool(client, card), tag=agent.card.id)
    agent.ability_manager.add(card)
    fake_model_server.add_reply(tool_calls=[FakeToolCall(name="fetch_files")])
    fake_model_server.add_reply(text="I cannot open these.")
    try:
        await _answer(agent, "Fetch the recording and the report.", streaming=False, conversation_id="mcp_files_gap")
    finally:
        Runner.resource_mgr.remove_tool(card.id)

    last_request = fake_model_server.requests[-1].body
    kind, call_id, text = _conversation(last_request)[-1]
    assert (kind, call_id) == ("tool result", "call_1")
    assert _AUDIO_BASE64 in text and _PDF_BASE64 in text
    keys: set[str] = set()
    types: set[str] = set()
    _keys_and_types(last_request, keys, types)
    assert not types & {"input_audio", "audio", "file", "input_file"}
