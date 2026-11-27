# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ReActAgent closes the model stream as soon as its streaming loop stops."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from openjiuwen.core.foundation.llm import AssistantMessageChunk, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.rail.base import AgentRail
from tests.unit_tests.fixtures.mock_llm import MockLLMModel


def make_agent():
    os.environ.setdefault("LLM_SSL_VERIFY", "false")
    config = ReActAgentConfig(
        model_config_obj=ModelRequestConfig(model="gpt-3.5-turbo"),
        model_client_config=ModelClientConfig(
            client_provider="OpenAI", api_key="mock_key", api_base="mock_url", timeout=30, verify_ssl=False
        ),
        prompt_template=[{"role": "system", "content": "你是一个助手。"}],
    )
    return ReActAgent(card=AgentCard(description="stream close")).configure(config)


class StreamingModel(MockLLMModel):
    """Keeps every stream it hands out, so only an explicit close can end one."""

    def __init__(self):
        super().__init__()
        self.streams = []
        self.closed = 0

    def stream(self, *args, **kwargs):
        stream = self._chunks()
        self.streams.append(stream)
        return stream

    async def _chunks(self):
        try:
            yield AssistantMessageChunk(content="partial")
            yield AssistantMessageChunk(content="never read")
        finally:
            self.closed += 1


class FailingInspector(AgentRail):
    """Makes the agent's streaming loop stop on the first chunk."""

    async def before_model_call(self, ctx):
        ctx.extra["_stream_chunk_inspectors"] = [self.inspect]

    @staticmethod
    def inspect(ctx, chunk):
        raise RuntimeError("inspector failed")


@pytest.mark.asyncio
async def test_model_stream_is_closed_when_the_streaming_loop_stops():
    agent = make_agent()
    agent._recover_from_model_exception = AsyncMock(return_value=False)
    await agent.register_rail(FailingInspector())
    llm = StreamingModel()

    with patch.object(agent, "_get_llm", return_value=llm):
        chunks = [chunk async for chunk in agent.stream({"query": "hello"})]

    answers = [chunk.payload for chunk in chunks if getattr(chunk, "type", None) == "answer"]
    assert answers[-1]["output"] == "inspector failed" and answers[-1]["result_type"] == "error"
    assert len(llm.streams) == 1, "the model was streamed once"
    assert llm.closed == 1, "the model stream is closed, not left for garbage collection"
