# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Model.stream closes the client stream as soon as the reader stops reading."""

from contextlib import aclosing
from unittest.mock import patch

import pytest

from openjiuwen.core.foundation.llm import (
    AssistantMessageChunk,
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    ProviderType,
)


class _ClientStream:
    """Client stream that records whether it was closed.

    Every stream it hands out is kept, as a caller might, so only an explicit
    close can end one: garbage collection never does it in these tests.
    """

    def __init__(self, close_error: Exception | None = None):
        self.close_error = close_error
        self.streams = []
        self.closed = False

    def __call__(self, **kwargs):
        stream = self._chunks()
        self.streams.append(stream)
        return stream

    async def _chunks(self):
        try:
            yield AssistantMessageChunk(content="frame-1")
            yield AssistantMessageChunk(content="frame-2")
        finally:
            self.closed = True
            if self.close_error is not None:
                raise self.close_error


def _build_model_with_stream(stream_fn) -> Model:
    model = Model(
        model_client_config=ModelClientConfig(
            client_provider=ProviderType.OpenAI,
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
            verify_ssl=False,
        ),
        model_config=ModelRequestConfig(model="mock-model"),
    )
    model._client.stream = stream_fn
    return model


@pytest.mark.asyncio
async def test_model_stream_closes_client_stream_when_reader_stops_early():
    client_stream = _ClientStream()
    model = _build_model_with_stream(client_stream)

    async with aclosing(model.stream(messages=[])) as chunks:
        async for _ in chunks:
            break

    assert client_stream.closed is True


@pytest.mark.asyncio
async def test_model_stream_closes_client_stream_when_reader_raises():
    client_stream = _ClientStream()
    model = _build_model_with_stream(client_stream)

    with pytest.raises(RuntimeError, match="reader failed"):
        async with aclosing(model.stream(messages=[])) as chunks:
            async for _ in chunks:
                raise RuntimeError("reader failed")

    assert client_stream.closed is True


@pytest.mark.asyncio
async def test_model_stream_read_to_the_end_is_unchanged():
    client_stream = _ClientStream()
    model = _build_model_with_stream(client_stream)

    received = [chunk.content async for chunk in model.stream(messages=[])]

    assert received == ["frame-1", "frame-2"]
    assert client_stream.closed is True


@pytest.mark.asyncio
async def test_model_stream_logs_close_failure_instead_of_raising():
    client_stream = _ClientStream(close_error=ValueError("close failed"))
    model = _build_model_with_stream(client_stream)

    with patch("openjiuwen.core.foundation.llm.model.llm_logger") as logger:
        async with aclosing(model.stream(messages=[])) as chunks:
            async for _ in chunks:
                break

    assert client_stream.closed is True
    assert logger.warning.call_args.args[0] == "Failed to close LLM client stream: %s"
    assert isinstance(logger.warning.call_args.args[1], ValueError)
