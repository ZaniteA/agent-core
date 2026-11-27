# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the shared HTTP clients behind Responses mode (API key and OpenAI account clients)."""

from contextlib import aclosing
from unittest.mock import MagicMock, patch

import httpx
import pytest

from openjiuwen.core.foundation.llm import (
    Model,
    ModelClientConfig,
    ModelRequestConfig,
    ProviderType,
    UserMessage,
)
from openjiuwen.core.foundation.llm.model_clients.openai_account_model_client import (
    DEFAULT_OPENAI_ACCOUNT_BASE_URL,
    OpenAIAccountModelClient,
)
from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient
from openjiuwen.core.foundation.llm.utils.responses_transport import OpenAIAccountResponsesTransport

_DELTA = b'event: response.output_text.delta\ndata: {"delta":"ok"}\n\n'
_COMPLETED = (
    b"event: response.completed\n"
    b'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
)


class _TrackedBody(httpx.AsyncByteStream):
    """SSE body that records whether the client closed it."""

    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield _DELTA
        yield _DELTA
        yield _COMPLETED

    async def aclose(self) -> None:
        self.closed = True


class _Endpoint:
    """Fake Responses endpoint that also builds the HTTP clients the transports ask for."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.bodies: list[_TrackedBody] = []
        self.clients: list[httpx.AsyncClient] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = _TrackedBody()
        self.bodies.append(body)
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    def make_client(self) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        self.clients.append(client)
        return client

    def serve(self):
        """Patch OpenAIAccountResponsesTransport so every client it builds talks to this endpoint."""
        return patch.object(OpenAIAccountResponsesTransport, "_make_client", self.make_client)

    def read_timeouts(self) -> list[float]:
        return [request.extensions["timeout"]["read"] for request in self.requests]


def _client_config(use_shared: bool = True, api_key: str = "sk-test") -> ModelClientConfig:
    return ModelClientConfig(
        client_provider=ProviderType.OpenAI,
        api_key=api_key,
        api_base="https://api.openai.com/v1",
        verify_ssl=False,
        api_mode="responses",
        use_shared_llm_http_client=use_shared,
    )


def _make_model(use_shared: bool = True, api_key: str = "sk-test") -> Model:
    return Model(
        model_client_config=_client_config(use_shared=use_shared, api_key=api_key),
        model_config=ModelRequestConfig(model="gpt-5.4-mini"),
    )


class _FakeAuthManager:
    def resolve_access_token(self, *, force_refresh: bool = False) -> str:
        return "access-token"


def _account_config(use_shared: bool = True) -> ModelClientConfig:
    return ModelClientConfig(
        client_provider=ProviderType.OpenAIAccount,
        api_key="",
        api_base=DEFAULT_OPENAI_ACCOUNT_BASE_URL,
        verify_ssl=False,
        use_shared_llm_http_client=use_shared,
        openai_account_auth_manager=_FakeAuthManager(),
    )


def _make_account_client(use_shared: bool = True) -> OpenAIAccountModelClient:
    return OpenAIAccountModelClient(ModelRequestConfig(model="gpt-5.4-mini"), _account_config(use_shared))


async def _read_all(stream) -> None:
    async for _ in stream:
        pass


class TestSharedClientPooling:
    @pytest.mark.asyncio
    async def test_shared_client_is_built_once_and_reused_without_hot_path_close(self):
        model = _make_model()
        endpoint = _Endpoint()

        with endpoint.serve():
            await model.invoke(messages=[UserMessage(content="hi")])
            await model.invoke(messages=[UserMessage(content="hi again")])
            await _read_all(model.stream(messages=[UserMessage(content="stream")]))

        # Built once, reused for the next calls.
        assert len(endpoint.clients) == 1
        assert len(endpoint.requests) == 3
        # Never closed on the hot path.
        assert endpoint.clients[0].is_closed is False
        key = OpenAIModelClient.connection_key(model.model_client_config)
        assert OpenAIModelClient._responses_client_cache == {key: endpoint.clients[0]}

    @pytest.mark.asyncio
    async def test_per_request_timeout_applied_without_rebuilding_client(self):
        model = _make_model()
        endpoint = _Endpoint()

        with endpoint.serve():
            await model.invoke(messages=[UserMessage(content="hi")], timeout=12.5)
            await model.invoke(messages=[UserMessage(content="hi")], timeout=30)

        assert len(endpoint.clients) == 1
        assert endpoint.read_timeouts() == [12.5, 30.0]

    @pytest.mark.asyncio
    async def test_different_api_keys_never_share_a_client(self):
        endpoint = _Endpoint()

        with endpoint.serve():
            await _make_model(api_key="sk-tenant-a").invoke(messages=[UserMessage(content="hi")])
            await _make_model(api_key="sk-tenant-b").invoke(messages=[UserMessage(content="hi")])

        assert len(endpoint.clients) == 2
        assert len(OpenAIModelClient._responses_client_cache) == 2

    @pytest.mark.asyncio
    async def test_stream_closed_early_closes_response_but_keeps_shared_client(self):
        model = _make_model()
        endpoint = _Endpoint()

        with endpoint.serve():
            async with aclosing(model.stream(messages=[UserMessage(content="hi")])) as chunks:
                async for _ in chunks:
                    break
            # Closed at once, not when the abandoned stream is garbage collected.
            assert endpoint.bodies[0].closed is True
            assert endpoint.clients[0].is_closed is False

            await model.invoke(messages=[UserMessage(content="hi again")])

        assert len(endpoint.clients) == 1

    def test_built_client_keeps_idle_connections_for_60_seconds(self):
        with patch(
            "openjiuwen.core.foundation.llm.utils.responses_transport.httpx.AsyncHTTPTransport",
            return_value=MagicMock(),
        ) as http_transport_cls:
            OpenAIAccountResponsesTransport()._make_client()

        assert http_transport_cls.call_args.kwargs["limits"].keepalive_expiry == 60.0

    @pytest.mark.asyncio
    async def test_aclose_closes_cached_clients_and_clears_cache(self):
        model = _make_model()
        endpoint = _Endpoint()

        with endpoint.serve():
            await model.invoke(messages=[UserMessage(content="hi")])

        assert OpenAIModelClient._responses_client_cache

        await OpenAIModelClient.aclose()

        assert endpoint.clients[0].is_closed is True
        assert OpenAIModelClient._responses_client_cache == {}


class TestAcloseConnections:
    @pytest.mark.asyncio
    async def test_aclose_connections_closes_only_given(self):
        model_keep = _make_model(api_key="sk-keep")
        model_drop = _make_model(api_key="sk-drop")
        endpoint = _Endpoint()

        with endpoint.serve():
            await model_keep.invoke(messages=[UserMessage(content="hi")])
            await model_drop.invoke(messages=[UserMessage(content="hi")])

        await OpenAIModelClient.aclose_connections([model_drop.model_client_config])

        # Only the targeted connection is closed; the untouched one stays.
        keep_client, drop_client = endpoint.clients
        assert drop_client.is_closed is True
        assert keep_client.is_closed is False
        assert OpenAIModelClient._responses_client_cache == {
            OpenAIModelClient.connection_key(model_keep.model_client_config): keep_client
        }


class TestFallbackClient:
    @pytest.mark.asyncio
    async def test_fallback_builds_and_closes_client_per_call(self):
        model = _make_model(use_shared=False)
        endpoint = _Endpoint()

        with endpoint.serve():
            await model.invoke(messages=[UserMessage(content="hi")])
            await _read_all(model.stream(messages=[UserMessage(content="hi again")]))

        # A fresh client per call, each closed after use.
        assert len(endpoint.clients) == 2
        assert all(client.is_closed for client in endpoint.clients)
        # Shared cache stays empty in fallback mode.
        assert OpenAIModelClient._responses_client_cache == {}

    @pytest.mark.asyncio
    async def test_request_is_the_same_with_and_without_shared_client(self):
        endpoint = _Endpoint()

        with endpoint.serve():
            await _make_model(use_shared=True).invoke(messages=[UserMessage(content="hi")], timeout=12.5)
            await _make_model(use_shared=False).invoke(messages=[UserMessage(content="hi")], timeout=12.5)

        shared, fallback = endpoint.requests
        assert shared.url == fallback.url
        assert shared.content == fallback.content
        assert dict(shared.headers) == dict(fallback.headers)
        assert endpoint.read_timeouts() == [12.5, 12.5]


class TestOpenAIAccountClientPooling:
    @pytest.mark.asyncio
    async def test_one_shared_client_serves_calls_with_any_timeout(self):
        client = _make_account_client()
        endpoint = _Endpoint()

        with endpoint.serve():
            await client.invoke("hi")
            # A timeout other than the config's used to build a new client that was never closed.
            await client.invoke("hi", timeout=7.5)
            await _read_all(client.stream("hi", timeout=7.5))

        assert len(endpoint.clients) == 1
        assert endpoint.clients[0].is_closed is False
        assert endpoint.read_timeouts() == [client.model_client_config.timeout, 7.5, 7.5]
        key = OpenAIAccountModelClient.connection_key(client.model_client_config)
        assert OpenAIAccountModelClient._responses_client_cache == {key: endpoint.clients[0]}

    @pytest.mark.asyncio
    async def test_stream_closed_early_closes_response_but_keeps_shared_client(self):
        client = _make_account_client()
        endpoint = _Endpoint()

        with endpoint.serve():
            async with aclosing(client.stream("hi")) as chunks:
                async for _ in chunks:
                    break

        assert endpoint.bodies[0].closed is True
        assert endpoint.clients[0].is_closed is False

    @pytest.mark.asyncio
    async def test_aclose_and_aclose_connections_close_cached_clients(self):
        client = _make_account_client()
        other_config = _account_config().model_copy(update={"api_base": "https://other.example/backend-api"})
        other = OpenAIAccountModelClient(ModelRequestConfig(model="gpt-5.4-mini"), other_config)
        endpoint = _Endpoint()

        with endpoint.serve():
            await client.invoke("hi")
            await other.invoke("hi")

        await OpenAIAccountModelClient.aclose_connections([other_config])

        assert [c.is_closed for c in endpoint.clients] == [False, True]
        assert list(OpenAIAccountModelClient._responses_client_cache) == [
            OpenAIAccountModelClient.connection_key(client.model_client_config)
        ]

        await OpenAIAccountModelClient.aclose()

        assert all(c.is_closed for c in endpoint.clients)
        assert OpenAIAccountModelClient._responses_client_cache == {}

    @pytest.mark.asyncio
    async def test_fallback_builds_and_closes_client_per_call(self):
        client = _make_account_client(use_shared=False)
        endpoint = _Endpoint()

        with endpoint.serve():
            await client.invoke("hi")
            await _read_all(client.stream("hi"))

        assert len(endpoint.clients) == 2
        assert all(c.is_closed for c in endpoint.clients)
        assert OpenAIAccountModelClient._responses_client_cache == {}
