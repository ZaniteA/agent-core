# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""HTTP transport shared by API key and account Responses clients.

This module is responsible only for HTTP I/O: sending requests and
iterating SSE events. All request-body construction and response
parsing is handled by :mod:`responses_utils`.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Hashable, MutableMapping, Optional, Union

import httpx

from openjiuwen.core.common.logging import llm_logger as logger
from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
from openjiuwen.core.foundation.llm.schema.message_chunk import AssistantMessageChunk
from openjiuwen.core.foundation.llm.utils.request_parameter_fallback import RequestParameterFallback
from openjiuwen.core.runner.callback import trigger
from openjiuwen.core.runner.callback.events import LLMCallEvents
from openjiuwen.extensions.external_provider.openai_auth.openai_account_auth import DEFAULT_OPENAI_ACCOUNT_BASE_URL
from openjiuwen.core.foundation.llm.utils.responses_utils import (
    build_headers,
    message_from_stream_chunk,
    OpenAIAccountResponsesError,
    parse_sse_block,
    ResponsesStreamParser,
    raise_for_http_error,
)


class OpenAIAccountResponsesTransport:
    """HTTP-only transport for the OpenAI account Responses API.

    Handles streaming, SSE iteration, and auth headers.  Format
    conversion (messages → input items, tools, parsing response
    payloads) lives in :mod:`responses_utils`.

    By default the transport owns one HTTP client and closes it in
    :meth:`aclose`. With ``shared_clients`` it borrows a long-lived client
    from that mapping instead (one per ``shared_client_key``) and never
    closes it: the mapping's owner does. ``timeout_seconds`` is applied to
    each request, so transports with different timeouts can share a client.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OPENAI_ACCOUNT_BASE_URL,
        timeout_seconds: float = 60.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        verify: Union[bool, str, Any] = True,
        proxy: Optional[str] = None,
        max_retries: int = 0,
        request_fallback: Optional[RequestParameterFallback] = None,
        shared_clients: Optional[MutableMapping[Hashable, httpx.AsyncClient]] = None,
        shared_client_key: Hashable = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._verify = verify
        self._proxy = proxy
        self._max_retries = max(0, int(max_retries or 0))
        self._request_fallback = request_fallback or RequestParameterFallback(
            "OpenAI", "OpenAI Responses", retry_without_stop=True,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._shared_clients = shared_clients
        self._shared_client_key = shared_client_key

    async def create_response(
        self,
        *,
        body: dict[str, Any],
        access_token: str,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> AssistantMessage:
        """Send a request and accumulate the streamed chunks into a single message."""
        final_chunk: Optional[AssistantMessageChunk] = None
        async for chunk in self.stream_response(
            body=body,
            access_token=access_token,
            model_name=model_name,
            session_id=session_id,
            extra_headers=extra_headers,
        ):
            final_chunk = final_chunk + chunk if final_chunk else chunk
        return message_from_stream_chunk(final_chunk)

    async def stream_response(
        self,
        *,
        body: dict[str, Any],
        access_token: str,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[AssistantMessageChunk]:
        """Stream response chunks from the Responses API."""
        stream_body = dict(body)
        stream_body["stream"] = True
        effective_model = model_name or str(body.get("model") or "")
        client = await self._get_client()
        headers = build_headers(access_token=access_token, session_id=session_id, extra_headers=extra_headers)
        async with AsyncExitStack() as connection:
            async def open_response(**request_body) -> httpx.Response:
                response = await connection.enter_async_context(
                    client.stream(
                        "POST",
                        self._responses_url(),
                        json=request_body,
                        headers=headers,
                        timeout=self._request_timeout(),
                    )
                )
                if response.status_code >= 400:
                    await response.aread()
                    try:
                        raise_for_http_error(response)
                    except OpenAIAccountResponsesError:
                        await connection.aclose()
                        raise
                return response

            # Retry only while opening the response. SSE events belong to the accepted request.
            response = await self._request_fallback.call(open_response, stream_body, is_stream=True)
            # httpx transport retries only cover failed connects and are not
            # reported back, so the retry count is unknown here.
            await trigger(
                LLMCallEvents.LLM_RESPONSE_STARTED,
                model_name=effective_model,
                retry_count=None,
            )
            parser = ResponsesStreamParser(model_name=effective_model)
            buffer: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    event = parse_sse_block(buffer)
                    buffer = []
                    chunk = parser.parse(event)
                    if chunk:
                        yield chunk
                else:
                    buffer.append(line)
            if buffer:
                event = parse_sse_block(buffer)
                chunk = parser.parse(event)
                if chunk:
                    yield chunk

    async def aclose(self) -> None:
        """Close the HTTP client this transport owns, if one has been created.

        A borrowed shared client is left open for the other transports using it.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _responses_url(self) -> str:
        return f"{self.base_url}/responses"

    def _request_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(max(5.0, float(self.timeout_seconds)))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._shared_clients is not None:
            return self._get_shared_client()
        if self._client is None:
            self._client = self._make_client()
        elif self._client.is_closed:
            old_client = self._client
            self._client = None
            try:
                await old_client.aclose()
            except Exception as exc:
                logger.warning(
                    "Failed to close stale OpenAI account Responses client before recreating it: %s",
                    exc,
                )
            self._client = self._make_client()
        return self._client

    def _get_shared_client(self) -> httpx.AsyncClient:
        # Fully synchronous (no ``await``), so under one asyncio event loop the
        # get/build/set below is atomic and needs no lock.
        client = self._shared_clients.get(self._shared_client_key)
        if client is None or client.is_closed:
            client = self._make_client()
            self._shared_clients[self._shared_client_key] = client
            logger.info("Created shared long-lived Responses HTTP client.")
        return client

    def _make_client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {"timeout": self._request_timeout()}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        else:
            kwargs["transport"] = httpx.AsyncHTTPTransport(
                verify=self._verify,
                proxy=self._proxy,
                retries=self._max_retries,
                # Same pool as the Chat client: httpx's default 5s keep-alive
                # drops idle connections between calls spaced further apart.
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=60.0,
                ),
            )
        return httpx.AsyncClient(**kwargs)
