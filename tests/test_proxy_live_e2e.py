"""End-to-end proxy test: real ASGI app over the wire, stubbed upstream.

The unit tests call handlers directly. This boots the real FastAPI app with
httpx's ASGI transport and POSTs to /v1/responses and /v1/chat/completions,
so routing, body preparation, protocol translation, and SSE framing are all
exercised the way an agent exercises them.
"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from skillclaw.api_server import SkillClawAPIServer  # noqa: E402
from skillclaw.config import SkillClawConfig  # noqa: E402
from skillclaw.protocols.openai_responses import from_openai_chat_payload  # noqa: E402

STUB_CHAT_BODY = {
    "id": "chatcmpl-live-1",
    "created": 1700000000,
    "model": "upstream-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"ls -la"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}

TEXT_CHAT_BODY = {
    "id": "chatcmpl-live-2",
    "created": 1700000000,
    "model": "upstream-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def _build_app(chat_body: dict) -> SkillClawAPIServer:
    """Create a server whose upstream always returns `chat_body`."""
    config = SkillClawConfig(
        llm_provider="custom",
        llm_api_base="http://upstream.test/v1",
        llm_api_key="k",
        llm_model_id="upstream-model",
        served_model_name="skillclaw-model",
        use_skills=False,
        use_prm=False,
    )
    server = SkillClawAPIServer(config)
    server.app = server._build_app()
    return server


def _stub_upstream(monkeypatch, chat_body: dict):
    """Route only the server's OUTBOUND calls to the stub.

    Patching httpx.AsyncClient wholesale would also capture the test's own
    ASGI client, which is what made the stream assertions see JSON.
    Instead patch the server's forwarding helpers directly.
    """

    def _translated(request_url: str) -> dict:
        if "/chat/completions" in str(request_url):
            return chat_body
        return from_openai_chat_payload(chat_body, "upstream-model")

    class _StubTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_translated(str(request.url)), request=request)

    _RealClient = httpx.AsyncClient

    class _StubClient(_RealClient):
        """Real client, but outbound calls without a transport hit the stub."""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", _StubTransport())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        events.append(json.loads(body))
    return events


@pytest.mark.asyncio
async def test_responses_endpoint_non_stream_returns_full_json(monkeypatch) -> None:
    """Without stream:true the endpoint answers with a complete JSON response."""
    server = _build_app(TEXT_CHAT_BODY)
    _stub_upstream(monkeypatch, TEXT_CHAT_BODY)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "skillclaw-model", "input": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["output_text"] == "Hello world"
    assert data["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


@pytest.mark.asyncio
async def test_responses_endpoint_streams_parseable_sse(monkeypatch) -> None:
    """POST /v1/responses with stream:true must yield well-formed SSE."""
    server = _build_app(TEXT_CHAT_BODY)
    _stub_upstream(monkeypatch, TEXT_CHAT_BODY)

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        # Must use stream(): a plain POST buffers the whole body and reports
        # the transport's content-type rather than the endpoint's.
        async with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "skillclaw-model",
                "input": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200, await response.aread()
            content_type = response.headers.get("content-type", "")
            assert content_type.startswith("text/event-stream"), content_type
            body = "".join([chunk async for chunk in response.aiter_text()])

    events = _sse_events(body)
    types = [e["type"] for e in events]
    assert "response.created" in types, types
    assert "response.completed" in types, types

    # Every output_item.added must be an in-progress shell.
    for event in events:
        if event["type"] == "response.output_item.added":
            item = event["item"]
            assert item.get("status") == "in_progress"
            assert item.get("content", []) == []
            assert item.get("arguments", "") == ""


@pytest.mark.asyncio
async def test_chat_completions_returns_tool_call(monkeypatch) -> None:
    server = _build_app(STUB_CHAT_BODY)
    _stub_upstream(monkeypatch, STUB_CHAT_BODY)

    app = server.app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "list files"}]},
        )

    assert response.status_code == 200, response.text
    data = response.json()
    choice = data["choices"][0]
    calls = choice["message"].get("tool_calls") or []
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "ls -la"}


@pytest.mark.asyncio
async def test_malformed_client_input_does_not_500(monkeypatch) -> None:
    """The 500s fixed in this audit must stay fixed at the HTTP boundary."""
    server = _build_app(TEXT_CHAT_BODY)
    _stub_upstream(monkeypatch, TEXT_CHAT_BODY)

    app = server.app

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
        for body in (
            {"model": "m", "messages": ["bare string"]},
            {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": None}]}]},
            {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": "auto"},
        ):
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code < 500, f"{body} -> {response.status_code}: {response.text[:200]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", [512, 2048, 8192, 100_000])
async def test_truncation_survives_oversized_max_tokens(monkeypatch, max_tokens) -> None:
    """A huge client max_tokens used to drive the prompt budget negative, which
    skipped truncation and forwarded an over-limit prompt for the upstream to
    reject. Truncation must stay active whatever the client asks for.
    """
    import logging

    from skillclaw.api_server import _estimate_openai_body_input_tokens

    logging.disable(logging.CRITICAL)
    try:
        sent: dict = {}
        config = SkillClawConfig(
            llm_provider="custom",
            llm_api_base="http://upstream.test/v1",
            llm_api_key="k",
            llm_model_id="upstream-model",
            served_model_name="skillclaw-model",
            use_skills=False,
            use_prm=False,
            max_context_tokens=2000,
        )
        server = SkillClawAPIServer(config)

        async def _forward(body):
            sent["body"] = body
            return {
                "id": "c",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }

        server._forward_to_llm = _forward
        _stub_upstream(monkeypatch, TEXT_CHAT_BODY)

        messages = [{"role": "system", "content": "sys"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": "word " * 120} for i in range(30)
        ]

        await server._handle_request(
            body={"model": "skillclaw-model", "messages": messages, "max_tokens": max_tokens},
            session_id=f"s-{max_tokens}",
            turn_type="side",
            session_done=False,
        )

        forwarded = sent["body"]
        assert len(forwarded["messages"]) < len(messages), "truncation was skipped entirely"
        assert _estimate_openai_body_input_tokens(forwarded) <= 2000, "forwarded prompt is still over the context limit"
    finally:
        logging.disable(logging.NOTSET)
