"""Regression tests for the behaviour defects found in the 2026-09 audit.

Each test here reproduced a real failure before the fix:

* a chat SSE ``tool_calls`` delta without ``index`` crashed the openai SDK
  mid-stream (``TypeError: list indices must be integers``);
* deterministic upstream 4xx (400/401) were retried 6 times, burning ~33s per
  request that could never succeed;
* a ``tool_result`` with no ``tool_use_id`` became ``role:tool`` with an empty
  ``tool_call_id``, and a conversation ending on an unanswered ``tool_use``
  was rejected upstream — both retried into a 502;
* the chat→Responses bridge dropped ``tools`` and every ``function_call``;
* ``previous_response_id`` was accepted from any session, leaking one
  session's prompt into another's upstream call;
* the compressed system prompt was cached in one global file, overwriting the
  system prompt of every other session (including across restarts);
* native ``/v1/responses`` did no truncation at all;
* TUI sessions were keyed by model only, so two clients shared one session;
* a 4xx retry loop with a large ``max_tokens`` silently disabled truncation.

The stub upstream runs in ``strict`` mode: it rejects the exact message shapes
that real OpenAI-compatible upstreams reject with 400 invalid_request_error, so
these tests assert behaviour rather than status codes.
"""

from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from skillclaw.api_server import (  # noqa: E402
    SkillClawAPIServer,
    _deduplicate_tool_calls,
    _sanitize_forward_messages,
    _truncate_responses_input,
)
from skillclaw.config import SkillClawConfig  # noqa: E402



# --------------------------------------------------------------------- stubs


def _chat_body(*, tool_calls: list[dict] | None = None, content: str = "ok") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-up-1",
        "created": 1700000000,
        "model": "upstream-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


class _Recorder:
    """Captures every body the server sends upstream."""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls = 0
        self.bodies: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, request: "httpx.Request") -> "httpx.Response":
        self.calls += 1
        self.urls.append(str(request.url))
        self.bodies.append(json.loads(request.content.decode() or "{}"))
        return self.responder(request, self.calls)


def _install(monkeypatch, recorder: _Recorder) -> None:
    class _StubTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: "httpx.Request") -> "httpx.Response":
            return recorder(request)

    real = httpx.AsyncClient

    class _StubClient(real):  # type: ignore[misc, valid-type]
        """Real client, so the test's own ASGI client still works; only
        outbound calls without an explicit transport reach the stub."""

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", _StubTransport())
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)


def _ok_responder(body: dict):
    def _respond(request: "httpx.Request", _calls: int) -> "httpx.Response":
        return httpx.Response(200, json=body, request=request)

    return _respond


def _strict_chat_responder(body: dict):
    """Mimic a real upstream: reject the three deterministic-400 shapes."""

    def _respond(request: "httpx.Request", _calls: int) -> "httpx.Response":
        payload = json.loads(request.content.decode() or "{}")
        for message in payload.get("messages") or []:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "tool" and not str(message.get("tool_call_id") or "").strip():
                return httpx.Response(400, json=_invalid("tool message without tool_call_id"), request=request)
        issued = {
            call["id"]
            for message in payload.get("messages") or []
            if isinstance(message, dict)
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("id")
        }
        for message in payload.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "tool":
                if str(message.get("tool_call_id") or "") not in issued:
                    return httpx.Response(400, json=_invalid("orphan tool message"), request=request)
        messages = [m for m in payload.get("messages") or [] if isinstance(m, dict)]
        if messages and _ends_on_unanswered_tool_call(messages):
            return httpx.Response(400, json=_invalid("unanswered tool_call"), request=request)
        if messages and not any(m.get("role") == "user" for m in messages):
            return httpx.Response(400, json=_invalid("no user turn"), request=request)
        return httpx.Response(200, json=body, request=request)

    return _respond


def _invalid(message: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": "messages",
            "code": None,
        }
    }


def _ends_on_unanswered_tool_call(messages: list[dict]) -> bool:
    last = messages[-1]
    calls = [c for c in (last.get("tool_calls") or []) if isinstance(c, dict) and c.get("id")]
    if not calls:
        return False
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    return any(c["id"] not in answered for c in calls)


def _server(**overrides) -> SkillClawAPIServer:
    config = SkillClawConfig(
        llm_provider="custom",
        llm_api_base="http://upstream.test/v1",
        llm_api_key="k",
        llm_model_id="upstream-model",
        served_model_name="skillclaw-model",
        use_skills=False,
        use_prm=False,
        **overrides,
    )
    server = SkillClawAPIServer(config)
    server.app = server._build_app()
    return server


# ------------------------------------------------------- 1. streaming index


@pytest.mark.asyncio
async def test_stream_tool_call_delta_carries_index(monkeypatch) -> None:
    """A message-shaped tool_calls list must be re-keyed for streaming.

    Before the fix the delta was forwarded without ``index`` and the openai
    SDK raised TypeError: list indices must be integers, not NoneType.
    """
    tool_calls = [{"id": "call_a", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]
    server = _server()
    recorder = _Recorder(_ok_responder(_chat_body(tool_calls=tool_calls)))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        chunks: list[dict] = []
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            body = "".join([part async for part in response.aiter_text()])
    for line in body.splitlines():
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
            chunks.append(json.loads(line[5:].strip()))

    deltas = [
        call
        for chunk in chunks
        for choice in chunk.get("choices") or []
        for call in ((choice.get("delta") or {}).get("tool_calls") or [])
    ]
    assert deltas, "tool_calls never reached the client in the stream"
    for call in deltas:
        assert isinstance(call.get("index"), int), f"delta tool_call has no index: {call}"


# ------------------------------------------------------- 2. retry on 4xx


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_deterministic_4xx_is_not_retried(monkeypatch, status: int) -> None:
    """A deterministic 4xx must fail once, not six times over ~33 seconds."""
    server = _server()
    attempts = {"n": 0}

    def _respond(request: "httpx.Request", _calls: int) -> "httpx.Response":
        attempts["n"] += 1
        return httpx.Response(status, json=_invalid("nope"), request=request)

    _install(monkeypatch, _Recorder(_respond))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == status, "the client's own 400 must be surfaced, not masked as 502"
    assert attempts["n"] == 1, f"deterministic {status} was retried {attempts['n']} times"


# --------------------------------------- 3. anthropic orphan/dangling shapes


@pytest.mark.asyncio
async def test_anthropic_tool_result_without_id_is_not_rejected(monkeypatch) -> None:
    """A ``tool_result`` with no ``tool_use_id`` must not become a 400/502."""
    server = _server()
    recorder = _Recorder(_strict_chat_responder(_chat_body()))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "skillclaw-model",
                "max_tokens": 128,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "run it"},
                            {"type": "tool_result", "content": "ok"},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200, f"got {response.status_code}: {response.text[:200]}"
    assert recorder.calls == 1, f"retried {recorder.calls} times"


@pytest.mark.asyncio
async def test_conversation_ending_on_tool_use_is_completed(monkeypatch) -> None:
    """A conversation ending on an unanswered tool_use must still forward."""
    server = _server()
    recorder = _Recorder(_strict_chat_responder(_chat_body()))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "skillclaw-model",
                "max_tokens": 128,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "list files"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_1", "name": "shell", "input": {"cmd": "ls"}}],
                    },
                ],
            },
        )

    assert response.status_code == 200, f"got {response.status_code}: {response.text[:200]}"
    forwarded = recorder.bodies[-1]["messages"]
    assert any(m.get("role") == "user" for m in forwarded), "forwarded body lost its user turn"


# ------------------------------------------------- 4. chat->Responses bridge


@pytest.mark.asyncio
async def test_bridge_forwards_tools_and_returns_tool_calls(monkeypatch) -> None:
    """tools must reach the Responses upstream and tool calls must come back."""
    server = _server(llm_api_mode="responses")
    upstream = {
        "id": "resp_1",
        "created_at": 1,
        "model": "upstream-model",
        "status": "completed",
        "output": [
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_a",
                "name": "shell",
                "arguments": '{"cmd":"ls"}',
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    recorder = _Recorder(_ok_responder(upstream))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "skillclaw-model",
                "messages": [{"role": "user", "content": "list files"}],
                "tools": [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}],
            },
        )

    assert recorder.bodies[-1].get("tools"), "tools were dropped before reaching the Responses upstream"
    message = response.json()["choices"][0]["message"]
    assert message.get("tool_calls"), f"tool call was dropped on the way back: {message}"
    assert message["tool_calls"][0]["function"]["name"] == "shell"
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_bridge_reports_truncation_honestly(monkeypatch) -> None:
    """A length-capped Responses result must not be reported as a clean stop."""
    server = _server(llm_api_mode="responses")
    upstream = {
        "id": "resp_2",
        "created_at": 1,
        "model": "upstream-model",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"id": "m1", "type": "message", "content": [{"type": "output_text", "text": "partial"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    _install(monkeypatch, _Recorder(_ok_responder(upstream)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.json()["choices"][0]["finish_reason"] == "length"


# ------------------------------------------------- 5. cross-session leakage


@pytest.mark.asyncio
async def test_previous_response_id_is_scoped_to_its_session(monkeypatch) -> None:
    """Session B must not be able to continue session A's response."""
    server = _server()
    recorder = _Recorder(_ok_responder(_chat_body()))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        first = await client.post(
            "/v1/responses",
            headers={"X-Session-Id": "SESSION-A"},
            json={"model": "skillclaw-model", "input": [{"role": "user", "content": "ALICE-SECRET"}]},
        )
        response_id = first.json().get("id")
        before = recorder.calls
        second = await client.post(
            "/v1/responses",
            headers={"X-Session-Id": "SESSION-B"},
            json={
                "model": "skillclaw-model",
                "previous_response_id": response_id,
                "input": [{"role": "user", "content": "BOB-UNRELATED"}],
            },
        )

    assert second.status_code in (400, 404), f"cross-session continuation was allowed: {second.status_code}"
    leaked = [body for body in recorder.bodies[before:] if "ALICE-SECRET" in json.dumps(body)]
    assert not leaked, "session A's prompt was forwarded on session B's request"


@pytest.mark.asyncio
async def test_response_lookup_is_scoped_to_its_session(monkeypatch) -> None:
    """GET /v1/responses/{id} must refuse a foreign session."""
    server = _server()
    _install(monkeypatch, _Recorder(_ok_responder(_chat_body())))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        first = await client.post(
            "/v1/responses",
            headers={"X-Session-Id": "SESSION-A"},
            json={"model": "skillclaw-model", "input": [{"role": "user", "content": "hi"}]},
        )
        response_id = first.json().get("id")
        foreign = await client.get(
            f"/v1/responses/{response_id}",
            headers={"X-Session-Id": "SESSION-B"},
        )

    assert foreign.status_code in (403, 404), f"foreign session read the response: {foreign.status_code}"


# ------------------------------------- 6. per-session system prompt caching


@pytest.mark.asyncio
async def test_compressed_prompt_cache_is_not_shared_across_sessions(monkeypatch) -> None:
    """One session's compressed prompt must not overwrite another's."""
    server = _server()
    recorder = _Recorder(_ok_responder(_chat_body()))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        await client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "SESSION-A"},
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "alpha"}]},
        )
        await client.post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "SESSION-B"},
            json={"model": "skillclaw-model", "messages": [{"role": "user", "content": "beta"}]},
        )

    first_system = _system_text(recorder.bodies[0])
    second_system = _system_text(recorder.bodies[1])
    if first_system and second_system:
        assert first_system != second_system, "both sessions were served the same cached system prompt"


def _system_text(body: dict) -> str:
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            return json.dumps(message.get("content"), sort_keys=True, ensure_ascii=False)
    return ""


# ------------------------------------------------- 7. native responses budget


def test_native_responses_input_is_truncated_to_budget() -> None:
    """The native Responses path had no truncation and forwarded everything."""
    items: list[dict] = []
    for index in range(40):
        items.append({"role": "user", "content": f"old question {index} " + "x" * 4000})
        items.append(
            {
                "type": "function_call",
                "call_id": f"call_{index}",
                "name": "shell",
                "arguments": "{}",
            }
        )
        items.append(
            {
                "type": "function_call_output",
                "call_id": f"call_{index}",
                "output": "y" * 2000,
            }
        )
    items.append({"role": "user", "content": "newest question"})

    trimmed = _truncate_responses_input(items, 3000)

    assert len(trimmed) < len(items), "nothing was dropped despite a 3000-token budget"
    assert trimmed[-1] == items[-1], "the newest item must be kept"
    call_ids = {i.get("call_id") for i in trimmed if isinstance(i, dict) and i.get("type") == "function_call"}
    for item in trimmed:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            assert item.get("call_id") in call_ids, "truncation orphaned a function_call_output"


def test_native_responses_truncation_keeps_a_user_turn() -> None:
    items = [
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": "another answer " + "z" * 5000},
    ]
    trimmed = _truncate_responses_input(items, 500)
    assert any(isinstance(i, dict) and i.get("role") == "user" for i in trimmed)


# ---------------------------------------------------- 8. TUI session keying


@pytest.mark.asyncio
async def test_two_tui_clients_do_not_share_a_session(monkeypatch) -> None:
    """Without a client id, two TUI users on one model shared one session."""
    server = _server()
    recorder = _Recorder(_ok_responder(_chat_body()))
    _install(monkeypatch, recorder)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        for authorization in ("Bearer key-alice", "Bearer key-bob"):
            await client.post(
                "/v1/responses",
                headers={"Authorization": authorization},
                json={"model": "skillclaw-model", "input": [{"role": "user", "content": "hi"}]},
            )

    sessions = [b.get("session_id") for b in recorder.bodies]
    assert sessions[0] != sessions[1], f"both clients shared session {sessions[0]!r}"


# ------------------------------------------- 9. large max_tokens vs budget


@pytest.mark.asyncio
async def test_huge_max_tokens_does_not_disable_truncation(monkeypatch) -> None:
    """A large client max_tokens used to drive the prompt budget to zero."""
    server = _server(max_context_tokens=8000)
    recorder = _Recorder(_strict_chat_responder(_chat_body()))
    _install(monkeypatch, recorder)

    filler = [{"role": "user", "content": f"message {i} " + "w" * 6000} for i in range(6)]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "skillclaw-model",
                "messages": filler + [{"role": "user", "content": "newest"}],
                "max_tokens": 100_000,
            },
        )

    assert response.status_code == 200, f"got {response.status_code}: {response.text[:200]}"
    forwarded = recorder.bodies[-1]["messages"]
    assert len(forwarded) < len(filler) + 1, "an over-budget prompt was forwarded verbatim"


# --------------------------------------------------- 10. helper invariants


def test_sanitize_drops_orphan_and_idless_tool_results() -> None:
    cleaned = _sanitize_forward_messages(
        [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "", "content": "orphan"},
            {"role": "tool", "tool_call_id": "call_missing", "content": "orphan"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ]
    )
    ids = [m.get("tool_call_id") for m in cleaned if m.get("role") == "tool"]
    assert ids == ["call_1"], f"unexpected tool results: {ids}"


def test_sanitize_completes_a_dangling_tool_call() -> None:
    cleaned = _sanitize_forward_messages(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
        ]
    )
    assert not _ends_on_unanswered_tool_call(cleaned), "left a dangling tool_call on the wire"


def test_sanitize_guarantees_a_user_turn() -> None:
    cleaned = _sanitize_forward_messages([{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}])
    assert any(m.get("role") == "user" for m in cleaned)


def test_dedup_keeps_distinct_idless_parallel_calls() -> None:
    """Falling back to (name, args) merged two genuinely separate calls."""
    calls = [
        {"function": {"name": "read", "arguments": '{"path":"a"}'}},
        {"function": {"name": "read", "arguments": '{"path":"b"}'}},
        {"id": "call_1", "function": {"name": "f", "arguments": "{}"}},
        {"id": "call_1", "function": {"name": "f", "arguments": "{}"}},
    ]
    deduped = _deduplicate_tool_calls(calls)
    assert len(deduped) == 3, f"merged distinct calls: {deduped}"
