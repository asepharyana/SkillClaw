"""Consume SkillClaw's Responses SSE through the real OpenAI SDK accumulator.

The unit tests assert on the emitted events. These assert on what an actual
SDK client *accumulates* — which is what Codex acts on when it decides to run a
tool before `response.completed` arrives. That is the path where the
output_item.added regression shipped doubled tool arguments.
"""

from __future__ import annotations

import json
import typing

import pytest

pytest.importorskip("openai")

from openai._models import construct_type  # noqa: E402
from openai._types import NOT_GIVEN  # noqa: E402
from openai.lib.streaming.responses import ResponseStreamState  # noqa: E402
from openai.types.responses import Response, ResponseStreamEvent  # noqa: E402

from skillclaw.protocols.openai_responses import (  # noqa: E402
    _in_progress_item,
    from_openai_chat_payload,
    stream_response,
)

_EVENT_UNION = typing.get_args(ResponseStreamEvent)[0]


def _drain(payload: dict) -> ResponseStreamState:
    """Feed a chat payload through SkillClaw's stream into the SDK state."""
    import asyncio

    response_payload = from_openai_chat_payload(payload, "skillclaw-model")

    async def collect() -> list[str]:
        return [line async for line in stream_response(response_payload)]

    lines = asyncio.run(collect())

    state = ResponseStreamState(input_tools=[], text_format=NOT_GIVEN)
    for line in lines:
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        raw = json.loads(body)
        if isinstance(raw.get("response"), dict):
            raw["response"] = construct_type(value=raw["response"], type_=Response)
        # handle_event already calls accumulate_event internally; calling both
        # would count every delta twice.
        state.handle_event(construct_type(value=raw, type_=_EVENT_UNION))
    return state


def _tool_payload(*args_spec: tuple[str, str]) -> dict:
    return {
        "id": "chatcmpl-t",
        "created": 1,
        "model": "skillclaw-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}}
                        for i, (n, a) in enumerate(args_spec)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def test_single_tool_call_is_not_doubled() -> None:
    """Pre-fix the accumulator held '{"c":"ls"}{"c":"ls"}', so an agent acting
    on the streamed snapshot would run a malformed command."""
    state = _drain(_tool_payload(("shell", '{"command":"ls -la"}')))
    live = [i for i in state._output_items.values() if i.type == "function_call"]
    assert len(live) == 1
    assert live[0].arguments == '{"command":"ls -la"}'
    json.loads(live[0].arguments)  # must parse on its own


def test_parallel_tool_calls_are_not_doubled() -> None:
    state = _drain(
        _tool_payload(
            ("shell", '{"command":"ls -la /tmp"}'),
            ("read", '{"path":"README.md"}'),
        )
    )
    live = {i.name: i.arguments for i in state._output_items.values() if i.type == "function_call"}
    assert live == {
        "shell": '{"command":"ls -la /tmp"}',
        "read": '{"path":"README.md"}',
    }
    for args in live.values():
        json.loads(args)


def test_text_is_not_doubled() -> None:
    payload = {
        "id": "chatcmpl-x",
        "created": 1,
        "model": "skillclaw-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello world"}, "finish_reason": "stop"}
        ],
    }
    state = _drain(payload)
    texts = [
        part.text
        for item in state._output_items.values()
        if item.type == "message"
        for part in item.content
        if part.type == "output_text" and part.text
    ]
    assert texts == ["Hello world"]


def test_sdk_accepts_every_event() -> None:
    """No event may be rejected by the SDK's own validation."""
    import asyncio

    response_payload = from_openai_chat_payload(
        _tool_payload(("shell", '{"command":"ls"}')), "skillclaw-model"
    )

    async def collect() -> list[str]:
        return [line async for line in stream_response(response_payload)]

    state = ResponseStreamState(input_tools=[], text_format=NOT_GIVEN)
    seen: list[str] = []
    for line in asyncio.run(collect()):
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if not body or body == "[DONE]":
            continue
        raw = json.loads(body)
        if isinstance(raw.get("response"), dict):
            raw["response"] = construct_type(value=raw["response"], type_=Response)
        event = construct_type(value=raw, type_=_EVENT_UNION)
        state.handle_event(event)  # raises if the SDK rejects it
        seen.append(event.type)

    assert "response.created" in seen
    assert "response.completed" in seen
    assert "response.output_item.added" in seen
    assert "response.function_call_arguments.delta" in seen


def test_in_progress_item_carries_no_content() -> None:
    """A client appends deltas to the item from .added; a pre-populated one
    doubles everything."""
    message = {
        "type": "message",
        "id": "m1",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi"}],
        "status": "completed",
    }
    call = {
        "type": "function_call",
        "id": "f1",
        "call_id": "c1",
        "name": "shell",
        "arguments": '{"command":"ls"}',
        "status": "completed",
    }
    assert _in_progress_item(message)["content"] == []
    assert _in_progress_item(call)["arguments"] == ""
    for item in (message, call):
        assert _in_progress_item(item)["status"] == "in_progress"
