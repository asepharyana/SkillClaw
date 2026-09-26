"""Tool results must survive the forward sanitizer intact.

The sanitizer runs on every request right before it is forwarded upstream. It
reorders tool results after the turn that issued them, and it drops results
whose issuer is absent. Both steps used to consume healthy traffic: the
reorder dropped *every* result, because it only flushed a buffered result when
it saw its issuer, and in a well-formed conversation the issuer has already
been emitted by the time the result arrives. The result was then replaced by
the "client ended the turn" placeholder, so the model saw an empty tool result
and retried the same call forever.
"""

import pytest

from skillclaw.api_server import (
    _hoist_tool_results_after_assistant,
    _repair_orphan_tool_results,
    _sanitize_forward_messages,
)


def _tool_call(call_id: str, name: str = "exec") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _round(output: str = "file-a\nfile-b") -> list[dict]:
    """A user turn, one tool call, and its result -- the healthy shape."""
    return [
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_1")]},
        {"role": "tool", "tool_call_id": "call_1", "content": output},
    ]


def _tool_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "tool"]


def test_hoister_keeps_a_result_that_already_follows_its_issuer():
    messages = _round() + [{"role": "user", "content": "and now?"}]

    hoisted = _hoist_tool_results_after_assistant(messages)

    assert _tool_contents(hoisted) == ["file-a\nfile-b"]


def test_hoister_moves_a_result_that_precedes_its_issuer():
    messages = [
        {"role": "user", "content": "list the files"},
        {"role": "tool", "tool_call_id": "call_1", "content": "file-a\nfile-b"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_1")]},
    ]

    hoisted = _hoist_tool_results_after_assistant(messages)

    assert [m["role"] for m in hoisted] == ["user", "assistant", "tool"]


def test_hoister_keeps_every_result_of_a_parallel_round():
    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1", "exec"), _tool_call("call_2", "read")],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "second"},
        {"role": "tool", "tool_call_id": "call_1", "content": "first"},
    ]

    hoisted = _hoist_tool_results_after_assistant(messages)

    assert sorted(_tool_contents(hoisted)) == ["first", "second"]


def test_hoister_keeps_a_trailing_result():
    messages = _round("done")

    assert _tool_contents(_hoist_tool_results_after_assistant(messages)) == ["done"]


def test_sanitize_forwards_the_real_result_not_a_placeholder():
    messages = _round() + [{"role": "user", "content": "and now?"}]

    forwarded = _sanitize_forward_messages(messages)

    assert _tool_contents(forwarded) == ["file-a\nfile-b"]


def test_sanitize_does_not_synthesize_a_result_that_was_supplied():
    messages = _round("real output") + [{"role": "user", "content": "and now?"}]

    forwarded = _sanitize_forward_messages(messages)

    assert not any("tool result unavailable" in str(m.get("content")) for m in forwarded)


@pytest.mark.parametrize(
    "output",
    ["", "0", "null", "[]", "plain text", "trailing newline\n"],
)
def test_sanitize_preserves_whatever_the_tool_returned(output):
    messages = _round(output) + [{"role": "user", "content": "and now?"}]

    forwarded = _sanitize_forward_messages(messages)

    assert _tool_contents(forwarded) == [output]


def test_sanitize_drops_a_result_with_no_issuer():
    messages = [
        {"role": "user", "content": "list the files"},
        {"role": "tool", "tool_call_id": "call_ghost", "content": "unissued"},
        {"role": "user", "content": "and now?"},
    ]

    forwarded = _sanitize_forward_messages(messages)

    assert _tool_contents(forwarded) == []


def test_sanitize_drops_a_result_with_no_id():
    messages = [
        {"role": "user", "content": "list the files"},
        {"role": "tool", "tool_call_id": "", "content": "unidentified"},
        {"role": "user", "content": "and now?"},
    ]

    forwarded = _sanitize_forward_messages(messages)

    assert _tool_contents(forwarded) == []


def test_sanitize_keeps_two_consecutive_rounds():
    messages = _round("first") + [
        {"role": "assistant", "content": "and again", "tool_calls": [_tool_call("call_2", "read")]},
        {"role": "tool", "tool_call_id": "call_2", "content": "second"},
        {"role": "user", "content": "and now?"},
    ]

    forwarded = _sanitize_forward_messages(messages)

    assert _tool_contents(forwarded) == ["first", "second"]


def test_repair_drops_an_orphan_but_keeps_an_answered_result():
    orphan = [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "call_ghost", "content": "unissued"},
    ]
    answered = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("call_1")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "answered"},
    ]

    assert _tool_contents(_repair_orphan_tool_results(orphan)) == []
    assert _tool_contents(_repair_orphan_tool_results(answered)) == ["answered"]


def test_sanitize_never_reorders_a_result_before_its_issuer():
    """Whatever the arrival order, a result must end up after its issuer."""
    messages = [
        {"role": "user", "content": "run all three"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1", "a"), _tool_call("call_2", "b"), _tool_call("call_3", "c")],
        },
        {"role": "tool", "tool_call_id": "call_3", "content": "third"},
        {"role": "tool", "tool_call_id": "call_1", "content": "first"},
        {"role": "tool", "tool_call_id": "call_2", "content": "second"},
        {"role": "user", "content": "and now?"},
    ]

    forwarded = _sanitize_forward_messages(messages)

    issuer_at = {
        str(call["id"]): index
        for index, message in enumerate(forwarded)
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }
    for index, message in enumerate(forwarded):
        if message.get("role") == "tool":
            assert index > issuer_at[message["tool_call_id"]]


def test_sanitize_preserves_a_long_multi_round_conversation():
    messages = [{"role": "user", "content": "go"}]
    for round_index in range(8):
        call_id = f"call_{round_index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call(call_id, "exec"), _tool_call(f"{call_id}_b", "read")],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call_id, "content": f"out-{round_index}"})
        messages.append({"role": "tool", "tool_call_id": f"{call_id}_b", "content": f"read-{round_index}"})
    messages.append({"role": "user", "content": "done"})

    forwarded = _sanitize_forward_messages(messages)

    assert len(_tool_contents(forwarded)) == 16
    assert all(output for output in _tool_contents(forwarded))
    assert "out-7" in _tool_contents(forwarded)
    assert "read-7" in _tool_contents(forwarded)


# --------------------------------------------------------------------------- #
# Protocol-level coverage: all three client wirings go through the same
# sanitizer, so each must arrive at the upstream with its tool output intact.
# --------------------------------------------------------------------------- #


def _forwarded_messages(server, body):
    """Run *body* through the live handler, returning what went upstream."""
    import asyncio
    import json

    seen: dict = {}

    async def fake_forward(forwarded):
        seen["body"] = json.loads(json.dumps(forwarded))
        return {
            "id": "x",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    server._forward_to_llm = fake_forward
    asyncio.run(server._handle_request(body, session_id="s", turn_type="main", session_done=False))
    return seen["body"]["messages"]


def _server(tmp_path):
    from skillclaw.api_server import SkillClawAPIServer
    from skillclaw.config import SkillClawConfig

    return SkillClawAPIServer(
        SkillClawConfig(
            proxy_api_key="skillclaw",
            record_enabled=False,
            record_dir=str(tmp_path),
            use_skills=False,
            llm_model_id="gpt-4o",
            llm_api_base="https://api.openai.com/v1",
        )
    )


def test_chat_completions_wiring_forwards_tool_output(tmp_path):
    """The Hermes/OpenAI chat path: role:tool with string content."""
    body = {
        "model": "m",
        "messages": _round("4a57433 feat: enhance usage handling") + [{"role": "user", "content": "ok"}],
    }

    forwarded = _forwarded_messages(_server(tmp_path), body)

    assert "4a57433 feat: enhance usage handling" in _tool_contents(forwarded)


def test_anthropic_wiring_forwards_tool_output(tmp_path):
    """The /v1/messages path, after Anthropic->OpenAI conversion."""
    from skillclaw.protocols.anthropic_messages import to_openai_body

    raw = {
        "model": "m",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "run git log"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "exec", "input": {"command": "git log"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "4a57433 feat: enhance usage handling"}],
                    }
                ],
            },
            {"role": "user", "content": "ok"},
        ],
    }

    forwarded = _forwarded_messages(_server(tmp_path), to_openai_body(raw))

    assert "4a57433 feat: enhance usage handling" in _tool_contents(forwarded)


def test_responses_wiring_forwards_tool_output(tmp_path):
    """The Codex /v1/responses path, where the payload lives in `output`."""
    from skillclaw.protocols.openai_responses import to_openai_body

    raw = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "run git log"},
            {"type": "function_call", "call_id": "call_1", "name": "exec", "arguments": '{"command":"git log"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "4a57433 feat: enhance usage handling"},
            {"type": "message", "role": "user", "content": "ok"},
        ],
    }

    forwarded = _forwarded_messages(_server(tmp_path), to_openai_body(raw, "m"))

    assert "4a57433 feat: enhance usage handling" in _tool_contents(forwarded)


def test_previous_response_replay_preserves_tool_output(tmp_path):
    """Codex resumes by replaying stored history; the result must survive it."""
    from skillclaw.api_server import _merge_previous_response_messages

    previous = _round("REAL-OUTPUT-1")
    merged = _merge_previous_response_messages(previous, previous + [{"role": "user", "content": "q2"}])

    forwarded = _forwarded_messages(_server(tmp_path), {"model": "m", "messages": merged})

    assert "REAL-OUTPUT-1" in _tool_contents(forwarded)
