"""Regression: context truncation must never orphan a role:tool message.

Upstream (OpenAI-compatible) rejects a `role:tool` message whose issuing
assistant turn was truncated away, with `400 invalid_request_error`. Naive
oldest-first dropping produces exactly that shape, so truncation must drop
whole tool-call groups instead of individual messages.
"""

from skillclaw.api_server import SkillClawAPIServer, _estimate_openai_body_input_tokens


def _truncate(messages: list[dict], max_prompt_tokens: int) -> list[dict]:
    server = object.__new__(SkillClawAPIServer)
    return server._truncate_messages(messages, tools=None, max_prompt_tokens=max_prompt_tokens)


def _assistant_tool_call(*call_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"x"}'},
            }
            for call_id in call_ids
        ],
    }


def _orphan_ids(messages: list[dict]) -> set[str]:
    """tool_call_ids answered by a role:tool message with no issuing assistant turn."""
    issued: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                issued.add(str(call["id"]))
    return {
        str(m["tool_call_id"])
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") and str(m["tool_call_id"]) not in issued
    }


def test_truncation_never_orphans_a_tool_result() -> None:
    system = {"role": "system", "content": "system"}
    first_user = {"role": "user", "content": "read the file"}
    assistant = _assistant_tool_call("call_1")
    tool_result = {"role": "tool", "tool_call_id": "call_1", "content": "file body"}

    # Budget that fits system + the tool pair, but not the older user turn.
    limit = _estimate_openai_body_input_tokens(
        {"messages": [system, assistant, tool_result], "tools": None}
    )

    result = _truncate([system, first_user, assistant, tool_result], max_prompt_tokens=limit)

    assert not _orphan_ids(result), f"orphaned tool results: {_orphan_ids(result)}"


def test_truncation_keeps_assistant_and_tool_result_together() -> None:
    system = {"role": "system", "content": "system"}
    first_user = {"role": "user", "content": "read the file"}
    assistant = _assistant_tool_call("call_1")
    tool_result = {"role": "tool", "tool_call_id": "call_1", "content": "file body"}

    limit = _estimate_openai_body_input_tokens(
        {"messages": [system, assistant, tool_result], "tools": None}
    )

    result = _truncate([system, first_user, assistant, tool_result], max_prompt_tokens=limit)

    assert assistant in result, "tool pair should survive when it fits the budget"
    assert tool_result in result, "tool result must travel with its assistant turn"


def test_truncation_drops_a_parallel_tool_call_group_together() -> None:
    system = {"role": "system", "content": "system"}
    first_user = {"role": "user", "content": "read two files"}
    assistant = _assistant_tool_call("a", "b")
    result_a = {"role": "tool", "tool_call_id": "a", "content": "aaa"}
    result_b = {"role": "tool", "tool_call_id": "b", "content": "bbb"}

    limit = _estimate_openai_body_input_tokens(
        {"messages": [system, assistant, result_a, result_b], "tools": None}
    )

    result = _truncate(
        [system, first_user, assistant, result_a, result_b], max_prompt_tokens=limit
    )

    assert not _orphan_ids(result), f"orphaned tool results: {_orphan_ids(result)}"
    if assistant in result:
        kept = {m.get("tool_call_id") for m in result if m.get("role") == "tool"}
        assert kept == {"a", "b"}, f"partial parallel tool group kept: {kept}"


def test_truncation_drops_both_sides_of_a_tool_group_when_it_must_drop() -> None:
    """When the budget cannot fit the tool pair, it goes entirely — not half."""
    system = {"role": "system", "content": "system"}
    user = {"role": "user", "content": "read the file"}
    assistant = _assistant_tool_call("call_1")
    tool_result = {"role": "tool", "tool_call_id": "call_1", "content": "x" * 40_000}

    result = _truncate(
        [system, user, assistant, tool_result], max_prompt_tokens=10_000
    )

    assert not _orphan_ids(result), f"orphaned tool results: {_orphan_ids(result)}"
    assert (assistant in result) == (tool_result in result), "tool group was split"


def test_truncation_still_drops_plain_history_without_tool_calls() -> None:
    system = {"role": "system", "content": "system"}
    older = {"role": "user", "content": "x" * 20_000}
    newest = {"role": "user", "content": "keep me"}
    limit = _estimate_openai_body_input_tokens({"messages": [system, newest], "tools": None})

    assert _truncate([system, older, newest], max_prompt_tokens=limit) == [system, newest]
